"""Time-align TTS segments and assemble the final dubbed audio track."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from mazinger.utils import get_audio_duration

log = logging.getLogger(__name__)

TARGET_SR = 24_000

# Segments prepared (loaded, tempo-stretched, trimmed) in parallel by
# assemble_timeline.  Most of that time is spent waiting on ffmpeg.
ASSEMBLE_WORKERS = min(8, os.cpu_count() or 1)

# Slow-down floor of the "sync" tempo mode (see assemble_synced).
SYNC_MIN_TEMPO_DEFAULT = 0.8

# Formats _load_and_resample reads without ffmpeg.  Lossy formats always go
# through ffmpeg: decoders disagree on encoder-delay padding.
_DIRECT_FORMATS = ("WAV", "WAVEX", "RF64", "FLAC", "AIFF")


def _load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    """Load an audio file as mono float32 at *target_sr*.

    Lossless mono files already at *target_sr* (the TTS segments) are read
    directly, which is ~100× faster than starting ffmpeg and gives the same
    samples; anything else is converted with ffmpeg.
    """
    try:
        info = sf.info(wav_path)
    except Exception:  # noqa: BLE001 — not a format soundfile reads
        info = None
    if (info is not None and info.samplerate == target_sr and info.channels == 1
            and info.format in _DIRECT_FORMATS):
        data, _ = sf.read(wav_path, dtype="float32")
        return data

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-ar", str(target_sr), "-ac", "1", "-f", "f32le", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _tempo_stretch(
    wav_path: str,
    factor: float,
    out_path: str,
    sr: int,
) -> np.ndarray:
    """Change playback speed by *factor* using the ffmpeg ``atempo`` filter.

    ``factor > 1`` speeds up, ``factor < 1`` slows down.
    """
    filters: list[str] = []
    remaining = factor
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-filter:a", ",".join(filters),
            "-ar", str(sr), "-ac", "1", out_path,
        ],
        capture_output=True,
        check=True,
    )
    data, _ = sf.read(out_path, dtype="float32")
    return data


def _fade(
    audio: np.ndarray,
    sr: int,
    fade_in_ms: int = 15,
    fade_out_ms: int = 50,
) -> np.ndarray:
    """Apply a raised-cosine (Hann) fade-in/out for natural-sounding edges.

    Short fade-in (default 15 ms) prevents clicks; longer fade-out
    (default 50 ms) mirrors how speech naturally trails off.
    """
    audio = audio.copy()
    n = len(audio)

    fi = min(int(sr * fade_in_ms / 1000), n // 2)
    if fi >= 2:
        # Hann fade-in: 0.5 * (1 - cos(pi * t))  — starts gentle, ends steep
        ramp_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi))).astype(np.float32)
        audio[:fi] *= ramp_in

    fo = min(int(sr * fade_out_ms / 1000), n // 2)
    if fo >= 2:
        ramp_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo))).astype(np.float32)
        audio[-fo:] *= ramp_out

    return audio


def _rms_energy(audio: np.ndarray, frame_len: int) -> np.ndarray:
    """Compute per-frame RMS energy (non-overlapping windows)."""
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.array([0.0], dtype=np.float32)
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def _find_last_silence(audio: np.ndarray, sr: int, budget_samps: int,
                        silence_thresh_db: float = -40.0) -> int:
    """Find the last silence boundary before *budget_samps*.

    Returns a sample index where the audio can be safely trimmed without
    cutting through voiced speech.  Falls back to the lowest-energy frame
    in the search range to minimise audible cuts.
    """
    frame_len = int(sr * 0.02)  # 20 ms frames
    energy = _rms_energy(audio, frame_len)
    thresh = 10 ** (silence_thresh_db / 20.0)
    budget_frame = min(budget_samps // frame_len, len(energy))

    # Search backwards over 80% of the budget range (not just 50%)
    search_floor = max(int(budget_frame * 0.2), 1)

    # Walk backwards from the budget boundary to find a silent frame
    for i in range(budget_frame - 1, search_floor, -1):
        if energy[i] < thresh:
            return (i + 1) * frame_len

    # No silence found — fall back to the lowest-energy frame in the range
    # so we at least cut at the quietest point rather than at an arbitrary
    # boundary that may be mid-vowel.
    search_region = energy[search_floor:budget_frame]
    if len(search_region) > 0:
        min_idx = int(np.argmin(search_region)) + search_floor
        return (min_idx + 1) * frame_len

    return budget_samps


def _speech_density(audio: np.ndarray, sr: int,
                     silence_thresh_db: float = -40.0) -> float:
    """Fraction of frames containing voiced speech (0.0–1.0)."""
    frame_len = int(sr * 0.02)
    energy = _rms_energy(audio, frame_len)
    thresh = 10 ** (silence_thresh_db / 20.0)
    if len(energy) == 0:
        return 1.0
    return float(np.mean(energy >= thresh))


def _ordered_map(
    pool: Executor, fn: Callable, items: Iterable, window: int,
) -> Iterator:
    """``pool.map(fn, items)`` with at most *window* results in flight.

    Results come back in order; memory stays bounded however many items
    there are (``Executor.map`` submits everything up front).
    """
    pending: deque = deque()
    try:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for fut in pending:
            fut.cancel()


# Whole-timeline scans run over blocks of this many samples.  A 2 h timeline
# is ~690 MB; full-array temporaries (np.abs, np.nonzero's int64 indices)
# would multiply its peak memory several times over.
_SCAN_BLOCK = 1 << 20


def _last_nonzero(a: np.ndarray) -> int:
    """Index of the last non-zero element of *a*, or ``-1`` if all zero."""
    for hi in range(len(a), 0, -_SCAN_BLOCK):
        lo = max(0, hi - _SCAN_BLOCK)
        nz = np.flatnonzero(a[lo:hi])
        if len(nz):
            return lo + int(nz[-1])
    return -1


def _peak_abs(a: np.ndarray) -> float:
    """``np.max(np.abs(a))`` without a full-size temporary."""
    peak = 0.0
    for lo in range(0, len(a), _SCAN_BLOCK):
        block = a[lo:lo + _SCAN_BLOCK]
        peak = max(peak, float(block.max()), -float(block.min()))
    return peak


def assemble_timeline(
    segment_info: list[dict],
    original_duration: float,
    output_path: str,
    *,
    sample_rate: int = TARGET_SR,
    speed_threshold: float = 0.05,
    min_speed_ratio: float = 0.82,
    target_fill: float = 0.92,
    tempo_mode: str = "auto",
    fixed_tempo: float | None = None,
    max_tempo: float = 1.5,
    crossfade_ms: int = 50,
    segment_gap_ms: int = 50,
    speech_map=None,
    vocals_path: str | None = None,
    min_tempo: float = SYNC_MIN_TEMPO_DEFAULT,
    report_path: str | None = None,
) -> str:
    """Assemble per-segment TTS WAVs into a single time-aligned audio file.

    ``tempo_mode="sync"`` hands over to :func:`assemble_synced`: every line
    is fitted exactly to the original speech it replaces (from
    *speech_map*) and the output has exactly the original's length.

    Smart tempo approach:
      1. Place each segment at its SRT start time.
      2. If a segment overflows its time slot → tempo-stretch up to *max_tempo*.
      3. If a segment is shorter than its slot → slow it down just enough
         to reach *target_fill* of the window (default 92%), capping at
         *min_speed_ratio* (default 0.82×) so speech never sounds
         unnaturally slow.
      4. If it *still* overflows after the cap → trim at the quietest
         point and apply a long fade-out to mask the cut.

    The *target_fill* parameter prevents the algorithm from trying to fill
    100% of the window — a small natural gap is left.  The *min_speed_ratio*
    acts as a hard floor: segments that would need more aggressive slowdown
    are left partially unfilled rather than distorted.

    Parameters:
        segment_info:      List of dicts from :func:`mazinger.tts.synthesize_segments`.
        original_duration: Duration of the original audio in seconds.
        output_path:       Where to write the final WAV.
        sample_rate:       Target sample rate.
        speed_threshold:   Fractional tolerance before a short segment is slowed
                           down.  Overflows are always sped up, since the
                           overflowing part would otherwise be trimmed.
        min_speed_ratio:   Hard floor for slowdown (default 0.82 = max ~22% slower).
                           Below this speech starts sounding unnatural.
        target_fill:       Target fraction of the time window to fill when
                           slowing down (default 0.92). A value < 1.0 leaves a
                           small natural gap instead of stretching to the edge.
        tempo_mode:        ``sync`` — exact fit to the original speech (see above);
                           ``auto`` — speed up overflows AND slow down short
                           segments toward *target_fill* (default);
                           ``off`` — no tempo adjustment;
                           ``dynamic`` — same as auto (legacy alias);
                           ``fixed`` — apply *fixed_tempo* to every segment.
        fixed_tempo:       Tempo rate applied when ``tempo_mode="fixed"``.
        max_tempo:         Upper speed limit for dynamic/auto mode (default 1.5).
        crossfade_ms:      Fade-in/out at segment edges (default 50).
        segment_gap_ms:    Silence gap reserved between segments (default 50).

    Returns:
        The *output_path*.
    """
    if tempo_mode == "sync":
        return assemble_synced(
            segment_info, original_duration, output_path,
            speech_map=speech_map, sample_rate=sample_rate,
            max_tempo=max_tempo, min_tempo=min_tempo, vocals_path=vocals_path,
            report_path=report_path,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Allow a small tail so the last segment is never hard-clipped.
    tail_pad_sec = 2.0
    total_samples = int((original_duration + tail_pad_sec) * sample_rate)
    timeline = np.zeros(total_samples, dtype=np.float32)

    stats = {"sped_up": 0, "slowed_down": 0, "ok": 0, "skipped": 0, "trimmed": 0}
    overflow_total = 0.0

    valid_segs = [s for s in segment_info if s.get("wav_path") is not None]
    valid_segs.sort(key=lambda s: s["start"])

    def prepare(seg_i: int) -> tuple[int, np.ndarray, str, float] | None:
        """Steps 1–2 for one segment: load, tempo-stretch, trim and fade.

        Depends only on the segment and the next one's start, so segments
        are prepared in parallel; placing them (step 3) stays in order.
        Returns ``(start_samp, audio, outcome, trimmed_secs)``, or ``None``
        for an empty segment.
        """
        seg = valid_segs[seg_i]
        raw_audio = _load_and_resample(seg["wav_path"], sample_rate)
        actual_dur = len(raw_audio) / sample_rate
        if actual_dur <= 0:
            return None
        trimmed_secs = 0.0

        target_dur = seg["target_dur"]
        start_samp = int(seg["start"] * sample_rate)

        # Budget = available time window for dynamic tempo decisions.
        # Uses the real gap to the next segment so tempo-stretch targets
        # the actual available space (the original behavior).
        is_last = (seg_i + 1 >= len(valid_segs))
        if not is_last:
            next_start = valid_segs[seg_i + 1]["start"]
            budget_dur = max(next_start - seg["start"] - segment_gap_ms / 1000, target_dur)
        else:
            budget_dur = max(original_duration - seg["start"], target_dur)

        budget_samps = int(budget_dur * sample_rate)
        speed_ratio = actual_dur / budget_dur

        # -- Step 1: tempo-stretch if needed ------------------------------
        if tempo_mode == "fixed" and fixed_tempo is not None:
            stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
            audio = _tempo_stretch(seg["wav_path"], fixed_tempo, stretched_path, sample_rate)
            outcome = "sped_up"

        elif tempo_mode in ("auto", "dynamic"):
            if speed_ratio > 1.0:
                # Segment overflows — speed it up.  Even a tiny overflow is
                # stretched: left as-is it would be trimmed off below.
                effective_ratio = min(speed_ratio, max_tempo)
                stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, stretched_path, sample_rate)
                outcome = "sped_up"
            elif speed_ratio < 1.0 - speed_threshold:
                # Segment is shorter than its slot — slow it down toward
                # target_fill of the window.  This avoids trying to fill
                # 100% (which would need aggressive slowdown) while still
                # closing most of the gap.
                #
                # needed_ratio = fill / target_fill  (the atempo value
                # that would make TTS fill exactly target_fill of the window).
                # Clamped to min_speed_ratio so speech never sounds drunk.
                fill = speed_ratio              # current fill fraction
                needed_ratio = fill / target_fill  # e.g. 0.80/0.92 = 0.87
                effective_ratio = max(needed_ratio, min_speed_ratio)

                # Only bother stretching if the correction is meaningful
                if effective_ratio < 1.0 - speed_threshold:
                    slowed_path = seg["wav_path"].replace(".wav", "_slowed.wav")
                    audio = _tempo_stretch(seg["wav_path"], effective_ratio, slowed_path, sample_rate)
                    outcome = "slowed_down"
                    log.debug(
                        "Seg %s: slowed %.2fx (%.1fs → %.1fs, "
                        "fill %.0f%% → %.0f%% of %.1fs window)",
                        seg["idx"], effective_ratio, actual_dur,
                        len(audio) / sample_rate,
                        fill * 100, min(actual_dur / effective_ratio / budget_dur, 1.0) * 100,
                        budget_dur,
                    )
                else:
                    audio = raw_audio
                    outcome = "ok"
            else:
                audio = raw_audio
                outcome = "ok"
        else:
            # tempo_mode == "off"
            audio = raw_audio
            outcome = "ok"

        # -- Step 2: handle overflow after stretch -------------------------
        if is_last:
            # Last segment: trim only if it exceeds the generous tail pad.
            clip_samps = int((budget_dur + tail_pad_sec) * sample_rate)
            if len(audio) > clip_samps:
                trim_at = _find_last_silence(audio, sample_rate, clip_samps)
                trimmed_secs = (len(audio) - trim_at) / sample_rate
                audio = audio[:trim_at]
                if trimmed_secs > 0.2:
                    log.warning(
                        "Seg %s (last): trimmed %.2fs to fit budget+pad %.2fs",
                        seg["idx"], trimmed_secs, budget_dur + tail_pad_sec,
                    )
                audio = _fade(audio, sample_rate, fade_in_ms=15, fade_out_ms=150)
            else:
                audio = _fade(audio, sample_rate, fade_in_ms=15, fade_out_ms=200)
        else:
            # Non-last segments: cap the spill so we *never* get two voices
            # speaking at the same time.  A small overlap (≤ segment_gap_ms)
            # is harmless and blends naturally; anything larger is trimmed
            # at the quietest point with a fade-out (same logic as the
            # last-segment branch).  This protects the listener from
            # cascading TTS overruns when a translation is much longer
            # than the source slot or when ``--max-tempo`` can't squeeze
            # the audio to fit.
            next_start_samp = int(valid_segs[seg_i + 1]["start"] * sample_rate)
            # Maximum samples we may write into the timeline starting at
            # start_samp without colliding with the next segment.
            max_len = max(0, next_start_samp - start_samp)

            if len(audio) > max_len:
                overflow_secs = trimmed_secs = (len(audio) - max_len) / sample_rate
                trim_at = _find_last_silence(audio, sample_rate, max_len)
                # Guarantee no overlap even if no silence was found.
                trim_at = min(trim_at, max_len)
                audio = audio[:trim_at]
                if overflow_secs > 0.2:
                    log.warning(
                        "Seg %s: trimmed %.2fs to prevent overlapping the "
                        "next segment (budget %.2fs, audio after stretch "
                        "%.2fs). Source SRT entry is likely too long for "
                        "its time slot — consider re-running with "
                        "--force-reset, or lower --duration-budget / raise "
                        "--max-tempo.",
                        seg["idx"], overflow_secs,
                        budget_dur, len(raw_audio) / sample_rate,
                    )
                audio = _fade(audio, sample_rate, fade_in_ms=15, fade_out_ms=120)
            else:
                audio = _fade(audio, sample_rate, fade_in_ms=15, fade_out_ms=50)

        return start_samp, audio, outcome, trimmed_secs

    workers = max(1, min(ASSEMBLE_WORKERS, len(valid_segs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = _ordered_map(pool, prepare, range(len(valid_segs)), window=4 * workers)
        for result in tqdm(prepared, total=len(valid_segs), desc="Aligning"):
            if result is None:
                stats["skipped"] += 1
                continue
            start_samp, audio, outcome, trimmed_secs = result
            stats[outcome] += 1
            if trimmed_secs > 0:
                stats["trimmed"] += 1
                overflow_total += trimmed_secs

            # -- Step 3: paste at SRT start time --------------------------
            end_samp = min(start_samp + len(audio), total_samples)
            seg_len = end_samp - start_samp
            if seg_len > 0:
                timeline[start_samp:end_samp] += audio[:seg_len]

    stats["skipped"] += len(segment_info) - len(valid_segs)

    # Trim tail padding — keep up to the last placed sample or original
    # duration, whichever is longer, plus a small cushion for the fade.
    orig_samples = int(original_duration * sample_rate)
    # Find actual last non-zero sample (= where audio content ends)
    last_nz = _last_nonzero(timeline)
    content_end = last_nz + 1 if last_nz >= 0 else orig_samples

    # Apply a gentle fade-out at the actual content boundary so the
    # listener doesn't hear a hard cut when the last segment finishes.
    # The fade ramps down the *audio content*, not trailing silence.
    tail_fade_ms = 300
    tail_fade_samps = min(int(sample_rate * tail_fade_ms / 1000), content_end // 2)
    if tail_fade_samps >= 2:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tail_fade_samps))).astype(np.float32)
        timeline[content_end - tail_fade_samps:content_end] *= ramp

    # Keep a tiny silence cushion (100 ms) after the content fade-out
    # so the ending doesn't feel abrupt, then trim.
    cushion = int(sample_rate * 0.1)
    placed_end = max(content_end + cushion, orig_samples)
    placed_end = min(placed_end, total_samples)
    timeline = timeline[:placed_end]

    peak = _peak_abs(timeline)
    if peak > 1.0:
        log.info("Normalising peak %.2f to 1.0", peak)
        timeline /= peak

    sf.write(output_path, timeline, sample_rate)

    if overflow_total > 0.1:
        log.warning(
            "Total overflow: %.2fs of TTS audio was trimmed to fit the timeline. "
            "Consider reducing translation word count (--duration-budget) "
            "or increasing --max-tempo.",
            overflow_total,
        )

    log.info(
        "Timeline assembled: %.2fs | sped_up=%d slowed=%d ok=%d trimmed=%d skipped=%d",
        len(timeline) / sample_rate,
        stats["sped_up"], stats["slowed_down"], stats["ok"],
        stats["trimmed"], stats["skipped"],
    )
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Exact sync ("sync" tempo mode)
# ═══════════════════════════════════════════════════════════════════════════════

# Stretch limits for exact sync.  Inside them a clip is stretched to the
# exact length of the speech it replaces; outside them it is stretched to
# the limit, then (when too long) allowed into the silence that follows and
# only then trimmed.
SYNC_MAX_TEMPO = 1.35
SYNC_MIN_TEMPO = SYNC_MIN_TEMPO_DEFAULT
SYNC_TOLERANCE = 0.01      # |rate - 1| below this is padded/trimmed, not stretched
SYNC_GAP = 0.04            # silence kept before the next line's onset
PASSTHROUGH_MAX = 2.0      # longest untranscribed vocal region kept (seconds)


def has_rubberband() -> bool:
    """Whether ffmpeg has the ``rubberband`` filter (higher-quality time stretch)."""
    global _RUBBERBAND
    if _RUBBERBAND is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, check=True,
            ).stdout
            _RUBBERBAND = any(line.split()[1:2] == ["rubberband"] for line in out.splitlines())
        except (OSError, subprocess.CalledProcessError):
            _RUBBERBAND = False
    return _RUBBERBAND


_RUBBERBAND: bool | None = None


def _atempo_chain(rate: float) -> str:
    filters, remaining = [], rate
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def stretch_audio(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch mono *audio* by *rate* (``> 1`` is faster), keeping pitch.

    Streams through ffmpeg (no temporary files) using rubberband when
    available, ``atempo`` otherwise.  The result is ``≈ len(audio) / rate``
    samples; callers that need an exact length use :func:`fit_length`.
    """
    if abs(rate - 1.0) < 1e-4 or not len(audio):
        return audio
    filt = f"rubberband=tempo={rate:.6f}" if has_rubberband() else _atempo_chain(rate)
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-af", filt, "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
        input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def fit_length(audio: np.ndarray, n: int, sr: int) -> np.ndarray:
    """*audio* padded with silence or trimmed (with a short fade) to exactly *n* samples."""
    if len(audio) == n:
        return audio
    if len(audio) < n:
        return np.pad(audio, (0, n - len(audio)))
    out = audio[:n].copy()
    fade = min(n, int(sr * 0.01))
    if fade > 1:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def _edge_fades(audio: np.ndarray, sr: int, fade_in_ms: float = 5, fade_out_ms: float = 10) -> np.ndarray:
    fi = min(len(audio) // 2, int(sr * fade_in_ms / 1000))
    fo = min(len(audio) // 2, int(sr * fade_out_ms / 1000))
    if fi > 1:
        audio[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 1:
        audio[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return audio


def sync_plan(
    natural: float, target: float, room: float, *,
    max_tempo: float = SYNC_MAX_TEMPO, min_tempo: float = SYNC_MIN_TEMPO,
    tolerance: float = SYNC_TOLERANCE,
) -> tuple[float, float, str]:
    """How to fit a clip of *natural* seconds to *target* seconds.

    *room* is the time available before the next line.  Returns
    ``(rate, length, outcome)``: the stretch rate, the length to place
    (seconds), and one of ``exact``, ``sped_up``, ``slowed_down``,
    ``too_long`` (still longer than the target at *max_tempo*; uses the
    following silence and is trimmed to *room* at most) or ``too_short``
    (shorter than the target even at *min_tempo*; ends early).
    """
    target = max(target, 1e-3)
    rate = natural / target
    if abs(rate - 1.0) <= tolerance:
        return 1.0, target, "exact"
    if rate > max_tempo:
        return max_tempo, min(natural / max_tempo, max(room, target)), "too_long"
    if rate < min_tempo:
        return min_tempo, natural / min_tempo, "too_short"
    return rate, target, "sped_up" if rate > 1.0 else "slowed_down"


def _place_phrases(
    audio: np.ndarray,
    clip_regions: list[tuple[float, float]],
    src_phrases: list[tuple[float, float]],
    onset: float,
    end: float,
    sr: int,
    *,
    max_tempo: float,
    min_tempo: float,
    min_piece: float = 0.25,
) -> tuple[np.ndarray, int] | None:
    """Fit a clip to an original line phrase by phrase; ``None`` when it does not map.

    *clip_regions* are the clip's speech regions (seconds, relative to
    *audio*); *src_phrases* the original phrases of the line (absolute),
    which run from *onset* to *end*.  The clip's own pauses are matched to
    the original's by relative position (:func:`mazinger.speech.match_pauses`);
    each piece between matched pauses is stretched to its original phrase
    group and placed on it, with the original's pause between pieces.
    Returns ``(audio, pieces)`` — *audio* starts at *onset* and lasts
    ``end - onset`` — or ``None`` when there are fewer than two phrases on
    either side, no pause matches, or a piece would need a stretch outside
    ``[min_tempo, max_tempo]``.
    """
    from mazinger.speech import CLIP_PAUSE, match_pauses, phrases

    if len(src_phrases) < 2:
        return None
    clip_ph = phrases(clip_regions, CLIP_PAUSE)
    if len(clip_ph) < 2:
        return None
    pairs = match_pauses(src_phrases, clip_ph)
    if not pairs:
        return None

    # Piece boundaries: original [s0, s1] <- clip [c0, c1].
    src_cuts = [(src_phrases[i][1], src_phrases[i + 1][0]) for i, _ in pairs]
    clip_cuts = [(clip_ph[j][1], clip_ph[j + 1][0]) for _, j in pairs]
    src_bounds = [onset] + [x for cut in src_cuts for x in cut] + [end]
    clip_bounds = [clip_ph[0][0]] + [x for cut in clip_cuts for x in cut] + [clip_ph[-1][1]]
    pieces = []
    for k in range(0, len(src_bounds), 2):
        s0, s1 = src_bounds[k], src_bounds[k + 1]
        c0, c1 = clip_bounds[k], clip_bounds[k + 1]
        if s1 - s0 < min_piece or c1 - c0 < min_piece:
            return None
        rate = (c1 - c0) / (s1 - s0)
        if not (min_tempo <= rate <= max_tempo):
            return None
        pieces.append((s0, s1, c0, c1, rate))

    out = np.zeros(int(round((end - onset) * sr)), dtype=np.float32)
    for s0, s1, c0, c1, rate in pieces:
        piece = audio[int(c0 * sr):int(np.ceil(c1 * sr))]
        if abs(rate - 1.0) > SYNC_TOLERANCE:
            piece = stretch_audio(piece, sr, rate)
        a = int(round((s0 - onset) * sr))
        n = min(int(round((s1 - s0) * sr)), len(out) - a)
        if n <= 0:
            continue
        out[a:a + n] += _edge_fades(fit_length(piece, n, sr), sr)
    return out, len(pieces)


def assemble_synced(
    segment_info: list[dict],
    original_duration: float,
    output_path: str,
    *,
    speech_map=None,
    sample_rate: int = TARGET_SR,
    max_tempo: float = SYNC_MAX_TEMPO,
    min_tempo: float = SYNC_MIN_TEMPO,
    vocals_path: str | None = None,
    passthrough_max: float = PASSTHROUGH_MAX,
    report_path: str | None = None,
) -> str:
    """Assemble segments so every line replaces the original speech exactly.

    For each segment, the speech span it replaces is looked up in
    *speech_map* (:func:`mazinger.speech.speech_spans`; without a map the
    subtitle span is used).  The clip's own leading and trailing silence is
    removed, it is stretched to the length of that span, padded or trimmed
    to the exact sample count, and placed at the span's onset.  Clips that
    cannot reach the target within ``[min_tempo, max_tempo]`` are stretched
    to the limit (see :func:`sync_plan`).

    The output has exactly ``round(original_duration * sample_rate)``
    samples.  With *vocals_path* (the source's vocals stem), vocal activity
    no line covers — laughs, breaths, short missed phrases up to
    *passthrough_max* seconds — is copied from the original so it is not
    lost from the dub.

    A per-segment report is written as JSON to *report_path* (default:
    beside *output_path*, ``<name>.sync.json``).
    """
    import json

    from mazinger.speech import (
        clip_regions, span_phrases, speech_spans, trust_unvoiced, uncovered_regions,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sr = sample_rate
    total = int(round(original_duration * sr))
    timeline = np.zeros(total, dtype=np.float32)

    segs = [s for s in segment_info if s.get("wav_path")]
    spans = speech_spans([(s["start"], s["end"]) for s in segs], speech_map)
    skip_unvoiced = speech_map is not None and trust_unvoiced(spans)
    order = sorted(range(len(segs)), key=lambda i: spans[i].onset)
    gap = SYNC_GAP

    def prepare(k: int):
        i = order[k]
        seg, span = segs[i], spans[i]
        if not span.voiced and skip_unvoiced:
            return "unvoiced"  # no speech in the source here: an ASR hallucination
        onset = min(span.onset, original_duration)
        nxt = spans[order[k + 1]].onset - gap if k + 1 < len(order) else original_duration
        room = max(0.0, min(nxt, original_duration) - onset)
        raw = _load_and_resample(seg["wav_path"], sr)
        regions = clip_regions(raw, sr)
        i0 = max(0, int(regions[0][0] * sr))
        i1 = min(len(raw), int(np.ceil(regions[-1][1] * sr)))
        audio = raw[i0:i1]
        natural = len(audio) / sr
        if natural <= 0 or room <= 0:
            return None
        target = min(span.duration, room)
        start = int(round(onset * sr))
        base = {"idx": seg["idx"], "onset": round(onset, 3), "target": round(target, 3),
                "natural": round(natural, 3), "voiced": span.voiced}

        # A line over several original phrases: fit piece by piece, so the
        # dub pauses where the speaker paused.
        if speech_map is not None and span.voiced:
            placed = _place_phrases(
                audio, [(a - i0 / sr, b - i0 / sr) for a, b in regions],
                span_phrases(speech_map, span), onset, onset + target, sr,
                max_tempo=max_tempo, min_tempo=min_tempo,
            )
            if placed is not None:
                audio, pieces = placed
                return start, audio, {**base, "rate": round(natural / target, 4),
                                      "placed": round(len(audio) / sr, 3),
                                      "outcome": "phrased", "pieces": pieces, "trimmed": 0.0}

        rate, length, outcome = sync_plan(natural, target, room,
                                          max_tempo=max_tempo, min_tempo=min_tempo)
        if rate != 1.0:
            audio = stretch_audio(audio, sr, rate)
        n = int(round(length * sr))
        trimmed = 0.0
        if outcome == "too_long" and len(audio) > n:
            cut = _find_last_silence(audio, sr, n)
            trimmed = (len(audio) - min(cut, n)) / sr
            audio = audio[:min(cut, n)]
        audio = _edge_fades(fit_length(audio, n, sr), sr)
        return start, audio, {**base, "rate": round(rate, 4), "placed": round(len(audio) / sr, 3),
                              "outcome": outcome, "trimmed": round(trimmed, 3)}

    stats = {k: 0 for k in ("exact", "phrased", "sped_up", "slowed_down", "too_long",
                            "too_short", "skipped", "unvoiced")}
    report: list[dict] = []
    workers = max(1, min(ASSEMBLE_WORKERS, len(order)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(_ordered_map(pool, prepare, range(len(order)), window=4 * workers),
                           total=len(order), desc="Syncing"):
            if result is None or result == "unvoiced":
                stats["skipped" if result is None else "unvoiced"] += 1
                continue
            start, audio, rec = result
            end = min(start + len(audio), total)
            timeline[start:end] += audio[:end - start]
            stats[rec["outcome"]] += 1
            report.append(rec)

    kept = 0.0
    if vocals_path and speech_map is not None and passthrough_max > 0 and os.path.isfile(vocals_path):
        info = sf.info(vocals_path)
        resampled = None  # decoded once, only when the stem is not at *sr*
        for s0, s1 in uncovered_regions(speech_map, spans):
            if s1 - s0 > passthrough_max:
                continue
            a, b = int(round(s0 * sr)), min(int(round(s1 * sr)), total)
            if b <= a:
                continue
            if info.samplerate == sr:
                clip, _ = sf.read(vocals_path, start=a, stop=b, dtype="float32", always_2d=True)
                clip = clip.mean(axis=1)
            else:
                if resampled is None:
                    resampled = _load_and_resample(vocals_path, sr)
                clip = resampled[a:b]
            clip = _edge_fades(clip[:b - a].copy(), sr, 20, 30)
            timeline[a:a + len(clip)] += clip
            kept += len(clip) / sr

    peak = _peak_abs(timeline)
    if peak > 1.0:
        log.info("Normalising peak %.2f to 1.0", peak)
        timeline /= peak
    sf.write(output_path, timeline, sr)

    dev = [abs(r["placed"] - r["target"]) / max(r["target"], 1e-3) for r in report]
    summary = {
        "duration": round(total / sr, 6),
        "samples": total,
        "segments": len(segs),
        **stats,
        "passthrough_seconds": round(kept, 2),
        "mean_length_error": round(float(np.mean(dev)) if dev else 0.0, 4),
        "max_length_error": round(float(np.max(dev)) if dev else 0.0, 4),
    }
    report_path = report_path or os.path.splitext(output_path)[0] + ".sync.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "segments": report}, fh, indent=1)
    except OSError as exc:
        log.warning("Could not write the sync report: %s", exc)

    log.info(
        "Synced timeline: %.2fs | exact=%d phrased=%d sped_up=%d slowed=%d too_long=%d "
        "too_short=%d skipped=%d unvoiced=%d | mean length error %.1f%% | kept %.1fs of "
        "untranscribed vocals",
        total / sr, stats["exact"], stats["phrased"], stats["sped_up"], stats["slowed_down"],
        stats["too_long"], stats["too_short"], stats["skipped"], stats["unvoiced"],
        summary["mean_length_error"] * 100, kept,
    )
    return output_path


def force_length(path: str, samples: int) -> None:
    """Pad or trim the audio file at *path* in place to exactly *samples* frames."""
    info = sf.info(path)
    if info.frames == samples:
        return
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if len(data) > samples:
        data = data[:samples]
    else:
        data = np.pad(data, ((0, samples - len(data)), (0, 0)))
    sf.write(path, data, sr, subtype=info.subtype)


def _loudness_or_none(path: str) -> float | None:
    """Integrated loudness (LUFS) of an audio file via ffmpeg, or ``None``."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    import json as _json, re as _re
    m = _re.search(r'\{[^}]+"input_i"[^}]+\}', result.stderr, _re.DOTALL)
    if m:
        try:
            value = float(_json.loads(m.group())["input_i"])
        except (ValueError, KeyError):
            return None
        return value if np.isfinite(value) else None
    return None


def _measure_loudness(path: str) -> float:
    """Return integrated loudness (LUFS) of an audio file via ffmpeg."""
    value = _loudness_or_none(path)
    return -24.0 if value is None else value


def measure_loudness_cached(audio_path: str, cache_path: str) -> float:
    """Integrated loudness of *audio_path*, kept in *cache_path* (JSON).

    Measuring takes about a minute per hour of audio, and the source of a
    project never changes, so the value is reused while *audio_path* keeps
    the size and modification time it was measured with.  A failed
    measurement (the -24 LUFS fallback) is not cached.
    """
    import json

    st = os.stat(audio_path)
    stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("source") == stamp:
            return float(cached["integrated_lufs"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass

    value = _loudness_or_none(audio_path)
    if value is None:
        return -24.0
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"source": stamp, "integrated_lufs": value}, fh)
    os.replace(tmp, cache_path)
    log.info("Source loudness cached: %.1f LUFS (%s)", value, cache_path)
    return value


# Demucs runs over blocks of the source so memory stays flat on long videos:
# the separated stems of a whole 2 h file would need ~10 GB of RAM.  Each
# block is decoded with extra context on both sides, which is separated and
# then discarded so block seams are inaudible.
DEMUCS_BLOCK_SEC = 300.0
DEMUCS_CONTEXT_SEC = 5.0


def _decode_audio(audio_path: str, sr: int, channels: int,
                  start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode a range of *audio_path* to float32 ``(samples, channels)`` via ffmpeg."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", audio_path, "-ar", str(sr), "-ac", str(channels), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, channels)


def _load_demucs(device: str | None) -> tuple[object, str]:
    """Load htdemucs and return ``(model, device)``."""
    import torch
    from demucs.pretrained import get_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    return model, device


def _unload_demucs(model: object, device: str) -> None:
    import gc
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def _demucs_stems_block(
    model, block: np.ndarray, sr: int, *,
    device: str, segment: float | None, overlap: float, vocals: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Separate one ``(samples, channels)`` block; return mono ``(background, vocals)`` at *sr*.

    With ``vocals=False`` the vocals stem is neither kept nor resampled (``None``).
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    wav = torch.from_numpy(np.ascontiguousarray(block.T))
    with torch.no_grad():
        # split=True runs the model over `segment`-second windows moved to
        # *device* one at a time, so GPU memory does not grow with the block.
        sources = apply_model(
            model, wav[None], device=device, split=True,
            segment=segment, overlap=overlap, progress=False,
        )
    stems = sources[0].cpu().numpy()  # (stems, channels, samples)
    vocals_idx = model.sources.index("vocals")
    bg = (stems.sum(axis=0) - stems[vocals_idx]).mean(axis=0).astype(np.float32)
    voc = stems[vocals_idx].mean(axis=0).astype(np.float32) if vocals else None
    if model.samplerate != sr:
        bg = torchaudio.functional.resample(
            torch.from_numpy(bg), model.samplerate, sr,
        ).numpy()
        if voc is not None:
            voc = torchaudio.functional.resample(
                torch.from_numpy(voc), model.samplerate, sr,
            ).numpy()
    return bg, voc


def _demucs_background_block(
    model, block: np.ndarray, sr: int, *,
    device: str, segment: float | None, overlap: float,
) -> np.ndarray:
    """Separate one ``(samples, channels)`` block; return mono background at *sr*."""
    return _demucs_stems_block(
        model, block, sr, device=device, segment=segment, overlap=overlap, vocals=False,
    )[0]


def _extract_background_demucs(
    audio_path: str, out_path: str, sr: int, *,
    device: str | None = None,
    segment: float | None = None,
    overlap: float = 0.25,
    block_sec: float = DEMUCS_BLOCK_SEC,
    context_sec: float = DEMUCS_CONTEXT_SEC,
    vocals_path: str | None = None,
) -> None:
    """Write the background stem to *out_path* (and the vocals stem to *vocals_path*)."""
    model, device = _load_demucs(device)
    vocals_out = None
    try:
        total = get_audio_duration(audio_path)
        n_blocks = max(1, int(np.ceil(total / block_sec)))
        log.info(
            "Extracting background%s with demucs on %s (%.0fs in %d block(s))",
            " and vocals" if vocals_path else "", device, total, n_blocks,
        )
        if vocals_path:
            vocals_out = sf.SoundFile(vocals_path, "w", samplerate=sr, channels=1, format="WAV")
        with sf.SoundFile(out_path, "w", samplerate=sr, channels=1, format="WAV") as out:
            for b in tqdm(range(n_blocks), desc="Separating", disable=n_blocks == 1):
                core_start = b * block_sec
                is_last = b == n_blocks - 1
                core_end = total if is_last else (b + 1) * block_sec
                read_start = max(0.0, core_start - context_sec)
                read_end = None if is_last else core_end + context_sec

                block = _decode_audio(
                    audio_path, model.samplerate, model.audio_channels,
                    start=read_start,
                    duration=None if read_end is None else read_end - read_start,
                )
                if not len(block):
                    break
                kw = dict(device=device, segment=segment, overlap=overlap)
                if vocals_out is not None:
                    stems = _demucs_stems_block(model, block, sr, **kw)
                else:
                    stems = (_demucs_background_block(model, block, sr, **kw),)
                # Slice positions come from absolute times so rounding never
                # accumulates across blocks.
                offset = round(core_start * sr) - round(read_start * sr)
                n = None if is_last else round(core_end * sr) - round(core_start * sr)
                for stem, dest in zip(stems, (out, vocals_out)):
                    core = stem[offset:] if n is None else stem[offset:offset + n]
                    if n is not None and len(core) < n:
                        core = np.pad(core, (0, n - len(core)))
                    dest.write(core)
    finally:
        if vocals_out is not None:
            vocals_out.close()
        _unload_demucs(model, device)


def _extract_stems(
    audio_path: str, out_path: str, sr: int = TARGET_SR, *, vocals_path: str | None = None,
) -> str:
    """Write the background stem (and optionally the vocals stem); return the method used.

    Uses demucs (htdemucs model) for high-quality source separation, block
    by block (see :data:`DEMUCS_BLOCK_SEC`) and on the GPU when available;
    returns ``"demucs"``.  Falls back to spectral masking via librosa when
    demucs is unavailable and returns ``"hpss"`` — its vocals stem is the mix
    minus the background, so it still carries some music.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        if vocals_path:
            _extract_background_demucs(audio_path, out_path, sr, vocals_path=vocals_path)
        else:
            _extract_background_demucs(audio_path, out_path, sr)
        return "demucs"
    except Exception as exc:
        log.info("Demucs unavailable (%s), using spectral masking fallback", exc)
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.stft(y)
        H, P = librosa.decompose.hpss(np.abs(S), kernel_size=31, margin=4.0)
        mask = P / (H + P + 1e-10)
        bg = librosa.istft(S * mask, length=len(y))
        sf.write(out_path, bg, sr)
        if vocals_path:
            sf.write(vocals_path, y - bg, sr)
        return "hpss"


def _extract_background(audio_path: str, out_path: str, sr: int = TARGET_SR) -> str:
    """Extract the non-vocal background of *audio_path* to *out_path* (see :func:`_extract_stems`)."""
    _extract_stems(audio_path, out_path, sr)
    return out_path


def background_cache_path(audio_path: str, sr: int = TARGET_SR) -> str:
    """Return the cache path of the background stem for *audio_path*.

    ``source/audio.mp3`` → ``source/background.<sr>.wav``.
    """
    return os.path.join(os.path.dirname(audio_path) or ".", f"background.{sr}.wav")


def extract_background_cached(
    audio_path: str,
    cache_path: str | None = None,
    sr: int = TARGET_SR,
) -> str:
    """Return a background stem for *audio_path*, extracting it only when needed.

    The stem at *cache_path* (default: :func:`background_cache_path`) is
    reused when it is at least as new as *audio_path*; otherwise it is
    re-extracted.  The file is replaced atomically, so a concurrent reader
    never sees a partial stem.
    """
    cache_path = cache_path or background_cache_path(audio_path, sr)
    try:
        fresh = (
            os.path.getsize(cache_path) > 0
            and os.path.getmtime(cache_path) >= os.path.getmtime(audio_path)
        )
    except OSError:
        fresh = False
    if fresh:
        log.info("Reusing cached background stem: %s", cache_path)
        return cache_path

    tmp_path = f"{os.path.splitext(cache_path)[0]}.{os.getpid()}.part.wav"
    try:
        _extract_background(audio_path, tmp_path, sr=sr)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    log.info("Background stem cached: %s", cache_path)
    return cache_path


def _fresh(path: str, source: str) -> bool:
    try:
        return os.path.getsize(path) > 0 and os.path.getmtime(path) >= os.path.getmtime(source)
    except OSError:
        return False


def extract_stems_cached(
    audio_path: str,
    background_path: str,
    vocals_path: str,
    sr: int = TARGET_SR,
) -> tuple[str, str, str]:
    """Background and vocals stems of *audio_path*, separated once and cached.

    Both stems come from one separation pass and are reused while newer than
    *audio_path*.  Returns ``(background_path, vocals_path, method)`` where
    *method* is ``"demucs"`` or ``"hpss"`` (see :func:`_extract_stems`).
    """
    import json

    meta_path = os.path.join(os.path.dirname(background_path) or ".", "stems.json")
    method = None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            method = json.load(fh).get("method")
    except (OSError, ValueError, AttributeError):
        pass
    if method and _fresh(background_path, audio_path) and _fresh(vocals_path, audio_path):
        log.info("Reusing cached stems (%s): %s", method, os.path.dirname(background_path))
        return background_path, vocals_path, method

    pid = os.getpid()
    tmp_bg = f"{os.path.splitext(background_path)[0]}.{pid}.part.wav"
    tmp_vo = f"{os.path.splitext(vocals_path)[0]}.{pid}.part.wav"
    try:
        method = _extract_stems(audio_path, tmp_bg, sr, vocals_path=tmp_vo)
        os.replace(tmp_bg, background_path)
        os.replace(tmp_vo, vocals_path)
    finally:
        for tmp in (tmp_bg, tmp_vo):
            if os.path.exists(tmp):
                os.remove(tmp)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"method": method}, fh)
    log.info("Stems cached (%s): %s, %s", method, background_path, vocals_path)
    return background_path, vocals_path, method


def post_process(
    dubbed_path: str,
    original_audio: str,
    output_path: str,
    *,
    loudness_match: bool = True,
    mix_background: bool = True,
    background_volume: float = 0.15,
    background_cache: str | None = None,
    loudness_cache: str | None = None,
    voice_reference: str | None = None,
    exact: bool = False,
) -> str:
    """Apply loudness normalisation and background audio mixing.

    With ``exact=True`` (the ``sync`` tempo mode) the dub is treated as a
    replacement voice track: it is matched to the loudness of
    *voice_reference* (the original's vocals stem) when given, the
    background is added at *background_volume* without amix's automatic
    down-scaling, a limiter guards against clipping, and the output keeps
    the exact sample count of *dubbed_path*.

    Parameters:
        dubbed_path:       Path to the assembled TTS audio.
        original_audio:    Path to the original source audio.
        output_path:       Where to write the processed result.
        loudness_match:    Match dubbed loudness to the original.
        mix_background:    Extract and mix background from original.
        background_volume: Gain multiplier for the background layer (0.0–1.0).
        background_cache:  Where to keep the extracted background stem so
                           later calls reuse it (see
                           :func:`extract_background_cached`).  When
                           ``None``, the stem is re-extracted every time
                           into ``background.wav`` beside *output_path*.
        loudness_cache:    JSON file keeping the loudness of *original_audio*
                           so later calls skip measuring it again (see
                           :func:`measure_loudness_cached`).
        voice_reference:   Loudness reference for the dub instead of
                           *original_audio* (exact mode only).
        exact:             Replacement-voice mixing, see above.
    """
    frames = sf.info(dubbed_path).frames if exact else None
    if not loudness_match and not mix_background:
        if dubbed_path != output_path:
            shutil.copy2(dubbed_path, output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    work = dubbed_path

    # -- loudness matching ------------------------------------------------
    if loudness_match:
        if exact and voice_reference and os.path.isfile(voice_reference):
            target_lufs = measure_loudness_cached(
                voice_reference, os.path.splitext(voice_reference)[0] + ".loudness.json",
            )
        elif loudness_cache:
            target_lufs = measure_loudness_cached(original_audio, loudness_cache)
        else:
            target_lufs = _measure_loudness(original_audio)
        target_lufs = max(target_lufs, -30.0)  # safety floor
        norm_path = output_path + ".norm.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", work,
             "-af", f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11",
             "-ar", str(TARGET_SR), "-ac", "1", norm_path],
            capture_output=True, check=True,
        )
        work = norm_path
        log.info("Loudness matched to %.1f LUFS", target_lufs)

    # -- background mixing ------------------------------------------------
    if mix_background:
        if background_cache:
            bg_path = extract_background_cached(original_audio, background_cache, sr=TARGET_SR)
        else:
            bg_path = os.path.join(os.path.dirname(output_path), "background.wav")
            _extract_background(original_audio, bg_path, sr=TARGET_SR)
            log.info("Background audio saved: %s", bg_path)

        dur_dub = get_audio_duration(work)
        mix_path = output_path + ".mix.wav"
        if exact:
            base = (
                f"[1:a]atrim=0:{dur_dub:.6f},asetpts=PTS-STARTPTS,"
                f"volume={background_volume:.3f}[bg];"
                f"[0:a][bg]amix=inputs=2:duration=first:normalize=0"
            )
            # alimiter's latency compensation (ffmpeg 5+) keeps the voice on
            # its onsets; older builds get a plain peak normalise instead.
            filters = [base + ",alimiter=limit=0.97:level=0:latency=1[out]", base + "[out]"]
        else:
            filters = [(
                f"[1:a]atrim=0:{dur_dub:.3f},asetpts=PTS-STARTPTS,"
                f"volume={background_volume:.2f}[bg];"
                f"[0:a][bg]amix=inputs=2:duration=first:weights=1 {background_volume:.2f}[out]"
            )]
        for n, filt in enumerate(filters, 1):
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", work, "-i", bg_path,
                     "-filter_complex", filt, "-map", "[out]",
                     "-ar", str(TARGET_SR), "-ac", "1", mix_path],
                    capture_output=True, check=True,
                )
                break
            except subprocess.CalledProcessError:
                if n == len(filters):
                    raise
        if exact and n > 1:
            data, rate = sf.read(mix_path, dtype="float32")
            peak = _peak_abs(data)
            if peak > 0.97:
                sf.write(mix_path, data * (0.97 / peak), rate)
        work = mix_path
        log.info("Mixed background at volume %.0f%%", background_volume * 100)

    # -- move final result into place ------------------------------------
    if work != output_path:
        shutil.move(work, output_path)

    # cleanup temp files (background.wav is kept for inspection)
    for suffix in (".norm.wav", ".mix.wav"):
        tmp = output_path + suffix
        if os.path.exists(tmp) and tmp != output_path:
            os.remove(tmp)

    if frames is not None:
        force_length(output_path, frames)
    return output_path


def mux_video(video_path: str, audio_path: str, output_path: str) -> str | None:
    """Replace the audio track of *video_path* with *audio_path*.

    Uses ffmpeg to copy the video stream and encode the new audio.
    Returns *output_path*, or ``None`` if ffmpeg is not installed.
    """
    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found — cannot produce dubbed video. "
            "Install ffmpeg (e.g. 'apt install ffmpeg' or 'brew install ffmpeg') "
            "and re-run with --output-type video."
        )
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log.info("Muxed video saved: %s", output_path)
    return output_path
