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
) -> str:
    """Assemble per-segment TTS WAVs into a single time-aligned audio file.

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
        speed_threshold:   Fractional tolerance before tempo-stretching is applied.
        min_speed_ratio:   Hard floor for slowdown (default 0.82 = max ~22% slower).
                           Below this speech starts sounding unnatural.
        target_fill:       Target fraction of the time window to fill when
                           slowing down (default 0.92). A value < 1.0 leaves a
                           small natural gap instead of stretching to the edge.
        tempo_mode:        ``auto`` — speed up overflows AND slow down short
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
            if speed_ratio > 1.0 + speed_threshold:
                # Segment overflows — speed it up
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


def _demucs_background_block(
    model, block: np.ndarray, sr: int, *,
    device: str, segment: float | None, overlap: float,
) -> np.ndarray:
    """Separate one ``(samples, channels)`` block; return mono background at *sr*."""
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
    if model.samplerate != sr:
        bg = torchaudio.functional.resample(
            torch.from_numpy(bg), model.samplerate, sr,
        ).numpy()
    return bg


def _extract_background_demucs(
    audio_path: str, out_path: str, sr: int, *,
    device: str | None = None,
    segment: float | None = None,
    overlap: float = 0.25,
    block_sec: float = DEMUCS_BLOCK_SEC,
    context_sec: float = DEMUCS_CONTEXT_SEC,
) -> None:
    model, device = _load_demucs(device)
    try:
        total = get_audio_duration(audio_path)
        n_blocks = max(1, int(np.ceil(total / block_sec)))
        log.info(
            "Extracting background with demucs on %s (%.0fs in %d block(s))",
            device, total, n_blocks,
        )
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
                bg = _demucs_background_block(
                    model, block, sr, device=device, segment=segment, overlap=overlap,
                )
                # Slice positions come from absolute times so rounding never
                # accumulates across blocks.
                offset = round(core_start * sr) - round(read_start * sr)
                if is_last:
                    core = bg[offset:]
                else:
                    n = round(core_end * sr) - round(core_start * sr)
                    core = bg[offset:offset + n]
                    if len(core) < n:
                        core = np.pad(core, (0, n - len(core)))
                out.write(core)
    finally:
        _unload_demucs(model, device)


def _extract_background(audio_path: str, out_path: str, sr: int = TARGET_SR) -> str:
    """Extract non-vocal background from *audio_path*.

    Uses demucs (htdemucs model) for high-quality source separation, block
    by block (see :data:`DEMUCS_BLOCK_SEC`) and on the GPU when available.
    Falls back to spectral masking via librosa when demucs is unavailable.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        _extract_background_demucs(audio_path, out_path, sr)
    except Exception as exc:
        log.info("Demucs unavailable (%s), using spectral masking fallback", exc)
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.stft(y)
        H, P = librosa.decompose.hpss(np.abs(S), kernel_size=31, margin=4.0)
        mask = P / (H + P + 1e-10)
        bg = librosa.istft(S * mask, length=len(y))
        sf.write(out_path, bg, sr)
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
) -> str:
    """Apply loudness normalisation and background audio mixing.

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
    """
    if not loudness_match and not mix_background:
        if dubbed_path != output_path:
            shutil.copy2(dubbed_path, output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    work = dubbed_path

    # -- loudness matching ------------------------------------------------
    if loudness_match:
        if loudness_cache:
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
        filt = (
            f"[1:a]atrim=0:{dur_dub:.3f},asetpts=PTS-STARTPTS,"
            f"volume={background_volume:.2f}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:weights=1 {background_volume:.2f}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", work, "-i", bg_path,
             "-filter_complex", filt, "-map", "[out]",
             "-ar", str(TARGET_SR), "-ac", "1", mix_path],
            capture_output=True, check=True,
        )
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
