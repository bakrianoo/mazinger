"""Time-align TTS segments and assemble the final dubbed audio track."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from mazinger.utils import get_audio_duration

log = logging.getLogger(__name__)

TARGET_SR = 24_000


def _load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    """Load a WAV and convert to mono at *target_sr* using ffmpeg."""
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

    gap_samps = int(sample_rate * segment_gap_ms / 1000)

    stats = {"sped_up": 0, "slowed_down": 0, "ok": 0, "skipped": 0, "trimmed": 0}
    overflow_total = 0.0

    valid_segs = [s for s in segment_info if s.get("wav_path") is not None]
    valid_segs.sort(key=lambda s: s["start"])

    for seg_i, seg in enumerate(tqdm(valid_segs, desc="Aligning")):
        raw_audio = _load_and_resample(seg["wav_path"], sample_rate)
        actual_dur = len(raw_audio) / sample_rate
        if actual_dur <= 0:
            stats["skipped"] += 1
            continue

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
            stats["sped_up"] += 1

        elif tempo_mode in ("auto", "dynamic"):
            if speed_ratio > 1.0 + speed_threshold:
                # Segment overflows — speed it up
                effective_ratio = min(speed_ratio, max_tempo)
                stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, stretched_path, sample_rate)
                stats["sped_up"] += 1
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
                    stats["slowed_down"] += 1
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
                    stats["ok"] += 1
            else:
                audio = raw_audio
                stats["ok"] += 1
        else:
            # tempo_mode == "off"
            audio = raw_audio
            stats["ok"] += 1

        # -- Step 2: handle overflow after stretch -------------------------
        if is_last:
            # Last segment: trim only if it exceeds the generous tail pad.
            clip_samps = int((budget_dur + tail_pad_sec) * sample_rate)
            if len(audio) > clip_samps:
                trim_at = _find_last_silence(audio, sample_rate, clip_samps)
                trimmed_secs = (len(audio) - trim_at) / sample_rate
                audio = audio[:trim_at]
                overflow_total += trimmed_secs
                stats["trimmed"] += 1
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
                overflow_secs = (len(audio) - max_len) / sample_rate
                overflow_total += overflow_secs
                trim_at = _find_last_silence(audio, sample_rate, max_len)
                # Guarantee no overlap even if no silence was found.
                trim_at = min(trim_at, max_len)
                audio = audio[:trim_at]
                stats["trimmed"] += 1
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

        # -- Step 3: paste at SRT start time ------------------------------
        end_samp = min(start_samp + len(audio), total_samples)
        seg_len = end_samp - start_samp
        if seg_len > 0:
            timeline[start_samp:end_samp] += audio[:seg_len]

    stats["skipped"] += len(segment_info) - len(valid_segs)

    # Trim tail padding — keep up to the last placed sample or original
    # duration, whichever is longer, plus a small cushion for the fade.
    orig_samples = int(original_duration * sample_rate)
    # Find actual last non-zero sample (= where audio content ends)
    nz = np.nonzero(timeline)[0]
    content_end = int(nz[-1]) + 1 if len(nz) else orig_samples

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

    peak = np.max(np.abs(timeline))
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


def _measure_loudness(path: str) -> float:
    """Return integrated loudness (LUFS) of an audio file via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    import json as _json, re as _re
    m = _re.search(r'\{[^}]+"input_i"[^}]+\}', result.stderr, _re.DOTALL)
    if m:
        data = _json.loads(m.group())
        return float(data["input_i"])
    return -24.0


def _extract_background(audio_path: str, out_path: str, sr: int = TARGET_SR) -> str:
    """Extract non-vocal background from *audio_path*.

    Uses demucs (htdemucs model) for high-quality source separation.
    Falls back to spectral masking via librosa when demucs is unavailable.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import torch, torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        log.info("Extracting background with demucs")
        model = get_model("htdemucs")
        model.eval()
        wav, wav_sr = torchaudio.load(audio_path)
        if wav_sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, wav_sr, model.samplerate)
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0))
        # sources: (1, num_stems, channels, samples)
        vocals_idx = model.sources.index("vocals")
        bg = sources[0]
        bg[vocals_idx] = 0
        bg_np = bg.sum(dim=0).mean(dim=0).cpu().numpy()
        import librosa
        if model.samplerate != sr:
            bg_np = librosa.resample(bg_np, orig_sr=model.samplerate, target_sr=sr)
        sf.write(out_path, bg_np, sr)
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


def post_process(
    dubbed_path: str,
    original_audio: str,
    output_path: str,
    *,
    loudness_match: bool = True,
    mix_background: bool = True,
    background_volume: float = 0.15,
) -> str:
    """Apply loudness normalisation and background audio mixing.

    Parameters:
        dubbed_path:       Path to the assembled TTS audio.
        original_audio:    Path to the original source audio.
        output_path:       Where to write the processed result.
        loudness_match:    Match dubbed loudness to the original.
        mix_background:    Extract and mix background from original.
        background_volume: Gain multiplier for the background layer (0.0–1.0).
    """
    if not loudness_match and not mix_background:
        if dubbed_path != output_path:
            shutil.copy2(dubbed_path, output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    work = dubbed_path

    # -- loudness matching ------------------------------------------------
    if loudness_match:
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
