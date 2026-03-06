"""Time-align TTS segments and assemble the final dubbed audio track."""

from __future__ import annotations

import logging
import os
import subprocess

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

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


def assemble_timeline(
    segment_info: list[dict],
    original_duration: float,
    output_path: str,
    *,
    sample_rate: int = TARGET_SR,
    speed_threshold: float = 0.05,
    min_speed_ratio: float = 0.5,
) -> str:
    """Assemble per-segment TTS WAVs into a single time-aligned audio file.

    Each segment is placed at its SRT start time on a silence-filled
    timeline matching *original_duration*.  Segments that are too long are
    sped up; segments that are too short are **slowed down** to fill the
    target duration so dubbed audio stays in sync with the original speaker.

    Parameters:
        segment_info:      List of dicts from :func:`mazinger_dubber.tts.synthesize_segments`.
        original_duration: Duration of the original audio in seconds.
        output_path:       Where to write the final WAV.
        sample_rate:       Target sample rate.
        speed_threshold:   Fractional tolerance before tempo-stretching is applied.
        min_speed_ratio:   Lowest allowed slowdown factor (default 0.5 = max 2× slower).
                           Prevents extreme stretching that sounds unnatural.

    Returns:
        The *output_path*.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_samples = int(original_duration * sample_rate)
    timeline = np.zeros(total_samples, dtype=np.float32)

    stats = {"sped_up": 0, "slowed_down": 0, "ok": 0, "skipped": 0}

    for seg in tqdm(segment_info, desc="Aligning"):
        if seg["wav_path"] is None:
            stats["skipped"] += 1
            continue

        target_dur = seg["target_dur"]
        target_samps = int(target_dur * sample_rate)
        start_samp = int(seg["start"] * sample_rate)

        raw_audio = _load_and_resample(seg["wav_path"], sample_rate)
        actual_dur = len(raw_audio) / sample_rate

        if actual_dur <= 0:
            stats["skipped"] += 1
            continue

        speed_ratio = actual_dur / target_dur

        if speed_ratio > 1.0 + speed_threshold:
            # TTS too long → speed up
            stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
            audio = _tempo_stretch(seg["wav_path"], speed_ratio, stretched_path, sample_rate)
            stats["sped_up"] += 1
        elif speed_ratio < 1.0 - speed_threshold:
            # TTS too short → slow down to fill the target duration.
            # Clamp to min_speed_ratio to avoid extreme / unnatural stretching.
            effective_ratio = max(speed_ratio, min_speed_ratio)
            slowed_path = seg["wav_path"].replace(".wav", "_slowed.wav")
            audio = _tempo_stretch(seg["wav_path"], effective_ratio, slowed_path, sample_rate)
            stats["slowed_down"] += 1
            if effective_ratio != speed_ratio:
                log.debug(
                    "Seg %s: clamped slowdown %.2f→%.2f (min_speed_ratio=%.2f)",
                    seg["idx"], speed_ratio, effective_ratio, min_speed_ratio,
                )
        else:
            audio = raw_audio
            stats["ok"] += 1

        if len(audio) > target_samps:
            audio = audio[:target_samps]
        elif len(audio) < target_samps:
            audio = np.pad(audio, (0, target_samps - len(audio)))

        end_samp = min(start_samp + len(audio), total_samples)
        actual_len = end_samp - start_samp
        timeline[start_samp:end_samp] = audio[:actual_len]

    sf.write(output_path, timeline, sample_rate)

    log.info(
        "Timeline assembled: %.2fs (sped_up=%d, slowed_down=%d, ok=%d, skipped=%d)",
        total_samples / sample_rate,
        stats["sped_up"], stats["slowed_down"], stats["ok"], stats["skipped"],
    )
    return output_path
