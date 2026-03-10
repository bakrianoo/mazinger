"""Voice-cloned text-to-speech synthesis using Qwen3-TTS or Chatterbox."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Literal

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  TTS Engine Type
# ═══════════════════════════════════════════════════════════════════════════════

TTSEngine = Literal["qwen", "chatterbox"]

SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
)


def validate_language(language: str) -> None:
    """Raise *ValueError* if *language* is not supported by Qwen TTS."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Qwen3-TTS Backend
# ═══════════════════════════════════════════════════════════════════════════════

def _load_qwen_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> Any:
    """Load a Qwen3-TTS model and return it."""
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model = Qwen3TTSModel.from_pretrained(
        model_name, device_map=device, dtype=torch_dtype,
    )
    log.info("Loaded Qwen TTS model: %s on %s (%s)", model_name, device, dtype)
    return model


def _create_qwen_voice_prompt(model: Any, ref_audio: str, ref_text: str) -> Any:
    """Build a reusable voice-clone prompt for Qwen3-TTS."""
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    log.info("Qwen voice clone prompt created from %s", ref_audio)
    return prompt


def _synthesize_qwen(
    model: Any,
    voice_prompt: Any,
    text: str,
    language: str = "English",
) -> tuple[np.ndarray, int]:
    """Generate audio using Qwen3-TTS. Returns (audio_array, sample_rate)."""
    validate_language(language)
    wavs, sr = model.generate_voice_clone(
        text=text, language=language, voice_clone_prompt=voice_prompt,
    )
    return wavs[0], sr


# ═══════════════════════════════════════════════════════════════════════════════
#  Chatterbox Backend
# ═══════════════════════════════════════════════════════════════════════════════

def _load_chatterbox_model(device: str = "cuda", model_name: str = "ResembleAI/chatterbox") -> Any:
    """Load a Chatterbox TTS model and return it."""
    from chatterbox.tts import ChatterboxTTS

    # Chatterbox expects device without index for simple cases
    device_clean = device.split(":")[0] if ":" in device else device
    model = ChatterboxTTS.from_pretrained(device=device_clean)
    log.info("Loaded Chatterbox TTS model: %s on %s", model_name, device_clean)
    return model


def _synthesize_chatterbox(
    model: Any,
    audio_prompt_path: str,
    text: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Generate audio using Chatterbox. Returns (audio_array, sample_rate)."""
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    # Chatterbox returns a torch tensor, convert to numpy
    audio_data = wav.squeeze().cpu().numpy()
    return audio_data, model.sr


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified Public API
# ═══════════════════════════════════════════════════════════════════════════════

class TTSWrapper:
    """Unified wrapper for TTS engines (Qwen or Chatterbox)."""

    def __init__(
        self,
        engine: TTSEngine,
        model: Any,
        voice_prompt: Any = None,
        ref_audio_path: str | None = None,
        chatterbox_exaggeration: float = 0.5,
        chatterbox_cfg: float = 0.5,
    ):
        self.engine = engine
        self.model = model
        self.voice_prompt = voice_prompt  # For Qwen
        self.ref_audio_path = ref_audio_path  # For Chatterbox
        self.chatterbox_exaggeration = chatterbox_exaggeration
        self.chatterbox_cfg = chatterbox_cfg

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        """Generate audio for the given text. Returns (audio_array, sample_rate)."""
        if self.engine == "qwen":
            return _synthesize_qwen(self.model, self.voice_prompt, text, language)
        else:  # chatterbox
            return _synthesize_chatterbox(
                self.model,
                self.ref_audio_path,
                text,
                exaggeration=self.chatterbox_exaggeration,
                cfg_weight=self.chatterbox_cfg,
            )


def load_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    engine: TTSEngine = "qwen",
    chatterbox_model: str = "ResembleAI/chatterbox",
) -> Any:
    """Load a TTS model and return it.

    Parameters:
        model_name: HuggingFace model identifier (used for Qwen only).
        device:     Target device (e.g. ``cuda:0``).
        dtype:      Weight dtype for Qwen (``bfloat16``, ``float16``, ``float32``).
        engine:     TTS engine to use: ``qwen`` or ``chatterbox``.
        chatterbox_model: HuggingFace model identifier for Chatterbox.

    Returns:
        The loaded model instance.
    """
    if engine == "qwen":
        return _load_qwen_model(model_name, device, dtype)
    else:  # chatterbox
        return _load_chatterbox_model(device, chatterbox_model)


def create_voice_prompt(
    model: Any,
    ref_audio: str,
    ref_text: str,
    engine: TTSEngine = "qwen",
    chatterbox_exaggeration: float = 0.5,
    chatterbox_cfg: float = 0.5,
) -> TTSWrapper:
    """Build a reusable voice-clone prompt from a reference recording.

    Parameters:
        model:     A loaded TTS model.
        ref_audio: Path to the reference audio file.
        ref_text:  Transcript of the reference audio (required for Qwen,
                   ignored for Chatterbox).
        engine:    TTS engine: ``qwen`` or ``chatterbox``.
        chatterbox_exaggeration: Exaggeration level for Chatterbox (0.0-1.0).
        chatterbox_cfg:          CFG weight for Chatterbox (0.0-1.0).

    Returns:
        A :class:`TTSWrapper` instance ready for synthesis.
    """
    if engine == "qwen":
        voice_prompt = _create_qwen_voice_prompt(model, ref_audio, ref_text)
        return TTSWrapper(engine, model, voice_prompt=voice_prompt)
    else:  # chatterbox
        log.info("Chatterbox voice clone configured from %s", ref_audio)
        return TTSWrapper(
            engine,
            model,
            ref_audio_path=ref_audio,
            chatterbox_exaggeration=chatterbox_exaggeration,
            chatterbox_cfg=chatterbox_cfg,
        )


def synthesize_segments(
    model: Any,
    voice_prompt: TTSWrapper | Any,
    srt_entries: list[dict],
    output_dir: str,
    *,
    language: str = "English",
    force_reset: bool = False,
) -> list[dict]:
    """Generate TTS audio for each SRT entry and save as WAV files.

    Parameters:
        model:        A loaded TTS model (can be ignored if voice_prompt is TTSWrapper).
        voice_prompt: The voice-clone prompt from :func:`create_voice_prompt`.
                      Can be a :class:`TTSWrapper` or a legacy Qwen prompt.
        srt_entries:  Parsed SRT entries (list of dicts with ``idx``, ``start``,
                      ``end``, ``text``).
        output_dir:   Directory in which to save individual segment WAV files.
        language:     Target language name (e.g. ``English``).
        force_reset:  When ``True``, delete all existing segment files in
                      *output_dir* before generating, so every segment is
                      re-synthesised from scratch.

    Returns:
        A list of segment info dicts with keys ``idx``, ``start``, ``end``,
        ``target_dur``, ``wav_path``, and ``actual_dur``.
    """
    if force_reset and os.path.isdir(output_dir):
        import glob
        for f in glob.glob(os.path.join(output_dir, "seg_*.wav")):
            os.remove(f)
        log.info("Force-reset: cleared existing segments in %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    segment_info: list[dict] = []

    # Determine if we're using the new TTSWrapper or legacy Qwen API
    use_wrapper = isinstance(voice_prompt, TTSWrapper)

    for entry in tqdm(srt_entries, desc="Synthesising"):
        target_dur = entry["end"] - entry["start"]
        text = entry["text"].strip()

        if not text:
            segment_info.append({
                "idx": entry["idx"],
                "start": entry["start"],
                "end": entry["end"],
                "target_dur": target_dur,
                "wav_path": None,
                "actual_dur": 0,
            })
            continue

        wav_path = os.path.join(output_dir, f"seg_{entry['idx'].zfill(4)}.wav")

        # Resume support: skip segments that were already generated.
        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            info = sf.info(wav_path)
            actual_dur = info.duration
            log.debug("Skipping existing segment %s (%.2fs)", wav_path, actual_dur)
            segment_info.append({
                "idx": entry["idx"],
                "start": entry["start"],
                "end": entry["end"],
                "target_dur": target_dur,
                "wav_path": wav_path,
                "actual_dur": actual_dur,
                "_skipped": True,
            })
            continue

        if use_wrapper:
            audio_data, sr = voice_prompt.synthesize(text, language)
        else:
            # Legacy Qwen API (backward compatibility)
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, voice_clone_prompt=voice_prompt,
            )
            audio_data = wavs[0]

        actual_dur = len(audio_data) / sr
        sf.write(wav_path, audio_data, sr)

        segment_info.append({
            "idx": entry["idx"],
            "start": entry["start"],
            "end": entry["end"],
            "target_dur": target_dur,
            "wav_path": wav_path,
            "actual_dur": actual_dur,
        })

    produced = sum(1 for s in segment_info if s["wav_path"])
    skipped = sum(1 for s in segment_info if s.get("_skipped"))
    overflow_segs = [
        s for s in segment_info
        if s["wav_path"] and s["actual_dur"] > s["target_dur"] * 1.05
    ]
    log.info(
        "Synthesised %d/%d segments (%d cached) -> %s",
        produced, len(srt_entries), skipped, output_dir,
    )
    if overflow_segs:
        total_overflow = sum(s["actual_dur"] - s["target_dur"] for s in overflow_segs)
        log.warning(
            "%d/%d segments exceed target duration (total overflow: %.2fs). "
            "Segments: %s",
            len(overflow_segs), len(srt_entries), total_overflow,
            ", ".join(
                f'{s["idx"]}({s["actual_dur"]:.1f}s/{s["target_dur"]:.1f}s)'
                for s in overflow_segs
            ),
        )
    return segment_info


def unload_model(model: Any) -> None:
    """Delete the model and free GPU memory."""
    import torch

    # Handle TTSWrapper case
    if isinstance(model, TTSWrapper):
        actual_model = model.model
        del model
        del actual_model
    else:
        del model

    gc.collect()
    torch.cuda.empty_cache()
    log.info("TTS model unloaded, GPU memory freed.")
