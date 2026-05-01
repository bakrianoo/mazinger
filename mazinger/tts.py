"""Voice-cloned text-to-speech synthesis using Qwen3-TTS, Chatterbox, or MLX."""

from __future__ import annotations

import abc
import gc
import logging
import os
from typing import Any, Literal

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# Module-level model cache — keeps loaded TTS models in memory for reuse
_model_cache: dict[str, Any] = {}


def _cache_key(engine: str, model_name: str, device: str, dtype: str) -> str:
    """Build a unique key for the model cache."""
    return f"{engine}|{model_name}|{device}|{dtype}"


def _remove_from_cache(obj: Any) -> None:
    keys = [k for k, v in _model_cache.items() if v is obj]
    for k in keys:
        del _model_cache[k]

# ═══════════════════════════════════════════════════════════════════════════════
#  TTS Engine Type
# ═══════════════════════════════════════════════════════════════════════════════

TTSEngine = Literal["qwen", "chatterbox", "mlx", "omnivoice"]
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_OMNIVOICE_MODEL = "k2-fsa/OmniVoice"

SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
)

_LANG_TO_CODE = {
    "Chinese": "zh", "English": "en", "Japanese": "ja", "Korean": "ko",
    "German": "de", "French": "fr", "Russian": "ru", "Portuguese": "pt",
    "Spanish": "es", "Italian": "it",
}


def validate_language(language: str) -> None:
    """Raise *ValueError* if *language* is not supported by Qwen TTS."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Base Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TTSWrapper(abc.ABC):
    """Unified adapter for TTS engines.

    Subclasses implement :meth:`synthesize` for single-segment generation
    and may override :meth:`synthesize_batch` for true batch support.
    """

    engine: str

    @abc.abstractmethod
    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        """Generate audio for *text*.  Returns ``(audio_array, sample_rate)``."""

    def synthesize_batch(
        self, items: list[tuple[str, str]],
    ) -> list[tuple[np.ndarray, int]]:
        """Synthesize multiple ``(text, language)`` pairs.

        The default implementation loops sequentially.
        """
        total = len(items)
        results = []
        for i, (t, lang) in enumerate(items, 1):
            log.info("Synthesising segment %d/%d", i, total)
            results.append(self.synthesize(t, lang))
        return results

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU memory held by this engine."""


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

    # Auto-detect compatible dtype when the device doesn't support the
    # requested precision (e.g. CPU or older GPUs without bfloat16).
    _is_cpu = "cpu" in str(device)
    if torch_dtype == torch.bfloat16 and (
        _is_cpu or not torch.cuda.is_bf16_supported()
    ):
        log.warning(
            "bfloat16 not supported on %s — falling back to float32", device,
        )
        torch_dtype = torch.float32
        dtype = "float32"
    elif torch_dtype == torch.float16 and _is_cpu:
        log.warning(
            "float16 not efficient on CPU — falling back to float32",
        )
        torch_dtype = torch.float32
        dtype = "float32"

    model = Qwen3TTSModel.from_pretrained(
        model_name, device_map=device, dtype=torch_dtype,
    )
    log.info("Loaded Qwen TTS model: %s on %s (%s)", model_name, device, dtype)
    return model


def _create_qwen_voice_prompt(model: Any, ref_audio: str, ref_text: str | None = None) -> Any:
    """Build a reusable voice-clone prompt for Qwen3-TTS."""
    x_vector_only = ref_text is None
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text or "",
        x_vector_only_mode=x_vector_only,
    )
    log.info("Qwen voice clone prompt created from %s (x_vector_only=%s)", ref_audio, x_vector_only)
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


class _QwenTTSWrapper(TTSWrapper):

    engine = "qwen"

    def __init__(self, model: Any, voice_prompt: Any):
        self.model = model
        self.voice_prompt = voice_prompt

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        return _synthesize_qwen(self.model, self.voice_prompt, text, language)

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        del self.voice_prompt, self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Qwen TTS model unloaded, GPU memory freed.")


VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def design_voice(
    text: str,
    language: str,
    instruct: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[np.ndarray, int]:
    """Synthesise a reference clip using the Qwen3-TTS VoiceDesign model.

    The clip matches the voice described by *instruct* and is intended as
    input for :func:`create_voice_clone_prompt`.  The VoiceDesign model is
    freed from GPU memory after generation.
    """
    validate_language(language)
    model = load_model(VOICE_DESIGN_MODEL, device=device, dtype=dtype, engine="qwen")
    wavs, sr = model.generate_voice_design(
        text=text, language=language, instruct=instruct,
    )
    unload_model(model, force=True)
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


class _ChatterboxTTSWrapper(TTSWrapper):

    engine = "chatterbox"

    def __init__(
        self,
        model: Any,
        ref_audio_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ):
        self.model = model
        self.ref_audio_path = ref_audio_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        return _synthesize_chatterbox(
            self.model, self.ref_audio_path, text,
            self.exaggeration, self.cfg_weight,
        )

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Chatterbox TTS model unloaded, GPU memory freed.")


def _load_mlx_model(
    mlx_model: str = DEFAULT_MLX_MODEL,
) -> Any:
    import platform
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine != "arm64":
        raise RuntimeError(
            "MLX TTS requires Apple Silicon (M1/M2/M3/M4/M5). "
            f"Current platform: {system} ({platform.machine()}). "
            "Use engine='qwen' or engine='chatterbox' instead."
        )
    try:
        from mlx_audio.tts.utils import load_model as mlx_load_model
    except ImportError:
        raise ImportError(
            "mlx-audio not installed. Install with: pip install 'mazinger[tts-mlx]'\n"
            "Or use engine='qwen' for standard Qwen3-TTS."
        ) from None

    model = mlx_load_model(mlx_model)
    log.info("Loaded MLX Qwen3-TTS model: %s", mlx_model)
    return model


class _MLXTTSWrapper(TTSWrapper):

    engine = "mlx"

    def __init__(self, model: Any, ref_audio: str, ref_text: str | None = None):
        self.model = model
        self.ref_audio = ref_audio
        self.ref_text = ref_text

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        lang_code = _LANG_TO_CODE.get(language, "auto")
        if lang_code == "auto":
            log.warning("Unknown language %r for MLX TTS, falling back to 'auto'", language)
        results = list(self.model.generate(
            text=text,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text or "",
            lang_code=lang_code,
        ))
        if not results:
            raise RuntimeError(
                f"MLX TTS generate() returned no results "
                f"(model={self.model!r}, text={text!r}, language={language!r})"
            )
        audio = np.array(results[0].audio)
        return audio, results[0].sample_rate

    def unload(self) -> None:
        # MLX uses Metal GPU memory cache - clear it to free GPU RAM
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass  # Best effort - mlx may not be installed
        _remove_from_cache(self.model)
        del self.model
        gc.collect()
        log.info("MLX TTS model unloaded, GPU memory freed.")


# ═══════════════════════════════════════════════════════════════════════════════
#  OmniVoice Backend
# ═══════════════════════════════════════════════════════════════════════════════

_OMNIVOICE_SAMPLE_RATE = 24_000

# Maximum length of the audio reference passed back to OmniVoice when
# locking the auto/design voice after the first segment.  OmniVoice
# warns above 20 s and recommends 3–10 s; longer references OOM on
# 16 GB GPUs (e.g. Colab T4) and degrade clone quality.
_OMNIVOICE_LOCK_REF_SECONDS = 8.0


def _load_omnivoice_model(
    model_name: str = DEFAULT_OMNIVOICE_MODEL,
    device: str = "cuda:0",
    dtype: str = "float16",
) -> Any:
    """Load an OmniVoice model and return it."""
    import torch
    try:
        from omnivoice import OmniVoice
    except ImportError:
        raise ImportError(
            "omnivoice not installed. Install with: pip install 'mazinger[tts-omnivoice]'\n"
            "Or use engine='qwen' for Qwen3-TTS."
        ) from None

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    _is_cpu = "cpu" in str(device)
    if torch_dtype == torch.bfloat16 and (
        _is_cpu or not torch.cuda.is_bf16_supported()
    ):
        log.warning(
            "bfloat16 not supported on %s — falling back to float16", device,
        )
        torch_dtype = torch.float16
        dtype = "float16"
    elif torch_dtype == torch.float16 and _is_cpu:
        log.warning(
            "float16 not efficient on CPU — falling back to float32",
        )
        torch_dtype = torch.float32
        dtype = "float32"

    model = OmniVoice.from_pretrained(
        model_name, device_map=device, dtype=torch_dtype,
    )
    log.info("Loaded OmniVoice model: %s on %s (%s)", model_name, device, dtype)
    return model


class _OmniVoiceTTSWrapper(TTSWrapper):
    """OmniVoice in voice-clone mode — clones the speaker from ``ref_audio``.

    The reference audio is encoded into a reusable
    :class:`VoiceClonePrompt` once on construction; every segment then
    shares the *exact* same prompt, guaranteeing voice consistency and
    avoiding the cost of re-encoding the reference for each call.
    """

    engine = "omnivoice"

    def __init__(self, model: Any, ref_audio: str, ref_text: str | None = None):
        self.model = model
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        # Build the prompt eagerly so the same encoded reference is shared
        # across all segments.
        self._voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text,
        )

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        audio_list = self.model.generate(
            text=text, voice_clone_prompt=self._voice_clone_prompt,
        )
        if not audio_list:
            raise RuntimeError(
                f"OmniVoice generate() returned no results "
                f"(text={text!r}, ref_audio={self.ref_audio!r})"
            )
        return audio_list[0], _OMNIVOICE_SAMPLE_RATE

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("OmniVoice model unloaded, GPU memory freed.")


def _omnivoice_build_clone_prompt(
    model: Any, audio: np.ndarray, ref_text: str,
) -> Any:
    """Build a reusable :class:`VoiceClonePrompt` from a generated waveform.

    Used by the auto-voice and voice-design wrappers to *lock* the voice
    after the first segment so every subsequent segment is cloned from the
    exact same speaker.

    The reference is trimmed to at most ``_OMNIVOICE_LOCK_REF_SECONDS`` of
    audio.  OmniVoice itself warns that references longer than ~20 s
    cause OOMs and degrade clone quality, and the model documentation
    recommends 3–10 s — anything longer here only hurts.  Trimming the
    *lock reference* does **not** affect the generated segment audio
    returned to the caller; only the encoded prompt used to clone
    subsequent segments is shortened.

    When the audio is trimmed, the paired ``ref_text`` is also shortened
    proportionally so the prompt's audio↔text alignment stays
    reasonable.
    """
    import torch
    max_samples = int(_OMNIVOICE_LOCK_REF_SECONDS * _OMNIVOICE_SAMPLE_RATE)
    if audio.shape[-1] > max_samples:
        ratio = max_samples / audio.shape[-1]
        audio = audio[..., :max_samples]
        words = ref_text.split()
        if words:
            keep = max(1, int(len(words) * ratio))
            ref_text = " ".join(words[:keep])
    waveform = torch.from_numpy(np.ascontiguousarray(audio))
    return model.create_voice_clone_prompt(
        ref_audio=(waveform, _OMNIVOICE_SAMPLE_RATE),
        ref_text=ref_text,
    )


class _OmniVoiceAutoTTSWrapper(TTSWrapper):
    """OmniVoice in auto-voice mode — the model picks a voice automatically.

    The model would otherwise sample a different random voice on every
    :meth:`synthesize` call.  To keep all segments in a single voice we
    let the model pick once on the first call, then build a reusable
    voice-clone prompt from that output and use it for every subsequent
    segment.
    """

    engine = "omnivoice"

    def __init__(self, model: Any):
        self.model = model
        self._voice_clone_prompt: Any = None

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        if self._voice_clone_prompt is not None:
            audio_list = self.model.generate(
                text=text, voice_clone_prompt=self._voice_clone_prompt,
            )
        else:
            log.info(
                "OmniVoice auto-voice: generating first segment to lock the voice"
            )
            audio_list = self.model.generate(text=text)

        if not audio_list:
            raise RuntimeError(
                f"OmniVoice auto-voice generate() returned no results "
                f"(text={text!r})"
            )

        audio = audio_list[0]
        if self._voice_clone_prompt is None:
            try:
                self._voice_clone_prompt = _omnivoice_build_clone_prompt(
                    self.model, audio, text,
                )
                log.info(
                    "OmniVoice auto-voice: voice locked — subsequent segments "
                    "will be cloned from segment 1"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to lock OmniVoice auto-voice (%s) — segments may "
                    "use different voices", exc,
                )
        return audio, _OMNIVOICE_SAMPLE_RATE

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        self._voice_clone_prompt = None
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("OmniVoice model unloaded, GPU memory freed.")


class _OmniVoiceDesignTTSWrapper(TTSWrapper):
    """OmniVoice voice-design mode — voice controlled via instruct string.

    The ``instruct`` text only describes voice *characteristics* (gender,
    pitch, accent, …); the model still samples a fresh random voice
    matching that description on every :meth:`generate` call.  To keep
    all segments in a single voice we generate the first segment with
    the instruct, then lock the voice by building a reusable
    voice-clone prompt from that output and use it for every subsequent
    segment.
    """

    engine = "omnivoice"

    def __init__(self, model: Any, instruct: str):
        self.model = model
        self.instruct = instruct
        self._voice_clone_prompt: Any = None

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        if self._voice_clone_prompt is not None:
            audio_list = self.model.generate(
                text=text, voice_clone_prompt=self._voice_clone_prompt,
            )
        else:
            log.info(
                "OmniVoice voice-design: generating first segment with "
                "instruct=%r to lock the voice", self.instruct,
            )
            audio_list = self.model.generate(text=text, instruct=self.instruct)

        if not audio_list:
            raise RuntimeError(
                f"OmniVoice voice-design generate() returned no results "
                f"(text={text!r}, instruct={self.instruct!r})"
            )

        audio = audio_list[0]
        if self._voice_clone_prompt is None:
            try:
                self._voice_clone_prompt = _omnivoice_build_clone_prompt(
                    self.model, audio, text,
                )
                log.info(
                    "OmniVoice voice-design: voice locked — subsequent "
                    "segments will be cloned from segment 1"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to lock OmniVoice voice-design voice (%s) — "
                    "segments may use different voices", exc,
                )
        return audio, _OMNIVOICE_SAMPLE_RATE

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        self._voice_clone_prompt = None
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("OmniVoice model unloaded, GPU memory freed.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    engine: TTSEngine = "qwen",
    chatterbox_model: str = "ResembleAI/chatterbox",
    mlx_model: str = DEFAULT_MLX_MODEL,
    omnivoice_model: str = DEFAULT_OMNIVOICE_MODEL,
) -> Any:
    """Load a TTS model and return it.

    Parameters:
        model_name:        HuggingFace model identifier (used for Qwen).
        device:            Target device (e.g. ``cuda:0``).
        dtype:             Weight dtype (``bfloat16``, ``float16``, ``float32``).
        engine:            TTS engine: ``qwen``, ``chatterbox``, ``mlx``, or ``omnivoice``.
        chatterbox_model:  HuggingFace model identifier for Chatterbox.
        omnivoice_model:   HuggingFace model identifier for OmniVoice.

    Returns:
        The loaded model instance.
    """
    if engine == "chatterbox":
        name = chatterbox_model
    elif engine == "mlx":
        name = mlx_model
    elif engine == "omnivoice":
        name = omnivoice_model
    else:
        name = model_name
    key = _cache_key(engine, name, device, dtype)
    if key in _model_cache:
        log.info("Reusing cached TTS model: %s", key)
        return _model_cache[key]

    if engine == "qwen":
        model = _load_qwen_model(model_name, device, dtype)
    elif engine == "chatterbox":
        model = _load_chatterbox_model(device, chatterbox_model)
    elif engine == "mlx":
        model = _load_mlx_model(mlx_model)
    elif engine == "omnivoice":
        model = _load_omnivoice_model(omnivoice_model, device, dtype)
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")

    _model_cache[key] = model
    return model


def create_voice_prompt(
    model: Any,
    ref_audio: str,
    ref_text: str | None = None,
    engine: TTSEngine = "qwen",
    chatterbox_exaggeration: float = 0.5,
    chatterbox_cfg: float = 0.5,
    mlx_model: str = DEFAULT_MLX_MODEL,
    omnivoice_model: str = DEFAULT_OMNIVOICE_MODEL,
    voice_design_instruct: str | None = None,
) -> TTSWrapper:
    """Build a reusable voice-clone prompt from a reference recording.

    Parameters:
        model:     A loaded TTS model (from :func:`load_model`).
        ref_audio: Path to the reference audio file.
        ref_text:  Transcript of the reference audio.  When ``None``,
                   Qwen uses x-vector-only mode (no transcript needed).
                   Ignored for Chatterbox and OmniVoice.
        engine:    TTS engine: ``qwen``, ``chatterbox``, ``mlx``, or ``omnivoice``.
        chatterbox_exaggeration: Exaggeration level for Chatterbox (0.0-1.0).
        chatterbox_cfg:          CFG weight for Chatterbox (0.0-1.0).
        voice_design_instruct:   OmniVoice voice-design instruct string
                   (e.g. ``"female, low pitch, british accent"``).  When
                   provided with ``engine="omnivoice"``, uses OmniVoice's
                   built-in voice design instead of cloning from *ref_audio*.

    Returns:
        A :class:`TTSWrapper` instance ready for synthesis.
    """
    if engine == "qwen":
        voice_prompt = _create_qwen_voice_prompt(model, ref_audio, ref_text)
        return _QwenTTSWrapper(model, voice_prompt)
    elif engine == "chatterbox":
        log.info("Chatterbox voice clone configured from %s", ref_audio)
        return _ChatterboxTTSWrapper(
            model, ref_audio, chatterbox_exaggeration, chatterbox_cfg,
        )
    elif engine == "mlx":
        log.info("MLX Qwen3-TTS voice clone configured from %s", ref_audio)
        return _MLXTTSWrapper(model, ref_audio, ref_text)
    elif engine == "omnivoice":
        if voice_design_instruct:
            log.info("OmniVoice voice-design mode (instruct=%r)", voice_design_instruct)
            return _OmniVoiceDesignTTSWrapper(model, voice_design_instruct)
        elif ref_audio:
            log.info("OmniVoice voice clone configured from %s", ref_audio)
            return _OmniVoiceTTSWrapper(model, ref_audio, ref_text)
        else:
            log.info("OmniVoice auto-voice mode (no reference audio)")
            return _OmniVoiceAutoTTSWrapper(model)
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")


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
    pending: list[tuple[int, str, str]] = []  # (index, text, wav_path)

    for entry in srt_entries:
        target_dur = entry["end"] - entry["start"]
        text = entry["text"].strip()
        wav_path = os.path.join(output_dir, f"seg_{entry['idx'].zfill(4)}.wav")

        rec: dict[str, Any] = {
            "idx": entry["idx"],
            "start": entry["start"],
            "end": entry["end"],
            "target_dur": target_dur,
        }

        if not text:
            rec.update(wav_path=None, actual_dur=0)
        elif os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            actual_dur = sf.info(wav_path).duration
            log.debug("Skipping existing segment %s (%.2fs)", wav_path, actual_dur)
            rec.update(wav_path=wav_path, actual_dur=actual_dur, _skipped=True)
        else:
            rec.update(wav_path=wav_path, actual_dur=0)
            pending.append((len(segment_info), text, wav_path))

        segment_info.append(rec)

    # Synthesize pending segments one-by-one, saving each WAV immediately
    # so that already-produced files survive a crash and are cached on retry.
    if pending:
        log.info("TTS: %d segments to synthesize (%d cached)",
                 len(pending), len(srt_entries) - len(pending))
        use_wrapper = isinstance(voice_prompt, TTSWrapper)
        total = len(pending)

        for i, (seg_idx, text, wav_path) in enumerate(pending, 1):
            log.info("Synthesising segment %d/%d", i, total)
            if use_wrapper:
                audio_data, sr = voice_prompt.synthesize(text, language)
            else:
                # Legacy Qwen API (backward compatibility)
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=voice_prompt,
                )
                audio_data = wavs[0]

            sf.write(wav_path, audio_data, sr)
            segment_info[seg_idx]["actual_dur"] = len(audio_data) / sr

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


def unload_model(model: Any, *, force: bool = False) -> None:
    """Unload a TTS model and free GPU memory.

    By default the model is kept in the module-level cache so subsequent
    calls to :func:`load_model` with the same parameters return instantly.
    Pass ``force=True`` to actually remove the model from memory.
    """
    if not force:
        log.info("TTS model kept in memory for reuse (pass force=True to free)")
        return

    if isinstance(model, TTSWrapper):
        model.unload()
        return

    import torch

    _remove_from_cache(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("TTS model unloaded, GPU memory freed.")
