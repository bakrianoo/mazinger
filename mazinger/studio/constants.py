"""Shared constants for Mazinger Studio."""

import os

# Qwen3-TTS supported languages
QWEN_LANGUAGES = [
    "Chinese", "English", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
]

# OmniVoice supported languages
OMNIVOICE_LANGUAGES = [
    "Arabic", "Cantonese", "Chinese", "Czech", "Dutch",
    "English", "Finnish", "French", "German", "Greek",
    "Hindi", "Indonesian", "Italian", "Japanese", "Korean",
    "Polish", "Portuguese", "Romanian", "Russian", "Spanish",
    "Thai", "Turkish", "Ukrainian", "Vietnamese",
]

# Default language list (union, sorted)
LANGUAGES = sorted(set(QWEN_LANGUAGES + OMNIVOICE_LANGUAGES))

VOICE_PRESETS = ["abubakr", "daheeh-v1", "italian-v1", "trump-v1"]

# Pre-defined voice themes — grouped by category for the UI.
# Each entry maps a display label to the internal theme key.
VOICE_THEMES = {
    "🎙️ Narrator": {
        "Narrator — Male":   "narrator-m",
        "Narrator — Female": "narrator-f",
    },
    "✨ Young": {
        "Young — Male":   "young-m",
        "Young — Female": "young-f",
    },
    "🔊 Deep": {
        "Deep — Male":   "deep-m",
        "Deep — Female": "deep-f",
    },
    "☀️ Warm": {
        "Warm — Male":   "warm-m",
        "Warm — Female": "warm-f",
    },
    "📰 News": {
        "News — Male":   "news-m",
        "News — Female": "news-f",
    },
    "📖 Storyteller": {
        "Storyteller — Male":   "storyteller-m",
        "Storyteller — Female": "storyteller-f",
    },
    "🧒 Kids": {
        "Kid — Boy":  "kid-m",
        "Kid — Girl": "kid-f",
    },
    "🎓 Teens": {
        "Teen — Male":   "teen-m",
        "Teen — Female": "teen-f",
    },
}

# Flat lookup: display label → theme key
THEME_CHOICES: list[str] = []
THEME_KEY_MAP: dict[str, str] = {}
for _group_items in VOICE_THEMES.values():
    for _label, _key in _group_items.items():
        THEME_CHOICES.append(_label)
        THEME_KEY_MAP[_label] = _key

QUALITY_MAP = {"Low (360p)": "low", "Medium (720p)": "medium", "High (best)": "high"}

METHOD_MAP = {
    "Faster Whisper (local GPU)": "faster-whisper",
    "Deepgram Nova 3 (cloud)": "deepgram",
    "OpenAI Whisper (cloud)": "openai",
    "WhisperX (local GPU)": "whisperx",
    "CohereX (local GPU, 14 languages)": "coherex",
}

# Source languages CohereX can transcribe.  Other languages must use a
# Whisper-family backend — Cohere Transcribe would otherwise transcribe
# confidently in the wrong language rather than fail.
COHEREX_SOURCE_LANGUAGES = [
    "Arabic", "Chinese (Simplified)", "Dutch", "English", "French", "German",
    "Greek", "Italian", "Japanese", "Korean", "Polish", "Portuguese",
    "Spanish", "Vietnamese",
]

# Gated HuggingFace repositories used by the pipeline.  Access has to be
# requested once per account on the model page, in addition to signing in.
# Labels stay short because they are also rendered as a single row of links
# in the Studio's Hugging Face card.
GATED_MODELS = [
    ("Cohere", "CohereLabs/cohere-transcribe-03-2026"),
    ("Cohere Arabic", "CohereLabs/cohere-transcribe-arabic-07-2026"),
    ("pyannote", "pyannote/segmentation-3.0"),
]

SEGMENT_MODE_MAP = {
    "Short": "short",
    "Long (default)": "long",
    "Auto": "auto",
}

from mazinger.ollama_setup import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL  # noqa: E402
