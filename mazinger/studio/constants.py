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
    "OpenAI Whisper (cloud)": "openai",
    "Faster Whisper (local GPU)": "faster-whisper",
    "WhisperX (local GPU)": "whisperx",
}

SEGMENT_MODE_MAP = {
    "Short": "short",
    "Long (default)": "long",
    "Auto": "auto",
}

OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b-q8_0")
