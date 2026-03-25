"""Shared constants for Mazinger Studio."""

LANGUAGES = [
    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
    "Czech", "Danish", "Dutch", "English", "Finnish", "French",
    "German", "Greek", "Hebrew", "Hindi", "Hungarian",
    "Indonesian", "Italian", "Japanese", "Korean", "Malay",
    "Norwegian", "Persian", "Polish", "Portuguese",
    "Romanian", "Russian", "Spanish", "Swedish",
    "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese",
]

VOICE_PRESETS = ["abubakr", "daheeh-v1", "italian-v1", "trump-v1"]

QUALITY_MAP = {"Low (360p)": "low", "Medium (720p)": "medium", "High (best)": "high"}

METHOD_MAP = {
    "OpenAI Whisper (cloud)": "openai",
    "Faster Whisper (local GPU)": "faster-whisper",
    "WhisperX (local GPU)": "whisperx",
}

OLLAMA_DEFAULT_MODEL = "qwen3.5:2b-q8_0"
