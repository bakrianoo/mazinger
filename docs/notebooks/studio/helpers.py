"""Helper classes and functions for Mazinger Studio."""

import logging
import subprocess as sp
import threading
import time

from constants import OLLAMA_DEFAULT_MODEL


class LogCollector(logging.Handler):
    """Thread-safe log handler that buffers formatted messages."""

    def __init__(self):
        super().__init__()
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self._lines.append(self.format(record))

    def read(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def clear(self):
        with self._lock:
            self._lines.clear()


def ensure_ollama(model_id: str | None = None):
    """Start Ollama server (if needed) and pull the requested model."""
    model_id = model_id or OLLAMA_DEFAULT_MODEL

    # Start server if not already running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        sp.Popen(["ollama", "serve"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        time.sleep(3)

    # Check if model is already available
    result = sp.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    for line in result.stdout.strip().splitlines()[1:]:
        if line.split()[0] == model_id:
            return  # already pulled

    # Pull the model
    sp.run(["ollama", "pull", model_id], check=True, timeout=600)


# ═══════════════════════════════════════════════════════════════════════
#  Phase detection — parse log lines to show current pipeline stage
# ═══════════════════════════════════════════════════════════════════════

# Patterns matched against the LAST log lines (checked bottom-up).
# First match wins, so order = most-recent stage first.
PHASE_PATTERNS = [
    ("Done. Final audio",           "✅ Finalizing…"),
    ("TTS model unloaded",          "⏳ Assembling final audio…"),
    ("TTS model kept in memory",    "⏳ Assembling final audio…"),
    ("assemble",                    "⏳ Assembling final audio…"),
    ("Synthesised",                 "⏳ Assembling final audio…"),
    ("Synthesising",                "⏳ Synthesizing speech… (TTS)"),
    ("Loaded Qwen TTS",            "⏳ Synthesizing speech… (TTS)"),
    ("Loaded Chatterbox TTS",      "⏳ Synthesizing speech… (TTS)"),
    ("voice clone prompt created",  "⏳ Preparing voice clone…"),
    ("Reusing cached TTS",         "⏳ Preparing voice clone…"),
    ("Reusing saved voice profile", "⏳ Loading saved voice profile…"),
    ("Reusing auto-cloned voice",   "⏳ Reusing auto-cloned voice profile…"),
    ("Auto-cloned voice profile",   "⏳ Auto-cloning voice from source…"),
    ("Generating voice from theme", "⏳ Generating voice from theme… (VoiceDesign)"),
    ("VoiceDesign model loaded",   "⏳ Generating voice from theme… (VoiceDesign)"),
    ("Voice profile saved",        "⏳ Voice profile ready"),
    ("Re-segmentation",            "⏳ Re-segmenting subtitles…"),
    ("Skipping re-segmentation",   "⏳ Re-segmenting subtitles…"),
    ("Translation complete",       "⏳ Re-segmenting subtitles…"),
    ("Translating",                "⏳ Translating subtitles… (LLM)"),
    ("Skipping translation",       "⏳ Translating subtitles… (LLM)"),
    ("Skipping description",       "⏳ Translating subtitles… (LLM)"),
    ("describe_content",           "⏳ Analyzing video content… (LLM)"),
    ("Skipping thumbnails",        "⏳ Analyzing video content… (LLM)"),
    ("select_timestamps",          "⏳ Extracting thumbnails… (LLM)"),
    ("Estimated SRT tokens",       "⏳ Preparing translation…"),
    ("Using raw SRT",              "⏳ Preparing translation…"),
    ("Using resegmented SRT",      "⏳ Preparing translation…"),
    ("Transcription complete",     "⏳ Transcription done, extracting thumbnails…"),
    ("faster-whisper transcription","⏳ Transcription done, extracting thumbnails…"),
    ("Transcribing with",          "⏳ Transcribing audio…"),
    ("Detected language",          "⏳ Transcribing audio…"),
    ("Skipping transcription",     "⏳ Transcription found (cached)"),
    ("Audio extracted",            "⏳ Transcribing audio…"),
    ("Video saved",                "⏳ Extracting audio…"),
    ("Requesting quality",         "⏳ Downloading video…"),
    ("Resolved slug",              "⏳ Downloading video…"),
    ("Skipping download",          "⏳ Download found (cached)"),
    ("Project:",                   "⏳ Starting pipeline…"),
]


def detect_phase(log_text: str) -> str:
    """Return a human-friendly status string based on the latest log lines."""
    if not log_text:
        return "⏳ Starting pipeline…"
    # Check last 20 lines (most recent activity)
    lines = log_text.strip().splitlines()[-20:]
    for line in reversed(lines):
        for pattern, label in PHASE_PATTERNS:
            if pattern in line:
                return label
    return "⏳ Processing…"


def check_ollama_health() -> str | None:
    """Return a warning string if Ollama is not responding, else None."""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        resp.read()
        return None  # healthy
    except Exception:
        return " ⚠️ Ollama server not responding!"


def free_gpu_and_restart_ollama() -> str:
    """Kill GPU-holding processes, clear CUDA cache, and restart Ollama."""
    import os
    import signal
    msgs: list[str] = []

    # 1. Kill lingering Ollama runner processes (hold GPU for loaded models)
    try:
        out = sp.run(
            ["pgrep", "-f", "ollama runner"],
            capture_output=True, text=True, timeout=5,
        )
        for pid in out.stdout.strip().splitlines():
            pid = pid.strip()
            if pid:
                os.kill(int(pid), signal.SIGKILL)
                msgs.append(f"Killed ollama runner (PID {pid})")
    except Exception:
        pass

    # 2. Stop the Ollama server
    try:
        sp.run(["pkill", "-f", "ollama serve"], timeout=5)
        msgs.append("Stopped Ollama server")
        time.sleep(1)
    except Exception:
        pass

    # 3. Clear PyTorch CUDA cache if loaded
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            msgs.append("Cleared CUDA cache")
    except ImportError:
        pass

    # 4. Restart Ollama server
    try:
        sp.Popen(["ollama", "serve"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        time.sleep(2)
        msgs.append("Ollama server restarted")
    except Exception as exc:
        msgs.append(f"Failed to restart Ollama: {exc}")

    # 5. Report GPU state
    try:
        nv = sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, total = nv.stdout.strip().split(",")
        msgs.append(f"GPU memory: {used.strip()} / {total.strip()} MiB")
    except Exception:
        pass

    return "\n".join(msgs) if msgs else "Done (no actions needed)"
