"""Helper classes and functions for Mazinger Studio."""

import logging
import os
import subprocess as sp
import threading
import time

from mazinger.studio.constants import GATED_MODELS, OLLAMA_DEFAULT_MODEL


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


class LLMStreamCollector:
    """Thread-safe buffer that accumulates streamed LLM tokens.

    Used as the callback for :func:`mazinger.llm.set_stream_callback`.
    The Gradio polling loop reads :meth:`read` to show live output.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: list[str] = []

    # Callable — this is the stream callback itself
    def __call__(self, token: str) -> None:
        with self._lock:
            self._chunks.append(token)

    def read(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()


def ensure_ollama(model_id: str | None = None, extra_models=(), progress=None):
    """Install Ollama (if missing), start the server and pull the models.

    Self-healing so a fresh installation works even when Studio was launched
    without ``mazinger web --with-ollama``. Raises
    :class:`mazinger.ollama_setup.OllamaSetupError` with an actionable message.
    """
    from mazinger.ollama_setup import ensure_ready

    return ensure_ready(
        model_id or OLLAMA_DEFAULT_MODEL,
        extra_models=extra_models,
        progress=progress,
    )


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
    ("Synthesising segment",        "⏳ Synthesizing speech… (TTS)"),
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
                # Enrich TTS status with segment progress numbers
                if pattern == "Synthesising segment":
                    import re
                    m = re.search(r"Synthesising segment (\d+)/(\d+)", line)
                    if m:
                        return f"⏳ Synthesizing speech… segment {m.group(1)}/{m.group(2)}"
                return label
    return "⏳ Processing…"


def check_ollama_health() -> str | None:
    """Return a warning string if Ollama is not responding, else None."""
    from mazinger.ollama_setup import is_running

    return None if is_running() else " ⚠️ Ollama server not responding!"


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
        from mazinger.ollama_setup import start_server
        start_server()
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


# ═══════════════════════════════════════════════════════════════════════
#  HuggingFace authentication
# ═══════════════════════════════════════════════════════════════════════
#
# ``huggingface_hub.notebook_login()`` renders its prompt through
# ``IPython.display`` into a notebook output cell, and falls back to a
# blocking terminal prompt when IPython is absent.  Neither reaches a Gradio
# UI — and a shared Gradio link has no notebook to draw into at all.  So the
# Studio drives the same OAuth device-code flow that ``notebook_login()``
# uses internally, and renders it inside the app instead.

_HF_MODEL_LINKS = "\n".join(
    f"- [{label}](https://huggingface.co/{repo})" for label, repo in GATED_MODELS
)

# Same repositories on one line, for the Studio's Hugging Face card.
HF_MODEL_LINKS_INLINE = "Accept the licence on each model page: " + "  ·  ".join(
    f"[{label}](https://huggingface.co/{repo})" for label, repo in GATED_MODELS
)

_ACCESS_HINT = (
    "Signing in is not enough on its own — open each gated model once and "
    "click **Agree and access repository**:\n\n" + _HF_MODEL_LINKS
)


def _apply_token(token: str) -> None:
    """Make *token* visible to every part of Mazinger that reads it."""
    os.environ["HF_TOKEN"] = token
    # mazinger.profiles captures HF_TOKEN into a module global at import
    # time, so setting the environment variable alone would not reach it.
    try:
        from mazinger import profiles
        profiles.HF_TOKEN = token
    except Exception:  # noqa: BLE001 — auth must not break on an import error
        pass


def hf_status() -> str:
    """Return a Markdown line describing the current HuggingFace login."""
    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        return "⚠️  `huggingface_hub` is not installed."

    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        return "🔒  **Not signed in.** Gated models (Cohere Transcribe) will fail to download."

    _apply_token(token)
    try:
        user = whoami(token=token)
        return f"✅  Signed in as **{user.get('name', 'unknown')}**."
    except Exception as exc:  # noqa: BLE001 — expired/invalid token
        return f"⚠️  A token is set but could not be verified: {exc}"


def hf_login_flow():
    """Drive the Hub's device-code login, yielding Markdown status updates.

    Used as a Gradio generator handler so the verification code appears
    immediately and the panel keeps updating while we wait for the user to
    authorise in another tab.
    """
    try:
        from huggingface_hub import get_token, login, whoami
    except ImportError:
        yield "⚠️  `huggingface_hub` is not installed. Run `pip install huggingface_hub`."
        return

    existing = get_token() or os.environ.get("HF_TOKEN")
    if existing:
        _apply_token(existing)
        try:
            user = whoami(token=existing)
            yield (f"✅  Already signed in as **{user.get('name', 'unknown')}**. "
                   f"Use *Sign out* to switch accounts.\n\n{_ACCESS_HINT}")
            return
        except Exception:  # noqa: BLE001 — stale token, fall through to re-login
            pass

    try:
        from huggingface_hub._login import poll_device_token, request_device_code
    except ImportError:
        yield ("⚠️  This version of `huggingface_hub` has no device-code login. "
               "Paste an access token below instead — "
               "[create one here](https://huggingface.co/settings/tokens).")
        return

    try:
        info = request_device_code()
    except Exception as exc:  # noqa: BLE001 — offline, proxy, endpoint down
        yield (f"⚠️  Could not start HuggingFace login: {exc}\n\n"
               "Paste an access token below instead — "
               "[create one here](https://huggingface.co/settings/tokens).")
        return

    uri = info.get("verification_uri_complete") or info["verification_uri"]
    code = info["user_code"]
    panel = (
        f"### 👉  [Open this link to authorise]({uri})\n\n"
        f"Then confirm this code:\n\n## `{code}`\n\n"
    )

    # Show the link and code before polling starts — otherwise a fast
    # authorisation could resolve the flow without the user ever seeing them.
    yield panel + "⏳  Waiting for authorisation…"

    result: dict = {}

    def _poll() -> None:
        try:
            result["token"] = poll_device_token(info)["access_token"]
        except Exception as exc:  # noqa: BLE001 — denied, expired, network
            result["error"] = exc

    worker = threading.Thread(target=_poll, daemon=True)
    worker.start()

    deadline = time.monotonic() + info.get("expires_in", 900)
    while worker.is_alive() and time.monotonic() < deadline:
        worker.join(timeout=2)
        if worker.is_alive():
            yield panel + (
                f"⏳  Waiting for authorisation… "
                f"({int(deadline - time.monotonic())}s left)"
            )

    if "token" in result:
        token = result["token"]
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001 — cache write failed
            log_msg = f" (token not persisted to disk: {exc})"
        else:
            log_msg = ""
        _apply_token(token)
        try:
            name = whoami(token=token).get("name", "unknown")
        except Exception:  # noqa: BLE001
            name = "unknown"
        yield f"✅  Signed in as **{name}**.{log_msg}\n\n{_ACCESS_HINT}"
    elif "error" in result:
        yield f"❌  Login failed: {result['error']}"
    else:
        yield "❌  Login timed out. Click **Sign in with Hugging Face** to try again."


def hf_login_with_token(token: str) -> str:
    """Sign in with a pasted access token (fallback for the device flow)."""
    token = (token or "").strip()
    if not token:
        return "⚠️  Paste an access token first."
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        return "⚠️  `huggingface_hub` is not installed."

    try:
        name = whoami(token=token).get("name", "unknown")
    except Exception as exc:  # noqa: BLE001 — invalid token
        return f"❌  That token was rejected: {exc}"

    try:
        login(token=token, add_to_git_credential=False)
    except Exception:  # noqa: BLE001 — non-fatal, env var still applies
        pass
    _apply_token(token)
    return f"✅  Signed in as **{name}**.\n\n{_ACCESS_HINT}"


def hf_logout() -> str:
    """Forget the stored HuggingFace token."""
    os.environ.pop("HF_TOKEN", None)
    try:
        from mazinger import profiles
        profiles.HF_TOKEN = None
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub import logout
        logout()
    except Exception:  # noqa: BLE001 — nothing stored on disk
        pass
    return "🔒  Signed out."
