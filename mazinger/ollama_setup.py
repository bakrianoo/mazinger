"""Ollama runtime management — install, serve and pull models on demand.

Single source of truth shared by ``mazinger web --with-ollama`` (eager setup at
launch) and Mazinger Studio (lazy setup when a mission starts), so a fresh
installation works no matter how Mazinger was started.

Every step is idempotent: an already-installed binary, a running server and an
already-pulled model are all detected and skipped.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b-q8_0")
DEFAULT_TRANSLATION_MODEL = "translategemma"

_INSTALL_URL = "https://ollama.com/install.sh"
_PULL_TIMEOUT = int(os.environ.get("OLLAMA_PULL_TIMEOUT", "1800"))

#: Appended to errors the user can resolve from the Studio UI.
_FALLBACK_HINT = (
    "Install it manually (https://ollama.com/download) or switch the LLM "
    "provider to \"OpenAI (Cloud)\"."
)


class OllamaSetupError(RuntimeError):
    """Ollama could not be installed, started, or a model could not be pulled."""


def _report(progress, message: str) -> None:
    log.info(message)
    if progress:
        try:
            progress(message)
        except Exception:  # never let UI plumbing break setup
            pass


# ── Discovery ─────────────────────────────────────────────────────

def host() -> str:
    """Base URL of the Ollama server (honours ``OLLAMA_HOST``)."""
    raw = (os.environ.get("OLLAMA_HOST") or "").strip() or "localhost:11434"
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def is_local() -> bool:
    """True when the server we manage runs in this container."""
    return any(h in host() for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def is_installed() -> bool:
    """True when the ``ollama`` binary is on ``PATH``."""
    return shutil.which("ollama") is not None


def is_running(timeout: float = 3) -> bool:
    """True when the Ollama server answers on :func:`host`."""
    try:
        with urllib.request.urlopen(f"{host()}/api/tags", timeout=timeout) as r:
            r.read()
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _normalise(model: str) -> str:
    """``llama3`` and ``llama3:latest`` name the same model."""
    model = model.strip()
    return model if ":" in model else f"{model}:latest"


def installed_models() -> set[str]:
    """Model tags known to the running server (empty set if unreachable)."""
    try:
        with urllib.request.urlopen(f"{host()}/api/tags", timeout=10) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return set()
    return {_normalise(m.get("name", "")) for m in data.get("models", []) if m.get("name")}


def has_model(model: str) -> bool:
    return _normalise(model) in installed_models()


# ── Setup steps ───────────────────────────────────────────────────

def install(progress=None) -> None:
    """Install Ollama via the official script. No-op when already installed."""
    if is_installed():
        return

    if not is_local():
        raise OllamaSetupError(
            f"Ollama is not installed here and OLLAMA_HOST points at {host()}, "
            f"which is not reachable. Start Ollama on that host, or unset "
            f"OLLAMA_HOST to run it locally."
        )
    if shutil.which("curl") is None:
        raise OllamaSetupError(f"'curl' is required to install Ollama automatically. {_FALLBACK_HINT}")
    if os.geteuid() != 0 and shutil.which("sudo") is None:
        raise OllamaSetupError(f"Installing Ollama needs root (or sudo), which is unavailable. {_FALLBACK_HINT}")

    _report(progress, "Installing Ollama (one-time download, ~1 min)…")

    # The install script unpacks a .zst archive and probes the GPU.
    if shutil.which("apt-get"):
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "zstd", "pciutils"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )

    result = subprocess.run(
        ["bash", "-c", f"curl -fsSL {_INSTALL_URL} | bash"],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0 or not is_installed():
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        raise OllamaSetupError(
            "Ollama installation failed: " + (" / ".join(tail) or "unknown error")
            + f". {_FALLBACK_HINT}"
        )
    _report(progress, "Ollama installed")


def start_server(progress=None, retries: int = 15, delay: float = 2) -> None:
    """Start ``ollama serve`` in the background and wait for it. No-op if running."""
    if is_running():
        return

    if not is_local():
        raise OllamaSetupError(
            f"No Ollama server responding at {host()}. Start it on that host, "
            f"or unset OLLAMA_HOST to run one locally."
        )
    install(progress)

    _report(progress, "Starting Ollama server…")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise OllamaSetupError(f"Could not start the Ollama server: {exc}. {_FALLBACK_HINT}") from exc

    for _ in range(retries):
        if is_running():
            _report(progress, "Ollama server is ready")
            return
        time.sleep(delay)

    raise OllamaSetupError(
        f"The Ollama server did not become ready within {int(retries * delay)}s. "
        f"Check it manually with: ollama serve"
    )


def pull_model(model: str, progress=None, attempts: int = 3) -> None:
    """Pull ``model`` (retrying transient registry failures). No-op if present."""
    model = model.strip()
    if not model or has_model(model):
        return

    _report(progress, f"Downloading model '{model}' (one-time, may take several minutes)…")

    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True, text=True, timeout=_PULL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timed out after {_PULL_TIMEOUT}s"
        except OSError as exc:
            last_err = str(exc)
        else:
            if result.returncode == 0:
                _report(progress, f"Model ready: {model}")
                return
            output = (result.stderr or result.stdout or "").strip().splitlines()
            last_err = output[-1] if output else "unknown error"

        log.warning("Pull attempt %d/%d failed for %s: %s", attempt, attempts, model, last_err)
        if attempt < attempts:
            time.sleep(3)

    raise OllamaSetupError(
        f"Could not download the model '{model}' after {attempts} attempts: {last_err}. "
        f"Check the model name and your connection, then retry: ollama pull {model}"
    )


def warm_up(model: str, timeout: float = 120) -> None:
    """Load ``model`` into memory so the first real request is fast. Best-effort."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with: ready"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        f"{host()}/api/chat", body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        log.info("Ollama warm-up OK: %s", data.get("message", {}).get("content", "")[:80])
    except Exception as exc:
        log.warning("Ollama warm-up failed: %s", exc)


# ── One-call entry point ──────────────────────────────────────────

def ensure_ready(
    model: str | None = None,
    extra_models=(),
    progress=None,
    warm: bool = False,
) -> str:
    """Guarantee Ollama is installed, running and holds every requested model.

    Returns the resolved primary model name. Raises :class:`OllamaSetupError`
    with an actionable message if any step cannot be completed.
    """
    model = (model or "").strip() or DEFAULT_MODEL

    install(progress)
    start_server(progress)

    for name in (model, *extra_models):
        pull_model(name, progress)

    if warm:
        _report(progress, "Warming up the model…")
        warm_up(model)

    return model
