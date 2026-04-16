"""mazinger web — launch the Gradio studio UI."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "web",
        help="Launch the Mazinger Studio web UI (Gradio).",
    )
    p.add_argument(
        "--with-ollama",
        action="store_true",
        default=False,
        help="Install Ollama (if missing), start the server, and pull the model before launching.",
    )
    p.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model to pull when --with-ollama is used (default: env OLLAMA_MODEL or qwen3.5:2b-q8_0).",
    )
    p.add_argument(
        "--with-faster-whisper",
        action="store_true",
        default=False,
        help="Pre-download the Faster Whisper model before launching.",
    )
    p.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Faster Whisper model to download (default: large-v3).",
    )
    p.add_argument(
        "--no-share",
        action="store_true",
        default=False,
        help="Disable Gradio public share link (default: share is enabled).",
    )
    p.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0).",
    )
    p.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging.")


# ── Ollama helpers ────────────────────────────────────────────────

def _ollama_is_running() -> bool:
    """Return True if the Ollama server is responding."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_ollama(retries: int = 15, delay: float = 2) -> bool:
    """Poll Ollama until it responds (up to retries * delay seconds)."""
    for _ in range(retries):
        if _ollama_is_running():
            return True
        time.sleep(delay)
    return False


def _setup_ollama(model: str) -> None:
    """Install Ollama if missing, start the server, pull and warm up the model."""

    # 1. Install Ollama if not present
    if not shutil.which("ollama"):
        log.info("Installing Ollama …")
        has_apt = os.path.exists("/usr/bin/apt-get")
        if has_apt:
            try:
                subprocess.run(
                    ["apt-get", "install", "-y", "-qq", "zstd", "pciutils"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False,
                )
            except FileNotFoundError:
                pass
        subprocess.check_call(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | bash"],
        )
        log.info("Ollama installed")
    else:
        log.info("Ollama already installed")

    # 2. Start the server if not running
    if not _ollama_is_running():
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log.info("Waiting for Ollama server …")
        if _wait_for_ollama():
            log.info("Ollama server is ready")
        else:
            log.warning("Ollama server did not become ready in time — pull may fail")
    else:
        log.info("Ollama server already running")

    # 3. Pull the model (up to 3 attempts)
    log.info("Pulling Ollama model: %s", model)
    pulled = False
    for attempt in range(1, 4):
        result = subprocess.run(
            ["ollama", "pull", model],
            timeout=600, capture_output=True, text=True,
        )
        if result.returncode == 0:
            pulled = True
            log.info("Ollama model ready: %s", model)
            break
        err = (result.stderr or result.stdout or "").strip()
        log.warning("Pull attempt %d/3 failed: %s", attempt, err)
        time.sleep(3)

    if not pulled:
        log.error("Ollama model pull failed after 3 attempts. Try: ollama pull %s", model)
        return

    # 4. Warm up — load the model into memory
    log.info("Warming up Ollama model …")
    try:
        import json
        import urllib.request
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with: ready"}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat", body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        reply = data.get("message", {}).get("content", "")
        log.info("Ollama warm-up OK: %s", reply[:80])
    except Exception as exc:
        log.warning("Warm-up failed: %s", exc)


def _setup_faster_whisper(model: str) -> None:
    """Pre-download a faster-whisper model from HuggingFace."""
    log.info("Downloading Faster Whisper model: %s …", model)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(f"Systran/faster-whisper-{model}")
        log.info("Faster Whisper model ready: %s", model)
    except Exception as exc:
        log.warning("Faster Whisper download failed: %s", exc)


# ── Handler ───────────────────────────────────────────────────────

def handler(args: argparse.Namespace) -> None:
    if args.with_ollama:
        model = (
            args.ollama_model
            or os.environ.get("OLLAMA_MODEL")
            or "qwen3.5:2b-q8_0"
        )
        os.environ["OLLAMA_MODEL"] = model
        _setup_ollama(model)

    if args.with_faster_whisper:
        _setup_faster_whisper(args.whisper_model)

    from mazinger.studio.app import launch

    launch(
        share=not args.no_share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
