"""mazinger web — launch the Gradio studio UI."""

from __future__ import annotations

import argparse
import logging
import os

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
        "--translation-model",
        default="translategemma",
        help=(
            "Dedicated Ollama translation model to also pull when --with-ollama "
            "is used. Set to an empty string to skip. Default: translategemma."
        ),
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

def _setup_ollama(model: str, extra_models: list[str]) -> None:
    """Install Ollama, start the server, pull the models and warm the main one.

    Setup failures are logged, not fatal: Studio retries the same steps lazily
    when a mission starts (see :mod:`mazinger.ollama_setup`).
    """
    from mazinger.ollama_setup import OllamaSetupError, ensure_ready

    try:
        ensure_ready(model, extra_models=extra_models, warm=True)
    except OllamaSetupError as exc:
        log.error("Ollama setup failed: %s", exc)
        log.warning("Launching Studio anyway — it will retry when a mission starts.")


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
        from mazinger.ollama_setup import DEFAULT_MODEL

        model = args.ollama_model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        os.environ["OLLAMA_MODEL"] = model

        translation_model = (args.translation_model or "").strip()
        extra = [translation_model] if translation_model and translation_model != model else []
        _setup_ollama(model, extra)

        if extra:
            os.environ["MAZINGER_TRANSLATION_MODEL"] = translation_model

    if args.with_faster_whisper:
        _setup_faster_whisper(args.whisper_model)

    from mazinger.studio.app import launch

    launch(
        share=not args.no_share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
