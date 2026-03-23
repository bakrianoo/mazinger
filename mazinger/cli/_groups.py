"""Reusable argument groups shared across CLI commands."""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_BASE_DIR = "./mazinger_output"

log = __import__("logging").getLogger(__name__)


def detect_device() -> str:
    """Return 'cuda' if CUDA is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_device(value: str) -> str:
    """Resolve 'auto' to the actual device, pass others through."""
    return detect_device() if value == "auto" else value


def add_source(p: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add optional/required positional *source* plus project-resolution flags."""
    kw = {} if required else {"nargs": "?", "default": None}
    p.add_argument("source", help="Video URL, local video path, or local audio path.", **kw)
    p.add_argument("--slug", default=None, help="Override project slug.")
    p.add_argument(
        "--quality", default=None,
        help="Video download quality: low (360p), medium (720p, default), "
             "high (best), or a resolution like 144, 480, 1080.",
    )
    add_cookies(p)


def _apply_slice(proj, *, start: str | None, end: str | None) -> None:
    """Slice the project video and/or audio in-place."""
    from mazinger.download import slice_project
    slice_project(proj, start=start, end=end)


def resolve_project(args: argparse.Namespace):
    """Download/ingest *source* and return a :class:`ProjectPaths`.

    Returns ``None`` when no source was provided (caller should fall back to
    explicit ``--srt`` / ``--video`` flags).
    """
    source = getattr(args, "source", None)
    if not source:
        return None

    from mazinger import download
    from mazinger.paths import ProjectPaths

    is_remote = download.is_url(source)
    slug = getattr(args, "slug", None)
    base_dir = getattr(args, "base_dir", DEFAULT_BASE_DIR)

    if slug is None:
        if is_remote:
            slug, _ = download.resolve_slug(
                source,
                cookies_from_browser=getattr(args, "cookies_from_browser", None),
                cookies=getattr(args, "cookies", None),
            )
        else:
            slug = download.slug_from_path(source)

    proj = ProjectPaths(slug, base_dir=base_dir).ensure_dirs()
    is_local_audio = not is_remote and download.is_audio_file(source)

    if is_local_audio:
        if not os.path.exists(proj.audio):
            download.ingest_local_audio(source, proj.audio)
    elif is_remote:
        if not os.path.exists(proj.video):
            download.download_video(
                source, proj.video,
                quality=getattr(args, "quality", None),
                cookies_from_browser=getattr(args, "cookies_from_browser", None),
                cookies=getattr(args, "cookies", None),
            )
        download.extract_audio(proj.video, proj.audio)
    else:
        if not os.path.exists(proj.video):
            download.ingest_local_video(source, proj.video, proj.audio)
        else:
            download.extract_audio(proj.video, proj.audio)

    # -- Apply time slicing if requested ----------------------------------
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if start or end:
        _apply_slice(proj, start=start, end=end)

    log.info("Project: %s", proj.root)
    return proj


def ensure_transcription(proj, args: argparse.Namespace) -> None:
    """Run transcription for *proj* if the SRT does not exist yet."""
    if os.path.exists(proj.source_srt):
        log.info("Skipping transcription (SRT exists)")
        return
    from mazinger.transcribe import transcribe
    transcribe(
        proj.audio, proj.source_srt,
        method=getattr(args, "transcribe_method", "openai"),
        model=getattr(args, "whisper_model", None),
        device=getattr(args, "device", "cuda"),
        openai_api_key=getattr(args, "openai_api_key", None),
        openai_base_url=getattr(args, "openai_base_url", None),
    )


def _language_type(value: str) -> str:
    from mazinger.translate import resolve_language
    try:
        return resolve_language(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _source_language_type(value: str) -> str:
    from mazinger.translate import resolve_source_language
    try:
        return resolve_source_language(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help=f"Root directory for project folders (default: {DEFAULT_BASE_DIR}).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging.")


def add_openai(p: argparse.ArgumentParser) -> None:
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Base URL for OpenAI-compatible API.")


def add_llm(p: argparse.ArgumentParser) -> None:
    add_openai(p)
    p.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"), help="LLM model for translation/analysis.")


def add_voice(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--clone-profile", default=None,
        help="Name of a voice profile from the HuggingFace dataset (e.g. 'abubakr').",
    )
    p.add_argument("--voice-sample", default=None, help="Path to voice reference audio.")
    p.add_argument("--voice-script", default=None, help="Path to voice reference transcript.")


def add_tts_engine(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Qwen TTS model name.")
    p.add_argument("--chatterbox-model", default="ResembleAI/chatterbox", help="Chatterbox TTS model name.")
    p.add_argument("--tts-language", default=None, type=_language_type,
                   help="Target TTS language (defaults to --target-language).")
    p.add_argument(
        "--tts-engine", default="qwen", choices=["qwen", "chatterbox"],
        help="TTS engine: 'qwen' (Qwen3-TTS) or 'chatterbox' (ResembleAI Chatterbox).",
    )
    p.add_argument("--chatterbox-exaggeration", type=float, default=0.5,
                   help="Chatterbox exaggeration level (0.0-1.0, default 0.5).")
    p.add_argument("--chatterbox-cfg", type=float, default=0.5,
                   help="Chatterbox CFG weight (0.0-1.0, default 0.5).")


def add_tempo(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dynamic-tempo", action="store_true",
                   help="Enable per-segment dynamic tempo adjustment.")
    p.add_argument("--fixed-tempo", type=float, default=None,
                   help="Apply a fixed tempo rate to all segments (e.g. 1.1). Overrides --dynamic-tempo.")
    p.add_argument("--max-tempo", type=float, default=1.3,
                   help="Maximum speed-up factor for dynamic tempo (default: 1.3).")


def add_transcription(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--transcribe-method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
        help="Transcription backend: 'openai', 'faster-whisper', or 'whisperx'.",
    )
    p.add_argument("--whisper-model", default=None, help="Whisper model name.")


def add_cookies(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cookies-from-browser", default=None,
                   help="Pass through to yt-dlp --cookies-from-browser.")
    p.add_argument("--cookies", default=None,
                   help="Pass through to yt-dlp --cookies (path to Netscape cookie file).")


def add_slice(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", default=None,
                   help="Start timestamp for slicing (e.g. '00:01:30' or '90').")
    p.add_argument("--end", default=None,
                   help="End timestamp for slicing (e.g. '00:05:00' or '300').")


def add_translation(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source-language", default="auto", type=_source_language_type,
        help="Source language for translation, or 'auto' to detect (default: auto).",
    )
    p.add_argument(
        "--target-language", default="English", type=_language_type,
        help="Target language for translation (default: English).",
    )
    p.add_argument("--words-per-second", type=float, default=None,
                   help="Target speech rate in words/sec for duration matching (default: 2.0).")
    p.add_argument("--duration-budget", type=float, default=None,
                   help="Fraction of time window to fill with translated speech, 0.0-1.0 (default: 0.80).")
    p.add_argument("--translate-technical-terms", action="store_true", default=False,
                   help="Translate technical terms into the target language. "
                        "When omitted, technical terms are kept in their original language.")


def resolve_voice(args: argparse.Namespace) -> tuple[str | None, str | None]:
    voice_sample = args.voice_sample
    voice_script = args.voice_script
    if args.clone_profile:
        from mazinger.profiles import fetch_profile
        pv, ps = fetch_profile(args.clone_profile)
        voice_sample = voice_sample or pv
        voice_script = voice_script or ps
    return voice_sample, voice_script


def require_voice(args: argparse.Namespace) -> tuple[str, str]:
    sample, script = resolve_voice(args)
    if not sample or not script:
        sys.exit("Error: --voice-sample and --voice-script are required (or use --clone-profile).")
    return sample, script


def make_openai_client(args: argparse.Namespace):
    from openai import OpenAI
    kw = {}
    if args.openai_api_key:
        kw["api_key"] = args.openai_api_key
    if args.openai_base_url:
        kw["base_url"] = args.openai_base_url
    return OpenAI(**kw)


def tempo_mode_from_args(args: argparse.Namespace) -> str:
    if args.fixed_tempo:
        return "fixed"
    if args.dynamic_tempo:
        return "dynamic"
    return "auto"


def add_subtitle_style(p: argparse.ArgumentParser) -> None:
    """Add subtitle styling arguments."""
    g = p.add_argument_group("subtitle styling")
    g.add_argument("--subtitle-font", default="Arial",
                   help="Subtitle font family (default: Arial).")
    g.add_argument("--subtitle-font-file", default=None,
                   help="Path to a local TTF/OTF font file.")
    g.add_argument("--subtitle-google-font", default=None,
                   help="Google Font name to download and use (e.g. 'Noto Sans Arabic').")
    g.add_argument("--subtitle-font-size", type=int, default=14,
                   help="Subtitle font size (default: 14).")
    g.add_argument("--subtitle-font-color", default="white",
                   help="Subtitle text color: name or #RRGGBB (default: white).")
    g.add_argument("--subtitle-bg-color", default="black",
                   help="Subtitle background color (default: black).")
    g.add_argument("--subtitle-bg-alpha", type=float, default=0.6,
                   help="Subtitle background opacity, 0.0-1.0 (default: 0.6).")
    g.add_argument("--subtitle-outline-color", default="black",
                   help="Subtitle outline color (default: black).")
    g.add_argument("--subtitle-outline-width", type=int, default=1,
                   help="Subtitle outline width (default: 1).")
    g.add_argument("--subtitle-position", default="bottom",
                   choices=["bottom", "top", "center"],
                   help="Subtitle position (default: bottom).")
    g.add_argument("--subtitle-margin", type=int, default=20,
                   help="Subtitle vertical margin in pixels (default: 20).")
    g.add_argument("--subtitle-bold", action="store_true",
                   help="Use bold subtitle text.")
    g.add_argument("--subtitle-line-spacing", type=int, default=8,
                   help="Extra vertical spacing between subtitle lines in pixels (default: 8).")


def add_subtitles(p: argparse.ArgumentParser) -> None:
    """Add --burn-subtitles flag, --subtitle-source, and styling arguments."""
    p.add_argument("--embed-subtitles", action="store_true",
                   help="Embed subtitles into the output video (implies --output-type video).")
    p.add_argument("--subtitle-source", default="translated",
                   help="SRT to burn: 'original', 'translated' (default), or a file path.")
    add_subtitle_style(p)


def subtitle_style_from_args(args: argparse.Namespace):
    """Construct a :class:`~mazinger.subtitle.SubtitleStyle` from parsed CLI arguments."""
    from mazinger.subtitle import SubtitleStyle, download_google_font

    font_file = getattr(args, "subtitle_font_file", None)
    google_font = getattr(args, "subtitle_google_font", None)
    if google_font and not font_file:
        font_file = download_google_font(google_font)

    return SubtitleStyle(
        font=args.subtitle_font,
        font_file=font_file,
        font_size=args.subtitle_font_size,
        font_color=args.subtitle_font_color,
        bg_color=args.subtitle_bg_color,
        bg_alpha=args.subtitle_bg_alpha,
        outline_color=args.subtitle_outline_color,
        outline_width=args.subtitle_outline_width,
        position=args.subtitle_position,
        margin_v=args.subtitle_margin,
        bold=args.subtitle_bold,
        line_spacing=args.subtitle_line_spacing,
    )
