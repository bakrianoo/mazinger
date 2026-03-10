"""Command-line interface for Mazinger Dubber."""

from __future__ import annotations

import argparse
import logging
import sys


DEFAULT_BASE_DIR = "./mazinger_dubber_output"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help=f"Root directory for project folders (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Sub-command handlers
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_dub(args: argparse.Namespace) -> None:
    from mazinger_dubber.pipeline import MazingerDubber

    voice_sample = args.voice_sample
    voice_script = args.voice_script

    if args.clone_profile:
        from mazinger_dubber.profiles import fetch_profile
        pv, ps = fetch_profile(args.clone_profile)
        voice_sample = voice_sample or pv
        voice_script = voice_script or ps

    if not voice_sample or not voice_script:
        sys.exit("Error: --voice-sample and --voice-script are required "
                 "(or use --clone-profile).")

    rs = MazingerDubber(
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        llm_model=args.llm_model,
        base_dir=args.base_dir,
    )
    proj = rs.dub(
        source=args.source,
        voice_sample=voice_sample,
        voice_script=voice_script,
        slug=args.slug,
        device=args.device,
        transcribe_method=args.transcribe_method,
        whisper_model=args.whisper_model,
        tts_model_name=args.tts_model,
        tts_language=args.tts_language,
        tts_engine=args.tts_engine,
        chatterbox_model=args.chatterbox_model,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg=args.chatterbox_cfg,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        force_reset=args.force_reset,
        use_resegmented=args.use_resegmented,
        output_type=args.output_type,
        tempo_mode="fixed" if args.fixed_tempo else ("dynamic" if args.dynamic_tempo else "auto"),
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
        words_per_second=args.words_per_second,
        duration_budget=args.duration_budget,
    )
    print(proj.summary())


def _cmd_download(args: argparse.Namespace) -> None:
    from mazinger_dubber.download import (
        is_url, is_audio_file, resolve_slug, slug_from_path,
        download_video, extract_audio, ingest_local_video, ingest_local_audio,
    )
    from mazinger_dubber.paths import ProjectPaths

    source = args.source
    remote = is_url(source)

    slug = args.slug
    if slug is None:
        if remote:
            slug, _ = resolve_slug(
                source,
                cookies_from_browser=args.cookies_from_browser,
                cookies=args.cookies,
            )
        else:
            slug = slug_from_path(source)

    proj = ProjectPaths(slug, base_dir=args.base_dir).ensure_dirs()

    if not remote and is_audio_file(source):
        ingest_local_audio(source, proj.audio)
    elif remote:
        download_video(
            source,
            proj.video,
            cookies_from_browser=args.cookies_from_browser,
            cookies=args.cookies,
        )
        extract_audio(proj.video, proj.audio)
    else:
        ingest_local_video(source, proj.video, proj.audio)

    print(proj.summary())


def _cmd_transcribe(args: argparse.Namespace) -> None:
    from mazinger_dubber.transcribe import transcribe

    transcribe(
        args.audio,
        args.output,
        method=args.method,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        language=args.language,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        skip_resegment=args.no_resegment,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    print(f"SRT saved: {args.output}")


def _cmd_thumbnails(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from mazinger_dubber.thumbnails import select_timestamps, extract_frames
    from mazinger_dubber.utils import save_json

    client_kwargs = {}
    if args.openai_api_key:
        client_kwargs["api_key"] = args.openai_api_key
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client = OpenAI(**client_kwargs)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()

    timestamps = select_timestamps(srt_text, client, llm_model=args.llm_model)
    results = extract_frames(args.video, timestamps, args.output_dir)

    meta_path = args.meta or f"{args.output_dir}/meta.json"
    save_json(results, meta_path)
    print(f"Extracted {len(results)} thumbnails -> {args.output_dir}")


def _cmd_describe(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from mazinger_dubber.describe import describe_content
    from mazinger_dubber.utils import load_json, save_json

    client_kwargs = {}
    if args.openai_api_key:
        client_kwargs["api_key"] = args.openai_api_key
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client = OpenAI(**client_kwargs)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()
    thumb_paths = load_json(args.thumbnails_meta)

    desc = describe_content(srt_text, thumb_paths, client, llm_model=args.llm_model)
    save_json(desc, args.output)
    print(f"Description saved: {args.output}")


def _cmd_translate(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from mazinger_dubber.translate import translate_srt
    from mazinger_dubber.utils import load_json

    client_kwargs = {}
    if args.openai_api_key:
        client_kwargs["api_key"] = args.openai_api_key
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client = OpenAI(**client_kwargs)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()
    description = load_json(args.description)
    thumb_paths = load_json(args.thumbnails_meta) if args.thumbnails_meta else []

    result = translate_srt(
        srt_text, description, thumb_paths, client, llm_model=args.llm_model,
        target_language=args.target_language,
        **(dict(words_per_second=args.words_per_second) if args.words_per_second is not None else {}),
        **(dict(duration_budget=args.duration_budget) if args.duration_budget is not None else {}),
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"Translated SRT saved: {args.output}")


def _cmd_resegment(args: argparse.Namespace) -> None:
    from mazinger_dubber.resegment import resegment_srt

    client = None
    if args.openai_api_key or args.openai_base_url:
        from openai import OpenAI
        client_kwargs = {}
        if args.openai_api_key:
            client_kwargs["api_key"] = args.openai_api_key
        if args.openai_base_url:
            client_kwargs["base_url"] = args.openai_base_url
        client = OpenAI(**client_kwargs)

    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()

    result = resegment_srt(
        srt_text,
        client=client,
        llm_model=args.llm_model,
        max_chars=args.max_chars,
        max_dur=args.max_dur,
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"Re-segmented SRT saved: {args.output}")


def _cmd_tts(args: argparse.Namespace) -> None:
    from mazinger_dubber import tts, assemble
    from mazinger_dubber.srt import parse_file
    from mazinger_dubber.utils import get_audio_duration

    srt_entries = parse_file(args.srt)
    original_duration = get_audio_duration(args.original_audio)

    voice_sample = args.voice_sample
    voice_script = args.voice_script

    if getattr(args, "clone_profile", None):
        from mazinger_dubber.profiles import fetch_profile
        pv, ps = fetch_profile(args.clone_profile)
        voice_sample = voice_sample or pv
        voice_script = voice_script or ps

    if not voice_sample or not voice_script:
        sys.exit("Error: --voice-sample and --voice-script are required "
                 "(or use --clone-profile).")

    with open(voice_script, encoding="utf-8") as fh:
        ref_text = fh.read().strip()

    engine = getattr(args, "tts_engine", "qwen")
    model = tts.load_model(
        args.tts_model, device=args.device, dtype=args.dtype, engine=engine,
        chatterbox_model=getattr(args, "chatterbox_model", "ResembleAI/chatterbox"),
    )
    voice_prompt = tts.create_voice_prompt(
        model, voice_sample, ref_text,
        engine=engine,
        chatterbox_exaggeration=getattr(args, "chatterbox_exaggeration", 0.5),
        chatterbox_cfg=getattr(args, "chatterbox_cfg", 0.5),
    )
    segment_info = tts.synthesize_segments(
        model, voice_prompt, srt_entries, args.segments_dir,
        language=args.language,
        force_reset=getattr(args, "force_reset", False),
    )
    tts.unload_model(voice_prompt)

    tempo_mode = "fixed" if args.fixed_tempo else ("dynamic" if args.dynamic_tempo else "auto")
    assemble.assemble_timeline(
        segment_info, original_duration, args.output,
        tempo_mode=tempo_mode,
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
    )
    print(f"Dubbed audio saved: {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser construction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mazinger-dubber",
        description="Mazinger Dubber -- End-to-end video dubbing pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- dub (full pipeline) ---------------------------------------------------
    p = subparsers.add_parser("dub", help="Run the full dubbing pipeline.")
    p.add_argument("source", help="Video URL, local video path, or local audio path.")
    p.add_argument("--clone-profile", default=None,
                   help="Name of a voice profile to fetch from the HuggingFace dataset "
                        "(e.g. 'abubakr'). Provides --voice-sample and --voice-script "
                        "automatically.")
    p.add_argument("--voice-sample", default=None, help="Path to voice reference audio.")
    p.add_argument("--voice-script", default=None, help="Path to voice reference transcript.")
    p.add_argument("--slug", default=None, help="Override project slug.")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu.")
    p.add_argument(
        "--transcribe-method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
        help="Transcription backend: 'openai' (cloud API, default), 'faster-whisper' (fast local), or 'whisperx' (local with alignment).",
    )
    p.add_argument("--whisper-model", default=None, help="Whisper model name.")
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Qwen TTS model name.")
    p.add_argument("--chatterbox-model", default="ResembleAI/chatterbox", help="Chatterbox TTS model name.")
    p.add_argument("--tts-language", default="English", help="Target TTS language.")
    p.add_argument(
        "--tts-engine", default="qwen", choices=["qwen", "chatterbox"],
        help="TTS engine: 'qwen' (Qwen3-TTS) or 'chatterbox' (ResembleAI Chatterbox).",
    )
    p.add_argument(
        "--chatterbox-exaggeration", type=float, default=0.5,
        help="Chatterbox exaggeration level (0.0-1.0, default 0.5).",
    )
    p.add_argument(
        "--chatterbox-cfg", type=float, default=0.5,
        help="Chatterbox CFG weight (0.0-1.0, default 0.5). Lower for slower/expressive speech.",
    )
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model for translation/analysis.")
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    p.add_argument(
        "--cookies-from-browser", default=None,
        help="Pass through to yt-dlp --cookies-from-browser (e.g., chrome or firefox:default).",
    )
    p.add_argument(
        "--cookies", default=None,
        help="Pass through to yt-dlp --cookies (path to Netscape cookie file).",
    )
    p.add_argument(
        "--use-resegmented", action="store_true",
        help="Translate and dub from the resegmented SRT (source.srt) instead of "
             "the raw WhisperX output (source.raw.srt). Resegmented SRT has "
             "shorter, more readable segments.",
    )
    p.add_argument(
        "--output-type", choices=["audio", "video"], default="audio",
        help="Output type: 'audio' (default) produces dubbed WAV; "
             "'video' also muxes the dubbed audio into the source video.",
    )
    p.add_argument(
        "--dynamic-tempo", action="store_true",
        help="Enable per-segment dynamic tempo adjustment to match original timing.",
    )
    p.add_argument(
        "--fixed-tempo", type=float, default=None,
        help="Apply a fixed tempo rate to all segments (e.g. 1.1). Overrides --dynamic-tempo.",
    )
    p.add_argument(
        "--max-tempo", type=float, default=1.3,
        help="Maximum speed-up factor for dynamic tempo (default: 1.3).",
    )
    p.add_argument(
        "--words-per-second", type=float, default=None,
        help="Target English speech rate in words/sec for translation duration matching (default: 2.0).",
    )
    p.add_argument(
        "--duration-budget", type=float, default=None,
        help="Fraction of time window to fill with translated speech, 0.0-1.0 (default: 0.80).",
    )
    p.add_argument(
        "--force-reset", action="store_true",
        help="Discard all cached/intermediate outputs and re-run every stage from scratch.",
    )
    _add_common_args(p)

    # -- download ---------------------------------------------------------------
    p = subparsers.add_parser("download", help="Download video / ingest local file and extract audio.")
    p.add_argument("source", help="Video URL, local video path, or local audio path.")
    p.add_argument("--slug", default=None, help="Override project slug.")
    p.add_argument(
        "--cookies-from-browser", default=None,
        help="Pass through to yt-dlp --cookies-from-browser (e.g., chrome or firefox:default).",
    )
    p.add_argument(
        "--cookies", default=None,
        help="Pass through to yt-dlp --cookies (path to Netscape cookie file).",
    )
    _add_common_args(p)

    # -- transcribe -------------------------------------------------------------
    p = subparsers.add_parser("transcribe", help="Transcribe audio to SRT.")
    p.add_argument("audio", help="Path to audio file.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    p.add_argument(
        "--method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
        help="Transcription backend: 'openai' (cloud API, default), 'faster-whisper' (fast local), or 'whisperx' (local with alignment).",
    )
    p.add_argument(
        "--model", default=None,
        help="Model name. Defaults to 'whisper-1' for OpenAI, 'large-v3' for local methods.",
    )
    p.add_argument("--device", default="cuda", help="Device for local methods (cuda/cpu).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for local methods.")
    p.add_argument("--compute-type", default="float16", help="Compute type: float16, int8, int8_float16.")
    p.add_argument("--language", default=None, help="Force language code (e.g., en, ar).")
    p.add_argument("--max-chars", type=int, default=120, help="Max chars per subtitle.")
    p.add_argument("--max-duration", type=float, default=10.0, help="Max seconds per subtitle.")
    p.add_argument("--no-resegment", action="store_true", help="Skip resegmentation.")
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key (for --method openai).")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    _add_common_args(p)

    # -- thumbnails -------------------------------------------------------------
    p = subparsers.add_parser("thumbnails", help="Extract key-frame thumbnails.")
    p.add_argument("--video", required=True, help="Path to video file.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("--output-dir", required=True, help="Output directory for thumbnails.")
    p.add_argument("--meta", default=None, help="Path to save metadata JSON.")
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model.")
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    _add_common_args(p)

    # -- describe ---------------------------------------------------------------
    p = subparsers.add_parser("describe", help="Generate video content description.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("--thumbnails-meta", required=True, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", required=True, help="Output JSON path.")
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model.")
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    _add_common_args(p)

    # -- translate --------------------------------------------------------------
    p = subparsers.add_parser("translate", help="Translate SRT to English.")
    p.add_argument("--srt", required=True, help="Path to source SRT file.")
    p.add_argument("--description", required=True, help="Path to description JSON.")
    p.add_argument("--thumbnails-meta", default=None, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    p.add_argument("--target-language", default="English", help="Target language for translation (default: English).")
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model.")
    p.add_argument(
        "--words-per-second", type=float, default=None,
        help="Target English speech rate in words/sec for duration matching (default: 2.0).",
    )
    p.add_argument(
        "--duration-budget", type=float, default=None,
        help="Fraction of time window to fill with translated speech, 0.0-1.0 (default: 0.80).",
    )
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    _add_common_args(p)

    # -- resegment --------------------------------------------------------------
    p = subparsers.add_parser("resegment", help="Re-segment SRT for readability.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    p.add_argument("--max-chars", type=int, default=84, help="Max chars per subtitle.")
    p.add_argument("--max-dur", type=float, default=4.0, help="Max seconds per subtitle.")
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model (optional).")
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key (enables LLM splitting).")
    p.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible API.")
    _add_common_args(p)

    # -- tts --------------------------------------------------------------------
    p = subparsers.add_parser("tts", help="Synthesise dubbed audio from SRT.")
    p.add_argument("--srt", required=True, help="Path to translated SRT.")
    p.add_argument("--original-audio", required=True, help="Original audio (for duration matching).")
    p.add_argument("--clone-profile", default=None,
                   help="Name of a voice profile to fetch from the HuggingFace dataset "
                        "(e.g. 'abubakr'). Provides --voice-sample and --voice-script "
                        "automatically.")
    p.add_argument("--voice-sample", default=None, help="Path to voice reference audio.")
    p.add_argument("--voice-script", default=None, help="Path to voice reference transcript.")
    p.add_argument("-o", "--output", required=True, help="Output dubbed WAV path.")
    p.add_argument("--segments-dir", default="./tts_segments", help="Directory for segment WAVs.")
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Qwen TTS model name.")
    p.add_argument("--chatterbox-model", default="ResembleAI/chatterbox", help="Chatterbox TTS model name.")
    p.add_argument("--device", default="cuda:0", help="Device.")
    p.add_argument("--dtype", default="bfloat16", help="Weight dtype (for Qwen engine).")
    p.add_argument("--language", default="English", help="Target language.")
    p.add_argument(
        "--tts-engine", default="qwen", choices=["qwen", "chatterbox"],
        help="TTS engine: 'qwen' (Qwen3-TTS) or 'chatterbox' (ResembleAI Chatterbox).",
    )
    p.add_argument(
        "--chatterbox-exaggeration", type=float, default=0.5,
        help="Chatterbox exaggeration level (0.0-1.0, default 0.5).",
    )
    p.add_argument(
        "--chatterbox-cfg", type=float, default=0.5,
        help="Chatterbox CFG weight (0.0-1.0, default 0.5). Lower for slower/expressive speech.",
    )
    p.add_argument(
        "--dynamic-tempo", action="store_true",
        help="Enable per-segment dynamic tempo adjustment to match original timing.",
    )
    p.add_argument(
        "--fixed-tempo", type=float, default=None,
        help="Apply a fixed tempo rate to all segments (e.g. 1.1). Overrides --dynamic-tempo.",
    )
    p.add_argument(
        "--max-tempo", type=float, default=1.3,
        help="Maximum speed-up factor for dynamic tempo (default: 1.3).",
    )
    p.add_argument(
        "--force-reset", action="store_true",
        help="Delete existing TTS segment files and re-synthesise from scratch.",
    )
    _add_common_args(p)

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

_HANDLERS = {
    "dub": _cmd_dub,
    "download": _cmd_download,
    "transcribe": _cmd_transcribe,
    "thumbnails": _cmd_thumbnails,
    "describe": _cmd_describe,
    "translate": _cmd_translate,
    "resegment": _cmd_resegment,
    "tts": _cmd_tts,
}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    _HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
