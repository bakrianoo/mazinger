"""mazinger transcribe — audio to SRT."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    DEFAULT_MLX_WHISPER_MODEL,
    add_common, add_deepgram, add_openai, add_source, resolve_project,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("transcribe", help="Transcribe audio to SRT.")
    add_source(p)
    p.add_argument("--audio", default=None, help="Path to audio file (overrides source).")
    p.add_argument("-o", "--output", default=None,
                   help="Output SRT path (default: project transcription/source.srt).")
    p.add_argument("--method", default="faster-whisper",
                   choices=["openai", "faster-whisper", "whisperx", "mlx-whisper", "deepgram"],
                   help="Transcription backend.")
    p.add_argument("--model", default=None,
                   help="Model name. Defaults to 'whisper-1' for OpenAI, "
                        "'large-v3' for local, 'nova-3' for Deepgram.")
    p.add_argument("--mlx-whisper-model", default=DEFAULT_MLX_WHISPER_MODEL,
                   help=f"MLX Whisper model name (default: {DEFAULT_MLX_WHISPER_MODEL}).")
    p.add_argument("--device", default="auto", help="Device: auto (default), cuda, or cpu.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for local methods.")
    p.add_argument("--compute-type", default="float16", help="Compute type: float16, int8, int8_float16.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (default: 5).")
    p.add_argument("--language", default=None, help="Force language code (e.g., en, ar).")
    p.add_argument("--initial-prompt", default=None,
                   help="Initial text prompt to condition the model (e.g., domain-specific terms).")
    p.add_argument("--no-condition-on-previous-text", action="store_true",
                   help="Disable conditioning on previous segment text.")
    p.add_argument("--max-chars", type=int, default=84, help="Max chars per subtitle.")
    p.add_argument("--max-duration", type=float, default=5.0, help="Max seconds per subtitle.")
    p.add_argument("--no-resegment", action="store_true", help="Skip resegmentation.")
    p.add_argument("--refine", action="store_true",
                   help="Use LLM to add punctuation and fix misheard words.")
    p.add_argument("--asr-review", action="store_true", default=False,
                   help="Review transcript with LLM to fix typos, punctuation, "
                        "and optionally normalise technical terms.")
    p.add_argument("--keep-technical-english", action="store_true", default=False,
                   help="Convert technical terms to English in the transcript "
                        "(requires --asr-review).")
    p.add_argument("--llm-model", default="gpt-4.1", help="LLM model for refinement.")
    add_openai(p)
    add_deepgram(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import sys
    from mazinger.transcribe import transcribe
    from mazinger.cli._groups import resolve_device

    args.device = resolve_device(args.device)
    proj = resolve_project(args)

    audio = args.audio or (proj.audio if proj else None)
    output = args.output or (proj.source_srt if proj else None)

    if not audio:
        sys.exit("Error: provide a source (positional) or --audio.")
    if not output:
        sys.exit("Error: provide a source (positional) or -o/--output.")

    transcribe(
        audio, output,
        method=args.method,
        model=args.model,
        mlx_whisper_model=args.mlx_whisper_model,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        skip_resegment=args.no_resegment,
        refine=args.refine,
        llm_model=args.llm_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        deepgram_api_key=args.deepgram_api_key,
        initial_prompt=args.initial_prompt,
        condition_on_previous_text=not args.no_condition_on_previous_text,
    )
    print(f"SRT saved: {output}")

    if args.asr_review:
        from mazinger.llm import build_client
        from mazinger.describe import describe_content
        from mazinger.review import review_srt

        with open(output, encoding="utf-8") as fh:
            srt_text = fh.read()

        client = build_client(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )

        description = describe_content(
            srt_text, [], client, llm_model=args.llm_model,
        )

        reviewed = review_srt(
            srt_text, description, client,
            llm_model=args.llm_model,
            source_language=getattr(args, "language", "auto") or "auto",
            keep_technical_english=args.keep_technical_english,
        )
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(reviewed)
        print(f"ASR review applied: {output}")
