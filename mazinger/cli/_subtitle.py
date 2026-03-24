"""mazinger subtitle — burn subtitles into a video."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_llm, add_source, add_subtitle_style, add_transcription,
    add_translation, ensure_transcription, llm_extra_body,
    make_openai_client, resolve_project, subtitle_style_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("subtitle", help="Burn subtitles into a video.")
    add_source(p)
    p.add_argument("--video", default=None, help="Path to source video (overrides project video).")
    p.add_argument("--srt", default=None,
                   help="Path to SRT file (overrides auto-translate; default: translated SRT).")
    p.add_argument("--audio", default=None,
                   help="Replacement audio track (omit to keep original audio).")
    p.add_argument("-o", "--output", default=None, help="Output video path.")
    add_subtitle_style(p)
    add_llm(p)
    add_transcription(p)
    add_translation(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import os
    import sys
    from mazinger.subtitle import burn_subtitles

    proj = resolve_project(args)

    video = args.video or (proj.video if proj and os.path.exists(proj.video) else None)
    if not video:
        sys.exit("Error: provide a video source (positional) or --video.")

    srt_path = args.srt
    if not srt_path and proj:
        # Auto-transcribe + translate + resegment if needed
        ensure_transcription(proj, args)
        if not os.path.exists(proj.final_srt):
            from mazinger.translate import translate_srt
            from mazinger.resegment import resegment_srt
            from mazinger.utils import load_json
            client = make_openai_client(args)
            with open(proj.source_srt, encoding="utf-8") as fh:
                srt_text = fh.read()
            description = load_json(proj.description) if os.path.exists(proj.description) else {}
            thumb_paths = load_json(proj.thumbs_meta) if os.path.exists(proj.thumbs_meta) else []
            translated = translate_srt(
                srt_text, description, thumb_paths, client, llm_model=args.llm_model,
                source_language=args.source_language,
                target_language=args.target_language,
                extra_body=llm_extra_body(args),
                **(dict(words_per_second=args.words_per_second) if args.words_per_second is not None else {}),
                **(dict(duration_budget=args.duration_budget) if args.duration_budget is not None else {}),
            )
            # Save raw translation, then resegment for readable captions
            os.makedirs(os.path.dirname(proj.translated_raw_srt) or ".", exist_ok=True)
            with open(proj.translated_raw_srt, "w", encoding="utf-8") as fh:
                fh.write(translated)
            resegmented = resegment_srt(translated, client=client, llm_model=args.llm_model,
                                         extra_body=llm_extra_body(args))
            os.makedirs(os.path.dirname(proj.final_srt) or ".", exist_ok=True)
            with open(proj.final_srt, "w", encoding="utf-8") as fh:
                fh.write(resegmented)
        srt_path = proj.final_srt
    if not srt_path:
        sys.exit("Error: provide a source (positional) or --srt.")

    output = args.output or (proj.final_video if proj else
                             os.path.splitext(video)[0] + "_subtitled.mp4")

    style = subtitle_style_from_args(args)
    result = burn_subtitles(video, output, srt_path, style, audio_path=args.audio)
    if result:
        print(f"Subtitled video saved: {result}")
