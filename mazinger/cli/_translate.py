"""mazinger translate — translate SRT to target language."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_llm, add_source, add_subtitles, add_transcription,
    add_translation, ensure_transcription, llm_extra_body,
    make_openai_client, resolve_project, subtitle_style_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("translate", help="Translate SRT to target language.")
    add_source(p)
    p.add_argument("--srt", default=None, help="Path to source SRT (overrides auto-transcription).")
    p.add_argument("--description", default=None,
                   help="Path to description JSON (optional, improves translation quality).")
    p.add_argument("--thumbnails-meta", default=None, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", default=None,
                   help="Output SRT path (default: project subtitles/translated.srt).")
    p.add_argument("--video", default=None,
                   help="Source video for subtitle embedding (overrides project video).")
    p.add_argument("--video-output", default=None,
                   help="Output video path (default: <output>.mp4 alongside the SRT).")
    add_llm(p)
    add_transcription(p)
    add_translation(p)
    add_subtitles(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import os
    import sys
    from mazinger.translate import translate_srt
    from mazinger.utils import load_json

    proj = resolve_project(args)

    # Resolve SRT: explicit flag > auto-transcribe from project
    srt_path = args.srt
    if not srt_path and proj:
        ensure_transcription(proj, args)
        srt_path = proj.source_srt
    if not srt_path:
        sys.exit("Error: provide a source (positional) or --srt.")

    output = args.output or (proj.final_srt if proj else
                             os.path.join(os.path.dirname(srt_path), "translated.srt"))

    client = make_openai_client(args)
    with open(srt_path, encoding="utf-8") as fh:
        srt_text = fh.read()
    description = load_json(args.description) if args.description else {}
    thumb_paths = load_json(args.thumbnails_meta) if args.thumbnails_meta else []

    result = translate_srt(
        srt_text, description, thumb_paths, client, llm_model=args.llm_model,
        source_language=args.source_language,
        target_language=args.target_language,
        translate_technical_terms=args.translate_technical_terms,
        extra_body=llm_extra_body(args),
        **(dict(words_per_second=args.words_per_second) if args.words_per_second is not None else {}),
        **(dict(duration_budget=args.duration_budget) if args.duration_budget is not None else {}),
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"Translated SRT saved: {output}")

    if args.embed_subtitles:
        from mazinger.resegment import resegment_srt
        from mazinger.subtitle import burn_subtitles

        # Resegment for readable captions before burning
        with open(output, encoding="utf-8") as fh:
            raw = fh.read()
        resegmented = resegment_srt(raw, client=client, llm_model=args.llm_model,
                                     extra_body=llm_extra_body(args))
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(resegmented)
        print(f"Resegmented SRT saved: {output}")

        video = args.video or (proj.video if proj and os.path.exists(proj.video) else None)
        if not video:
            sys.exit("Error: --video is required when using --embed-subtitles (or provide a video source).")
        video_out = args.video_output or os.path.splitext(output)[0] + ".mp4"
        style = subtitle_style_from_args(args)
        burn_subtitles(video, video_out, output, style)
        print(f"Subtitled video saved: {video_out}")
