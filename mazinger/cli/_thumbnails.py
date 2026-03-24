"""mazinger thumbnails — extract key-frame thumbnails."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_llm, add_source, add_transcription,
    ensure_transcription, llm_extra_body, make_openai_client, resolve_project,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("thumbnails", help="Extract key-frame thumbnails.")
    add_source(p)
    p.add_argument("--video", default=None, help="Path to video file (overrides project video).")
    p.add_argument("--srt", default=None, help="Path to SRT file (overrides auto-transcription).")
    p.add_argument("--output-dir", default=None, help="Output directory for thumbnails.")
    p.add_argument("--meta", default=None, help="Path to save metadata JSON.")
    add_llm(p)
    add_transcription(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import os
    import sys
    from mazinger.thumbnails import select_timestamps, extract_frames
    from mazinger.utils import save_json

    proj = resolve_project(args)

    video = args.video or (proj.video if proj and os.path.exists(proj.video) else None)
    if not video:
        sys.exit("Error: provide a video source (positional) or --video.")

    srt_path = args.srt
    if not srt_path and proj:
        ensure_transcription(proj, args)
        srt_path = proj.source_srt
    if not srt_path:
        sys.exit("Error: provide a source (positional) or --srt.")

    output_dir = args.output_dir or (proj.thumbnails_dir if proj else None)
    if not output_dir:
        sys.exit("Error: provide a source (positional) or --output-dir.")

    client = make_openai_client(args)
    with open(srt_path, encoding="utf-8") as fh:
        srt_text = fh.read()

    timestamps = select_timestamps(srt_text, client, llm_model=args.llm_model,
                                    extra_body=llm_extra_body(args))
    results = extract_frames(video, timestamps, output_dir)

    meta_path = args.meta or (proj.thumbs_meta if proj else f"{output_dir}/meta.json")
    save_json(results, meta_path)
    print(f"Extracted {len(results)} thumbnails -> {output_dir}")
