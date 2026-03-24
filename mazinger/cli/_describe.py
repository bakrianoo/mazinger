"""mazinger describe — generate video content description."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_llm, add_source, add_transcription,
    ensure_transcription, llm_extra_body, make_openai_client, resolve_project,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("describe", help="Generate video content description.")
    add_source(p)
    p.add_argument("--srt", default=None, help="Path to SRT file (overrides auto-transcription).")
    p.add_argument("--thumbnails-meta", default=None, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", default=None, help="Output JSON path.")
    add_llm(p)
    add_transcription(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import os
    import sys
    from mazinger.describe import describe_content
    from mazinger.thumbnails import select_timestamps, extract_frames
    from mazinger.utils import load_json, save_json

    proj = resolve_project(args)

    srt_path = args.srt
    if not srt_path and proj:
        ensure_transcription(proj, args)
        srt_path = proj.source_srt
    if not srt_path:
        sys.exit("Error: provide a source (positional) or --srt.")

    output = args.output or (proj.description if proj else None)
    if not output:
        sys.exit("Error: provide a source (positional) or -o/--output.")

    client = make_openai_client(args)
    with open(srt_path, encoding="utf-8") as fh:
        srt_text = fh.read()

    # Resolve or generate thumbnails
    if args.thumbnails_meta:
        thumb_paths = load_json(args.thumbnails_meta)
    elif proj and os.path.exists(proj.thumbs_meta):
        thumb_paths = load_json(proj.thumbs_meta)
    elif proj and os.path.exists(proj.video):
        ts = select_timestamps(srt_text, client, llm_model=args.llm_model,
                               extra_body=llm_extra_body(args))
        thumb_paths = extract_frames(proj.video, ts, proj.thumbnails_dir)
        save_json(thumb_paths, proj.thumbs_meta)
    else:
        sys.exit("Error: provide --thumbnails-meta or a video source for auto-extraction.")

    desc = describe_content(srt_text, thumb_paths, client, llm_model=args.llm_model,
                            extra_body=llm_extra_body(args))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    save_json(desc, output)
    print(f"Description saved: {output}")
