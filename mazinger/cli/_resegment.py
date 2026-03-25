"""mazinger resegment — re-segment SRT for readability."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_llm


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("resegment", help="Re-segment SRT for readability.")
    p.add_argument("--srt", required=True, help="Path to SRT file.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    p.add_argument("--max-chars", type=int, default=84, help="Max chars per subtitle.")
    p.add_argument("--max-dur", type=float, default=4.0, help="Max seconds per subtitle.")
    add_llm(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.resegment import resegment_srt

    client = None
    if args.openai_api_key or args.openai_base_url:
        from mazinger.cli._groups import make_llm_client
        client = make_llm_client(args)

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
