"""mazinger slice — extract a time range from a video or audio file."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_slice, add_source, resolve_project


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("slice", help="Extract a time range from a video or audio file.")
    add_source(p, required=True)
    add_slice(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    proj = resolve_project(args)
    print(proj.summary())
