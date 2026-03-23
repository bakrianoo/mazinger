"""mazinger slice — extract a time range from a video or audio file."""

from __future__ import annotations

import argparse
import logging

from mazinger.cli._groups import add_common, add_slice, add_source, resolve_project

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("slice", help="Extract a time range from a video or audio file.")
    add_source(p, required=True)
    add_slice(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if not start and not end:
        raise SystemExit("Error: at least one of --start or --end is required for slicing.")

    # resolve_project downloads/ingests + applies the slice via _apply_slice
    proj = resolve_project(args)
    if proj is None:
        raise SystemExit("Error: could not resolve project from the given source.")

    log.info("Slice complete: %s", proj.root)
    print(proj.summary())
