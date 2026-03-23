"""mazinger download — fetch video / ingest local file and extract audio."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_slice, add_source, resolve_project


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download", help="Download video / ingest local file and extract audio.")
    add_source(p, required=True)
    add_slice(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    proj = resolve_project(args)
    print(proj.summary())
