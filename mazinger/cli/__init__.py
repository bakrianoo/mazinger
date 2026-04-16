"""Mazinger CLI — command-line entry point.

Each sub-command lives in its own module under ``mazinger.cli``.
Adding a new command is a two-step process:

1. Create ``_mycommand.py`` with ``register(subparsers)`` and ``handler(args)``.
2. Append the module to ``_COMMANDS`` below.
"""

from __future__ import annotations

import argparse
import logging

from mazinger.cli import (
    _describe,
    _download,
    _dub,
    _profile,
    _resegment,
    _slice,
    _speak,
    _subtitle,
    _thumbnails,
    _transcribe,
    _translate,
    _web,
)

_COMMANDS = (
    _dub,
    _download,
    _slice,
    _transcribe,
    _thumbnails,
    _describe,
    _translate,
    _resegment,
    _speak,
    _subtitle,
    _profile,
    _web,
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mazinger",
        description="Mazinger — End-to-end video dubbing pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in _COMMANDS:
        cmd.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    dispatch = {cmd.__name__.rsplit("_", 1)[-1]: cmd.handler for cmd in _COMMANDS}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
