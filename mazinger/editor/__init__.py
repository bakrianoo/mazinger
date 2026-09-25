"""Mazinger Editor — segment-level review and repair of a finished dub.

The Editor loads a completed project, presents it as a list of chunks (one per
dubbed segment) and lets the user fix text, timing and chunk boundaries, re-do
single stages for individual chunks, and re-assemble the final output.

This package holds the logic only; the Gradio UI lives in
:mod:`mazinger.studio.editor_ui`.

Modules:
    session  Chunk/Session model, import from a project, edits, stale rules.
    store    Persistence of a session (snapshot + append-only change log).
    ops      Re-transcribe, re-translate, re-dub and assemble operations.
    media    Original-audio clip extraction and its cache.
"""

from mazinger.editor.session import (  # noqa: E402
    STALE_CHECK, STALE_DUB, STALE_KINDS, STALE_TRANSLATION, Chunk, Session, UndoError,
)

__all__ = [
    "Chunk", "Session", "UndoError",
    "STALE_KINDS", "STALE_TRANSLATION", "STALE_DUB", "STALE_CHECK",
]
