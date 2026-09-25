"""Chunk and session model for the Editor.

A *chunk* is one entry of the final ``subtitles/translated.srt`` — the unit
that was dubbed as one TTS segment.  Its transcription is mapped from the
source SRT by time overlap on first open (:meth:`Session.import_project`).

Edits never run anything.  They change the chunk and mark the steps that
depend on the change as *stale* (see :data:`STALE_KINDS`); the Editor's
operations (:mod:`mazinger.editor.ops`) redo those steps on request and
clear the flags they fix through the ``apply_*`` methods.

Every mutating method returns the ids of the rows it changed (or created),
so the UI refreshes only those, and writes one record to the change log
(:mod:`mazinger.editor.store`).
"""

from __future__ import annotations

import logging
import os
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from mazinger.editor.store import MERGE_EVERY, SessionStore
from mazinger.paths import ProjectPaths
from mazinger.runinfo import load_run_info, project_relpath, resolve_project_path
from mazinger.srt import parse_file

log = logging.getLogger(__name__)

# Stale flags.
STALE_TRANSLATION = "translation"           # source text changed since the translation
STALE_DUB = "dub"                           # target text / boundaries changed since the dub
STALE_CHECK = "transcription_check"         # timing moved; transcription may not match
STALE_KINDS = (STALE_TRANSLATION, STALE_DUB, STALE_CHECK)

MIN_CHUNK_SEC = 0.2       # shortest chunk a timing edit or split may produce
HISTORY_LIMIT = 10        # undo steps kept per chunk
BAD_FIT = 1.15            # fit ratio above which a dub counts as too long

# Fields captured by a history entry and restored by undo.
_STATE_FIELDS = ("start", "end", "source_text", "target_text", "dub_wav", "dub_dur")


class UndoError(RuntimeError):
    """The requested undo cannot be applied in the session's current state."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Text helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Scripts written without spaces between words: Thai, Lao, Myanmar, Khmer,
# and CJK (Chinese, Japanese).  Hangul is excluded — Korean uses spaces.
_UNSPACED_RANGES = (
    (0x0E00, 0x0EFF), (0x1000, 0x109F), (0x1780, 0x17FF),
    (0x2E80, 0x312F), (0x31A0, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF),
    (0x20000, 0x3FFFF),
)


def _is_unspaced_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _UNSPACED_RANGES)


def _is_unspaced(text: str) -> bool:
    """True for text without spaces in a script that does not use them."""
    return bool(text) and not any(ch.isspace() for ch in text) and any(map(_is_unspaced_char, text))


def text_units(text: str) -> list[str]:
    """Split *text* into the units a split point counts: words, or characters
    for scripts written without spaces."""
    text = text.strip()
    if not (_is_unspaced(text) and len(text) > 1):
        return text.split()
    units: list[str] = []
    for ch in text:
        # Keep combining marks (Thai vowels and tones) on their base character.
        if units and unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            units[-1] += ch
        else:
            units.append(ch)
    return units


def split_text(text: str, at: int) -> tuple[str, str]:
    """Split *text* after its first *at* :func:`text_units` (clamped)."""
    units = text_units(text)
    at = max(0, min(at, len(units)))
    joiner = "" if _is_unspaced(text.strip()) else " "
    return joiner.join(units[:at]), joiner.join(units[at:])


def default_split_index(text: str, fraction: float) -> int:
    """Split point that divides *text* in proportion to time (*fraction* of the chunk)."""
    return round(len(text_units(text)) * max(0.0, min(1.0, fraction)))


def join_texts(a: str, b: str) -> str:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return a or b
    return a + ("" if _is_unspaced(a) and _is_unspaced(b) else " ") + b


# ═══════════════════════════════════════════════════════════════════════════════
#  Chunk
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """One dubbed segment.

    ``dub_wav`` is relative to the project root (see
    :func:`mazinger.runinfo.project_relpath`).  ``history`` holds up to
    :data:`HISTORY_LIMIT` undo entries, newest last.
    """

    id: str
    start: float
    end: float
    source_text: str = ""
    target_text: str = ""
    dub_wav: str | None = None
    dub_dur: float = 0.0
    stale: set[str] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    rev: int = 0

    # -- derived -----------------------------------------------------------

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def fit_ratio(self) -> float | None:
        """Dub length over slot length; ``None`` without a dub."""
        if not self.dub_wav or self.duration <= 0:
            return None
        return self.dub_dur / self.duration

    @property
    def has_text(self) -> bool:
        return bool(self.target_text.strip())

    def badges(self) -> list[str]:
        """Every status flag of the chunk, most urgent first."""
        if not self.has_text:
            return ["empty"]
        out = []
        if STALE_TRANSLATION in self.stale:
            out.append("⚠ translation")
        if not self.dub_wav:
            out.append("⚠ no dub")
        elif STALE_DUB in self.stale:
            out.append("⚠ dub")
        if STALE_CHECK in self.stale:
            out.append("🔍 check")
        fit = self.fit_ratio
        if fit is not None and fit > BAD_FIT:
            out.append("⏱ long")
        return out or ["✓ ok"]

    @property
    def status(self) -> str:
        """Short status label for the chunk table."""
        return self.badges()[0]

    @property
    def needs_translation(self) -> bool:
        return STALE_TRANSLATION in self.stale and bool(self.source_text.strip())

    @property
    def needs_dub(self) -> bool:
        return self.has_text and (STALE_DUB in self.stale or not self.dub_wav)

    # -- serialisation -----------------------------------------------------

    def state(self) -> dict:
        s = {f: getattr(self, f) for f in _STATE_FIELDS}
        s["stale"] = sorted(self.stale)
        return s

    def to_dict(self) -> dict:
        return {"id": self.id, **self.state(), "history": self.history, "rev": self.rev}

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            id=d["id"], start=float(d["start"]), end=float(d["end"]),
            source_text=d.get("source_text", ""), target_text=d.get("target_text", ""),
            dub_wav=d.get("dub_wav"), dub_dur=float(d.get("dub_dur", 0.0)),
            stale=set(d.get("stale", ())), history=list(d.get("history", ())),
            rev=int(d.get("rev", 0)),
        )


def _mark_dub_stale(c: Chunk) -> None:
    # A chunk without text has nothing to dub, so its dub cannot be stale.
    if c.has_text:
        c.stale.add(STALE_DUB)
    else:
        c.stale.discard(STALE_DUB)


def _field_history_only(d: dict) -> dict:
    return {**d, "history": [h for h in d.get("history", ()) if "state" in h]}


def _for_embedding(c: Chunk) -> dict:
    """*c* as stored inside a split/merge undo entry.

    *c* keeps its whole history, so its own split or merge can still be
    undone after this one is.  The chunks nested inside those entries keep
    field-level history only: nesting deeper would grow without bound over
    repeated split/merge cycles.  Undo therefore steps back through at most
    two consecutive splits/merges.
    """
    history = []
    for h in c.history:
        if h.get("op") == "split":
            h = {**h, "original": _field_history_only(h["original"])}
        elif h.get("op") == "merge":
            h = {**h, "originals": [_field_history_only(o) for o in h["originals"]]}
        history.append(h)
    return {**c.to_dict(), "history": history}


# ═══════════════════════════════════════════════════════════════════════════════
#  Import helpers
# ═══════════════════════════════════════════════════════════════════════════════

def map_source_to_chunks(
    sources: list[tuple[float, float, str]],
    chunks: list[tuple[float, float]],
) -> list[list[str]]:
    """Assign each source entry to the chunk it overlaps most.

    Both lists must be sorted by start time.  One pass over both (O(n + m)
    for non-overlapping chunks).  An entry that overlaps no chunk goes to the
    nearest one, so no transcription text is lost.

    Returns:
        For each chunk, the texts assigned to it in time order.
    """
    out: list[list[str]] = [[] for _ in chunks]
    n = len(chunks)
    if not n:
        return out
    j = 0
    for s_start, s_end, text in sources:
        while j < n - 1 and chunks[j][1] <= s_start:
            j += 1
        best, best_ov = -1, 0.0
        k = j
        while k < n and chunks[k][0] < s_end:
            ov = min(s_end, chunks[k][1]) - max(s_start, chunks[k][0])
            if ov > best_ov:
                best, best_ov = k, ov
            k += 1
        if best < 0:
            # No overlap: the nearer of the chunk before and the one at j.
            best = j
            if j > 0:
                gap_before = max(0.0, s_start - chunks[j - 1][1])
                gap_after = max(0.0, chunks[j][0] - s_end)
                if gap_before < gap_after:
                    best = j - 1
        out[best].append(text)
    return out


def _source_srt_for(proj: ProjectPaths, run_info: dict | None) -> str | None:
    """The source SRT the translation was made from (see ``run.json``)."""
    candidates = []
    if run_info and run_info.get("translation_source_srt"):
        candidates.append(resolve_project_path(proj, run_info["translation_source_srt"]))
    candidates += [proj.reviewed_srt, proj.source_srt, proj.source_raw_srt]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


def _file_stamp(path: str) -> list | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


def _wav_duration(path: str) -> float | None:
    import soundfile as sf
    try:
        return float(sf.info(path).duration)  # header only, no decoding
    except Exception:  # noqa: BLE001 — missing or unreadable
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Session
# ═══════════════════════════════════════════════════════════════════════════════

class Session:
    """The chunks of one project language, their edits and their persistence.

    Create with :meth:`open` (load or import), :meth:`import_project` or
    :meth:`load`.  Methods are safe to call from several threads.
    """

    def __init__(
        self,
        proj: ProjectPaths,
        chunks: Iterable[Chunk],
        *,
        run_info: dict | None = None,
        rev: int = 0,
        next_id: int | None = None,
        output_stale: bool = False,
        duration: float | None = None,
        source_stamp: list | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.proj = proj
        self.language = proj.target_language
        self.run_info = run_info
        self.chunks: list[Chunk] = list(chunks)
        self.rev = rev
        self.output_stale = output_stale
        self.duration = duration              # source media length, for clamping
        self.source_stamp = source_stamp      # final SRT at import (mtime_ns, size)
        self.store = store
        self._lock = threading.RLock()
        self._index: dict[str, int] = {}
        self._reindex()
        if next_id is None:
            nums = [int(c.id[1:]) for c in self.chunks if c.id[1:].isdigit()]
            next_id = max(nums, default=0) + 1
        self.next_id = next_id

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def open(cls, proj: ProjectPaths) -> Session:
        """Load the saved session of *proj*, or import one on first open."""
        store = SessionStore(proj.editor_dir)
        return cls.load(proj) if store.exists() else cls.import_project(proj)

    @classmethod
    def import_project(cls, proj: ProjectPaths, *, save: bool = True) -> Session:
        """Build a session from a finished dub.

        Chunks come from ``subtitles/translated.srt``, their transcription from
        the source SRT (by time overlap), and their dub from the existing
        ``tts/segments/seg_NNNN.wav`` files, which are referenced, not copied.
        A chunk whose segment is missing is marked for re-dubbing.

        Raises:
            FileNotFoundError: if the project has no final translated SRT.
        """
        if not os.path.isfile(proj.final_srt):
            raise FileNotFoundError(f"No dubbed subtitles to edit: {proj.final_srt}")
        run_info = load_run_info(proj)

        entries = sorted(parse_file(proj.final_srt), key=lambda e: e["start"])
        src_path = _source_srt_for(proj, run_info)
        sources = sorted(
            ((e["start"], e["end"], e["text"]) for e in parse_file(src_path)),
            key=lambda s: s[0],
        ) if src_path else []
        texts = map_source_to_chunks(sources, [(e["start"], e["end"]) for e in entries])

        chunks = []
        for n, (e, src) in enumerate(zip(entries, texts), 1):
            c = Chunk(
                id=f"c{n}", start=e["start"], end=e["end"],
                source_text=" ".join(t.strip() for t in src if t.strip()),
                target_text=e["text"].strip(),
            )
            wav = os.path.join(proj.tts_segments_dir, f"seg_{e['idx'].zfill(4)}.wav")
            dur = _wav_duration(wav) if c.has_text else None
            if dur is not None:
                c.dub_wav, c.dub_dur = project_relpath(proj, wav), dur
            elif c.has_text:
                c.stale.add(STALE_DUB)
            chunks.append(c)

        duration = None
        if os.path.isfile(proj.audio):
            try:
                from mazinger.utils import get_audio_duration
                duration = get_audio_duration(proj.audio)
            except Exception as exc:  # noqa: BLE001 — clamping falls back to chunk bounds
                log.warning("Could not read the source duration: %s", exc)

        session = cls(
            proj, chunks, run_info=run_info, duration=duration,
            source_stamp=_file_stamp(proj.final_srt),
            store=SessionStore(proj.editor_dir) if save else None,
        )
        log.info(
            "Editor session imported: %d chunks (%s), transcription from %s",
            len(chunks), proj.target_language, src_path or "nowhere",
        )
        if save:
            session.save()
        return session

    @classmethod
    def load(cls, proj: ProjectPaths) -> Session:
        """Load the saved session: snapshot + change-log replay.

        Chunks whose dub WAV no longer exists are marked for re-dubbing.
        The log is merged into a new snapshot when anything was replayed.
        """
        store = SessionStore(proj.editor_dir)
        snap, records = store.load()
        session = cls(
            proj, (Chunk.from_dict(d) for d in snap.get("chunks", ())),
            run_info=load_run_info(proj),
            rev=snap.get("rev", 0),
            next_id=snap.get("next_id"),
            output_stale=snap.get("output_stale", False),
            duration=snap.get("duration"),
            source_stamp=snap.get("source_stamp"),
            store=store,
        )
        replayed = 0
        for rec in records:
            try:
                session._apply_record(rec)
            except (KeyError, ValueError) as exc:
                log.error("Stopping change-log replay at rev %s: %s", rec.get("rev"), exc)
                break
            replayed += 1

        missing = 0
        for c in session.chunks:
            if c.dub_wav and not os.path.isfile(session.abs_path(c.dub_wav)):
                c.dub_wav, c.dub_dur = None, 0.0
                _mark_dub_stale(c)
                missing += 1
        if missing:
            log.warning("%d dubbed segment file(s) are missing; those chunks need re-dubbing", missing)
        if replayed or missing or store.pending:
            session.save()
        return session

    # ── lookup ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def index(self, chunk_id: str) -> int:
        try:
            return self._index[chunk_id]
        except KeyError:
            raise KeyError(f"No chunk {chunk_id!r}") from None

    def chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self.index(chunk_id)]

    def neighbours(self, chunk_id: str) -> tuple[Chunk | None, Chunk | None]:
        i = self.index(chunk_id)
        prev = self.chunks[i - 1] if i > 0 else None
        nxt = self.chunks[i + 1] if i + 1 < len(self.chunks) else None
        return prev, nxt

    def abs_path(self, rel: str | None) -> str | None:
        """Absolute path of a project-relative path stored in a chunk."""
        return resolve_project_path(self.proj, rel)

    def ids(self, kind: str | None = None) -> list[str]:
        """Ids of all chunks, or of those needing *kind* (a :data:`STALE_KINDS` value)."""
        if kind is None:
            return [c.id for c in self.chunks]
        if kind == STALE_DUB:
            return [c.id for c in self.chunks if c.needs_dub]
        if kind == STALE_TRANSLATION:
            return [c.id for c in self.chunks if c.needs_translation]
        return [c.id for c in self.chunks if kind in c.stale]

    def counts(self) -> dict:
        """Totals for the summary bar."""
        out = {"chunks": len(self.chunks), STALE_TRANSLATION: 0, STALE_DUB: 0,
               STALE_CHECK: 0, "long": 0, "empty": 0}
        for c in self.chunks:
            if not c.has_text:
                out["empty"] += 1
            if c.needs_translation:
                out[STALE_TRANSLATION] += 1
            if c.needs_dub:
                out[STALE_DUB] += 1
            if STALE_CHECK in c.stale:
                out[STALE_CHECK] += 1
            fit = c.fit_ratio
            if fit is not None and fit > BAD_FIT:
                out["long"] += 1
        return out

    def out_of_sync(self) -> bool:
        """True when the project's final SRT changed after this session was
        imported — the project was dubbed again, so the session is outdated."""
        return self.source_stamp is not None and _file_stamp(self.proj.final_srt) != self.source_stamp

    # ── edits ────────────────────────────────────────────────────────────────

    def set_source_text(self, chunk_id: str, text: str) -> list[str]:
        """Edit the transcription → translation and dub become stale."""
        text = text.strip()
        with self._lock:
            c = self.chunk(chunk_id)
            if text == c.source_text:
                return []
            self._push_history(c, "set_source_text")
            c.source_text = text
            c.stale.add(STALE_TRANSLATION)
            _mark_dub_stale(c)
            return self._commit_edit("set_source_text", c)

    def set_target_text(self, chunk_id: str, text: str) -> list[str]:
        """Edit the translation → dub becomes stale.

        Emptying the translation removes the chunk's dub (it will be silent).
        """
        text = text.strip()
        with self._lock:
            c = self.chunk(chunk_id)
            if text == c.target_text:
                return []
            self._push_history(c, "set_target_text")
            c.target_text = text
            if not c.has_text:
                c.dub_wav, c.dub_dur = None, 0.0
            _mark_dub_stale(c)
            return self._commit_edit("set_target_text", c)

    def clamp_timing(self, chunk_id: str, start: float, end: float) -> tuple[float, float]:
        """Clamp a proposed ``[start, end]`` between the neighbouring chunks.

        Keeps at least :data:`MIN_CHUNK_SEC`; when the range is too short, the
        edge that moved gives way.  Raises ``ValueError`` if the gap between
        the neighbours is itself too small.
        """
        c = self.chunk(chunk_id)
        prev, nxt = self.neighbours(chunk_id)
        # Never force a pre-existing overlap to be resolved by this edit.
        lo = min(prev.end, c.start) if prev else 0.0
        hi = max(nxt.start, c.end) if nxt else max(self.duration or c.end, c.end)
        if hi - lo < MIN_CHUNK_SEC:
            raise ValueError(f"Not enough room for chunk {chunk_id}: {hi - lo:.2f}s")
        start = round(max(lo, min(float(start), hi - MIN_CHUNK_SEC)), 3)
        end = round(min(hi, max(float(end), lo + MIN_CHUNK_SEC)), 3)
        if end - start < MIN_CHUNK_SEC - 1e-9:
            if start != c.start:
                start = round(end - MIN_CHUNK_SEC, 3)
            else:
                end = round(start + MIN_CHUNK_SEC, 3)
        return start, end

    def set_timing(self, chunk_id: str, start: float | None = None, end: float | None = None) -> list[str]:
        """Move the chunk's start and/or end, clamped by :meth:`clamp_timing`.

        The dub is kept (its fit ratio changes); the transcription is flagged
        for a check because the audio under the chunk changed.
        """
        with self._lock:
            c = self.chunk(chunk_id)
            start, end = self.clamp_timing(
                chunk_id, c.start if start is None else start, c.end if end is None else end,
            )
            if (start, end) == (c.start, c.end):
                return []
            self._push_history(c, "set_timing")
            c.start, c.end = start, end
            c.stale.add(STALE_CHECK)
            return self._commit_edit("set_timing", c)

    def split(
        self,
        chunk_id: str,
        at_time: float,
        src_split_idx: int | None = None,
        tgt_split_idx: int | None = None,
    ) -> list[str]:
        """Split a chunk at *at_time* into two new chunks (new ids).

        The texts are cut after *src_split_idx* / *tgt_split_idx* units (words,
        or characters for unspaced scripts); by default in proportion to the
        time split (:func:`default_split_index`).  Both halves need a dub.

        Returns:
            The two new ids.
        """
        with self._lock:
            i = self.index(chunk_id)
            c = self.chunks[i]
            at_time = round(float(at_time), 3)
            if not (c.start + MIN_CHUNK_SEC <= at_time <= c.end - MIN_CHUNK_SEC):
                raise ValueError(
                    f"Split point {at_time:.3f}s must leave at least {MIN_CHUNK_SEC}s on "
                    f"each side of {c.start:.3f}–{c.end:.3f}s"
                )
            frac = (at_time - c.start) / c.duration
            if src_split_idx is None:
                src_split_idx = default_split_index(c.source_text, frac)
            if tgt_split_idx is None:
                tgt_split_idx = default_split_index(c.target_text, frac)
            src_a, src_b = split_text(c.source_text, src_split_idx)
            tgt_a, tgt_b = split_text(c.target_text, tgt_split_idx)

            parts = [
                Chunk(self._new_id(), c.start, at_time, src_a, tgt_a, stale=set(c.stale)),
                Chunk(self._new_id(), at_time, c.end, src_b, tgt_b, stale=set(c.stale)),
            ]
            entry = {"op": "split", "token": f"split:{self.rev + 1}",
                     "original": _for_embedding(c), "parts": [p.id for p in parts]}
            for p in parts:
                _mark_dub_stale(p)
                p.history.append(entry)
            return self._commit("split", i, [c.id], parts)

    def merge_with_next(self, chunk_id: str) -> list[str]:
        """Merge a chunk with the following one into one new chunk.

        Returns:
            The new chunk's id.
        """
        with self._lock:
            i = self.index(chunk_id)
            if i + 1 >= len(self.chunks):
                raise ValueError(f"Chunk {chunk_id} is the last chunk; nothing to merge with")
            a, b = self.chunks[i], self.chunks[i + 1]
            merged = Chunk(
                self._new_id(), a.start, b.end,
                join_texts(a.source_text, b.source_text),
                join_texts(a.target_text, b.target_text),
                stale=a.stale | b.stale,
            )
            _mark_dub_stale(merged)
            merged.history.append({
                "op": "merge", "token": f"merge:{self.rev + 1}",
                "originals": [_for_embedding(a), _for_embedding(b)],
            })
            return self._commit("merge", i, [a.id, b.id], [merged])

    def undo(self, chunk_id: str) -> list[str]:
        """Revert the chunk's last change.

        Undoing a split restores the original chunk (both halves must be
        unchanged since the split); undoing a merge restores the two
        originals.

        Returns:
            Ids of the affected chunks; ``[]`` when there is nothing to undo.

        Raises:
            UndoError: if the undo would conflict with later edits.
        """
        with self._lock:
            i = self.index(chunk_id)
            c = self.chunks[i]
            if not c.history:
                return []
            entry = c.history[-1]

            if entry.get("op") == "split":
                ids = entry["parts"]
                at = self._index.get(ids[0])
                span = self.chunks[at:at + len(ids)] if at is not None else []
                if [p.id for p in span] != ids or any(
                    not p.history or p.history[-1].get("token") != entry["token"] for p in span
                ):
                    raise UndoError(
                        "This chunk was split and a part has changed since; "
                        "undo the changes to both parts first"
                    )
                original = Chunk.from_dict(entry["original"])
                return self._commit("undo_split", at, ids, [original])

            if entry.get("op") == "merge":
                originals = [Chunk.from_dict(d) for d in entry["originals"]]
                return self._commit("undo_merge", i, [c.id], originals)

            state = entry["state"]
            if (state["start"], state["end"]) != (c.start, c.end):
                prev, nxt = self.neighbours(chunk_id)
                if (prev and state["start"] < min(prev.end, c.start)) or \
                        (nxt and state["end"] > max(nxt.start, c.end)):
                    raise UndoError("The previous timing now overlaps a neighbouring chunk")
            c.history.pop()
            for f in _STATE_FIELDS:
                setattr(c, f, state[f])
            c.stale = set(state["stale"])
            return self._commit_edit("undo", c)

    def dismiss(self, chunk_id: str, kind: str) -> list[str]:
        """Clear a stale flag without redoing the step ("keep as it is")."""
        if kind not in STALE_KINDS:
            raise ValueError(f"Unknown stale kind {kind!r}")
        with self._lock:
            c = self.chunk(chunk_id)
            if kind not in c.stale:
                return []
            self._push_history(c, "dismiss")
            c.stale.discard(kind)
            return self._commit_edit("dismiss", c, output_stale=False)

    # ── results of operations (used by editor.ops) ──────────────────────────

    def apply_transcription(self, chunk_id: str, text: str) -> list[str]:
        """Store a re-transcription: clears the check flag.  A changed text
        makes translation and dub stale, as a manual edit does."""
        text = text.strip()
        with self._lock:
            c = self.chunk(chunk_id)
            if text == c.source_text and STALE_CHECK not in c.stale:
                return []
            self._push_history(c, "retranscribe")
            c.stale.discard(STALE_CHECK)
            changed = text != c.source_text
            if changed:
                c.source_text = text
                c.stale.add(STALE_TRANSLATION)
                _mark_dub_stale(c)
            return self._commit_edit("retranscribe", c, output_stale=changed or None)

    def apply_translation(self, chunk_id: str, text: str) -> list[str]:
        """Store a re-translation: clears the translation flag.  A changed
        text makes the dub stale."""
        text = text.strip()
        with self._lock:
            c = self.chunk(chunk_id)
            if text == c.target_text and STALE_TRANSLATION not in c.stale:
                return []
            self._push_history(c, "retranslate")
            c.stale.discard(STALE_TRANSLATION)
            changed = text != c.target_text
            if changed:
                c.target_text = text
                if not c.has_text:
                    c.dub_wav, c.dub_dur = None, 0.0
                _mark_dub_stale(c)
            return self._commit_edit("retranslate", c, output_stale=changed or None)

    def apply_dub(self, chunk_id: str, wav_path: str, duration: float) -> list[str]:
        """Store a new dub for the chunk: clears the dub flag."""
        with self._lock:
            c = self.chunk(chunk_id)
            self._push_history(c, "redub")
            c.dub_wav = project_relpath(self.proj, wav_path)
            c.dub_dur = float(duration)
            c.stale.discard(STALE_DUB)
            return self._commit_edit("redub", c)

    def mark_output_current(self) -> None:
        """Record that the final output was rebuilt from this session (assemble).

        Assembly rewrites ``subtitles/translated.srt`` from the session, so the
        stamp used by :meth:`out_of_sync` is refreshed too.
        """
        with self._lock:
            self.output_stale = False
            self.source_stamp = _file_stamp(self.proj.final_srt)
            self._log({"rev": self._bump(), "op": "assembled", "output_stale": False,
                       "source_stamp": self.source_stamp})
            self.save()

    def new_dub_path(self, chunk_id: str) -> str:
        """A fresh versioned path ``editor/segments/<id>_v<n>.wav`` for a re-dub."""
        seg_dir = os.path.join(self.proj.editor_dir, "segments")
        n = 1
        while os.path.exists(os.path.join(seg_dir, f"{chunk_id}_v{n}.wav")):
            n += 1
        return os.path.join(seg_dir, f"{chunk_id}_v{n}.wav")

    # ── persistence ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "language": self.language,
            "rev": self.rev,
            "next_id": self.next_id,
            "output_stale": self.output_stale,
            "duration": self.duration,
            "source_stamp": self.source_stamp,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def save(self) -> None:
        """Write a full snapshot and empty the change log."""
        with self._lock:
            if self.store is not None:
                self.store.write_snapshot(self.snapshot())

    # ── internals ────────────────────────────────────────────────────────────

    def _reindex(self) -> None:
        self._index = {c.id: i for i, c in enumerate(self.chunks)}

    def _new_id(self) -> str:
        cid = f"c{self.next_id}"
        self.next_id += 1
        return cid

    def _bump(self) -> int:
        self.rev += 1
        return self.rev

    @staticmethod
    def _push_history(c: Chunk, op: str) -> None:
        c.history.append({"op": op, "state": c.state()})
        if len(c.history) > HISTORY_LIMIT:
            del c.history[: len(c.history) - HISTORY_LIMIT]

    def _commit_edit(self, op: str, c: Chunk, *, output_stale: bool | None = True) -> list[str]:
        c.rev += 1
        return self._commit(op, self._index[c.id], [c.id], [c], output_stale=output_stale)

    def _commit(
        self, op: str, at: int, removed: list[str], new: list[Chunk],
        *, output_stale: bool | None = True,
    ) -> list[str]:
        """Replace ``chunks[at:at+len(removed)]`` by *new* and log it.

        ``output_stale=None`` leaves the flag unchanged.
        """
        self.chunks[at:at + len(removed)] = new
        if [c.id for c in new] != removed:
            self._reindex()
        if output_stale:
            self.output_stale = True
        self._log({
            "rev": self._bump(), "op": op, "at": at, "remove": removed,
            "chunks": [c.to_dict() for c in new],
            "next_id": self.next_id, "output_stale": self.output_stale,
        })
        return [c.id for c in new]

    def _log(self, record: dict) -> None:
        if self.store is None:
            return
        self.store.append(record)
        if self.store.pending >= MERGE_EVERY:
            self.save()

    def _apply_record(self, rec: dict) -> None:
        """Re-apply one change-log record (after-image) during load."""
        if "at" in rec:
            at, removed = rec["at"], rec["remove"]
            if [c.id for c in self.chunks[at:at + len(removed)]] != removed:
                raise ValueError(f"record does not match the session at position {at}")
            self.chunks[at:at + len(removed)] = [Chunk.from_dict(d) for d in rec["chunks"]]
            self._reindex()
            self.next_id = rec.get("next_id", self.next_id)
        self.output_stale = rec.get("output_stale", self.output_stale)
        self.source_stamp = rec.get("source_stamp", self.source_stamp)
        self.rev = rec["rev"]
