"""Gradio UI for the Editor tab of Mazinger Studio.

Layout, top to bottom: project picker and summary bar (bulk actions,
assemble), a settings form for projects without ``run.json``, the paginated
chunk table, the detail panel of the selected chunk, and the output area.

Sessions are shared by every browser tab (one :class:`Session` per project
language, kept in :data:`_SESSIONS`), so two tabs on the same project never
write conflicting change logs.  Per-tab view state (page, filter, selection)
lives in a ``gr.State`` dict.

Handlers return ``{name: value}`` dicts; :func:`build` maps the names to
components.  Only the current page of the table (:data:`PAGE_SIZE` rows) is
ever sent to the browser.
"""

import glob
import html
import logging
import os
import re
import threading
from datetime import datetime

import gradio as gr

from mazinger.editor import media, ops
from mazinger.editor.session import (
    BAD_FIT, STALE_CHECK, STALE_DUB, STALE_TRANSLATION, Session, UndoError,
    default_split_index, split_text, text_units,
)
from mazinger.gpu import GPUBusy
from mazinger.paths import ProjectPaths

log = logging.getLogger(__name__)

# Studio dubs into ./mazinger_output (MazingerDubber's default base_dir).
BASE_DIR = os.path.abspath(os.environ.get("MAZINGER_OUTPUT_DIR", "./mazinger_output"))

PAGE_SIZE = 50
TRUNCATE = 70
COLUMNS = ["#", "Start", "End", "Transcription", "Translation", "Status", "Fit"]

FILTER_ALL = "All"
FILTER_NEEDS_WORK = "Needs work"
FILTER_LONG = f"Fit > {round(BAD_FIT * 100)}%"
FILTER_ERRORS = "Errors"
FILTERS = [FILTER_ALL, FILTER_NEEDS_WORK, FILTER_LONG, FILTER_ERRORS]

_KIND_LABEL = {
    STALE_TRANSLATION: ("Re-translate", "re-translate"),
    STALE_DUB: ("Re-dub", "re-dub"),
    STALE_CHECK: ("Re-transcribe", "re-transcribe"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared state
# ═══════════════════════════════════════════════════════════════════════════════

_SESSIONS: dict = {}
_CLIPS: dict = {}
_ERRORS: dict = {}          # key → {chunk_id: last error}
_REGISTRY_LOCK = threading.Lock()
_RESOURCES = ops.ResourceManager()


def paths_for(key: str) -> ProjectPaths:
    """``<base>/projects/<slug>/lang/<language>`` → its :class:`ProjectPaths`."""
    lang_dir = os.path.abspath(key)
    language = os.path.basename(lang_dir)
    project_root = os.path.dirname(os.path.dirname(lang_dir))
    slug = os.path.basename(project_root)
    base = os.path.dirname(os.path.dirname(project_root))
    return ProjectPaths(slug, base_dir=base, target_language=language)


def key_for(proj: ProjectPaths) -> str:
    return os.path.abspath(os.path.dirname(proj.run_info))


def key_from_output(path: str | None) -> str | None:
    """Project key from any file inside ``lang/<language>/`` (e.g. a dub result)."""
    if not path:
        return None
    parts = os.path.abspath(path).split(os.sep)
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "lang" and i >= 2 and parts[i - 2] == "projects":
            return os.sep.join(parts[:i + 2])
    return None


def get_session(key: str, *, reload: bool = False) -> Session:
    with _REGISTRY_LOCK:
        session = _SESSIONS.get(key)
        if session is None or reload:
            session = Session.open(paths_for(key))
            _SESSIONS[key] = session
            _ERRORS.setdefault(key, {})
            _CLIPS[key] = media.clip_cache_for(session)
        return session


def reimport_session(key: str) -> Session:
    """Start the session over from the project's latest dub (drops edits)."""
    with _REGISTRY_LOCK:
        session = Session.import_project(paths_for(key))
        _SESSIONS[key] = session
        _ERRORS[key] = {}
        _CLIPS[key] = media.clip_cache_for(session)
        return session


def errors_for(key: str) -> dict:
    return _ERRORS.setdefault(key, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  Projects (5.3)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_projects(base_dir: str = BASE_DIR) -> list[dict]:
    """Finished dubs under *base_dir*, newest first."""
    out = []
    pattern = os.path.join(base_dir, "projects", "*", "lang", "*", "subtitles", "translated.srt")
    for srt in glob.glob(pattern):
        lang_dir = os.path.dirname(os.path.dirname(srt))
        if not os.path.isdir(os.path.join(lang_dir, "tts", "segments")):
            continue
        try:
            with open(srt, encoding="utf-8") as fh:
                chunks = fh.read().count("-->")
            mtime = os.path.getmtime(srt)
        except OSError:
            continue
        out.append({
            "key": os.path.abspath(lang_dir),
            "slug": os.path.basename(os.path.dirname(os.path.dirname(lang_dir))),
            "language": os.path.basename(lang_dir),
            "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "mtime": mtime,
            "chunks": chunks,
            "has_session": os.path.isfile(os.path.join(lang_dir, "editor", "session.json")),
            "has_run_info": os.path.isfile(os.path.join(lang_dir, "run.json")),
        })
    out.sort(key=lambda p: p["mtime"], reverse=True)
    return out


def project_label(p: dict) -> str:
    label = f"{p['slug']} · {p['language']} · {p['date']} · {p['chunks']:,} chunks"
    if p["has_session"]:
        label += " · ✏️ edited"
    return label


def project_choices(base_dir: str = BASE_DIR) -> list[tuple[str, str]]:
    return [(project_label(p), p["key"]) for p in scan_projects(base_dir)]


# ═══════════════════════════════════════════════════════════════════════════════
#  Table (5.5)
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_time(seconds: float) -> str:
    """``75.25`` → ``1:15.25``; ``3725.5`` → ``1:02:05.50``."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    return f"{int(m)}:{s:05.2f}"


_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+(?:[.,]\d*)?)\s*$")


def parse_time(text) -> float:
    """Parse ``ss``, ``m:ss`` or ``h:mm:ss`` (decimals allowed) into seconds.

    Raises:
        ValueError: on anything else.
    """
    if isinstance(text, (int, float)):
        return float(text)
    m = _TIME_RE.match(str(text or ""))
    if not m:
        raise ValueError(f"Not a time: {text!r} (use m:ss or h:mm:ss)")
    a, b, sec = m.groups()
    sec = float(sec.replace(",", "."))
    if a is not None and b is not None:
        return int(a) * 3600 + int(b) * 60 + sec
    if a is not None:
        return int(a) * 60 + sec
    return sec


def _truncate(text: str, n: int = TRUNCATE) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def row_status(chunk, error: str | None) -> str:
    return "❌ error" if error else " ".join(chunk.badges())


def filter_indices(session: Session, flt: str, query: str, errors: dict) -> list[int]:
    """Positions of the chunks matching the filter and the text search."""
    q = (query or "").strip().lower()
    out = []
    for i, c in enumerate(session.chunks):
        if flt == FILTER_NEEDS_WORK and not (c.stale or c.needs_dub):
            continue
        if flt == FILTER_LONG and not ((c.fit_ratio or 0) > BAD_FIT):
            continue
        if flt == FILTER_ERRORS and c.id not in errors:
            continue
        if q and q not in c.source_text.lower() and q not in c.target_text.lower():
            continue
        out.append(i)
    return out


def page_count(n: int) -> int:
    return max(1, -(-n // PAGE_SIZE))


def page_of_position(indices: list[int], position: int) -> int | None:
    """Page (0-based) of the chunk at session *position* within *indices*."""
    import bisect
    k = bisect.bisect_left(indices, position)
    if k < len(indices) and indices[k] == position:
        return k // PAGE_SIZE
    return None


_ROW_STYLE = {
    "error": "background-color: rgba(231, 111, 81, 0.28)",
    "stale": "background-color: rgba(244, 162, 97, 0.18)",
    "check": "background-color: rgba(42, 157, 143, 0.18)",
    "selected": "outline: 2px solid rgba(233, 196, 106, 0.9); outline-offset: -2px",
}


def table_page(session: Session, indices: list[int], page: int, errors: dict,
               selected: str | None = None):
    """The rows of one page as a styled DataFrame, plus their chunk ids."""
    import pandas as pd

    page = max(0, min(page, page_count(len(indices)) - 1))
    rows, ids, styles = [], [], []
    for pos in indices[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
        c = session.chunks[pos]
        err = errors.get(c.id)
        fit = c.fit_ratio
        rows.append([
            pos + 1, fmt_time(c.start), fmt_time(c.end),
            _truncate(c.source_text), _truncate(c.target_text),
            row_status(c, err), f"{fit * 100:.0f}%" if fit is not None else "—",
        ])
        ids.append(c.id)
        style = ""
        if err:
            style = _ROW_STYLE["error"]
        elif c.stale - {STALE_CHECK} or c.needs_dub:
            style = _ROW_STYLE["stale"]
        elif STALE_CHECK in c.stale:
            style = _ROW_STYLE["check"]
        if c.id == selected:
            style = "; ".join(s for s in (style, _ROW_STYLE["selected"]) if s)
        styles.append(style)

    df = pd.DataFrame(rows, columns=COLUMNS)
    styler = df.style.apply(lambda row: [styles[row.name]] * len(row), axis=1)
    return styler, ids, page


def summary_text(session: Session) -> str:
    n = session.counts()
    parts = [f"**{n['chunks']:,}** chunks"]
    if n[STALE_TRANSLATION]:
        parts.append(f"⚠ {n[STALE_TRANSLATION]:,} to re-translate")
    if n[STALE_DUB]:
        parts.append(f"⚠ {n[STALE_DUB]:,} to re-dub")
    if n[STALE_CHECK]:
        parts.append(f"🔍 {n[STALE_CHECK]:,} to check")
    if n["long"]:
        parts.append(f"⏱ {n['long']:,} too long")
    parts.append("**output out of date**" if session.output_stale else "output up to date")
    return " · ".join(parts)


def bulk_label(session: Session, kind: str) -> tuple[str, bool]:
    n = len(ops.stale_ids(session, kind))
    verb = _KIND_LABEL[kind][0]
    return (f"{verb} {n:,} stale" if n else f"{verb} stale"), bool(n)


# ═══════════════════════════════════════════════════════════════════════════════
#  Detail panel (5.6)
# ═══════════════════════════════════════════════════════════════════════════════

def detail_values(session: Session, chunk_id: str | None, errors: dict,
                  clips: media.ClipCache | None) -> dict:
    """Values for the detail panel of *chunk_id* (empty panel when ``None``)."""
    if not chunk_id or chunk_id not in session.ids():
        return {"detail_md": "*Select a row to edit it.*", "orig_audio": None, "dub_audio": None,
                "src": "", "tgt": "", "start": "", "end": "", "row_error": "", "dirty": ""}
    i = session.index(chunk_id)
    c = session.chunks[i]
    fit = c.fit_ratio
    head = (f"### #{i + 1} · {fmt_time(c.start)} – {fmt_time(c.end)} · {c.duration:.2f} s"
            + (f" · dub {c.dub_dur:.2f} s ({fit * 100:.0f}%)" if fit is not None else ""))
    badges = " ".join(f"`{b}`" for b in c.badges())

    orig = None
    if clips is not None and os.path.isfile(session.proj.audio):
        try:
            orig = clips.clip(c.start, c.end)
            clips.prefetch(media.neighbour_ranges(session, chunk_id))
        except Exception as exc:  # noqa: BLE001 — the panel still works without audio
            log.warning("Could not cut the original clip: %s", exc)
    err = errors.get(chunk_id)
    return {
        "detail_md": f"{head}\n\n{badges}",
        "orig_audio": orig,
        "dub_audio": session.abs_path(c.dub_wav) if c.dub_wav else None,
        "src": c.source_text, "tgt": c.target_text,
        "start": fmt_time(c.start), "end": fmt_time(c.end),
        "row_error": f"❌ **Last action failed:** {html.escape(err)}" if err else "",
        "dirty": "",
    }


def split_defaults(session: Session, chunk_id: str, at: float | None = None) -> dict:
    c = session.chunk(chunk_id)
    if at is None:
        at = round((c.start + c.end) / 2, 2)
    frac = (at - c.start) / c.duration if c.duration else 0.5
    return {
        "at": at,
        "src_max": len(text_units(c.source_text)), "tgt_max": len(text_units(c.target_text)),
        "src_idx": default_split_index(c.source_text, frac),
        "tgt_idx": default_split_index(c.target_text, frac),
    }


def split_preview(session: Session, chunk_id: str, at: float, src_idx: int, tgt_idx: int) -> str:
    c = session.chunk(chunk_id)
    src_a, src_b = split_text(c.source_text, int(src_idx))
    tgt_a, tgt_b = split_text(c.target_text, int(tgt_idx))
    return (f"**1st** {fmt_time(c.start)}–{fmt_time(at)}: {src_a or '∅'} → *{tgt_a or '∅'}*\n\n"
            f"**2nd** {fmt_time(at)}–{fmt_time(c.end)}: {src_b or '∅'} → *{tgt_b or '∅'}*")


# ═══════════════════════════════════════════════════════════════════════════════
#  Settings form for projects without run.json (5.4)
# ═══════════════════════════════════════════════════════════════════════════════

VOICE_FROM_SEGMENT = "Use one of the dubbed segments"
VOICE_FROM_FILE = "Upload the voice sample"
TRANSCRIBE_METHODS = ["faster-whisper", "whisperx", "coherex", "openai", "deepgram", "mlx-whisper"]
TTS_ENGINES = ["qwen", "omnivoice", "chatterbox", "mlx"]


def settings_from_form(
    proj: ProjectPaths, *, source_language: str, transcribe_method: str, whisper_model: str,
    llm_model: str, llm_base_url: str, translation_model: str,
    tts_engine: str, tts_model: str, tts_dtype: str,
    voice_mode: str, voice_file: str | None, voice_script: str,
    tempo_mode: str, max_tempo: float, loudness_match: bool, mix_background: bool,
    background_volume: float, output_type: str,
) -> dict:
    """A run record (``run.json`` shape) from the Editor's settings form."""
    from mazinger.runinfo import project_relpath

    if voice_mode == VOICE_FROM_FILE:
        if not voice_file:
            raise ValueError("Upload the voice sample the project was dubbed with")
        from mazinger.profiles import keep_voice_reference
        wav, script = keep_voice_reference(voice_file, (voice_script or "").strip() or None,
                                           proj.voice_reference_dir)
        voice = {"kind": "sample", "sample": project_relpath(proj, wav),
                 "script": project_relpath(proj, script) if script else None}
    else:
        voice = {"kind": "dub-segment", "sample": None, "script": None}
    return {
        "slug": proj.slug,
        "target_language": proj.target_language,
        "source_language": source_language or "auto",
        "transcription": {"method": transcribe_method, "model": (whisper_model or "").strip() or None},
        "llm": {"model": (llm_model or "").strip() or None,
                "base_url": (llm_base_url or "").strip() or None, "think": None},
        "translation": {"translation_model": (translation_model or "").strip() or None},
        "tts": {"engine": tts_engine, "model": (tts_model or "").strip() or None,
                "dtype": tts_dtype, "language": proj.target_language},
        "voice": voice,
        "assembly": {"tempo_mode": tempo_mode, "fixed_tempo": None, "max_tempo": float(max_tempo),
                     "loudness_match": bool(loudness_match), "mix_background": bool(mix_background),
                     "background_volume": float(background_volume)},
        "output": {"output_type": output_type, "subtitle_style": None, "subtitle_source": "translated"},
        "entered_in_editor": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Handlers (return {name: value}; see build())
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_view() -> dict:
    return {"key": None, "filter": FILTER_ALL, "query": "", "page": 0,
            "selected": None, "page_ids": [], "pending": None}


def refresh(view: dict, *, detail: bool = True) -> dict:
    """Everything that depends on the session: table page, summary, detail."""
    key = view.get("key")
    if not key:
        return {"view": view}
    session = get_session(key)
    errors = errors_for(key)
    indices = filter_indices(session, view["filter"], view["query"], errors)
    if view.get("selected") not in session.ids():
        view["selected"] = None
    styler, ids, page = table_page(session, indices, view["page"], errors, view["selected"])
    view = {**view, "page": page, "page_ids": ids}
    tr_label, tr_on = bulk_label(session, STALE_TRANSLATION)
    dub_label, dub_on = bulk_label(session, STALE_DUB)
    chk_label, chk_on = bulk_label(session, STALE_CHECK)
    out = {
        "view": view,
        "table": styler,
        "page_md": f"Page {page + 1} of {page_count(len(indices))} · {len(indices):,} rows",
        "summary": summary_text(session),
        "bulk_translate": gr.update(value=tr_label, interactive=tr_on),
        "bulk_dub": gr.update(value=dub_label, interactive=dub_on),
        "bulk_check": gr.update(value=chk_label, interactive=chk_on),
    }
    if detail:
        out.update(detail_values(session, view["selected"], errors, _CLIPS.get(key)))
        out["split_group"] = gr.update(visible=False)
    return out


def _progress_refresh(view: dict) -> dict:
    """:func:`refresh` for updates while an operation runs: the table and
    summary, but not the (disabled) action buttons."""
    out = refresh(view, detail=False)
    for name in ("bulk_translate", "bulk_dub", "bulk_check"):
        out.pop(name, None)
    return out


def banner_for(session: Session, notices: list[str] | None = None) -> str:
    lines = []
    if session.out_of_sync():
        lines.append("⚠️ This project was dubbed again after these edits were started. "
                     "Use **Start over from the latest dub** to edit the new dub (your edits are dropped).")
    if session.run_info is None:
        lines.append("⚙️ This project has no record of its dub settings. "
                     "Fill in **Dub settings** below before re-doing any step.")
    lines += [f"ℹ️ {n}" for n in (notices or [])]
    return "\n\n".join(lines)


def h_open(key: str | None, view: dict) -> dict:
    view = _empty_view()
    if not key:
        return {"view": view, "status": "", "settings_group": gr.update(visible=False)}
    try:
        session = get_session(key)
    except Exception as exc:  # noqa: BLE001 — shown to the user
        log.exception("Could not open %s", key)
        return {"view": view, "status": f"❌ Could not open the project: {html.escape(str(exc))}"}
    view["key"] = key
    view["selected"] = session.chunks[0].id if session.chunks else None
    return {
        **refresh(view),
        "status": banner_for(session),
        "settings_group": gr.update(visible=session.run_info is None),
        "log": "", **_outputs(None, None, None, list(ops.previous_outputs(session).values())),
    }


def h_filter(view: dict, flt: str, query: str) -> dict:
    return refresh({**view, "filter": flt, "query": query or "", "page": 0}, detail=False)


def h_page(view: dict, delta: int) -> dict:
    return refresh({**view, "page": view.get("page", 0) + delta}, detail=False)


def _select(view: dict, chunk_id: str | None) -> dict:
    """Select *chunk_id* and move to the page that shows it."""
    if not view.get("key") or not chunk_id:
        return refresh(view)
    session = get_session(view["key"])
    view = {**view, "selected": chunk_id}
    indices = filter_indices(session, view["filter"], view["query"], errors_for(view["key"]))
    page = page_of_position(indices, session.index(chunk_id))
    if page is None:  # outside the filter: show it unfiltered
        view.update(filter=FILTER_ALL, query="")
        indices = list(range(len(session)))
        page = page_of_position(indices, session.index(chunk_id))
    view["page"] = page
    return {**refresh(view), "filter": view["filter"], "search": view["query"]}


def h_select_row(view: dict, row: int) -> dict:
    ids = view.get("page_ids") or []
    if not 0 <= row < len(ids):
        return {}
    return _select(view, ids[row])


def h_step(view: dict, delta: int) -> dict:
    """◀ Prev / Next ▶ within the filtered rows."""
    if not view.get("key") or not view.get("selected"):
        return {}
    session = get_session(view["key"])
    indices = filter_indices(session, view["filter"], view["query"], errors_for(view["key"]))
    pos = session.index(view["selected"])
    order = indices if pos in indices else list(range(len(session)))
    k = order.index(pos) + delta
    if not 0 <= k < len(order):
        return {}
    return _select(view, session.chunks[order[k]].id)


def h_jump(view: dict, text: str) -> dict:
    if not view.get("key"):
        return {}
    try:
        t = parse_time(text)
    except ValueError as exc:
        return {"status": f"❌ {exc}"}
    session = get_session(view["key"])
    target = next((c for c in session.chunks if c.end > t), session.chunks[-1] if session.chunks else None)
    return {**_select(view, target.id if target else None), "status": ""}


def h_save(view: dict, src: str, tgt: str, start: str, end: str) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    session = get_session(view["key"])
    try:
        s, e = parse_time(start), parse_time(end)
    except ValueError as exc:
        return {"row_error": f"❌ {exc}"}
    c = session.chunk(cid)
    changed = []
    changed += session.set_source_text(cid, src or "")
    changed += session.set_target_text(cid, tgt or "")
    if (round(s, 2), round(e, 2)) != (round(c.start, 2), round(c.end, 2)):
        changed += session.set_timing(cid, s, e)
    out = refresh(view)
    if changed:
        errors_for(view["key"]).pop(cid, None)
        out["row_error"] = ""
    return out


def h_undo(view: dict) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    session = get_session(view["key"])
    try:
        ids = session.undo(cid)
    except UndoError as exc:
        return {"row_error": f"❌ {html.escape(str(exc))}"}
    if not ids:
        return {"row_error": "Nothing to undo for this chunk."}
    return _select(view, ids[0])


def h_merge(view: dict) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    try:
        (new,) = get_session(view["key"]).merge_with_next(cid)
    except ValueError as exc:
        return {"row_error": f"❌ {html.escape(str(exc))}"}
    return _select(view, new)


def h_split_open(view: dict) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    session = get_session(view["key"])
    d = split_defaults(session, cid)
    return {
        "split_group": gr.update(visible=True),
        "split_time": fmt_time(d["at"]),
        "src_split": gr.update(maximum=max(1, d["src_max"]), value=d["src_idx"]),
        "tgt_split": gr.update(maximum=max(1, d["tgt_max"]), value=d["tgt_idx"]),
        "split_preview": split_preview(session, cid, d["at"], d["src_idx"], d["tgt_idx"]),
    }


def h_split_time(view: dict, at_text: str) -> dict:
    """A new split time re-proposes the text split points."""
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    session = get_session(view["key"])
    try:
        at = parse_time(at_text)
    except ValueError as exc:
        return {"split_preview": f"❌ {exc}"}
    d = split_defaults(session, cid, at)
    return {"src_split": d["src_idx"], "tgt_split": d["tgt_idx"],
            "split_preview": split_preview(session, cid, at, d["src_idx"], d["tgt_idx"])}


def h_split_preview(view: dict, at_text: str, src_idx: int, tgt_idx: int) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    try:
        at = parse_time(at_text)
    except ValueError as exc:
        return {"split_preview": f"❌ {exc}"}
    return {"split_preview": split_preview(get_session(view["key"]), cid, at, src_idx, tgt_idx)}


def h_split_apply(view: dict, at_text: str, src_idx: int, tgt_idx: int) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    try:
        a, _b = get_session(view["key"]).split(cid, parse_time(at_text), int(src_idx), int(tgt_idx))
    except ValueError as exc:
        return {"split_preview": f"❌ {html.escape(str(exc))}"}
    return _select(view, a)


def h_dismiss_check(view: dict) -> dict:
    cid = view.get("selected")
    if not view.get("key") or not cid:
        return {}
    get_session(view["key"]).dismiss(cid, STALE_CHECK)
    return refresh(view)


def h_ask(view: dict, kind: str) -> dict:
    """First click of a bulk button: ask for confirmation with the count."""
    if not view.get("key"):
        return {}
    n = len(ops.stale_ids(get_session(view["key"]), kind))
    if not n:
        return {"confirm_group": gr.update(visible=False), "view": {**view, "pending": None}}
    verb = _KIND_LABEL[kind][1]
    note = " Models are loaded on first use; this can take a while." if kind != STALE_TRANSLATION else ""
    return {
        "view": {**view, "pending": {"kind": kind, "count": n}},
        "confirm_group": gr.update(visible=True),
        "confirm_md": f"**{verb.capitalize()} {n:,} chunk{'s' if n != 1 else ''}?**{note}",
    }


def h_cancel(view: dict) -> dict:
    return {"view": {**view, "pending": None}, "confirm_group": gr.update(visible=False)}


def h_save_settings(view: dict, *form) -> dict:
    if not view.get("key"):
        return {}
    from mazinger.runinfo import load_run_info, save_run_info
    session = get_session(view["key"])
    names = ["source_language", "transcribe_method", "whisper_model", "llm_model", "llm_base_url",
             "translation_model", "tts_engine", "tts_model", "tts_dtype", "voice_mode", "voice_file",
             "voice_script", "tempo_mode", "max_tempo", "loudness_match", "mix_background",
             "background_volume", "output_type"]
    try:
        info = settings_from_form(session.proj, **dict(zip(names, form)))
        save_run_info(session.proj, info)
    except (ValueError, OSError) as exc:
        return {"settings_msg": f"❌ {html.escape(str(exc))}"}
    session.run_info = load_run_info(session.proj)
    _RESOURCES.free()
    return {"settings_msg": "✅ Saved to run.json", "settings_group": gr.update(visible=False),
            "status": banner_for(session)}


def h_reimport(view: dict) -> dict:
    if not view.get("key"):
        return {}
    reimport_session(view["key"])
    return h_open(view["key"], view)


def h_free_gpu() -> dict:
    _RESOURCES.free()
    return {"status": "✅ Editor models unloaded; GPU memory freed."}


def _resources(session: Session, api_key: str | None, dub_api_key: str | None) -> ops.Resources:
    res = _RESOURCES.get(session)
    res.set_api_key((api_key or "").strip() or (dub_api_key or "").strip() or None)
    return res


def _chunk_label(session: Session, cid: str | None) -> str:
    try:
        return f"#{session.index(cid) + 1}"
    except (KeyError, TypeError):
        return str(cid)


_RUNNERS = {
    STALE_CHECK: ops.retranscribe,
    STALE_TRANSLATION: ops.retranslate,
    STALE_DUB: ops.redub,
}


def h_run(view: dict, kind: str, scope: str, api_key: str = "", dub_api_key: str = ""):
    """Run an operation on the selected chunk (``scope="one"``) or on every
    stale chunk after confirmation (``scope="stale"``).  Generator."""
    if not view.get("key"):
        yield {}
        return
    session = get_session(view["key"])
    errors = errors_for(view["key"])
    pending, view = view.get("pending"), {**view, "pending": None}
    yield {"busy": True, "view": view, "confirm_group": gr.update(visible=False),
           "log": f"⏳ {_KIND_LABEL[kind][0]}…"}
    lines: list[str] = []
    try:
        res = _resources(session, api_key, dub_api_key)
        if scope == "one":
            if not view.get("selected"):
                raise ValueError("Select a row first")
            gen = _RUNNERS[kind](session, [view["selected"]], res)
        else:
            gen = ops.redo_stale(session, kind, res,
                                 expected=pending["count"] if pending and pending["kind"] == kind else None)
        for p in gen:
            if p.chunk_id:
                label = _chunk_label(session, p.chunk_id)
                if p.error:
                    errors[p.chunk_id] = p.error
                    lines.append(f"❌ {label}: {p.error}")
                else:
                    errors.pop(p.chunk_id, None)
                    lines.append(f"✓ {label} ({p.done}/{p.total})")
            if p.message:
                lines.append(p.message)
            yield {**_progress_refresh(view), "log": "\n".join(lines[-300:])}
        notices = list(res.notices)
        res.notices.clear()
        status = banner_for(session, notices)
    except (GPUBusy, ops.SettingsMissing, ops.VoiceUnavailable, ValueError) as exc:
        lines.append(f"❌ {exc}")
        status = f"❌ {html.escape(str(exc))}"
    except Exception as exc:  # noqa: BLE001 — keep the UI alive
        log.exception("Editor operation failed")
        lines.append(f"❌ {type(exc).__name__}: {exc}")
        status = f"❌ {html.escape(str(exc))}"
    yield {**refresh(view), "busy": False, "log": "\n".join(lines[-300:]), "status": status}


def _outputs(audio, video, files, prev) -> dict:
    """Output players and file lists, each shown only when it has content."""
    return {
        "out_audio": gr.update(value=audio, visible=bool(audio)),
        "out_video": gr.update(value=video, visible=bool(video)),
        "out_files": gr.update(value=files or None, visible=bool(files)),
        "prev_files": gr.update(value=prev or None, visible=bool(prev)),
    }


def h_assemble(view: dict):
    """Rebuild the final output.  Generator."""
    if not view.get("key"):
        yield {}
        return
    session = get_session(view["key"])
    yield {"busy": True, "log": "⏳ Assembling…"}
    lines: list[str] = []
    result: dict = {}
    try:
        for p in ops.assemble(session):
            lines.append(p.message)
            result = p.outputs or result
            yield {"log": "\n".join(lines)}
        status = "✅ Output rebuilt." + (" Previous version kept." if result.get("previous_audio") else "")
    except GPUBusy as exc:
        lines.append(f"❌ {exc}")
        status = f"❌ {html.escape(str(exc))}"
    except Exception as exc:  # noqa: BLE001
        log.exception("Assembly failed")
        lines.append(f"❌ Assembly failed: {exc}. The previous output was left in place.")
        status = f"❌ Assembly failed: {html.escape(str(exc))}"
    files = [result[k] for k in ("audio", "video", "srt", "source_srt", "display_srt")
             if result.get(k) and os.path.isfile(result[k])]
    prev = [p for p in ops.previous_outputs(session).values()]
    yield {
        **refresh(view, detail=False), "busy": False, "log": "\n".join(lines), "status": status,
        **_outputs(result.get("audio"), result.get("video"), files, prev),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Layout and wiring
# ═══════════════════════════════════════════════════════════════════════════════

class EditorTab:
    """Handles to the components other parts of Studio wire into."""

    def __init__(self, project, view, open_fn, outputs, components):
        self.components = components
        self.project = project
        self.view = view
        self.open_fn = open_fn
        self.outputs = outputs


def build(dub_api_key: gr.components.Component | None = None) -> EditorTab:
    """Build the Editor tab's contents inside the current ``gr.Tab``.

    Parameters:
        dub_api_key: The Dub tab's API-key textbox; its value is used when the
                     Editor's own key field is empty.
    """
    c: dict = {}
    view = gr.State(_empty_view())
    c["view"] = view

    gr.Markdown(
        "Fix a finished dub chunk by chunk. Edits only **mark** what needs redoing; "
        "nothing runs until you ask, and **Assemble** rebuilds the output.",
        elem_classes="openai-info",
    )

    # ── Project + summary (5.3, 5.7) ─────────────────────────────────────────
    gr.Markdown("#### 📂  PROJECT", elem_classes="section-title")
    with gr.Group(elem_classes="card"):
        with gr.Row(equal_height=True):
            c["project"] = gr.Dropdown(choices=project_choices(), value=None, label="Finished dub",
                                       scale=5, allow_custom_value=False)
            refresh_btn = gr.Button("🔄 Refresh list", scale=1)
        c["status"] = gr.Markdown("")
        c["summary"] = gr.Markdown("")
        with gr.Row():
            c["bulk_translate"] = gr.Button("Re-translate stale", interactive=False)
            c["bulk_dub"] = gr.Button("Re-dub stale", interactive=False)
            c["bulk_check"] = gr.Button("Re-transcribe stale", interactive=False)
            c["assemble"] = gr.Button("🎬 Assemble", variant="primary")
            c["free_gpu"] = gr.Button("🧹 Free GPU")
        with gr.Group(visible=False) as confirm_group:
            c["confirm_md"] = gr.Markdown("")
            with gr.Row():
                c["confirm_yes"] = gr.Button("Yes, go", variant="primary")
                c["confirm_no"] = gr.Button("Cancel")
        c["confirm_group"] = confirm_group
        with gr.Accordion("🔑 LLM API key", open=False):
            c["api_key"] = gr.Textbox(
                label="API key (not saved)", type="password",
                placeholder="Leave empty to use the Dub tab's key or OPENAI_API_KEY",
            )
        with gr.Row(visible=False) as reimport_row:
            c["reimport"] = gr.Button("Start over from the latest dub", variant="stop")
        c["reimport_row"] = reimport_row

    # ── Settings for projects without run.json (5.4) ────────────────────────
    from mazinger.studio.constants import LANGUAGES
    with gr.Group(visible=False, elem_classes="card") as settings_group:
        gr.Markdown("#### ⚙️  DUB SETTINGS\nThis project was dubbed before Mazinger recorded its "
                    "settings. Enter them once; they are saved to `run.json`.")
        with gr.Row():
            s_source = gr.Dropdown(["auto"] + LANGUAGES, value="auto", label="Source language")
            s_method = gr.Dropdown(TRANSCRIBE_METHODS, value="faster-whisper", label="Transcription")
            s_whisper = gr.Textbox(label="Transcription model", placeholder="default")
        with gr.Row():
            s_llm = gr.Textbox(label="LLM model", placeholder="gpt-4.1")
            s_base = gr.Textbox(label="LLM base URL", placeholder="empty = OpenAI; http://localhost:11434/v1 = Ollama")
            s_trmodel = gr.Textbox(label="Translation model (optional)", placeholder="e.g. translategemma")
        with gr.Row():
            s_engine = gr.Dropdown(TTS_ENGINES, value="qwen", label="TTS engine")
            s_tts_model = gr.Textbox(label="TTS model", placeholder="default")
            s_dtype = gr.Dropdown(["bfloat16", "float16", "float32"], value="bfloat16", label="TTS dtype")
        s_voice_mode = gr.Radio([VOICE_FROM_SEGMENT, VOICE_FROM_FILE], value=VOICE_FROM_SEGMENT,
                                label="Voice for re-dubs")
        with gr.Row():
            s_voice_file = gr.File(label="Voice sample", type="filepath", file_types=["audio"])
            s_voice_script = gr.Textbox(label="Voice sample transcript (optional)", lines=3)
        with gr.Row():
            s_tempo = gr.Dropdown(["auto", "off"], value="auto", label="Tempo")
            s_max_tempo = gr.Slider(1.0, 2.0, value=1.5, step=0.05, label="Max tempo")
            s_bg_vol = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Background volume")
        with gr.Row():
            s_loud = gr.Checkbox(value=True, label="Match loudness")
            s_mix = gr.Checkbox(value=True, label="Mix background")
            s_output = gr.Radio(["audio", "video"], value="audio", label="Output")
        with gr.Row():
            save_settings = gr.Button("Save settings", variant="primary")
            c["settings_msg"] = gr.Markdown("")
    c["settings_group"] = settings_group
    settings_inputs = [s_source, s_method, s_whisper, s_llm, s_base, s_trmodel, s_engine,
                       s_tts_model, s_dtype, s_voice_mode, s_voice_file, s_voice_script,
                       s_tempo, s_max_tempo, s_loud, s_mix, s_bg_vol, s_output]

    # ── Chunk table (5.5) ────────────────────────────────────────────────────
    gr.Markdown("#### 🧩  CHUNKS", elem_classes="section-title")
    with gr.Group(elem_classes="card"):
        with gr.Row(equal_height=True):
            c["filter"] = gr.Radio(FILTERS, value=FILTER_ALL, label="Show", scale=3)
            c["search"] = gr.Textbox(label="Search text", placeholder="press Enter", scale=2)
            jump_box = gr.Textbox(label="Jump to time", placeholder="m:ss", scale=1)
        c["table"] = gr.Dataframe(
            headers=COLUMNS, datatype=["number"] + ["str"] * 6, interactive=False,
            wrap=True, max_height=560, column_widths=["6%", "9%", "9%", "30%", "30%", "10%", "6%"],
        )
        with gr.Row(equal_height=True):
            prev_page = gr.Button("◀ Page", scale=1)
            c["page_md"] = gr.Markdown("", elem_classes="page-info")
            next_page = gr.Button("Page ▶", scale=1)

    # ── Detail panel (5.6) ───────────────────────────────────────────────────
    gr.Markdown("#### ✏️  SELECTED CHUNK", elem_classes="section-title")
    with gr.Group(elem_classes="card"):
        c["detail_md"] = gr.Markdown("*Select a row to edit it.*")
        with gr.Row():
            c["orig_audio"] = gr.Audio(label="Original", type="filepath", interactive=False)
            c["dub_audio"] = gr.Audio(label="Dubbed", type="filepath", interactive=False)
        with gr.Row():
            c["src"] = gr.Textbox(label="Transcription", lines=3)
            c["tgt"] = gr.Textbox(label="Translation", lines=3)
        with gr.Row():
            c["start"] = gr.Textbox(label="Start", scale=1)
            c["end"] = gr.Textbox(label="End", scale=1)
            c["dirty"] = gr.Markdown("", elem_classes="dirty-hint")
        c["row_error"] = gr.Markdown("")
        with gr.Row():
            c["save"] = gr.Button("💾 Save", variant="primary")
            c["undo"] = gr.Button("↶ Undo")
            c["one_check"] = gr.Button("🎙 Re-transcribe")
            c["one_translate"] = gr.Button("🌐 Re-translate")
            c["one_dub"] = gr.Button("🔊 Re-dub")
        with gr.Row():
            c["prev_row"] = gr.Button("◀ Prev")
            c["next_row"] = gr.Button("Next ▶")
            c["split_open"] = gr.Button("✂️ Split…")
            c["merge"] = gr.Button("🔗 Merge with next")
            c["dismiss"] = gr.Button("✓ Timing is fine")
        with gr.Group(visible=False) as split_group:
            with gr.Row():
                c["split_time"] = gr.Textbox(label="Split at", scale=1)
                c["src_split"] = gr.Slider(0, 1, step=1, label="Transcription: words in the 1st part", scale=2)
                c["tgt_split"] = gr.Slider(0, 1, step=1, label="Translation: words in the 1st part", scale=2)
            c["split_preview"] = gr.Markdown("")
            with gr.Row():
                c["split_apply"] = gr.Button("Split", variant="primary")
                split_cancel = gr.Button("Cancel")
        c["split_group"] = split_group

    # ── Output (5.8) ─────────────────────────────────────────────────────────
    gr.Markdown("#### 📦  OUTPUT", elem_classes="section-title")
    with gr.Group(elem_classes="results-card"):
        c["log"] = gr.Textbox(label="Log", lines=6, max_lines=16, interactive=False,
                              autoscroll=True, elem_classes="log-box")
        with gr.Row():
            c["out_audio"] = gr.Audio(label="Dubbed audio", type="filepath", interactive=False,
                                      visible=False)
            c["out_video"] = gr.Video(label="Dubbed video", interactive=False, visible=False)
        c["out_files"] = gr.File(label="Downloads", file_count="multiple", interactive=False,
                                 visible=False)
        c["prev_files"] = gr.File(label="Previous version", file_count="multiple",
                                  interactive=False, visible=False)

    # Controls disabled while an operation runs (5.10).
    action_names = ["bulk_translate", "bulk_dub", "bulk_check", "assemble", "save", "undo",
                    "one_check", "one_translate", "one_dub", "split_open", "split_apply",
                    "merge", "dismiss", "confirm_yes", "reimport"]
    outputs = [comp for comp in c.values()]

    def to_updates(result: dict) -> dict:
        out = {}
        busy = result.pop("busy", None)
        if busy is not None:
            for name in action_names:
                out[c[name]] = gr.update(interactive=not busy)
        view_val = result.get("view")
        for name, value in result.items():
            if name in c:
                out[c[name]] = value
        if view_val is not None and view_val.get("key"):
            session = _SESSIONS.get(view_val["key"])
            if session is not None:
                out[c["reimport_row"]] = gr.update(visible=session.out_of_sync())
        return out

    def wrap(fn):
        def run(*args):
            return to_updates(fn(*args))
        return run

    def wrap_gen(fn):
        def run(*args):
            for result in fn(*args):
                yield to_updates(result)
        return run

    key_inputs = [c["api_key"]] + ([dub_api_key] if dub_api_key is not None else [])

    def op(kind, scope):
        def run(v, *keys):
            yield from wrap_gen(h_run)(v, kind, scope, *keys)
        return run

    # Project
    open_fn = wrap(h_open)
    c["project"].change(open_fn, [c["project"], view], outputs)
    refresh_btn.click(lambda: gr.update(choices=project_choices()), None, c["project"])
    c["reimport"].click(wrap(h_reimport), view, outputs)
    save_settings.click(wrap(h_save_settings), [view] + settings_inputs, outputs)
    c["free_gpu"].click(wrap(h_free_gpu), None, outputs)

    # Table
    c["filter"].change(wrap(h_filter), [view, c["filter"], c["search"]], outputs)
    c["search"].submit(wrap(h_filter), [view, c["filter"], c["search"]], outputs)
    prev_page.click(wrap(lambda v: h_page(v, -1)), view, outputs)
    next_page.click(wrap(lambda v: h_page(v, +1)), view, outputs)
    jump_box.submit(wrap(h_jump), [view, jump_box], outputs)

    def on_select(v, evt: gr.SelectData):
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return to_updates(h_select_row(v, int(row)))
    c["table"].select(on_select, view, outputs)

    # Detail
    for box in ("src", "tgt", "start", "end"):
        c[box].input(lambda: "● Unsaved changes", None, c["dirty"], queue=False, show_progress="hidden")
    c["save"].click(wrap(h_save), [view, c["src"], c["tgt"], c["start"], c["end"]], outputs)
    c["undo"].click(wrap(h_undo), view, outputs)
    c["merge"].click(wrap(h_merge), view, outputs)
    c["dismiss"].click(wrap(h_dismiss_check), view, outputs)
    c["prev_row"].click(wrap(lambda v: h_step(v, -1)), view, outputs)
    c["next_row"].click(wrap(lambda v: h_step(v, +1)), view, outputs)
    c["split_open"].click(wrap(h_split_open), view, outputs)
    c["split_time"].submit(wrap(h_split_time), [view, c["split_time"]], outputs)
    for s in ("src_split", "tgt_split"):
        c[s].change(wrap(h_split_preview),
                     [view, c["split_time"], c["src_split"], c["tgt_split"]], outputs)
    c["split_apply"].click(wrap(h_split_apply),
                           [view, c["split_time"], c["src_split"], c["tgt_split"]], outputs)
    split_cancel.click(lambda: gr.update(visible=False), None, c["split_group"])

    # Operations on the selected chunk
    c["one_check"].click(op(STALE_CHECK, "one"), [view] + key_inputs, outputs)
    c["one_translate"].click(op(STALE_TRANSLATION, "one"), [view] + key_inputs, outputs)
    c["one_dub"].click(op(STALE_DUB, "one"), [view] + key_inputs, outputs)

    # Bulk operations: ask, then run on confirm
    c["bulk_translate"].click(wrap(lambda v: h_ask(v, STALE_TRANSLATION)), view, outputs)
    c["bulk_dub"].click(wrap(lambda v: h_ask(v, STALE_DUB)), view, outputs)
    c["bulk_check"].click(wrap(lambda v: h_ask(v, STALE_CHECK)), view, outputs)
    c["confirm_no"].click(wrap(h_cancel), view, outputs)

    def confirmed(v, *keys):
        kind = (v.get("pending") or {}).get("kind")
        if not kind:
            yield to_updates(h_cancel(v))
            return
        yield from wrap_gen(h_run)(v, kind, "stale", *keys)
    c["confirm_yes"].click(confirmed, [view] + key_inputs, outputs)

    c["assemble"].click(wrap_gen(h_assemble), view, outputs)

    return EditorTab(c["project"], view, open_fn, outputs, c)


def open_from_dub(render_paths: dict | None):
    """"Open in Editor" after a dub: the project key and the refreshed choices."""
    key = None
    for path in (render_paths or {}).values():
        key = key_from_output(path)
        if key:
            break
    base = paths_for(key).base_dir if key else BASE_DIR
    return gr.update(choices=project_choices(base), value=key)
