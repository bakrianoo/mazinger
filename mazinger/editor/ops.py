"""Editor operations: re-transcribe, re-translate, re-dub and assemble.

Operations take their settings from the project's ``run.json`` (see
:mod:`mazinger.runinfo`), or from a dict of the same shape for projects
dubbed before run records existed.  Models are loaded once per session by
:class:`Resources` and kept between single-chunk actions.

The per-chunk operations are generators that yield one :class:`Progress`
per chunk.  They keep going when a chunk fails (the failure is reported in
its ``Progress``), clear only the stale flags they fix, and hold the shared
GPU lock (:mod:`mazinger.gpu`) while they run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from mazinger.editor.session import (
    STALE_CHECK, STALE_DUB, STALE_KINDS, STALE_TRANSLATION, Session, text_units,
)
from mazinger.gpu import GPUBusy, gpu_lock, register_releaser
from mazinger.runinfo import resolve_project_path

log = logging.getLogger(__name__)

# Chunks of source text sent as context on each side of a re-translation.
CONTEXT_CHUNKS = 2
# Texts per batched TTS call, for engines that support batching.
REDUB_BATCH_SIZE = 4
# Longest on-screen subtitle line built from a chunk.
DISPLAY_MAX_CHARS = 42

ASSEMBLY_DEFAULTS = dict(
    tempo_mode="auto", fixed_tempo=None, max_tempo=1.5,
    loudness_match=True, mix_background=True, background_volume=0.15,
)


@dataclass
class Progress:
    """One step of an operation.

    ``changed`` lists the rows to refresh.  ``error`` is set when this step
    failed; the operation carries on with the next chunk.
    """

    op: str
    done: int
    total: int
    chunk_id: str | None = None
    error: str | None = None
    message: str = ""
    changed: list[str] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def finished(self) -> bool:
        return self.done >= self.total


class SettingsMissing(RuntimeError):
    """The project has no run record and no settings were supplied."""


class VoiceUnavailable(RuntimeError):
    """No voice reference is available to re-dub in the original voice."""


def _section(settings: dict | None, name: str) -> dict:
    value = (settings or {}).get(name)
    return value if isinstance(value, dict) else {}


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  Resources (4.1)
# ═══════════════════════════════════════════════════════════════════════════════

class Resources:
    """Models, clients and context for one session, loaded on first use.

    Parameters:
        session:  The Editor session.
        settings: Run settings; defaults to ``session.run_info``.
        device:   ``cuda`` / ``cpu``; detected when ``None``.
        api_key:  LLM API key; falls back to ``OPENAI_API_KEY``.  Never saved.
        llm_client, voice_prompt: Pre-built objects (tests, or callers that
                  already hold them).
    """

    def __init__(
        self,
        session: Session,
        *,
        settings: dict | None = None,
        device: str | None = None,
        api_key: str | None = None,
        llm_client: Any = None,
        voice_prompt: Any = None,
    ) -> None:
        self.session = session
        self.proj = session.proj
        self.settings = settings if settings is not None else session.run_info
        self.device = device
        self.api_key = api_key
        self._llm = llm_client
        self._voice = voice_prompt
        self._context: dict | None = None
        # Messages for the user, e.g. that a voice reference was created.
        self.notices: list[str] = []

    @property
    def key(self) -> tuple:
        return (os.path.abspath(self.proj.root), self.session.language)

    def require_settings(self) -> dict:
        if not self.settings:
            raise SettingsMissing(
                "This project has no run.json (it was dubbed with an older Mazinger). "
                "Enter the settings it was dubbed with first."
            )
        return self.settings

    def get_device(self) -> str:
        if self.device is None:
            self.device = _default_device()
        return self.device

    # -- languages ---------------------------------------------------------

    @property
    def target_language(self) -> str:
        return (self.settings or {}).get("target_language") or self.session.language

    @property
    def tts_language(self) -> str:
        return _section(self.settings, "tts").get("language") or self.target_language

    @property
    def source_language(self) -> str:
        return (self.settings or {}).get("source_language") or "auto"

    def source_language_code(self) -> str | None:
        """ISO code for the ASR backend, or ``None`` to detect."""
        from mazinger.translate import lang_code_from_name, lang_name_from_code
        detected = (self.settings or {}).get("detected_source_language")
        if detected:
            return detected
        name = self.source_language
        if name and name != "auto":
            code = lang_code_from_name(name)
            if lang_name_from_code(code) == name:
                return code
        return None

    def source_language_name(self) -> str:
        """Canonical source language name (for template translators)."""
        from mazinger.translate import lang_name_from_code
        if self.source_language != "auto":
            return self.source_language
        return lang_name_from_code(self.source_language_code()) or "English"

    # -- LLM ---------------------------------------------------------------

    @property
    def llm_model(self) -> str:
        return (_section(self.settings, "llm").get("model")
                or os.environ.get("OPENAI_MODEL") or "gpt-4.1")

    def llm(self) -> Any:
        if self._llm is None:
            from mazinger.llm import build_client
            llm = _section(self.settings, "llm")
            self._llm = build_client(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=llm.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
                think=llm.get("think"),
            )
        return self._llm

    def context(self) -> dict:
        """Content description, video metadata and thumbnails of the project."""
        if self._context is None:
            from mazinger.utils import is_valid_json_file, load_json
            ctx: dict[str, Any] = {"description": {}, "video_meta": None, "thumb_paths": []}
            if is_valid_json_file(self.proj.description):
                ctx["description"] = load_json(self.proj.description)
            if os.path.isfile(self.proj.video_meta):
                ctx["video_meta"] = load_json(self.proj.video_meta)
            if os.path.isfile(self.proj.thumbs_meta):
                ctx["thumb_paths"] = self._existing_thumbs(load_json(self.proj.thumbs_meta))
            self._context = ctx
        return self._context

    def _existing_thumbs(self, thumbs: Any) -> list[dict]:
        # Paths are recorded as the dub saw them (often relative to its cwd).
        out = []
        for tp in thumbs if isinstance(thumbs, list) else []:
            path = tp.get("path", "")
            if not os.path.isfile(path):
                path = os.path.join(self.proj.thumbnails_dir, os.path.basename(path))
            if os.path.isfile(path):
                out.append({**tp, "path": path})
        return out

    # -- ASR ---------------------------------------------------------------

    def asr_settings(self) -> dict:
        """Keyword arguments for :func:`mazinger.transcribe.transcribe_clip`."""
        from mazinger.transcribe import build_initial_prompt
        t = _section(self.require_settings(), "transcription")
        method = t.get("method") or "faster-whisper"
        kw: dict[str, Any] = dict(
            method=method,
            model=t.get("model"),
            device=self.get_device(),
            language=self.source_language_code(),
            vad_method=t.get("vad_method") or "pyannote",
            initial_prompt=build_initial_prompt(self.context()["video_meta"]),
        )
        if method == "mlx-whisper":
            kw["beam_size"] = None
            if t.get("mlx_whisper_model"):
                kw["mlx_whisper_model"] = t["mlx_whisper_model"]
        else:
            kw["beam_size"] = t.get("beam_size") or 5
        if method == "openai":
            kw["openai_api_key"] = self.api_key or os.environ.get("OPENAI_API_KEY")
            kw["openai_base_url"] = _section(self.settings, "llm").get("base_url")
        return kw

    # -- TTS ---------------------------------------------------------------

    def resolve_voice(self) -> tuple[str, str | None]:
        """``(reference_audio, reference_text)`` that reproduce the dub's voice.

        OmniVoice voice-design and auto-voice runs have no reference clip;
        one of the dub's own segments is promoted to a reference instead
        (see :func:`mazinger.profiles.select_reference_segment`), and a
        notice is added for the user.

        Raises:
            VoiceUnavailable: when no reference can be found or made.
        """
        voice = _section(self.require_settings(), "voice")
        kind = voice.get("kind")
        sample = resolve_project_path(self.proj, voice.get("sample"))
        script = resolve_project_path(self.proj, voice.get("script"))

        if kind in _PROMOTED_VOICE_DIRS:
            ref_dir = os.path.join(self.proj.voice_profile_dir, _PROMOTED_VOICE_DIRS[kind])
            wav = os.path.join(ref_dir, "voice.wav")
            if kind != "omnivoice-auto" or not (sample and os.path.isfile(sample)):
                if not os.path.isfile(wav):
                    self._promote_reference(ref_dir)
                sample, script = wav, os.path.join(ref_dir, "script.txt")
        elif not (sample and os.path.isfile(sample)):
            for d in (self.proj.voice_reference_dir, self.proj.voice_profile_dir):
                found = _find_voice(d)
                if found:
                    sample, script = found, os.path.join(d, "script.txt")
                    break

        if not (sample and os.path.isfile(sample)):
            raise VoiceUnavailable(
                "No voice reference is left in this project, so a re-dub would not match "
                "the original voice."
            )
        ref_text = None
        if script and os.path.isfile(script):
            with open(script, encoding="utf-8") as fh:
                ref_text = fh.read().strip() or None
        return sample, ref_text

    def _promote_reference(self, ref_dir: str) -> None:
        from mazinger.profiles import select_reference_segment
        segs, texts = [], []
        for c in self.session.chunks:
            path = self.session.abs_path(c.dub_wav)
            if path and c.has_text:
                segs.append({"idx": c.id, "wav_path": path, "actual_dur": c.dub_dur})
                texts.append({"idx": c.id, "text": c.target_text})
        if not select_reference_segment(segs, texts, ref_dir):
            raise VoiceUnavailable("No dubbed segment is usable as a voice reference.")
        self.notices.append(
            "This dub had no voice reference, so one of its segments was kept as "
            f"the reference for re-dubs ({os.path.relpath(ref_dir, self.proj.root)})."
        )

    def voice(self) -> Any:
        """The loaded TTS voice prompt (a :class:`mazinger.tts.TTSWrapper`)."""
        if self._voice is None:
            from mazinger import tts
            t = _section(self.require_settings(), "tts")
            engine = t.get("engine") or "qwen"
            model_name = t.get("model")
            dtype = t.get("dtype") or ("float16" if engine == "omnivoice" else "bfloat16")
            device = self.get_device()
            tts_device = device if ":" in device or device == "cpu" else f"{device}:0"
            ref_audio, ref_text = self.resolve_voice()

            kw: dict[str, Any] = dict(device=tts_device, dtype=dtype, engine=engine)
            if model_name:
                kw[{"chatterbox": "chatterbox_model", "mlx": "mlx_model",
                    "omnivoice": "omnivoice_model"}.get(engine, "model_name")] = model_name
            model = tts.load_model(**kw)
            self._voice = tts.create_voice_prompt(
                model, ref_audio, ref_text, engine=engine,
                chatterbox_exaggeration=t.get("chatterbox_exaggeration", 0.5),
                chatterbox_cfg=t.get("chatterbox_cfg", 0.5),
                **({"mlx_model": model_name} if engine == "mlx" and model_name else {}),
            )
        return self._voice

    def set_api_key(self, api_key: str | None) -> None:
        """Use *api_key* for the LLM from now on (models stay loaded)."""
        api_key = api_key or None
        if api_key != self.api_key:
            self.api_key = api_key
            self._llm = None

    # -- lifetime ----------------------------------------------------------

    def free(self) -> None:
        """Unload every model this session loaded."""
        if self._voice is not None:
            from mazinger import tts
            try:
                tts.unload_model(self._voice, force=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not unload the TTS model: %s", exc)
            self._voice = None
        try:
            from mazinger import transcribe
            transcribe.clear_cache()
        except Exception as exc:  # noqa: BLE001 — torch may be absent
            log.debug("ASR cache not cleared: %s", exc)
        if self._llm is not None and hasattr(self._llm, "unload_model"):
            try:
                self._llm.unload_model(self.llm_model)
            except Exception:  # noqa: BLE001
                pass
        self._llm = None


# Voice kinds without a reference clip of their own: a dubbed segment is
# promoted to one, into voice_profile/<dir>/.
#   omnivoice-auto   OmniVoice picked a voice itself
#   theme-instruct   OmniVoice voice design from an instruct string
#   dub-segment      chosen in the Editor for projects without a run record
_PROMOTED_VOICE_DIRS = {
    "omnivoice-auto": "omnivoice_auto",
    "theme-instruct": "omnivoice_design",
    "dub-segment": "dub_segment",
}


def _find_voice(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        if name.startswith("voice.") and not name.endswith(".part"):
            return os.path.join(directory, name)
    return None


class ResourceManager:
    """Keeps the :class:`Resources` of the open session; one at a time.

    Switching to another project or language frees the previous one's
    models.  Registered with :func:`mazinger.gpu.release_idle`, so a full dub
    in the Dub tab frees them too.
    """

    def __init__(self) -> None:
        self._res: Resources | None = None
        register_releaser(self.free)

    def get(self, session: Session, **kwargs: Any) -> Resources:
        key = (os.path.abspath(session.proj.root), session.language)
        if self._res is not None and self._res.key == key and self._res.session is session and not kwargs:
            return self._res
        self.free()
        self._res = Resources(session, **kwargs)
        return self._res

    def free(self) -> None:
        """Free GPU memory ("Free GPU" button, project switch, full dub)."""
        if self._res is not None:
            self._res.free()
            self._res = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-chunk operations (4.3–4.5)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_each(
    op: str, ids: list[str], step: Callable[[str], list[str]],
) -> Iterator[Progress]:
    total = len(ids)
    if not total:
        yield Progress(op, 0, 0, message="Nothing to do")
        return
    failed = 0
    for n, cid in enumerate(ids, 1):
        try:
            changed = step(cid)
        except (GPUBusy, SettingsMissing, VoiceUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 — report and carry on
            failed += 1
            log.warning("%s failed for chunk %s: %s", op, cid, exc)
            yield Progress(op, n, total, cid, error=str(exc) or type(exc).__name__)
            continue
        yield Progress(op, n, total, cid, changed=changed,
                       message=_summary(op, n, total, failed) if n == total else "")


def _summary(op: str, n: int, total: int, failed: int) -> str:
    ok = total - failed
    return f"{op}: {ok}/{total} done" + (f", {failed} failed" if failed else "")


def retranscribe(session: Session, ids: list[str], res: Resources) -> Iterator[Progress]:
    """Re-transcribe the audio under each chunk (clears the check flag)."""
    from mazinger.transcribe import transcribe_clip

    def step(cid: str) -> list[str]:
        c = session.chunk(cid)
        span = (c.start, c.end)
        text = transcribe_clip(session.proj.audio, c.start, c.end, **res.asr_settings())
        if not text.strip():
            raise ValueError("No speech recognised; the transcription was kept")
        if (session.chunk(cid).start, session.chunk(cid).end) != span:
            raise RuntimeError("The chunk's timing changed while it was transcribed")
        return session.apply_transcription(cid, text)

    with gpu_lock.hold("an Editor re-transcription"):
        res.require_settings()
        yield from _run_each("Re-transcribe", list(ids), step)


def retranslate(session: Session, ids: list[str], res: Resources) -> Iterator[Progress]:
    """Re-translate each chunk from its source text (clears the translation flag)."""
    from mazinger import translate

    settings = res.require_settings()
    tr = _section(settings, "translation")
    ctx = res.context()

    def step(cid: str) -> list[str]:
        i = session.index(cid)
        c = session.chunks[i]
        source = c.source_text
        if not source.strip():
            raise ValueError("The chunk has no transcription to translate")
        if tr.get("translation_model"):
            text = translate.translate_text_simple(
                source, res.llm(),
                llm_model=tr["translation_model"],
                source_language=translate.resolve_language(res.source_language_name()),
                target_language=translate.resolve_language(res.target_language),
            )
            if not text:
                raise ValueError("The translation model returned nothing")
        else:
            chunks = session.chunks
            text = translate.translate_chunk(
                source,
                prev_ctx=[p.source_text for p in chunks[max(0, i - CONTEXT_CHUNKS):i]],
                next_ctx=[n.source_text for n in chunks[i + 1:i + 1 + CONTEXT_CHUNKS]],
                duration=c.duration,
                description=ctx["description"],
                client=res.llm(),
                llm_model=res.llm_model,
                source_language=res.source_language,
                target_language=res.target_language,
                words_per_second=tr.get("words_per_second"),
                duration_budget=tr.get("duration_budget") or translate.DURATION_BUDGET,
                translate_technical_terms=bool(tr.get("translate_technical_terms")),
                user_instructions=tr.get("user_instructions") or "",
                video_meta=ctx["video_meta"],
                thumb_paths=ctx["thumb_paths"],
                start=c.start,
            )
        if session.chunk(cid).source_text != source:
            raise RuntimeError("The transcription changed while it was translated")
        return session.apply_translation(cid, text)

    with gpu_lock.hold("an Editor re-translation"):
        yield from _run_each("Re-translate", list(ids), step)


def redub(
    session: Session, ids: list[str], res: Resources, *, batch_size: int = REDUB_BATCH_SIZE,
) -> Iterator[Progress]:
    """Re-synthesise each chunk's translation in the dub's voice (clears the dub flag).

    New audio goes to ``editor/segments/<id>_v<n>.wav``; earlier versions
    are kept so undo can return to them.  Engines that implement batch
    synthesis get up to *batch_size* chunks per call.
    """
    from mazinger import tts

    ids = list(ids)
    total = len(ids)
    with gpu_lock.hold("an Editor re-dub"):
        voice = res.voice()
        language = res.tts_language
        batched = type(voice).synthesize_batch is not tts.TTSWrapper.synthesize_batch
        if not total:
            yield Progress("Re-dub", 0, 0, message="Nothing to do")
            return

        done = failed = 0
        step = max(1, batch_size) if batched else 1
        for g in range(0, total, step):
            group = ids[g:g + step]
            texts: dict[str, str] = {}
            errors: dict[str, str] = {}
            for cid in group:
                try:
                    text = session.chunk(cid).target_text
                except KeyError as exc:
                    errors[cid] = str(exc)
                    continue
                if not text.strip():
                    errors[cid] = "The chunk has no translation to dub"
                else:
                    texts[cid] = text

            audio: dict[str, tuple] = {}
            if batched and len(texts) > 1:
                try:
                    results = voice.synthesize_batch([(t, language) for t in texts.values()])
                    audio = dict(zip(texts, results))
                except Exception as exc:  # noqa: BLE001 — retry one by one
                    log.warning("Batch re-dub failed (%s); retrying chunk by chunk", exc)

            for cid in group:
                done += 1
                try:
                    if cid in errors:
                        raise ValueError(errors[cid])
                    path = session.new_dub_path(cid)
                    if cid in audio:
                        data, sr = audio[cid]
                        dur = tts.write_segment(path, data, sr)
                    else:
                        _, dur = tts.synthesize_one(voice, texts[cid], path, language)
                    if session.chunk(cid).target_text != texts[cid]:
                        raise RuntimeError("The translation changed while it was dubbed")
                    changed = session.apply_dub(cid, path, dur)
                except Exception as exc:  # noqa: BLE001 — report and carry on
                    failed += 1
                    log.warning("Re-dub failed for chunk %s: %s", cid, exc)
                    yield Progress("Re-dub", done, total, cid, error=str(exc) or type(exc).__name__)
                    continue
                yield Progress("Re-dub", done, total, cid, changed=changed,
                               message=_summary("Re-dub", done, total, failed) if done == total else "")


_REDO = {
    STALE_CHECK: retranscribe,
    STALE_TRANSLATION: retranslate,
    STALE_DUB: redub,
}


def stale_ids(session: Session, kind: str) -> list[str]:
    """Chunks :func:`redo_stale` would process — show ``len()`` in the confirm step."""
    if kind not in STALE_KINDS:
        raise ValueError(f"Unknown stale kind {kind!r}")
    return session.ids(kind)


def redo_stale(
    session: Session, kind: str, res: Resources, *, expected: int | None = None,
) -> Iterator[Progress]:
    """Redo every chunk stale for *kind*.

    Pass the count the user confirmed as *expected*; if the set changed
    since, nothing runs and ``ValueError`` is raised so the UI can ask again.
    """
    ids = stale_ids(session, kind)
    if expected is not None and len(ids) != expected:
        raise ValueError(f"{len(ids)} chunks are now stale, not {expected}; please confirm again")
    return _REDO[kind](session, ids, res)


# ═══════════════════════════════════════════════════════════════════════════════
#  Assemble (4.7)
# ═══════════════════════════════════════════════════════════════════════════════

def display_entries(
    entries: list[tuple[float, float, str]], max_chars: int = DISPLAY_MAX_CHARS,
) -> list[tuple[float, float, str]]:
    """Split long subtitle entries into lines of at most *max_chars*.

    Lines break at sentence, then clause, then word boundaries (characters
    for scripts without spaces).  Each line gets a share of the entry's time
    proportional to its word count.
    """
    from mazinger.subtitle import _split_text_for_display

    out: list[tuple[float, float, str]] = []
    for start, end, text in entries:
        text = " ".join(text.split())
        if not text:
            continue
        lines: list[str] = []
        for piece in _split_text_for_display(text, max_chars):
            if len(piece) <= max_chars:
                lines.append(piece)
            else:  # an unspaced script: cut by characters
                units = text_units(piece)
                lines += ["".join(units[k:k + max_chars]) for k in range(0, len(units), max_chars)]
        weights = [max(1, len(text_units(line))) for line in lines]
        total = sum(weights)
        cursor = start
        for n, (line, w) in enumerate(zip(lines, weights)):
            line_end = end if n == len(lines) - 1 else cursor + (end - start) * w / total
            out.append((round(cursor, 3), round(line_end, 3), line))
            cursor = line_end
    return out


def _write_srt(path: str, entries: list[tuple[float, float, str]]) -> str:
    from mazinger.srt import blocks_to_text
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(blocks_to_text([(str(i), s, e, t) for i, (s, e, t) in enumerate(entries, 1)]))
    os.replace(tmp, path)
    return path


def prev_path(path: str) -> str:
    """``dubbed.wav`` → ``dubbed.prev.wav``: where the previous output is kept."""
    base, ext = os.path.splitext(path)
    return f"{base}.prev{ext}"


def _install(tmp: str, final: str) -> None:
    if os.path.exists(final):
        os.replace(final, prev_path(final))
    os.replace(tmp, final)


def segment_info(session: Session) -> tuple[list[dict], dict]:
    """The session as ``assemble_timeline`` input, plus counts for the log.

    Chunks without text or without a dub file are left silent.  Stale dubs
    are used as they are — assembling never redoes anything.
    """
    segs, counts = [], {"used": 0, "stale": 0, "silent": 0}
    for c in session.chunks:
        path = session.abs_path(c.dub_wav)
        if not c.has_text or not path or not os.path.isfile(path):
            counts["silent"] += c.has_text
            continue
        segs.append({
            "idx": c.id, "start": c.start, "end": c.end, "target_dur": c.duration,
            "wav_path": path, "actual_dur": c.dub_dur,
        })
        counts["used"] += 1
        counts["stale"] += STALE_DUB in c.stale
    return segs, counts


def assemble(session: Session, *, settings: dict | None = None) -> Iterator[Progress]:
    """Rebuild the final audio (and video) from the session.

    Outputs are built beside the old ones and swapped in only when every
    step succeeded; the old files are kept as ``*.prev.*``.  Writes the
    session's ``subtitles/translated.srt`` (old one kept as
    ``translated.prev.srt``) and ``editor/source.edited.srt`` — the shared
    ``transcription/source.srt`` is never modified.
    """
    from mazinger import assemble as asm
    from mazinger.utils import get_audio_duration

    proj = session.proj
    settings = settings if settings is not None else (session.run_info or {})
    opts = {**ASSEMBLY_DEFAULTS, **{k: v for k, v in _section(settings, "assembly").items()
                                    if k in ASSEMBLY_DEFAULTS and v is not None}}
    output = _section(settings, "output")
    style = output.get("subtitle_style")
    sub_source = output.get("subtitle_source") or "translated"
    want_video = (output.get("output_type") == "video" or style is not None
                  or os.path.isfile(proj.final_video))
    total = 5
    op = "Assemble"

    with gpu_lock.hold("an Editor assembly"):
        segs, counts = segment_info(session)
        note = f"{counts['used']} segments"
        if counts["stale"]:
            note += f", {counts['stale']} with an out-of-date dub"
        if counts["silent"]:
            note += f", {counts['silent']} without a dub (silent)"
        yield Progress(op, 0, total, message=f"Building the timeline: {note}")

        duration = session.duration or get_audio_duration(proj.audio)
        tmp_wav = os.path.join(proj.tts_dir, "dubbed.editor.tmp.wav")
        tmp_mp4 = os.path.join(proj.tts_dir, "dubbed.editor.tmp.mp4")
        tmp_srt = proj.final_srt + ".editor.tmp"
        try:
            asm.assemble_timeline(
                segs, duration, tmp_wav,
                tempo_mode=opts["tempo_mode"], fixed_tempo=opts["fixed_tempo"],
                max_tempo=opts["max_tempo"],
            )
            yield Progress(op, 1, total, message="Mixing: loudness and background")

            if opts["loudness_match"] or opts["mix_background"]:
                asm.post_process(
                    tmp_wav, proj.audio, tmp_wav,
                    loudness_match=opts["loudness_match"],
                    mix_background=opts["mix_background"],
                    background_volume=opts["background_volume"],
                    background_cache=proj.background_audio(asm.TARGET_SR),
                    loudness_cache=proj.source_loudness,
                )
            yield Progress(op, 2, total, message="Writing subtitles")

            targets = [(c.start, c.end, c.target_text) for c in session.chunks if c.has_text]
            sources = [(c.start, c.end, c.source_text) for c in session.chunks if c.source_text.strip()]
            _write_srt(tmp_srt, targets)
            edited_source = _write_srt(os.path.join(proj.editor_dir, "source.edited.srt"), sources)
            if sub_source == "translated":
                display_srt = _write_srt(
                    os.path.join(proj.editor_dir, "subtitles.display.srt"), display_entries(targets))
            elif sub_source == "original":
                display_srt = _write_srt(
                    os.path.join(proj.editor_dir, "source.display.srt"), display_entries(sources))
            else:
                display_srt = resolve_project_path(proj, sub_source)

            video_out = None
            if want_video and os.path.isfile(proj.video):
                yield Progress(op, 3, total, message="Rendering the video")
                if style is not None:
                    from mazinger.subtitle import SubtitleStyle, burn_subtitles
                    video_out = burn_subtitles(
                        proj.video, tmp_mp4, display_srt, SubtitleStyle(**style), audio_path=tmp_wav,
                    )
                else:
                    video_out = asm.mux_video(proj.video, tmp_wav, tmp_mp4)
            elif want_video:
                log.warning("No source video in the project — skipping video output")

            _install(tmp_wav, proj.final_audio)
            _install(tmp_srt, proj.final_srt)
            outputs = {"audio": proj.final_audio, "srt": proj.final_srt,
                       "source_srt": edited_source, "display_srt": display_srt}
            if video_out:
                _install(tmp_mp4, proj.final_video)
                outputs["video"] = proj.final_video
            for key, path in (("previous_audio", proj.final_audio), ("previous_video", proj.final_video)):
                if os.path.isfile(prev_path(path)):
                    outputs[key] = prev_path(path)
        finally:
            for tmp in (tmp_wav, tmp_mp4, tmp_srt):
                if os.path.exists(tmp):
                    os.remove(tmp)

        session.mark_output_current()
        yield Progress(op, total, total, message="Output rebuilt", outputs=outputs)


def previous_outputs(session: Session) -> dict:
    """Paths of the previous outputs kept by the last assembly (may be empty)."""
    out = {}
    for key, path in (("audio", session.proj.final_audio), ("video", session.proj.final_video),
                      ("srt", session.proj.final_srt)):
        if os.path.isfile(prev_path(path)):
            out[key] = prev_path(path)
    return out


__all__ = [
    "Progress", "Resources", "ResourceManager", "SettingsMissing", "VoiceUnavailable",
    "retranscribe", "retranslate", "redub", "stale_ids", "redo_stale",
    "assemble", "display_entries", "segment_info", "prev_path", "previous_outputs",
]