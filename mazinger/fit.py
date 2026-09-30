"""Fit check: shorten dubbed lines whose speech does not fit their time slot.

After TTS, each segment's audio is compared with the time it may occupy on
the timeline (the same window :func:`mazinger.assemble.assemble_timeline`
uses).  Lines that would need more than a mild speed-up are rewritten
shorter by the LLM — tighter phrasing, same content — and re-synthesised.
A rewrite is kept only when its audio really is shorter, so the check can
never make a line worse.  Assembly then handles the small overruns that
remain with a gentle tempo change instead of a hard speed-up or a cut.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import json_repair

from mazinger.srt import blocks_to_text

log = logging.getLogger(__name__)

DEFAULT_MAX_FIT = 1.15   # speed-up above which a line is rewritten
DEFAULT_ROUNDS = 2       # rewrite / re-synthesise rounds at most
_AIM = 1.10              # shorten only to what a mild speed-up absorbs, so
                         # as little wording as possible is lost
_MAX_CUT = 0.3           # never shorten a line by more than 30% per round
_BATCH = 20              # lines per LLM request
_SEGMENT_GAP = 0.05      # matches assemble_timeline's segment_gap_ms

_SHORTEN_SYSTEM = """\
You are a dubbing script editor. Each line below is a {language} dubbing \
line that runs too long for its time slot. Rewrite each line in {language} \
so it has at most "max_{unit}" {unit}.

RULES:
- Keep the FULL meaning: every point, name, number, example and the \
  speaker's point of view. Do not summarise or drop content.
- Shorten by tighter phrasing: remove filler words and redundant \
  wording, prefer shorter synonyms and simpler constructions.
- Keep the same language, tone and tense. Keep technical terms as they are.
- Return ONLY a JSON array of {{"index": ..., "text": ...}} objects, one \
per input line, no commentary."""


def slot_seconds(segments: list[dict], i: int, original_duration: float) -> float:
    """Time segment *i* may occupy on the timeline, as assembly computes it."""
    seg = segments[i]
    if i + 1 < len(segments):
        window = segments[i + 1]["start"] - seg["start"] - _SEGMENT_GAP
    else:
        window = original_duration - seg["start"]
    return max(window, seg["target_dur"], 1e-3)


def fit_ratios(
    segments: list[dict], original_duration: float, *, sync: bool = False,
) -> list[float | None]:
    """Speech length over slot length per segment; ``None`` without audio.

    Like assembly, only segments with audio take part, in start order.
    With *sync* (the ``sync`` tempo mode) the slot is the original speech
    the line replaces (``target_dur``) and the clip's own edge silence
    does not count (``speech_dur``).
    """
    if sync:
        return [
            (s.get("speech_dur") or s["actual_dur"]) / max(s["target_dur"], 1e-3)
            if s.get("wav_path") and s.get("actual_dur") else None
            for s in segments
        ]
    placed = sorted((s for s in segments if s.get("wav_path")), key=lambda s: s["start"])
    ratio = {
        id(s): s["actual_dur"] / slot_seconds(placed, i, original_duration)
        for i, s in enumerate(placed) if s.get("actual_dur")
    }
    return [ratio.get(id(s)) for s in segments]


def _shorten(
    items: list[dict], client: Any, llm_model: str, language: str, unit: str,
    usage_tracker: Any = None,
) -> dict[str, str]:
    """Ask the LLM for shorter versions of *items*; returns ``{idx: text}``."""
    from mazinger.translate import _clean_llm_text

    out: dict[str, str] = {}
    system = _SHORTEN_SYSTEM.format(language=language, unit=unit)
    for b in range(0, len(items), _BATCH):
        batch = items[b:b + _BATCH]
        payload = json.dumps(
            [{"index": it["idx"], "text": it["text"], f"max_{unit}": it["max"]} for it in batch],
            ensure_ascii=False, indent=2,
        )
        resp = client.chat.completions.create(
            model=llm_model, temperature=0.2, top_p=0.9, num_predict=4000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
        )
        if usage_tracker is not None:
            usage_tracker.record("fit", llm_model, resp)
        try:
            rows = json_repair.loads(resp.choices[0].message.content or "")
        except Exception:  # noqa: BLE001 — an unusable reply keeps the lines
            rows = []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and "index" in row and "text" in row:
                idx = str(row["index"]).strip()
                out[idx] = _clean_llm_text(str(row["text"]), idx)
    return out


def _synthesize_trials(
    trials: list[tuple], voice_prompt: Any, language: str, model: Any, sync: bool,
) -> list[tuple[float, float] | None]:
    """Synthesise each rewrite to its trial path; ``(duration, speech_dur)`` or ``None``.

    Engines that batch get one call for all trials.
    """
    from mazinger import tts

    targets = [item["seg"]["target_dur"] if sync else None for item, *_ in trials]
    out: list[tuple[float, float] | None] = [None] * len(trials)
    if (isinstance(voice_prompt, tts.TTSWrapper) and voice_prompt.batched and len(trials) > 1):
        try:
            for k in range(0, len(trials), tts.TTS_BATCH_SIZE):
                chunk = list(range(k, min(k + tts.TTS_BATCH_SIZE, len(trials))))
                results = voice_prompt.synthesize_batch_to(
                    [(trials[i][1], language, targets[i]) for i in chunk])
                for i, (audio, sr) in zip(chunk, results):
                    dur = tts.write_segment(trials[i][3], audio, sr)
                    out[i] = (dur, tts.speech_duration(trials[i][3]) if sync else dur)
            return out
        except Exception as exc:  # noqa: BLE001 — fall back to one by one
            log.warning("Fit check: batch re-synthesis failed (%s); retrying one by one", exc)
    for i, (item, new_text, _, trial) in enumerate(trials):
        if out[i] is not None:
            continue
        try:
            _, dur = tts.synthesize_one(voice_prompt, new_text, trial, language,
                                        model=model, target_dur=targets[i])
            out[i] = (dur, tts.speech_duration(trial) if sync else dur)
        except Exception as exc:  # noqa: BLE001 — keep the original line
            log.warning("Fit check: re-synthesis of line %s failed: %s", item["idx"], exc)
    return out


def fit_segments(
    segments: list[dict],
    srt_entries: list[dict],
    *,
    client: Any,
    llm_model: str,
    voice_prompt: Any,
    tts_language: str,
    target_language: str,
    original_duration: float,
    srt_path: str | None = None,
    model: Any = None,
    max_fit: float = DEFAULT_MAX_FIT,
    rounds: int = DEFAULT_ROUNDS,
    max_tempo: float = 1.5,
    usage_tracker: Any = None,
    sync: bool = False,
) -> dict:
    """Rewrite and re-synthesise segments that overflow their slot.

    With *sync*, a line overflows when its speech is longer than the
    original speech it replaces (see :func:`fit_ratios`), and re-synthesis
    aims at that length.

    *segments* and *srt_entries* are the outputs of
    :func:`mazinger.tts.synthesize_segments` and the SRT it was given; both
    are updated in place, and the SRT is rewritten at *srt_path* when a line
    changed so later stages and the Editor see the spoken text.

    Returns statistics: lines rewritten, time spent, and how many lines
    still exceed *max_fit* and *max_tempo* (the latter get trimmed).
    """
    from mazinger import tts
    from mazinger.translate import _CHAR_BUDGET_LANGUAGES, _count_units, resolve_language

    started = time.monotonic()
    lang = resolve_language(target_language)
    unit = "characters" if lang in _CHAR_BUDGET_LANGUAGES else "words"
    text_by_idx = {e["idx"]: e for e in srt_entries}
    stats = {"rewritten": 0, "rounds": 0}

    for round_no in range(1, rounds + 1):
        ratios = fit_ratios(segments, original_duration, sync=sync)
        flagged = []
        for seg, fit in zip(segments, ratios):
            entry = text_by_idx.get(seg["idx"])
            if fit is None or fit <= max_fit or not entry or not entry["text"].strip():
                continue
            units = _count_units(entry["text"], lang)
            if units < 3:
                continue
            target = max(int(units * _AIM / fit), int(units * (1 - _MAX_CUT)), 1)
            if target < units:
                flagged.append({"idx": seg["idx"], "text": entry["text"].strip(),
                                "max": target, "units": units, "seg": seg})
        if not flagged:
            break
        stats["rounds"] = round_no
        log.info("Fit check round %d: %d/%d lines need more than %.2fx — shortening",
                 round_no, len(flagged), len(segments), max_fit)

        rewrites = _shorten(flagged, client, llm_model, lang, unit, usage_tracker)
        if hasattr(client, "unload_model"):
            client.unload_model(llm_model)

        trials = []
        for item in flagged:
            new_text = rewrites.get(item["idx"], "")
            new_units = _count_units(new_text, lang)
            # Shorter, but not a summary: at most the per-round cut plus slack.
            if not new_text or new_units >= item["units"] \
                    or new_units < item["units"] * (1 - _MAX_CUT) - 2:
                continue
            trials.append((item, new_text, new_units, item["seg"]["wav_path"] + ".fit.wav"))

        for (item, new_text, new_units, trial), result in zip(
                trials, _synthesize_trials(trials, voice_prompt, tts_language, model, sync)):
            seg = item["seg"]
            if result is None:
                continue
            new_dur, new_speech = result
            before = (seg.get("speech_dur") or seg["actual_dur"]) if sync else seg["actual_dur"]
            after = new_speech if sync else new_dur
            if after >= before:
                os.remove(trial)
                continue
            os.replace(trial, seg["wav_path"])
            log.info("Fit check: line %s %.1fs -> %.1fs (%d -> %d %s)", item["idx"],
                     before, after, item["units"], new_units, unit)
            seg["actual_dur"] = new_dur
            if sync:
                seg["speech_dur"] = new_speech
            text_by_idx[item["idx"]]["text"] = new_text
            stats["rewritten"] += 1

    if stats["rewritten"] and srt_path:
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(blocks_to_text(
                [(e["idx"], e["start"], e["end"], e["text"]) for e in srt_entries]))

    ratios = [r for r in fit_ratios(segments, original_duration, sync=sync) if r is not None]
    stats.update(
        seconds=round(time.monotonic() - started, 1),
        over_max_fit=sum(r > max_fit for r in ratios),
        over_max_tempo=sum(r > max_tempo for r in ratios),
        max_ratio=round(max(ratios, default=0.0), 2),
    )
    log.info(
        "Fit check: %d lines rewritten in %.1fs; %d still above %.2fx, "
        "%d above max tempo %.2fx (will be trimmed); worst %.2fx",
        stats["rewritten"], stats["seconds"], stats["over_max_fit"], max_fit,
        stats["over_max_tempo"], max_tempo, stats["max_ratio"],
    )
    return stats
