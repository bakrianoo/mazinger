"""Re-segment translated SRT into readable, properly-timed caption blocks."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import json_repair

from mazinger_dubber.srt import parse_blocks, build

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

MAX_CHARS = 84
MAX_DUR = 4.0
MIN_DUR = 1.0

_SPLIT_SYSTEM = """\
You are a professional subtitle editor. Split the given long subtitle text \
into shorter, readable caption segments.

RULES:
1. Split at natural sentence or clause boundaries (periods, commas, semicolons, \
   conjunctions like "and", "but", "so", "because", "which").
2. Each segment should be 40-84 characters long.
3. Never break in the middle of a word or a technical term.
4. Preserve the EXACT original text -- do not rephrase, rewrite, add, or remove \
   any words. Only add split points.
5. Return a JSON array of strings.
6. Return ONLY the JSON array -- no markdown fences, no commentary."""


def _llm_split(text: str, client: OpenAI, llm_model: str = "gpt-4.1") -> list[str]:
    """Use an LLM to split long text into caption-sized pieces."""
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _SPLIT_SYSTEM},
            {"role": "user", "content": f"Split this subtitle text:\n\n{text}"},
        ],
    )
    segments = json_repair.loads(resp.choices[0].message.content)
    return [s.strip() for s in segments if s.strip()]


def _rule_based_split(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Deterministic fallback: split on sentence and clause boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            segments.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        segments.append(current.strip())

    final: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            parts = re.split(r"(?<=,)\s+|(?<=\u2014)\s*|\s+(?=\u2014)", seg)
            buf = ""
            for p in parts:
                if buf and len(buf) + len(p) + 1 > max_chars:
                    final.append(buf.strip())
                    buf = p
                else:
                    buf = (buf + " " + p).strip() if buf else p
            if buf:
                final.append(buf.strip())
    return final


def _distribute_timestamps(
    segments: list[str],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """Assign timestamps proportionally by word count."""
    total_dur = end - start
    word_counts = [len(s.split()) for s in segments]
    total_words = sum(word_counts)

    if total_words == 0:
        per_seg = total_dur / len(segments)
        return [(start + i * per_seg, start + (i + 1) * per_seg) for i in range(len(segments))]

    result: list[tuple[float, float]] = []
    t = start
    for i, seg in enumerate(segments):
        proportion = word_counts[i] / total_words
        seg_dur = total_dur * proportion
        if i < len(segments) - 1:
            seg_dur = max(MIN_DUR, min(MAX_DUR, seg_dur))
        seg_end = min(t + seg_dur, end)
        result.append((t, seg_end))
        t = seg_end

    if result:
        s, _ = result[-1]
        result[-1] = (s, end)
    return result


def resegment_srt(
    srt_text: str,
    *,
    client: OpenAI | None = None,
    llm_model: str = "gpt-4.1",
    max_chars: int = MAX_CHARS,
    max_dur: float = MAX_DUR,
) -> str:
    """Re-segment an SRT string into properly sized caption blocks.

    When *client* is provided, long entries are split using an LLM for
    natural-sounding boundaries.  Otherwise a deterministic rule-based
    splitter is used.

    Returns:
        The re-segmented SRT as a string.
    """
    blocks = parse_blocks(srt_text)
    resegmented: list[tuple[float, float, str]] = []
    llm_calls = 0

    for _, start, end, text in blocks:
        text = text.strip()
        dur = end - start

        if len(text) <= max_chars and dur <= max_dur:
            resegmented.append((start, end, text))
            continue

        if len(text) <= max_chars:
            resegmented.append((start, end, text))
            continue

        # Try LLM split, fall back to rules
        segments: list[str] | None = None
        if client is not None:
            try:
                segments = _llm_split(text, client, llm_model)
                llm_calls += 1
                joined = " ".join(segments)
                if joined.replace("  ", " ").strip() != text.replace("  ", " ").strip():
                    segments = None
            except Exception:
                segments = None

        if segments is None:
            segments = _rule_based_split(text, max_chars)

        time_ranges = _distribute_timestamps(segments, start, end)
        for (s, e), seg_text in zip(time_ranges, segments):
            resegmented.append((s, e, seg_text))

    log.info(
        "Re-segmented %d -> %d entries (LLM calls: %d)",
        len(blocks), len(resegmented), llm_calls,
    )
    return build(resegmented)
