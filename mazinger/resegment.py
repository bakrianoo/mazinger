"""Re-segment translated SRT into readable, properly-timed caption blocks.

Two-phase approach:
1. **Merge** – consecutive SRT entries that are fragments of the same sentence
   or speech phrase are combined so TTS never generates audio for an incomplete
   thought (avoids unnatural mid-sentence pauses).
2. **Split** – entries that are too long for comfortable subtitles are broken
   at natural clause/sentence boundaries.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import json_repair

from mazinger.srt import parse_blocks, build, sanitize
from mazinger.utils import LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

MAX_CHARS = 300
MAX_DUR = 30.0
MIN_DUR = 1.0
_MERGE_BATCH_SIZE = 40
_MAX_MERGE_GROUP = 8
# Maximum silence gap (seconds) between entries that may be merged.
# Gaps larger than this indicate a natural pause in the original audio
# that should be preserved to maintain sync.
_MERGE_MAX_GAP = 1.5

# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 – Merge fragments into complete phrases
# ═══════════════════════════════════════════════════════════════════════════════

_MERGE_SYSTEM = """\
You are a speech-aware subtitle editor preparing subtitles for \
text-to-speech synthesis. Your goal is to merge consecutive subtitle \
entries so that each resulting entry is a COMPLETE SPOKEN THOUGHT — \
a full phrase or group of closely related sentences that a speaker would \
naturally say without a long pause.

This is critical: TTS will generate audio for each entry independently. \
If an entry is an incomplete fragment (e.g. "But what" or \
"One of the biggest challenges people face when working in"), the \
generated speech will sound broken and unnatural.

MERGE AGGRESSIVELY:
- Merge any entries that are fragments of the same sentence.
- Merge follow-up sentences that continue the same idea or thought \
  (e.g. a rhetorical question immediately followed by its answer).
- Merge short transitional phrases ("But let's move on—") with the \
  sentence that follows.
- It is BETTER to merge too much than too little. A longer entry that \
  reads as one spoken paragraph is fine.

LIMITS:
- Maximum 8 consecutive entries per group.
- Do NOT merge entries that are clearly about different topics or \
  separated by a natural topic shift.
- Each merged result should be at most ~500 characters.

FORMAT:
- Return a JSON array of arrays. Each inner array lists the 1-based \
  entry numbers to merge.
- Every entry number must appear in exactly one group.
- Only consecutive entries may be grouped.
- Return ONLY the JSON array — no markdown fences, no commentary.

EXAMPLE INPUT:
1: "But what"
2: "makes this application special, and why is it important?"
3: "One of the biggest challenges people face when working in"
4: "data science is that their software engineering knowledge is often quite limited. Why is that?"
5: "While learning—whether it's data science, machine learning, or analytics—"
6: "most of the time,"
7: "you're guided to experiment with certain things."
8: "Notebooks are your testing ground."

EXAMPLE OUTPUT: [[1, 2], [3, 4], [5, 6, 7], [8]]"""

# Only real sentence-ending punctuation — NOT em-dashes, commas, colons, or
# trailing quotes/parens which often appear mid-thought.
_SENTENCE_END_RE = re.compile(r'[.!?…؟。！？]\s*$')


def _validate_merge_groups(groups: list, n: int) -> bool:
    """Check that *groups* cover entries 1..n exactly once, in order."""
    if not isinstance(groups, list):
        return False
    seen: set[int] = set()
    prev_max = 0
    for group in groups:
        if not isinstance(group, list) or not group:
            return False
        if len(group) > _MAX_MERGE_GROUP:
            return False
        nums: list[int] = []
        for idx in group:
            if not isinstance(idx, (int, float)):
                return False
            idx = int(idx)
            if idx < 1 or idx > n or idx in seen:
                return False
            seen.add(idx)
            nums.append(idx)
        nums_sorted = sorted(nums)
        if nums_sorted != list(range(nums_sorted[0], nums_sorted[-1] + 1)):
            return False
        if nums_sorted[0] <= prev_max:
            return False
        prev_max = nums_sorted[-1]
    return len(seen) == n


def _fix_merge_groups(groups: list, n: int) -> list[list[int]] | None:
    """Try to repair an LLM merge response that is close but invalid.

    Common issues:
    - Some entry numbers are missing (the LLM skipped them).
    - Groups are otherwise valid (consecutive, in order, no duplicates).

    Repair strategy: insert any missing entries as singleton groups in the
    correct position.  Returns *None* if the response is too broken to fix.
    """
    if not isinstance(groups, list):
        return None

    # Normalise: ensure every group is a list of ints, consecutive, in order.
    normalised: list[list[int]] = []
    seen: set[int] = set()
    prev_max = 0
    for group in groups:
        if not isinstance(group, list) or not group:
            return None
        nums: list[int] = []
        for idx in group:
            if not isinstance(idx, (int, float)):
                return None
            idx = int(idx)
            if idx < 1 or idx > n or idx in seen:
                return None
            seen.add(idx)
            nums.append(idx)
        nums_sorted = sorted(nums)
        if nums_sorted != list(range(nums_sorted[0], nums_sorted[-1] + 1)):
            return None
        if nums_sorted[0] <= prev_max:
            return None
        if len(nums_sorted) > _MAX_MERGE_GROUP:
            return None
        prev_max = nums_sorted[-1]
        normalised.append(nums_sorted)

    if not normalised:
        return None

    # All present already — nothing to fix.
    if len(seen) == n:
        return normalised

    # Fill in gaps: insert missing entries as singletons.
    result: list[list[int]] = []
    next_expected = 1
    for group in normalised:
        # Insert singletons for any entries before this group.
        while next_expected < group[0]:
            result.append([next_expected])
            next_expected += 1
        result.append(group)
        next_expected = group[-1] + 1
    # Trailing entries after the last group.
    while next_expected <= n:
        result.append([next_expected])
        next_expected += 1

    return result


def _llm_merge_batch(
    blocks: list[tuple[str, float, float, str]],
    client: OpenAI,
    llm_model: str,
    usage_tracker: LLMUsageTracker | None = None,
) -> list[list[int]] | None:
    """Ask the LLM which consecutive entries should be merged."""
    lines = []
    for i, (_, _s, _e, text) in enumerate(blocks, 1):
        lines.append(f'{i}: "{text.strip()}"')
    entries_text = "\n".join(lines)

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _MERGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Merge these subtitle entries as needed:\n\n"
                        + entries_text
                    ),
                },
            ],
            repeat_penalty=1.3,
            top_p=0.9,
            num_predict=8000,
            frequency_penalty=0.3,
        )
        if usage_tracker is not None:
            usage_tracker.record("resegment-merge", llm_model, resp)
        groups = json_repair.loads(resp.choices[0].message.content)
        if _validate_merge_groups(groups, len(blocks)):
            return groups
        # Try to repair (e.g. LLM skipped some entries)
        fixed = _fix_merge_groups(groups, len(blocks))
        if fixed is not None:
            log.info("LLM merge response repaired (missing entries filled as singletons)")
            return fixed
        log.warning(
            "LLM merge response failed validation – falling back. "
            "Expected %d entries, got: %s",
            len(blocks),
            resp.choices[0].message.content[:500],
        )
    except Exception:
        log.warning("LLM merge call failed – falling back", exc_info=True)
    return None


def _rule_based_merge(
    blocks: list[tuple[str, float, float, str]],
    max_gap: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Merge entries whose text doesn't end with sentence-ending punctuation.

    This is the deterministic fallback when no LLM client is available.
    """
    if not blocks:
        return []

    merged: list[tuple[float, float, str]] = []
    buf_start: float | None = None
    buf_end: float = 0.0
    buf_texts: list[str] = []

    for _, start, end, text in blocks:
        text = text.strip()

        # Large silence gap → flush current buffer first
        if buf_start is not None and (start - buf_end) > max_gap:
            merged.append((buf_start, buf_end, " ".join(buf_texts)))
            buf_start = start
            buf_end = end
            buf_texts = [text]
        else:
            if buf_start is None:
                buf_start = start
            buf_end = end
            buf_texts.append(text)

        # Sentence-ending punctuation → flush
        if _SENTENCE_END_RE.search(text):
            merged.append((buf_start, buf_end, " ".join(buf_texts)))
            buf_start = None
            buf_end = 0.0
            buf_texts = []

    if buf_texts and buf_start is not None:
        merged.append((buf_start, buf_end, " ".join(buf_texts)))

    return merged


def _merge_phrases(
    blocks: list[tuple[str, float, float, str]],
    client: OpenAI | None,
    llm_model: str,
    usage_tracker: LLMUsageTracker | None = None,
) -> list[tuple[float, float, str]]:
    """Merge consecutive SRT blocks that belong to the same speech phrase."""
    if not blocks:
        return []

    if client is None:
        return _rule_based_merge(blocks)

    merged: list[tuple[float, float, str]] = []
    merge_calls = 0

    for batch_start in range(0, len(blocks), _MERGE_BATCH_SIZE):
        batch = blocks[batch_start : batch_start + _MERGE_BATCH_SIZE]

        groups = _llm_merge_batch(batch, client, llm_model, usage_tracker=usage_tracker)
        merge_calls += 1

        if groups is None:
            merged.extend(_rule_based_merge(batch))
            continue

        # Post-process LLM groups: split any group whose entries span a
        # silence gap > _MERGE_MAX_GAP — the LLM doesn't see timestamps
        # and may merge across natural pauses in the original audio.
        final_groups: list[list[int]] = []
        for group in groups:
            indices = [g - 1 for g in group]
            sub: list[int] = [indices[0]]
            for k in range(1, len(indices)):
                prev_end = batch[indices[k - 1]][2]
                cur_start = batch[indices[k]][1]
                if cur_start - prev_end > _MERGE_MAX_GAP:
                    final_groups.append(sub)
                    sub = [indices[k]]
                else:
                    sub.append(indices[k])
            final_groups.append(sub)

        for indices in final_groups:
            texts = [batch[i][3].strip() for i in indices]
            start = batch[indices[0]][1]
            end = batch[indices[-1]][2]
            merged.append((start, end, " ".join(texts)))

    log.info(
        "Merge phase: %d -> %d entries (LLM merge calls: %d)",
        len(blocks), len(merged), merge_calls,
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 – Split long entries at natural boundaries
# ═══════════════════════════════════════════════════════════════════════════════

_SPLIT_SYSTEM = """\
You are a speech-aware subtitle editor. Split long text into segments \
for text-to-speech. Each segment MUST be a complete spoken thought.

RULES:
1. ONLY split at sentence-ending punctuation (. ! ?). \
   NEVER split at commas, conjunctions, dashes, or mid-sentence.
2. Each segment must sound natural spoken on its own — a complete idea.
3. Segments can be up to 300 characters. Do NOT split just for length. \
   Only split when there are genuinely separate sentences.
4. Preserve the EXACT original text — do not rephrase or remove words.
5. Return a JSON array of strings. No markdown fences, no commentary."""


def _llm_split(
    text: str,
    client: OpenAI,
    llm_model: str = "gpt-4.1",
    usage_tracker: LLMUsageTracker | None = None,
) -> list[str]:
    """Use an LLM to split long text into caption-sized pieces."""
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _SPLIT_SYSTEM},
            {"role": "user", "content": f"Split this subtitle text:\n\n{text}"},
        ],
        repeat_penalty=1.3,
        top_p=0.9,
        num_predict=2048,
        frequency_penalty=0.3,
    )
    if usage_tracker is not None:
        usage_tracker.record("resegment-split", llm_model, resp)
    segments = json_repair.loads(resp.choices[0].message.content)
    return [s.strip() for s in segments if s.strip()]


def _rule_based_split(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Deterministic fallback: split ONLY at sentence boundaries.

    Keeps full sentences together. Only falls back to clause-level splitting
    when a single sentence exceeds *max_chars*.
    """
    # Split at sentence-ending punctuation only
    sentences = re.split(r'(?<=[.!?…؟])\s+', text)
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

    # Only break further if a single sentence is still over the limit
    final: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            # Last resort: split at semicolons or em-dashes
            parts = re.split(r'(?<=[;])\s+|(?<=\u2014)\s*|\s+(?=\u2014)', seg)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Timestamp distribution
# ═══════════════════════════════════════════════════════════════════════════════

def _distribute_timestamps(
    segments: list[str],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """Assign timestamps proportionally by word count.

    Uses iterative redistribution to handle MIN_DUR/MAX_DUR clamping so that
    the sub-segments always sum exactly to the parent window.
    """
    total_dur = end - start
    n = len(segments)

    if n == 0:
        return []
    if n == 1:
        return [(start, end)]

    word_counts = [max(1, len(s.split())) for s in segments]
    total_words = sum(word_counts)

    if total_words == 0:
        per_seg = total_dur / n
        return [(start + i * per_seg, start + (i + 1) * per_seg) for i in range(n)]

    # Iterative distribution: compute proportional durations, clamp, and
    # redistribute excess/deficit from clamped segments to unclamped ones.
    durations = [total_dur * (wc / total_words) for wc in word_counts]
    locked = [False] * n

    for _ in range(5):  # converges in 2-3 iterations
        excess = 0.0
        unlocked_words = 0
        for i in range(n):
            if locked[i]:
                continue
            if durations[i] < MIN_DUR:
                excess += durations[i] - MIN_DUR  # negative = deficit
                durations[i] = MIN_DUR
                locked[i] = True
            elif durations[i] > MAX_DUR:
                excess += durations[i] - MAX_DUR  # positive = surplus
                durations[i] = MAX_DUR
                locked[i] = True
            else:
                unlocked_words += word_counts[i]
        if abs(excess) < 0.001 or unlocked_words == 0:
            break
        # Distribute excess proportionally among unlocked segments
        for i in range(n):
            if not locked[i]:
                durations[i] += excess * (word_counts[i] / unlocked_words)

    # Force-normalise to exactly match total_dur
    dur_sum = sum(durations)
    if abs(dur_sum - total_dur) > 0.001:
        scale = total_dur / dur_sum
        durations = [d * scale for d in durations]

    result: list[tuple[float, float]] = []
    t = start
    for i in range(n):
        seg_end = t + durations[i]
        if i == n - 1:
            seg_end = end  # ensure exact alignment
        result.append((t, seg_end))
        t = seg_end

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_short_segments(
    segments: list[tuple[float, float, str]],
    *,
    min_dur: float = MIN_DUR,
    max_dur: float = MAX_DUR,
    max_chars: int = MAX_CHARS,
) -> list[tuple[float, float, str]]:
    """Merge segments shorter than *min_dur* into the closest neighbour.

    "Closest" is decided by the silence gap on each side: a short tail
    1 s after the previous segment ends but 5 s before the next one
    starts is merged backwards; a short snippet just before a tightly
    following segment is merged forwards.

    Merge candidates that would push the resulting segment past
    *max_dur* or *max_chars* are skipped (kept as-is) to avoid producing
    over-long captions just to absorb a tiny tail.
    """
    if len(segments) < 2:
        return list(segments)

    out = [list(s) for s in segments]
    i = 0
    merged_count = 0
    while i < len(out):
        start, end, text = out[i]
        dur = end - start
        if dur >= min_dur:
            i += 1
            continue

        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None

        # Decide which neighbour is "closer" (smaller silence gap).
        gap_prev = (start - prev[1]) if prev is not None else float("inf")
        gap_next = (nxt[0] - end) if nxt is not None else float("inf")

        # Reject neighbours that would exceed the limits after merge.
        def _fits(neighbour):
            new_start = min(start, neighbour[0])
            new_end = max(end, neighbour[1])
            new_text_len = len(neighbour[2]) + 1 + len(text)
            return (new_end - new_start) <= max_dur and new_text_len <= max_chars

        prev_ok = prev is not None and _fits(prev)
        next_ok = nxt is not None and _fits(nxt)

        target = None
        if prev_ok and next_ok:
            target = "prev" if gap_prev <= gap_next else "next"
        elif prev_ok:
            target = "prev"
        elif next_ok:
            target = "next"

        # Hard fallback: if both neighbours technically exceed max_dur but
        # one is available, still absorb the short tail — the resulting
        # slight overflow is far better than a 0.4 s standalone segment
        # that TTS can't speak coherently.
        if target is None:
            if prev is not None and nxt is not None:
                target = "prev" if gap_prev <= gap_next else "next"
            elif prev is not None:
                target = "prev"
            elif nxt is not None:
                target = "next"

        if target == "prev":
            prev[1] = max(prev[1], end)
            prev[2] = (prev[2].rstrip() + " " + text.lstrip()).strip()
            del out[i]
            merged_count += 1
            # Re-check the merged segment from its position (no index inc).
            continue
        if target == "next":
            nxt[0] = min(nxt[0], start)
            nxt[2] = (text.rstrip() + " " + nxt[2].lstrip()).strip()
            del out[i]
            merged_count += 1
            # Don't increment: the slot now holds the (former) next entry,
            # which is already long enough — but loop will re-evaluate.
            continue

        # Neither neighbour fits — leave the short segment alone.
        i += 1

    if merged_count:
        log.info(
            "Short-segment merge: absorbed %d entries shorter than %.1fs "
            "into their nearest neighbour (%d -> %d)",
            merged_count, min_dur, len(segments), len(out),
        )
    return [tuple(s) for s in out]


def absorb_short_segments(
    srt_text: str,
    *,
    min_dur: float = MIN_DUR,
    max_dur: float = MAX_DUR,
    max_chars: int = MAX_CHARS,
) -> str:
    """Public helper: absorb tiny segments in *srt_text* into a neighbour.

    Pure-text wrapper around :func:`_merge_short_segments` so callers
    that already have a final SRT (e.g. a cached one) can apply the
    same cleanup without re-running the LLM merge/split phases.
    """
    blocks = parse_blocks(sanitize(srt_text))
    segs = [(s, e, t) for _, s, e, t in blocks]
    cleaned = _merge_short_segments(
        segs, min_dur=min_dur, max_dur=max_dur, max_chars=max_chars,
    )
    return build(cleaned)


def resegment_srt(
    srt_text: str,
    *,
    client: OpenAI | None = None,
    llm_model: str = "gpt-4.1",
    max_chars: int = MAX_CHARS,
    max_dur: float = MAX_DUR,
    usage_tracker: LLMUsageTracker | None = None,
) -> str:
    """Re-segment an SRT string so every entry is a complete speech phrase.

    **Phase 1 – Merge**: consecutive entries that are fragments of the same
    sentence are combined (via LLM when *client* is provided, otherwise by a
    punctuation-based heuristic).

    **Phase 2 – Split**: entries that exceed *max_chars* are broken at natural
    clause / sentence boundaries (via LLM or rule-based fallback).

    Returns:
        The re-segmented SRT as a string.
    """
    blocks = parse_blocks(sanitize(srt_text))

    # ── Phase 1: merge fragments into complete phrases ────────────────
    merged = _merge_phrases(blocks, client, llm_model, usage_tracker=usage_tracker)

    # ── Phase 2: split overly-long entries ────────────────────────────
    resegmented: list[tuple[float, float, str]] = []
    split_calls = 0

    for start, end, text in merged:
        text = text.strip()

        if len(text) <= max_chars and (end - start) <= max_dur:
            resegmented.append((start, end, text))
            continue

        if len(text) <= max_chars:
            resegmented.append((start, end, text))
            continue

        # Try LLM split, fall back to rules
        segments: list[str] | None = None
        if client is not None:
            try:
                segments = _llm_split(text, client, llm_model, usage_tracker=usage_tracker)
                split_calls += 1
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

    # ── Phase 3: absorb tiny tail segments into their nearest neighbour ──
    before_short_merge = len(resegmented)
    resegmented = _merge_short_segments(
        resegmented, min_dur=MIN_DUR, max_dur=max_dur, max_chars=max_chars,
    )

    log.info(
        "Re-segmented %d -> %d -> %d -> %d "
        "(original -> merged -> split -> short-absorbed, LLM split calls: %d)",
        len(blocks), len(merged), before_short_merge, len(resegmented), split_calls,
    )
    return build(resegmented)


# ═══════════════════════════════════════════════════════════════════════════════
#  Long-segment merge (rule-based, no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

_SENTENCE_END = frozenset(".!?…؟。！？")


def _find_silence_points(
    audio_path: str,
    blocks: list[tuple[str, float, float, str]],
    silence_thresh_db: float = -40.0,
    min_gap: float = 0.3,
) -> set[int]:
    """Return indices *i* where the gap between block *i* and *i+1* is silent."""
    from mazinger.assemble import _load_and_resample, _rms_energy, TARGET_SR

    audio = _load_and_resample(audio_path, TARGET_SR)
    total_samples = len(audio)
    thresh = 10 ** (silence_thresh_db / 20.0)
    silent = set()

    for i in range(len(blocks) - 1):
        gap_start = blocks[i][2]   # end of block i
        gap_end = blocks[i + 1][1]  # start of block i+1
        if gap_end - gap_start < min_gap:
            continue
        s0 = min(int(gap_start * TARGET_SR), total_samples)
        s1 = min(int(gap_end * TARGET_SR), total_samples)
        if s1 - s0 < TARGET_SR // 50:  # < 20ms
            continue
        region = audio[s0:s1]
        rms = _rms_energy(region, len(region))
        if rms[0] < thresh:
            silent.add(i)

    return silent


def merge_long_segments(
    srt_text: str,
    source_audio: str | None = None,
    *,
    min_duration: float = 8.0,
    max_duration: float = 30.0,
    silence_thresh_db: float = -40.0,
    min_silence_gap: float = 0.3,
) -> str:
    """Merge translated SRT blocks into longer chunks for TTS.

    Uses punctuation boundaries, inter-block gaps, and optionally source
    audio silence detection to decide where to split.  No LLM calls.

    Returns:
        Re-merged SRT string.
    """
    blocks = parse_blocks(sanitize(srt_text))
    if not blocks:
        return srt_text

    silent_gaps: set[int] = set()
    if source_audio:
        try:
            silent_gaps = _find_silence_points(
                source_audio, blocks, silence_thresh_db, min_silence_gap,
            )
        except Exception:
            log.debug("Silence detection failed; merging by punctuation only")

    merged: list[tuple[float, float, str]] = []
    texts: list[str] = []
    chunk_start: float | None = None
    chunk_end: float = 0.0

    for i, (_, start, end, text) in enumerate(blocks):
        text = text.strip()
        if not text:
            continue

        # Would including this block exceed the hard limit?
        if chunk_start is not None and (end - chunk_start) > max_duration:
            merged.append((chunk_start, chunk_end, " ".join(texts)))
            texts, chunk_start = [], None

        if chunk_start is None:
            chunk_start = start
        chunk_end = end
        texts.append(text)

        dur = chunk_end - chunk_start

        # Check for a natural break after this block
        last_char = text[-1] if text else ""
        is_sentence_end = last_char in _SENTENCE_END
        has_gap = (i in silent_gaps) or (
            i < len(blocks) - 1 and blocks[i + 1][1] - end >= min_silence_gap
        )

        if dur >= min_duration and (is_sentence_end or has_gap):
            merged.append((chunk_start, chunk_end, " ".join(texts)))
            texts, chunk_start = [], None

    if texts and chunk_start is not None:
        merged.append((chunk_start, chunk_end, " ".join(texts)))

    # Validation
    durations = [e - s for s, e, _ in merged]
    if durations:
        durations.sort()
        median = durations[len(durations) // 2]
        log.info(
            "Long-segment merge: %d -> %d chunks "
            "(min=%.1fs, max=%.1fs, median=%.1fs)",
            len(blocks), len(merged),
            durations[0], durations[-1], median,
        )
        over = [d for d in durations if d > max_duration * 1.1]
        if over:
            log.warning("%d chunks exceed max_duration (%.1fs)", len(over), max_duration)

    return build(merged)
