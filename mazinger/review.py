"""Review and correct ASR-generated SRT subtitles using an LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import json_repair
from tqdm.auto import tqdm

from mazinger.srt import parse_blocks, blocks_to_text
from mazinger.translate import _clean_llm_text
from mazinger.utils import LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

BLOCKS_PER_BATCH = 20
OVERLAP_SIZE = 4

def _build_system_prompt(
    description: dict,
    source_language: str = "auto",
    keep_technical_english: bool = False,
    video_meta: dict | None = None,
) -> str:
    """
    Build a prompt optimized for small/quantized LLMs (e.g. Gemma3 4b).
    Improvements:
    - Simpler, imperative language (fewer abstract rules)
    - Concrete dos/don'ts instead of vague guidelines
    - Reduced token count while preserving intent
    - Language-adaptive examples
    - Stricter output format anchoring
    """
    summary = description.get("summary", "")
    keywords = description.get("keywords", [])
    keypoints = description.get("keypoints", [])

    kw_str = ", ".join(keywords[:10]) if keywords else ""
    kp_str = "; ".join(keypoints[:5]) if keypoints else ""

    # ── Language block ────────────────────────────────────────────────────────
    if source_language == "auto":
        lang_instruction = (
            "Detect the language of each subtitle entry and correct it in that same language. "
            "Do NOT translate anything."
        )
    else:
        lang_instruction = (
            f"All subtitles are in {source_language}. "
            f"Correct them in {source_language} only. Do NOT translate."
        )

    # ── Context block (only if data exists) ──────────────────────────────────
    context_lines = []
    if video_meta:
        if video_meta.get("title"):
            context_lines.append(f"Video title: {video_meta['title']}")
        if video_meta.get("description"):
            desc = video_meta["description"]
            if len(desc) > 500:
                desc = desc[:500] + "…"
            context_lines.append(f"Video description: {desc}")
        if video_meta.get("tags"):
            context_lines.append(f"Tags: {', '.join(video_meta['tags'][:15])}")
    if summary:
        context_lines.append(f"Topic summary: {summary}")
    if kw_str:
        context_lines.append(f"Key terms: {kw_str}")
    if kp_str:
        context_lines.append(f"Key points: {kp_str}")
    context_block = "\n".join(context_lines) if context_lines else ""

    # ── Technical-English block ───────────────────────────────────────────────
    tech_rule = ""
    tech_example_block = ""
    if keep_technical_english:
        tech_rule = (
            "- Write technical/scientific terms (programming languages, frameworks, "
            "medical terms, software names, etc.) in their standard English form "
            "(e.g. write 'Python' not 'بايثون', 'React' not 'ريأكت'). "
            "Adjust surrounding words for grammar if needed.\n"
        )
        tech_example_block = """\

EXAMPLE — technical terms (Arabic input):
INPUT:  [{"index":"5","text":"نثبت المكتبة باستخدام بايثون وهذا الفريم وورك يدعم ريأكت"}]
OUTPUT: [{"index":"5","text":"نثبت المكتبة باستخدام Python وهذا الـ Framework يدعم React."}]
"""

    # ── Determine example language for the main example ───────────────────────
    # Use a generic English example always — small LLMs generalise better from English.
    main_example = """\

EXAMPLE — typo + punctuation (English):
INPUT:
[
  {"index":"1","text":"so lets talk about the importent featurs"},
  {"index":"2","text":"first  we need to instal the dependancies"},
  {"index":"3","text":"you can use pip instal or conda"}
]
OUTPUT:
[
  {"index":"1","text":"So let's talk about the important features."},
  {"index":"2","text":"First, we need to install the dependencies."},
  {"index":"3","text":"You can use pip install or conda."}
]

EXAMPLE — multilingual punctuation:
INPUT:  [{"index":"1","text":"bonjour comment allez vous aujourdui"}]
OUTPUT: [{"index":"1","text":"Bonjour, comment allez-vous aujourd'hui ?"}]
INPUT:  [{"index":"2","text":"今天我们来讨论一下这个问题吧"}]
OUTPUT: [{"index":"2","text":"今天我们来讨论一下这个问题吧。"}]"""

    # ── Assemble ──────────────────────────────────────────────────────────────
    context_section = f"\nCONTENT CONTEXT:\n{context_block}\n" if context_block else ""

    return f"""\
You are an ASR transcript corrector. {lang_instruction}
{context_section}
CORRECT only these errors:
- Obvious typos and misspellings (fix only when you are certain)
- Missing sentence-ending punctuation (. ? !)
- Clearly wrong or misplaced punctuation marks
- Words wrongly split or merged by ASR (e.g. "im portant" → "important", "ofthe" → "of the")
- Capitalize sentence starts (only for languages that use capitalization)
- Multiple spaces → single space
- Use punctuation conventions appropriate for the subtitle language
  (Chinese: 。，？！ Japanese: 。、 Spanish: ¿¡ French: space before : ; ! ?)
{tech_rule}
DO NOT:
- Change meaning, rephrase, or paraphrase
- Invent new words or terms not present in the original (use the key terms \
  above only to recognise misheard words, never to insert them)
- Add or remove words (except to fix obvious ASR split/merge errors)
- Translate any text to another language
- Change word order
- Skip any entry or reorder entries

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown, no explanation:
[{{"index":"<same index>","text":"<corrected text>"}}, ...]
{main_example}{tech_example_block}"""
                                                                                                                                               
def _is_safe_edit(original: str, corrected: str) -> bool:
    """Reject edits that add or remove content (likely hallucination).

    A correction fixes typos, punctuation, spacing and word boundaries, so the
    letters and digits of the line stay about the same.  Only those are
    compared — punctuation and spaces are free — with a few characters of
    slack so short lines can still get a real typo fix.
    """
    if not corrected.strip():
        return False
    orig = sum(1 for ch in original if ch.isalnum())
    corr = sum(1 for ch in corrected if ch.isalnum())
    slack = 4
    return orig * 0.75 - slack <= corr <= orig * 1.33 + slack


def _blocks_to_json(blocks):
    entries = []
    for idx, _start, _end, text in blocks:
        entries.append({"index": idx, "text": text})
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _blocks_to_context(blocks):
    return "\n".join(f'{idx}: "{text.strip()}"' for idx, _s, _e, text in blocks)


def _parse_response(raw, core_blocks):
    try:
        items = json_repair.loads(raw)
        if isinstance(items, list) and items:
            text_map = {}
            for item in items:
                if isinstance(item, dict) and "index" in item and "text" in item:
                    idx = str(item["index"])
                    text_map[idx] = _clean_llm_text(str(item["text"]), idx)

            if not text_map:
                raise ValueError("No valid entries in LLM response")

            result = []
            for idx, start, end, original in core_blocks:
                reviewed = text_map.get(idx, "")
                if reviewed and _is_safe_edit(original, reviewed):
                    result.append((idx, start, end, reviewed))
                else:
                    if reviewed:
                        log.debug(
                            "Unsafe edit rejected for block %s: %r → %r",
                            idx, original, reviewed,
                        )
                    result.append((idx, start, end, original))
            return result
    except Exception:
        pass

    log.warning("Review parse failed, keeping original text for batch")
    return list(core_blocks)


def review_srt(
    srt_text: str,
    description: dict,
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    source_language: str = "auto",
    keep_technical_english: bool = False,
    video_meta: dict | None = None,
    blocks_per_batch: int = BLOCKS_PER_BATCH,
    overlap_size: int = OVERLAP_SIZE,
    usage_tracker: LLMUsageTracker | None = None,
) -> str:
    """Review ASR-generated SRT and fix typos/punctuation via batched LLM calls.

    Parameters:
        srt_text:               Full SRT string to review.
        description:            Content description dict (summary, keywords, keypoints).
        client:                 An initialised OpenAI client.
        llm_model:              Model identifier.
        source_language:        Language of the subtitles (or 'auto').
        keep_technical_english: When True, convert technical terms to English.
        blocks_per_batch:       SRT blocks per LLM call.
        overlap_size:           Context blocks before/after each batch.

    Returns:
        The reviewed SRT as a string.
    """
    system_prompt = _build_system_prompt(
        description,
        source_language=source_language,
        keep_technical_english=keep_technical_english,
        video_meta=video_meta,
    )

    all_blocks = parse_blocks(srt_text)
    if not all_blocks:
        return srt_text

    # Pre-process: collapse multiple spaces (common ASR artifact)
    all_blocks = [
        (idx, start, end, re.sub(r"  +", " ", text).strip())
        for idx, start, end, text in all_blocks
    ]

    log.info("Reviewing %d SRT blocks in batches of %d", len(all_blocks), blocks_per_batch)

    half_overlap = overlap_size // 2
    reviewed_blocks = []

    batch_ranges = []
    for i in range(0, len(all_blocks), blocks_per_batch):
        batch_ranges.append((i, min(i + blocks_per_batch, len(all_blocks))))

    for batch_idx, (core_start, core_end) in enumerate(tqdm(batch_ranges, desc="Reviewing")):
        core = all_blocks[core_start:core_end]
        before = all_blocks[max(0, core_start - half_overlap):core_start]
        after = all_blocks[core_end:min(len(all_blocks), core_end + half_overlap)]

        batch_json = _blocks_to_json(core)

        payload = ""
        if before:
            payload += "== CONTEXT BEFORE (reference only, do NOT return) ==\n"
            payload += _blocks_to_context(before) + "\n\n"
        payload += "== REVIEW THESE ENTRIES (correct and return these) ==\n" + batch_json
        if after:
            payload += "\n\n== CONTEXT AFTER (reference only, do NOT return) ==\n"
            payload += _blocks_to_context(after)

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ]

        resp = client.chat.completions.create(
            model=llm_model, temperature=0.2, messages=msgs, think=False,
            top_p=0.9,
            num_predict=8000,
        )
        if usage_tracker is not None:
            usage_tracker.record("review", llm_model, resp)

        raw = resp.choices[0].message.content.strip()
        reviewed_blocks.extend(_parse_response(raw, core))

    result = blocks_to_text(reviewed_blocks)
    log.info("Review complete: %d entries", len(reviewed_blocks))
    return result


# ── SRT source selection ─────────────────────────────────────────────────────

_SELECT_PROMPT = """\
You are given two SRT subtitle transcriptions of the same video.

SOURCE A — produced by an ASR model (Whisper).
SOURCE B — auto-generated or uploaded by the video platform (YouTube).

Video metadata:
{meta}

Compare the two on these criteria:
1. Language correctness — grammar, spelling, proper word boundaries
2. Topic relevance — consistency with the video title/description/tags
3. Timestamp accuracy — natural segment boundaries, no excessive overlap
4. Completeness — coverage of the full audio without missing segments
5. Readability — clean punctuation, no garbled or hallucinated text

Return ONLY a JSON object (no markdown, no commentary):
{{"choice": "A" or "B", "reason": "one sentence"}}"""


def select_srt(
    srt_a: str,
    srt_b: str,
    client,
    *,
    llm_model: str = "gpt-4.1",
    video_meta: dict | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> str:
    """Ask an LLM to pick the better SRT between two candidates.

    Returns ``"A"`` or ``"B"``.
    """
    meta_lines = []
    if video_meta:
        for k in ("title", "description", "channel", "tags"):
            v = video_meta.get(k)
            if v:
                if isinstance(v, list):
                    v = ", ".join(str(i) for i in v[:10])
                meta_lines.append(f"{k}: {v}")
    meta_block = "\n".join(meta_lines) if meta_lines else "(not available)"

    # Truncate to keep prompt manageable.
    max_chars = 4000
    a_sample = srt_a[:max_chars]
    b_sample = srt_b[:max_chars]

    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        think=False,
        messages=[
            {"role": "system", "content": _SELECT_PROMPT.format(meta=meta_block)},
            {"role": "user", "content": (
                f"=== SOURCE A (ASR) ===\n{a_sample}\n\n"
                f"=== SOURCE B (YouTube) ===\n{b_sample}"
            )},
        ],
        num_predict=128,
    )
    if usage_tracker is not None:
        usage_tracker.record("select_srt", llm_model, resp)

    raw = resp.choices[0].message.content.strip()
    try:
        result = json_repair.loads(raw)
        choice = str(result.get("choice", "A")).upper()
        reason = result.get("reason", "")
    except Exception:
        choice = "A"
        reason = "parse error, defaulting to ASR"

    if choice not in ("A", "B"):
        choice = "A"

    log.info("SRT selection: %s — %s", choice, reason)
    return choice
