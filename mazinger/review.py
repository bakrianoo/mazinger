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
- Invent new words or terms not present in the original
- Add or remove words (except to fix obvious ASR split/merge errors)
- Translate any text to another language
- Change word order
- Skip any entry or reorder entries

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown, no explanation:
[{{"index":"<same index>","text":"<corrected text>"}}, ...]
{main_example}{tech_example_block}
Now correct the following JSON array:"""
                                                                                                                                               
def _is_safe_edit(original: str, corrected: str) -> bool:
    """Reject edits that change text length drastically (likely hallucination)."""
    if not corrected.strip():
        return False
    orig_len = max(len(original), 1)
    corr_len = len(corrected)
    if orig_len <= 5:
        return corr_len <= 30
    return orig_len / 3 <= corr_len <= orig_len * 3


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
                    text_map[str(item["index"])] = _clean_llm_text(str(item["text"]))

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
        payload += "== REVIEW THESE ENTRIES ==\n" + batch_json
        if after:
            payload += "\n\n== CONTEXT AFTER (reference only, do NOT return) ==\n"
            payload += _blocks_to_context(after)

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ]

        resp = client.chat.completions.create(
            model=llm_model, temperature=0.2, messages=msgs, think=False,
        )
        if usage_tracker is not None:
            usage_tracker.record("review", llm_model, resp)

        raw = resp.choices[0].message.content.strip()
        reviewed_blocks.extend(_parse_response(raw, core))

    result = blocks_to_text(reviewed_blocks)
    log.info("Review complete: %d entries", len(reviewed_blocks))
    return result
