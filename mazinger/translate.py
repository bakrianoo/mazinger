"""Translate SRT subtitles to a target language using an LLM with visual context."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import json_repair
from tqdm.auto import tqdm

from mazinger.srt import parse_blocks, blocks_to_text, sanitize
from mazinger.utils import make_image_content, LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

# ── Patterns for cleaning common weak-LLM artifacts from translated text ─────

# Timestamp tags: [MM:SS], [HH:MM:SS], [0:23], [12:05:03], [MM:SS.ms]
_TIMESTAMP_TAG_RE = re.compile(r"\[\d{1,2}(?::\d{2}){1,2}(?:[.,]\d+)?\]")

# Duration/target annotations echoed back: [duration: 4.0s | target: ~6 words]
_DURATION_TAG_RE = re.compile(
    r"\[duration:\s*[\d.]+s?\s*\|\s*target:\s*~?\d+\s*words?\]",
    re.IGNORECASE,
)

# SRT timestamp arrows: 00:00:01,000 --> 00:00:05,000
_SRT_ARROW_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")

# XML/HTML tags that LLMs commonly hallucinate
_LLM_XML_TAG_RE = re.compile(
    r"</?(?:index|translated[_ ]?text|original[_ ]?text|start|end|"
    r"subtitle|entry|segment|translation|text|source|target|item|lang)>",
    re.IGNORECASE,
)

# Markdown code fences
_CODE_FENCE_RE = re.compile(r"```(?:json|srt|text)?")

# Leading index prefix like "1." or "1:" at the very start of text
_LEADING_INDEX_RE = re.compile(r"^\d+[.:]\s+")


def _clean_llm_text(text: str) -> str:
    """Strip common weak-LLM artifacts from a translated subtitle text.

    Removes timestamp tags, duration annotations, SRT arrows, XML tags,
    code fences, and leading index prefixes that weak models may echo back.
    """
    text = _TIMESTAMP_TAG_RE.sub("", text)
    text = _DURATION_TAG_RE.sub("", text)
    text = _SRT_ARROW_RE.sub("", text)
    text = _LLM_XML_TAG_RE.sub("", text)
    text = _CODE_FENCE_RE.sub("", text)
    # Collapse whitespace before checking leading index (prior removals may
    # leave leading spaces that prevent the anchor from matching).
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = _LEADING_INDEX_RE.sub("", text)
    return text.strip()


SUPPORTED_LANGUAGES = (
    "Arabic",
    "Bengali",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Persian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swedish",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Vietnamese",
)

_LANG_LOOKUP = {lang.lower(): lang for lang in SUPPORTED_LANGUAGES}


def _format_language_list() -> str:
    """Format the supported language list as a readable multi-column block."""
    col_width = max(len(lang) for lang in SUPPORTED_LANGUAGES) + 4
    cols = 3
    lines = []
    for i in range(0, len(SUPPORTED_LANGUAGES), cols):
        row = SUPPORTED_LANGUAGES[i:i + cols]
        lines.append("  ".join(lang.ljust(col_width) for lang in row).rstrip())
    return "\n".join(lines)


def resolve_language(value: str) -> str:
    """Return the canonical language name, or raise ``ValueError``."""
    canonical = _LANG_LOOKUP.get(value.lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported language: '{value}'\n\n"
            f"Supported languages:\n{_format_language_list()}"
        )
    return canonical


def resolve_source_language(value: str) -> str:
    """Like ``resolve_language`` but also accepts ``'auto'``."""
    if value.lower() == "auto":
        return "auto"
    return resolve_language(value)


BLOCKS_PER_BATCH = 24
OVERLAP_SIZE = 8

# Average English speech rate in words per second (used for duration targeting).
WORDS_PER_SECOND = 2.0
# Fraction of duration-based word count to use as the target.  Keeps dubbed
# audio from overrunning each segment's time window.
DURATION_BUDGET = 0.80


def _build_system_prompt(
    keywords: list[str],
    keypoints: list[str],
    target_language: str = "English",
    source_language: str = "auto",
    words_per_second: float = WORDS_PER_SECOND,
    duration_budget: float = DURATION_BUDGET,
    translate_technical_terms: bool = False,
) -> str:
    kw_examples = ", ".join(f'"{ k}"' for k in keywords[:10])
    kp_summary = "; ".join(keypoints[:8])
    budget_pct = int(duration_budget * 100)
    example_dur = 20.0
    example_target = int(example_dur * words_per_second * duration_budget)

    if source_language == "auto":
        source_ctx = (
            " The source subtitles may contain speech in one or more "
            "languages \u2014 identify the language(s) present and translate "
            f"all content into {target_language}."
        )
    else:
        source_ctx = f" The source subtitles are in {source_language}."

    return f"""\
You are a professional {target_language} dubbing script writer for technical / \
programming tutorial videos.{source_ctx} You are given subtitle texts as a JSON \
array (with index, text, and a target word count), video screenshots, and a \
keyword/keypoint list. Produce natural, well-phrased {target_language} dubbing \
scripts -- not a literal word-for-word translation, but also NOT a compressed \
summary.

QUALITY GOALS:
- The {target_language} must sound like a fluent {target_language}-speaking instructor \
  naturally explaining the topic in a friendly, conversational teaching tone.
- Clean up false starts, unintelligible fragments, and obvious speech errors. \
  However, PRESERVE the speaker's natural elaboration, rhetorical questions, \
  examples, and storytelling flow.
- Do NOT compress or summarize. The translation should convey the SAME level \
  of detail and explanation as the original.
- When the transcript is vague, incomplete, or references on-screen visuals, \
  use the screenshots and keypoint context to write a clear {target_language} sentence.
- If the original uses repetition or restates an idea for emphasis, rephrase \
  it into clean {target_language} that keeps the same emphasis without crude \
  repetition.

DURATION MATCHING (CRITICAL FOR DUBBING):
- Each entry has a "target_words" field — the maximum number of words for \
  your translation.
- The target word count is already set to ~{budget_pct}% of the available time window \
  (at ~{words_per_second} {target_language} words/second). This {budget_pct}% budget ensures the generated \
  dubbed voice finishes naturally BEFORE the next segment starts, leaving \
  a small breathing room.
- Your translation for each entry MUST stay WITHIN the target word count. \
  Do NOT exceed it -- going over causes the dubbed audio to be cut off \
  mid-sentence.
- Example: a "target_words": {example_target} entry needs around {example_target} words.
- Aim for 90-100% of the target word count. Significantly fewer words \
  create awkward silences; more words cause cut-off speech.
- If the original content is too dense for the word budget, prioritise the \
  core meaning and drop minor asides -- but never compress to less than \
  ~70% of the target.
- Do NOT pad with meaningless filler. Every word should contribute to a \
  natural, fluent delivery.

STRUCTURAL RULES:
1. Translate EVERY entry in the MAIN BLOCK. Do NOT skip, merge, split, or \
   reorder entries.
2. Return a JSON array of objects, one per input entry, in the SAME order. \
   Each object must have exactly two keys: \
   "index" (the original index) and "text" (the translated {target_language} text).
3. {_technical_terms_instruction(kw_examples, target_language, translate_technical_terms)}
4. The video covers: {kp_summary}. Use this to disambiguate unclear references.
5. Return ONLY the JSON array -- no markdown fences, no commentary, no XML \
   tags, no timestamps, no SRT formatting.

EXAMPLE OUTPUT:
[
  {{"index": "1", "text": "Translated sentence here."}},
  {{"index": "2", "text": "Next translated sentence."}}
]

You may receive CONTEXT BEFORE and CONTEXT AFTER sections. They are for \
reference only -- translate and return ONLY the MAIN BLOCK entries."""



def _technical_terms_instruction(
    kw_examples: str,
    target_language: str,
    translate_technical_terms: bool,
) -> str:
    if translate_technical_terms:
        return (
            f"Translate technical terms into professional, widely-accepted "
            f"{target_language} equivalents. Where a standard {target_language} "
            f"term exists for a concept (e.g. {kw_examples}), use the "
            f"{target_language} term. If no established translation exists, "
            f"transliterate or keep the original and integrate it naturally "
            f"into the {target_language} sentence."
        )
    return (
        f"Keep technical terms in their original language: {kw_examples}. "
        f"Embed them naturally within the {target_language} sentence so the "
        f"result reads fluently — adjust surrounding grammar, prepositions, "
        f"and word order as needed to accommodate the foreign-language term."
    )


def _blocks_to_json_entries(
    blocks: list[tuple[str, float, float, str]],
    words_per_second: float = WORDS_PER_SECOND,
    duration_budget: float = DURATION_BUDGET,
) -> str:
    """Convert blocks to a JSON array of {index, text, target_words} for LLM input."""
    entries = []
    for idx, start, end, text in blocks:
        dur = end - start
        target_words = max(1, round(dur * words_per_second * duration_budget))
        entries.append({
            "index": idx,
            "text": text,
            "target_words": target_words,
        })
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _blocks_to_context_text(
    blocks: list[tuple[str, float, float, str]],
) -> str:
    """Convert blocks to a simple numbered text list for LLM context (no timestamps)."""
    lines = []
    for idx, _start, _end, text in blocks:
        lines.append(f'{idx}: "{text.strip()}"')
    return "\n".join(lines)


def _find_thumbnails_for_range(
    thumb_paths: list[dict],
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    return [
        tp for tp in thumb_paths
        if start_sec <= float(tp["seconds"]) <= end_sec
    ]


def _build_messages(
    system_prompt: str,
    batch_json: str,
    batch_thumbs: list[dict],
    keypoints: list[str],
    keywords: list[str],
    context_before: str = "",
    context_after: str = "",
    target_language: str = "English",
) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    user_parts: list[dict] = []

    ctx = (
        "VIDEO CONTEXT:\n"
        f"Keypoints: {'; '.join(keypoints)}\n"
        f"Keywords: {', '.join(keywords)}\n\n"
    )
    user_parts.append({"type": "text", "text": ctx})

    if batch_thumbs:
        # Cap images per batch to keep prompt size reasonable for smaller models.
        if len(batch_thumbs) > 4:
            step = len(batch_thumbs) / 4
            batch_thumbs = [batch_thumbs[int(i * step)] for i in range(4)]
        user_parts.append({"type": "text", "text": "SCREENSHOTS from this segment:"})
        for tp in batch_thumbs:
            user_parts.append({"type": "text", "text": f"  [{tp['timestamp']}] {tp['reason']}"})
            user_parts.append(make_image_content(tp["path"]))

    payload = ""
    if context_before:
        payload += "== CONTEXT BEFORE (do NOT translate, for reference only) ==\n" + context_before + "\n\n"
    payload += "== MAIN BLOCK (translate these entries) ==\n" + batch_json
    if context_after:
        payload += "\n\n== CONTEXT AFTER (do NOT translate, for reference only) ==\n" + context_after

    user_parts.append({
        "type": "text",
        "text": (
            f"\nTranslate the MAIN BLOCK entries into natural, full-length {target_language} "
            "suitable for dubbing. Use CONTEXT BEFORE/AFTER for surrounding context "
            "but ONLY return translations for the MAIN BLOCK. Use the screenshots "
            "and context to resolve vague or incomplete references.\n"
            "Match the target_words count for each entry -- this is critical for "
            "dubbing timing.\n"
            "Return a JSON array of {\"index\": ..., \"text\": ...} objects in order.\n\n"
            + payload
        ),
    })

    msgs.append({"role": "user", "content": user_parts})
    return msgs


def _parse_translation_response(
    raw_content: str,
    core_blocks: list[tuple[str, float, float, str]],
) -> list[tuple[str, float, float, str]]:
    """Parse LLM JSON response and reconstruct blocks with original timestamps.

    Falls back to treating the response as raw SRT if JSON parsing fails,
    and ultimately falls back to keeping original text if nothing works.
    """
    # Try JSON parse first (expected path)
    try:
        translations = json_repair.loads(raw_content)
        if isinstance(translations, list) and translations:
            result: list[tuple[str, float, float, str]] = []
            # Build index -> translated text map (clean artifacts)
            trans_map: dict[str, str] = {}
            for item in translations:
                if isinstance(item, dict) and "index" in item and "text" in item:
                    trans_map[str(item["index"])] = _clean_llm_text(
                        str(item["text"])
                    )

            # Reconstruct blocks in original order with original timestamps
            for idx, start, end, original_text in core_blocks:
                translated_text = trans_map.get(idx, "")
                if translated_text:
                    result.append((idx, start, end, translated_text))
                else:
                    log.warning("Missing translation for index %s, keeping original", idx)
                    result.append((idx, start, end, original_text))

            if len(result) == len(core_blocks):
                return result
            log.warning(
                "JSON translation count mismatch: expected %d, got %d",
                len(core_blocks), len(result),
            )
    except Exception:
        pass

    # Fallback: try parsing as SRT (in case LLM ignored JSON instruction)
    log.warning("JSON parse failed, attempting SRT fallback parse")
    translated_srt_blocks = parse_blocks(sanitize(raw_content))
    if translated_srt_blocks:
        result = []
        srt_map = {b[0]: _clean_llm_text(b[3]) for b in translated_srt_blocks}
        for idx, start, end, original_text in core_blocks:
            translated_text = srt_map.get(idx, "")
            if translated_text:
                result.append((idx, start, end, translated_text))
            else:
                result.append((idx, start, end, original_text))
        return result

    # Last resort: return original blocks unchanged
    log.warning("All parsing failed, returning original text for batch")
    return list(core_blocks)


def translate_srt(
    srt_text: str,
    description: dict,
    thumb_paths: list[dict],
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    source_language: str = "auto",
    target_language: str = "English",
    blocks_per_batch: int = BLOCKS_PER_BATCH,
    overlap_size: int = OVERLAP_SIZE,
    words_per_second: float = WORDS_PER_SECOND,
    duration_budget: float = DURATION_BUDGET,
    translate_technical_terms: bool = False,
    usage_tracker: LLMUsageTracker | None = None,
) -> str:
    """Translate an SRT file to the target language using batched LLM calls with visual context.

    Parameters:
        srt_text:         Full source-language SRT string.
        description:      Content description dict (must have ``keypoints`` and
                          ``keywords``).
        thumb_paths:      List of thumbnail metadata dicts.
        client:           An initialised OpenAI client.
        llm_model:        Model identifier.
        target_language:  Target language for translation (default: ``English``).
        blocks_per_batch: Number of core SRT blocks per LLM call.
        overlap_size:     Number of context blocks before/after each batch.
        translate_technical_terms: When ``True`` translate technical terms
                          into professional target-language equivalents;
                          when ``False`` (default) keep them in the original
                          language.

    Returns:
        The translated SRT as a string.
    """
    source_language = resolve_source_language(source_language)
    target_language = resolve_language(target_language)

    keywords = description.get("keywords", [])
    keypoints = description.get("keypoints", [])
    system_prompt = _build_system_prompt(
        keywords, keypoints, target_language,
        source_language=source_language,
        words_per_second=words_per_second,
        duration_budget=duration_budget,
        translate_technical_terms=translate_technical_terms,
    )

    all_blocks = parse_blocks(srt_text)
    log.info("Translating %d SRT blocks in batches of %d", len(all_blocks), blocks_per_batch)

    batch_ranges = []
    for i in range(0, len(all_blocks), blocks_per_batch):
        batch_ranges.append((i, min(i + blocks_per_batch, len(all_blocks))))

    half_overlap = overlap_size // 2
    translated_blocks: list[tuple[str, float, float, str]] = []

    for batch_idx, (core_start, core_end) in enumerate(tqdm(batch_ranges, desc="Translating")):
        core_blocks = all_blocks[core_start:core_end]

        ctx_before_start = max(0, core_start - half_overlap)
        ctx_after_end = min(len(all_blocks), core_end + half_overlap)

        before_blocks = all_blocks[ctx_before_start:core_start]
        after_blocks = all_blocks[core_end:ctx_after_end]

        # Build JSON payload — no timestamps sent to LLM
        batch_json = _blocks_to_json_entries(
            core_blocks,
            words_per_second=words_per_second,
            duration_budget=duration_budget,
        )
        # Context blocks as simple numbered text (no timestamps)
        context_before = _blocks_to_context_text(before_blocks) if before_blocks else ""
        context_after = _blocks_to_context_text(after_blocks) if after_blocks else ""

        full_start = before_blocks[0][1] if before_blocks else core_blocks[0][1]
        full_end = after_blocks[-1][2] if after_blocks else core_blocks[-1][2]
        batch_thumbs = _find_thumbnails_for_range(thumb_paths, full_start, full_end)

        log.debug(
            "Batch %d: blocks %d-%d (core=%d, ctx_before=%d, ctx_after=%d)",
            batch_idx + 1, core_start + 1, core_end,
            len(core_blocks), len(before_blocks), len(after_blocks),
        )

        msgs = _build_messages(
            system_prompt, batch_json, batch_thumbs,
            keypoints, keywords, context_before, context_after,
            target_language=target_language,
        )
        resp = client.chat.completions.create(
            model=llm_model, temperature=0.3, messages=msgs,
        )
        if usage_tracker is not None:
            usage_tracker.record("translate", llm_model, resp)

        # Parse JSON response and reconstruct SRT with original timestamps
        raw_content = resp.choices[0].message.content.strip()
        batch_translated = _parse_translation_response(raw_content, core_blocks)
        translated_blocks.extend(batch_translated)

    result = blocks_to_text(translated_blocks)

    original_count = len(all_blocks)
    translated_count = len(translated_blocks)
    log.info("Translation complete: %d -> %d entries", original_count, translated_count)
    if original_count != translated_count:
        log.warning("Entry count mismatch: %d original vs %d translated", original_count, translated_count)

    return result
