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

# Baseline TTS speech rate by language (words per second).
# Measured empirically from Qwen3-TTS output at default settings.
_TTS_WPS: dict[str, float] = {
    "English": 3.2,
    "French": 3.4,
    "German": 3.0,
    "Spanish": 3.5,
    "Italian": 3.5,
    "Portuguese": 3.4,
    "Russian": 2.8,
    "Chinese (Simplified)": 3.0,
    "Chinese (Traditional)": 3.0,
    "Japanese": 4.0,
    "Korean": 3.2,
    "Arabic": 3.0,
    "Dutch": 3.0,
}
_DEFAULT_WPS = 3.0

# Fraction of duration-based word count to use as the target.
DURATION_BUDGET = 0.85
# Minimum words per segment — shorter budgets produce unusable fragments.
MIN_TARGET_WORDS = 4


def estimate_wps(
    blocks: list[tuple[str, float, float, str]],
    target_language: str = "English",
) -> float:
    """Estimate the target words-per-second for duration budgeting.

    Uses the source speech rate (words/time in the source SRT) scaled by the
    known TTS output rate for the target language.  Falls back to the
    language-specific TTS baseline when the source is too short or sparse.
    """
    tts_wps = _TTS_WPS.get(target_language, _DEFAULT_WPS)

    # Measure source speech density
    total_words = sum(len(text.split()) for _, _, _, text in blocks)
    total_dur = sum(end - start for _, start, end, _ in blocks)
    if total_dur < 1.0 or total_words < 5:
        return tts_wps

    source_wps = total_words / total_dur

    # The source speaker may be much faster than TTS can reproduce.
    # Cap at the TTS baseline — requesting more words than TTS can speak
    # just causes overflow and truncation.
    return min(source_wps, tts_wps)


def _build_system_prompt(
    keywords: list[str],
    keypoints: list[str],
    target_language: str = "English",
    source_language: str = "auto",
    words_per_second: float = _DEFAULT_WPS,
    duration_budget: float = DURATION_BUDGET,
    translate_technical_terms: bool = False,
    summary: str = "",
    dialect: str = "",
    tone: str = "",
    speakers: list[dict] | None = None,
    languages: list[str] | None = None,
    user_instructions: str = "",
) -> str:
    kw_examples = ", ".join(f'"{ k}"' for k in keywords[:10])
    kp_summary = "; ".join(keypoints[:8])
    budget_pct = int(duration_budget * 100)
    example_dur = 20.0
    example_target = int(example_dur * words_per_second * duration_budget)
    over_example = example_target + 5

    if source_language == "auto":
        source_ctx = (
            " The source subtitles may contain speech in one or more "
            "languages \u2014 identify the language(s) present and translate "
            f"all content into {target_language}."
        )
    else:
        source_ctx = f" The source subtitles are in {source_language}."

    if languages and len(languages) > 1:
        langs = ", ".join(languages)
        source_ctx += (
            f" The speaker uses multiple languages ({langs}) —"
            " interpret mixed-language passages in context and translate"
            f" everything into {target_language}."
        )

    genre_ctx = f" The video is about: {summary}" if summary else ""
    dialect_ctx = f" The source dialect is {dialect}." if dialect else ""
    tone_ctx = f" The delivery style is {tone}." if tone else ""
    speaker_ctx = ""
    if speakers:
        roles = ", ".join(
            f"{s['role']} ({s.get('desc', '')})" for s in speakers
        )
        speaker_ctx = f" Speakers: {roles}."

    _prompt = f"""\
You are a professional {target_language} dubbing script writer.{source_ctx}{genre_ctx}{dialect_ctx}{tone_ctx}{speaker_ctx} You are given subtitle texts as a JSON \
array (with index, text, and a target word count), video screenshots, and a \
keyword/keypoint list. Produce natural, well-phrased {target_language} dubbing \
scripts -- not a literal word-for-word translation, but also NOT a compressed \
summary.

QUALITY GOALS:
- The {target_language} must sound like a fluent native {target_language} speaker \
  naturally explaining the topic in a conversational tone matching the original \
  register.
- The source speech may use colloquial dialect rather than formal/standard \
  language. Always interpret words in the spoken dialect — colloquial or \
  dialectal meanings take priority over literary/formal ones.
- Preserve the speaker's point of view and self-references (e.g. references \
  to "our show", "previous episodes", "as we mentioned"). Keep the original \
  tense — do not shift past to future or vice versa.
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
- Each entry has a "target_words" field — the HARD MAXIMUM number of words \
  for your translation. Exceeding it causes the dubbed audio to be CUT OFF \
  mid-sentence, ruining the viewer experience.
- The target word count equals ~{budget_pct}% of the available time window \
  (at ~{words_per_second:.1f} {target_language} words/second).
- ALWAYS count your output words and ensure they are ≤ target_words. \
  For example, if "target_words": {example_target}, write exactly \
  {example_target} words or fewer — never {over_example}.
- Aim for 85-100% of the target. Fewer words = awkward silence; \
  more words = speech cut off.
- If the original content is too dense for the word budget, PRIORITISE \
  the core meaning and drop minor asides or redundant phrases. \
  Never pad with filler.

STRUCTURAL RULES:
1. Translate EVERY entry in the MAIN BLOCK. Do NOT skip or reorder entries.
   MERGING: If two or three ADJACENT entries are clearly fragments of the \
   same sentence or one continuous spoken thought (e.g. entry N introduces \
   a person/concept and entry N+1 immediately continues describing it), \
   you SHOULD merge them into a single entry. Use a hyphenated index like \
   "2-3" and the combined word budget. Do NOT merge entries that are about \
   different topics or separated by a clear topic shift.
2. Return a JSON array of objects in the SAME order. \
   Each object must have exactly two keys: \
   "index" (the original index, or a merged range like "2-3") and \
   "text" (the translated {target_language} text).
3. {_technical_terms_instruction(kw_examples, target_language, translate_technical_terms)}
4. The video covers: {kp_summary}. Use this to disambiguate unclear references.
5. Return ONLY the JSON array -- no markdown fences, no commentary, no XML \
   tags, no timestamps, no SRT formatting.

EXAMPLE OUTPUT (with a merge):
[
  {{"index": "1", "text": "Translated sentence here."}},
  {{"index": "2-3", "text": "Merged translation when entries 2 and 3 form one thought."}},
  {{"index": "4", "text": "Next translated sentence."}}
]

You may receive CONTEXT BEFORE and CONTEXT AFTER sections. They are for \
reference only -- translate and return ONLY the MAIN BLOCK entries."""
    if user_instructions and user_instructions.strip():
        _prompt += (
            "\n\nCONTENT & TRANSLATION GUIDELINES FROM THE USER:\n"
            + user_instructions.strip()
            + "\nApply these guidelines throughout your translation."
        )
    return _prompt


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
    words_per_second: float = _DEFAULT_WPS,
    duration_budget: float = DURATION_BUDGET,
) -> str:
    """Convert blocks to a JSON array of {index, text, target_words} for LLM input."""
    entries = []
    for idx, start, end, text in blocks:
        dur = end - start
        target_words = max(MIN_TARGET_WORDS, round(dur * words_per_second * duration_budget))
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
    video_meta: dict | None = None,
) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    user_parts: list[dict] = []

    ctx = (
        "VIDEO CONTEXT:\n"
        f"Keypoints: {'; '.join(keypoints)}\n"
        f"Keywords: {', '.join(keywords)}\n"
    )
    if video_meta:
        if video_meta.get("title"):
            ctx += f"Video title: {video_meta['title']}\n"
        if video_meta.get("description"):
            desc = video_meta["description"]
            if len(desc) > 500:
                desc = desc[:500] + "…"
            ctx += f"Video description: {desc}\n"
        if video_meta.get("channel") or video_meta.get("uploader"):
            ctx += f"Channel: {video_meta.get('channel') or video_meta.get('uploader')}\n"
        if video_meta.get("tags"):
            ctx += f"Tags: {', '.join(video_meta['tags'][:15])}\n"
    ctx += "\n"
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


# Regex to detect merged range indices like "2-3" or "2–3"
_RANGE_INDEX_RE = re.compile(r'^(\d+)\s*[-\u2013]\s*(\d+)$')


def _parse_translation_response(
    raw_content: str,
    core_blocks: list[tuple[str, float, float, str]],
) -> list[tuple[str, float, float, str]]:
    """Parse LLM JSON response and reconstruct blocks with original timestamps.

    Supports merged entries where the LLM combined adjacent blocks into one
    (indicated by a range index like ``"2-3"``).  Merged entries get the
    timestamp span of the combined source blocks.

    Falls back to treating the response as raw SRT if JSON parsing fails,
    and ultimately falls back to keeping original text if nothing works.
    """
    block_by_idx: dict[str, tuple[str, float, float, str]] = {
        idx: (idx, start, end, text) for idx, start, end, text in core_blocks
    }

    # Try JSON parse first (expected path)
    try:
        translations = json_repair.loads(raw_content)
        if isinstance(translations, list) and translations:
            result: list[tuple[str, float, float, str]] = []
            absorbed: set[str] = set()  # indices merged into a range

            for item in translations:
                if not isinstance(item, dict) or "index" not in item or "text" not in item:
                    continue
                raw_idx = str(item["index"]).strip()
                text = _clean_llm_text(str(item["text"]))

                range_m = _RANGE_INDEX_RE.match(raw_idx)
                if range_m:
                    first = range_m.group(1)
                    last = range_m.group(2)
                    first_block = block_by_idx.get(first)
                    last_block = block_by_idx.get(last)
                    if first_block and last_block and text:
                        merged_start = first_block[1]
                        merged_end = last_block[2]
                        result.append((first, merged_start, merged_end, text))
                        for i in range(int(first) + 1, int(last) + 1):
                            absorbed.add(str(i))
                        log.info("Merged translation entries %s-%s", first, last)
                    elif first_block and text:
                        result.append((first, first_block[1], first_block[2], text))
                else:
                    if raw_idx in absorbed:
                        continue
                    block = block_by_idx.get(raw_idx)
                    if block:
                        if text:
                            result.append((raw_idx, block[1], block[2], text))
                        else:
                            log.warning("Empty translation for index %s, keeping original", raw_idx)
                            result.append(block)

            # Add any blocks not covered by translation or absorption
            covered = {r[0] for r in result} | absorbed
            for idx, start, end, original_text in core_blocks:
                if idx not in covered:
                    log.warning("Missing translation for index %s, keeping original", idx)
                    result.append((idx, start, end, original_text))

            # Sort by start time to maintain order
            result.sort(key=lambda x: x[1])

            if result:
                if len(absorbed) > 0:
                    log.info(
                        "Translation batch: %d entries merged into %d (absorbed %d)",
                        len(core_blocks), len(result), len(absorbed),
                    )
                return result
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


def _validate_word_counts(
    translated_blocks: list[tuple[str, float, float, str]],
    words_per_second: float,
    duration_budget: float,
    tolerance: float = 1.5,
) -> list[tuple[str, float, float, str, int, int]]:
    """Return blocks that exceed their word budget by more than *tolerance*.

    Each returned tuple appends ``(actual_words, target_words)`` to the block.
    """
    violations = []
    for idx, start, end, text in translated_blocks:
        dur = end - start
        target = max(MIN_TARGET_WORDS, round(dur * words_per_second * duration_budget))
        actual = len(text.split())
        if actual > target * tolerance:
            violations.append((idx, start, end, text, actual, target))
    return violations


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
    words_per_second: float | None = None,
    duration_budget: float = DURATION_BUDGET,
    translate_technical_terms: bool = False,
    video_meta: dict | None = None,
    usage_tracker: LLMUsageTracker | None = None,
    user_instructions: str = "",
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
        words_per_second: Target speech rate.  When ``None`` (default), estimated
                          automatically from the source speech density and the
                          target language TTS rate.
        translate_technical_terms: When ``True`` translate technical terms
                          into professional target-language equivalents;
                          when ``False`` (default) keep them in the original
                          language.

    Returns:
        The translated SRT as a string.
    """
    source_language = resolve_source_language(source_language)
    target_language = resolve_language(target_language)

    all_blocks = parse_blocks(srt_text)

    if words_per_second is None:
        words_per_second = estimate_wps(all_blocks, target_language)
    log.info("Translation WPS: %.2f (budget: %.0f%%)", words_per_second, duration_budget * 100)

    # Use dialect from describe phase as source language when auto-detected
    described_dialect = description.get("dialect", "")
    if source_language == "auto" and described_dialect:
        source_language = described_dialect

    keywords = description.get("keywords", [])
    keypoints = description.get("keypoints", [])
    system_prompt = _build_system_prompt(
        keywords, keypoints, target_language,
        source_language=source_language,
        words_per_second=words_per_second,
        duration_budget=duration_budget,
        translate_technical_terms=translate_technical_terms,
        summary=description.get("summary", ""),
        dialect=description.get("dialect", ""),
        tone=description.get("tone", ""),
        speakers=description.get("speakers"),
        languages=description.get("languages"),
        user_instructions=user_instructions,
    )

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
            video_meta=video_meta,
        )
        resp = client.chat.completions.create(
            model=llm_model, temperature=0.3, messages=msgs,
            repeat_penalty=1.2,
            top_p=0.9,
            num_predict=8000,
            frequency_penalty=0.3,
        )
        if usage_tracker is not None:
            usage_tracker.record("translate", llm_model, resp)

        # Parse JSON response and reconstruct SRT with original timestamps
        raw_content = resp.choices[0].message.content.strip()
        batch_translated = _parse_translation_response(raw_content, core_blocks)
        translated_blocks.extend(batch_translated)

    # Validation report (assembly handles overflow via tempo stretch)
    violations = _validate_word_counts(
        translated_blocks, words_per_second, duration_budget,
    )
    if violations:
        over_total = sum(a - t for _, _, _, _, a, t in violations)
        log.warning(
            "%d/%d segments over word budget (excess: %d words). "
            "Assembly will compensate via tempo stretch. "
            "Segments: %s",
            len(violations), len(translated_blocks), over_total,
            ", ".join(
                "{}({}w/{}w)".format(idx, actual, target)
                for idx, _, _, _, actual, target in violations[:10]
            ),
        )

    result = blocks_to_text(translated_blocks)

    original_count = len(all_blocks)
    translated_count = len(translated_blocks)
    if translated_count < original_count:
        log.info(
            "Translation complete: %d -> %d entries (%d merged)",
            original_count, translated_count, original_count - translated_count,
        )
    elif original_count != translated_count:
        log.warning("Entry count mismatch: %d original vs %d translated", original_count, translated_count)
    else:
        log.info("Translation complete: %d entries", translated_count)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Simple per-segment translation (template-based, e.g. for `translategemma`)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical language name → ISO 639-1 code.  Used by template-based
# translators that expect explicit source/target codes.
_LANG_TO_ISO_CODE: dict[str, str] = {
    "Arabic": "ar", "Bengali": "bn",
    "Chinese (Simplified)": "zh", "Chinese (Traditional)": "zh",
    "Czech": "cs", "Danish": "da", "Dutch": "nl",
    "English": "en", "Finnish": "fi", "French": "fr",
    "German": "de", "Greek": "el", "Hebrew": "he",
    "Hindi": "hi", "Hungarian": "hu", "Indonesian": "id",
    "Italian": "it", "Japanese": "ja", "Korean": "ko",
    "Malay": "ms", "Norwegian": "no", "Persian": "fa",
    "Polish": "pl", "Portuguese": "pt", "Romanian": "ro",
    "Russian": "ru", "Spanish": "es", "Swedish": "sv",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk",
    "Urdu": "ur", "Vietnamese": "vi",
}

# Reverse: ISO code → canonical language name.  Whisper emits codes
# (``en``, ``ar`` …) so we need this to translate the detected language
# into a name the template can render.
_ISO_CODE_TO_LANG: dict[str, str] = {}
for _name, _code in _LANG_TO_ISO_CODE.items():
    _ISO_CODE_TO_LANG.setdefault(_code, _name)


def lang_name_from_code(code: str | None) -> str | None:
    """Return the canonical language name for an ISO 639-1 *code*.

    Returns ``None`` when *code* is empty or unknown.
    """
    if not code:
        return None
    return _ISO_CODE_TO_LANG.get(code.lower().split("-")[0])


def lang_code_from_name(name: str) -> str:
    """Return the ISO 639-1 code for a canonical language *name*.

    Falls back to ``"en"`` if *name* is not in the map.
    """
    return _LANG_TO_ISO_CODE.get(name, "en")


# User-facing translation template — kept in a constant so the same
# wording the model was fine-tuned on is used at runtime.
SIMPLE_TRANSLATION_TEMPLATE = (
    "You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to "
    "{TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately "
    "convey the meaning and nuances of the original {SOURCE_LANG} text "
    "while adhering to {TARGET_LANG} grammar, vocabulary, and cultural "
    "sensitivities.\n"
    "Produce only the {TARGET_LANG} translation, without any additional "
    "explanations or commentary. Please translate the following "
    "{SOURCE_LANG} text into {TARGET_LANG}:\n\n\n"
    "{TEXT}"
)


def _build_simple_prompt(
    text: str, source_language: str, target_language: str,
) -> str:
    return SIMPLE_TRANSLATION_TEMPLATE.format(
        SOURCE_LANG=source_language,
        SOURCE_CODE=lang_code_from_name(source_language),
        TARGET_LANG=target_language,
        TARGET_CODE=lang_code_from_name(target_language),
        TEXT=text,
    )


def _strip_simple_translation(reply: str) -> str:
    """Clean up a raw translation reply.

    Some chat-tuned translation models echo the prompt back, wrap the
    answer in code fences, or prefix it with ``"Translation:"``. Strip
    those common artifacts before returning.
    """
    text = reply.strip()
    text = _CODE_FENCE_RE.sub("", text)
    # Drop a leading "Translation:" / "Output:" label (case-insensitive).
    text = re.sub(r"^\s*(translation|output|answer)\s*[:\-]\s*", "", text, flags=re.I)
    # Some models repeat the source language name in the lead-in.
    text = re.sub(
        r"^\s*Here(?:'s| is) the [^\n:]+translation[^:]*:\s*", "",
        text, flags=re.I,
    )
    return text.strip()


# Maximum source-block duration before pre-splitting kicks in for the
# simple-template translator.  Without splitting, a 30 s source line can
# expand to a 60–90 s translation that the assembler would have to trim
# (the simple path has no per-segment word budgeting).  Splitting at
# sentence boundaries first keeps each block within a slot the
# assembler can realistically fit.
_SIMPLE_MAX_BLOCK_SEC = 12.0
_SIMPLE_MIN_SUBBLOCK_SEC = 2.5

# Sentence-terminator characters that we will split on, in priority
# order.  Includes Latin, Arabic (؟ ،), CJK (。！？), Devanagari (।)
# punctuation so the splitter behaves on multilingual input.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[\.!\?。！？؟।])\s+(?=\S)"
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?<=[,;:،؛])\s+(?=\S)"
)


def _split_text_for_duration(text: str, max_pieces: int) -> list[str]:
    """Split *text* into at most *max_pieces* readable chunks.

    Tries sentence boundaries first, then clause boundaries, then word
    boundaries — preferring natural breaks over fixed character counts
    so each chunk is independently translatable.
    """
    if max_pieces <= 1 or not text.strip():
        return [text]

    parts = _SENTENCE_BOUNDARY_RE.split(text)
    if len(parts) >= 2:
        return _balance_pieces(parts, max_pieces)

    parts = _CLAUSE_BOUNDARY_RE.split(text)
    if len(parts) >= 2:
        return _balance_pieces(parts, max_pieces)

    # Word-level fallback: cut into roughly equal piles of words.
    words = text.split()
    if len(words) < max_pieces:
        return [text]
    chunk = max(1, len(words) // max_pieces)
    return [
        " ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)
    ]


def _balance_pieces(parts: list[str], max_pieces: int) -> list[str]:
    """Greedily merge adjacent *parts* until at most *max_pieces* remain.

    Targets evenly-sized pieces by repeatedly merging the smallest
    adjacent pair — this avoids the degenerate "one giant + many tiny"
    output that naive truncation produces.
    """
    pieces = [p.strip() for p in parts if p.strip()]
    while len(pieces) > max_pieces:
        # Find the smallest adjacent pair (by combined length).
        sizes = [len(pieces[i]) + len(pieces[i + 1]) for i in range(len(pieces) - 1)]
        i = int(min(range(len(sizes)), key=lambda k: sizes[k]))
        pieces[i:i + 2] = [pieces[i] + " " + pieces[i + 1]]
    return pieces


def _presplit_blocks_for_simple(
    blocks: list[tuple[str, float, float, str]],
    max_block_sec: float = _SIMPLE_MAX_BLOCK_SEC,
    min_subblock_sec: float = _SIMPLE_MIN_SUBBLOCK_SEC,
) -> tuple[list[tuple[str, float, float, str]], int]:
    """Split overly-long source blocks before per-segment translation.

    Each block longer than *max_block_sec* is broken into N sub-blocks
    where ``N = ceil(duration / max_block_sec)``.  Sub-block timings are
    distributed proportionally to text length so dense passages keep
    more time than short tails.  Sub-block IDs are suffixed (e.g.
    ``"7a"``, ``"7b"``) to remain unique without colliding with
    original numeric IDs.

    Returns a tuple of ``(new_blocks, n_split)`` where *n_split* is the
    number of original blocks that were split (for logging).
    """
    out: list[tuple[str, float, float, str]] = []
    n_split = 0
    for idx, start, end, text in blocks:
        dur = end - start
        if dur <= max_block_sec:
            out.append((idx, start, end, text))
            continue

        # How many sub-blocks?  Use ceil so even a tiny excess triggers
        # an extra piece (e.g. 13 s / 12 s → 2 pieces, not 1).  Cap so
        # each piece stays above the readable minimum.
        import math
        n_pieces = max(2, math.ceil(dur / max_block_sec))
        n_pieces = min(n_pieces, max(2, int(dur // min_subblock_sec)))

        pieces = _split_text_for_duration(text, n_pieces)
        if len(pieces) < 2:
            out.append((idx, start, end, text))
            continue

        # Distribute time proportionally to character count.
        lengths = [max(1, len(p)) for p in pieces]
        total = sum(lengths)
        cursor = start
        for i, piece in enumerate(pieces):
            share = dur * (lengths[i] / total)
            piece_end = end if i == len(pieces) - 1 else cursor + share
            sub_idx = f"{idx}{chr(ord('a') + i)}" if len(pieces) <= 26 else f"{idx}.{i + 1}"
            out.append((sub_idx, cursor, piece_end, piece))
            cursor = piece_end
        n_split += 1
    return out, n_split


def translate_srt_simple(
    srt_text: str,
    client: OpenAI,
    *,
    llm_model: str = "translategemma",
    source_language: str = "auto",
    target_language: str = "English",
    usage_tracker: LLMUsageTracker | None = None,
    temperature: float = 0.1,
    max_block_sec: float = _SIMPLE_MAX_BLOCK_SEC,
) -> str:
    """Translate every SRT entry independently with a template prompt.

    Designed for lightweight task-specific translation models such as
    ``translategemma`` that expect a single-sentence prompt of the form
    described in :data:`SIMPLE_TRANSLATION_TEMPLATE`.

    Differs from :func:`translate_srt` in that it does **not** use
    visual context, batching, or duration budgeting — each subtitle
    block is translated on its own.  This is faster on small models
    and avoids the JSON-formatting failure modes that weak LLMs hit
    when asked to return structured arrays.

    Source blocks longer than *max_block_sec* are pre-split on
    sentence/clause boundaries before translation so the assembler can
    fit each translated chunk into a realistic time slot — this
    prevents the "85 s of audio in a 29 s slot" failure mode that
    happens when a small translation model expands a long input.

    Parameters:
        srt_text:         Source SRT string.
        client:           Initialised OpenAI-compatible client.
        llm_model:        Translation model name.
        source_language:  Canonical language name (e.g. ``English``) or
                          ``auto`` — when auto, defaults to ``English``.
        target_language:  Canonical target language name.
        max_block_sec:    Source blocks longer than this (seconds) are
                          pre-split on sentence boundaries.

    Returns:
        Translated SRT preserving the original timestamps (sub-block
        boundaries inserted where pre-splitting was applied).
    """
    src = source_language if source_language and source_language != "auto" else "English"
    tgt = resolve_language(target_language)
    src = resolve_language(src)

    raw_blocks = parse_blocks(srt_text)
    blocks, n_split = _presplit_blocks_for_simple(
        raw_blocks, max_block_sec=max_block_sec,
    )
    if n_split:
        log.info(
            "Pre-split %d source blocks > %.1fs → %d total entries",
            n_split, max_block_sec, len(blocks),
        )
    log.info(
        "Simple-translate %d entries via %s (%s → %s)",
        len(blocks), llm_model, src, tgt,
    )

    translated: list[tuple[str, float, float, str]] = []
    for idx, start, end, text in tqdm(blocks, desc="Translating"):
        original = text.strip()
        if not original:
            translated.append((idx, start, end, original))
            continue

        prompt = _build_simple_prompt(original, src, tgt)
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if usage_tracker is not None:
                usage_tracker.record("translate", llm_model, resp)
            reply = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Simple translation failed for entry %s: %s — keeping original",
                idx, exc,
            )
            translated.append((idx, start, end, original))
            continue

        cleaned = _strip_simple_translation(_clean_llm_text(reply))
        if not cleaned:
            log.warning(
                "Empty translation for entry %s — keeping original", idx,
            )
            cleaned = original
        translated.append((idx, start, end, cleaned))

    # Renumber sequentially — pre-splitting may have produced
    # non-numeric IDs (e.g. ``"7a"``, ``"7b"``) that downstream tools
    # expecting plain integer SRT indices would reject.
    renumbered = [
        (str(i), start, end, text)
        for i, (_idx, start, end, text) in enumerate(translated, 1)
    ]
    return blocks_to_text(renumbered)
