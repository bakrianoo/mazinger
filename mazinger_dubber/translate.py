"""Translate SRT subtitles to a target language using an LLM with visual context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from mazinger_dubber.srt import parse_blocks, blocks_to_text, format_time
from mazinger_dubber.utils import make_image_content

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

BLOCKS_PER_BATCH = 24
OVERLAP_SIZE = 8


def _build_system_prompt(keywords: list[str], keypoints: list[str], target_language: str = "English") -> str:
    kw_examples = ", ".join(f'"{k}"' for k in keywords[:10])
    kp_summary = "; ".join(keypoints[:8])

    return f"""\
You are a professional {target_language} dubbing script writer for technical / \
programming tutorial videos. You are given SRT subtitles with duration \
annotations, video screenshots, and a keyword/keypoint list. Produce natural, \
well-phrased {target_language} dubbing scripts -- not a literal word-for-word \
translation, but also NOT a compressed summary.

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
- Each entry includes a [duration: Xs | target: ~N words] annotation.
- English speech averages ~2.5 words per second. Your translation for each \
  entry MUST contain approximately the target number of words so the dubbed \
  audio fills the same time window as the original speech.
- Example: a [duration: 20.0s | target: ~50 words] entry needs about 50 words.
- This is the MOST IMPORTANT constraint. Short translations create awkward \
  silences in the dubbed output. If you find yourself writing significantly \
  fewer words than the target, expand your phrasing: add natural elaboration, \
  gentle transitions, or brief clarifications the speaker implied.
- Do NOT pad with meaningless filler. Every word should contribute to a \
  natural, fluent delivery.

STRUCTURAL RULES:
1. Translate EVERY subtitle entry in the MAIN BLOCK. Do NOT skip, merge, \
   split, or reorder entries.
2. Keep the EXACT SRT index numbers and timestamps -- only replace the \
   source text with {target_language}. Remove the [duration/target] \
   annotations from your output.
3. Preserve these technical terms exactly: {kw_examples}. \
   Use them verbatim when the speaker refers to them.
4. The video covers: {kp_summary}. Use this to disambiguate unclear references.
5. Return ONLY the translated SRT block for the MAIN BLOCK entries -- \
   no fences, no commentary.
6. Each subtitle entry format:
   <index>
   <start> --> <end>
   <translated text>

   (blank line)

You will receive CONTEXT BEFORE and CONTEXT AFTER sections. They are for \
reference only -- translate and return ONLY the MAIN BLOCK entries."""


# Average English speech rate in words per second (used for duration targeting).
WORDS_PER_SECOND = 2.5


def _blocks_to_annotated_text(blocks: list[tuple[str, float, float, str]]) -> str:
    """Like ``blocks_to_text`` but adds duration & word-count annotations."""
    parts: list[str] = []
    for idx, start, end, text in blocks:
        dur = end - start
        target_words = max(1, round(dur * WORDS_PER_SECOND))
        parts.append(
            f"{idx}\n{format_time(start)} --> {format_time(end)}\n"
            f"[duration: {dur:.1f}s | target: ~{target_words} words]\n{text}\n"
        )
    return "\n".join(parts)


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
    srt_batch: str,
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
        user_parts.append({"type": "text", "text": "SCREENSHOTS from this segment:"})
        for tp in batch_thumbs:
            user_parts.append({"type": "text", "text": f"  [{tp['timestamp']}] {tp['reason']}"})
            user_parts.append(make_image_content(tp["path"]))

    srt_payload = ""
    if context_before:
        srt_payload += "== CONTEXT BEFORE (do NOT translate) ==\n" + context_before + "\n\n"
    srt_payload += "== MAIN BLOCK (translate these entries) ==\n" + srt_batch
    if context_after:
        srt_payload += "\n\n== CONTEXT AFTER (do NOT translate) ==\n" + context_after

    user_parts.append({
        "type": "text",
        "text": (
            f"\nTranslate the MAIN BLOCK entries into natural, full-length {target_language} "
            "suitable for dubbing. Use CONTEXT BEFORE/AFTER for surrounding context "
            "but ONLY return translations for the MAIN BLOCK. Use the screenshots "
            "and context to resolve vague or incomplete references.\n"
            "Keep index numbers and timestamps EXACTLY as-is. "
            "Match the target word count shown in each entry's [duration/target] "
            "annotation -- this is critical for dubbing timing.\n\n"
            + srt_payload
        ),
    })

    msgs.append({"role": "user", "content": user_parts})
    return msgs


def translate_srt(
    srt_text: str,
    description: dict,
    thumb_paths: list[dict],
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    target_language: str = "English",
    blocks_per_batch: int = BLOCKS_PER_BATCH,
    overlap_size: int = OVERLAP_SIZE,
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

    Returns:
        The translated SRT as a string.
    """
    keywords = description.get("keywords", [])
    keypoints = description.get("keypoints", [])
    system_prompt = _build_system_prompt(keywords, keypoints, target_language)

    all_blocks = parse_blocks(srt_text)
    log.info("Translating %d SRT blocks in batches of %d", len(all_blocks), blocks_per_batch)

    batch_ranges = []
    for i in range(0, len(all_blocks), blocks_per_batch):
        batch_ranges.append((i, min(i + blocks_per_batch, len(all_blocks))))

    half_overlap = overlap_size // 2
    translated_parts: list[str] = []

    for batch_idx, (core_start, core_end) in enumerate(tqdm(batch_ranges, desc="Translating")):
        core_blocks = all_blocks[core_start:core_end]
        core_indices = {b[0] for b in core_blocks}

        ctx_before_start = max(0, core_start - half_overlap)
        ctx_after_end = min(len(all_blocks), core_end + half_overlap)

        before_blocks = all_blocks[ctx_before_start:core_start]
        after_blocks = all_blocks[core_end:ctx_after_end]

        batch_srt = _blocks_to_annotated_text(core_blocks)
        context_before = blocks_to_text(before_blocks) if before_blocks else ""
        context_after = blocks_to_text(after_blocks) if after_blocks else ""

        full_start = before_blocks[0][1] if before_blocks else core_blocks[0][1]
        full_end = after_blocks[-1][2] if after_blocks else core_blocks[-1][2]
        batch_thumbs = _find_thumbnails_for_range(thumb_paths, full_start, full_end)

        log.debug(
            "Batch %d: blocks %d-%d (core=%d, ctx_before=%d, ctx_after=%d)",
            batch_idx + 1, core_start + 1, core_end,
            len(core_blocks), len(before_blocks), len(after_blocks),
        )

        msgs = _build_messages(
            system_prompt, batch_srt, batch_thumbs,
            keypoints, keywords, context_before, context_after,
            target_language=target_language,
        )
        resp = client.chat.completions.create(
            model=llm_model, temperature=0.3, messages=msgs,
        )
        translated_batch = resp.choices[0].message.content.strip()

        # Filter to only core indices
        translated_blocks = parse_blocks(translated_batch)
        filtered = [b for b in translated_blocks if b[0] in core_indices]

        if len(filtered) != len(core_blocks):
            log.warning(
                "Batch %d: expected %d entries, got %d (raw=%d). Using raw output.",
                batch_idx + 1, len(core_blocks), len(filtered), len(translated_blocks),
            )
            translated_parts.append(translated_batch)
        else:
            translated_parts.append(blocks_to_text(filtered))

    result = "\n\n".join(translated_parts) + "\n"

    original_count = len(all_blocks)
    translated_count = len(parse_blocks(result))
    log.info("Translation complete: %d -> %d entries", original_count, translated_count)
    if original_count != translated_count:
        log.warning("Entry count mismatch: %d original vs %d translated", original_count, translated_count)

    return result
