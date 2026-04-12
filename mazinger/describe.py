"""Analyse video content and produce a structured description via LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import json_repair

from mazinger.srt import parse_blocks
from mazinger.utils import make_image_content, LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

_DESCRIBE_SYSTEM = """\
You are a concise, factual video content analyst.
You will receive screenshots (thumbnails) and subtitle texts from a video.

Your task: produce a SHORT structured JSON description of the video.

STRICT RULES:
- ONLY state facts directly supported by the subtitles and images.
- Do NOT invent, guess, or hallucinate any information.
- Do NOT repeat yourself — every keypoint and keyword must be UNIQUE.
- Keep keypoints to 5-10 items MAX. Each must be a distinct concept.
- Keep keywords to 10-20 items MAX. No duplicates, no near-duplicates.
- If the content is in a non-English language, still write title/summary/keypoints in English.
- Keywords should preserve original casing and include technical terms, names, tools.
- For "dialect": identify the specific spoken dialect or register from the subtitle text
  (e.g. "Egyptian Arabic", "Brazilian Portuguese", "American English", "formal MSA").
  If uncertain, write the broad language name.
- For "languages": list ALL languages that appear in the speech, in order of prevalence.
  Include the primary language and any secondary languages the speaker switches to
  (e.g. ["Arabic", "English"] when the speaker uses English terms within Arabic speech).
- For "tone": describe the delivery style in 2-4 words (e.g. "serious analytical",
  "casual storytelling", "news report", "comedic", "academic lecture").
- For "speakers": list ONLY people who actually speak on-camera in the video,
  NOT people who are merely quoted, cited, or referenced by the speaker.
  Each entry has "role" (e.g. "host", "narrator", "interviewer", "guest") and
  a short "desc". If only one speaker, still include them.

Return EXACTLY this JSON structure (no markdown fences, no extra text):
{
  "title": "short descriptive title",
  "summary": "2-3 sentences summarising the video content",
  "dialect": "specific spoken dialect or register",
  "languages": ["primary language", "secondary language if any"],
  "tone": "2-4 word delivery style",
  "speakers": [{"role": "host", "desc": "short description"}],
  "keypoints": ["unique point 1", "unique point 2", ...],
  "keywords": ["term1", "Term2", ...]
}"""


def describe_content(
    srt_text: str,
    thumb_paths: list[dict],
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    video_meta: dict | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> dict:
    """Send thumbnails and the full SRT to an LLM for content analysis.

    Parameters:
        srt_text:    Full SRT content string.
        thumb_paths: List of thumbnail dicts, each containing ``path``,
                     ``timestamp``, and ``reason`` keys.
        client:      An initialised OpenAI client.
        llm_model:   Model identifier to use.

    Returns:
        A dict with ``title``, ``summary``, ``keypoints``, and ``keywords``.
    """
    # Evenly sample thumbnails — keep the payload manageable for smaller models.
    MAX_IMAGES = 8
    if len(thumb_paths) > MAX_IMAGES:
        step = len(thumb_paths) / MAX_IMAGES
        thumb_paths = [thumb_paths[int(i * step)] for i in range(MAX_IMAGES)]
        log.info("Sampled %d/%d thumbnails for description", MAX_IMAGES, len(thumb_paths))

    user_parts: list[dict] = []

    # Inject video metadata (title, description, tags, etc.) when available.
    if video_meta:
        meta_lines = []
        if video_meta.get("title"):
            meta_lines.append(f"Title: {video_meta['title']}")
        if video_meta.get("description"):
            # Truncate very long descriptions to avoid overwhelming the prompt.
            desc = video_meta["description"]
            if len(desc) > 1000:
                desc = desc[:1000] + "…"
            meta_lines.append(f"Description: {desc}")
        if video_meta.get("uploader") or video_meta.get("channel"):
            meta_lines.append(f"Channel: {video_meta.get('channel') or video_meta.get('uploader')}")
        if video_meta.get("tags"):
            meta_lines.append(f"Tags: {', '.join(video_meta['tags'][:20])}")
        if video_meta.get("categories"):
            meta_lines.append(f"Categories: {', '.join(video_meta['categories'])}")
        if meta_lines:
            user_parts.append({
                "type": "text",
                "text": "VIDEO METADATA:\n" + "\n".join(meta_lines) + "\n",
            })

    for tp in thumb_paths:
        user_parts.append({"type": "text", "text": f"[{tp['timestamp']}] {tp['reason']}"})
        user_parts.append(make_image_content(tp["path"]))

    # Send only subtitle texts — timestamps are irrelevant for content analysis.
    # Limit subtitle lines to avoid overwhelming small models.
    blocks = parse_blocks(srt_text)
    MAX_LINES = 120
    if len(blocks) > MAX_LINES:
        step = len(blocks) / MAX_LINES
        blocks = [blocks[int(i * step)] for i in range(MAX_LINES)]
    numbered_lines = [f"{idx}. {text.strip()}" for idx, _s, _e, text in blocks]
    texts_only = "\n".join(numbered_lines)
    user_parts.append({"type": "text", "text": f"\n\nSubtitle texts:\n\n{texts_only}"})
    user_parts.append({
        "type": "text",
        "text": (
            "\nNow return the JSON object. Remember:\n"
            "- NO duplicate keypoints or keywords.\n"
            "- ONLY facts from the subtitles above.\n"
            "- Keep it concise."
        ),
    })

    log.info("Requesting content description from %s...", llm_model)
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.15,
        think=False,
        messages=[
            {"role": "system", "content": _DESCRIBE_SYSTEM},
            {"role": "user", "content": user_parts},
        ],
        # Sampling options — penalise repetition, cap output length
        repeat_penalty=1.3,
        top_p=0.9,
        num_predict=1024,
        frequency_penalty=0.5,
    )
    if usage_tracker is not None:
        usage_tracker.record("describe", llm_model, resp)
    description = json_repair.loads(resp.choices[0].message.content)

    # Guard: small/local LLMs may return a non-dict (list, string, etc.).
    # Wrap into the expected schema so downstream code doesn't crash.
    if not isinstance(description, dict):
        log.warning("LLM returned non-dict description (%s), wrapping into schema", type(description).__name__)
        description = {"title": "", "summary": str(description), "keypoints": [], "keywords": []}

    # ── Post-process: deduplicate keypoints and keywords ──────────
    if isinstance(description.get("keypoints"), list):
        seen = set()
        deduped = []
        for kp in description["keypoints"]:
            key = kp.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(kp.strip())
        description["keypoints"] = deduped[:10]

    if isinstance(description.get("keywords"), list):
        seen = set()
        deduped = []
        for kw in description["keywords"]:
            key = kw.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(kw.strip())
        description["keywords"] = deduped[:20]

    log.info("Description generated: %s", description.get("title", ""))
    return description
