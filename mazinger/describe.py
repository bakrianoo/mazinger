"""Analyse video content and produce a structured description via LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import json_repair

from mazinger.utils import make_image_content, LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

_DESCRIBE_SYSTEM = """\
You are an expert technical content analyst. You will be given:
1. Screenshots (thumbnails) from a tutorial video with their timestamps.
2. The full SRT subtitle file of the video.

Produce a structured JSON object with:
- "title": concise English title
- "summary": 2-4 sentence English summary
- "keypoints": list of 8-15 key topics/concepts (English, short phrases)
- "keywords": list of 15-30 technical terms, library names, tool names, and
  domain-specific vocabulary (preserve original casing, e.g. "FastAPI")

Return ONLY valid JSON -- no markdown fences, no extra text."""


def describe_content(
    srt_text: str,
    thumb_paths: list[dict],
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    usage_tracker: LLMUsageTracker | None = None,
    extra_body: dict | None = None,
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
    # Evenly sample thumbnails across the video timeline to stay concise.
    MAX_IMAGES = 50
    if len(thumb_paths) > MAX_IMAGES:
        step = len(thumb_paths) / MAX_IMAGES
        thumb_paths = [thumb_paths[int(i * step)] for i in range(MAX_IMAGES)]
        log.info("Sampled %d thumbnails (original count exceeded API limit)", MAX_IMAGES)

    user_parts: list[dict] = []
    for tp in thumb_paths:
        user_parts.append({"type": "text", "text": f"[{tp['timestamp']}] {tp['reason']}"})
        user_parts.append(make_image_content(tp["path"]))

    user_parts.append({"type": "text", "text": f"\n\nFull SRT:\n\n{srt_text}"})
    user_parts.append({
        "type": "text",
        "text": "\nReturn the JSON object with title, summary, keypoints, and keywords.",
    })

    log.info("Requesting content description from %s...", llm_model)
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _DESCRIBE_SYSTEM},
            {"role": "user", "content": user_parts},
        ],
        **({"extra_body": extra_body} if extra_body else {}),
    )
    if usage_tracker is not None:
        usage_tracker.record("describe", llm_model, resp)
    description = json_repair.loads(resp.choices[0].message.content)
    log.info("Description generated: %s", description.get("title", ""))
    return description
