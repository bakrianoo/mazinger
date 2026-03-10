"""Extract key-frame thumbnails from a video using LLM-selected timestamps."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

import json_repair
from PIL import Image

from mazinger_dubber.srt import parse_blocks, blocks_to_text
from mazinger_dubber.utils import estimate_tokens, LLMUsageTracker

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Timestamp selection via LLM
# ---------------------------------------------------------------------------

_TIMESTAMP_SYSTEM = """\
You are an expert video analyst. You will receive an SRT subtitle file (or a \
segment of one) from a tutorial / educational video. Your job is to select \
timestamps where extracting a screenshot would add meaningful visual context \
that the subtitles alone cannot convey.

Good reasons to pick a timestamp:
- The speaker shows a code editor, terminal, browser, or UI.
- A diagram, architecture drawing, or slide is presented.
- The speaker demonstrates a live result (running a server, opening a URL).
- A new topic or section begins and the screen likely changed.
- The speaker references something visual ("as you can see here").

Guidelines:
- Pick 10-25 timestamps for a full video, or 5-12 per segment if partial.
- Spread them so every major visual moment is covered.
- Prefer moments a few seconds *after* a topic is mentioned.
- Return ONLY a JSON array with no markdown fences or explanation.

Each element:
  - "timestamp": "HH:MM:SS" or "MM:SS"
  - "seconds": float total seconds
  - "reason": one-line description
"""

_TOKEN_THRESHOLD = 12_000
_BATCH_MINUTES = 5
_OVERLAP_SECONDS = 30


def _request_timestamps(
    client: OpenAI,
    srt_segment: str,
    segment_label: str = "",
    llm_model: str = "gpt-4.1",
    usage_tracker: LLMUsageTracker | None = None,
) -> list[dict]:
    label = f" (segment: {segment_label})" if segment_label else ""
    user_msg = (
        f"Here is the SRT file{label}:\n\n{srt_segment}\n\n"
        "Return the JSON array of timestamps."
    )
    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _TIMESTAMP_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    if usage_tracker is not None:
        usage_tracker.record("thumbnails", llm_model, resp)
    return json_repair.loads(resp.choices[0].message.content)


def _deduplicate(ts_list: list[dict], min_gap: float = 5.0) -> list[dict]:
    if not ts_list:
        return ts_list
    ordered = sorted(ts_list, key=lambda t: float(t["seconds"]))
    result = [ordered[0]]
    for t in ordered[1:]:
        if float(t["seconds"]) - float(result[-1]["seconds"]) >= min_gap:
            result.append(t)
    return result


def select_timestamps(
    srt_text: str,
    client: OpenAI,
    *,
    llm_model: str = "gpt-4.1",
    min_gap: float = 5.0,
    usage_tracker: LLMUsageTracker | None = None,
) -> list[dict]:
    """Analyse an SRT and return a list of timestamps worth capturing.

    For short SRTs the entire text is sent in a single request.  Longer
    transcripts are split into overlapping time-based windows.

    Returns:
        A de-duplicated list of ``{"timestamp", "seconds", "reason"}`` dicts.
    """
    est = estimate_tokens(srt_text)
    log.info("Estimated SRT tokens: ~%d", est)

    if est <= _TOKEN_THRESHOLD:
        timestamps = _request_timestamps(client, srt_text, llm_model=llm_model,
                                         usage_tracker=usage_tracker)
        return _deduplicate(timestamps, min_gap)

    blocks = parse_blocks(srt_text)
    total_end = max(b[2] for b in blocks)
    batch_sec = _BATCH_MINUTES * 60

    all_timestamps: list[dict] = []
    window_start = 0.0
    batch_num = 0

    while window_start < total_end:
        window_end = window_start + batch_sec + _OVERLAP_SECONDS
        batch_blocks = [b for b in blocks if b[2] > window_start and b[1] < window_end]
        if not batch_blocks:
            window_start += batch_sec
            continue

        batch_num += 1
        segment_srt = blocks_to_text(batch_blocks)
        label = f"{batch_blocks[0][1] / 60:.0f}min-{batch_blocks[-1][2] / 60:.0f}min"
        log.info("Batch %d: %s (%d subtitles)", batch_num, label, len(batch_blocks))

        batch_ts = _request_timestamps(client, segment_srt, segment_label=label,
                                         llm_model=llm_model, usage_tracker=usage_tracker)
        all_timestamps.extend(batch_ts)
        window_start += batch_sec

    return _deduplicate(all_timestamps, min_gap)


# ---------------------------------------------------------------------------
#  Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    timestamps: list[dict],
    output_dir: str,
    *,
    max_size: int = 768,
    jpeg_quality: int = 85,
) -> list[dict]:
    """Extract and resize video frames for each timestamp entry.

    Returns:
        A list of dicts, each containing the original timestamp fields plus
        a ``path`` key pointing to the saved JPEG.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    for i, ts in enumerate(timestamps):
        sec = float(ts["seconds"])
        fname = f"thumb_{i:03d}_{sec:.1f}s.jpg"
        out_path = os.path.join(output_dir, fname)
        raw_path = out_path.replace(".jpg", "_raw.png")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(sec), "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", raw_path,
                ],
                check=True,
                capture_output=True,
            )
            img = Image.open(raw_path).convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=jpeg_quality, optimize=True)
            os.remove(raw_path)
            results.append({"path": out_path, **ts})
            log.debug("Extracted %s (%dx%d)", fname, img.size[0], img.size[1])
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"")[:200]
            log.warning("Failed to extract %s: %s", fname, stderr)

    log.info("Extracted %d/%d thumbnails -> %s", len(results), len(timestamps), output_dir)
    return results
