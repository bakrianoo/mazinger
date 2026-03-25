"""Shared helper functions used across multiple pipeline stages."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Filename helpers
# ---------------------------------------------------------------------------

def sanitize_filename(title: str) -> str:
    """Normalise a human-readable title into a filesystem-safe slug."""
    from slugify import slugify
    return slugify(title, allow_unicode=True)


# ---------------------------------------------------------------------------
#  Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count (~3 chars per token, conservative for multilingual)."""
    return len(text) // 3


# ---------------------------------------------------------------------------
#  Image / vision helpers (OpenAI multimodal API)
# ---------------------------------------------------------------------------

def image_to_base64(path: str) -> str:
    """Read an image and return its base-64 encoded string."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def make_image_content(path: str, detail: str = "low") -> dict:
    """Build an OpenAI vision-compatible ``image_url`` content block."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_to_base64(path)}",
            "detail": detail,
        },
    }


# ---------------------------------------------------------------------------
#  JSON persistence
# ---------------------------------------------------------------------------

def save_json(data: object, path: str) -> None:
    """Write *data* as pretty-printed JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_json(path: str) -> object:
    """Read and parse a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
#  Audio duration via ffprobe
# ---------------------------------------------------------------------------

def get_audio_duration(path: str) -> float:
    """Return the duration of an audio file in seconds (requires ``ffprobe``)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# ---------------------------------------------------------------------------
#  LLM Usage Tracking
# ---------------------------------------------------------------------------

class LLMUsageTracker:
    """Accumulates LLM token usage across pipeline stages.

    Each call to :meth:`record` stores input/output token counts with
    a stage label and model name.  :meth:`report` returns a formatted
    summary string.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, stage: str, model: str, response: object) -> None:
        """Extract usage from an OpenAI response and log it.

        Parameters:
            stage:    Pipeline stage name (e.g. ``"thumbnails"``, ``"translate"``).
            model:    Model identifier used for the call.
            response: The ``ChatCompletion`` response object.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        entry = {
            "stage": stage,
            "model": model,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        self.records.append(entry)
        log.info(
            "LLM usage [%s] model=%s  in=%d  out=%d",
            stage, model, entry["input_tokens"], entry["output_tokens"],
        )

    # ------------------------------------------------------------------

    def summary_by_stage(self) -> dict[str, dict]:
        """Aggregate totals grouped by stage.

        Returns:
            ``{stage: {"model": str, "calls": int, "input_tokens": int,
            "output_tokens": int}}``
        """
        agg: dict[str, dict] = {}
        for r in self.records:
            s = r["stage"]
            if s not in agg:
                agg[s] = {"model": r["model"], "calls": 0,
                           "input_tokens": 0, "output_tokens": 0}
            agg[s]["calls"] += 1
            agg[s]["input_tokens"] += r["input_tokens"]
            agg[s]["output_tokens"] += r["output_tokens"]
        return agg

    @property
    def total_input(self) -> int:
        return sum(r["input_tokens"] for r in self.records)

    @property
    def total_output(self) -> int:
        return sum(r["output_tokens"] for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def report(self) -> str:
        """Return a human-readable usage report string."""
        if not self.records:
            return "LLM Usage: (no calls recorded)"
        lines = ["", "═══ LLM Usage Report ═══"]
        for stage, data in self.summary_by_stage().items():
            lines.append(
                f"  {stage:<16s}  model={data['model']:<28s}  "
                f"calls={data['calls']}  "
                f"in={data['input_tokens']:>8,}  "
                f"out={data['output_tokens']:>7,}"
            )
        lines.append(f"  {'─' * 72}")
        lines.append(
            f"  {'TOTAL':<16s}  {'':28s}  "
            f"calls={len(self.records)}  "
            f"in={self.total_input:>8,}  "
            f"out={self.total_output:>7,}"
        )
        lines.append("═" * 26)
        return "\n".join(lines)
