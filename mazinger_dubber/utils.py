"""Shared helper functions used across multiple pipeline stages."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess


# ---------------------------------------------------------------------------
#  Filename helpers
# ---------------------------------------------------------------------------

def sanitize_filename(title: str) -> str:
    """Normalise a human-readable title into a filesystem-safe slug."""
    name = title.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


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
