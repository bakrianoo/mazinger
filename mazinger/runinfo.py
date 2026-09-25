"""Record of the settings a completed dub used (``lang/<language>/run.json``).

Re-doing a single stage later — re-translating or re-dubbing one segment in
the Editor — needs the exact engine, model, voice and assembly settings of the
original run.  Those exist only as arguments to :meth:`MazingerDubber.dub`,
so the pipeline writes them here once the dub completes.

Paths inside the record are stored relative to the project root so a project
folder stays valid after it is moved.  Credentials are never stored: API keys,
tokens and cookie files are either re-entered or read from the environment.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from mazinger.paths import ProjectPaths

log = logging.getLogger(__name__)

RUN_INFO_VERSION = 1

# Key fragments that must never appear in a saved record.  Checked on save so
# a future field added carelessly fails loudly instead of leaking a secret.
_SECRET_FRAGMENTS = ("api_key", "apikey", "token", "password", "secret", "cookie")


def project_relpath(proj: ProjectPaths, path: str | None) -> str | None:
    """Return *path* relative to the project root, or absolute if outside it."""
    if not path:
        return None
    abs_path = os.path.abspath(path)
    root = os.path.abspath(proj.root)
    if os.path.commonpath([abs_path, root]) == root:
        return os.path.relpath(abs_path, root)
    return abs_path


def resolve_project_path(proj: ProjectPaths, path: str | None) -> str | None:
    """Inverse of :func:`project_relpath`."""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(proj.root, path)


def _find_secret_keys(obj: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if any(frag in str(key).lower() for frag in _SECRET_FRAGMENTS):
                found.append(dotted)
            found.extend(_find_secret_keys(value, dotted))
    return found


def save_run_info(proj: ProjectPaths, info: dict) -> str:
    """Write *info* to ``proj.run_info`` atomically and return the path.

    ``version``, ``mazinger_version`` and ``created_at`` are filled in here.

    Raises:
        ValueError: if any key looks like it holds a credential.
    """
    leaked = _find_secret_keys(info)
    if leaked:
        raise ValueError(f"Refusing to save credentials in run.json: {', '.join(leaked)}")

    from mazinger import __version__

    record = {
        "version": RUN_INFO_VERSION,
        "mazinger_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **info,
    }
    os.makedirs(os.path.dirname(proj.run_info), exist_ok=True)
    tmp = proj.run_info + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, proj.run_info)
    log.info("Run settings saved: %s", proj.run_info)
    return proj.run_info


def load_run_info(proj: ProjectPaths) -> dict | None:
    """Return the saved run record, or ``None`` when absent or unreadable.

    Projects dubbed before run records existed have none; callers must then
    ask the user for the settings instead.
    """
    if not os.path.isfile(proj.run_info):
        return None
    try:
        with open(proj.run_info, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Ignoring unreadable run record %s: %s", proj.run_info, exc)
        return None
    if not isinstance(record, dict):
        log.warning("Ignoring malformed run record %s", proj.run_info)
        return None
    if record.get("version", 0) > RUN_INFO_VERSION:
        log.warning(
            "Run record %s is version %s; this Mazinger understands up to %s. "
            "Unknown fields are ignored.",
            proj.run_info, record.get("version"), RUN_INFO_VERSION,
        )
    return record
