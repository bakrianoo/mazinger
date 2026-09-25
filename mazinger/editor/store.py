"""Persistence of an Editor session: ``session.json`` + ``changes.jsonl``.

``session.json`` is a full snapshot.  Every edit after it appends one line to
``changes.jsonl``, so saving an edit costs one small write however long the
video is.  :meth:`SessionStore.write_snapshot` merges the log back into a new
snapshot (the session does this every :data:`MERGE_EVERY` changes, on
assemble and on load).

Crash safety:

- The snapshot is written to a temp file and renamed into place.
- Each log record carries the session ``rev`` it produced.  Records at or
  below the snapshot's ``rev`` are skipped on load, so a crash between
  writing a snapshot and truncating the log replays nothing twice.
- A torn last line (a crash mid-append) is ignored.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

SESSION_VERSION = 1

# Merge the change log into the snapshot after this many appended records.
MERGE_EVERY = 200

SNAPSHOT_NAME = "session.json"
LOG_NAME = "changes.jsonl"


class SessionStore:
    """Reads and writes the session files inside ``lang/<language>/editor/``."""

    def __init__(self, editor_dir: str) -> None:
        self.dir = editor_dir
        self.snapshot_path = os.path.join(editor_dir, SNAPSHOT_NAME)
        self.log_path = os.path.join(editor_dir, LOG_NAME)
        # Records appended since the last snapshot.
        self.pending = 0

    def exists(self) -> bool:
        return os.path.isfile(self.snapshot_path)

    # ------------------------------------------------------------------

    def load(self) -> tuple[dict, list[dict]]:
        """Return ``(snapshot, records)``: the records still to replay, in order.

        Raises:
            FileNotFoundError: if there is no snapshot.
            ValueError: if the snapshot is unreadable or from a newer Mazinger.
        """
        try:
            with open(self.snapshot_path, encoding="utf-8") as fh:
                snapshot = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unreadable Editor session {self.snapshot_path}: {exc}") from exc
        if not isinstance(snapshot, dict):
            raise ValueError(f"Malformed Editor session {self.snapshot_path}")
        if snapshot.get("version", 0) > SESSION_VERSION:
            raise ValueError(
                f"Editor session {self.snapshot_path} is version {snapshot.get('version')}; "
                f"this Mazinger understands up to {SESSION_VERSION}. Update Mazinger to open it."
            )

        base_rev = snapshot.get("rev", 0)
        records = [r for r in self._read_log() if r.get("rev", 0) > base_rev]
        self.pending = len(records)
        return snapshot, records

    def _read_log(self) -> list[dict]:
        if not os.path.isfile(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        records: list[dict] = []
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                is_last = not any(rest.strip() for rest in lines[n + 1:])
                if is_last:
                    log.warning("Ignoring a torn last line in %s", self.log_path)
                else:
                    log.error(
                        "Corrupt line %d in %s — later changes are not replayed",
                        n + 1, self.log_path,
                    )
                break
            records.append(rec)
        return records

    # ------------------------------------------------------------------

    def append(self, record: dict) -> None:
        """Append one change record to the log."""
        os.makedirs(self.dir, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.pending += 1

    def write_snapshot(self, snapshot: dict) -> None:
        """Atomically replace the snapshot, then empty the change log."""
        os.makedirs(self.dir, exist_ok=True)
        data = {"version": SESSION_VERSION, **snapshot}
        tmp = self.snapshot_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.snapshot_path)
        # Records up to snapshot["rev"] are now in the snapshot; a crash
        # before this truncation is harmless (see the module docstring).
        with open(self.log_path, "w", encoding="utf-8"):
            pass
        self.pending = 0
