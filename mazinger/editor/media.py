"""Original-audio clips for the Editor's players, cut on demand and cached.

Opening a row plays the source audio under that chunk.  Decoding a two-hour
MP3 to cut it would be slow, so each clip is cut with an input seek (``-ss``
before ``-i``) into a small mono Opus file under
``lang/<language>/editor/cache/orig_<start>_<end>.ogg``.  The cache is capped
in size and evicts the least recently used clips; neighbouring rows can be
pre-cut in the background so stepping through rows is instant.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable

log = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 200 * 1024 * 1024
CLIP_BITRATE = "32k"


class ClipCache:
    """Cut and cache clips of one source audio file.

    Parameters:
        audio_path: The project's source audio.
        cache_dir:  Where clips are kept (``editor/cache``).
        max_bytes:  Size cap; least recently used clips are evicted beyond it.
    """

    def __init__(self, audio_path: str, cache_dir: str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.audio_path = audio_path
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    def path_for(self, start: float, end: float) -> str:
        return os.path.join(
            self.cache_dir, f"orig_{round(start * 1000)}_{round(end * 1000)}.ogg",
        )

    def clip(self, start: float, end: float) -> str:
        """Return the path of the clip for ``[start, end]``, cutting it if needed."""
        if end <= start:
            raise ValueError(f"Empty clip range: {start}–{end}")
        path = self.path_for(start, end)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            try:
                os.utime(path)  # mark as recently used
            except OSError:
                pass
            return path

        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part.ogg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                    "-i", self.audio_path,
                    "-vn", "-ac", "1", "-c:a", "libopus", "-b:a", CLIP_BITRATE,
                    tmp,
                ],
                capture_output=True, check=True,
            )
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.evict(keep=path)
        return path

    def prefetch(self, ranges: Iterable[tuple[float, float]]) -> Future:
        """Cut *ranges* in a background thread (e.g. the neighbouring rows)."""
        ranges = [(s, e) for s, e in ranges if e > s]
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-prefetch")
        return self._pool.submit(self._prefetch, ranges)

    def _prefetch(self, ranges: list[tuple[float, float]]) -> None:
        for s, e in ranges:
            try:
                self.clip(s, e)
            except Exception as exc:  # noqa: BLE001 — a failed pre-cut is retried on open
                log.debug("Pre-cutting clip %.2f–%.2f failed: %s", s, e, exc)

    def evict(self, keep: str | None = None) -> int:
        """Remove least recently used clips until the cache fits; return how many."""
        with self._lock:
            try:
                names = [n for n in os.listdir(self.cache_dir)
                         if n.startswith("orig_") and n.endswith(".ogg") and ".part." not in n]
            except FileNotFoundError:
                return 0
            files = []
            for n in names:
                p = os.path.join(self.cache_dir, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
            total = sum(size for _, size, _ in files)
            removed = 0
            for _, size, p in sorted(files):
                if total <= self.max_bytes:
                    break
                if p == keep:
                    continue
                try:
                    os.remove(p)
                except OSError:
                    continue
                total -= size
                removed += 1
            return removed

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


def clip_cache_for(session, max_bytes: int = DEFAULT_MAX_BYTES) -> ClipCache:
    """The clip cache of an Editor session."""
    return ClipCache(
        session.proj.audio, os.path.join(session.proj.editor_dir, "cache"), max_bytes,
    )


def neighbour_ranges(session, chunk_id: str, radius: int = 2) -> list[tuple[float, float]]:
    """Time ranges of the chunks around *chunk_id*, nearest first."""
    i = session.index(chunk_id)
    out = []
    for d in range(1, radius + 1):
        for j in (i + d, i - d):
            if 0 <= j < len(session.chunks):
                c = session.chunks[j]
                out.append((c.start, c.end))
    return out
