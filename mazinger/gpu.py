"""One GPU job at a time, across Studio's Dub tab and the Editor.

A full dub and an Editor operation (re-dub, re-transcribe, assemble …) each
need most of the VRAM.  Both take :data:`gpu_lock` for the duration of their
work.  Editor operations do not wait: when a dub holds the lock they fail
at once with :class:`GPUBusy`, naming the holder, so the UI can say why.

Components that keep models loaded between jobs (the Editor's resource
manager) register a *releaser*; :func:`release_idle` calls them before a
full dub starts so the dub gets the whole GPU.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Callable, Iterator

log = logging.getLogger(__name__)


class GPUBusy(RuntimeError):
    """The GPU is in use by another job."""

    def __init__(self, holder: str) -> None:
        super().__init__(f"The GPU is busy: {holder} is running. Try again when it finishes.")
        self.holder = holder


class GPULock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.holder: str | None = None

    @contextlib.contextmanager
    def hold(self, owner: str, *, wait: bool = False) -> Iterator[None]:
        """Hold the GPU as *owner* for the ``with`` block.

        Raises:
            GPUBusy: when *wait* is false and another job holds the GPU.
        """
        if not self._lock.acquire(blocking=wait):
            raise GPUBusy(self.holder or "another job")
        self.holder = owner
        try:
            yield
        finally:
            self.holder = None
            self._lock.release()

    def busy(self) -> bool:
        return self._lock.locked()


gpu_lock = GPULock()

_releasers: list[Callable[[], None]] = []


def register_releaser(fn: Callable[[], None]) -> None:
    """Register *fn* to free cached models when a full dub needs the GPU."""
    if fn not in _releasers:
        _releasers.append(fn)


def unregister_releaser(fn: Callable[[], None]) -> None:
    with contextlib.suppress(ValueError):
        _releasers.remove(fn)


def release_idle() -> None:
    """Ask every registered component to free the models it keeps loaded."""
    for fn in list(_releasers):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — freeing is best effort
            log.warning("Could not free cached models: %s", exc)
