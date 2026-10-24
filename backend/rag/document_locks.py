"""Single-worker, per-document mutation exclusion."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator


class DocumentBusyError(RuntimeError):
    pass


class DocumentLockRegistry:
    def __init__(self) -> None:
        self._guard = Lock()
        self._locks: dict[int, Lock] = {}

    @contextmanager
    def acquire(self, document_id: int) -> Iterator[None]:
        with self._guard:
            lock = self._locks.setdefault(int(document_id), Lock())
        if not lock.acquire(blocking=False):
            raise DocumentBusyError(f"document {document_id} is busy")
        try:
            yield
        finally:
            lock.release()


document_locks = DocumentLockRegistry()
