"""Process- and thread-safe locks for shared Lean caches."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from .errors import LeanTimeoutError

__all__ = ["async_exclusive_file_lock", "exclusive_file_lock"]

_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[Path, threading.Lock] = {}


def _local_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(resolved, threading.Lock())


@contextmanager
def exclusive_file_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Serialize work across threads and Unix processes using ``flock``."""
    deadline = None if timeout is None else time.monotonic() + timeout
    local_lock = _local_lock(path)
    if not local_lock.acquire(timeout=-1 if timeout is None else max(0, timeout)):
        raise LeanTimeoutError(f"Timed out waiting for Lean cache lock: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as lock_file:
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - guarded by platform support
                raise RuntimeError("Lean tooling requires Unix file locking") from exc

            if deadline is None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            else:
                while True:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise LeanTimeoutError(f"Timed out waiting for Lean cache lock: {path}") from None
                        time.sleep(min(0.05, remaining))
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        local_lock.release()


@asynccontextmanager
async def async_exclusive_file_lock(path: Path, *, timeout: float | None = None) -> AsyncIterator[None]:
    """Acquire the same cache lock without blocking the event loop.

    Cancellation while waiting never leaves a background worker acquiring an
    abandoned lock. Synchronous callers, other loops, and processes all share
    the same exclusion.
    """
    deadline = None if timeout is None else time.monotonic() + timeout

    async def wait() -> None:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise LeanTimeoutError(f"Timed out waiting for Lean cache lock: {path}")
        await asyncio.sleep(0.05 if remaining is None else min(0.05, remaining))

    local_lock = _local_lock(path)
    while not local_lock.acquire(blocking=False):
        await wait()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as lock_file:
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - guarded by platform support
                raise RuntimeError("Lean tooling requires Unix file locking") from exc

            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await wait()
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        local_lock.release()
