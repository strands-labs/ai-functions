"""Cache locks shared by synchronous and asynchronous Lean consumers."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

__all__ = ["async_exclusive_file_lock", "exclusive_file_lock"]

@contextmanager
def exclusive_file_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]: ...

@asynccontextmanager
async def async_exclusive_file_lock(path: Path, *, timeout: float | None = None) -> AsyncIterator[None]: ...
