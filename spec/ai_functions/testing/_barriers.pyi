"""Shared barrier registry used by ``ScriptedModel`` and ``RuntimeHarness``.

Private to ``ai_functions.testing``. The contextvar is set by
``RuntimeHarness.__aenter__`` so scripted models spawned inside the
``async with`` block resolve barrier names against the active harness.
"""

import asyncio
from contextvars import ContextVar


class BarrierRegistry:
    """Named ``asyncio.Event`` store shared by a harness and its scripted models.

    Creates an empty registry.
    """

    def __init__(self) -> None: ...

    def get(self, name: str) -> asyncio.Event:
        """Return the (lazily created) event for ``name``.

        Args:
            name: Barrier name.

        Returns:
            The ``asyncio.Event`` for ``name``; created on first access.
        """
        ...

    def release(self, name: str) -> None:
        """Set the event for ``name``, unblocking all current and future awaits.

        Args:
            name: Barrier name.
        """
        ...

    def release_all(self) -> None:
        """Set every known event so pending awaiters unwind on harness teardown."""
        ...


current_registry: ContextVar[BarrierRegistry | None]
"""Contextvar carrying the active harness's registry; ``None`` outside a
harness."""


async def await_barrier(name: str) -> None:
    """Block until the named barrier is released by the active harness.

    Args:
        name: Barrier name.

    Raises:
        RuntimeError: No ``RuntimeHarness`` is active in this context.
    """
    ...
