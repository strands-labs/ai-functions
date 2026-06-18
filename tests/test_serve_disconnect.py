"""``aserve`` must fail loudly when the coordinator connection drops.

The host's wait loop polls ``handle.status()`` once a second. Before the
fix, any exception from that poll — including ``ConnectionClosedError`` when
the websocket to the coordinator dropped mid-run — was caught and treated as
a terminal status, so the process exited 0 with nothing logged. These tests
pin the corrected behavior: a dropped connection propagates out of
``_wait_for_shutdown`` (and thus ``aserve``), while a real shutdown signal
still wins as a clean exit.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_functions.network import ConnectionClosedError
from ai_functions.serve import _wait_for_shutdown, _wait_until_terminal
from ai_functions.types import ThreadStatus


class _FakeHandle:
    """Minimal stand-in for ThreadHandle exposing ``status``.

    ``status`` returns ``statuses`` in order, then raises ``raise_after`` (if
    set) on the next call. A ``None`` entry models "still running".
    """

    def __init__(
        self,
        *,
        statuses: list[ThreadStatus | None] | None = None,
        raise_after: Exception | None = None,
    ) -> None:
        self._statuses = list(statuses or [])
        self._raise_after = raise_after
        self.status_calls = 0

    async def status(self) -> ThreadStatus:
        self.status_calls += 1
        if self._statuses:
            nxt = self._statuses.pop(0)
            if nxt is not None:
                return nxt
            return ThreadStatus.RUNNING
        if self._raise_after is not None:
            raise self._raise_after
        return ThreadStatus.RUNNING


async def test_wait_until_terminal_propagates_connection_drop() -> None:
    """A status-poll failure is no longer swallowed."""
    handle = _FakeHandle(raise_after=ConnectionClosedError("channel closed"))
    with pytest.raises(ConnectionClosedError):
        await _wait_until_terminal(handle)  # type: ignore[arg-type]


async def test_wait_for_shutdown_reraises_connection_drop() -> None:
    """A dropped coordinator connection escapes the wait loop (loud death)."""
    handle = _FakeHandle(raise_after=ConnectionClosedError("channel closed"))
    shutdown = asyncio.Event()
    with pytest.raises(ConnectionClosedError):
        await _wait_for_shutdown(handle, shutdown, None)  # type: ignore[arg-type]


async def test_wait_for_shutdown_signal_wins_clean() -> None:
    """A requested shutdown exits cleanly even though status() would raise."""
    handle = _FakeHandle(raise_after=ConnectionClosedError("channel closed"))
    shutdown = asyncio.Event()
    shutdown.set()  # signal already requested
    # Must return normally — no exception — because shutdown takes precedence.
    await _wait_for_shutdown(handle, shutdown, None)  # type: ignore[arg-type]


async def test_wait_for_shutdown_terminal_status_clean() -> None:
    """A genuinely terminal thread status still ends the loop cleanly."""
    handle = _FakeHandle(statuses=[ThreadStatus.TERMINATED])
    shutdown = asyncio.Event()
    await _wait_for_shutdown(handle, shutdown, None)  # type: ignore[arg-type]
