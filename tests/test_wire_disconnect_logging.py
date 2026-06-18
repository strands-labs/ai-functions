"""A dropped wire transport must leave a legible reason behind.

When the WebSocket to the coordinator closes mid-run, the ``websockets``
``ConnectionClosed`` exception carries the close code and reason (1009
"message too big", 1011 keepalive timeout, 1006 abnormal, …). The reader
loop used to discard it with a bare ``except Exception: break``, so the
single most diagnostic fact about a drop was lost. These tests pin the
corrected behavior: the close is summarised and logged — at ERROR for an
abnormal close, at DEBUG for a clean going-away.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from ai_functions.network.channel import WireChannel, _describe_transport_close


def test_describe_too_big_close_is_abnormal() -> None:
    """A 1009 close is summarised with code + reason and flagged abnormal."""
    exc = ConnectionClosedError(Close(1009, "message too big"), None)
    description, is_clean = _describe_transport_close(exc)
    assert "1009" in description
    assert "message too big" in description
    assert is_clean is False


def test_describe_keepalive_close_is_abnormal() -> None:
    """A 1011 keepalive-timeout close is flagged abnormal."""
    exc = ConnectionClosedError(Close(1011, "keepalive ping timeout"), None)
    description, is_clean = _describe_transport_close(exc)
    assert "1011" in description
    assert is_clean is False


def test_describe_going_away_is_clean() -> None:
    """A 1001 going-away close is clean (expected shutdown)."""
    exc = ConnectionClosedOK(Close(1001, "going away"), None)
    _description, is_clean = _describe_transport_close(exc)
    assert is_clean is True


def test_describe_abnormal_no_frame() -> None:
    """A 1006-style close with no frame falls back to the exception text."""
    exc = ConnectionClosedError(None, None)
    description, is_clean = _describe_transport_close(exc)
    assert is_clean is False
    assert description  # non-empty


def test_describe_non_websockets_error() -> None:
    """A raw transport error (not a ConnectionClosed) is reported verbatim."""
    description, is_clean = _describe_transport_close(OSError("broken pipe"))
    assert "broken pipe" in description
    assert is_clean is False


class _DropTransport:
    """Transport whose ``recv`` raises a given close exception once started."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def recv(self) -> str:
        raise self._exc

    async def close(self) -> None:
        return None


async def test_read_loop_logs_abnormal_close(caplog: pytest.LogCaptureFixture) -> None:
    """An abnormal transport close is logged at ERROR with the code."""
    transport = _DropTransport(ConnectionClosedError(Close(1009, "message too big"), None))
    channel = WireChannel(transport)
    with caplog.at_level(logging.ERROR, logger="ai_functions.network.channel"):
        async with channel:
            # The reader task runs eagerly; give it a turn to hit recv().
            await asyncio.sleep(0.01)
    assert any("1009" in rec.message and rec.levelno == logging.ERROR for rec in caplog.records)


async def test_read_loop_clean_close_not_error(caplog: pytest.LogCaptureFixture) -> None:
    """A clean going-away close does not log at ERROR."""
    transport = _DropTransport(ConnectionClosedOK(Close(1001, "going away"), None))
    channel = WireChannel(transport)
    with caplog.at_level(logging.DEBUG, logger="ai_functions.network.channel"):
        async with channel:
            await asyncio.sleep(0.01)
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)
