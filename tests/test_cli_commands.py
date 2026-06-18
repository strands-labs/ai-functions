"""Tests for the non-TUI ``ai_functions`` command bodies — ``notify`` / ``submit``.

The command functions in :mod:`ai_functions.cli.commands` open their own
short-lived client via :func:`ai_functions.connect`. These tests swap that
``connect`` for a fake async context manager yielding a stub client, so
the verbs can be exercised without a running coordinator. Stdout/stderr
are captured via ``capsys`` and the integer exit code is asserted
directly (the Typer wiring in ``app.py`` just forwards it).
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pytest

from ai_functions.cli import commands
from ai_functions.network import CoordinatorClient
from ai_functions.runtime.errors import ThreadNotFoundError
from ai_functions.types import (
    Event,
    EventId,
    InputShape,
    ThreadId,
    ThreadInfo,
    ThreadStatus,
    TokenUsage,
    TokenUsageEvent,
    WorkerId,
)

_TID = ThreadId("thread-test")


class _FakeClient:
    """Records cross-thread calls and returns canned values."""

    def __init__(
        self,
        *,
        info: ThreadInfo | None = None,
        result: object = "ok",
        missing: bool = False,
        prior_events: list[Event] | None = None,
        cycle_events: list[Event] | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        self._info = info
        self._result = result
        self._missing = missing
        # The log starts with `prior_events`; calling `submit` appends
        # `cycle_events`, mirroring how a real cycle emits events during
        # execution (after the caller has watermarked the log).
        self._log: list[Event] = list(prior_events or [])
        self._cycle_events = list(cycle_events or [])
        self._submit_error = submit_error
        self.notified: list[tuple[ThreadId, str]] = []
        self.submitted: list[tuple[ThreadId, str]] = []
        self.cancelled: list[ThreadId] = []

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        if self._missing or self._info is None:
            raise ThreadNotFoundError(thread_id)
        return self._info

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        if self._missing:
            raise ThreadNotFoundError(thread_id)
        self.notified.append((thread_id, text))

    async def submit(self, thread_id: ThreadId, text: str) -> object:
        if self._missing:
            raise ThreadNotFoundError(thread_id)
        if self._submit_error is not None:
            raise self._submit_error
        self.submitted.append((thread_id, text))
        self._log.extend(self._cycle_events)
        return self._result

    async def cancel(self, thread_id: ThreadId) -> None:
        self.cancelled.append(thread_id)

    async def get_events(self, thread_id: ThreadId, since_id: EventId | None = None) -> list[Event]:
        del thread_id
        if since_id is None:
            return list(self._log)
        # Mirror the coordinator: events strictly after the cursor id.
        ids = [getattr(e, "id", None) for e in self._log]
        if since_id not in ids:
            return list(self._log)
        cut = ids.index(since_id) + 1
        return list(self._log[cut:])


def _info(shape: InputShape = InputShape.STR_PROMPT) -> ThreadInfo:
    return ThreadInfo(
        thread_id=_TID,
        worker_id=WorkerId("worker-1"),
        thread_name="t",
        input_shape=shape,
        status=ThreadStatus.RUNNING,
        parent_id=None,
    )


def _patch_connect(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    @asynccontextmanager
    async def _fake_connect(url: str | None = None) -> AsyncGenerator[CoordinatorClient]:
        del url
        yield cast("CoordinatorClient", client)

    monkeypatch.setattr(commands, "connect", _fake_connect)


# ── notify ────────────────────────────────────────────────────────────────


def test_notify_delivers_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ai-functions notify`` calls ``notify`` once and exits 0."""
    client = _FakeClient(info=_info())
    _patch_connect(monkeypatch, client)
    code = commands.notify(_TID, "hello")
    assert code == 0
    assert client.notified == [(_TID, "hello")]


def test_notify_thread_not_found(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing thread yields exit 1 and a clean error (no traceback)."""
    client = _FakeClient(missing=True)
    _patch_connect(monkeypatch, client)
    code = commands.notify(_TID, "hello")
    assert code == 1
    assert "not found" in capsys.readouterr().err


# ── submit ──────────────────────────────────────────────────────────────────


def test_submit_runs_cycle_and_prints_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ai-functions submit`` runs one cycle and prints the string result."""
    client = _FakeClient(info=_info(), result="the answer")
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "the question")
    assert code == 0
    assert client.submitted == [(_TID, "the question")]
    assert "the answer" in capsys.readouterr().out


def test_submit_reprs_non_string_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A non-string cycle result is printed via ``repr``."""
    client = _FakeClient(info=_info(), result={"count": 42})
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q")
    assert code == 0
    assert "{'count': 42}" in capsys.readouterr().out


def test_submit_rejects_non_str_prompt_shape(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-``STR_PROMPT`` thread is refused with a clear error, no submit."""
    client = _FakeClient(info=_info(InputShape.STRUCTURED))
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q")
    assert code == 1
    err = capsys.readouterr().err
    assert "structured" in err
    assert "str_prompt" in err
    assert client.submitted == []


def test_submit_thread_not_found(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing thread is reported before any cycle starts."""
    client = _FakeClient(missing=True)
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q")
    assert code == 1
    assert "not found" in capsys.readouterr().err
    assert client.submitted == []


# ── submit --json ────────────────────────────────────────────────────────────


def _token_event(*, input_tokens: int = 0, output_tokens: int = 0, timestamp: float) -> TokenUsageEvent:
    return TokenUsageEvent(
        thread_id=_TID,
        timestamp=timestamp,
        token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_submit_json_includes_result_and_summed_token_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` emits a parseable object with summed token usage."""
    cycle_events: list[Event] = [
        _token_event(input_tokens=100, output_tokens=10, timestamp=1000.0),
        _token_event(input_tokens=50, output_tokens=5, timestamp=1002.0),
    ]
    client = _FakeClient(info=_info(), result="the answer", cycle_events=cycle_events)
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q", as_json=True)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["result"] == "the answer"
    assert payload["token_usage"]["input_tokens"] == 150
    assert payload["token_usage"]["output_tokens"] == 15
    assert payload["token_usage"]["total_tokens"] == 165
    assert payload["timing"]["duration_seconds"] == pytest.approx(2.0)


def test_submit_json_scopes_events_to_this_cycle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Token usage from before the watermark is not counted."""
    prior = _token_event(input_tokens=999, output_tokens=999, timestamp=1.0)
    this_cycle = _token_event(input_tokens=7, output_tokens=3, timestamp=2.0)
    client = _FakeClient(info=_info(), result="ok", prior_events=[prior], cycle_events=[this_cycle])
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q", as_json=True)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # The watermark is `prior.id`, so only `this_cycle` is summed.
    assert payload["token_usage"]["input_tokens"] == 7
    assert payload["token_usage"]["output_tokens"] == 3


def test_submit_json_dict_result_embedded_not_repr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A JSON-safe dict result is embedded as JSON, not repr'd."""
    client = _FakeClient(info=_info(), result={"city": "Kyoto"})
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q", as_json=True)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == {"city": "Kyoto"}
    assert payload["token_usage"]["total_tokens"] == 0
    assert payload["timing"] is None


def test_submit_json_still_rejects_non_str_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` does not relax the input-shape gate."""
    client = _FakeClient(info=_info(InputShape.NO_ARGS))
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q", as_json=True)
    assert code == 1
    assert "no_args" in capsys.readouterr().err
    assert client.submitted == []


# ── graceful error translation ───────────────────────────────────────────────


def test_submit_connection_error_is_clean(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """An ``OSError`` reaching the coordinator exits 2, not a traceback."""

    @asynccontextmanager
    async def _failing_connect(url: str | None = None) -> AsyncGenerator[CoordinatorClient]:
        del url
        raise OSError("connection refused")
        yield  # pragma: no cover  -- unreachable; makes this an async generator

    monkeypatch.setattr(commands, "connect", _failing_connect)
    code = commands.submit(_TID, "q")
    assert code == 2
    assert "could not reach coordinator" in capsys.readouterr().err


def test_submit_failed_cycle_is_clean(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A cycle that raises is reported as a clean error, exit 1."""
    client = _FakeClient(info=_info(), submit_error=ValueError("model exploded"))
    _patch_connect(monkeypatch, client)
    code = commands.submit(_TID, "q")
    assert code == 1
    err = capsys.readouterr().err
    assert "cycle failed" in err
    assert "model exploded" in err


def test_notify_connection_error_is_clean(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """``notify`` also translates a connection ``OSError`` to exit 2."""

    @asynccontextmanager
    async def _failing_connect(url: str | None = None) -> AsyncGenerator[CoordinatorClient]:
        del url
        raise OSError("connection refused")
        yield  # pragma: no cover  -- unreachable; makes this an async generator

    monkeypatch.setattr(commands, "connect", _failing_connect)
    code = commands.notify(_TID, "hi")
    assert code == 2
    assert "could not reach coordinator" in capsys.readouterr().err


# ── main(): usage-error translation ──────────────────────────────────────────


def test_main_missing_argument_exits_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing required argument is a clean usage error (exit 1)."""
    from ai_functions.cli import main

    monkeypatch.setattr(sys, "argv", ["ai_functions", "submit"])
    code = main()
    assert code == 1
    # Click prints a usage hint; the key point is no traceback escaped.
    assert "Usage:" in capsys.readouterr().err


def test_main_unknown_command_exits_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown command is a clean usage error (exit 1)."""
    from ai_functions.cli import main

    monkeypatch.setattr(sys, "argv", ["ai_functions", "bogus"])
    code = main()
    assert code == 1
    assert "No such command" in capsys.readouterr().err


def test_main_propagates_command_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command's nonzero ``typer.Exit`` code reaches the process exit code.

    With ``standalone_mode=False`` Click returns the code rather than
    raising ``SystemExit``; ``main`` must honour that return value.
    ``ps`` with no discoverable coordinator returns 2, so it exercises
    the return-value path (a regression guard: a bare ``main`` that
    ignored the return value would wrongly report 0).
    """
    from ai_functions.cli import main

    # Point discovery at an empty dir + no env override, so no coordinator
    # is found and `ps` exits 2.
    monkeypatch.delenv("AI_FUNCTIONS_COORDINATOR_URL", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ai_functions", "ps"])
    code = main()
    assert code == 2
    assert "coordinator" in capsys.readouterr().err.lower()
