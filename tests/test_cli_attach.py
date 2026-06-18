"""Behavioural tests for ``ai-functions attach`` — the Textual TUI.

Covers the event formatter's two render modes and the ``_AttachApp``
buffer / view-toggle behaviour that backs ``Ctrl+O``. The app is driven
headlessly through Textual's ``run_test`` pilot against a fake
coordinator client, so no server or terminal is required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from rich.console import Console, RenderableType

from ai_functions.cli.attach import _AttachApp
from ai_functions.cli.events import (
    STRUCTURED_OUTPUT_TOOL,
    filter_events_full,
    format_event,
    format_event_full,
)
from ai_functions.network import CoordinatorClient
from ai_functions.types import (
    Event,
    InputShape,
    MessageAssistantTokenEvent,
    MessageUserEvent,
    StartedEvent,
    ThreadId,
    ThreadInfo,
    ThreadStatus,
    TokenUsage,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkerId,
)

_TID = ThreadId("thread-test")


def _render_to_text(renderable: RenderableType | None) -> str:
    """Flatten a Rich renderable to plain text (empty string for None)."""
    if renderable is None:
        return ""
    console = Console(file=None, color_system=None, force_terminal=False, width=10_000)
    with console.capture() as capture:
        console.print(renderable, end="")
    return capture.get()


# ── format_event_full: full content, conversation filter ──────────────────


def test_format_event_full_keeps_full_user_text() -> None:
    """A long user message is rendered without truncation."""
    long_text = "x" * 5_000
    out = _render_to_text(format_event_full(MessageUserEvent(thread_id=_TID, text=long_text)))
    assert long_text in out


def test_format_event_truncates_what_full_does_not() -> None:
    """The compact formatter truncates the same message the full one keeps."""
    long_text = "y" * 5_000
    event = MessageUserEvent(thread_id=_TID, text=long_text)
    compact = _render_to_text(format_event(event))
    full = _render_to_text(format_event_full(event))
    assert "..." in compact
    assert len(full) > len(compact)
    assert long_text in full


def test_format_event_full_includes_tool_result_body() -> None:
    """Tool result content is shown in full in the conversation view."""
    event = ToolResultEvent(
        thread_id=_TID,
        tool_use_id="tu-1",
        status="success",
        content=[{"text": "the full tool output here"}],
    )
    out = _render_to_text(format_event_full(event))
    assert "the full tool output here" in out


def test_filter_events_full_skips_non_conversation_kinds() -> None:
    """Lifecycle / token-usage / streaming events are not in the transcript."""
    assert filter_events_full(StartedEvent(thread_id=_TID, thread_name="t")) is False
    assert filter_events_full(MessageAssistantTokenEvent(thread_id=_TID, text="x")) is False
    assert filter_events_full(TokenUsageEvent(thread_id=_TID, token_usage=TokenUsage(input_tokens=1))) is False


def test_format_event_full_truncates_tool_noise() -> None:
    """Tool-call args and tool-result bodies are abbreviated, not dumped."""
    big = "z" * 5_000
    call = ToolCallEvent(thread_id=_TID, tool_use_id="tu-1", tool_name="write", arguments={"content": big})
    result = ToolResultEvent(thread_id=_TID, tool_use_id="tu-1", status="success", content=[{"text": big}])
    call_out = _render_to_text(format_event_full(call))
    result_out = _render_to_text(format_event_full(result))
    assert "..." in call_out
    assert big not in call_out
    assert "..." in result_out
    assert big not in result_out


def test_filter_events_full_hides_structured_output_call() -> None:
    """The FinalAnswer wrapper call is hidden; a normal tool call is not."""
    final = ToolCallEvent(
        thread_id=_TID,
        tool_use_id="tu-fa",
        tool_name=STRUCTURED_OUTPUT_TOOL,
        arguments={"answer": 42},
    )
    normal = ToolCallEvent(thread_id=_TID, tool_use_id="tu-2", tool_name="read", arguments={})
    assert filter_events_full(final) is False
    assert filter_events_full(normal) is True


# ── _AttachApp: buffer + Ctrl+O toggle ────────────────────────────────────


class _FakeClient:
    """Minimal stand-in for ``CoordinatorClient`` used by ``_AttachApp``."""

    def __init__(self, events: list[Event]) -> None:
        self._events = events

    async def get_events(self, thread_id: ThreadId) -> list[Event]:
        del thread_id
        return list(self._events)

    def on(self, callback: Callable[[Event], None], *, thread_id: ThreadId) -> object:
        del callback, thread_id

        class _Sub:
            def unsubscribe(self) -> None:
                return None

        return _Sub()


def _info(shape: InputShape = InputShape.STR_PROMPT) -> ThreadInfo:
    return ThreadInfo(
        thread_id=_TID,
        worker_id=WorkerId("worker-1"),
        thread_name="t",
        input_shape=shape,
        status=ThreadStatus.RUNNING,
        parent_id=None,
    )


def _app(events: list[Event], shape: InputShape = InputShape.STR_PROMPT) -> _AttachApp:
    """Build an ``_AttachApp`` against a fake client (cast for the checker)."""
    client = cast("CoordinatorClient", _FakeClient(events))
    return _AttachApp(client, _info(shape))


def _mixed_events() -> list[Event]:
    return [
        StartedEvent(thread_id=_TID, thread_name="t"),
        MessageUserEvent(thread_id=_TID, text="hello"),
        ToolCallEvent(thread_id=_TID, tool_use_id="tu-1", tool_name="read", arguments={}),
        TokenUsageEvent(thread_id=_TID, token_usage=TokenUsage(input_tokens=3)),
    ]


async def test_buffer_holds_all_events_regardless_of_view() -> None:
    """Every replayed event is buffered even if not shown in the view."""
    app = _app(_mixed_events())
    async with app.run_test():
        # 4 events buffered, but only some render in the default view.
        assert len(app._events) == 4  # pyright: ignore[reportPrivateUsage]


async def test_toggle_view_switches_mode_and_is_reversible() -> None:
    """Conversation is the default; Ctrl+O flips and toggling back restores it."""
    app = _app(_mixed_events())
    async with app.run_test() as pilot:
        assert app._conversation_view is True  # pyright: ignore[reportPrivateUsage]
        await pilot.press("ctrl+o")
        assert app._conversation_view is False  # pyright: ignore[reportPrivateUsage]
        # Buffer is untouched by the re-render.
        assert len(app._events) == 4  # pyright: ignore[reportPrivateUsage]
        await pilot.press("ctrl+o")
        assert app._conversation_view is True  # pyright: ignore[reportPrivateUsage]
        assert len(app._events) == 4  # pyright: ignore[reportPrivateUsage]


async def test_conversation_view_hides_structured_output_call_and_result() -> None:
    """In conversation view, both the FinalAnswer call and its result vanish."""
    events: list[Event] = [
        MessageUserEvent(thread_id=_TID, text="question"),
        ToolCallEvent(
            thread_id=_TID,
            tool_use_id="tu-fa",
            tool_name=STRUCTURED_OUTPUT_TOOL,
            arguments={"answer": 42},
        ),
        ToolResultEvent(
            thread_id=_TID,
            tool_use_id="tu-fa",
            status="success",
            content=[{"text": "Successfully validated FinalAnswer structured output"}],
        ),
    ]
    app = _app(events)
    async with app.run_test():
        # Conversation is the default view.
        assert app._conversation_view is True  # pyright: ignore[reportPrivateUsage]
        log = app.query_one("#log")
        # Only the user message should render; call + result are hidden.
        assert len(log.lines) == 1  # pyright: ignore[reportAttributeAccessIssue]
        # All three are still buffered.
        assert len(app._events) == 3  # pyright: ignore[reportPrivateUsage]


async def test_live_event_appends_to_buffer() -> None:
    """A live event delivered after mount lands in the buffer."""
    app = _app(_mixed_events())
    async with app.run_test():
        app._append_event(MessageUserEvent(thread_id=_TID, text="live one"))  # pyright: ignore[reportPrivateUsage]
        assert len(app._events) == 5  # pyright: ignore[reportPrivateUsage]
        assert any(
            isinstance(e, MessageUserEvent) and e.text == "live one"
            for e in app._events  # pyright: ignore[reportPrivateUsage]
        )
