"""Shared event-formatter used by ``ai-functions logs`` and ``ai-functions attach``.

The logic originated in ``examples/05_claude_code.py`` — the pretty-
printer that turns :class:`~ai_functions.types.Event` instances into one-line
console output. Factored into the package so the CLI commands and the
TUI share it.

Output is a ``rich`` :class:`~rich.console.RenderableType` so both the
plain-stdout logger (``ai-functions logs``) and the Textual ``RichLog`` widget
(``ai-functions attach``) consume it without double-formatting. Callers that
need plain text can ``str(Console().render(renderable))`` or pass
``markup=False``.
"""

from __future__ import annotations

from typing import cast

from rich.console import Console, RenderableType
from rich.text import Text

from ..types import (
    CancelledEvent,
    CompletedEvent,
    CustomEvent,
    Event,
    FailedEvent,
    MessageAssistantCompleteEvent,
    MessageAssistantStartEvent,
    MessageAssistantThinkingEvent,
    MessageAssistantTokenEvent,
    MessageUserEvent,
    StartedEvent,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_MAX_PREVIEW = 240
_MAX_USER_PREVIEW = 120
_MAX_ARG_PREVIEW = 60
_MAX_TOOL_RESULT_PREVIEW = 500

STRUCTURED_OUTPUT_TOOL = "FinalAnswer"
"""Name of the structured-output wrapper tool.

When a thread has a non-string typed output, the answer is returned by
calling an auto-generated ``FinalAnswer`` tool (see
``ai_functions.ai_thread.ai_thread`` — ``create_model("FinalAnswer", ...)``)
rather than as assistant prose. In the conversation view that tool call
and its result are redundant with the surfaced answer, so they are
hidden.
"""


def filter_events(event: Event) -> bool:
    """Report whether an event should appear in the "all events" feed.

    This is the render filter for :func:`format_event`: callers decide
    whether to render an event with this predicate, then format the ones
    that pass. The only events filtered out are the intra-turn streaming
    fragments — the assistant ``start`` marker
    (:class:`MessageAssistantStartEvent`) and the streamed ``token`` /
    ``thinking`` chunks (:class:`MessageAssistantTokenEvent` /
    :class:`MessageAssistantThinkingEvent`). An assistant turn streams as
    a start marker followed by many such chunks and is then closed by a
    single :class:`MessageAssistantCompleteEvent` carrying the full
    aggregated content; rendering each chunk would emit one line per
    token that duplicates the complete event.

    Args:
        event: Any :class:`~ai_functions.types.Event` subclass.

    Returns:
        ``True`` if the event should be rendered in the feed, ``False``
        for the streaming-fragment chunk events described above.
    """
    return not isinstance(
        event,
        MessageAssistantStartEvent | MessageAssistantTokenEvent | MessageAssistantThinkingEvent,
    )


def filter_events_full(event: Event) -> bool:
    """Report whether an event belongs in the "conversation" view.

    This is the render filter for :func:`format_event_full`. The
    conversation view shows only the events that carry conversational
    content — user turns, completed assistant turns, and tool activity.
    Every other event kind (lifecycle, token usage, custom, and the
    streaming fragments) is filtered out, which is what makes the view a
    clean transcript.

    The structured-output wrapper call (:data:`STRUCTURED_OUTPUT_TOOL`)
    is also filtered out: its answer is already surfaced as the turn's
    content, so the call would be redundant. Its result event carries no
    tool name and is suppressed by the caller via ``tool_use_id``.

    Args:
        event: Any :class:`~ai_functions.types.Event` subclass.

    Returns:
        ``True`` if the event is part of the conversation view, ``False``
        otherwise.
    """
    if isinstance(event, ToolCallEvent) and event.tool_name == STRUCTURED_OUTPUT_TOOL:
        return False
    return isinstance(event, MessageUserEvent | MessageAssistantCompleteEvent | ToolCallEvent | ToolResultEvent)


def format_event(event: Event, *, markup: bool = True) -> RenderableType:
    """Render one ai-functions event as a one-line Rich renderable.

    The renderable is typically a :class:`~rich.text.Text` instance;
    callers should not depend on the concrete type. Unknown / unhandled
    event kinds render as ``"• <kind>"`` so the feed never drops an event
    silently. Callers decide *whether* to render an event with
    :func:`filter_events`; this function always returns a renderable.

    Args:
        event: Any :class:`~ai_functions.types.Event` subclass.
        markup: When ``True`` (default), the returned renderable
            includes ANSI colours / bold attributes. Pass ``False`` for
            log-file output where colours would produce escape noise.

    Returns:
        A Rich renderable that prints on a single line when measured
        against an infinite-width console (multi-line content such as
        long assistant messages is truncated with an ellipsis).
    """
    del markup  # colour is baked into Text styles; strip via console render.
    match event:
        case StartedEvent(thread_name=name):
            return Text(f"  ▶ {name or 'thread'} started", style="green")
        case CompletedEvent(thread_name=name):
            return Text(f"  ✓ {name or 'thread'} completed", style="green bold")
        case CancelledEvent(thread_name=name):
            return Text(f"  ✗ {name or 'thread'} cancelled", style="yellow")
        case FailedEvent(thread_name=name, error=error):
            return Text(f"  ✗ {name or 'thread'} failed: {error}", style="red bold")
        case MessageUserEvent(text=text):
            return Text(f"  ▷ user: {_truncate(text, _MAX_USER_PREVIEW)}", style="cyan")
        case MessageAssistantCompleteEvent(content=content):
            joined = _join_assistant_text(content)
            preview = _truncate(joined, _MAX_PREVIEW) if joined else "<no text content>"
            return Text(f"  ◁ assistant: {preview}", style="magenta")
        case ToolCallEvent():
            return Text(
                f"    ⚙ tool call: {event.tool_name}({_format_args(event.arguments)})",
                style="blue",
            )
        case ToolResultEvent():
            return Text(
                f"    ⚙ tool result: {event.tool_use_id} [{event.status}]",
                style="blue dim",
            )
        case TokenUsageEvent(token_usage=usage):
            total_input = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            total = total_input + usage.output_tokens
            return Text(
                f"  Σ tokens: in={usage.input_tokens} "
                f"cache_r={usage.cache_read_tokens} cache_w={usage.cache_write_tokens} "
                f"out={usage.output_tokens} (total={total})",
                style="dim",
            )
        case CustomEvent(kind=kind):
            return Text(f"  • custom event: {kind}", style="dim")
        case _:
            return Text(f"  • {event.kind}", style="dim")


def format_event_full(event: Event) -> RenderableType:
    """Render a conversation event with full, untruncated content.

    The content a user is meant to read — user turns and completed
    assistant turns — is rendered untruncated. Tool activity is machine
    noise, so it is still abbreviated: tool-call arguments use the same
    per-value cap as :func:`format_event`, and a tool result's body is
    shown but capped to a short preview. Callers decide *whether* to
    render an event in the conversation view with
    :func:`filter_events_full`; this function always returns a
    renderable.

    Args:
        event: Any :class:`~ai_functions.types.Event` subclass.

    Returns:
        A Rich renderable with the event's complete content.
    """
    match event:
        case MessageUserEvent(text=text):
            return Text(f"  ▷ user: {text}", style="cyan")
        case MessageAssistantCompleteEvent(content=content):
            joined = _join_assistant_text(content)
            body = joined if joined else "<no text content>"
            return Text(f"  ◁ assistant: {body}", style="magenta")
        case ToolCallEvent():
            return Text(
                f"    ⚙ tool call: {event.tool_name}({_format_args(event.arguments)})",
                style="blue",
            )
        case ToolResultEvent():
            body = _truncate(_join_tool_result_text(event.content), _MAX_TOOL_RESULT_PREVIEW)
            suffix = f"\n{body}" if body else ""
            return Text(
                f"    ⚙ tool result: {event.tool_use_id} [{event.status}]{suffix}",
                style="blue dim",
            )
        case _:
            # Non-conversation kinds are excluded by ``filter_events_full``
            # before reaching here; fall back to the compact renderer.
            return format_event(event)


def format_event_plain(event: Event) -> str:
    """Plain-text convenience wrapper for ``ai-functions logs`` without ``--color``.

    Equivalent to rendering :func:`format_event` with ``markup=False``
    through a Rich console and stripping trailing whitespace.

    Args:
        event: Event to format.

    Returns:
        A single-line, ANSI-free representation of the event, or the
        empty string for events that :func:`filter_events` excludes from
        the feed (the streaming-fragment chunk events).
    """
    if not filter_events(event):
        return ""
    renderable = format_event(event, markup=False)
    console = Console(file=None, color_system=None, force_terminal=False, width=1000)
    with console.capture() as capture:
        console.print(renderable, end="")
    return capture.get().rstrip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _join_assistant_text(content: object) -> str:
    """Extract text from the assistant content list."""
    if not isinstance(content, list):
        return ""
    raw_blocks: list[object] = list(content)  # pyright: ignore[reportUnknownArgumentType]
    texts: list[str] = []
    for block in raw_blocks:
        if isinstance(block, dict):
            block_map = cast("dict[str, object]", block)
            text = block_map.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts)


def _format_args(arguments: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in arguments.items():
        rendered = repr(value)
        if len(rendered) > _MAX_ARG_PREVIEW:
            rendered = rendered[: _MAX_ARG_PREVIEW - 3] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _join_tool_result_text(content: object) -> str:
    """Extract text from a tool result's content blocks.

    Mirrors :func:`_join_assistant_text` for the Strands
    ``ToolResultContent`` shape: each block is a dict that may carry a
    ``text`` entry. Non-text blocks (e.g. ``json``) are rendered via
    ``repr`` so nothing is silently dropped from the full view.

    Args:
        content: The ``ToolResultEvent.content`` value.

    Returns:
        The concatenated block text, one block per line.
    """
    if not isinstance(content, list):
        return ""
    raw_blocks: list[object] = list(content)  # pyright: ignore[reportUnknownArgumentType]
    parts: list[str] = []
    for block in raw_blocks:
        if isinstance(block, dict):
            block_map = cast("dict[str, object]", block)
            text = block_map.get("text")
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(repr(block_map))
        else:
            parts.append(repr(block))
    return "\n".join(parts)
