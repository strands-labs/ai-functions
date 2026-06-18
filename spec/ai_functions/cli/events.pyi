"""Shared event-formatter used by ``ai-functions logs`` and ``ai-functions attach``.

The logic originated as the pretty-
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

from rich.console import RenderableType

from ..types import Event

STRUCTURED_OUTPUT_TOOL: str
"""Name of the structured-output wrapper tool (``"FinalAnswer"``).

When a thread has a non-string typed output, the answer is returned by
calling an auto-generated ``FinalAnswer`` tool rather than as assistant
prose. The conversation view hides this tool call (via
:func:`format_event_full`) and its result, since the answer is already
surfaced as the turn's content.
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
    ...


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
    ...


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
    ...


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
    ...


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
    ...
