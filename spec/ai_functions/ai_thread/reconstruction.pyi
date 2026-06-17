"""Rebuild Strands-format message history from a stored event sequence."""

from __future__ import annotations

from strands.types.content import Messages

from ..types import Event, RenderableEvent


def render_renderable_events(events: list[RenderableEvent]) -> Messages:
    """Render a sequence of :data:`RenderableEvent` s into Strands ``Messages``.

    The pure-rendering core. Assumes the caller has already handled any
    ``ContextSummarizedEvent`` boundaries and filtered out
    observability-only events that carry no conversational content; this
    function walks what remains and emits the matching message list. It
    does not run the I10 healing pass — that is
    :func:`reconstruct_messages`'s responsibility, so the healer runs
    exactly once on the fully-assembled message list.

    Args:
        events: Renderable events in chronological order.

    Returns:
        The Strands ``Messages`` list these events produce.

    Ensures:
        - Each :class:`MessageUserEvent` yields one user ``Message`` with
          a single text block (see I7).
        - Each :class:`MessageAssistantCompleteEvent` yields one assistant
          ``Message`` whose content is the stored list of content blocks.
        - A contiguous run of :class:`ToolResultEvent` events collapses
          into one user ``Message`` carrying one ``toolResult`` block per
          event; a subsequent ``MessageUserEvent``,
          ``MessageAssistantCompleteEvent``, or end-of-list flushes the
          group.
        - :class:`ToolCallEvent` is observability-only and produces no
          message.
    """
    ...


def reconstruct_messages(events: list[Event]) -> Messages:
    """Convert stored events into a Strands-format ``Messages`` list.

    Args:
        events: Events for one thread in chronological order.

    Returns:
        A message list ready for a Strands agent.

    Requires:
        ``events`` is sorted by append order (oldest first).

    Ensures:
        - If the input contains one or more
          :class:`ContextSummarizedEvent` events, the one with the
          greatest index (the last one in append order) defines a
          boundary: every event at that index or earlier is dropped from
          rendering and ``event.new_history`` is rendered in its place
          via :func:`render_renderable_events`, then events strictly
          after the boundary render normally and append to the result.
        - Events outside the renderable subset — lifecycle, token-usage,
          session, approval, streaming fragments — are filtered out
          before rendering.
        - The output obeys every ``Ensures`` of
          :func:`render_renderable_events`.
        - Every ``toolUse`` block in the reconstructed history is
          followed by a matching ``toolResult`` block (I10). Where the
          event log contains no matching ``TOOL_RESULT``, the missing
          blocks are synthesized by Strands'
          ``generate_missing_tool_result_content`` helper — the same text
          and shape Strands' own session manager produces when it detects
          orphaned ``toolUse``. The I10 healing pass runs exactly once,
          on the fully-assembled message list, after any summary splice.
    """
    ...
