"""TextGrad-style optimization over reconstructed computation graphs."""

from __future__ import annotations

from ..memory.base import MemoryBackend
from ..types.events import Event
from ..types.graph import ThreadNode
from .textgrad import TextGradOptimizer

__all__ = [
    "TextGradOptimizer",
    "build_graph",
]


def build_graph(events: list[Event], backends: list[MemoryBackend]) -> ThreadNode:
    """Reconstruct one thread's computation node from its event log.

    Builds **exactly one** :class:`ThreadNode` from a single thread's
    pre-fetched events (the caller does ``await coordinator.get_events(tid)``).
    It does not recurse into other threads and has no coordinator handle, so
    cross-thread structure — one thread's output feeding another — is wired by
    the caller afterward::

        email = build_graph(await coord.get_events("email-1"), [memory])
        joke1 = build_graph(await coord.get_events("joke-1"), [memory])
        joke2 = build_graph(await coord.get_events("joke-2"), [memory])
        email.child_threads = [joke1, joke2]

    Args:
        events: One thread's events in append order (oldest first), as
            returned by :meth:`Coordinator.get_events`.
        backends: Live memory backends whose ``backend_id`` is matched against
            each ``ParameterRecalledEvent`` so the resulting ``ParameterNode``
            can reference the owning backend for consolidation.

    Returns:
        A single childless :class:`ThreadNode`.

    Ensures:
        - One :class:`ParameterNode` per distinct ``(backend_id, name)``; when
          the same parameter is recalled more than once, the latest event's
          value and metadata win (last-write-wins) and the value is kept in
          its recalled type.
        - A ``ParameterRecalledEvent`` whose ``backend_id`` matches no entry in
          ``backends`` is skipped (no node, logged).
        - Each ``ParameterNode.value`` is ``backend.deserialize_value`` applied
          to the event's recalled value.
        - Tool activity is paired by ``tool_use_id`` into ``ToolCallNode`` s
          (a ``ToolCallEvent`` supplies name/arguments; the matching
          ``ToolResultEvent`` supplies result and ``status``).
        - ``node.messages`` is ``reconstruct_messages(events)`` (parameter and
          other non-renderable events are inert to it).
        - ``node.value`` is the text of the last ``MessageAssistantCompleteEvent``
          (its concatenated text blocks), or ``None`` if the thread produced no
          assistant turn.
        - ``node.child_threads`` is empty; the caller wires cross-thread edges.
    """
    ...
