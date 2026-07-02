"""Graph types for post-hoc computation-graph reconstruction.

These types represent a graph reconstructed from the coordinator's event log
*after* execution completes, not built during it. The same reconstruction
works whether threads run standalone or on separate workers.

Three node types:
- ThreadNode: an AIThread / AIFunction execution (messages, tool calls, params).
- ParameterNode: a memory parameter that was recalled / queried / searched.
- ToolCallNode: a tool invocation within a thread.

All grad-bearing nodes share a common base carrying ``node_id``, ``value``,
``requires_grad``, and ``gradients``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from strands.types.content import Message

if TYPE_CHECKING:
    from ..memory.base import MemoryBackend


@dataclass
class Node:
    """Base class for all grad-bearing graph nodes."""

    node_id: str
    value: Any = None  # pyright: ignore[reportExplicitAny]
    requires_grad: bool = True
    gradients: list[str] = field(default_factory=list)


@dataclass
class ToolCallNode:
    """A tool invocation extracted from the event log."""

    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]
    result: str | None = None
    status: Literal["success", "error"] = "success"


@dataclass
class ParameterNode(Node):
    """A memory parameter that was recalled during thread execution.

    Reconstructed from ``ParameterRecalledEvent`` events. Holds a **direct
    reference** to the ``MemoryBackend`` that owns this parameter, so the
    optimizer can call ``backend.consolidate(name, feedbacks)`` with no lookup
    table.
    """

    name: str = ""
    derivation: Literal["full", "query", "search"] = "full"
    backend: MemoryBackend | None = field(default=None, repr=False)
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]


@dataclass
class ThreadNode(Node):
    """A thread execution reconstructed from the event log.

    The primary graph node. The optimizer walks these in reverse topological
    order, distributing feedback to parameters and child threads.
    """

    thread_id: str = ""
    func_name: str | None = None
    messages: list[Message] = field(default_factory=list)

    parameters: list[ParameterNode] = field(default_factory=list)
    tool_calls: list[ToolCallNode] = field(default_factory=list)
    child_threads: list[ThreadNode] = field(default_factory=list)
    parent: ThreadNode | None = field(default=None, repr=False)

    events: list[Any] = field(default_factory=list, repr=False)  # pyright: ignore[reportExplicitAny]
