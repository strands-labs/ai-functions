"""Reconstruct a single ``ThreadNode`` from one thread's event log.

The optimization graph is a pure function of the event log + the live memory
backends, built after a run. ``build_graph`` rebuilds one thread's node;
cross-thread structure (one thread's output feeding another) is wired by the
caller, since that Python-level dataflow is recorded in no event log.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..ai_thread.reconstruction import reconstruct_messages
from ..types.events import (
    MessageAssistantCompleteEvent,
    ParameterRecalledEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ..types.graph import ParameterNode, ThreadNode, ToolCallNode
from ._formatting import unique_name

if TYPE_CHECKING:
    from ..memory.base import MemoryBackend
    from ..types.events import Event

logger = logging.getLogger(__name__)


def topological_sort(node: ThreadNode) -> list[ThreadNode]:
    """Return ThreadNodes in reverse topological order, pruning grad-free subtrees."""
    visited: set[int] = set()
    order: list[ThreadNode] = []
    _has_grad_cache: dict[int, bool] = {}

    def _has_grad_parameter(n: ThreadNode) -> bool:
        nid = id(n)
        if nid in _has_grad_cache:
            return _has_grad_cache[nid]
        result = any(p.requires_grad for p in n.parameters) or any(_has_grad_parameter(c) for c in n.child_threads)
        _has_grad_cache[nid] = result
        return result

    def _dfs(n: ThreadNode) -> None:
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)
        for child in n.child_threads:
            if _has_grad_parameter(child):
                _dfs(child)
        order.append(n)

    _dfs(node)
    return list(reversed(order))


def _assistant_text(content: list[Any]) -> str:  # pyright: ignore[reportExplicitAny]
    """Concatenate the text blocks of an assistant turn's content."""
    return "".join(block["text"] for block in content if isinstance(block, dict) and "text" in block)


def build_graph(events: list[Event], backends: list[MemoryBackend]) -> ThreadNode:
    """Reconstruct one thread's computation node from its event log.

    Builds exactly one ``ThreadNode`` from a single thread's pre-fetched events.
    Cross-thread edges (``child_threads``) are wired by the caller.

    Args:
        events: One thread's events in append order (oldest first).
        backends: Live memory backends, matched by ``backend_id``.

    Returns:
        A single childless ``ThreadNode``.
    """
    backend_map = {b.backend_id: b for b in backends}

    thread_id = ""
    func_name: str | None = None
    for evt in events:
        tid = getattr(evt, "thread_id", None)
        if tid:
            thread_id = str(tid)
        name = getattr(evt, "thread_name", None)
        if name and func_name is None:
            func_name = str(name)

    # Parameters: one ParameterNode per (backend_id, name); last recall wins.
    param_nodes: dict[tuple[str, str], ParameterNode] = {}
    tool_calls: list[ToolCallNode] = []
    tc_map: dict[str, ToolCallNode] = {}
    result_value: Any = None  # pyright: ignore[reportExplicitAny]

    for evt in events:
        if isinstance(evt, ParameterRecalledEvent):
            backend = backend_map.get(evt.backend_id)
            if backend is None:
                logger.warning("build_graph: no backend for id '%s' (param '%s')", evt.backend_id, evt.name)
                continue
            value = backend.deserialize_value(evt.name, evt.value)
            key = (evt.backend_id, evt.name)
            param_nodes[key] = ParameterNode(
                node_id=unique_name(evt.name),
                value=value,
                requires_grad=evt.requires_grad,
                name=evt.name,
                derivation=evt.derivation,
                backend=backend,
                description=evt.description,
                meta=dict(evt.meta),
            )
        elif isinstance(evt, ToolCallEvent):
            tc = ToolCallNode(
                tool_use_id=evt.tool_use_id,
                tool_name=evt.tool_name,
                arguments=dict(evt.arguments),
            )
            tc_map[evt.tool_use_id] = tc
            tool_calls.append(tc)
        elif isinstance(evt, ToolResultEvent):
            tc = tc_map.get(evt.tool_use_id)
            if tc is not None:
                tc.result = "".join(
                    block["text"] for block in evt.content if isinstance(block, dict) and "text" in block
                )
                tc.status = "error" if evt.status == "error" else "success"
        elif isinstance(evt, MessageAssistantCompleteEvent):
            result_value = _assistant_text(list(evt.content)) or result_value

    messages = reconstruct_messages(events)

    nid = f"{func_name or 'thread'}-{thread_id[:4]}"
    return ThreadNode(
        node_id=nid,
        thread_id=thread_id,
        func_name=func_name,
        messages=messages,
        value=result_value,
        parameters=list(param_nodes.values()),
        tool_calls=tool_calls,
        child_threads=[],
        events=list(events),
    )
