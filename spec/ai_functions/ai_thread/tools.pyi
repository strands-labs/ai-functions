"""Coordinator-facing tools exposed to LLM agents.

Each tool is a Strands ``@tool``-decorated coroutine that closes over a
:class:`ThreadContext` so the calling agent can reach the coordinator.
The default ``config_hook`` installed on every ``AIFunction`` calls
:func:`coordinator_tools` with the cycle's ``ctx`` and appends the
result to ``cycle_config.tools``.

Two tools are exposed, ``list_threads()`` and
``send_message(thread_id, message, mode="wait")``. Their semantics, wire
descriptions, and argument schema live in
:mod:`ai_functions.runtime.coordinator_tools_core` so that every runtime adapter
offers the identical tool; this module is the Strands binding for them, and
Strands derives each schema from the wrapper's signature.

These tools are LLM-facing. Application code that wants the old
inject-then-no-cycle semantics should call
``ctx.coordinator.notify(...)`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from strands.types.tools import AgentTool

from ..types import ThreadContext


def coordinator_tools(ctx: ThreadContext) -> Sequence[AgentTool]:
    """Build the list of coordinator-facing tools bound to ``ctx``.

    Args:
        ctx: The current cycle's context. Captured by the returned
            tools so they can reach ``ctx.coordinator`` when invoked.

    Returns:
        A fresh list of ``AgentTool`` instances — one per tool. Each
        invocation uses ``ctx.coordinator`` and ``ctx.thread_id``
        captured at build time.
    """
    ...
