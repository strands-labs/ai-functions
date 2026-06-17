"""Coordinator-facing tools exposed to LLM agents.

Each tool is a Strands ``@tool``-decorated coroutine that closes over a
:class:`ThreadContext` so the calling agent can reach the coordinator.
The default ``config_hook`` installed on every ``AIFunction`` calls
:func:`coordinator_tools` with the cycle's ``ctx`` and appends the
result to ``cycle_config.tools``.

Two tools are exposed:

- ``list_threads()`` — return a JSON-friendly snapshot of every thread
  registered with the calling agent's coordinator, including a
  ``is_self`` flag marking the calling thread.
- ``send_message(thread_id, message, mode="wait")`` — invoke a peer
  thread via its typed ``run(message)`` entry point. The peer must have
  ``input_shape == STR_PROMPT``. ``mode`` selects how the sender relates
  to the peer's result:

  - ``"wait"`` (default): await the peer's cycle; return its reply as
    the tool's result. Blocks the sender's cycle on the peer's cycle.
  - ``"fire_and_forget"``: schedule the peer's cycle as a background
    task; return immediately. The peer's reply is discarded.
  - ``"continue_then_receive"``: schedule the peer's cycle, return
    immediately, and when the peer completes, enqueue a fresh cycle on
    the sender with the peer's reply formatted as the user turn. This
    mode requires the sender itself to have ``input_shape == STR_PROMPT``
    — the tool returns an error otherwise, asking the agent to use
    ``"wait"``.

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
