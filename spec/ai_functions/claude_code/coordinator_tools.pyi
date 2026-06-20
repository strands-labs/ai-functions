"""Runtime-facing tools exposed to a Claude Agent SDK session.

Mirror of :mod:`ai_functions.ai_thread.tools` for the Claude Agent runtime.
``coordinator_tools(ctx)`` returns a ``Sequence[SdkMcpTool]`` that can
be packaged via :func:`runtime_mcp_server` into the
``McpSdkServerConfig`` consumed by ``ClaudeAgentOptions.mcp_servers``.

Two tools are exposed:

- ``list_threads()`` — return a JSON-friendly snapshot of every thread
  registered with the calling agent's coordinator, including a
  ``is_self`` flag marking the calling thread.
- ``send_message(thread_id, message, mode="wait")`` — invoke a peer
  thread via its typed ``run(message)`` entry point. The peer must
  have ``input_shape == STR_PROMPT``. ``mode`` selects how the sender
  relates to the peer's result, with the same ``"wait"`` /
  ``"fire_and_forget"`` / ``"continue_then_receive"`` semantics as
  the Strands-side ``send_message``.

The MCP server reserves the name ``_ai_functions_runtime`` in the
``mcp_servers`` mapping. Users may not register their own server
under that name; ``ClaudeAgentThread`` raises on collision at spawn
time.
"""

from __future__ import annotations

from collections.abc import Sequence

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool

from ..types import ThreadContext


def coordinator_tools(ctx: ThreadContext) -> Sequence[SdkMcpTool[object]]:
    """Build SDK MCP tools bound to ``ctx`` for runtime-facing dispatch."""
    ...


def runtime_mcp_server(ctx: ThreadContext) -> McpSdkServerConfig:
    """Package the coordinator tools as an in-process MCP server."""
    ...
