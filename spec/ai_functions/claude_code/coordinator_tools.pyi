"""Runtime-facing tools exposed to a Claude Agent SDK session.

Claude Agent SDK binding for the two runtime-facing tools.
``coordinator_tools(ctx)`` returns a ``Sequence[SdkMcpTool]`` that can
be packaged via :func:`runtime_mcp_server` into the
``McpSdkServerConfig`` consumed by ``ClaudeAgentOptions.mcp_servers``.

The tools are ``list_threads()`` and
``send_message(thread_id, message, mode="wait")``. Their semantics, wire
descriptions, and argument schemas live in
:mod:`ai_functions.runtime.coordinator_tools_core`, shared with the Strands
binding in :mod:`ai_functions.ai_thread.tools`, so an agent sees the same tool
whichever runtime hosts it. This module adds only the SDK decorator and the MCP
content packing.

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
