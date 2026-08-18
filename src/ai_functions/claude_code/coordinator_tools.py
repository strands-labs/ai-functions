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

from collections.abc import Callable, Sequence
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server, tool

from ..runtime.coordinator_tools_core import (
    LIST_THREADS_DESCRIPTION,
    LIST_THREADS_INPUT_SCHEMA,
    SEND_MESSAGE_DESCRIPTION,
    SEND_MESSAGE_INPUT_SCHEMA,
)
from ..runtime.coordinator_tools_core import list_threads as _list_threads
from ..runtime.coordinator_tools_core import send_message as _send_message
from ..types import ThreadContext

_RUNTIME_SERVER_NAME = "_ai_functions_runtime"
"""Reserved key under which the runtime MCP server registers in
``ClaudeAgentOptions.mcp_servers``. Users may not register their own
server under this name."""


def _text_result(text: str) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # SDK callback contract
    """Wrap ``text`` in the single-text-block result shape the SDK expects."""
    return {"content": [{"type": "text", "text": text}]}


def coordinator_tools(ctx: ThreadContext) -> Sequence[SdkMcpTool[object]]:
    """Build SDK MCP tools bound to ``ctx`` for runtime-facing dispatch."""
    return _coordinator_tools_with_provider(lambda: ctx)


def _coordinator_tools_with_provider(
    ctx_provider: Callable[[], ThreadContext | None],
) -> Sequence[SdkMcpTool[object]]:
    """Build SDK MCP tools that resolve their ``ThreadContext`` lazily.

    Used by :class:`ClaudeAgentThread` to share one MCP server across
    many cycles: the SDK client connects once, but every tool
    invocation reads the current cycle's ctx via ``ctx_provider``.
    The dispatcher serialises cycles, so ``ctx_provider`` always
    returns the active cycle's ctx during a tool call.
    """

    @tool("list_threads", LIST_THREADS_DESCRIPTION, LIST_THREADS_INPUT_SCHEMA)
    async def list_threads(
        _args: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # SDK callback contract
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # SDK callback contract
        """Return a JSON-friendly snapshot of registered threads."""
        ctx = ctx_provider()
        if ctx is None:
            return _text_result("error: no active cycle")
        result = await _list_threads(ctx.coordinator, str(ctx.thread_id))
        return _text_result(result.model_dump_json())

    @tool("send_message", SEND_MESSAGE_DESCRIPTION, SEND_MESSAGE_INPUT_SCHEMA)
    async def send_message(
        args: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # SDK callback contract
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # SDK callback contract
        """Dispatch ``message`` to ``thread_id`` according to ``mode``."""
        ctx = ctx_provider()
        if ctx is None:
            return _text_result("error: no active cycle")
        text = await _send_message(
            ctx.coordinator,
            str(ctx.thread_id),
            str(args["thread_id"]),
            str(args["message"]),
            str(args.get("mode", "wait")),
        )
        return _text_result(text)

    return [list_threads, send_message]


def runtime_mcp_server(ctx: ThreadContext) -> McpSdkServerConfig:
    """Package the coordinator tools as an in-process MCP server."""
    return _runtime_mcp_server_with_provider(lambda: ctx)


def _runtime_mcp_server_with_provider(
    ctx_provider: Callable[[], ThreadContext | None],
) -> McpSdkServerConfig:
    """Build the runtime MCP server with a lazy ``ThreadContext`` provider."""
    return create_sdk_mcp_server(
        name=_RUNTIME_SERVER_NAME,
        version="1.0.0",
        tools=list(_coordinator_tools_with_provider(ctx_provider)),
    )
