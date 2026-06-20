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

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server, tool

from ..types import InputShape, ThreadContext, ThreadId

_RUNTIME_SERVER_NAME = "_ai_functions_runtime"
"""Reserved key under which the runtime MCP server registers in
``ClaudeAgentOptions.mcp_servers``. Users may not register their own
server under this name."""


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

    @tool(
        "list_threads",
        (
            "List every thread registered with the current coordinator. "
            "Returns a JSON object with a 'threads' array; each entry has "
            "'thread_id', 'thread_name' (may be null), 'status', "
            "'input_shape', 'parent_id' (may be null), and 'is_self' "
            "(true for the calling thread). Use this to discover peers "
            "before calling send_message. Only threads with "
            "'input_shape' == 'str_prompt' can receive send_message calls."
        ),
        {},
    )
    async def list_threads(
        _args: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # SDK callback contract
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # SDK callback contract
        """Return a JSON-friendly snapshot of registered threads."""
        ctx = ctx_provider()
        if ctx is None:
            return {"content": [{"type": "text", "text": "error: no active cycle"}]}
        self_id = str(ctx.thread_id)
        infos = await ctx.coordinator.list_threads()
        threads_json: list[dict[str, object]] = []
        for info in infos:
            threads_json.append(
                {
                    "thread_id": str(info.thread_id),
                    "thread_name": info.thread_name,
                    "status": str(info.status),
                    "input_shape": str(info.input_shape),
                    "parent_id": None if info.parent_id is None else str(info.parent_id),
                    "is_self": str(info.thread_id) == self_id,
                },
            )
        return {
            "content": [
                {"type": "text", "text": json.dumps({"threads": threads_json})},
            ],
        }

    @tool(
        "send_message",
        (
            "Send a message to a peer thread by invoking its run(message) "
            "entry point. The peer must have input_shape='str_prompt'. "
            "'mode' selects how the sender relates to the peer's result:\n"
            "  - 'wait' (default): await the peer and return its reply as "
            "the tool result. Blocks this cycle on the peer's cycle.\n"
            "  - 'fire_and_forget': schedule the peer's cycle in the "
            "background and return immediately; the peer's reply is "
            "discarded.\n"
            "  - 'continue_then_receive': schedule the peer's cycle and "
            "return immediately; when the peer completes, a fresh cycle "
            "is scheduled on THIS thread with the peer's reply as the "
            "user turn. Requires this thread to have "
            "input_shape='str_prompt'. If not, the tool returns an error "
            "and you should use mode='wait' instead.\n"
            "Use list_threads to discover valid thread_ids."
        ),
        {"thread_id": str, "message": str, "mode": str},
    )
    async def send_message(
        args: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # SDK callback contract
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # SDK callback contract
        """Dispatch ``message`` to ``thread_id`` according to ``mode``."""
        ctx = ctx_provider()
        if ctx is None:
            return {"content": [{"type": "text", "text": "error: no active cycle"}]}
        thread_id = str(args["thread_id"])
        message = str(args["message"])
        mode = str(args.get("mode", "wait"))
        text = await _dispatch(ctx, str(ctx.thread_id), thread_id, message, mode)
        return {"content": [{"type": "text", "text": text}]}

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


# ── Internals ────────────────────────────────────────────────────────


async def _dispatch(
    ctx: ThreadContext,
    self_id: str,
    thread_id: str,
    message: str,
    mode: str,
) -> str:
    """Route ``message`` to ``thread_id`` per ``mode``; return the tool's text."""
    if thread_id == self_id:
        return "error: cannot send_message to self"

    coord = ctx.coordinator
    try:
        peer_info = await coord.get_thread_info(ThreadId(thread_id))
    except Exception:
        return f"error: no thread with id {thread_id}"
    if peer_info.input_shape != InputShape.STR_PROMPT:
        return (
            f"error: thread {thread_id} has input_shape={peer_info.input_shape!s}; "
            "send_message requires a str_prompt peer."
        )

    peer = coord.get_handle(ThreadId(thread_id))

    if mode == "wait":
        try:
            result = await peer.run(message)
        except Exception as exc:
            return f"error: {exc}"
        return str(result)

    if mode == "fire_and_forget":
        fut = peer.run(message)

        async def _swallow() -> None:
            try:
                _ = await fut
            except Exception:
                pass

        _ = asyncio.create_task(_swallow())
        return f"ok: dispatched to {thread_id}"

    if mode == "continue_then_receive":
        try:
            self_info = await coord.get_thread_info(ctx.thread_id)
        except Exception:
            return "error: calling thread is no longer registered"
        if self_info.input_shape != InputShape.STR_PROMPT:
            return (
                "error: continue_then_receive requires this thread to "
                "have input_shape='str_prompt'. Use mode='wait' instead."
            )
        sender = coord.get_handle(ctx.thread_id)
        fut = peer.run(message)

        async def _notify_on_complete() -> None:
            try:
                peer_result = await fut
                notification = f"[Reply from {thread_id}] {peer_result}"
            except Exception as exc:
                notification = f"[Reply from {thread_id}] error: {exc}"
            try:
                _ = sender.run(notification)
            except Exception:
                pass

        _ = asyncio.create_task(_notify_on_complete())
        return f"ok: dispatched to {thread_id}; reply will arrive as a new user turn"

    return f"error: unknown mode {mode!r}; valid modes are 'wait', 'fire_and_forget', 'continue_then_receive'"
