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

import asyncio
import json
from collections.abc import Sequence

from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.tools import AgentTool

from ..types import InputShape, ThreadContext, ThreadId

# ── Deadlock detection for blocking ``send_message(mode="wait")`` ────────────
#
# A blocking wait enqueues a cycle behind the peer's single serial dispatcher
# and suspends the caller's cycle until it drains. On its own that is only
# latency — the peer finishes what it is doing, then runs the enqueued cycle.
# It deadlocks *only* when the peer is (directly or transitively) already
# blocked in a wait back on the caller, so the two dispatchers can never drain
# each other.
#
# We track the "waits-for" graph of in-flight blocking waits as
# ``{coordinator_id: {waiter_id: target_id}}`` and refuse a new wait only when
# committing to it would close a cycle. Each waiter has at most one outstanding
# edge: while suspended in a wait its cycle cannot issue another tool call. The
# check-and-register step runs with no ``await`` in between, so on the single
# event loop it is atomic — of two peers waiting on each other, exactly one
# registers first and the other observes that edge and refuses.
#
# Keyed by coordinator identity: peers that can actually deadlock this way
# share one coordinator object (the in-memory coordinator the worker's executor
# holds). Waits across separate ``CoordinatorClient`` instances are not tracked,
# matching the existing single-coordinator scope of these tools.
_wait_edges: dict[int, dict[str, str]] = {}


def _would_close_wait_cycle(coord_key: int, waiter: str, target: str) -> bool:
    """Return whether adding ``waiter -> target`` closes a wait-for cycle.

    Walks the existing waits-for chain from ``target``; a cycle would form iff
    that chain leads back to ``waiter``. The chain is acyclic by construction
    (every edge passed this check before being added), but a ``seen`` guard
    keeps the walk finite regardless.
    """
    edges = _wait_edges.get(coord_key)
    if not edges:
        return False
    seen: set[str] = set()
    cur: str | None = target
    while cur is not None and cur not in seen:
        if cur == waiter:
            return True
        seen.add(cur)
        cur = edges.get(cur)
    return False


def _release_wait_edge(coord_key: int, waiter: str) -> None:
    """Drop ``waiter``'s outstanding wait edge, pruning empty coordinator maps."""
    edges = _wait_edges.get(coord_key)
    if edges is None:
        return
    _ = edges.pop(waiter, None)
    if not edges:
        _ = _wait_edges.pop(coord_key, None)


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
    self_id = str(ctx.thread_id)
    coord = ctx.coordinator

    @_strands_tool(
        name="list_threads",
        description=(
            "List every thread registered with the current coordinator. Returns a JSON "
            "object with a 'threads' array; each entry has 'thread_id', "
            "'thread_name' (may be null), 'status', 'input_shape', 'parent_id' "
            "(may be null), and 'is_self' (true for the calling thread). Use this "
            "to discover peers before calling send_message. Only threads with "
            "'input_shape' == 'str_prompt' can receive send_message calls."
        ),
    )
    async def list_threads() -> str:
        infos = await coord.list_threads()
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
        return json.dumps({"threads": threads_json})

    @_strands_tool(
        name="send_message",
        description=(
            "Send a message to a peer thread by invoking its run(message) entry "
            "point. The peer must have input_shape='str_prompt'. 'mode' "
            "selects how the sender relates to the peer's result:\n"
            "  - 'wait' (default): await the peer and return its reply as the "
            "tool result. Blocks this cycle on the peer's cycle.\n"
            "  - 'fire_and_forget': schedule the peer's cycle in the background "
            "and return immediately; the peer's reply is discarded.\n"
            "  - 'continue_then_receive': schedule the peer's cycle and return "
            "immediately; when the peer completes, a fresh cycle is scheduled on "
            "THIS thread with the peer's reply as the user turn. Requires this "
            "thread to have input_shape='str_prompt'. If not, the tool returns "
            "an error and you should use mode='wait' instead.\n"
            "Use list_threads to discover valid thread_ids."
        ),
    )
    async def send_message(
        thread_id: str,
        message: str,
        mode: str = "wait",
    ) -> str:
        if thread_id == self_id:
            return "error: cannot send_message to self"
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
            # A blocking wait enqueues a cycle behind the peer's single serial
            # dispatcher and suspends this cycle until it drains. That is only a
            # true deadlock when the peer is (directly or transitively) already
            # waiting back on us: then neither dispatcher can drain the other.
            # Waiting on a merely-busy peer that is *not* waiting on us is safe —
            # it finishes its work, then runs our enqueued cycle. So we refuse
            # only when committing to this wait would close a cycle in the
            # waits-for graph. See ``_would_close_wait_cycle`` above.
            coord_key = id(coord)
            if _would_close_wait_cycle(coord_key, self_id, thread_id):
                return (
                    f"error: thread {thread_id} is already waiting on this thread; "
                    "send_message(mode='wait') would deadlock. Use "
                    "mode='fire_and_forget' or mode='continue_then_receive' instead."
                )
            # Register our outstanding edge before awaiting, with no intervening
            # ``await`` — so a peer that tries to wait back on us observes it and
            # refuses (breaking the cycle on exactly one side).
            _wait_edges.setdefault(coord_key, {})[self_id] = thread_id
            try:
                result = await peer.run(message)
            except Exception as exc:
                return f"error: {exc}"
            finally:
                _release_wait_edge(coord_key, self_id)
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

    return [list_threads, send_message]
