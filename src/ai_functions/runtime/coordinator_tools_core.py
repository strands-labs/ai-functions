"""Transport-agnostic bodies for the two runtime-facing coordinator tools.

Every runtime adapter that exposes ``list_threads`` / ``send_message`` to an
agent — the Strands-native path in :mod:`ai_functions.ai_thread.tools` and the
Claude Agent SDK MCP path in :mod:`ai_functions.claude_code.coordinator_tools` —
calls into this module. Adapters own only their transport's plumbing (decorator,
schema handoff, result packing); the semantics, the wire-visible descriptions,
the argument schemas, and the deadlock guard live here once so no two agents can
be offered different versions of the same tool.

The bodies take ``(coordinator, self_thread_id)`` rather than a
:class:`~ai_functions.types.ThreadContext`, because that pair is all they need.
An adapter with no live cycle to draw a context from can therefore still serve
the tools.

Schemas are derived, never written twice: :class:`SendMessageArgs` is the single
declaration of ``send_message``'s arguments, and adapters either bind their
decorator to a matching signature or hand
:data:`SEND_MESSAGE_INPUT_SCHEMA` (``SendMessageArgs.model_json_schema()``)
straight to their SDK.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import Coordinator
from ..types import InputShape, ThreadId

__all__ = [
    "LIST_THREADS_DESCRIPTION",
    "LIST_THREADS_INPUT_SCHEMA",
    "SEND_MESSAGE_DESCRIPTION",
    "SEND_MESSAGE_INPUT_SCHEMA",
    "SEND_MESSAGE_MODES",
    "ListThreadsResult",
    "SendMessageArgs",
    "SendMessageMode",
    "ThreadSummary",
    "list_threads",
    "send_message",
]


# ── Argument and result models ──────────────────────────────────────────────

SendMessageMode = Literal["wait", "fire_and_forget", "continue_then_receive"]
"""How the sender relates to the peer's result. See :data:`SEND_MESSAGE_DESCRIPTION`."""

SEND_MESSAGE_MODES: tuple[str, ...] = get_args(SendMessageMode)
"""The legal :data:`SendMessageMode` values, for runtime validation."""


class ThreadSummary(BaseModel):
    """One entry of :class:`ListThreadsResult`; an agent-facing ``ThreadInfo``.

    Every field is widened to ``str`` because this crosses a tool boundary to a
    language model: :class:`~ai_functions.types.ThreadInfo` carries ``ThreadId``
    newtypes and ``StrEnum`` members, and a model reading the tool result has no
    use for that distinction. Frozen, like ``ThreadInfo``, so it round-trips over
    the wire via ``model_dump_json`` / ``model_validate_json``.
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str
    """Runtime-assigned id of this thread; the value to pass to ``send_message``."""

    thread_name: str | None
    """Human-readable name supplied at spawn time (may be ``None``)."""

    status: str
    """Lifecycle status of this thread at snapshot time."""

    input_shape: str
    """Coarse shape of the thread's ``execute`` input signature. Only
    ``"str_prompt"`` threads can receive ``send_message``."""

    parent_id: str | None
    """Id of the parent thread, if this thread was spawned with one."""

    is_self: bool
    """True for the thread whose agent is calling the tool."""


class ListThreadsResult(BaseModel):
    """Return shape of :func:`list_threads`. Frozen; serialises via ``model_dump_json``."""

    model_config = ConfigDict(frozen=True)

    threads: list[ThreadSummary]
    """One entry per thread registered with the coordinator, order coordinator-defined."""


class SendMessageArgs(BaseModel):
    """Arguments of :func:`send_message`; the sole declaration of its schema.

    Exists to be rendered as a schema rather than to be instantiated: adapters
    whose SDK wants a JSON Schema get :data:`SEND_MESSAGE_INPUT_SCHEMA` from it,
    and adapters that infer from a signature declare the same three parameters.
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="Id of the peer thread to send to.")
    message: str = Field(description="Message body delivered as the peer's user turn.")
    mode: SendMessageMode = Field(
        default="wait",
        description=(
            "How this thread relates to the peer's result: 'wait' blocks on the "
            "reply, 'fire_and_forget' discards it, 'continue_then_receive' "
            "delivers it as a later user turn on this thread."
        ),
    )


LIST_THREADS_DESCRIPTION = (
    "List every thread registered with the current coordinator. Returns a JSON "
    "object with a 'threads' array; each entry has 'thread_id', "
    "'thread_name' (may be null), 'status', 'input_shape', 'parent_id' "
    "(may be null), and 'is_self' (true for the calling thread). Use this "
    "to discover peers before calling send_message. Only threads with "
    "'input_shape' == 'str_prompt' can receive send_message calls."
)

SEND_MESSAGE_DESCRIPTION = (
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
)


def _tool_input_schema(model: type[BaseModel]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Render ``model`` as a tool input schema.

    Drops the model-level ``title`` and ``description`` Pydantic derives from the
    class name and docstring: both SDKs carry the tool's name and description
    out of band, and the docstring is written for this file's readers rather than
    for a model reading the wire schema. Per-property descriptions are kept.
    """
    schema = model.model_json_schema()
    for key in ("title", "description"):
        _ = schema.pop(key, None)
    return schema


LIST_THREADS_INPUT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}  # pyright: ignore[reportExplicitAny]
"""JSON Schema for ``list_threads``, for adapters whose SDK takes one directly.

Written out rather than derived from a model: the tool takes no arguments, so
there is nothing for a second declaration to drift from.
"""

SEND_MESSAGE_INPUT_SCHEMA: dict[str, Any] = _tool_input_schema(SendMessageArgs)  # pyright: ignore[reportExplicitAny]
"""JSON Schema for ``send_message``, for adapters whose SDK takes one directly.

Derived from :class:`SendMessageArgs`, so ``mode`` reaches every runtime as an
enum with a default rather than as a bare string.
"""


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
# matching the existing single-coordinator scope of these tools. Because the
# graph lives here rather than in one adapter, a cycle spanning runtimes — a
# Claude thread and an AI Function waiting on each other — is caught too.
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


# ── Tool bodies ─────────────────────────────────────────────────────────────


async def list_threads(coordinator: Coordinator, self_thread_id: str) -> ListThreadsResult:
    """Snapshot every thread registered with ``coordinator``.

    Args:
        coordinator: Coordinator to enumerate.
        self_thread_id: Id of the calling thread, marked ``is_self``.

    Returns:
        One :class:`ThreadSummary` per registered thread, order
        coordinator-defined.
    """
    infos = await coordinator.list_threads()
    return ListThreadsResult(
        threads=[
            ThreadSummary(
                thread_id=str(info.thread_id),
                thread_name=info.thread_name,
                status=str(info.status),
                input_shape=str(info.input_shape),
                parent_id=None if info.parent_id is None else str(info.parent_id),
                is_self=str(info.thread_id) == self_thread_id,
            )
            for info in infos
        ],
    )


async def send_message(
    coordinator: Coordinator,
    self_thread_id: str,
    thread_id: str,
    message: str,
    mode: str = "wait",
) -> str:
    """Dispatch ``message`` to ``thread_id`` according to ``mode``.

    Every failure is reported as an ``"error: ..."`` string rather than raised:
    the caller is a language model, and a returned message it can act on beats
    an exception that aborts its turn.

    ``mode`` is typed as ``str``, not :data:`SendMessageMode`, and validated at
    the bottom of this function. Adapters publish the enum in their schema, but
    a transport that lets an unconstrained value through must still get a usable
    answer instead of a crash.

    Args:
        coordinator: Coordinator hosting both threads.
        self_thread_id: Id of the calling thread.
        thread_id: Id of the peer to send to.
        message: Message body delivered as the peer's user turn.
        mode: One of :data:`SEND_MESSAGE_MODES`.

    Returns:
        The peer's reply for ``"wait"``, an acknowledgement for the two
        non-blocking modes, or an ``"error: ..."`` description.
    """
    if thread_id == self_thread_id:
        return "error: cannot send_message to self"
    try:
        peer_info = await coordinator.get_thread_info(ThreadId(thread_id))
    except Exception:  # noqa: BLE001 - any lookup failure means "no such peer"
        return f"error: no thread with id {thread_id}"
    if peer_info.input_shape != InputShape.STR_PROMPT:
        return (
            f"error: thread {thread_id} has input_shape={peer_info.input_shape!s}; "
            "send_message requires a str_prompt peer."
        )

    peer = coordinator.get_handle(ThreadId(thread_id))

    if mode == "wait":
        # A blocking wait enqueues a cycle behind the peer's single serial
        # dispatcher and suspends this cycle until it drains. That is only a
        # true deadlock when the peer is (directly or transitively) already
        # waiting back on us: then neither dispatcher can drain the other.
        # Waiting on a merely-busy peer that is *not* waiting on us is safe —
        # it finishes its work, then runs our enqueued cycle. So we refuse
        # only when committing to this wait would close a cycle in the
        # waits-for graph. See ``_would_close_wait_cycle`` above.
        coord_key = id(coordinator)
        if _would_close_wait_cycle(coord_key, self_thread_id, thread_id):
            return (
                f"error: thread {thread_id} is already waiting on this thread; "
                "send_message(mode='wait') would deadlock. Use "
                "mode='fire_and_forget' or mode='continue_then_receive' instead."
            )
        # Register our outstanding edge before awaiting, with no intervening
        # ``await`` — so a peer that tries to wait back on us observes it and
        # refuses (breaking the cycle on exactly one side).
        _wait_edges.setdefault(coord_key, {})[self_thread_id] = thread_id
        try:
            result = await peer.run(message)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            return f"error: {exc}"
        finally:
            _release_wait_edge(coord_key, self_thread_id)
        return str(result)

    if mode == "fire_and_forget":
        fut = peer.run(message)

        async def _swallow() -> None:
            try:
                _ = await fut
            except Exception:  # noqa: BLE001 - the reply is discarded by design
                pass

        _ = asyncio.create_task(_swallow())
        return f"ok: dispatched to {thread_id}"

    if mode == "continue_then_receive":
        self_id = ThreadId(self_thread_id)
        try:
            self_info = await coordinator.get_thread_info(self_id)
        except Exception:  # noqa: BLE001 - caller vanished mid-cycle
            return "error: calling thread is no longer registered"
        if self_info.input_shape != InputShape.STR_PROMPT:
            return (
                "error: continue_then_receive requires this thread to "
                "have input_shape='str_prompt'. Use mode='wait' instead."
            )
        sender = coordinator.get_handle(self_id)
        fut = peer.run(message)

        async def _notify_on_complete() -> None:
            try:
                peer_result = await fut
                notification = f"[Reply from {thread_id}] {peer_result}"
            except Exception as exc:  # noqa: BLE001 - reported to the sender as text
                notification = f"[Reply from {thread_id}] error: {exc}"
            try:
                _ = sender.run(notification)
            except Exception:  # noqa: BLE001 - sender may be gone; nothing to do
                pass

        _ = asyncio.create_task(_notify_on_complete())
        return f"ok: dispatched to {thread_id}; reply will arrive as a new user turn"

    return f"error: unknown mode {mode!r}; valid modes are 'wait', 'fire_and_forget', 'continue_then_receive'"
