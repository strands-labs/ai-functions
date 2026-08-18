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
decorator to a matching signature or hand :data:`SEND_MESSAGE_INPUT_SCHEMA`
straight to their SDK.
"""

from typing import Any, Literal

from pydantic import BaseModel

from ..protocols import Coordinator

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

SendMessageMode = Literal["wait", "fire_and_forget", "continue_then_receive"]
"""How the sender relates to the peer's result."""

SEND_MESSAGE_MODES: tuple[str, ...]
"""The legal :data:`SendMessageMode` values, for runtime validation."""

LIST_THREADS_DESCRIPTION: str
"""Wire-visible description of ``list_threads``, shared by every adapter."""

SEND_MESSAGE_DESCRIPTION: str
"""Wire-visible description of ``send_message``, shared by every adapter."""

LIST_THREADS_INPUT_SCHEMA: dict[str, Any]
"""JSON Schema for ``list_threads``, for adapters whose SDK takes one directly.

Written out rather than derived from a model: the tool takes no arguments, so
there is nothing for a second declaration to drift from.
"""

SEND_MESSAGE_INPUT_SCHEMA: dict[str, Any]
"""JSON Schema for ``send_message``, for adapters whose SDK takes one directly.

Derived from :class:`SendMessageArgs`, so ``mode`` reaches every runtime as an
enum with a default rather than as a bare string.
"""

class ThreadSummary(BaseModel):
    """One entry of :class:`ListThreadsResult`; an agent-facing ``ThreadInfo``.

    Every field is widened to ``str`` because this crosses a tool boundary to a
    language model: :class:`~ai_functions.types.ThreadInfo` carries ``ThreadId``
    newtypes and ``StrEnum`` members, and a model reading the tool result has no
    use for that distinction. Frozen, like ``ThreadInfo``, so it round-trips over
    the wire via ``model_dump_json`` / ``model_validate_json``.
    """

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

    threads: list[ThreadSummary]
    """One entry per thread registered with the coordinator, order coordinator-defined."""

class SendMessageArgs(BaseModel):
    """Arguments of :func:`send_message`; the sole declaration of its schema.

    Exists to be rendered as a schema rather than to be instantiated: adapters
    whose SDK wants a JSON Schema get :data:`SEND_MESSAGE_INPUT_SCHEMA` from it,
    and adapters that infer from a signature declare the same three parameters.
    """

    thread_id: str
    message: str
    mode: SendMessageMode

async def list_threads(coordinator: Coordinator, self_thread_id: str) -> ListThreadsResult:
    """Snapshot every thread registered with ``coordinator``.

    Args:
        coordinator: Coordinator to enumerate.
        self_thread_id: Id of the calling thread, marked ``is_self``.

    Returns:
        One :class:`ThreadSummary` per registered thread, order
        coordinator-defined.
    """
    ...

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

    ``mode`` is typed as ``str``, not :data:`SendMessageMode`, and validated
    inside the body. Adapters publish the enum in their schema, but a transport
    that lets an unconstrained value through must still get a usable answer
    instead of a crash.

    A blocking ``"wait"`` is refused when granting it would close a cycle in the
    graph of in-flight waits, so two peers waiting on each other cannot deadlock
    their dispatchers. Because that graph is held here rather than per adapter, a
    cycle spanning runtimes is caught too.

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
    ...
