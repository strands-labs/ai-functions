"""Runtime-maintained lifecycle status for registered threads."""

from __future__ import annotations

import enum
from pydantic import BaseModel

from .ids import ThreadId, WorkerId


class ThreadStatus(enum.StrEnum):
    """Runtime-maintained lifecycle state of a registered thread.

    Lifecycle:
        NOT_STARTED -> RUNNING -> {IDLE, PAUSED, CANCELLED} -> {TERMINATED, FAILED}.
    """

    NOT_STARTED = "not_started"
    """Thread is registered; no cycle has run yet."""

    RUNNING = "running"
    """Dispatcher is currently awaiting ``Thread.execute``."""

    IDLE = "idle"
    """Dispatcher is blocked on an empty queue after at least one cycle."""

    PAUSED = "paused"
    """Pause signal is set; queued work is not consumed until resume."""

    CANCELLED = "cancelled"
    """Last cycle ended via cooperative cancel; thread stays registered."""

    TERMINATED = "terminated"
    """Thread has been torn down; handle is no longer usable. Terminal."""

    FAILED = "failed"
    """Runtime removed the thread after an unrecoverable failure. Terminal."""

    @property
    def is_done(self) -> bool:
        """Whether this status is terminal.

        Returns:
            ``True`` iff the thread has been removed from the runtime.
        """
        ...


class InputShape(enum.StrEnum):
    """Coarse classification of a thread's ``execute`` input signature.

    Fine enough for the runtime-facing LLM tools to decide whether a
    thread is "chat-shaped" (accepts one string), fine-grained typed,
    or takes no arguments. Not intended to describe the full signature —
    if callers need structural detail, they should add a separate
    ``input_schema`` field in a later revision.
    """

    STR_PROMPT = "str_prompt"
    """Thread takes exactly one positional ``str`` argument.

    Chat-style peers; eligible for ``send_message`` tool invocations.
    """

    STRUCTURED = "structured"
    """Thread takes one or more typed positional arguments that are not
    ``str``. Not eligible for the default ``send_message`` tool; callers
    must invoke the thread through its typed ``run`` entry point."""

    NO_ARGS = "no_args"
    """Thread takes no positional arguments — a pure one-shot."""


class ThreadInfo(BaseModel):
    """Static snapshot of a thread's identity, host, and current status.

    Produced by :meth:`Coordinator.list_threads` and
    :meth:`Coordinator.get_thread_info` for discovery and introspection
    — both by application code and by the runtime-facing tools exposed
    to LLM agents. Pydantic frozen model so it round-trips over the
    wire via ``model_dump_json`` / ``model_validate_json``.
    """

    thread_id: ThreadId
    """Runtime-assigned id of this thread."""

    worker_id: WorkerId
    """Id of the worker that hosts this thread. The coordinator uses
    this to route operations via the worker's ``WorkerAdapter``."""

    thread_name: str | None
    """Human-readable name supplied at spawn time (may be ``None``)."""

    input_shape: InputShape
    """Coarse shape of the thread's ``execute`` input signature.

    Provided by the thread at spawn time; never changes over the
    thread's lifetime.
    """

    status: ThreadStatus
    """Runtime-maintained lifecycle status of this thread at snapshot time."""

    parent_id: ThreadId | None = None
    """Id of the parent thread, if this thread was spawned with one."""
