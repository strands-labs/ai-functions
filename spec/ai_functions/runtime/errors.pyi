"""Errors specific to runtime execution: dispatcher and distributed transport."""

from __future__ import annotations

from typing import Literal

from ..types import EventKind, ThreadId, WorkerId


class ThreadIdMismatchError(ValueError):
    """An event's ``thread_id`` disagrees with the routing context.

    ``worker._route_event`` stamps each event with the thread id of
    the cycle it's being emitted from. If the caller also set
    ``event.thread_id`` and the two disagree, the call is almost
    certainly a bug — refusing it surfaces the mismatch at the offending
    frame.

    Args:
        event_thread_id: ``event.thread_id`` as the caller set it.
        routing_thread_id: ``thread_id`` ``_route_event`` was called with.

    Invariants:
        Events stored via a ``Coordinator`` always carry a ``thread_id``
        that matches the thread whose log they live in.
    """

    event_thread_id: ThreadId
    routing_thread_id: ThreadId

    def __init__(self, event_thread_id: ThreadId, routing_thread_id: ThreadId) -> None: ...


class EventEmissionError(RuntimeError):
    """An event of the wrong kind was routed through ``worker._route_event``.

    The runtime routing gates all ``append_event`` calls through a single routing
    function that knows who the emitter is:

    - ``source="thread"`` — calls from inside a cycle, delivered via
      ``ThreadContext.on_event``. These may NOT emit lifecycle events
      (``STARTED``, ``COMPLETED``, ``FAILED``, ``CANCELLED``, ``RESULT``);
      lifecycle emission is the dispatcher's job (I5).
    - ``source="runtime"`` — calls from the runtime itself (dispatcher,
      message router, approval handling, session management). These may
      NOT emit ``MESSAGE_USER`` events; user-message emission is the
      bridge hook's job (I7).

    Args:
        kind: The ``EventKind`` the caller tried to emit.
        thread_id: Id of the thread whose log the event targeted.
        source: Where the call came from (``"runtime"`` or ``"thread"``).

    Invariants:
        I5, I7.
    """

    kind: EventKind
    thread_id: ThreadId
    source: Literal["runtime", "thread"]

    def __init__(
        self,
        kind: EventKind,
        thread_id: ThreadId,
        source: Literal["runtime", "thread"],
    ) -> None: ...


class ThreadNotFoundError(KeyError):
    """The coordinator has no thread registered under the given id.

    Raised by every ``Coordinator`` method that takes a ``thread_id``
    and must route to a worker. Callers that want an Option-style
    lookup should catch this explicitly.

    Args:
        thread_id: The id that was not found.
    """

    thread_id: ThreadId

    def __init__(self, thread_id: ThreadId) -> None: ...


class DistributedError(Exception):
    """Base error for distributed execution failures."""


class WorkerLostError(DistributedError):
    """Worker process crashed or disconnected.

    Args:
        worker_id: Identifier of the worker that was lost.
        thread_ids: Ids of threads previously hosted on this worker.
    """

    worker_id: WorkerId
    thread_ids: list[ThreadId]

    def __init__(self, worker_id: WorkerId, thread_ids: list[ThreadId]) -> None: ...


class SerializationError(DistributedError):
    """Spawnable could not be cloudpickled for remote execution.

    Args:
        function_name: Name of the spawnable that failed to serialise.
        reason: Cloudpickle error text.
    """

    function_name: str

    def __init__(self, function_name: str, reason: str) -> None: ...


class ConnectionLostError(DistributedError):
    """WebSocket connection to the server was lost and reconnection failed.

    Args:
        url: WebSocket URL that could not be reached.
        retries: Number of failed reconnection attempts.
    """

    url: str
    retries: int

    def __init__(self, url: str, retries: int) -> None: ...
