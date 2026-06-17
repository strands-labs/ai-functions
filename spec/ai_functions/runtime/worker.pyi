"""Worker: process-local execution engine.

A worker owns one asyncio dispatcher task per thread it hosts, plus the
queues, signals, and live ``Thread`` instances needed to drive them. It
registers itself with a :class:`Coordinator` and serves operations that
the coordinator routes to it via the :class:`WorkerAdapter` protocol.

A concrete :class:`LocalWorker` is provided. Users who want a different
execution strategy (multi-process, thread-pool-backed, etc.) can write
their own concrete class that implements ``WorkerAdapter``; there is no
``Worker`` protocol beyond the adapter.

Invariants:
    - I1 — the worker is the sole host for every thread it registers.
    - I4 — the dispatcher task for a thread is created only after all
      per-thread state (and any seeded history) is fully populated.
    - I5 — the dispatcher is the sole emitter of lifecycle events
      (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``, ``RESULT``).
    - I12 — if a dispatcher task dies with an uncaught exception,
      every pending ``PromptRequest`` future on that thread MUST be
      failed with that exception. No caller of ``handle.run(...)`` is
      allowed to hang because the dispatcher crashed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

from ..handle import ThreadHandle
from ..protocols import Coordinator, Spawnable
from ..types import ThreadId, WorkerId


# ── Work items ──


@dataclass
class PromptRequest[T]:
    """A ``handle.run(*args, **kwargs)`` request carried through the work queue."""

    args: tuple[object, ...]
    kwargs: dict[str, object]
    future: asyncio.Future[T]


class TerminateAfterIdle:
    """Graceful termination marker; dispatcher runs teardown and exits on dequeue."""


WorkItem = PromptRequest[Any] | TerminateAfterIdle  # pyright: ignore[reportExplicitAny]


# ── WorkerAdapter protocol ──


@runtime_checkable
class WorkerAdapter(Protocol):
    """Contract the coordinator uses to reach a worker.

    Public extension point: any class satisfying this protocol can act
    as a worker behind a :class:`Coordinator`. The library ships
    :class:`LocalWorker` as the default in-process implementation and
    builds an internal shim for remote workers served through the
    network endpoint. Most users never implement this protocol
    themselves — :class:`LocalWorker` is expected to cover in-process
    execution, and the network layer covers cross-process execution.

    Every method takes a ``thread_id`` the adapter is expected to host
    (the coordinator guarantees this via its registration table; the
    adapter may raise ``ThreadNotFoundError`` if asked to operate on a
    thread it doesn't host).
    """

    @property
    def worker_id(self) -> WorkerId:
        """Stable id assigned to this worker at registration time."""
        ...

    async def spawn[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId,
        thread_name: str | None,
        parent_id: ThreadId | None,
        metadata: dict[str, object],
    ) -> None:
        """Allocate per-thread state on this worker for ``thread_id``.

        Invoked by ``Coordinator.spawn`` after the coordinator has
        allocated the thread id and registered the ``ThreadInfo``.
        The adapter builds the live ``Thread`` from ``target``
        (in-process ``target.to_thread()`` for ``LocalWorker``;
        cloudpickle-then-``to_thread`` for remote workers), allocates
        its work queue and pause/cancel signals, and starts the
        dispatcher task.

        The coordinator, not the adapter, is responsible for calling
        ``register_thread`` and ``copy_events``; the adapter is only
        responsible for the worker-local per-thread state and the
        dispatcher lifecycle (I4).

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            thread_id: Id the coordinator has already allocated.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
            metadata: Application metadata attached to the thread.
        """
        ...

    def submit(
        self,
        thread_id: ThreadId,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on ``thread_id``'s work queue.

        Args:
            thread_id: Thread whose queue receives the request.
            args: Positional arguments forwarded to ``Thread.execute``.
            kwargs: Keyword arguments forwarded to ``Thread.execute``.

        Returns:
            A future the dispatcher resolves once the cycle completes.

        Concurrency:
            Synchronous enqueue; the cycle runs later on the dispatcher.

        Liveness (I12):
            The returned future MUST eventually resolve. If the
            dispatcher task dies with an uncaught exception, the
            adapter is responsible for failing this future (and every
            other pending future on ``thread_id``) with that exception
            rather than leaving callers awaiting forever.
            :class:`LocalWorker` attaches a done-callback on the
            dispatcher task to enforce this; remote adapters must
            surface the equivalent signal.
        """
        ...

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``'s live instance.

        Forwards to ``Thread.notify``. No cycle is started by
        this call.

        Args:
            thread_id: Thread receiving the message.
            text: Message body delivered to the live thread.
        """
        ...

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle on ``thread_id``.

        Args:
            thread_id: Thread whose in-flight cycle should be cancelled.
        """
        ...

    async def pause(self, thread_id: ThreadId) -> None:
        """Set ``thread_id``'s pause signal.

        Args:
            thread_id: Thread to pause.
        """
        ...

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear ``thread_id``'s pause signal.

        Args:
            thread_id: Thread to resume.
        """
        ...

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination behind currently-queued work.

        Args:
            thread_id: Thread to terminate.
        """
        ...

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear ``thread_id`` down immediately; cancel in-flight, drop queued work.

        Args:
            thread_id: Thread to tear down.
        """
        ...

    async def get_fork_spawnable(self, thread_id: ThreadId) -> Spawnable[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a resumption spawnable for ``thread_id``.

        Delegates to the live ``Thread.fork()``. The returned spawnable
        is safe to pass to ``Coordinator.spawn`` with ``seed_from`` to
        materialize the forked thread.

        Args:
            thread_id: Source thread whose fork spawnable is requested.

        Returns:
            A spawnable that, when instantiated, resumes the source
            thread's non-event state (if any).

        Raises:
            NotImplementedError: The thread does not support forking.
        """
        ...

    def resolve_approval(
        self,
        thread_id: ThreadId,
        approval_id: str,
        decision: str,
    ) -> bool:
        """Resolve a pending tool-approval future inside this worker.

        Invoked by the coordinator when a user calls
        ``Coordinator.resolve_approval``. The adapter completes the
        pending future held by its ``on_interrupt`` handler; the
        coordinator then appends the ``ApprovalDecidedEvent``.

        Args:
            thread_id: Thread whose approval future is being resolved.
            approval_id: Id of the ``APPROVAL_REQUEST`` this resolves.
            decision: Policy decision to return to the executor.

        Returns:
            True iff a matching pending future was found and resolved.
        """
        ...


# ── LocalWorker — default in-process implementation ──


class LocalWorker(WorkerAdapter):
    """In-process thread execution engine.

    Owns asyncio dispatcher tasks, per-thread work queues, pause/cancel
    signals, and the live ``Thread`` instances it hosts. Registers with
    the coordinator on first use (or explicitly via :meth:`register`)
    and deregisters on :meth:`close`.

    User-facing surface is narrow: construct with a coordinator, use
    :meth:`spawn_locally` for the in-process escape hatch (no
    serialization — the spawnable is passed by reference), call
    :meth:`close` to tear the worker down. Everything else flows through
    the coordinator.

    Args:
        coordinator: The coordinator this worker will register with.
        worker_id: Optional explicit id; a uuid is minted otherwise.

    Ensures:
        - Construction is pure: no I/O, no coroutine scheduling.
        - ``self`` is not yet registered with ``coordinator``.
        - No threads are hosted until :meth:`spawn_locally` or
          :meth:`spawn` is called.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        *,
        worker_id: WorkerId | None = None,
    ) -> None: ...

    @property
    def coordinator(self) -> Coordinator:
        """The coordinator this worker is registered with."""
        ...

    async def register(self) -> Self:
        """Register this worker with the coordinator.

        Idempotent. Called automatically on the first
        :meth:`spawn_locally` / :meth:`spawn`; callers may invoke it
        eagerly to surface registration failures before a spawn is
        attempted.

        Returns:
            ``self``, so construction and registration can be chained
            (e.g. ``worker = await LocalWorker(coord).register()``).

        Ensures:
            After return, ``self.worker_id`` is in the coordinator's
            worker pool.
        """
        ...

    # ── User-facing spawn (no serialization) ──

    async def spawn_locally[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId | None = None,
        thread_name: str | None = None,
        parent_id: ThreadId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ThreadHandle[P, T]:
        """Host a new thread on this worker and return a handle to it.

        The escape hatch for spawnables that cannot survive
        cloudpickling: ``target`` is passed by reference, never
        serialized. The created thread is registered with
        ``self.coordinator`` before the dispatcher starts (I4); the
        returned handle is backed by that coordinator.

        Named ``spawn_locally`` (rather than ``spawn``) to avoid
        collision with the ``WorkerAdapter.spawn`` method the coordinator
        calls when routing a generic ``Coordinator.spawn`` request.

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            thread_id: Explicit id; one is minted if omitted.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
            metadata: Application metadata attached to the thread.

        Returns:
            A ``ThreadHandle`` whose coordinator is ``self.coordinator``.

        Ensures:
            - ``self.coordinator`` knows about the thread before the
              dispatcher starts (I4).
            - The dispatcher task is running, blocked on ``q.get``.
        """
        ...

    def has_thread(self, thread_id: ThreadId) -> bool:
        """Return True iff this worker currently hosts ``thread_id``.

        Args:
            thread_id: Thread to probe.

        Returns:
            True if ``thread_id`` is hosted here, else False.
        """
        ...

    async def close(self) -> None:
        """Gracefully terminate every thread hosted by this worker.

        Each thread is ``terminate``-d (its ``TerminateAfterIdle`` flow
        runs, which tears down any local resources via
        ``Thread.teardown``). After every thread's dispatcher exits,
        the worker deregisters itself from the coordinator.

        Ensures:
            - No thread hosted by this worker remains registered with
              the coordinator.
            - ``self`` is deregistered from the coordinator's worker
              pool.

        Concurrency:
            Idempotent; closing an already-closed worker is a no-op.
        """
        ...
