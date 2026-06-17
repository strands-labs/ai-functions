"""Coordinator contract and cross-thread extension points.

This module defines the three user-facing protocols of the system:

- :class:`Thread` — the live, runnable instance contract a worker drives.
- :class:`Spawnable` — factory for a ``Thread``; what users pass to
  ``coordinator.spawn`` or ``worker.spawn_locally``.
- :class:`Coordinator` — the authoritative registry, event log, message
  router, and cross-worker dispatcher. Every user-facing operation on a
  thread routes through a coordinator.

The ``Worker`` / ``WorkerAdapter`` story lives in ``ai_functions.runtime.worker``
because it is an execution-engine concern; users rarely interact with
it directly beyond constructing a ``LocalWorker``.

Invariants:
    I1 — every registered thread has exactly one host worker, reachable
    via the adapter stored in the coordinator's routing table.

    I5 — the worker's dispatcher is the sole emitter of lifecycle events
    (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``, ``RESULT``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Protocol, runtime_checkable

from strands.interrupt import Interrupt
from strands.types.interrupt import InterruptResponseContent

from .types import (
    Event,
    EventId,
    EventKind,
    InputShape,
    MessageId,
    Policy,
    ThreadContext,
    ThreadId,
    ThreadInfo,
    ThreadStatus,
    WorkerId,
)

if TYPE_CHECKING:
    from .handle import ThreadHandle
    from .runtime.worker import WorkerAdapter


# ── Callback protocols ──

class OnEventCallback(Protocol):
    """Sink bound to ``Coordinator.append_event`` for a single event."""

    def __call__(self, event: Event) -> None:
        """Forward one event to the coordinator's append path.

        Args:
            event: The event to deliver.

        Ensures:
            ``event`` reaches ``Coordinator.append_event``.
        """
        ...


@runtime_checkable
class Subscription(Protocol):
    """Handle returned by ``Coordinator.on`` that tears down its own registration.

    Also usable as a context manager: exiting the ``with`` block calls ``unsubscribe``
    unconditionally, even on exception.
    """

    def unsubscribe(self) -> None:
        """Tear down this subscription so its callback receives no further events.

        Ensures:
            No further invocation of the registered callback for events appended after
            this call.

        Concurrency:
            Idempotent.
        """
        ...

    def __enter__(self) -> Subscription:
        """Return ``self`` for use in a ``with`` block."""
        ...

    def __exit__(self, *exc: Any) -> None:
        """Call ``unsubscribe`` on context exit.

        Args:
            exc: Exception info tuple from the context manager protocol.
        """
        ...


class OnInterruptCallback(Protocol):
    """Handler for a batch of executor-raised interrupts (e.g., tool approvals)."""

    async def __call__(self, interrupts: list[Interrupt]) -> list[InterruptResponseContent]:
        """Resolve one batch of Strands interrupts to decision responses.

        Args:
            interrupts: One ``strands.interrupt.Interrupt`` per pending approval, in
                executor order.

        Returns:
            One ``InterruptResponseContent`` per interrupt, in the same order; each
            carrying the matching ``interruptId``.

        Emits:
            APPROVAL_REQUEST (one per entry in ``interrupts``).

        Concurrency:
            Awaited inline by the executor; the dispatcher sees no event during the wait.
        """
        ...


# ── Thread — a live, runnable thread instance ──

@runtime_checkable
class Thread[**P, T](Protocol):
    """A live, runnable thread instance the worker drives.

    See Also:
        :class:`ThreadStatus`.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the thread, used for telemetry and error attribution.

        Need not be unique across threads.

        Concurrency:
            Synchronous and side-effect-free.
        """
        ...

    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        """Start one cycle against ``ctx`` with typed arguments and return its result.

        Args:
            ctx: Freshly built per-cycle context; never reused across cycles.
            args: Positional arguments forwarded from ``ThreadHandle.run``.
            kwargs: Keyword arguments forwarded from ``ThreadHandle.run``.

        Returns:
            The typed cycle result on natural completion.

        Requires:
            ``ctx`` is a fresh context built by the worker for this cycle.

        Obligations:
            - At each work boundary, raise ``CancelledError`` if
              ``ctx.cancel_signal`` is set.
            - At each work boundary, await ``ctx.pause_signal`` if set.
            - Invoke ``ctx.on_interrupt`` inline when the executor surfaces interrupts.
            - Observe any side-channel messages delivered via
              :meth:`notify` at a safe boundary of its choosing;
              threads that don't care about injections may ignore them.

        Raises:
            asyncio.CancelledError: ``ctx.cancel_signal`` was set at a work boundary.
        """
        ...

    async def notify(self, text: str) -> None:
        """Deliver a side-channel message into this thread's stream.

        Best-effort notification: the thread decides whether and when to
        surface the message. Typically threads buffer the text internally
        and drain it at a safe boundary inside their next (or in-flight)
        :meth:`execute` cycle. Calling ``notify`` does NOT start a
        cycle; an idle thread remains idle. A running thread may observe
        the message mid-cycle if it chooses.

        Default implementation is a no-op — threads that ignore injected
        messages need not override. Concrete implementations (e.g.
        ``AIThread``) maintain their own buffer.

        Args:
            text: Message body delivered by the worker or an external sender.

        Ensures:
            - No new cycle is started by this call.
            - Observation timing is the thread's choice.

        Concurrency:
            Awaited inline by the caller (worker delivery path). Must not
            block on network I/O.
        """
        ...

    def serialize_result(self, result: T) -> str:
        """Encode ``result`` as a string for storage in a ``ResultEvent``.

        Args:
            result: The typed cycle result to serialize.

        Returns:
            A string representation of ``result`` that
            :meth:`deserialize_result` can round-trip.

        Ensures:
            ``deserialize_result(serialize_result(result))`` is equal to ``result``.

        Concurrency:
            Synchronous and side-effect-free.
        """
        ...

    def deserialize_result(self, payload: str) -> T:
        """Recover a result from the string stored in a ``ResultEvent``.

        Args:
            payload: Value previously produced by :meth:`serialize_result`.

        Returns:
            The deserialized result of type ``T``.

        Raises:
            AIFunctionError: The payload is malformed or cannot be
                decoded as ``T``.
        """
        ...

    async def fork(self) -> Spawnable[P, T]:
        """Return a ``Spawnable`` that, when instantiated, resumes this thread's state.

        The coordinator uses this to build the forked thread's live
        instance. The returned spawnable MUST be safe to pass to
        :meth:`Coordinator.spawn` with ``seed_from=source_id``; the
        coordinator then seeds the new thread's event log before the
        dispatcher starts.

        For threads whose only state is the event log (e.g. :class:`AIThread`)
        this is a one-liner — the original template is stateless and
        reusing it is correct. Threads with external state (a CLI
        subprocess, a remote session, a checkpointed workspace) must
        return a spawnable that carries the resumption payload so that
        :meth:`Spawnable.to_thread` produces a live instance equivalent
        to ``self`` at the moment of the call.

        Returns:
            A spawnable whose :meth:`Spawnable.to_thread` produces a
            fresh live instance equivalent to ``self`` — the event log
            then diverges from the parent via the dispatcher.

        Ensures:
            The returned spawnable's ``to_thread()`` produces a live
            instance whose non-event state (if any) is equivalent to
            ``self``'s at the moment ``fork`` was called.

        Raises:
            NotImplementedError: This thread type does not support
                forking.
        """
        ...

    async def teardown(self) -> None:
        """Release any resources this thread holds; called on termination.

        Invoked once by :meth:`Coordinator.terminate` (after the dispatcher
        drains its queue and reaches the termination marker) and by
        :meth:`Coordinator.terminate_now` (after the in-flight task is
        cancelled). No cycle is in flight at the time of the call; no
        further :meth:`execute` will be invoked on this instance.

        Threads with no external state (e.g. :class:`AIThread`) may leave
        the default no-op implementation. Threads owning a subprocess,
        network connection, file handle, or remote session should close
        them here. Threads that buffer injected messages should drop the
        buffer here.

        Ensures:
            - Resources owned by this thread are released.
            - Failures are the thread's responsibility; raising from
              ``teardown`` is allowed but will surface as an unhandled
              exception in the dispatcher's teardown path.

        Concurrency:
            Awaited inline by the worker during teardown; not racing with
            ``execute``.
        """
        ...


# ── Spawnable — factory for a Thread ──

@runtime_checkable
class Spawnable[**P, T](Protocol):
    """Factory for a ``Thread``; input to ``coordinator.spawn`` / ``worker.spawn_locally``."""

    def to_thread(self) -> Thread[P, T]:
        """Produce a fresh live ``Thread`` bound to a single spawn."""
        ...

    @property
    def input_shape(self) -> InputShape:
        """Coarse classification of this spawnable's ``execute`` input signature.

        The coordinator stores this on :class:`ThreadInfo` at spawn time
        so tools (and clients) can discover which threads are chat-style
        peers. See :class:`InputShape` for the enum values.
        """
        ...


# ── Coordinator — broker, registry, event log, and cross-worker router ──

@runtime_checkable
class Coordinator(Protocol):
    """The authoritative broker of the system: registry + event log + router.

    The coordinator is the sole user-facing object for operations that
    take a ``thread_id``. It holds:

    - A registry of workers (``dict[WorkerId, WorkerAdapter]``).
    - A registry of threads (``dict[ThreadId, ThreadInfo]``), each
      carrying a ``worker_id`` the coordinator uses to route operations
      to the hosting worker's adapter.
    - The durable event log, through which ``append_event`` writes and
      ``get_events`` / ``on`` subscribers read.
    - Pause signals per-thread, driven by ``TOKEN_USAGE`` events.
    - The approval-decision routing path (approvals are events).

    Cross-process distribution is implemented by a ``RemoteCoordinator``
    client that proxies the full protocol over a network transport to
    a :class:`~ai_functions.network.CoordinatorEndpoint` fronting an in-memory
    coordinator on the server. Clients that connect to the same endpoint
    observe the same threads, events, and workers.

    Invariants:
        I1, I2, I3, I5, I11, I12.

        I11 (side-effect isolation): once :meth:`append_event` has
        durably stored an event, any failure in its side effects
        (status caching, subscriber fan-out, rate-limit hooks) MUST
        NOT propagate to the caller. The caller is typically a worker
        dispatcher task, and propagating would kill the thread and
        cause downstream ``handle.run(...)`` futures to hang. Failures
        must be logged and swallowed; subsequent side effects still run.

        I12 (liveness under failure): a :class:`Coordinator` MUST
        guarantee that every pending operation (a ``handle.run`` future,
        a pending ``spawn``, etc.) eventually resolves — either
        successfully or by raising — even if the backing transport,
        worker, or internal task dies. No code path may silently
        orphan a caller's await. Concretely, if a worker's dispatcher
        or a wire connection fails, the coordinator is responsible for
        failing pending futures with the root exception (or a clear
        surrogate like ``ConnectionClosedError``).
    """

    # ── Worker pool ──

    async def register_worker(self, adapter: WorkerAdapter) -> None:
        """Register a worker with this coordinator.

        Called by a worker at construction time. The coordinator records
        the adapter keyed by ``adapter.worker_id`` and uses it to route
        operations for any thread whose ``info.worker_id`` matches.

        Args:
            adapter: The adapter the coordinator will invoke for every
                operation on threads hosted by this worker.

        Ensures:
            Subsequent :meth:`register_thread` calls citing this worker
            will find an adapter to route to.

        Raises:
            ValueError: A worker is already registered under the same
                ``worker_id``.
        """
        ...

    async def deregister_worker(self, worker_id: WorkerId) -> None:
        """Remove a worker from the pool.

        All threads hosted by that worker are implicitly removed from
        the routing table as part of the worker's own ``close`` flow;
        this method should be called after the worker's threads are
        already terminated.

        Args:
            worker_id: Id of the worker to remove.

        Concurrency:
            Idempotent; removing an unknown id is a no-op.
        """
        ...

    # ── Thread registry ──

    async def register_thread(self, info: ThreadInfo) -> None:
        """Register a thread with this coordinator.

        Called by the hosting worker after allocating the thread's
        state but before starting its dispatcher. ``info.worker_id``
        must point at a worker already registered via
        :meth:`register_worker`.

        Args:
            info: Full snapshot of the thread's identity and host.

        Ensures:
            - Subsequent operations on ``info.thread_id`` route to the
              worker identified by ``info.worker_id``.
            - :meth:`list_threads` and :meth:`get_thread_info` return
              this info.
            - :meth:`notify`, :meth:`submit`, and peers find the
              thread.

        Raises:
            ValueError: ``info.worker_id`` is not a registered worker.
        """
        ...

    async def deregister_thread(self, thread_id: ThreadId) -> None:
        """Remove ``thread_id`` from the registry.

        Called by the hosting worker on thread teardown. Events for the
        thread remain in the log for audit and replay.

        Args:
            thread_id: Id of the thread being removed.

        Concurrency:
            Idempotent; removing an unknown id is a no-op.
        """
        ...

    # ── Discovery ──

    async def list_threads(self) -> list[ThreadInfo]:
        """Return a snapshot of every registered thread.

        Returns:
            One :class:`ThreadInfo` per registered thread, order
            implementation-defined.
        """
        ...

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        """Return the full info snapshot for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            The :class:`ThreadInfo` for this thread.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        ...

    def get_handle(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:
        """Return a handle bound to this coordinator for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            A typed-erased :class:`ThreadHandle`. Callers who know the
            expected ``P, T`` can annotate at the call site to narrow.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        ...

    async def get_thread_status(self, thread_id: ThreadId) -> ThreadStatus:
        """Return the current status of ``thread_id``.

        Served from the coordinator's own registry, which tracks status
        by subscribing to lifecycle events emitted by the hosting
        worker's dispatcher.

        Args:
            thread_id: Thread to query.

        Returns:
            The cached :class:`ThreadStatus` for this thread.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        ...

    # ── Spawning ──

    async def spawn[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        seed_from: ThreadId | None = None,
        seed_events: list[Event] | None = None,
        worker_id: WorkerId | None = None,
        thread_id: ThreadId | None = None,
        thread_name: str | None = None,
        parent_id: ThreadId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ThreadHandle[P, T]:
        """Create a new thread on a registered worker, optionally pre-seeded.

        Worker selection:

        - If ``worker_id`` is given, the named worker hosts the thread;
          raises if no such worker is registered.
        - Otherwise, the coordinator selects one by its default policy
          (implementation-defined; the in-memory coordinator picks the
          first registered worker).

        ``target`` must be cloudpickle-safe if the coordinator may
        dispatch to a remote worker. Clients that need to spawn a
        non-picklable thread must go through
        :meth:`LocalWorker.spawn_locally` directly on a local worker —
        that is the in-process escape hatch and never crosses a wire.

        Seeding (optional):

        - ``seed_from=source_id``: copy every event from the source
          thread's log into the new thread's log, rewriting
          ``thread_id`` and minting fresh event ids. Used for forking
          (see :meth:`fork`) and for resuming a thread from another
          live thread on the same coordinator.
        - ``seed_events=[...]``: rehydrate from an explicit event list
          — for example, a persisted snapshot.
        - At most one of ``seed_from`` / ``seed_events`` may be given.
          Passing neither produces a fresh thread with an empty log.

        Ordering (atomic w.r.t. external observers):

        1. Mint ``thread_id`` if not provided.
        2. Seed the event log bucket (when applicable).
        3. Register the ``ThreadInfo`` — the thread now appears in
           :meth:`list_threads`, :meth:`get_events` returns the seeded
           log, and peers may ``submit`` / ``notify``.
        4. Call ``WorkerAdapter.spawn`` on the hosting worker, which
           allocates per-thread state and starts the dispatcher.

        External observers never see a registered thread with an empty
        log when seeding was requested: the seed is always in place
        before registration completes.

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            seed_from: Source thread whose event log is copied onto the
                new thread. Mutually exclusive with ``seed_events``.
            seed_events: Explicit event list copied onto the new
                thread's log. Mutually exclusive with ``seed_from``.
            worker_id: Explicit worker selection; coordinator-default if ``None``.
            thread_id: Explicit id; one is minted if omitted.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
                No magic default — callers who want ``seed_from`` to
                also be the parent must pass it explicitly.
            metadata: Application metadata attached to the thread.

        Returns:
            A ``ThreadHandle`` whose coordinator is ``self``.

        Raises:
            ValueError: Both ``seed_from`` and ``seed_events`` are
                given; ``worker_id`` is given but not registered; no
                workers are registered; or ``seed_from`` is not a
                registered thread.
            SerializationError: ``target`` cannot be cloudpickled and
                the chosen worker is remote.
        """
        ...

    # ── Cross-thread operations (routed to hosting worker's adapter) ──

    def submit(
        self,
        thread_id: ThreadId,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Awaitable[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on ``thread_id``'s work queue.

        Args:
            thread_id: Id of the thread to run.
            args: Positional arguments forwarded to ``Thread.execute``.
            kwargs: Keyword arguments forwarded to ``Thread.execute``.

        Returns:
            An awaitable (typically an ``asyncio.Future``) that resolves
            with the cycle's typed result. The coordinator routes the
            request to the hosting worker's adapter.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.

        Concurrency:
            In-process: synchronous enqueue; future resolves later.
            Remote: the submit frame is dispatched over the wire; the
            returned awaitable resolves when the server relays the
            result back.

        Liveness (I12):
            The returned awaitable MUST eventually resolve — either
            with the cycle result, with the cycle's raised exception,
            with ``CancelledError`` on cancel / teardown, or with a
            transport / dispatcher error if the backing worker or
            connection dies. Implementations may NOT orphan the
            awaitable on any internal failure mode.
        """
        ...

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``.

        Routes to the hosting worker's ``notify`` method, which
        calls the live thread's ``Thread.notify``. No cycle is
        started by this call.

        Args:
            thread_id: Thread receiving the message.
            text: Message body delivered to the live thread.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def pause(self, thread_id: ThreadId) -> None:
        """Set the pause signal on ``thread_id``.

        Args:
            thread_id: Thread to pause.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear the pause signal on ``thread_id``.

        Args:
            thread_id: Thread to resume.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle on ``thread_id``.

        Args:
            thread_id: Thread to cancel.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination of ``thread_id``.

        Args:
            thread_id: Thread to terminate.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear ``thread_id`` down immediately.

        Args:
            thread_id: Thread to tear down.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def fork(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:
        """Fork ``thread_id`` into a new thread seeded with its history.

        Thin sugar built on :meth:`spawn`:

        1. Ask the hosting worker for a resumption spawnable via
           ``WorkerAdapter.get_fork_spawnable`` (which delegates to
           ``Thread.fork()``).
        2. Call ``self.spawn(new_spawnable, seed_from=thread_id,
           parent_id=thread_id)``.

        Callers who need finer control (different seeding source,
        different parent id, placing the child on a specific worker)
        should use :meth:`spawn` directly.

        Args:
            thread_id: Thread to fork.

        Returns:
            A new handle (type-erased) backed by this coordinator.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
            NotImplementedError: The source thread's ``fork`` raises.
        """
        ...

    # ── Approvals ──

    async def resolve_approval(
        self, thread_id: ThreadId, approval_id: str, decision: str,
    ) -> None:
        """Resolve a pending tool-approval request.

        Appends an ``ApprovalDecidedEvent`` that the worker's waiting
        ``on_interrupt`` handler observes via its event subscription.

        Args:
            thread_id: Thread whose approval is being resolved.
            approval_id: Id of the ``APPROVAL_REQUEST`` this resolves.
            decision: Policy decision to return to the executor.

        Emits:
            APPROVAL_DECIDED.
        """
        ...

    # ── Events: the single emit/store/broadcast path ──

    def append_event(self, event: Event) -> None:
        """Persist ``event`` and broadcast it to subscribers.

        Synchronous by design so it can be called from sync callbacks (Strands hooks,
        streaming callback). Implementations must not block on network I/O; remote
        variants schedule the post and return immediately, preserving per-producer
        order via an ordering queue.

        Args:
            event: The event to append; ``event.thread_id`` must be set.

        Ensures:
            - ``event`` is durably stored before any subscriber is notified.
            - Every matching subscriber registered via :meth:`on` observes ``event``.
            - ``TOKEN_USAGE`` events drive this coordinator's own pause-signal state.
            - For any single producer, events are persisted in the
              order ``append_event`` was called (I2, I3).
            - Side-effect isolation (I11): once the event is durably
              stored, any failure in a status-cache update, subscriber
              callback, or rate-limit hook is logged and swallowed.
              ``append_event`` itself raises ONLY for the misuses
              listed under ``Raises`` — never for downstream side-effect
              failures. This is a hard requirement: the common caller
              is a worker dispatcher task, and propagating would kill
              the thread and hang pending ``handle.run`` futures.
            - One failing subscriber MUST NOT prevent subsequent
              subscribers from receiving ``event``.

        Raises:
            ValueError: if ``event.thread_id`` is ``None``.
        """
        ...

    async def get_events(
        self,
        thread_id: ThreadId,
        since_id: EventId | None = None,
        kinds: list[EventKind] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Replay stored events for ``thread_id`` in chronological order.

        Args:
            thread_id: Thread whose events are being read.
            since_id: Return only events appended strictly after this id.
            kinds: Restrict to these event kinds; all kinds if ``None``.
            limit: Return at most this many events; unbounded if ``None``.

        Returns:
            Events for ``thread_id`` matching the filters, oldest first (I8).
        """
        ...

    async def new_message_id(self) -> MessageId:
        """Mint a fresh message id."""
        ...

    async def copy_events(
        self,
        source_id: ThreadId,
        target_id: ThreadId,
        until_event_id: EventId | None = None,
    ) -> None:
        """Copy every event from ``source_id`` onto ``target_id`` in order.

        Used by the coordinator to seed a forked thread's event log
        without streaming events back to the caller.

        Args:
            source_id: Thread whose events are read.
            target_id: Thread whose log receives rewritten copies.
            until_event_id: Copy events up to and including this id;
                copy the entire current history when ``None``.

        Ensures:
            - Each source event is re-appended onto ``target_id`` with a
              fresh ``event.id`` and ``thread_id=target_id``; relative
              order is preserved.
            - No events already on ``target_id`` are mutated or removed.
            - Subscribers registered via :meth:`on` are NOT notified for
              the copied events.
            - For remote implementations, outstanding writes on the
              source are flushed before reading (I8).

        Raises:
            ThreadNotFoundError: ``source_id`` or ``target_id`` is not registered.
            ValueError: ``until_event_id`` does not correspond to any
                event on ``source_id``.
        """
        ...

    # ── Event subscriptions (live tail) ──

    def on(
        self,
        callback: OnEventCallback,
        *,
        thread_id: ThreadId | None = None,
        kinds: list[EventKind] | None = None,
    ) -> Subscription:
        """Register a live subscriber for future events.

        Args:
            callback: Sink invoked for each matching event appended after this call.
            thread_id: If set, restrict deliveries to this thread.
            kinds: If set, restrict deliveries to these event kinds.

        Returns:
            A :class:`Subscription` whose ``unsubscribe`` tears down this registration.

        Ensures:
            - ``callback`` is invoked for every event appended after registration whose
              filters match (AND), until ``unsubscribe`` is called.
            - Calling ``unsubscribe`` is idempotent.
        """
        ...

    # ── Rate limiting ──

    async def is_paused(self, thread_id: ThreadId) -> bool:
        """Return whether the rate-limit / manual pause signal is set for ``thread_id``.

        Args:
            thread_id: Thread to query.

        Returns:
            True iff the thread is paused at this instant (I3, I8).
            Callers should await :meth:`wait_until_unpaused` to block.
        """
        ...

    def wait_until_unpaused(self, thread_id: ThreadId) -> Awaitable[None]:
        """Await until the pause signal for ``thread_id`` is clear.

        Args:
            thread_id: Thread whose pause signal to watch.

        Returns:
            An awaitable that resolves the next time the thread is unpaused (immediately
            if already unpaused).

        Ensures:
            Every :meth:`append_event` of ``TOKEN_USAGE`` that transitions this thread
            out of paused state wakes outstanding awaiters (I3).
        """
        ...


# ── Permission Store — tool approval policies ──

@runtime_checkable
class PermissionStore(Protocol):
    """Store and resolve tool permission policies used by ``on_interrupt``."""

    async def resolve_policy(self, thread_id: ThreadId, tool_name: str) -> Policy:
        """Resolve the effective policy for a tool on a thread."""
        ...

    async def set_global_policy(self, tool_name: str, policy: Policy) -> None:
        """Set the global default policy for ``tool_name``."""
        ...

    async def grant_session(self, thread_id: ThreadId, tool_name: str) -> None:
        """Grant ``tool_name`` on ``thread_id`` for the lifetime of the session."""
        ...

    async def revoke_session(self, thread_id: ThreadId, tool_name: str) -> None:
        """Revoke a session-scoped grant previously given by :meth:`grant_session`."""
        ...
