"""InMemoryCoordinator — default single-process ``Coordinator`` implementation.

Invariants:
    I2, I3.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast, final

from ..handle import ThreadHandle
from ..protocols import Coordinator, OnEventCallback, Spawnable, Subscription
from ..types import (
    ApprovalDecidedEvent,
    Event,
    EventId,
    EventKind,
    FailedEvent,
    MessageId,
    ThreadId,
    ThreadInfo,
    ThreadSpawnedEvent,
    ThreadStatus,
    WorkerId,
)
from .errors import ThreadNotFoundError, WorkerLostError
from .prices import PRICES, PriceTable

if TYPE_CHECKING:
    from .worker import WorkerAdapter


_logger: logging.Logger = logging.getLogger("ai_functions.runtime.coordinator")


def _log_side_effect_failure(
    where: str,
    event: Event,
    thread_id: ThreadId,
    exc: BaseException,
) -> None:
    """Log an isolated side-effect failure from ``append_event``.

    The event has already been durably appended; this is a broadcast /
    cache / hook failure that must not propagate. We log with traceback
    so the root cause is still surfaced — we just don't kill the
    producer.
    """
    _logger.exception(
        "append_event side effect failed: where=%s thread=%s event=%s err=%r",
        where,
        thread_id,
        event.kind,
        exc,
    )


async def _wait_event(event: asyncio.Event) -> None:
    """Await an ``asyncio.Event`` and discard its ``True`` return value."""
    await event.wait()


@final
class _Subscription(Subscription):
    """Concrete ``Subscription`` returned by :meth:`InMemoryCoordinator.on`."""

    __slots__ = ("_unsubscribe", "_active")

    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe: Callable[[], None] = unsubscribe
        self._active: bool = True

    def unsubscribe(self) -> None:
        """Tear down this subscription idempotently."""
        if self._active:
            self._active = False
            self._unsubscribe()

    def __enter__(self) -> Subscription:
        """Return ``self`` for use in a ``with`` block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Call ``unsubscribe`` on context exit."""
        del exc
        self.unsubscribe()


class _Subscriber:
    """Internal record for one registered subscriber."""

    __slots__ = ("callback", "thread_id", "kinds")

    def __init__(
        self,
        callback: OnEventCallback,
        thread_id: ThreadId | None,
        kinds: frozenset[str] | None,
    ) -> None:
        self.callback: OnEventCallback = callback
        self.thread_id: ThreadId | None = thread_id
        self.kinds: frozenset[str] | None = kinds

    def matches(self, event: Event) -> bool:
        """Return ``True`` iff this subscriber should receive ``event``."""
        if self.thread_id is not None:
            event_tid: object = getattr(event, "thread_id", None)
            if event_tid != self.thread_id:
                return False
        if self.kinds is not None and str(event.kind) not in self.kinds:
            return False
        return True


class InMemoryCoordinator(Coordinator):
    """Single-process ``Coordinator``; events in per-thread lists, callbacks in dicts.

    Implements:
        Coordinator.

    Args:
        prices: The table that prices each ``TOKEN_USAGE`` event by the model
            it names, for a caller that folds a run's usage into dollars (a
            team totaling its run). :data:`~ai_functions.runtime.prices.PRICES`
            by default; pass ``PRICES.with_prices({...})`` to add or correct a
            model.

    Invariants:
        - I2 — sole sink for every event this coordinator serves.
        - I3 — per-thread pause signals are driven by ``TOKEN_USAGE``
          events appended here.
    """

    __slots__ = (
        "_prices",
        "_workers",
        "_infos",
        "_events",
        "_subscribers",
        "_pause_events",
        "_lost_workers",
    )

    def __init__(self, *, prices: PriceTable = PRICES) -> None:
        self._prices: PriceTable = prices
        self._workers: dict[WorkerId, WorkerAdapter] = {}
        self._infos: dict[ThreadId, ThreadInfo] = {}
        self._events: dict[ThreadId, list[Event]] = {}
        self._subscribers: list[_Subscriber] = []
        self._pause_events: dict[ThreadId, asyncio.Event] = {}
        # Threads orphaned by each lost worker, in the order they were failed.
        # Read by ``_adapter_for`` to name the loss in ``WorkerLostError``.
        self._lost_workers: dict[WorkerId, tuple[ThreadId, ...]] = {}

    @property
    def prices(self) -> PriceTable:
        """The table this coordinator prices ``TOKEN_USAGE`` events with.

        Read by a caller that folds usage itself and must price it the same way
        every time, such as a team totaling its run.
        """
        return self._prices

    # ── Worker pool ─────────────────────────────────────────────────────────

    async def register_worker(self, adapter: WorkerAdapter) -> None:
        """Register a worker with this coordinator."""
        wid = adapter.worker_id
        if wid in self._workers:
            raise ValueError(f"worker {wid!r} already registered")
        self._workers[wid] = adapter

    async def deregister_worker(self, worker_id: WorkerId) -> None:
        """Remove a worker from the pool and fail the threads it still hosted.

        A thread whose host is gone can never run again, so leaving it
        ``idle`` in the registry publishes a thread that ``list_threads``
        offers and every routed operation refuses. Each non-terminal
        thread of ``worker_id`` therefore gets a ``FailedEvent`` naming
        the lost worker and moves to ``ThreadStatus.FAILED``, which is
        terminal — the registry and the routing table agree again.

        Concurrency:
            Idempotent; removing an unknown id is a no-op, and a second
            call finds no non-terminal thread left to fail.
        """
        self._workers.pop(worker_id, None)
        orphans = tuple(
            thread_id
            for thread_id, info in self._infos.items()
            if info.worker_id == worker_id and not info.status.is_done
        )
        # Subscribers run synchronously inside append_event. Record the whole
        # loss before notifying any of them, so callbacks see terminal statuses
        # and WorkerLostError carries every affected thread from the first event.
        if orphans:
            self._lost_workers[worker_id] = orphans
        for thread_id in orphans:
            info = self._infos[thread_id]
            self._infos[thread_id] = info.model_copy(update={"status": ThreadStatus.FAILED})
        for thread_id in orphans:
            self.append_event(
                FailedEvent(
                    thread_id=thread_id,
                    thread_name=self._infos[thread_id].thread_name,
                    error=f"WorkerLostError: worker {worker_id!r} was lost while hosting thread {thread_id!r}",
                )
            )

    # ── Thread registry ─────────────────────────────────────────────────────

    async def register_thread(self, info: ThreadInfo) -> None:
        """Register a thread; ``info.worker_id`` must point at a known worker."""
        if info.worker_id not in self._workers:
            raise ValueError(
                f"worker {info.worker_id!r} is not registered; call register_worker() first",
            )
        self._infos[info.thread_id] = info
        self._events.setdefault(info.thread_id, [])
        unpaused = self._pause_events.setdefault(info.thread_id, asyncio.Event())
        unpaused.set()

    async def deregister_thread(self, thread_id: ThreadId) -> None:
        """Remove ``thread_id`` from the registry; idempotent."""
        self._infos.pop(thread_id, None)
        unpaused = self._pause_events.pop(thread_id, None)
        if unpaused is not None:
            unpaused.set()

    # ── Discovery ───────────────────────────────────────────────────────────

    async def list_threads(self) -> list[ThreadInfo]:
        """Return a snapshot of every registered thread."""
        return list(self._infos.values())

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        """Return the full info snapshot for ``thread_id``."""
        info = self._infos.get(thread_id)
        if info is None:
            raise ThreadNotFoundError(thread_id)
        return info

    def get_handle(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a handle bound to this coordinator for ``thread_id``."""
        if thread_id not in self._infos:
            raise ThreadNotFoundError(thread_id)
        return ThreadHandle(thread_id, self)

    async def get_thread_status(self, thread_id: ThreadId) -> ThreadStatus:
        """Return the current status of ``thread_id``."""
        info = self._infos.get(thread_id)
        if info is None:
            raise ThreadNotFoundError(thread_id)
        return info.status

    # ── Spawning ────────────────────────────────────────────────────────────

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
        """Create a new thread on a registered worker, optionally pre-seeded."""
        if seed_from is not None and seed_events is not None:
            raise ValueError("spawn accepts at most one of 'seed_from' / 'seed_events'")

        # Pick a worker.
        if worker_id is not None:
            if worker_id not in self._workers:
                raise ValueError(f"worker {worker_id!r} is not registered")
            adapter = self._workers[worker_id]
        else:
            if not self._workers:
                raise ValueError("no workers registered; cannot spawn")
            adapter = next(iter(self._workers.values()))

        tid = thread_id if thread_id is not None else ThreadId(f"thread-{uuid.uuid4().hex[:12]}")
        if thread_name is None:
            # Default the display name from the spawnable so event logs (and
            # the optimization graphs reconstructed from them) carry the
            # function's name without every caller having to pass it.
            name = getattr(target, "name", None)
            thread_name = name if isinstance(name, str) and name else None

        # Seed the log BEFORE registering the thread, so external
        # observers never see a registered-but-empty-log window.
        if seed_from is not None:
            if seed_from not in self._infos:
                raise ThreadNotFoundError(seed_from)
            source_events = list(self._events.get(seed_from, []))
            self._events[tid] = [
                e.model_copy(  # type: ignore[union-attr]
                    update={"thread_id": tid, "id": EventId(f"evt-{uuid.uuid4().hex}")},
                )
                for e in source_events
            ]
        elif seed_events is not None:
            self._events[tid] = [
                e.model_copy(  # type: ignore[union-attr]
                    update={"thread_id": tid, "id": EventId(f"evt-{uuid.uuid4().hex}")},
                )
                for e in seed_events
            ]

        # Register the thread info.
        info = ThreadInfo(
            thread_id=tid,
            worker_id=adapter.worker_id,
            thread_name=thread_name,
            input_shape=target.input_shape,
            status=ThreadStatus.NOT_STARTED,
            parent_id=parent_id,
        )
        await self.register_thread(info)

        # A fresh sub-computation (not a fork, which carries seed_from) records
        # its parent→child edge in the parent's log for build_graph.
        if seed_from is None and parent_id is not None:
            self.append_event(ThreadSpawnedEvent(thread_id=parent_id, child_thread_id=tid))

        # Ask the worker to allocate per-thread state and start the dispatcher.
        await adapter.spawn(
            target,
            thread_id=tid,
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata if metadata is not None else {},
        )

        return ThreadHandle(tid, self)

    # ── Cross-thread operations (routed to the hosting worker's adapter) ──

    def _adapter_for(self, thread_id: ThreadId) -> WorkerAdapter:
        """Return the adapter hosting ``thread_id`` or raise."""
        info = self._infos.get(thread_id)
        if info is None:
            raise ThreadNotFoundError(thread_id)
        adapter = self._workers.get(info.worker_id)
        if adapter is None:
            # The thread is registered but its host is gone. Reporting
            # not-found here would contradict ``list_threads``, which still
            # answers with this thread; name the loss instead, and carry the
            # rest of that worker's blast radius so one caught error tells the
            # caller everything it lost.
            siblings = [tid for tid in self._lost_workers.get(info.worker_id, ()) if tid != thread_id]
            raise WorkerLostError(info.worker_id, [thread_id, *siblings])
        return adapter

    def submit(
        self,
        thread_id: ThreadId,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on ``thread_id``."""
        adapter = self._adapter_for(thread_id)
        return adapter.submit(thread_id, args, kwargs)

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``."""
        adapter = self._adapter_for(thread_id)
        await adapter.notify(thread_id, text)

    async def pause(self, thread_id: ThreadId) -> None:
        """Set the pause signal."""
        adapter = self._adapter_for(thread_id)
        await adapter.pause(thread_id)
        info = self._infos[thread_id]
        # The worker may have been lost while awaiting the pause response.
        if info.status.is_done:
            return
        # Pausing a non-running thread flips it to PAUSED; an in-flight
        # cycle keeps its RUNNING status until it yields at a work
        # boundary, at which point its next lifecycle event governs.
        if info.status != ThreadStatus.RUNNING:
            self._infos[thread_id] = info.model_copy(update={"status": ThreadStatus.PAUSED})

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear the pause signal."""
        adapter = self._adapter_for(thread_id)
        await adapter.resume(thread_id)
        info = self._infos[thread_id]
        if info.status == ThreadStatus.PAUSED:
            self._infos[thread_id] = info.model_copy(update={"status": ThreadStatus.IDLE})

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle."""
        adapter = self._adapter_for(thread_id)
        await adapter.cancel(thread_id)

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination."""
        adapter = self._adapter_for(thread_id)
        await adapter.terminate(thread_id)

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear the thread down immediately."""
        adapter = self._adapter_for(thread_id)
        await adapter.terminate_now(thread_id)

    async def fork(
        self,
        thread_id: ThreadId,
        *,
        parent_id: ThreadId | None = None,
    ) -> ThreadHandle[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Fork ``thread_id`` into a new thread seeded with its history.

        Thin sugar over :meth:`spawn`: ask the worker for a resumption
        spawnable, then spawn it with ``seed_from`` set.

        A fork is a divergent continuation of the source, not a sub-computation
        it delegated — so by default the fork inherits the source's own
        ``parent_id`` (becoming its sibling) rather than becoming its child.
        This keeps token rollup pointing at the true owner and keeps the fork
        out of the source's optimization ``child_threads``. Pass ``parent_id``
        to override — e.g. a thread forking a helper it takes responsibility for
        passes ``parent_id=<its own id>``.
        """
        source = self._infos.get(thread_id)
        if source is None:
            raise ThreadNotFoundError(thread_id)
        adapter = self._adapter_for(thread_id)
        new_spawnable = await adapter.get_fork_spawnable(thread_id)
        return await self.spawn(
            new_spawnable,
            seed_from=thread_id,
            parent_id=source.parent_id if parent_id is None else parent_id,
        )

    # ── Approvals ───────────────────────────────────────────────────────────

    async def resolve_approval(
        self,
        thread_id: ThreadId,
        approval_id: str,
        decision: str,
    ) -> None:
        """Resolve a pending tool-approval request on ``thread_id``.

        Dispatches to the hosting worker's ``resolve_approval`` (which
        resolves the awaiting future inside the worker's
        ``on_interrupt``), then appends an ``ApprovalDecidedEvent`` to
        the log as the canonical record.
        """
        adapter = self._adapter_for(thread_id)
        _ = adapter.resolve_approval(thread_id, approval_id, decision)
        self.append_event(
            ApprovalDecidedEvent(
                thread_id=thread_id,
                approval_id=approval_id,
                decision=decision,
            ),
        )

    # ── Events ──────────────────────────────────────────────────────────────

    def append_event(self, event: Event) -> None:
        """Persist ``event`` and broadcast it to subscribers.

        Durability first: the event is appended to the log before any
        side effects run. Status caching, subscriber fan-out, and
        rate-limit bookkeeping are best-effort — a failure in any one
        of them is logged and swallowed so that a buggy subscriber or
        a stale cache never breaks the producer (which is often the
        dispatcher task; an exception here would kill the thread).

        Args:
            event: The event to append; ``event.thread_id`` must be set.

        Raises:
            ValueError: ``event.thread_id`` is ``None``.
        """
        thread_id_obj: object = getattr(event, "thread_id", None)
        if thread_id_obj is None:
            raise ValueError(
                f"Cannot append event {event.kind!r}: ``thread_id`` is unset. "
                "Events must be stamped with a routing thread id before being "
                "appended (``LocalWorker._route_event`` does this automatically)."
            )
        thread_id: ThreadId = cast(ThreadId, thread_id_obj)
        # Durability: append before any side effects run.
        self._events.setdefault(thread_id, []).append(event)

        # Update the cached status from lifecycle events.
        try:
            self._apply_lifecycle(event, thread_id)
        except Exception as exc:  # noqa: BLE001 — side effects never propagate
            _log_side_effect_failure("status cache", event, thread_id, exc)

        # Broadcast to live subscribers; isolate each callback.
        for subscriber in list(self._subscribers):
            if not subscriber.matches(event):
                continue
            try:
                subscriber.callback(event)
            except Exception as exc:  # noqa: BLE001 — one bad subscriber must not affect others
                _log_side_effect_failure("subscriber", event, thread_id, exc)

        # I3: TOKEN_USAGE drives rate-limit pause state.
        if event.kind == EventKind.TOKEN_USAGE:
            try:
                self._on_token_usage(thread_id)
            except Exception as exc:  # noqa: BLE001
                _log_side_effect_failure("token-usage hook", event, thread_id, exc)

    def _apply_lifecycle(self, event: Event, thread_id: ThreadId) -> None:
        """Update cached status from lifecycle events."""
        info = self._infos.get(thread_id)
        if info is None:
            return
        if info.status.is_done:
            # A terminal status is the end of the thread's life. A late
            # lifecycle event from a cycle that was already in flight must
            # not resurrect it as ``idle`` — status only ever moves toward
            # terminal.
            return
        new_status: ThreadStatus | None = None
        if event.kind == EventKind.STARTED:
            new_status = ThreadStatus.RUNNING
        elif event.kind == EventKind.COMPLETED:
            # Respect pause state set while running.
            unpaused = self._pause_events.get(thread_id)
            paused = unpaused is not None and not unpaused.is_set()
            new_status = ThreadStatus.PAUSED if paused else ThreadStatus.IDLE
        elif event.kind == EventKind.CANCELLED:
            new_status = ThreadStatus.CANCELLED
        elif event.kind == EventKind.FAILED:
            # A cycle that raised leaves the thread alive and dequeuing, so a
            # dispatcher-emitted FAILED means idle. The coordinator writes
            # ``ThreadStatus.FAILED`` itself when the thread is really dead
            # (``deregister_worker``), which the terminal guard above protects.
            new_status = ThreadStatus.IDLE
        if new_status is not None and new_status != info.status:
            self._infos[thread_id] = info.model_copy(update={"status": new_status})

    def _on_token_usage(self, thread_id: ThreadId) -> None:
        """React to a ``TOKEN_USAGE`` event on ``thread_id``.

        Default no-op; subclasses override to drive pause state from
        cumulative usage.
        """
        del thread_id

    async def get_events(
        self,
        thread_id: ThreadId,
        since_id: EventId | None = None,
        kinds: list[EventKind] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Replay stored events for ``thread_id`` in chronological order."""
        stored = self._events.get(thread_id, [])
        kind_set: frozenset[str] | None = frozenset(str(k) for k in kinds) if kinds is not None else None

        results: list[Event] = []
        seen_cursor = since_id is None
        for event in stored:
            if not seen_cursor:
                if getattr(event, "id", None) == since_id:
                    seen_cursor = True
                continue
            if kind_set is not None and str(event.kind) not in kind_set:
                continue
            results.append(event)
            if limit is not None and len(results) >= limit:
                break
        return results

    async def new_message_id(self) -> MessageId:
        """Mint a fresh message id."""
        return MessageId(f"msg-{uuid.uuid4().hex}")

    async def copy_events(
        self,
        source_id: ThreadId,
        target_id: ThreadId,
        until_event_id: EventId | None = None,
    ) -> None:
        """Copy every event from ``source_id`` onto ``target_id`` in order."""
        if source_id not in self._infos:
            raise ThreadNotFoundError(source_id)
        if target_id not in self._infos:
            raise ThreadNotFoundError(target_id)

        source_events = self._events.get(source_id, [])

        if until_event_id is not None:
            cutoff: int | None = None
            for i, e in enumerate(source_events):
                if getattr(e, "id", None) == until_event_id:
                    cutoff = i
                    break
            if cutoff is None:
                raise ValueError(f"until_event_id={until_event_id!r} is not an event on source thread {source_id!r}")
            source_events = source_events[: cutoff + 1]

        target_bucket = self._events.setdefault(target_id, [])
        for event in source_events:
            copied = event.model_copy(  # type: ignore[union-attr]
                update={"thread_id": target_id, "id": EventId(f"evt-{uuid.uuid4().hex}")},
            )
            target_bucket.append(copied)

    # ── Subscriptions ───────────────────────────────────────────────────────

    def on(
        self,
        callback: OnEventCallback,
        *,
        thread_id: ThreadId | None = None,
        kinds: list[EventKind] | None = None,
    ) -> Subscription:
        """Register a live subscriber for future events."""
        kind_set: frozenset[str] | None = frozenset(str(k) for k in kinds) if kinds is not None else None
        subscriber = _Subscriber(callback=callback, thread_id=thread_id, kinds=kind_set)
        self._subscribers.append(subscriber)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

        return _Subscription(_unsubscribe)

    # ── Rate limiting ───────────────────────────────────────────────────────

    async def is_paused(self, thread_id: ThreadId) -> bool:
        """Whether the rate-limit / manual pause signal is set for ``thread_id``."""
        unpaused = self._pause_events.get(thread_id)
        if unpaused is None:
            return False
        return not unpaused.is_set()

    def wait_until_unpaused(self, thread_id: ThreadId) -> Awaitable[None]:
        """Await until the pause signal for ``thread_id`` is clear."""
        unpaused = self._pause_events.setdefault(thread_id, asyncio.Event())
        # Default: unregistered threads are not paused.
        if thread_id not in self._infos and not unpaused.is_set():
            unpaused.set()
        return _wait_event(unpaused)
