"""Test harness bundling a ``LocalWorker``, coordinator, and barrier registry.

``RuntimeHarness`` is an async context manager that wires up a worker,
gives tests ergonomic access to event logs, and provides synchronization
primitives for concurrency tests:

- ``wait_for`` blocks until a matching event is appended (scanning the
  existing log first, then subscribing via ``Coordinator.on``).
- ``release`` unblocks a barrier that a ``ScriptedModel`` is suspended on.

Barriers are wired via a ``ContextVar`` set during ``__aenter__``: any
``ScriptedModel`` used inside the ``async with`` block resolves its
barrier names against the harness automatically. Parallel harnesses have
independent registries because each ``async with`` runs in its own task
context.
"""

from types import TracebackType
from typing import Self, final

from strands.types.content import Messages

from ..runtime.worker import LocalWorker
from ..protocols import Coordinator, Spawnable
from ..types import Event, EventKind, ThreadId
from ..handle import ThreadHandle


@final
class RuntimeHarness:
    """Test wiring for a ``LocalWorker`` with observability and barrier release.

    Usage::

        async with RuntimeHarness() as h:
            handle = h.spawn(my_fn.replace(model=ScriptedModel([...])))
            result = await handle.run("hi")
            h.events(handle.thread_id)

    Args:
        coordinator: A pre-built coordinator; a fresh ``InMemoryCoordinator``
            is constructed on ``__aenter__`` when ``None``.
        worker: A pre-built ``LocalWorker`` bound to the coordinator; a fresh
            one is constructed on ``__aenter__`` when ``None``.

    Ensures:
        Barrier registry is empty until ``__aenter__`` runs.
    """

    def __init__(
        self,
        *,
        coordinator: Coordinator | None = None,
        worker: LocalWorker | None = None,
    ) -> None: ...

    async def __aenter__(self) -> Self:
        """Install the barrier contextvar and returns self.

        Ensures:
            - ``self.worker`` is a live ``LocalWorker``.
            - Any ``ScriptedModel`` constructed and used inside the ``async with`` block
              resolves barriers against this harness.
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Tear down the worker and release all barriers.

        Args:
            exc_type: Exception class raised inside the ``with`` block, if any.
            exc: Exception instance raised inside the ``with`` block, if any.
            tb: Traceback for the raised exception, if any.

        Ensures:
            - Every registered dispatcher is cancelled and awaited.
            - Every pending barrier is released so scripted models can unwind.
            - The contextvar token set in ``__aenter__`` is reset.
        """
        ...

    # ── Spawning ──

    async def spawn[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId | None = None,
        thread_name: str | None = None,
        parent_id: ThreadId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ThreadHandle[P, T]:
        """Spawn a thread on the underlying worker and remember it.

        Delegates to ``self.worker.spawn_locally`` and additionally
        hooks an ``AfterInvocationEvent`` listener that captures the
        final ``agent.messages`` for :meth:`agent_messages`.

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            thread_id: Explicit id; one is minted if omitted.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
            metadata: Application metadata attached to the thread.

        Returns:
            A ``ThreadHandle`` in ``NOT_STARTED`` state.
        """
        ...

    # ── Observation ──

    async def events(
        self,
        thread_id: ThreadId,
        *,
        kinds: list[EventKind] | None = None,
    ) -> list[Event]:
        """Snapshot of events emitted for ``thread_id``.

        Args:
            thread_id: Thread whose events are requested.
            kinds: Restrict to these event kinds; all kinds if ``None``.

        Returns:
            Events in append order.
        """
        ...

    def agent_messages(self, thread_id: ThreadId) -> Messages:
        """Return the Strands ``Messages`` captured after the most recent cycle.

        Args:
            thread_id: Thread whose captured messages are requested.

        Returns:
            The ``agent.messages`` list copied at cycle completion; an empty list if no
            cycle has completed yet on this thread.

        Requires:
            ``thread_id`` was spawned through ``self.spawn`` (the harness
            attaches the capture hook on spawn).
        """
        ...

    # ── Synchronization ──

    async def wait_for(
        self,
        thread_id: ThreadId,
        kind: EventKind,
        *,
        timeout: float = 2.0,
    ) -> Event:
        """Wait until an event of ``kind`` is appended for ``thread_id``.

        Scans the existing event log first, so a ``wait_for`` issued after the event has
        already arrived resolves immediately.

        Args:
            thread_id: Thread whose log is being watched.
            kind: Event kind to match.
            timeout: Seconds before raising ``asyncio.TimeoutError``.

        Returns:
            The matching event.

        Raises:
            asyncio.TimeoutError: No matching event arrived within ``timeout``.
        """
        ...

    def release(self, barrier: str) -> None:
        """Unblock a ``ScriptedModel`` barrier by name.

        Args:
            barrier: Name of the barrier (from ``Turn.await_before``,
                ``Turn.await_after``, or an ``AwaitBarrier`` sentinel).

        Ensures:
            Any current or future await on this barrier resolves.

        Concurrency:
            Idempotent — releasing an already-released barrier is a no-op.
        """
        ...

    # ── Properties ──

    @property
    def worker(self) -> LocalWorker:
        """The underlying worker.

        Raises if accessed before ``__aenter__``.
        """
        ...

    @property
    def coordinator(self) -> Coordinator:
        """The underlying coordinator backing the worker."""
        ...
