"""CoordinatorClient — ``Coordinator`` implementation over a WireChannel.

Connects to a :class:`CoordinatorEndpoint` over a single WebSocket and
proxies every :class:`~ai_functions.protocols.Coordinator` method as an outbound
RPC call. Also accepts *inbound* RPC calls from the endpoint when the
client hosts a worker (the endpoint must reach the worker's adapter
methods over the same connection).

Usage::

    client = await CoordinatorClient.connect("ws://localhost:9900/rpc")
    worker = LocalWorker(client)
    await worker.register()
    handle = await worker.spawn_locally(my_fn)

The client registers its hosted workers with the endpoint on first use;
thereafter, inbound ``worker.*`` calls from the endpoint are dispatched
to the matching ``LocalWorker`` instance via the shared ``WireChannel``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from types import TracebackType
from typing import Any, Self, final

from ..handle import ThreadHandle
from ..protocols import Coordinator, OnEventCallback, Spawnable, Subscription
from ..types import Event, EventId, EventKind, MessageId, ThreadId, ThreadInfo, ThreadStatus, WorkerId
from .wire import Transport


@final
class CoordinatorClient(Coordinator):
    """Remote :class:`~ai_functions.protocols.Coordinator` backed by a WebSocket.

    Every ``Coordinator`` method issues a ``CallFrame`` over the
    underlying :class:`WireChannel` and awaits a matching ``ResultFrame``
    or ``ErrorFrame``. Events broadcast by the endpoint arrive as
    ``EventFrame`` s and are delivered to subscribers registered via
    :meth:`on`.

    Workers hosted by this client (via ``LocalWorker(client)``) also
    install their ``WorkerAdapter`` methods as inbound handlers on the
    same channel, so the endpoint can route ``worker.spawn`` /
    ``worker.submit`` / etc. back to them.

    Args:
        transport: An open duplex string channel. Prefer
            :meth:`connect` for the common case; direct construction
            is for tests that provide a mock transport.

    Invariants:
        Implements :class:`Coordinator`.
    """

    @classmethod
    async def connect(cls, url: str) -> Self:
        """Open a WebSocket to ``url`` and return a connected client.

        Args:
            url: ``ws://host:port/rpc`` URL of the coordinator endpoint.

        Returns:
            A connected ``CoordinatorClient`` ready for use.

        Raises:
            OSError: The connection could not be established.
        """
        ...

    def __init__(self, transport: Transport) -> None: ...

    async def close(self) -> None:
        """Close the underlying channel; idempotent."""
        ...

    async def __aenter__(self) -> Self:
        """Return ``self`` for use in a ``with`` block."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close on context exit.

        Args:
            exc_type: Exception class raised inside the ``with`` block, if any.
            exc: Exception instance raised inside the ``with`` block, if any.
            tb: Traceback for the raised exception, if any.
        """
        ...

    # ── Worker pool ──

    async def register_worker(self, adapter: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Register a worker with the remote coordinator.

        Two-step: the adapter's methods are installed as inbound
        handlers on the shared channel (so the endpoint can route
        ``worker.*`` calls to the adapter), and a
        ``coordinator.register_worker`` RPC is issued to tell the
        endpoint this worker exists.

        Args:
            adapter: A :class:`~ai_functions.runtime.worker.WorkerAdapter`
                hosted by this client process.

        Raises:
            ValueError: Already registered under the same ``worker_id``.
        """
        ...

    async def deregister_worker(self, worker_id: WorkerId) -> None:
        """Remove a worker from the remote pool; idempotent.

        Args:
            worker_id: Worker to remove.
        """
        ...

    # ── Thread registry ──

    async def register_thread(self, info: ThreadInfo) -> None:
        """Register a thread with the remote coordinator.

        Args:
            info: Full snapshot of the thread's identity and host.

        Raises:
            ValueError: ``info.worker_id`` is not a registered worker.
        """
        ...

    async def deregister_thread(self, thread_id: ThreadId) -> None:
        """Remove ``thread_id`` from the remote registry; idempotent.

        Args:
            thread_id: Thread to remove.
        """
        ...

    # ── Discovery ──

    async def list_threads(self) -> list[ThreadInfo]:
        """Return a snapshot of every registered thread.

        Returns:
            One :class:`ThreadInfo` per registered thread.
        """
        ...

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        """Return the full info snapshot for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            The :class:`ThreadInfo`.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        ...

    def get_handle(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a handle bound to this client for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            A type-erased :class:`ThreadHandle`.
        """
        ...

    async def get_thread_status(self, thread_id: ThreadId) -> ThreadStatus:
        """Return the current status of ``thread_id``.

        Args:
            thread_id: Thread to query.

        Returns:
            The cached :class:`ThreadStatus`.

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
        """Spawn a thread on the remote coordinator.

        ``target`` is cloudpickled and shipped as a ``Binary`` param;
        the endpoint unpickles and dispatches to the hosting worker.

        Args:
            target: Spawnable whose ``to_thread`` produces the live
                instance on the hosting worker.
            seed_from: Source thread whose event log is copied.
            seed_events: Explicit event list to copy.
            worker_id: Explicit worker selection; endpoint default if ``None``.
            thread_id: Explicit id; one is minted if omitted.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread.
            metadata: Application metadata attached to the thread.

        Returns:
            A :class:`ThreadHandle` backed by this client.

        Raises:
            ValueError: Seeding / worker selection errors.
            RemoteError: The endpoint rejected the spawn.
        """
        ...

    # ── Cross-thread operations ──

    def submit(
        self,
        thread_id: ThreadId,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Awaitable[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on the remote thread's work queue.

        Args and kwargs are cloudpickled together as a single ``Binary``
        param. The returned awaitable resolves when the endpoint relays
        the cycle result back.

        Args:
            thread_id: Thread to run.
            args: Positional arguments forwarded to ``Thread.execute``.
            kwargs: Keyword arguments forwarded to ``Thread.execute``.

        Returns:
            An awaitable that resolves with the cycle's typed result.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        ...

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``.

        Args:
            thread_id: Thread receiving the message.
            text: Message body.

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

    async def fork(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Fork ``thread_id`` into a new thread seeded with its history.

        Args:
            thread_id: Thread to fork.

        Returns:
            A handle to the new forked thread.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
            NotImplementedError: The source thread does not support
                forking.
        """
        ...

    # ── Approvals ──

    async def resolve_approval(
        self,
        thread_id: ThreadId,
        approval_id: str,
        decision: str,
    ) -> None:
        """Resolve a pending tool-approval request.

        Args:
            thread_id: Thread whose approval is being resolved.
            approval_id: Id of the ``APPROVAL_REQUEST`` this resolves.
            decision: Policy decision to return to the executor.
        """
        ...

    # ── Events ──

    def append_event(self, event: Event) -> None:
        """Schedule ``event`` to be appended on the endpoint's coordinator.

        Synchronous by design (I2/I3): it schedules a fire-and-forget
        ``coordinator.append_event`` RPC and returns immediately without
        blocking on network I/O. The remote coordinator performs the durable
        append and subscriber broadcast; subscribers on this client observe it
        later via :meth:`on`. Per I11, a failure of the scheduled RPC is logged
        and swallowed rather than propagated to the (often sync) caller.

        Args:
            event: The event to append; ``event.thread_id`` must be set.
        """
        ...

    async def get_events(
        self,
        thread_id: ThreadId,
        since_id: EventId | None = None,
        kinds: list[EventKind] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Replay stored events for ``thread_id`` from the remote log.

        Args:
            thread_id: Thread whose events are being read.
            since_id: Return only events appended strictly after this id.
            kinds: Restrict to these event kinds; all kinds if ``None``.
            limit: Return at most this many events; unbounded if ``None``.

        Returns:
            Events for ``thread_id`` matching the filters, oldest first.
        """
        ...

    async def new_message_id(self) -> MessageId:
        """Mint a fresh message id on the endpoint.

        Returns:
            A freshly minted message id.
        """
        ...

    async def copy_events(
        self,
        source_id: ThreadId,
        target_id: ThreadId,
        until_event_id: EventId | None = None,
    ) -> None:
        """Copy events from one thread to another on the endpoint.

        Args:
            source_id: Thread whose events are read.
            target_id: Thread whose log receives rewritten copies.
            until_event_id: Copy events up to and including this id;
                copy the entire current history when ``None``.

        Raises:
            ThreadNotFoundError: Either thread is not registered.
            ValueError: ``until_event_id`` does not match any source event.
        """
        ...

    # ── Subscriptions ──

    def on(
        self,
        callback: OnEventCallback,
        *,
        thread_id: ThreadId | None = None,
        kinds: list[EventKind] | None = None,
    ) -> Subscription:
        """Subscribe to events broadcast by the endpoint.

        The client does not maintain a durable event log — subscriptions
        receive events broadcast over the wire after registration.

        Args:
            callback: Sink invoked for each matching event.
            thread_id: Restrict to this thread if set.
            kinds: Restrict to these event kinds if set.

        Returns:
            A :class:`Subscription` whose ``unsubscribe`` tears down
            this registration.
        """
        ...

    # ── Rate limiting ──

    async def is_paused(self, thread_id: ThreadId) -> bool:
        """Return whether ``thread_id`` is paused on the endpoint.

        Args:
            thread_id: Thread to query.

        Returns:
            True iff the thread is paused at this instant.
        """
        ...

    def wait_until_unpaused(self, thread_id: ThreadId) -> Awaitable[None]:
        """Await until ``thread_id`` is unpaused on the endpoint.

        Args:
            thread_id: Thread whose pause signal to watch.

        Returns:
            An awaitable that resolves when the thread is unpaused.
        """
        ...


class WorkerHandlers:
    """Bundled inbound ``worker.<wid>.*`` RPC handlers for one worker adapter.

    Installed on the shared channel by
    :meth:`CoordinatorClient.register_worker`, which uses
    :func:`~ai_functions.network.channel.bind_handlers` with a ``rename``
    callback so the decorator's ``worker.<op>`` tag expands to
    ``worker.<wid>.<op>`` — multiple workers hosted behind the same
    client share one channel without colliding.

    Args:
        adapter: The :class:`~ai_functions.runtime.worker.WorkerAdapter` whose
            methods every decorated handler forwards to.
    """

    def __init__(self, adapter: Any) -> None: ...  # pyright: ignore[reportExplicitAny]
