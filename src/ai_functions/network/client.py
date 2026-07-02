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

import asyncio
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast, final

from ..handle import ThreadHandle
from ..protocols import Coordinator, OnEventCallback, Spawnable, Subscription
from ..runtime.errors import ThreadNotFoundError
from ..types import (
    Event,
    EventId,
    EventKind,
    MessageId,
    ThreadId,
    ThreadInfo,
    ThreadStatus,
    WorkerId,
)
from ._transport_config import (
    MAX_MESSAGE_BYTES,
    PING_INTERVAL_SECONDS,
    PING_TIMEOUT_SECONDS,
)
from .channel import (
    EVENT_ADAPTER,
    WebsocketTransport,
    WireChannel,
    bind_handlers,
    cloudpickle_b64,
    rpc_method,
    unpickle_b64,
)
from .wire import Transport
from .wire_methods import (
    AppendEventParams,
    CopyEventsParams,
    DeregisterThreadParams,
    DeregisterWorkerParams,
    GetEventsParams,
    GetThreadInfoParams,
    GetThreadStatusParams,
    InjectMessageParams,
    ListThreadsParams,
    NewMessageIdParams,
    RegisterThreadParams,
    RegisterWorkerParams,
    ResolveApprovalParams,
    SpawnParams,
    SubmitParams,
    ThreadIdOnlyParams,
    WorkerInjectMessageParams,
    WorkerResolveApprovalParams,
    WorkerSpawnParams,
    WorkerSubmitParams,
    WorkerThreadIdParams,
)

if TYPE_CHECKING:
    from ..runtime.worker import WorkerAdapter


@final
class _ClientSubscription(Subscription):
    """Subscription returned by ``CoordinatorClient.on``."""

    __slots__ = ("_unsubscribe", "_active")

    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe: Callable[[], None] = unsubscribe
        self._active: bool = True

    def unsubscribe(self) -> None:
        if self._active:
            self._active = False
            self._unsubscribe()

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc
        self.unsubscribe()


class _Subscriber:
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
        if self.thread_id is not None:
            event_tid: object = getattr(event, "thread_id", None)
            if event_tid != self.thread_id:
                return False
        if self.kinds is not None and str(event.kind) not in self.kinds:  # type: ignore[union-attr]
            return False
        return True


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

    __slots__ = (
        "_channel",
        "_subscribers",
        "_workers",
    )

    _channel: WireChannel
    _subscribers: list[_Subscriber]
    _workers: dict[WorkerId, WorkerAdapter]

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
        import websockets

        ws = await websockets.connect(
            url,
            max_size=MAX_MESSAGE_BYTES,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
        )
        transport = WebsocketTransport(ws)
        self = cls(transport)
        _ = await self._channel.__aenter__()
        return self

    def __init__(self, transport: Transport) -> None:
        self._channel = WireChannel(transport)
        self._subscribers = []
        self._workers = {}
        _ = self._channel.on_event(self._dispatch_event)

    def _dispatch_event(self, event: Event) -> None:
        for sub in list(self._subscribers):
            if sub.matches(event):
                try:
                    sub.callback(event)
                except Exception:  # noqa: BLE001
                    pass

    async def __aenter__(self) -> Self:
        """Return ``self`` for use in a ``with`` block."""
        return self

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
        del exc_type, exc, tb
        await self.close()

    async def close(self) -> None:
        """Close the underlying channel; idempotent."""
        await self._channel.close()

    async def _call(self, method: str, params: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Send a typed params pydantic model over the channel."""
        return await self._channel.call(method, **params.model_dump(mode="json"))

    # ── Worker pool ─────────────────────────────────────────────────────────

    async def register_worker(self, adapter: WorkerAdapter) -> None:
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
        wid = adapter.worker_id
        if wid in self._workers:
            raise ValueError(f"worker {wid!r} already registered on this client")
        self._workers[wid] = adapter
        # Install inbound handlers so the endpoint can reach this adapter.
        # The WorkerHandlers methods are tagged with "worker.<method>";
        # we rename them to "worker.<wid>.<method>" at bind time so
        # multiple workers on the same client don't collide.
        wid_str = str(wid)
        bind_handlers(
            self._channel,
            WorkerHandlers(adapter),
            rename=lambda tag: tag.replace("worker.", f"worker.{wid_str}.", 1),
        )
        _ = await self._call(
            "coordinator.register_worker",
            RegisterWorkerParams(worker_id=wid),
        )

    async def deregister_worker(self, worker_id: WorkerId) -> None:
        """Remove a worker from the remote pool; idempotent.

        Args:
            worker_id: Worker to remove.
        """
        self._workers.pop(worker_id, None)
        _ = await self._call(
            "coordinator.deregister_worker",
            DeregisterWorkerParams(worker_id=worker_id),
        )

    # ── Thread registry ─────────────────────────────────────────────────────

    async def register_thread(self, info: ThreadInfo) -> None:
        """Register a thread with the remote coordinator.

        Args:
            info: Full snapshot of the thread's identity and host.

        Raises:
            ValueError: ``info.worker_id`` is not a registered worker.
        """
        _ = await self._call(
            "coordinator.register_thread",
            RegisterThreadParams(info=info),
        )

    async def deregister_thread(self, thread_id: ThreadId) -> None:
        """Remove ``thread_id`` from the remote registry; idempotent.

        Args:
            thread_id: Thread to remove.
        """
        _ = await self._call(
            "coordinator.deregister_thread",
            DeregisterThreadParams(thread_id=thread_id),
        )

    # ── Discovery ───────────────────────────────────────────────────────────

    async def list_threads(self) -> list[ThreadInfo]:
        """Return a snapshot of every registered thread.

        Returns:
            One :class:`ThreadInfo` per registered thread.
        """
        raw = await self._call("coordinator.list_threads", ListThreadsParams())
        return [ThreadInfo.model_validate(item) for item in cast("list[object]", raw)]

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        """Return the full info snapshot for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            The :class:`ThreadInfo`.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        raw = await self._call(
            "coordinator.get_thread_info",
            GetThreadInfoParams(thread_id=thread_id),
        )
        return ThreadInfo.model_validate(raw)

    def get_handle(self, thread_id: ThreadId) -> ThreadHandle[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a handle bound to this client for ``thread_id``.

        Args:
            thread_id: Thread to look up.

        Returns:
            A type-erased :class:`ThreadHandle`.
        """
        return ThreadHandle(thread_id, self)

    async def get_thread_status(self, thread_id: ThreadId) -> ThreadStatus:
        """Return the current status of ``thread_id``.

        Args:
            thread_id: Thread to query.

        Returns:
            The cached :class:`ThreadStatus`.

        Raises:
            ThreadNotFoundError: No thread is registered under this id.
        """
        raw = await self._call(
            "coordinator.get_thread_status",
            GetThreadStatusParams(thread_id=thread_id),
        )
        return ThreadStatus(raw)

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
        params = SpawnParams(
            target_pickle=cloudpickle_b64(target).encode("ascii"),
            seed_from=seed_from,
            seed_events=seed_events,
            worker_id=worker_id,
            thread_id=thread_id,
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata,
        )
        raw = await self._call("coordinator.spawn", params)
        return ThreadHandle(ThreadId(cast("str", raw)), self)

    # ── Cross-thread operations ─────────────────────────────────────────────

    def submit(
        self,
        thread_id: ThreadId,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
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
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()  # pyright: ignore[reportExplicitAny]

        async def _issue() -> None:
            params = SubmitParams(
                thread_id=thread_id,
                args_kwargs_pickle=cloudpickle_b64((args, kwargs)).encode("ascii"),
            )
            try:
                raw = await self._call("coordinator.submit", params)
            except Exception as exc:  # noqa: BLE001
                if not future.done():
                    future.set_exception(exc)
                return
            value = unpickle_b64(cast("str", raw))
            if not future.done():
                future.set_result(value)

        _ = asyncio.create_task(_issue())
        return future

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``.

        Args:
            thread_id: Thread receiving the message.
            text: Message body.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.notify",
            InjectMessageParams(thread_id=thread_id, text=text),
        )

    async def pause(self, thread_id: ThreadId) -> None:
        """Set the pause signal on ``thread_id``.

        Args:
            thread_id: Thread to pause.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.pause",
            ThreadIdOnlyParams(thread_id=thread_id),
        )

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear the pause signal on ``thread_id``.

        Args:
            thread_id: Thread to resume.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.resume",
            ThreadIdOnlyParams(thread_id=thread_id),
        )

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle on ``thread_id``.

        Args:
            thread_id: Thread to cancel.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.cancel",
            ThreadIdOnlyParams(thread_id=thread_id),
        )

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination of ``thread_id``.

        Args:
            thread_id: Thread to terminate.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.terminate",
            ThreadIdOnlyParams(thread_id=thread_id),
        )

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear ``thread_id`` down immediately.

        Args:
            thread_id: Thread to tear down.

        Raises:
            ThreadNotFoundError: ``thread_id`` is not registered.
        """
        _ = await self._call(
            "coordinator.terminate_now",
            ThreadIdOnlyParams(thread_id=thread_id),
        )

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
        raw = await self._call(
            "coordinator.fork",
            ThreadIdOnlyParams(thread_id=thread_id),
        )
        return ThreadHandle(ThreadId(cast("str", raw)), self)

    # ── Approvals ───────────────────────────────────────────────────────────

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
        _ = await self._call(
            "coordinator.resolve_approval",
            ResolveApprovalParams(
                thread_id=thread_id,
                approval_id=approval_id,
                decision=decision,
            ),
        )

    # ── Events ──────────────────────────────────────────────────────────────

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
        params = AppendEventParams(event=event)

        async def _send() -> None:
            try:
                _ = await self._call("coordinator.append_event", params)
            except Exception as exc:  # noqa: BLE001 -- log + swallow (I11)
                import logging

                logging.getLogger("ai_functions.network.client").warning(
                    "append_event RPC failed: %r",
                    exc,
                )

        _ = asyncio.create_task(_send())

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
        raw = await self._call(
            "coordinator.get_events",
            GetEventsParams(
                thread_id=thread_id,
                since_id=since_id,
                kinds=kinds,
                limit=limit,
            ),
        )
        return [EVENT_ADAPTER.validate_python(item) for item in cast("list[object]", raw)]

    async def new_message_id(self) -> MessageId:
        """Mint a fresh message id on the endpoint.

        Returns:
            A freshly minted message id.
        """
        raw = await self._call("coordinator.new_message_id", NewMessageIdParams())
        return MessageId(cast("str", raw))

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
        _ = await self._call(
            "coordinator.copy_events",
            CopyEventsParams(
                source_id=source_id,
                target_id=target_id,
                until_event_id=until_event_id,
            ),
        )

    # ── Subscriptions ───────────────────────────────────────────────────────

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
        kind_set: frozenset[str] | None = frozenset(str(k) for k in kinds) if kinds is not None else None
        sub = _Subscriber(callback=callback, thread_id=thread_id, kinds=kind_set)
        self._subscribers.append(sub)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

        return _ClientSubscription(_unsubscribe)

    # ── Rate limiting ───────────────────────────────────────────────────────

    async def is_paused(self, thread_id: ThreadId) -> bool:
        """Return whether ``thread_id`` is paused on the endpoint.

        Args:
            thread_id: Thread to query.

        Returns:
            True iff the thread is paused at this instant.
        """
        raw = await self._call(
            "coordinator.is_paused",
            ThreadIdOnlyParams(thread_id=thread_id),
        )
        return bool(raw)

    def wait_until_unpaused(self, thread_id: ThreadId) -> Awaitable[None]:
        """Await until ``thread_id`` is unpaused on the endpoint.

        Args:
            thread_id: Thread whose pause signal to watch.

        Returns:
            An awaitable that resolves when the thread is unpaused.
        """

        async def _wait() -> None:
            _ = await self._call(
                "coordinator.wait_until_unpaused",
                ThreadIdOnlyParams(thread_id=thread_id),
            )

        return _wait()


# ── WorkerHandlers — dispatch table for inbound worker.<wid>.* calls ─────────


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

    __slots__ = ("_adapter",)

    _adapter: WorkerAdapter

    def __init__(self, adapter: WorkerAdapter) -> None:
        self._adapter = adapter

    @rpc_method("worker.spawn")
    async def spawn(self, p: WorkerSpawnParams) -> None:
        target = cast(
            "Spawnable[..., Any]",  # pyright: ignore[reportExplicitAny]
            unpickle_b64(p.target_pickle.decode("ascii")),
        )
        await self._adapter.spawn(
            target,
            thread_id=p.thread_id,
            thread_name=p.thread_name,
            parent_id=p.parent_id,
            metadata=p.metadata,
        )

    @rpc_method("worker.submit")
    async def submit(self, p: WorkerSubmitParams) -> str:
        args, kwargs = cast(
            "tuple[tuple[object, ...], dict[str, object]]",
            unpickle_b64(p.args_kwargs_pickle.decode("ascii")),
        )
        fut = self._adapter.submit(p.thread_id, args, kwargs)
        result = await fut
        return cloudpickle_b64(result)

    @rpc_method("worker.notify")
    async def notify(self, p: WorkerInjectMessageParams) -> None:
        await self._adapter.notify(p.thread_id, p.text)

    @rpc_method("worker.cancel")
    async def cancel(self, p: WorkerThreadIdParams) -> None:
        await self._adapter.cancel(p.thread_id)

    @rpc_method("worker.pause")
    async def pause(self, p: WorkerThreadIdParams) -> None:
        await self._adapter.pause(p.thread_id)

    @rpc_method("worker.resume")
    async def resume(self, p: WorkerThreadIdParams) -> None:
        await self._adapter.resume(p.thread_id)

    @rpc_method("worker.terminate")
    async def terminate(self, p: WorkerThreadIdParams) -> None:
        await self._adapter.terminate(p.thread_id)

    @rpc_method("worker.terminate_now")
    async def terminate_now(self, p: WorkerThreadIdParams) -> None:
        await self._adapter.terminate_now(p.thread_id)

    @rpc_method("worker.get_fork_spawnable")
    async def get_fork_spawnable(self, p: WorkerThreadIdParams) -> str:
        spawnable = await self._adapter.get_fork_spawnable(p.thread_id)
        return cloudpickle_b64(spawnable)

    @rpc_method("worker.resolve_approval")
    async def resolve_approval(self, p: WorkerResolveApprovalParams) -> bool:
        return self._adapter.resolve_approval(p.thread_id, p.approval_id, p.decision)


# Keep ThreadNotFoundError importable from the well-known error list.
_ = ThreadNotFoundError
