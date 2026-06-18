"""CoordinatorEndpoint — WebSocket server fronting an in-memory coordinator.

A :class:`CoordinatorEndpoint` accepts WebSocket connections from
:class:`CoordinatorClient` instances. Each connection gets its own
:class:`WireChannel`; the endpoint registers ``coordinator.*`` handlers
on every channel so clients can call every
:class:`~ai_functions.protocols.Coordinator` method over the wire.

When a client registers a worker (via ``coordinator.register_worker``),
the endpoint builds a per-channel ``_RemoteWorkerAdapter`` shim that
translates ``WorkerAdapter`` calls (from the inner in-memory coordinator)
into outbound ``worker.*`` RPC calls on the originating channel. The
shim is registered with the inner coordinator's worker pool, so routing
from other threads / other workers transparently reaches the remote
process.

Events appended to the inner coordinator are fanned out to every
connected channel as :class:`EventFrame` broadcasts.

Transport note:
    The default server uses ``websockets.serve``. A FastAPI / Starlette
    app with a WebSocket route can be added later by adapting the
    channel's :class:`Transport` to Starlette's WebSocket API; all the
    RPC logic is transport-agnostic.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast, final

from ..protocols import Coordinator, Spawnable
from ..runtime.coordinator import InMemoryCoordinator
from ..types import Event, ThreadId, WorkerId
from ._transport_config import (
    MAX_MESSAGE_BYTES,
    PING_INTERVAL_SECONDS,
    PING_TIMEOUT_SECONDS,
)
from .channel import (
    WebsocketTransport,
    WireChannel,
    bind_handlers,
    cloudpickle_b64,
    rpc_method,
    unpickle_b64,
)
from .wire import DEFAULT_HOST, DEFAULT_PORT
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
    from ..types import ThreadInfo, ThreadStatus


@final
class CoordinatorEndpoint:
    """WebSocket server fronting an :class:`InMemoryCoordinator`.

    Usage::

        endpoint = CoordinatorEndpoint()
        await endpoint.serve(host="127.0.0.1", port=9900)

        # or as a context manager:
        async with CoordinatorEndpoint() as endpoint:
            await endpoint.start(host="127.0.0.1", port=9900)
            # endpoint.url → "ws://127.0.0.1:9900/rpc"
            ...
            await endpoint.stop()

    Args:
        coordinator: Pre-built inner coordinator; a fresh
            :class:`InMemoryCoordinator` is constructed when ``None``.

    Ensures:
        - On :meth:`serve` / :meth:`start`, the endpoint accepts
          WebSocket connections and multiplexes RPC calls + events over
          each one.
        - On :meth:`stop`, every connected channel is closed, worker
          shims are deregistered from the inner coordinator, and the
          server socket is released.
    """

    __slots__ = (
        "_coordinator",
        "_server",
        "_channels",
        "_host",
        "_port",
        "_url",
    )

    _coordinator: Coordinator
    _server: Any  # websockets.server.Server  # pyright: ignore[reportExplicitAny]
    _channels: set[WireChannel]
    _host: str | None
    _port: int | None
    _url: str | None

    def __init__(self, coordinator: Coordinator | None = None) -> None:
        self._coordinator = coordinator if coordinator is not None else InMemoryCoordinator()
        self._server = None
        self._channels = set()
        self._host = None
        self._port = None
        self._url = None

    @property
    def coordinator(self) -> Coordinator:
        """The inner coordinator this endpoint fronts."""
        return self._coordinator

    @property
    def url(self) -> str:
        """The ``ws://host:port/rpc`` URL of the running server.

        Raises:
            RuntimeError: The server is not running.
        """
        if self._url is None:
            raise RuntimeError("CoordinatorEndpoint: server not started")
        return self._url

    async def start(self, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        """Start accepting connections in the background.

        Args:
            host: Bind address; defaults to :data:`DEFAULT_HOST`.
            port: Bind port; defaults to :data:`DEFAULT_PORT`.

        Ensures:
            - The server is listening on ``host:port``.
            - :attr:`url` is available.
        """
        import websockets

        async def _handler(ws: Any) -> None:  # pyright: ignore[reportExplicitAny]
            await self._serve_connection(ws)

        self._server = await websockets.serve(
            _handler,
            host,
            port,
            max_size=MAX_MESSAGE_BYTES,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
        )
        self._host = host
        # When the caller asks for port 0, the OS assigns a free port; read it
        # back from the listening socket so self.url reflects reality.
        bound_port = port
        if port == 0:
            sockets = getattr(self._server, "sockets", None)
            if sockets:
                sock = next(iter(sockets))
                sockname: object = sock.getsockname()
                if isinstance(sockname, tuple):
                    tup = cast("tuple[object, ...]", sockname)
                    if len(tup) >= 2 and isinstance(tup[1], int):
                        bound_port = tup[1]
        self._port = bound_port
        self._url = f"ws://{host}:{bound_port}/rpc"

    async def stop(self) -> None:
        """Stop accepting new connections and close existing channels.

        Ensures:
            - No new connections are accepted.
            - Every connected channel is closed; remote workers are
              deregistered from the inner coordinator.
            - :meth:`start` / :meth:`serve` may be called again.

        Concurrency:
            Idempotent.
        """
        server = self._server
        if server is None:
            return
        server.close()
        with suppress(Exception):
            await server.wait_closed()
        self._server = None

        for channel in list(self._channels):
            await channel.close()
        self._channels.clear()

        self._host = None
        self._port = None
        self._url = None

    async def serve(self, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        """Start the server and block until cancelled.

        Convenience for long-running server processes::

            asyncio.run(CoordinatorEndpoint().serve())

        Args:
            host: Bind address; defaults to :data:`DEFAULT_HOST`.
            port: Bind port; defaults to :data:`DEFAULT_PORT`.
        """
        await self.start(host=host, port=port)
        try:
            await asyncio.Future()  # block forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def __aenter__(self) -> Self:
        """Return ``self`` for use in an ``async with`` block."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the server on context exit.

        Args:
            exc_type: Exception class raised inside the ``with`` block, if any.
            exc: Exception instance raised inside the ``with`` block, if any.
            tb: Traceback for the raised exception, if any.
        """
        del exc_type, exc, tb
        await self.stop()

    # ── Per-connection handling ─────────────────────────────────────────────

    async def _serve_connection(self, ws: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Serve one connected client until the WebSocket closes."""
        transport = WebsocketTransport(ws)
        channel = WireChannel(transport)
        self._channels.add(channel)

        handlers = CoordinatorHandlers(self._coordinator, channel)
        bind_handlers(channel, handlers)

        # Forward events from the inner coordinator to this client.
        event_sub = self._coordinator.on(
            lambda event: _forward_event(channel, event),
        )

        try:
            async with channel:
                while not channel.closed:
                    await asyncio.sleep(0.05)
        finally:
            # Deregister any workers this client hosted.
            for wid in list(handlers.hosted_workers.keys()):
                with suppress(Exception):
                    await self._coordinator.deregister_worker(wid)
            event_sub.unsubscribe()
            self._channels.discard(channel)


def _forward_event(channel: WireChannel, event: Event) -> None:
    """Schedule an event-forward to the connected client (fire and forget)."""
    if channel.closed:
        return

    async def _send() -> None:
        with suppress(Exception):
            await channel.send_event(event)

    _ = asyncio.create_task(_send())


# ── CoordinatorHandlers — dispatch table for inbound coordinator.* calls ─────


class CoordinatorHandlers:
    """Bundled inbound ``coordinator.*`` RPC handlers for a single channel.

    One instance per connected client. Each public method is decorated
    with :func:`~ai_functions.network.channel.rpc_method` so that
    :func:`~ai_functions.network.channel.bind_handlers` can install every
    ``coordinator.<op>`` handler on the channel at once.

    Attributes:
        hosted_workers: Maps :class:`WorkerId` → the remote worker
            adapter shim the endpoint built for this client. Populated
            by ``register_worker``, cleared on channel close.

    Args:
        coord: The inner in-memory coordinator this endpoint fronts.
        channel: The per-connection :class:`WireChannel` (used to
            build outbound ``worker.*`` RPC shims when a worker is
            registered).
    """

    __slots__ = ("_coord", "_channel", "hosted_workers")

    _coord: Coordinator
    _channel: WireChannel
    hosted_workers: dict[WorkerId, _RemoteWorkerAdapter]

    def __init__(self, coord: Coordinator, channel: WireChannel) -> None:
        self._coord = coord
        self._channel = channel
        self.hosted_workers = {}

    # ── Worker pool ──

    @rpc_method("coordinator.register_worker")
    async def register_worker(self, p: RegisterWorkerParams) -> None:
        adapter = _RemoteWorkerAdapter(self._channel, p.worker_id)
        self.hosted_workers[p.worker_id] = adapter
        await self._coord.register_worker(adapter)

    @rpc_method("coordinator.deregister_worker")
    async def deregister_worker(self, p: DeregisterWorkerParams) -> None:
        self.hosted_workers.pop(p.worker_id, None)
        await self._coord.deregister_worker(p.worker_id)

    # ── Thread registry ──

    @rpc_method("coordinator.register_thread")
    async def register_thread(self, p: RegisterThreadParams) -> None:
        await self._coord.register_thread(p.info)

    @rpc_method("coordinator.deregister_thread")
    async def deregister_thread(self, p: DeregisterThreadParams) -> None:
        await self._coord.deregister_thread(p.thread_id)

    # ── Discovery ──

    @rpc_method("coordinator.list_threads")
    async def list_threads(self, p: ListThreadsParams) -> list[ThreadInfo]:
        del p
        return await self._coord.list_threads()

    @rpc_method("coordinator.get_thread_info")
    async def get_thread_info(self, p: GetThreadInfoParams) -> ThreadInfo:
        return await self._coord.get_thread_info(p.thread_id)

    @rpc_method("coordinator.get_thread_status")
    async def get_thread_status(self, p: GetThreadStatusParams) -> ThreadStatus:
        return await self._coord.get_thread_status(p.thread_id)

    # ── Spawning ──

    @rpc_method("coordinator.spawn")
    async def spawn(self, p: SpawnParams) -> str:
        target = cast(
            "Spawnable[..., Any]",  # pyright: ignore[reportExplicitAny]
            unpickle_b64(p.target_pickle.decode("ascii")),
        )
        handle = await self._coord.spawn(
            target,
            seed_from=p.seed_from,
            seed_events=p.seed_events,
            worker_id=p.worker_id,
            thread_id=p.thread_id,
            thread_name=p.thread_name,
            parent_id=p.parent_id,
            metadata=p.metadata,
        )
        return str(handle.id)

    # ── Cross-thread operations ──

    @rpc_method("coordinator.submit")
    async def submit(self, p: SubmitParams) -> str:
        args, kwargs = cast(
            "tuple[tuple[object, ...], dict[str, object]]",
            unpickle_b64(p.args_kwargs_pickle.decode("ascii")),
        )
        fut = self._coord.submit(p.thread_id, *args, **kwargs)
        result = await fut
        return cloudpickle_b64(result)

    @rpc_method("coordinator.notify")
    async def notify(self, p: InjectMessageParams) -> None:
        await self._coord.notify(p.thread_id, p.text)

    @rpc_method("coordinator.pause")
    async def pause(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.pause(p.thread_id)

    @rpc_method("coordinator.resume")
    async def resume(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.resume(p.thread_id)

    @rpc_method("coordinator.cancel")
    async def cancel(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.cancel(p.thread_id)

    @rpc_method("coordinator.terminate")
    async def terminate(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.terminate(p.thread_id)

    @rpc_method("coordinator.terminate_now")
    async def terminate_now(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.terminate_now(p.thread_id)

    @rpc_method("coordinator.fork")
    async def fork(self, p: ThreadIdOnlyParams) -> str:
        handle = await self._coord.fork(p.thread_id)
        return str(handle.id)

    # ── Approvals ──

    @rpc_method("coordinator.resolve_approval")
    async def resolve_approval(self, p: ResolveApprovalParams) -> None:
        await self._coord.resolve_approval(p.thread_id, p.approval_id, p.decision)

    # ── Events ──

    @rpc_method("coordinator.append_event")
    async def append_event(self, p: AppendEventParams) -> None:
        # Synchronous on the inner coordinator; no await needed.
        self._coord.append_event(p.event)

    @rpc_method("coordinator.get_events")
    async def get_events(self, p: GetEventsParams) -> list[Event]:
        return await self._coord.get_events(
            p.thread_id,
            since_id=p.since_id,
            kinds=p.kinds,
            limit=p.limit,
        )

    @rpc_method("coordinator.new_message_id")
    async def new_message_id(self, p: NewMessageIdParams) -> str:
        del p
        return str(await self._coord.new_message_id())

    @rpc_method("coordinator.copy_events")
    async def copy_events(self, p: CopyEventsParams) -> None:
        await self._coord.copy_events(
            source_id=p.source_id,
            target_id=p.target_id,
            until_event_id=p.until_event_id,
        )

    # ── Rate limiting ──

    @rpc_method("coordinator.is_paused")
    async def is_paused(self, p: ThreadIdOnlyParams) -> bool:
        return await self._coord.is_paused(p.thread_id)

    @rpc_method("coordinator.wait_until_unpaused")
    async def wait_until_unpaused(self, p: ThreadIdOnlyParams) -> None:
        await self._coord.wait_until_unpaused(p.thread_id)


# ── Remote worker adapter ────────────────────────────────────────────────────


class _RemoteWorkerAdapter:
    """Server-side shim implementing ``WorkerAdapter`` over a wire channel.

    Built by the endpoint when a client registers a worker. Every method
    issues an outbound ``worker.<wid>.<method>`` call over the channel,
    which is dispatched on the client side by :class:`WorkerHandlers`.
    """

    __slots__ = ("_channel", "_worker_id")

    def __init__(self, channel: WireChannel, worker_id: WorkerId) -> None:
        self._channel = channel
        self._worker_id = worker_id

    @property
    def worker_id(self) -> WorkerId:
        return self._worker_id

    def _m(self, method: str) -> str:
        return f"worker.{self._worker_id}.{method}"

    async def spawn(
        self,
        target: Spawnable[..., Any],  # pyright: ignore[reportExplicitAny]
        *,
        thread_id: ThreadId,
        thread_name: str | None,
        parent_id: ThreadId | None,
        metadata: dict[str, object],
    ) -> None:
        params = WorkerSpawnParams(
            target_pickle=cloudpickle_b64(target).encode("ascii"),
            thread_id=thread_id,
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata,
        )
        _ = await self._channel.call(self._m("spawn"), **params.model_dump(mode="json"))

    def submit(
        self,
        thread_id: ThreadId,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()  # pyright: ignore[reportExplicitAny]

        async def _issue() -> None:
            params = WorkerSubmitParams(
                thread_id=thread_id,
                args_kwargs_pickle=cloudpickle_b64((args, kwargs)).encode("ascii"),
            )
            try:
                raw = await self._channel.call(self._m("submit"), **params.model_dump(mode="json"))
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
        params = WorkerInjectMessageParams(thread_id=thread_id, text=text)
        _ = await self._channel.call(self._m("notify"), **params.model_dump(mode="json"))

    async def cancel(self, thread_id: ThreadId) -> None:
        params = WorkerThreadIdParams(thread_id=thread_id)
        _ = await self._channel.call(self._m("cancel"), **params.model_dump(mode="json"))

    async def pause(self, thread_id: ThreadId) -> None:
        params = WorkerThreadIdParams(thread_id=thread_id)
        _ = await self._channel.call(self._m("pause"), **params.model_dump(mode="json"))

    async def resume(self, thread_id: ThreadId) -> None:
        params = WorkerThreadIdParams(thread_id=thread_id)
        _ = await self._channel.call(self._m("resume"), **params.model_dump(mode="json"))

    async def terminate(self, thread_id: ThreadId) -> None:
        params = WorkerThreadIdParams(thread_id=thread_id)
        _ = await self._channel.call(self._m("terminate"), **params.model_dump(mode="json"))

    async def terminate_now(self, thread_id: ThreadId) -> None:
        params = WorkerThreadIdParams(thread_id=thread_id)
        _ = await self._channel.call(self._m("terminate_now"), **params.model_dump(mode="json"))

    async def get_fork_spawnable(self, thread_id: ThreadId) -> Spawnable[..., Any]:  # pyright: ignore[reportExplicitAny]
        params = WorkerThreadIdParams(thread_id=thread_id)
        raw = await self._channel.call(self._m("get_fork_spawnable"), **params.model_dump(mode="json"))
        return cast(
            "Spawnable[..., Any]",  # pyright: ignore[reportExplicitAny]
            unpickle_b64(cast("str", raw)),
        )

    def resolve_approval(self, thread_id: ThreadId, approval_id: str, decision: str) -> bool:
        params = WorkerResolveApprovalParams(
            thread_id=thread_id,
            approval_id=approval_id,
            decision=decision,
        )

        async def _issue() -> None:
            with suppress(Exception):
                _ = await self._channel.call(
                    self._m("resolve_approval"),
                    **params.model_dump(mode="json"),
                )

        _ = asyncio.create_task(_issue())
        return True
