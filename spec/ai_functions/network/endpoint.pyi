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

from types import TracebackType
from typing import Any, Self, final

from ..protocols import Coordinator
from ..types import WorkerId
from .channel import WireChannel


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

    def __init__(self, coordinator: Coordinator | None = None) -> None: ...

    @property
    def coordinator(self) -> Coordinator:
        """The inner coordinator this endpoint fronts."""
        ...

    @property
    def url(self) -> str:
        """The ``ws://host:port/rpc`` URL of the running server.

        Raises:
            RuntimeError: The server is not running.
        """
        ...

    async def start(self, *, host: str = ..., port: int = ...) -> None:
        """Start accepting connections in the background.

        Args:
            host: Bind address; defaults to :data:`DEFAULT_HOST`.
            port: Bind port; defaults to :data:`DEFAULT_PORT`.

        Ensures:
            - The server is listening on ``host:port``.
            - :attr:`url` is available.
        """
        ...

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
        ...

    async def serve(self, *, host: str = ..., port: int = ...) -> None:
        """Start the server and block until cancelled.

        Convenience for long-running server processes::

            asyncio.run(CoordinatorEndpoint().serve())

        Args:
            host: Bind address; defaults to :data:`DEFAULT_HOST`.
            port: Bind port; defaults to :data:`DEFAULT_PORT`.
        """
        ...

    async def __aenter__(self) -> Self:
        """Return ``self`` for use in an ``async with`` block."""
        ...

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
        ...


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

    hosted_workers: dict[WorkerId, Any]  # pyright: ignore[reportExplicitAny]

    def __init__(self, coord: Coordinator, channel: WireChannel) -> None: ...
