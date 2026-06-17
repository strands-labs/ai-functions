"""WireChannel — symmetric RPC machinery shared by client and endpoint.

A :class:`WireChannel` wraps a :class:`Transport` and provides:

- ``call(method, **params)`` — initiate an outbound RPC; await the result.
- ``send_event(event)`` — broadcast a server-initiated event (no response).
- ``register(method_name, handler)`` — install an inbound dispatch handler.
- ``on_event(callback)`` — subscribe to inbound event frames.

Both sides of the wire use the same ``WireChannel`` class. The asymmetry
between "client" and "endpoint" is entirely in which handlers each side
registers (``coordinator.*`` on the endpoint; ``worker.*`` on a client
that hosts a worker) — the channel itself is symmetric.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Self

from ..types import Event
from .wire import Transport


Handler = Callable[[dict[str, Any]], Awaitable[Any]]  # pyright: ignore[reportExplicitAny]
"""Inbound call handler: takes parsed ``params``, returns a JSON-compatible value.

Raising an exception produces an ``ErrorFrame`` in response; the
exception's class name becomes the frame ``type``, ``str(exc)`` becomes
the ``message``.
"""

EventCallback = Callable[[Event], None]
"""Subscriber callback invoked for each inbound ``EventFrame``."""


class WireChannel:
    """Symmetric bidirectional JSON-frame RPC channel.

    Owns one reader task that decodes incoming frames and routes them:

    - ``call`` frames → dispatched via the handler table; response
      frames written back.
    - ``result`` / ``error`` frames → resolve / reject the matching
      pending future in the outbound-calls map.
    - ``event`` frames → fanned out to subscribed callbacks.

    Args:
        transport: The underlying duplex string channel.

    Concurrency:
        ``call``, ``send_event``, ``register``, and ``on_event`` are all
        safe to invoke from multiple tasks concurrently.
    """

    def __init__(self, transport: Transport) -> None: ...

    async def __aenter__(self) -> Self:
        """Start the reader task and return ``self``."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Cancel the reader, close the transport, reject pending calls.

        Args:
            exc_type: Exception class raised inside the ``with`` block, if any.
            exc: Exception instance raised inside the ``with`` block, if any.
            tb: Traceback for the raised exception, if any.
        """
        ...

    # ── Outbound ─────────────────────────────────────────────────────────────

    async def call(self, method: str, **params: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Send a ``CallFrame`` and await the peer's response.

        Args:
            method: Dotted method name dispatched by the peer.
            params: JSON-compatible keyword arguments.

        Returns:
            The JSON-compatible ``value`` field of the peer's
            ``ResultFrame``.

        Raises:
            RemoteError: The peer replied with an ``ErrorFrame`` whose
                ``type`` we can't rehydrate locally.
            ThreadNotFoundError, ValueError, NotImplementedError: Re-raised
                when the peer's ``type`` matches a known exception class.
            ConnectionClosedError: The transport closed before the
                response arrived.
        """
        ...

    async def send_event(self, event: Event) -> None:
        """Send an ``EventFrame`` to the peer; no response expected.

        Args:
            event: The event to broadcast.
        """
        ...

    # ── Inbound ──────────────────────────────────────────────────────────────

    def register(self, method: str, handler: Handler) -> None:
        """Install a dispatch handler for inbound ``CallFrame`` s.

        Args:
            method: Dotted method name the peer will address.
            handler: Called with the frame's ``params`` dict; its return
                value becomes the ``ResultFrame.value``.

        Raises:
            ValueError: A handler is already registered for ``method``.
        """
        ...

    def on_event(self, callback: EventCallback) -> Callable[[], None]:
        """Subscribe to inbound ``EventFrame`` s.

        Args:
            callback: Invoked synchronously for each inbound event.

        Returns:
            An unsubscribe callable; calling it removes ``callback``
            from the subscription list (idempotent).
        """
        ...

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def closed(self) -> bool:
        """Whether the channel has been closed (either side).

        After ``closed`` becomes ``True``, ``call`` and ``send_event``
        raise ``ConnectionClosedError``.
        """
        ...

    async def close(self) -> None:
        """Close the channel idempotently.

        Cancels the reader task, rejects every pending outbound call
        with :class:`ConnectionClosedError`, and closes the underlying
        transport. Safe to call multiple times.
        """
        ...


# ── Helpers / decorators for typed handler dispatch ──────────────────────────


EVENT_ADAPTER: Any  # pyright: ignore[reportExplicitAny]
"""TypeAdapter for the :class:`Event` tagged union; shared by client and endpoint."""


def cloudpickle_b64(value: object) -> str:
    """Cloudpickle-then-base64 helper for ``target`` / args / results on the wire.

    Args:
        value: Any cloudpicklable Python object.

    Returns:
        The base64 string representation of ``cloudpickle.dumps(value)``.
    """
    ...


def unpickle_b64(blob: str) -> object:
    """Inverse of :func:`cloudpickle_b64`.

    Args:
        blob: The base64 text produced by :func:`cloudpickle_b64`.

    Returns:
        The rehydrated Python object.
    """
    ...


def event_to_dict(event: Event) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Serialize an :class:`Event` (tagged union) to a wire-safe dict.

    Args:
        event: An event from the :class:`Event` tagged union.

    Returns:
        A JSON-compatible dict round-trippable through ``EVENT_ADAPTER``.
    """
    ...


def rpc_method(
    name: str,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:  # pyright: ignore[reportExplicitAny]
    """Tag a coroutine method as an RPC handler for the given wire name.

    The decorated method must take exactly one positional argument
    besides ``self``: a pydantic ``BaseModel`` subclass (the params
    model). At bind time, the channel validates incoming
    ``CallFrame.params`` against that model before calling the handler;
    the handler's return value becomes the ``ResultFrame.value``.

    Args:
        name: Dotted wire method name (e.g. ``"coordinator.spawn"``).

    Returns:
        A decorator that stamps the wire name onto the handler.
    """
    ...


def bind_handlers(
    channel: WireChannel,
    instance: object,
    *,
    rename: Callable[[str], str] | None = None,
) -> None:
    """Install every :func:`rpc_method`-decorated method on ``instance`` onto ``channel``.

    For each decorated method, inspects the params annotation to locate
    the pydantic model class, and registers a wrapper that:

    - validates ``CallFrame.params`` via ``Model.model_validate`` before
      dispatch;
    - serializes pydantic results, lists, and enums on the way out;
    - passes primitives / base64 strings / dicts through unchanged.

    Args:
        channel: The channel whose handler table receives the registrations.
        instance: An object whose methods carry the :func:`rpc_method` tag.
        rename: Optional function from the decorator's declared method
            name to the name registered on the channel — for per-instance
            namespacing (e.g. prefixing a worker id onto the
            :class:`WorkerHandlers` tails).

    Raises:
        TypeError: A decorated method's signature isn't
            ``(self, params: <BaseModel subclass>)``.
    """
    ...


class WebsocketTransport:
    """Adapt a ``websockets`` connection to the :class:`Transport` protocol.

    Used for both sides: ``websockets.connect()`` returns a client
    WebSocket; the object yielded by ``websockets.serve``'s handler is a
    server WebSocket. Both expose the same ``send`` / ``recv`` / ``close``
    API, so one adapter covers both.

    Args:
        ws: A ``websockets`` connection object.
    """

    def __init__(self, ws: Any) -> None: ...  # pyright: ignore[reportExplicitAny]

    async def send(self, text: str) -> None:
        """Send one frame's JSON text over the WebSocket.

        Args:
            text: The frame JSON to send.
        """
        ...

    async def recv(self) -> str:
        """Receive the next frame's JSON text from the WebSocket.

        Returns:
            The raw frame JSON as a string.
        """
        ...

    async def close(self) -> None:
        """Close the WebSocket; idempotent."""
        ...
