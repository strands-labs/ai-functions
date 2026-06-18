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

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import TracebackType
from typing import Any, Self, cast

from pydantic import TypeAdapter

from ..runtime.errors import ThreadNotFoundError
from ..types import Event
from .wire import (
    CallFrame,
    ConnectionClosedError,
    ErrorFrame,
    EventFrame,
    Frame,
    RemoteError,
    ResultFrame,
    Transport,
)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]  # pyright: ignore[reportExplicitAny]
EventCallback = Callable[[Event], None]

_logger: logging.Logger = logging.getLogger("ai_functions.network.channel")

# WebSocket close codes that represent a clean, expected shutdown. Anything
# else (1006 abnormal, 1009 too-big, 1011 keepalive/internal, …) is logged at
# ERROR so a mid-run disconnect leaves a legible reason behind.
_CLEAN_CLOSE_CODES: frozenset[int] = frozenset({1000, 1001})


def _describe_transport_close(exc: BaseException) -> tuple[str, bool]:
    """Summarise why the transport's ``recv`` raised, for logging.

    Pulls the WebSocket close code and reason out of a ``websockets``
    ``ConnectionClosed`` exception when present, so an opaque "connection
    closed" becomes e.g. "code 1009 (message too big): ..." — the single
    most useful fact when diagnosing a mid-run drop.

    Args:
        exc: The exception raised by ``Transport.recv``.

    Returns:
        A ``(description, is_clean)`` pair. ``description`` is a one-line
        human summary; ``is_clean`` is ``True`` only when the close carried
        a normal/going-away code (1000 / 1001), so the caller can pick the
        log level.
    """
    rcvd = getattr(exc, "rcvd", None)
    sent = getattr(exc, "sent", None)
    close = rcvd if rcvd is not None else sent
    code = getattr(close, "code", None)
    if code is None:
        # Not a websockets ConnectionClosed (e.g. a raw OSError) — no close
        # frame to inspect; report the exception itself.
        return f"{type(exc).__name__}: {exc}", False
    reason = getattr(close, "reason", "") or ""
    origin = "peer" if rcvd is not None else "local"
    detail = f": {reason}" if reason else ""
    return f"code {code} from {origin}{detail}", code in _CLEAN_CLOSE_CODES


# Map well-known exception classes to / from their type names on the wire.
# Extend as needed; unknown names produce a RemoteError on the caller side.
_KNOWN_EXCEPTIONS: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "ThreadNotFoundError": ThreadNotFoundError,
}


_FRAME_ADAPTER: TypeAdapter[Frame] = TypeAdapter(Frame)  # pyright: ignore[reportInvalidTypeForm]


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

    __slots__ = (
        "_transport",
        "_handlers",
        "_event_subs",
        "_pending",
        "_reader_task",
        "_closed",
        "_closing_lock",
    )

    _transport: Transport
    _handlers: dict[str, Handler]
    _event_subs: list[EventCallback]
    _pending: dict[str, asyncio.Future[Any]]  # pyright: ignore[reportExplicitAny]
    _reader_task: asyncio.Task[None] | None
    _closed: bool
    _closing_lock: asyncio.Lock

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._handlers = {}
        self._event_subs = []
        self._pending = {}
        self._reader_task = None
        self._closed = False
        self._closing_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Start the reader task and return ``self``."""
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

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
        del exc_type, exc, tb
        await self.close()

    @property
    def closed(self) -> bool:
        """Whether the channel has been closed (either side).

        After ``closed`` becomes ``True``, ``call`` and ``send_event``
        raise ``ConnectionClosedError``.
        """
        return self._closed

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
        if self._closed:
            raise ConnectionClosedError(f"channel closed; cannot call {method!r}")
        call_id = f"c-{uuid.uuid4().hex[:16]}"
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()  # pyright: ignore[reportExplicitAny]
        self._pending[call_id] = future
        frame = CallFrame(id=call_id, method=method, params=params)
        try:
            await self._transport.send(frame.model_dump_json())
        except Exception:
            _ = self._pending.pop(call_id, None)
            raise
        return await future

    async def send_event(self, event: Event) -> None:
        """Send an ``EventFrame`` to the peer; no response expected.

        Args:
            event: The event to broadcast.
        """
        if self._closed:
            raise ConnectionClosedError("channel closed; cannot send event")
        # Pydantic event models serialize their own tagged-union shape
        # via ``model_dump``; the TypeAdapter on the peer side matches.
        frame = EventFrame(event=event.model_dump(mode="json"))  # type: ignore[union-attr]
        await self._transport.send(frame.model_dump_json())

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
        if method in self._handlers:
            raise ValueError(f"handler for {method!r} already registered")
        self._handlers[method] = handler

    def on_event(self, callback: EventCallback) -> Callable[[], None]:
        """Subscribe to inbound ``EventFrame`` s.

        Args:
            callback: Invoked synchronously for each inbound event.

        Returns:
            An unsubscribe callable; calling it removes ``callback``
            from the subscription list (idempotent).
        """
        self._event_subs.append(callback)

        def _unsubscribe() -> None:
            try:
                self._event_subs.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    # ── Close ────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the channel idempotently.

        Cancels the reader task, rejects every pending outbound call
        with :class:`ConnectionClosedError`, and closes the underlying
        transport. Safe to call multiple times.
        """
        async with self._closing_lock:
            if self._closed:
                return
            self._closed = True

            # Reject pending calls so awaiters wake up.
            err = ConnectionClosedError("channel closed")
            for call_id, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_exception(err)
                _ = self._pending.pop(call_id, None)

            # Cancel the reader if still running.
            if self._reader_task is not None and not self._reader_task.done():
                _ = self._reader_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self._reader_task

            with suppress(Exception):
                await self._transport.close()

    # ── Reader loop ──────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Consume frames until the transport closes.

        Catches every exception the transport may raise on close — the
        concrete ``websockets`` library signals normal shutdown with a
        ``ConnectionClosedOK`` (subclass of ``Exception``), which is
        not an error condition for us.
        """
        try:
            while not self._closed:
                try:
                    text = await self._transport.recv()
                except Exception as exc:  # noqa: BLE001 -- any transport close ends the loop
                    # Surface *why* the transport closed. The close code/reason
                    # is the most diagnostic fact about a mid-run drop and is
                    # otherwise lost when the loop simply breaks. A clean
                    # going-away close is expected (DEBUG); anything else —
                    # 1006 abnormal, 1009 too-big, 1011 keepalive — is logged at
                    # ERROR so it is never silent again.
                    description, is_clean = _describe_transport_close(exc)
                    if is_clean:
                        _logger.debug("wire transport closed (%s)", description)
                    else:
                        _logger.error("wire transport closed abnormally (%s)", description)
                    break
                try:
                    frame = _FRAME_ADAPTER.validate_json(text)
                except Exception:  # noqa: BLE001 -- malformed frame; skip, keep reading
                    continue
                await self._dispatch(frame)
        finally:
            # Any pending futures waiting on results will fail in close().
            if not self._closed:
                _ = asyncio.create_task(self.close())

    async def _dispatch(self, frame: Frame) -> None:  # pyright: ignore[reportInvalidTypeForm]
        """Route one decoded frame to the appropriate handler."""
        match frame:
            case ResultFrame(id=call_id, value=value):
                fut = self._pending.pop(call_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(value)
            case ErrorFrame(id=call_id, type=err_type, message=msg):
                fut = self._pending.pop(call_id, None)
                if fut is not None and not fut.done():
                    fut.set_exception(_rehydrate_error(err_type, msg))
            case CallFrame(id=call_id, method=method, params=params):
                # Dispatch async so multiple inbound calls can interleave.
                _ = asyncio.create_task(self._handle_call(call_id, method, params))
            case EventFrame(event=event_data):
                event = _decode_event(event_data)
                if event is None:
                    return
                for cb in list(self._event_subs):
                    try:
                        cb(event)
                    except Exception:  # noqa: BLE001 -- subscriber errors don't break the channel
                        pass

    async def _handle_call(self, call_id: str, method: str, params: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        """Dispatch one inbound CallFrame, send back a ResultFrame or ErrorFrame."""
        handler = self._handlers.get(method)
        if handler is None:
            err = ErrorFrame(
                id=call_id,
                type="NotImplementedError",
                message=f"no handler registered for method {method!r}",
            )
            with suppress(Exception):
                await self._transport.send(err.model_dump_json())
            return

        try:
            value = await handler(params)
        except Exception as exc:  # noqa: BLE001 -- every handler error becomes a wire frame
            err = ErrorFrame(
                id=call_id,
                type=type(exc).__name__,
                message=str(exc),
            )
            with suppress(Exception):
                await self._transport.send(err.model_dump_json())
            return

        result = ResultFrame(id=call_id, value=value)
        with suppress(Exception):
            await self._transport.send(result.model_dump_json())


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rehydrate_error(err_type: str, message: str) -> Exception:
    """Reconstruct a typed exception from a wire ErrorFrame."""
    cls = _KNOWN_EXCEPTIONS.get(err_type)
    if cls is None:
        return RemoteError(err_type, message)
    try:
        return cls(message)
    except Exception:  # noqa: BLE001 -- some exception classes have custom __init__
        return RemoteError(err_type, message)


EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)  # pyright: ignore[reportInvalidTypeForm]
"""TypeAdapter for the :class:`Event` tagged union; shared by client and endpoint."""


def cloudpickle_b64(value: object) -> str:
    """Cloudpickle-then-base64 helper for ``target`` / args / results on the wire.

    Args:
        value: Any cloudpicklable Python object.

    Returns:
        The base64 string representation of ``cloudpickle.dumps(value)``.
    """
    import base64

    import cloudpickle  # pyright: ignore[reportMissingTypeStubs]

    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")  # pyright: ignore[reportUnknownMemberType]


def unpickle_b64(blob: str) -> object:
    """Inverse of :func:`cloudpickle_b64`.

    Args:
        blob: The base64 text produced by :func:`cloudpickle_b64`.

    Returns:
        The rehydrated Python object.
    """
    import base64

    import cloudpickle  # pyright: ignore[reportMissingTypeStubs]

    return cloudpickle.loads(base64.b64decode(blob.encode("ascii")))  # pyright: ignore[reportUnknownMemberType]


def event_to_dict(event: Event) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Serialize an :class:`Event` (tagged union) to a wire-safe dict.

    Args:
        event: An event from the :class:`Event` tagged union.

    Returns:
        A JSON-compatible dict round-trippable through ``EVENT_ADAPTER``.
    """
    from typing import cast as _cast

    return _cast("dict[str, Any]", EVENT_ADAPTER.dump_python(event, mode="json"))  # pyright: ignore[reportExplicitAny]


class WebsocketTransport:
    """Adapt a ``websockets`` connection to the :class:`Transport` protocol.

    Used for both sides: ``websockets.connect()`` returns a client
    WebSocket; the object yielded by ``websockets.serve``'s handler is a
    server WebSocket. Both expose the same ``send`` / ``recv`` / ``close``
    API, so one adapter covers both.

    Args:
        ws: A ``websockets`` connection object.
    """

    __slots__ = ("_ws",)

    def __init__(self, ws: Any) -> None:  # pyright: ignore[reportExplicitAny]
        self._ws: Any = ws  # pyright: ignore[reportExplicitAny]

    async def send(self, text: str) -> None:
        """Send one frame's JSON text over the WebSocket.

        Args:
            text: The frame JSON to send.
        """
        await self._ws.send(text)

    async def recv(self) -> str:
        """Receive the next frame's JSON text from the WebSocket.

        Returns:
            The raw frame JSON as a string.
        """
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        from typing import cast as _cast

        return _cast("str", raw)

    async def close(self) -> None:
        """Close the WebSocket; idempotent."""
        try:
            await self._ws.close()
        except Exception:  # noqa: BLE001
            pass


def _decode_event(raw: object) -> Event | None:
    """Rehydrate a pydantic Event from its dumped dict."""
    if not isinstance(raw, dict):
        return None
    try:
        return EVENT_ADAPTER.validate_python(raw)
    except Exception:  # noqa: BLE001 -- unknown event kinds: skip
        return None


# ── Typed handler decorator + binder ─────────────────────────────────────────


_RPC_METHOD_ATTR: str = "__ai_functions_rpc_method__"


def rpc_method(name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:  # pyright: ignore[reportExplicitAny]
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

    def _decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:  # pyright: ignore[reportExplicitAny]
        setattr(fn, _RPC_METHOD_ATTR, name)
        return fn

    return _decorator


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
    import inspect

    from pydantic import BaseModel

    for attr_name in dir(instance):
        if attr_name.startswith("__"):
            continue
        method = getattr(instance, attr_name, None)
        if not callable(method):
            continue
        rpc_name = getattr(method, _RPC_METHOD_ATTR, None)
        if rpc_name is None:
            continue

        # Introspect the params model from the annotation.
        sig = inspect.signature(method)
        params_list = list(sig.parameters.values())
        if len(params_list) != 1:
            raise TypeError(
                f"@rpc_method handler {attr_name!r} on {type(instance).__name__} "
                f"must take exactly one param (the pydantic model), got {len(params_list)}",
            )
        param_ann = params_list[0].annotation
        # Handle string annotations (from `from __future__ import annotations`).
        if isinstance(param_ann, str):
            try:
                param_ann = eval(param_ann, getattr(method, "__globals__", None) or {})  # noqa: S307
            except Exception:  # noqa: BLE001
                raise TypeError(  # noqa: B904
                    f"@rpc_method handler {rpc_name!r}: cannot resolve annotation {param_ann!r}",
                ) from None
        if not (isinstance(param_ann, type) and issubclass(param_ann, BaseModel)):
            raise TypeError(
                f"@rpc_method handler {rpc_name!r} must annotate its params as a "
                f"pydantic BaseModel subclass; got {param_ann!r}",
            )
        model_cls = param_ann

        def _make_wrapper(
            bound_method: Any,  # pyright: ignore[reportExplicitAny]
            model: type[BaseModel],
        ) -> Handler:
            async def _wrapped(raw: dict[str, Any]) -> Any:  # pyright: ignore[reportExplicitAny]
                params = model.model_validate(raw)
                result = await bound_method(params)
                return _serialize_result(result)

            return _wrapped

        final_name = rename(rpc_name) if rename is not None else rpc_name
        channel.register(final_name, _make_wrapper(method, model_cls))


def _serialize_result(value: object) -> object:
    """Coerce a handler's return value into a JSON-compatible wire value.

    - ``None`` → ``None``.
    - pydantic ``BaseModel`` → ``.model_dump(mode="json")``.
    - ``list`` / ``tuple`` → element-wise serialization.
    - Enum / StrEnum → ``str(value)``.
    - Everything else passes through (JSON primitives, already-serialized
      dicts, base64 strings for cloudpickle payloads, etc.).
    """
    from enum import Enum

    from pydantic import BaseModel

    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, (list, tuple)):
        items: list[object] = list(cast("list[object]", value))
        return [_serialize_result(v) for v in items]
    return value
