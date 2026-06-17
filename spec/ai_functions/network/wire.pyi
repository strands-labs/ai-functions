"""Wire protocol — pydantic frames for the Coordinator/Worker network layer.

The network layer is a symmetric, bidirectional RPC channel over a single
WebSocket. Either peer (client or endpoint) may initiate a call; the
other answers. Events broadcast from endpoint to client are the same
frame shape with no correlation expected.

Frames are discriminated on ``kind``:

- ``call``  — request: method name + JSON params. Awaits a ``result``
  or ``error`` frame with matching ``id``.
- ``result`` — success response; carries a JSON value.
- ``error``  — failure response; carries an error type + message.
- ``event``  — server-initiated event broadcast; no response expected.

A single ``Binary`` field is used for cloudpickle payloads (Spawnables,
run-cycle args/kwargs/results) which cannot be JSON-encoded.

Encoding: ``frame.model_dump_json()``
Decoding: ``TypeAdapter(Frame).validate_json(raw)``

The wire layer is transport-agnostic: any async duplex string channel
satisfying :class:`Transport` is acceptable. The default implementation
uses ``websockets``; FastAPI / Starlette WebSockets can be plugged in
via a thin adapter.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


DEFAULT_HOST: str = "127.0.0.1"
"""Default host for :class:`CoordinatorEndpoint` and :class:`CoordinatorClient`."""

DEFAULT_PORT: int = 9900
"""Default port for :class:`CoordinatorEndpoint` and :class:`CoordinatorClient`."""


# ── Binary escape type ───────────────────────────────────────────────────────


Binary = Annotated[bytes, ...]
"""Binary field encoded as base64 in JSON; accepts raw bytes on construction.

Used for cloudpickle-serialized spawnables and prompt args / kwargs /
results — payloads that can't round-trip through JSON natively.
"""


# ── Frames ────────────────────────────────────────────────────────────────────


class CallFrame(BaseModel):
    """Outbound RPC request.

    Fields:
        kind: Discriminator (always ``"call"``).
        id: Correlation id chosen by the caller; echoed in the matching
            ``result`` or ``error`` frame.
        method: Dotted method name the peer should dispatch
            (e.g. ``"coordinator.spawn"``, ``"worker.submit"``).
        params: JSON-compatible keyword arguments. Non-JSON values
            (cloudpickle blobs) are carried as ``Binary`` base64 fields
            inside this dict at method-specific keys.
    """

    kind: Literal["call"]
    id: str
    method: str
    params: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class ResultFrame(BaseModel):
    """Successful RPC response.

    Fields:
        kind: Discriminator (always ``"result"``).
        id: Matches the ``id`` of the originating ``CallFrame``.
        value: JSON-compatible return value; may contain ``Binary``
            blobs at method-specific keys.
    """

    kind: Literal["result"]
    id: str
    value: Any  # pyright: ignore[reportExplicitAny]


class ErrorFrame(BaseModel):
    """Failed RPC response.

    Fields:
        kind: Discriminator (always ``"error"``).
        id: Matches the ``id`` of the originating ``CallFrame``.
        type: Exception class name (e.g. ``"ThreadNotFoundError"``,
            ``"ValueError"``). Used on the caller side to reconstruct a
            typed exception when possible; otherwise surfaced as
            :class:`RemoteError`.
        message: Human-readable error description.
    """

    kind: Literal["error"]
    id: str
    type: str
    message: str


class EventFrame(BaseModel):
    """Broadcast from endpoint to client.

    Fields:
        kind: Discriminator (always ``"event"``).
        event: One :class:`ai_functions.types.Event` pydantic model, serialized
            in place (the wire layer calls ``model_dump`` / ``model_validate``
            on the tagged union).

    No correlation id — events are fire-and-forget from the sender's POV.
    """

    kind: Literal["event"]
    event: Any  # pyright: ignore[reportExplicitAny]  # ai_functions.types.Event union


Frame = Annotated[
    CallFrame | ResultFrame | ErrorFrame | EventFrame,
    Field(discriminator="kind"),
]
"""Discriminated union of every frame shape that crosses the wire."""


# ── Transport protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Transport(Protocol):
    """Thin async duplex string channel.

    Both sides of the wire layer talk to a ``Transport``. The default
    implementation wraps a ``websockets`` client / server socket; a
    FastAPI / Starlette WebSocket can be adapted to this protocol with a
    trivial wrapper (both expose ``send`` / ``recv`` with different
    method names).

    Implementations must be safe to call concurrently for ``send`` and
    ``recv`` from different tasks (full-duplex).
    """

    async def send(self, text: str) -> None:
        """Send one frame's serialized JSON over the wire.

        Args:
            text: The ``frame.model_dump_json()`` output.
        """
        ...

    async def recv(self) -> str:
        """Receive one frame's serialized JSON from the wire.

        Returns:
            The next frame's JSON text, as sent by the peer.

        Raises:
            ConnectionClosedError: The peer closed the connection.
        """
        ...

    async def close(self) -> None:
        """Close the underlying connection idempotently."""
        ...


# ── Error hierarchy ──────────────────────────────────────────────────────────


class WireError(Exception):
    """Base class for wire-layer errors."""


class RemoteError(WireError):
    """A peer returned an ``ErrorFrame`` whose ``type`` we cannot rehydrate.

    Attributes:
        remote_type (str): The ``type`` field from the ErrorFrame — the
            peer's exception class name.
        message (str): The ``message`` field from the ErrorFrame.

    Args:
        remote_type: The ``type`` field from the ErrorFrame.
        message: The ``message`` field from the ErrorFrame.
    """

    remote_type: str
    message: str

    def __init__(self, remote_type: str, message: str) -> None: ...


class ConnectionClosedError(WireError):
    """The peer connection closed before a pending call resolved."""
