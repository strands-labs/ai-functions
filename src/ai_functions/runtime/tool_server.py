"""``CoordinatorToolServer`` — the runtime tools over streamable-HTTP MCP.

Serves the two coordinator tools from
:mod:`ai_functions.runtime.coordinator_tools_core` (``list_threads`` /
``send_message``) over HTTP, so agent runtimes that take an ``mcp_servers`` URL
— Codex among them — can join a team without per-runtime bridge code. Runs in
the process owning the coordinator.

Clients must send the token in the ``Authorization: Bearer`` header (Codex:
``codex mcp add --url ... --bearer-token-env-var``).

One server hosts any number of threads; each registration mints its own
capability URL::

    server = CoordinatorToolServer()
    await server.start()
    reg = server.register(coordinator, thread_id)
    # reg.url   -> http://127.0.0.1:<port>/mcp/<routing-id>
    # reg.token -> the secret for the Authorization header
    ...
    server.deregister(thread_id)
    await server.stop()

The server binds 127.0.0.1 on a system-assigned port; pass ``reg.url`` to the
runtime as it starts. The port is reachable by any local process, so the *token*
is the boundary: each registration mints a ``secrets``-grade token required in
the ``Authorization: Bearer`` header (constant-time compared), the ``Host``
header must name the bound address (DNS-rebinding defense per the MCP spec's
guidance for local HTTP servers), and deregistration revokes the token
immediately.

The MCP app runs stateless (``stateless_http=True``): tools and one-shot
reads only — no server-initiated messages, no resource subscriptions.
Session state lives in the coordinator, not the transport.

Requires the ``runtime-tools`` extra (``mcp``, ``uvicorn``).
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import secrets
import socket
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, final

from ..protocols import Coordinator
from ..types import ThreadId
from .coordinator_tools_core import (
    LIST_THREADS_DESCRIPTION,
    SEND_MESSAGE_DESCRIPTION,
    SendMessageMode,
)
from .coordinator_tools_core import list_threads as _core_list_threads
from .coordinator_tools_core import send_message as _core_send_message

try:
    import uvicorn
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "CoordinatorToolServer requires the optional 'runtime-tools' extra "
        "(the MCP server and an ASGI server). Install it with:\n"
        "    pip install 'strands-ai-functions[runtime-tools]'",
    ) from exc

_HOST = "127.0.0.1"
"""Bind address, and the only ``Host`` header the server answers to."""

_STARTUP_TIMEOUT = 10.0
"""Seconds to wait for uvicorn to report ``started`` before giving up."""


# ASGI shorthands (plain dicts/callables; no framework import needed).
_Scope = MutableMapping[str, Any]  # pyright: ignore[reportExplicitAny]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]  # pyright: ignore[reportExplicitAny]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]  # pyright: ignore[reportExplicitAny]


@dataclass(frozen=True)
class ToolServerRegistration:
    """One thread's capability to call the runtime tools."""

    url: str
    """Full MCP endpoint for this thread."""

    token: str
    """The bare secret, required in the ``Authorization: Bearer`` header on
    every request. Codex takes it via ``bearer_token_env_var``."""


@dataclass(frozen=True)
class _Registration:
    """Server-side binding of a routing id to the thread it acts as."""

    coordinator: Coordinator
    thread_id: ThreadId
    token: str


_active_registration: contextvars.ContextVar[_Registration | None] = contextvars.ContextVar(
    "ai_functions_tool_server_registration",
    default=None,
)


def _require_registration() -> _Registration:
    """Return the request's registration; the router guarantees one is set."""
    reg = _active_registration.get()
    if reg is None:  # pragma: no cover - unreachable behind the router
        raise RuntimeError("tool invoked outside an authenticated request")
    return reg


def _build_mcp_app() -> FastMCP:
    """Build the single stateless FastMCP app serving both runtime tools.

    Tool identity is per-request: the token router resolves which thread is
    calling and parks its registration in a context variable before handing
    the request to this app.
    """
    mcp = FastMCP("ai_functions_runtime", stateless_http=True)

    @mcp.tool(name="list_threads", description=LIST_THREADS_DESCRIPTION)
    async def list_threads() -> str:
        reg = _require_registration()
        result = await _core_list_threads(reg.coordinator, str(reg.thread_id))
        return result.model_dump_json()

    @mcp.tool(name="send_message", description=SEND_MESSAGE_DESCRIPTION)
    async def send_message(
        thread_id: str,
        message: str,
        mode: SendMessageMode = "wait",
    ) -> str:
        reg = _require_registration()
        return await _core_send_message(reg.coordinator, str(reg.thread_id), thread_id, message, mode)

    return mcp


async def _plain_response(send: _Send, status: int, body: str) -> None:
    """Emit a minimal text/plain ASGI response."""
    payload = body.encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(payload)).encode()),
            ],
        },
    )
    await send({"type": "http.response.body", "body": payload})


@final
class _TokenRouter:
    """Pure-ASGI wrapper enforcing the token and Host checks per request.

    Wraps the FastMCP app rather than subclassing any framework middleware so
    the inner app's lifespan passes through untouched (the streamable-HTTP
    session manager initialises in the lifespan; dropping it breaks every
    request).
    """

    def __init__(self, app: Callable[..., Awaitable[None]], server: CoordinatorToolServer) -> None:
        self._app = app
        self._server = server

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not self._host_allowed(scope):
            await _plain_response(send, 403, "forbidden: unrecognized Host")
            return

        path = str(scope.get("path", ""))
        prefix = "/mcp/"
        if not path.startswith(prefix):
            await _plain_response(send, 404, "not found")
            return
        path_id, _, rest = path[len(prefix) :].partition("/")

        registration = self._server._lookup(path_id)  # pyright: ignore[reportPrivateUsage]  # module-internal
        if registration is None:
            await _plain_response(send, 404, "not found")
            return

        bearer = self._bearer(scope)
        if bearer is None or not hmac.compare_digest(bearer, registration.token):
            await _plain_response(send, 401, "unauthorized")
            return

        # Rewrite to the path the FastMCP app is mounted on.
        scope["path"] = "/mcp" + (f"/{rest}" if rest else "")
        ctx_token = _active_registration.set(registration)
        try:
            await self._app(scope, receive, send)
        finally:
            _active_registration.reset(ctx_token)

    @staticmethod
    def _host_allowed(scope: _Scope) -> bool:
        """Accept only Host headers naming the bound host (DNS-rebinding defense)."""
        raw = dict(scope.get("headers") or []).get(b"host", b"")
        hostname = raw.decode("latin-1").rsplit(":", 1)[0].strip("[]").lower()
        return hostname == _HOST

    @staticmethod
    def _bearer(scope: _Scope) -> str | None:
        raw = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        scheme, _, value = raw.partition(" ")
        if scheme.lower() != "bearer" or not value:
            return None
        return value.strip()


@final
class CoordinatorToolServer:
    """A streamable-HTTP MCP server hosting the runtime tools.

    Lifecycle:
        constructed → ``start()`` (binds and serves) → ``register`` /
        ``deregister`` per thread → ``stop()``.

    Concurrency:
        ``start``/``stop`` must run on the same event loop — ``start`` spawns
        the serving task there and ``stop`` awaits it.
        ``register``/``deregister`` are synchronous dict operations, safe to
        call between requests; per-request identity travels in a context
        variable, so concurrent requests for different threads never observe
        each other's registration.
    """

    def __init__(self) -> None:
        """Configure the server; nothing binds until :meth:`start`."""
        self._registrations: dict[str, _Registration] = {}
        self._by_thread: dict[ThreadId, str] = {}
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Bind the socket and serve until :meth:`stop`.

        Ensures:
            - :attr:`base_url` is valid once this returns.
            - Requests are answered (all-404 until a thread registers).

        Raises:
            TimeoutError: Serving did not come up within
                :data:`_STARTUP_TIMEOUT` seconds.

        Concurrency:
            Idempotent; a started server ignores further ``start`` calls.
        """
        if self._server is not None:
            return

        sock = self._bind_socket()
        bound_port = int(sock.getsockname()[1])
        self._bound_port = bound_port

        app = _TokenRouter(_build_mcp_app().streamable_http_app(), self)
        config = uvicorn.Config(
            app,
            host=_HOST,
            port=bound_port,
            log_level="warning",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve(sockets=[sock]))
        deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT
        while not self._server.started:
            if self._serve_task.done():
                self._serve_task.result()  # surface the startup failure
                raise RuntimeError("tool server exited before startup completed")
            if asyncio.get_running_loop().time() >= deadline:
                await self.stop()
                raise TimeoutError(
                    f"tool server did not start within {_STARTUP_TIMEOUT}s",
                )
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Stop serving and revoke every registration.

        In-flight tool calls are dropped rather than drained: a client waiting
        on one (``send_message(mode="wait")``, say) sees the connection close
        mid-response.

        Concurrency:
            Idempotent; stopping a never-started server is a no-op.
        """
        self._registrations.clear()
        self._by_thread.clear()
        server, task = self._server, self._serve_task
        self._server = None
        self._serve_task = None
        self._bound_port = None
        if server is None or task is None:
            return
        server.should_exit = True
        await task

    @property
    def base_url(self) -> str:
        """Root URL of the running server (no registration path).

        Raises:
            RuntimeError: The server is not started.
        """
        if self._bound_port is None:
            raise RuntimeError("CoordinatorToolServer is not started")
        return f"http://{_HOST}:{self._bound_port}"

    # ── Registrations ──

    def register(self, coordinator: Coordinator, thread_id: ThreadId) -> ToolServerRegistration:
        """Mint a capability URL through which ``thread_id`` calls the tools.

        Re-registering a thread revokes its previous token and mints a fresh
        one.

        Args:
            coordinator: Coordinator the tools act on for this thread.
            thread_id: Identity the tools act *as* — ``is_self`` marking,
                ``send_message`` sender, and the reply target of
                ``continue_then_receive``.

        Returns:
            The registration; hand ``url`` and ``token`` to the agent runtime's
            MCP config.

        Raises:
            RuntimeError: The server is not started.
        """
        if self._bound_port is None:
            raise RuntimeError("CoordinatorToolServer is not started; call start() first")
        self.deregister(thread_id)
        path_id = secrets.token_urlsafe(8)
        token = secrets.token_urlsafe(32)
        self._registrations[path_id] = _Registration(
            coordinator=coordinator,
            thread_id=thread_id,
            token=token,
        )
        self._by_thread[thread_id] = path_id
        return ToolServerRegistration(url=f"{self.base_url}/mcp/{path_id}", token=token)

    def deregister(self, thread_id: ThreadId) -> None:
        """Revoke ``thread_id``'s registration; later requests to its URL 404.

        A request already dispatched keeps the registration it resolved for the
        rest of its lifetime; revocation is not a cancellation.

        Concurrency:
            Idempotent; deregistering an unknown thread is a no-op.
        """
        token = self._by_thread.pop(thread_id, None)
        if token is not None:
            _ = self._registrations.pop(token, None)

    # ── Internals ──

    def _lookup(self, path_id: str) -> _Registration | None:
        """Resolve a URL routing id to its registration; ``None`` when revoked/unknown."""
        return self._registrations.get(path_id)

    @staticmethod
    def _bind_socket() -> socket.socket:
        """Bind the listening socket on a system-assigned port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((_HOST, 0))
            sock.listen(2048)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        return sock
