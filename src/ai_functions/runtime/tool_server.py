"""``CoordinatorToolServer`` — the runtime tools over streamable-HTTP MCP.

Serves the two coordinator tools from
:mod:`ai_functions.runtime.coordinator_tools_core` (``list_threads`` /
``send_message``) to any MCP-speaking agent runtime over HTTP, so foreign
backends that cannot host an in-process MCP server (Codex, Kiro, anything
else with an ``mcp_servers`` config) can join a team without per-runtime
bridge code.

One server per worker. Each thread registers to receive its own capability
URL::

    server = CoordinatorToolServer()
    await server.start()
    reg = server.register(coordinator, thread_id)
    # reg.url  -> http://127.0.0.1:8787/mcp/<token>
    # reg.token, for transports that pass the secret out of band
    ...
    server.deregister(thread_id)
    await server.stop()

Security model: the port is reachable by any local process, so the
*token* is the boundary. Each registration mints a ``secrets``-grade token
that must appear both in the URL path and in the ``Authorization: Bearer``
header (constant-time compared); the ``Host`` header must resolve to the
bound host (DNS-rebinding defense per the MCP spec's guidance for local
HTTP servers); deregistration revokes the token immediately.

The MCP app runs stateless (``stateless_http=True``): tools and one-shot
reads only — no server-initiated messages, no resource subscriptions.
Session state lives in the coordinator, not the transport.

Requires the ``runtime-tools`` extra (``mcp``, ``uvicorn``).
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import hmac
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8787
"""Default fixed port. Fixed (not ephemeral) by design: MCP URL allowlists
(e.g. Claude Code's ``allowedMcpServers``) match on URL patterns that must be
written before the server exists, so one glob like ``http://127.0.0.1:8787/*``
can cover every thread for the deployment's lifetime."""

# ASGI shorthands (plain dicts/callables; no framework import needed).
_Scope = MutableMapping[str, Any]  # pyright: ignore[reportExplicitAny]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]  # pyright: ignore[reportExplicitAny]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]  # pyright: ignore[reportExplicitAny]


@dataclass(frozen=True)
class ToolServerRegistration:
    """One thread's capability to call the runtime tools."""

    url: str
    """Full MCP endpoint for this thread, token embedded in the path."""

    token: str
    """The bare secret, for transports that pass it out of band (e.g. Codex's
    ``bearer_token_env_var``). Required in the ``Authorization: Bearer`` header
    on every request."""


@dataclass(frozen=True)
class _Registration:
    """Server-side binding of a token to the thread it acts as."""

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
        token, _, rest = path[len(prefix) :].partition("/")

        registration = self._server._lookup(token)  # pyright: ignore[reportPrivateUsage]  # module-internal
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

    def _host_allowed(self, scope: _Scope) -> bool:
        """Accept only Host headers naming the bound host (DNS-rebinding defense)."""
        raw = dict(scope.get("headers") or []).get(b"host", b"")
        hostname = raw.decode("latin-1").rsplit(":", 1)[0].strip("[]").lower()
        return hostname in self._server._allowed_hostnames  # pyright: ignore[reportPrivateUsage]  # module-internal

    @staticmethod
    def _bearer(scope: _Scope) -> str | None:
        raw = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        scheme, _, value = raw.partition(" ")
        if scheme.lower() != "bearer" or not value:
            return None
        return value.strip()


@final
class CoordinatorToolServer:
    """One streamable-HTTP MCP server hosting the runtime tools for a worker.

    Lifecycle:
        constructed → ``start()`` (binds and serves) → ``register`` /
        ``deregister`` per thread → ``stop()``.

    Concurrency:
        ``start``/``stop`` are owned by the hosting worker's event loop.
        ``register``/``deregister`` are synchronous dict operations, safe to
        call between requests; per-request identity travels in a context
        variable, so concurrent requests for different threads never observe
        each other's registration.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        fallback_to_ephemeral: bool = True,
    ) -> None:
        """Configure the server; nothing binds until :meth:`start`.

        Args:
            host: Interface to bind. The default serves local agent
                subprocesses only; binding wider is the caller's decision and
                should come with real network auth in front.
            port: Fixed port to bind. Fixed by design — see
                :data:`DEFAULT_PORT`. ``0`` requests an ephemeral port
                outright.
            fallback_to_ephemeral: When the fixed port is taken, bind an
                ephemeral one instead of failing. The fallback logs a warning
                because a URL allowlist written for the fixed port will not
                match it.
        """
        self._host = host
        self._requested_port = port
        self._fallback = fallback_to_ephemeral
        self._registrations: dict[str, _Registration] = {}
        self._by_thread: dict[ThreadId, str] = {}
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None
        self._allowed_hostnames: frozenset[str] = frozenset(
            {host.lower(), "localhost", "127.0.0.1", "::1"},
        )

    # ── Lifecycle ──

    async def start(self) -> None:
        """Bind the socket and serve until :meth:`stop`.

        Ensures:
            - :attr:`base_url` is valid once this returns.
            - Requests are answered (all-404 until a thread registers).

        Raises:
            OSError: The fixed port is taken and ``fallback_to_ephemeral``
                is false.

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
            host=self._host,
            port=bound_port,
            log_level="warning",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve(sockets=[sock]))
        while not self._server.started:
            if self._serve_task.done():
                self._serve_task.result()  # surface the startup failure
                raise RuntimeError("tool server exited before startup completed")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Stop serving and revoke every registration.

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
        """Root URL of the running server (no token).

        Raises:
            RuntimeError: The server is not started.
        """
        if self._bound_port is None:
            raise RuntimeError("CoordinatorToolServer is not started")
        return f"http://{self._host}:{self._bound_port}"

    @property
    def port(self) -> int | None:
        """The bound port while running, else ``None``."""
        return self._bound_port

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
            The registration; hand ``url`` (and, for out-of-band transports,
            ``token``) to the agent runtime's MCP config.

        Raises:
            RuntimeError: The server is not started.
        """
        if self._bound_port is None:
            raise RuntimeError("CoordinatorToolServer is not started; call start() first")
        self.deregister(thread_id)
        token = secrets.token_urlsafe(32)
        self._registrations[token] = _Registration(
            coordinator=coordinator,
            thread_id=thread_id,
            token=token,
        )
        self._by_thread[thread_id] = token
        return ToolServerRegistration(url=f"{self.base_url}/mcp/{token}", token=token)

    def deregister(self, thread_id: ThreadId) -> None:
        """Revoke ``thread_id``'s token; requests with it fail immediately.

        Concurrency:
            Idempotent; deregistering an unknown thread is a no-op.
        """
        token = self._by_thread.pop(thread_id, None)
        if token is not None:
            _ = self._registrations.pop(token, None)

    # ── Internals ──

    def _lookup(self, token: str) -> _Registration | None:
        """Resolve a path token to its registration; ``None`` when revoked/unknown."""
        return self._registrations.get(token)

    def _bind_socket(self) -> socket.socket:
        """Bind the listening socket, honouring the ephemeral fallback."""
        try:
            return self._bind(self._requested_port)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or not self._fallback or self._requested_port == 0:
                raise
            logger.warning(
                "tool server port %d is in use; falling back to an ephemeral port — "
                "URL allowlists written for the fixed port will not match",
                self._requested_port,
            )
            return self._bind(0)

    def _bind(self, port: int) -> socket.socket:
        family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, port))
            sock.listen(2048)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        return sock
