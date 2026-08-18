"""``CoordinatorToolServer`` — the runtime tools over streamable-HTTP MCP.

Serves the two coordinator tools from
:mod:`ai_functions.runtime.coordinator_tools_core` (``list_threads`` /
``send_message``) to any MCP-speaking agent runtime over HTTP, so foreign
backends that cannot host an in-process MCP server (Codex, Kiro, anything
else with an ``mcp_servers`` config) can join a team without per-runtime
bridge code.

One server per worker; each thread registers to receive its own capability
URL of the form ``http://<host>:<port>/mcp/<token>``.

Security model: the port is reachable by any local process, so the *token*
is the boundary. Each registration mints a ``secrets``-grade token that must
appear both in the URL path and in the ``Authorization: Bearer`` header
(constant-time compared); the ``Host`` header must resolve to the bound host
(DNS-rebinding defense per the MCP spec's guidance for local HTTP servers);
deregistration revokes the token immediately.

The MCP app runs stateless (``stateless_http=True``): tools and one-shot
reads only — no server-initiated messages, no resource subscriptions.

Requires the ``runtime-tools`` extra (``mcp``, ``uvicorn``).
"""

from dataclasses import dataclass
from typing import final

from ..protocols import Coordinator
from ..types import ThreadId

DEFAULT_PORT: int
"""Default fixed port. Fixed (not ephemeral) by design: MCP URL allowlists
(e.g. Claude Code's ``allowedMcpServers``) match on URL patterns that must be
written before the server exists, so one glob like ``http://127.0.0.1:8787/*``
can cover every thread for the deployment's lifetime."""

@dataclass(frozen=True)
class ToolServerRegistration:
    """One thread's capability to call the runtime tools."""

    url: str
    """Full MCP endpoint for this thread, token embedded in the path."""

    token: str
    """The bare secret, for transports that pass it out of band (e.g. Codex's
    ``bearer_token_env_var``). Required in the ``Authorization: Bearer`` header
    on every request."""

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
        port: int = ...,
        fallback_to_ephemeral: bool = True,
    ) -> None:
        """Configure the server; nothing binds until :meth:`start`.

        Args:
            host: Interface to bind. The default serves local agent
                subprocesses only; binding wider is the caller's decision and
                should come with real network auth in front.
            port: Fixed port to bind, :data:`DEFAULT_PORT` by default — see
                its docstring for why fixed. ``0`` requests an ephemeral port
                outright.
            fallback_to_ephemeral: When the fixed port is taken, bind an
                ephemeral one instead of failing. The fallback logs a warning
                because a URL allowlist written for the fixed port will not
                match it.
        """
        ...

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
        ...

    async def stop(self) -> None:
        """Stop serving and revoke every registration.

        Concurrency:
            Idempotent; stopping a never-started server is a no-op.
        """
        ...

    @property
    def base_url(self) -> str:
        """Root URL of the running server (no token).

        Raises:
            RuntimeError: The server is not started.
        """
        ...

    @property
    def port(self) -> int | None:
        """The bound port while running, else ``None``."""
        ...

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
        ...

    def deregister(self, thread_id: ThreadId) -> None:
        """Revoke ``thread_id``'s token; requests with it fail immediately.

        Concurrency:
            Idempotent; deregistering an unknown thread is a no-op.
        """
        ...
