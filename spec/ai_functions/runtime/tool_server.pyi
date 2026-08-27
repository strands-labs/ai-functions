"""``CoordinatorToolServer`` — the runtime tools over streamable-HTTP MCP.

Serves the two coordinator tools from
:mod:`ai_functions.runtime.coordinator_tools_core` (``list_threads`` /
``send_message``) over HTTP, so agent runtimes that take an ``mcp_servers`` URL
— Codex among them — can join a team without per-runtime bridge code. Runs in
the process owning the coordinator.

Clients must send the token in the ``Authorization: Bearer`` header.

One server hosts any number of threads; each registration mints its own
capability URL of the form ``http://127.0.0.1:<port>/mcp/<routing-id>``.

The server binds 127.0.0.1 on a system-assigned port; pass the registration URL
to the runtime as it starts. The port is reachable by any local process, so the
*token* is the boundary: each registration mints a ``secrets``-grade token
required in the ``Authorization: Bearer`` header (constant-time compared), the
``Host`` header must name the bound address (DNS-rebinding defense per the MCP
spec's guidance for local HTTP servers), and deregistration revokes the token
immediately.

The MCP app runs stateless (``stateless_http=True``): tools and one-shot
reads only — no server-initiated messages, no resource subscriptions.

Requires the ``runtime-tools`` extra (``mcp``, ``uvicorn``).
"""

from dataclasses import dataclass
from typing import final

from ..protocols import Coordinator
from ..types import ThreadId

@dataclass(frozen=True)
class ToolServerRegistration:
    """One thread's capability to call the runtime tools."""

    url: str
    """Full MCP endpoint for this thread."""

    token: str
    """The bare secret, required in the ``Authorization: Bearer`` header on
    every request. Codex takes it via ``bearer_token_env_var``."""

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
        ...

    async def start(self) -> None:
        """Bind the socket and serve until :meth:`stop`.

        Ensures:
            - :attr:`base_url` is valid once this returns.
            - Requests are answered (all-404 until a thread registers).

        Raises:
            TimeoutError: Serving did not come up within the startup timeout.

        Concurrency:
            Idempotent; a started server ignores further ``start`` calls.
        """
        ...

    async def stop(self) -> None:
        """Stop serving and revoke every registration.

        In-flight tool calls are dropped rather than drained: a client waiting
        on one (``send_message(mode="wait")``, say) sees the connection close
        mid-response.

        Concurrency:
            Idempotent; stopping a never-started server is a no-op.
        """
        ...

    @property
    def base_url(self) -> str:
        """Root URL of the running server (no registration path).

        Raises:
            RuntimeError: The server is not started.
        """
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
            The registration; hand ``url`` and ``token`` to the agent runtime's
            MCP config.

        Raises:
            RuntimeError: The server is not started.
        """
        ...

    def deregister(self, thread_id: ThreadId) -> None:
        """Revoke ``thread_id``'s registration; later requests to its URL 404.

        A request already dispatched keeps the registration it resolved for the
        rest of its lifetime; revocation is not a cancellation.

        Concurrency:
            Idempotent; deregistering an unknown thread is a no-op.
        """
        ...
