"""``CoordinatorToolServer`` — HTTP MCP access to the runtime tools.

Hermetic: a stub coordinator and the ``mcp`` Python client, no agent binary
and no model. Pins the contract foreign runtimes rely on: tool schemas match
the shared core's declaration, per-thread tokens isolate identity, auth is
required in both the path and the header, and revocation is immediate.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from ai_functions.runtime.coordinator_tools_core import SEND_MESSAGE_INPUT_SCHEMA  # noqa: E402
from ai_functions.runtime.tool_server import CoordinatorToolServer  # noqa: E402
from ai_functions.types import InputShape, ThreadId, ThreadInfo, ThreadStatus, WorkerId  # noqa: E402


def _info(thread_id: str, name: str) -> ThreadInfo:
    return ThreadInfo(
        thread_id=ThreadId(thread_id),
        worker_id=WorkerId("w-1"),
        thread_name=name,
        input_shape=InputShape.STR_PROMPT,
        status=ThreadStatus.IDLE,
    )


class _Handle:
    """Peer handle whose run() resolves immediately with a canned reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def run(self, message: str) -> asyncio.Future[str]:
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        fut.set_result(f"{self._reply}:{message}")
        return fut


class _StubCoordinator:
    """Just enough Coordinator for the two tools: registry + handles."""

    def __init__(self) -> None:
        self.infos = {
            ThreadId("t-alice"): _info("t-alice", "alice"),
            ThreadId("t-bob"): _info("t-bob", "bob"),
        }

    async def list_threads(self) -> list[ThreadInfo]:
        return list(self.infos.values())

    async def get_thread_info(self, thread_id: ThreadId) -> ThreadInfo:
        return self.infos[thread_id]

    def get_handle(self, thread_id: ThreadId) -> _Handle:
        return _Handle("pong")


def _http(token: str) -> httpx.AsyncClient:
    """An httpx client carrying the bearer header on every request."""
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})


async def _call(url: str, token: str, tool: str, args: dict[str, Any]) -> str:  # pyright: ignore[reportExplicitAny]
    """Open a session against ``url`` and invoke one tool."""
    async with streamable_http_client(url, http_client=_http(token)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            block = result.content[0]
            assert block.type == "text"
            return block.text


@pytest.fixture
async def server():  # noqa: ANN201 - pytest fixture
    srv = CoordinatorToolServer(port=0)
    await srv.start()
    yield srv
    await srv.stop()


async def test_tools_and_schema_match_the_shared_core(server: CoordinatorToolServer) -> None:
    """The wire schema is the shared declaration: enum, default, required."""
    reg = server.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    async with streamable_http_client(reg.url, http_client=_http(reg.token)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name: t for t in (await session.list_tools()).tools}

    assert set(tools) == {"list_threads", "send_message"}
    schema = tools["send_message"].inputSchema
    core = SEND_MESSAGE_INPUT_SCHEMA["properties"]
    assert schema["properties"]["mode"]["enum"] == core["mode"]["enum"]
    assert schema["properties"]["mode"]["default"] == core["mode"]["default"]
    assert schema["required"] == SEND_MESSAGE_INPUT_SCHEMA["required"]


async def test_each_registration_acts_as_its_own_thread(server: CoordinatorToolServer) -> None:
    """Two tokens on one server resolve to distinct calling identities."""
    coord = _StubCoordinator()
    alice = server.register(coord, ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    bob = server.register(coord, ThreadId("t-bob"))  # pyright: ignore[reportArgumentType]

    import json

    as_alice = json.loads(await _call(alice.url, alice.token, "list_threads", {}))
    as_bob = json.loads(await _call(bob.url, bob.token, "list_threads", {}))
    self_of = lambda payload: next(t["thread_name"] for t in payload["threads"] if t["is_self"])  # noqa: E731
    assert self_of(as_alice) == "alice"
    assert self_of(as_bob) == "bob"


async def test_send_message_dispatches_to_the_peer(server: CoordinatorToolServer) -> None:
    """A wait-mode send returns the peer's reply through the tool result."""
    reg = server.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    reply = await _call(reg.url, reg.token, "send_message", {"thread_id": "t-bob", "message": "ping"})
    assert "pong:ping" in reply


async def test_tools_work_with_no_cycle_in_flight(server: CoordinatorToolServer) -> None:
    """Identity comes from the token, not a live cycle context.

    The in-process SDK transport answers ``error: no active cycle`` between
    cycles; over HTTP an idle thread's peers can still discover and message it.
    """
    reg = server.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    # No worker, no dispatcher, no ThreadContext anywhere in this test.
    text = await _call(reg.url, reg.token, "list_threads", {})
    assert '"is_self":true' in text.replace(" ", "")


async def test_header_token_is_required(server: CoordinatorToolServer) -> None:
    """The path token alone is not enough; the bearer header must match it."""
    coord = _StubCoordinator()
    alice = server.register(coord, ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    bob = server.register(coord, ThreadId("t-bob"))  # pyright: ignore[reportArgumentType]

    async with httpx.AsyncClient() as client:
        # No Authorization header at all.
        r = await client.post(alice.url, json={})
        assert r.status_code == 401
        # A *valid* token for a different thread in the header.
        r = await client.post(bob.url, json={}, headers={"Authorization": f"Bearer {alice.token}"})
        assert r.status_code == 401


async def test_unknown_token_is_not_found(server: CoordinatorToolServer) -> None:
    """A fabricated path token is rejected before any tool logic runs."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{server.base_url}/mcp/not-a-real-token",
            json={},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert r.status_code == 404


async def test_deregister_revokes_immediately(server: CoordinatorToolServer) -> None:
    """A revoked token stops working without a server restart."""
    reg = server.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    assert "pong" in await _call(reg.url, reg.token, "send_message", {"thread_id": "t-bob", "message": "x"})
    server.deregister(ThreadId("t-alice"))
    async with httpx.AsyncClient() as client:
        r = await client.post(reg.url, json={}, headers={"Authorization": f"Bearer {reg.token}"})
        assert r.status_code == 404


async def test_reregister_rotates_the_token(server: CoordinatorToolServer) -> None:
    """Registering the same thread again revokes the old token."""
    coord = _StubCoordinator()
    first = server.register(coord, ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    second = server.register(coord, ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    assert first.token != second.token
    async with httpx.AsyncClient() as client:
        r = await client.post(first.url, json={}, headers={"Authorization": f"Bearer {first.token}"})
        assert r.status_code == 404
    assert "pong" in await _call(second.url, second.token, "send_message", {"thread_id": "t-bob", "message": "x"})


async def test_unrecognized_host_is_rejected(server: CoordinatorToolServer) -> None:
    """A Host header naming a foreign origin is refused (DNS-rebinding defense)."""
    reg = server.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]
    async with httpx.AsyncClient() as client:
        r = await client.post(
            reg.url,
            json={},
            headers={"Authorization": f"Bearer {reg.token}", "Host": "evil.example.com"},
        )
        assert r.status_code == 403


async def test_fixed_port_falls_back_to_ephemeral_when_taken() -> None:
    """A taken fixed port falls back to an ephemeral bind with a warning."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]
    try:
        srv = CoordinatorToolServer(port=taken_port, fallback_to_ephemeral=True)
        await srv.start()
        try:
            assert srv.port is not None
            assert srv.port != taken_port
        finally:
            await srv.stop()
    finally:
        blocker.close()


async def test_fixed_port_without_fallback_raises() -> None:
    """fallback_to_ephemeral=False turns a taken port into a hard error."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]
    try:
        srv = CoordinatorToolServer(port=taken_port, fallback_to_ephemeral=False)
        with pytest.raises(OSError):
            await srv.start()
    finally:
        blocker.close()


async def test_register_before_start_raises() -> None:
    """A registration needs a bound port to mint a URL."""
    srv = CoordinatorToolServer(port=0)
    with pytest.raises(RuntimeError):
        srv.register(_StubCoordinator(), ThreadId("t-alice"))  # pyright: ignore[reportArgumentType]


async def test_stop_is_idempotent() -> None:
    """Stopping twice (or before starting) is a no-op."""
    srv = CoordinatorToolServer(port=0)
    await srv.stop()
    await srv.start()
    await srv.stop()
    await srv.stop()
