"""``CodexAgent`` / ``CodexAgentThread`` — the OpenAI Codex backend.

Two tiers. Unit tests feed synthetic SDK notification models straight into
the mapping and pin the Codex-to-event table without any subprocess.
Integration tests run the real ``codex app-server`` binary (bundled with the
``openai-codex`` dependency) against a mocked model backend
(``codex_mock_model``), so turn structure, notification framing, and token
accounting come from the real protocol implementation while staying hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("openai_codex")

from codex_mock_model import MockModelServer, assistant_turn, codex_test_config  # noqa: E402
from openai_codex.client import CodexConfig  # noqa: E402
from openai_codex.generated.v2_all import (  # noqa: E402
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    McpToolCallStatus,
    McpToolCallThreadItem,
    MessagePhase,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
)
from openai_codex.models import Notification, UnknownNotification  # noqa: E402

from ai_functions.codex.codex import (  # noqa: E402
    _RUNTIME_TOKEN_ENV,
    CodexAgent,
    CodexAgentThread,
    _config_with_runtime_tools,
    _TurnState,
)
from ai_functions.runtime import InMemoryCoordinator, LocalWorker  # noqa: E402
from ai_functions.runtime.tool_server import ToolServerRegistration  # noqa: E402
from ai_functions.types import Event, ThreadContext, ThreadId  # noqa: E402
from ai_functions.types.events import EventKind  # noqa: E402


class _NeverPausedCoordinator:
    """Just enough Coordinator for execute()'s cycle-boundary pause check."""

    async def wait_until_unpaused(self, thread_id: ThreadId) -> None:
        return None


class _Recorder:
    """A ThreadContext whose event sink records everything."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def ctx(self) -> ThreadContext:
        return ThreadContext(
            thread_id=ThreadId("t-codex"),
            coordinator=_NeverPausedCoordinator(),  # type: ignore[arg-type]  # structural stub
            on_event=self.events.append,
            on_interrupt=None,  # type: ignore[arg-type]  # mapping never touches it
            pause_signal=asyncio.Event(),
            cancel_signal=asyncio.Event(),
        )

    def kinds(self) -> list[object]:
        return [e.kind for e in self.events]


def _thread() -> CodexAgentThread:
    return CodexAgentThread(CodexAgent())


def _breakdown(input_tokens: int, cached: int, output: int) -> TokenUsageBreakdown:
    return TokenUsageBreakdown(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output,
        reasoning_output_tokens=0,
        total_tokens=input_tokens + output,
    )


def _usage_notification(turn_id: str, input_tokens: int, cached: int, output: int) -> Notification:
    payload = ThreadTokenUsageUpdatedNotification(
        thread_id="th-1",
        turn_id=turn_id,
        token_usage=ThreadTokenUsage(
            last=_breakdown(input_tokens, cached, output),
            total=_breakdown(input_tokens, cached, output),
        ),
    )
    return Notification(method="thread/tokenUsage/updated", payload=payload)


# ── Unit tier: the mapping table ─────────────────────────────────────────────


async def test_agent_message_spans_start_token_complete() -> None:
    """An agent message maps to START on started, TOKEN per delta, COMPLETE on completed."""
    rec = _Recorder()
    ctx = rec.ctx()
    thread = _thread()
    state = _TurnState(turn_id="turn-1")

    item = AgentMessageThreadItem(id="msg-1", text="", type="agentMessage", phase=MessagePhase.final_answer)
    thread._emit_item_started(item, ctx)  # pyright: ignore[reportPrivateUsage]
    thread._emit_notification(  # pyright: ignore[reportPrivateUsage]
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(item_id="msg-1", thread_id="th-1", turn_id="turn-1", delta="hel"),
        ),
        ctx,
        state,
    )
    done = AgentMessageThreadItem(id="msg-1", text="hello", type="agentMessage", phase=MessagePhase.final_answer)
    thread._emit_item_completed(done, ctx, state)  # pyright: ignore[reportPrivateUsage]

    assert rec.kinds() == [
        EventKind.MESSAGE_ASSISTANT_START,
        EventKind.MESSAGE_ASSISTANT_TOKEN,
        EventKind.MESSAGE_ASSISTANT_COMPLETE,
    ]
    assert state.final_answer == "hello"
    # All three events share the item's id as the message id.
    ids = {getattr(e, "message_id", None) for e in rec.events}
    assert ids == {"msg-1"}


async def test_command_execution_maps_to_tool_call_and_result() -> None:
    """A command item is a TOOL_CALL on start and a TOOL_RESULT on completion."""
    rec = _Recorder()
    ctx = rec.ctx()
    thread = _thread()
    state = _TurnState(turn_id="turn-1")

    running = CommandExecutionThreadItem(
        id="call-1",
        type="commandExecution",
        command="echo hi",
        cwd="/tmp",
        command_actions=[],
        status=CommandExecutionStatus.in_progress,
    )
    thread._emit_item_started(running, ctx)  # pyright: ignore[reportPrivateUsage]
    done = CommandExecutionThreadItem(
        id="call-1",
        type="commandExecution",
        command="echo hi",
        cwd="/tmp",
        command_actions=[],
        status=CommandExecutionStatus.completed,
        exit_code=0,
        aggregated_output="hi\n",
    )
    thread._emit_item_completed(done, ctx, state)  # pyright: ignore[reportPrivateUsage]

    call, result = rec.events
    assert call.kind == EventKind.TOOL_CALL
    assert call.tool_name == "command_execution"
    assert call.arguments["command"] == "echo hi"
    assert result.kind == EventKind.TOOL_RESULT
    assert result.status == "success"
    assert result.content == [{"text": "hi\n"}]
    assert call.tool_use_id == result.tool_use_id == "call-1"


async def test_failed_command_reports_error_status() -> None:
    """Non-zero exit code or failed/declined status becomes status='error'."""
    rec = _Recorder()
    thread = _thread()
    state = _TurnState(turn_id="turn-1")
    failed = CommandExecutionThreadItem(
        id="call-2",
        type="commandExecution",
        command="false",
        cwd="/tmp",
        command_actions=[],
        status=CommandExecutionStatus.completed,
        exit_code=1,
        aggregated_output="",
    )
    thread._emit_item_completed(failed, rec.ctx(), state)  # pyright: ignore[reportPrivateUsage]
    assert rec.events[0].status == "error"


async def test_mcp_tool_call_uses_server_dot_tool_name() -> None:
    """MCP tool items surface as '<server>.<tool>' with their arguments."""
    rec = _Recorder()
    thread = _thread()
    item = McpToolCallThreadItem(
        id="mcp-1",
        type="mcpToolCall",
        server="aif_runtime",
        tool="send_message",
        arguments={"thread_id": "t-2", "message": "hi"},
        status=McpToolCallStatus.in_progress,
    )
    thread._emit_item_started(item, rec.ctx())  # pyright: ignore[reportPrivateUsage]
    call = rec.events[0]
    assert call.tool_name == "aif_runtime.send_message"
    assert call.arguments == {"thread_id": "t-2", "message": "hi"}


async def test_token_usage_accumulates_and_subtracts_cached() -> None:
    """Per-call `last` breakdowns sum; cached tokens move out of input_tokens."""
    rec = _Recorder()
    ctx = rec.ctx()
    thread = _thread()
    state = _TurnState(turn_id="turn-1")

    thread._emit_notification(_usage_notification("turn-1", 100, 40, 10), ctx, state)  # pyright: ignore[reportPrivateUsage]
    thread._emit_notification(_usage_notification("turn-1", 200, 150, 5), ctx, state)  # pyright: ignore[reportPrivateUsage]
    # A different turn's usage must not leak in.
    thread._emit_notification(_usage_notification("turn-OTHER", 999, 0, 999), ctx, state)  # pyright: ignore[reportPrivateUsage]

    usage = state.usage
    assert usage is not None
    assert usage.input_tokens == (100 - 40) + (200 - 150)
    assert usage.cache_read_tokens == 40 + 150
    assert usage.output_tokens == 15
    assert usage.cache_write_tokens == 0
    # Usage is not emitted per notification; the turn emits exactly one at the end.
    assert EventKind.TOKEN_USAGE not in rec.kinds()


async def test_unknown_notification_becomes_custom_event() -> None:
    """Unknown methods degrade to CustomEvent('codex_<method>'), never raise."""
    rec = _Recorder()
    thread = _thread()
    state = _TurnState(turn_id="turn-1")
    thread._emit_notification(  # pyright: ignore[reportPrivateUsage]
        Notification(method="thread/somethingNew", payload=UnknownNotification(params={"x": 1})),
        rec.ctx(),
        state,
    )
    event = rec.events[0]
    assert event.kind == "codex_thread_somethingNew"
    assert event.payload == {"x": 1}


def test_runtime_tools_config_injects_url_and_token() -> None:
    """The launch config gains the MCP server overrides; the token rides the env."""
    reg = ToolServerRegistration(url="http://127.0.0.1:1234/mcp/tok", token="tok")
    base = CodexConfig(config_overrides=('model="o3"',), env={"KEEP": "1"})
    merged = _config_with_runtime_tools(base, reg)

    assert merged.config_overrides == (
        'model="o3"',
        'mcp_servers.ai_functions_runtime.url="http://127.0.0.1:1234/mcp/tok"',
        f'mcp_servers.ai_functions_runtime.bearer_token_env_var="{_RUNTIME_TOKEN_ENV}"',
    )
    assert merged.env == {"KEEP": "1", _RUNTIME_TOKEN_ENV: "tok"}
    # The original config is untouched (dataclasses.replace semantics).
    assert base.config_overrides == ('model="o3"',)
    assert base.env == {"KEEP": "1"}


def test_runtime_tools_config_accepts_none() -> None:
    """A template with no config still gets a wired launch config."""
    reg = ToolServerRegistration(url="http://127.0.0.1:1234/mcp/tok", token="tok")
    merged = _config_with_runtime_tools(None, reg)
    assert any(o.startswith("mcp_servers.ai_functions_runtime.url=") for o in merged.config_overrides)
    assert merged.env == {_RUNTIME_TOKEN_ENV: "tok"}


def test_runtime_tools_config_rejects_reserved_key() -> None:
    """User overrides may not squat on the reserved MCP server key."""
    reg = ToolServerRegistration(url="http://127.0.0.1:1234/mcp/tok", token="tok")
    base = CodexConfig(config_overrides=('mcp_servers.ai_functions_runtime.url="http://evil"',))
    with pytest.raises(ValueError, match="reserved"):
        _ = _config_with_runtime_tools(base, reg)


async def test_notify_while_idle_buffers() -> None:
    """With no turn in flight, notify() parks the message for the next turn."""
    thread = _thread()
    await thread.notify("a note")
    assert thread._inject_buffer == ["a note"]  # pyright: ignore[reportPrivateUsage]


async def test_fork_of_unconnected_thread_returns_template() -> None:
    """A never-connected thread's entire state is its template."""
    template = CodexAgent(name="forky")
    thread = CodexAgentThread(template)
    assert await thread.fork() is template


async def test_serialize_round_trip_is_identity() -> None:
    thread = _thread()
    assert thread.serialize_result("x") == "x"
    assert thread.deserialize_result("x") == "x"


async def test_teardown_never_connected_is_noop() -> None:
    thread = _thread()
    await thread.teardown()
    assert not thread.is_connected


async def test_interrupt_watcher_fires_on_cancel_signal() -> None:
    """The watcher calls handle.interrupt() when the cycle's cancel signal sets."""

    class _Handle:
        def __init__(self) -> None:
            self.interrupted = asyncio.Event()

        async def interrupt(self) -> None:
            self.interrupted.set()

    rec = _Recorder()
    ctx = rec.ctx()
    thread = _thread()
    handle = _Handle()
    watcher = asyncio.create_task(thread._interrupt_on_cancel(ctx, handle))  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    ctx.cancel_signal.set()
    await asyncio.wait_for(handle.interrupted.wait(), timeout=2)
    await watcher


# ── Integration tier: real app-server, mocked model ──────────────────────────


@pytest.mark.integration
async def test_execute_returns_answer_and_shadows_events(tmp_path: Path) -> None:
    """One cycle end-to-end: result text, message events, one TOKEN_USAGE."""
    with MockModelServer() as model:
        model.enqueue(assistant_turn("the answer is 4", input_tokens=100, cached_input_tokens=40, output_tokens=7))
        template = CodexAgent(config=codex_test_config(tmp_path, model.url))
        coord = InMemoryCoordinator()
        worker = await LocalWorker(coord).register()
        handle = await worker.spawn_locally(template, thread_name="codex")
        try:
            result = await asyncio.wait_for(handle.run("what is 2+2?"), timeout=120)
            assert result == "the answer is 4"

            events = await coord.get_events(handle.id)
            kinds = [e.kind for e in events]
            assert EventKind.MESSAGE_USER in kinds
            assert EventKind.MESSAGE_ASSISTANT_COMPLETE in kinds
            usage_events = [e for e in events if e.kind == EventKind.TOKEN_USAGE]
            assert len(usage_events) == 1
            usage = usage_events[0].token_usage
            assert usage.input_tokens == 60  # 100 raw minus 40 cached
            assert usage.cache_read_tokens == 40
            assert usage.output_tokens == 7
        finally:
            await handle.terminate_now()
            await worker.close()


@pytest.mark.integration
async def test_post_condition_failure_rides_next_turn(tmp_path: Path) -> None:
    """A failing post-condition feeds back as the next user turn, prompt sent once."""

    def must_mention_four(result: str):  # noqa: ANN202
        from ai_functions.ai_thread.postcondition import PostConditionResult

        if "4" in result:
            return PostConditionResult(passed=True)
        return PostConditionResult(passed=False, message="the answer must contain the digit 4")

    with MockModelServer() as model:
        model.enqueue(assistant_turn("I refuse to answer", response_id="r1"))
        model.enqueue(assistant_turn("fine, it is 4", response_id="r2"))
        template = CodexAgent(
            config=codex_test_config(tmp_path, model.url),
            post_conditions=(must_mention_four,),
            max_attempts=3,
        )
        coord = InMemoryCoordinator()
        worker = await LocalWorker(coord).register()
        handle = await worker.spawn_locally(template, thread_name="codex")
        try:
            result = await asyncio.wait_for(handle.run("what is 2+2?"), timeout=120)
            assert result == "fine, it is 4"

            assert len(model.requests) == 2
            first_turn_texts = " ".join(model.user_texts(0))
            retry_texts = " ".join(model.user_texts(1))
            assert "what is 2+2?" in first_turn_texts
            assert "Post-condition failures" in retry_texts
            assert "must contain the digit 4" in retry_texts
        finally:
            await handle.terminate_now()
            await worker.close()


@pytest.mark.integration
async def test_fork_resumes_a_distinct_codex_thread(tmp_path: Path) -> None:
    """fork() returns a template resuming a server-side fork of the transcript."""
    with MockModelServer() as model:
        model.enqueue(assistant_turn("first reply", response_id="r1"))
        template = CodexAgent(config=codex_test_config(tmp_path, model.url))
        thread = template.to_thread()
        rec = _Recorder()
        try:
            _ = await asyncio.wait_for(thread.execute(rec.ctx(), "hello"), timeout=120)
            source_id = thread.codex_thread_id
            assert source_id is not None

            forked = await thread.fork()
            assert isinstance(forked, CodexAgent)
            assert forked.resume_thread_id is not None
            assert forked.resume_thread_id != source_id
        finally:
            await thread.teardown()


@pytest.mark.integration
async def test_runtime_tools_reach_the_model(tmp_path: Path) -> None:
    """The app-server connects to this thread's tool server and offers the
    runtime tools to the model; teardown stops the server."""
    with MockModelServer() as model:
        model.enqueue(assistant_turn("ok", response_id="r1"))
        template = CodexAgent(config=codex_test_config(tmp_path, model.url))
        thread = template.to_thread()
        rec = _Recorder()
        try:
            _ = await asyncio.wait_for(thread.execute(rec.ctx(), "hello"), timeout=120)
            # The recorded model request proves the whole chain: the app-server
            # resolved our config overrides, fetched the tool list over HTTP
            # MCP (bearer token and all), and exposed it to the model.
            tool_names = [t.get("name") for t in model.requests[0].get("tools", [])]
            assert "mcp__ai_functions_runtime" in tool_names
        finally:
            await thread.teardown()
        assert thread._tool_server is None  # pyright: ignore[reportPrivateUsage]
