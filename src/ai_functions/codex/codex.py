"""``CodexAgent`` template and ``CodexAgentThread`` — OpenAI Codex-backed thread.

A ``CodexAgentThread`` drives a ``codex app-server`` subprocess through the
OpenAI Codex Python SDK (``openai_codex``). The app-server owns the
conversation transcript; ai_functions observes its notification stream and
re-emits each element through ``Coordinator.append_event`` as an observability
shadow. I7/I9 do not apply (the app-server, not the runtime, owns history);
the thread drains injected messages at work boundaries, supports pause, and
delivers ``notify`` mid-turn by steering the in-flight turn.

Codex-to-event mapping
──────────────────────

Every notification on ``AsyncTurnHandle.stream()`` maps to a fixed set of
ai_functions events. No new built-in event kinds are introduced. Lifecycle
events (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``, ``RESULT``) are
emitted by the runtime dispatcher, never by the thread.

- ``UserMessageThreadItem``: ignored (the thread already emitted
  ``MESSAGE_USER`` when it sent the turn, and again when it steered).
- ``AgentMessageThreadItem``: ``MESSAGE_ASSISTANT_START`` on ``item/started``
  and ``MESSAGE_ASSISTANT_COMPLETE`` on ``item/completed``, both keyed by
  ``MessageId(item.id)``; ``item/agentMessage/delta`` becomes
  ``MESSAGE_ASSISTANT_TOKEN`` (``complete=False``).
- ``ReasoningThreadItem`` on ``item/completed``: one
  ``MESSAGE_ASSISTANT_THINKING`` (``complete=True``) per summary/content
  entry; ``item/reasoning/textDelta`` and ``item/reasoning/summaryTextDelta``
  become ``MESSAGE_ASSISTANT_THINKING`` (``complete=False``).
- ``CommandExecutionThreadItem``, ``FileChangeThreadItem``,
  ``McpToolCallThreadItem``, ``DynamicToolCallThreadItem``,
  ``WebSearchThreadItem``: ``TOOL_CALL`` on ``item/started`` and
  ``TOOL_RESULT`` on ``item/completed``, with ``tool_use_id=item.id``.
  Command executions report ``status="error"`` for failed/declined runs or a
  non-zero exit code; MCP and dynamic tool calls follow their own status
  fields.
- ``thread/tokenUsage/updated``: accumulated per turn from the ``last``
  breakdown; exactly one ``TOKEN_USAGE`` is emitted per turn. Codex counts
  cached tokens *inside* ``inputTokens``, so the event reports
  ``input_tokens = inputTokens - cachedInputTokens`` and
  ``cache_read_tokens = cachedInputTokens``, preserving the
  ``input + cache_read + cache_write`` total identity.
- ``error`` (``ErrorNotification``): ``CustomEvent(kind="codex_error")``.
- ``turn/plan/updated`` / ``item/plan/delta`` / ``PlanThreadItem``:
  ``CustomEvent(kind="codex_plan", payload=...)``.
- Every other thread item on ``item/completed``:
  ``CustomEvent(kind=f"codex_item_{item.type}", payload=...)`` — the item union
  grows with the CLI, so unmapped variants degrade to observability rather than
  being dropped.
- ``turn/completed``: terminal. ``failed`` raises ``AIFunctionError`` with
  the turn error's message; ``interrupted`` raises
  ``asyncio.CancelledError``; ``completed`` yields the turn result.
- Every other notification: ``CustomEvent(kind=f"codex_{method}")`` with
  ``/`` replaced by ``_`` — the registry is large and versions with the CLI,
  so unknown methods degrade to observability rather than errors.

The turn's string result is the last ``final_answer``-phase agent message,
falling back to the last message with no phase (mirrors the SDK's own
``TurnResult`` collection). ``turn/completed`` carries ``items_view:
not_loaded`` with an empty ``items`` list, so items are accumulated from the
stream, never read from the completion payload.

Runtime tools
─────────────

Codex reaches the runtime tools (``list_threads`` / ``send_message``) over
HTTP MCP: each thread starts its own :class:`CoordinatorToolServer` when it
connects, registers itself for a capability URL, and injects the endpoint
into the app-server launch via ``--config`` overrides
(``mcp_servers.ai_functions_runtime.url`` plus a ``bearer_token_env_var``
naming an env var that carries the token). The server lives exactly as long
as the connection: ``teardown`` stops it after closing the SDK client.

Invariants:
    I2 — every emitted event goes through ``Coordinator.append_event``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final, override

from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.tools import AgentTool

from ..ai_thread.errors import AIFunctionError
from ..ai_thread.postcondition import PostCondition, run_post_condition_loop
from ..protocols import Spawnable, Thread
from ..runtime.tool_server import CoordinatorToolServer, ToolServerRegistration
from ..types import (
    CustomEvent,
    InputShape,
    MessageAssistantCompleteEvent,
    MessageAssistantStartEvent,
    MessageAssistantThinkingEvent,
    MessageAssistantTokenEvent,
    MessageId,
    MessageUserEvent,
    ThreadContext,
    TokenUsage,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)

try:
    from openai_codex import ApprovalMode, AsyncCodex, Sandbox
    from openai_codex.client import CodexConfig
    from openai_codex.generated.v2_all import (
        AgentMessageThreadItem,
        CommandExecutionThreadItem,
        DynamicToolCallThreadItem,
        FileChangeThreadItem,
        McpToolCallThreadItem,
        MessagePhase,
        PlanThreadItem,
        ReasoningThreadItem,
        TurnStatus,
        UserMessageThreadItem,
        WebSearchThreadItem,
    )

    # The notification types have a curated home in ``openai_codex.models``;
    # only the thread-item classes and enums must come from the generated module.
    from openai_codex.models import (
        AgentMessageDeltaNotification,
        ErrorNotification,
        ItemCompletedNotification,
        ItemStartedNotification,
        Notification,
        ReasoningSummaryTextDeltaNotification,
        ReasoningTextDeltaNotification,
        ThreadTokenUsageUpdatedNotification,
        TurnCompletedNotification,
        UnknownNotification,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "CodexAgent requires the optional 'codex' extra (the OpenAI Codex "
        "SDK and its bundled runtime). Install it with:\n"
        "    pip install 'strands-ai-functions[codex]'",
    ) from exc

if TYPE_CHECKING:
    from openai_codex import AsyncThread, AsyncTurnHandle
    from openai_codex.generated.v2_all import (
        Personality,
        ReasoningEffort,
        ReasoningSummary,
        ThreadTokenUsage,
    )

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_RUNTIME_SERVER_KEY = "ai_functions_runtime"
"""Key of the runtime-tools MCP server in the Codex config; tool calls surface
as ``TOOL_CALL`` events named ``ai_functions_runtime.<tool>``."""

_RUNTIME_TOKEN_ENV = "AI_FUNCTIONS_RUNTIME_TOKEN"
"""Env var (in the app-server subprocess only) carrying the bearer token."""


def _config_with_runtime_tools(config: CodexConfig | None, reg: ToolServerRegistration) -> CodexConfig:
    """Return a copy of ``config`` pointing Codex at the runtime tool server.

    Adds ``--config`` overrides registering ``reg.url`` as a streamable-HTTP
    MCP server and routes its bearer token through the subprocess environment
    (``bearer_token_env_var``), so the secret never appears on the command
    line. The caller's own overrides and env are preserved.

    Raises:
        ValueError: ``config.config_overrides`` already configures the
            reserved ``mcp_servers.ai_functions_runtime`` key.
    """
    base = config or CodexConfig()
    reserved = f"mcp_servers.{_RUNTIME_SERVER_KEY}"
    if any(override.lstrip().startswith(reserved) for override in base.config_overrides):
        raise ValueError(
            f"CodexConfig.config_overrides may not configure the reserved key "
            f"{reserved!r}; ai_functions uses it for the runtime MCP server.",
        )
    overrides = (
        f'{reserved}.url="{reg.url}"',
        f'{reserved}.bearer_token_env_var="{_RUNTIME_TOKEN_ENV}"',
    )
    env = dict(base.env or {})
    env[_RUNTIME_TOKEN_ENV] = reg.token
    return dataclasses.replace(base, config_overrides=(*base.config_overrides, *overrides), env=env)


_MAPPED_ITEM_TYPES = frozenset(
    {
        "userMessage",
        "agentMessage",
        "reasoning",
        "plan",
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "webSearch",
    },
)
"""Wire discriminators of the thread items this adapter maps to typed events.
Items outside this set are re-emitted as ``codex_item_*`` CustomEvents."""


def _item_root(item: object) -> object:
    """Unwrap the generated ``ThreadItem`` RootModel to its concrete variant."""
    return getattr(item, "root", item)


def _item_type(item: object) -> str:
    """Wire discriminator of a thread item (``agentMessage``, ``webSearch``, …)."""
    value = getattr(item, "type", None)
    return value if isinstance(value, str) else "unknown"


def _status_value(status: object) -> str:
    """Render a generated status enum (or anything else) as its wire string."""
    value = getattr(status, "value", status)
    return value if isinstance(value, str) else str(value)


def _payload_dict(payload: object) -> dict[str, object]:
    """Render a notification payload as a JSON-friendly dict for CustomEvent."""
    if isinstance(payload, UnknownNotification):
        return dict(payload.params)
    if isinstance(payload, BaseModel):
        dumped = payload.model_dump(mode="json", by_alias=True)
        return dumped if isinstance(dumped, dict) else {"payload": dumped}
    return {"payload": repr(payload)}


def _tool_result_text(*parts: object) -> list[dict[str, str]]:
    """Pack tool output fragments into Strands-shape text content blocks."""
    texts = [str(p) for p in parts if p is not None and str(p) != ""]
    return [{"text": t} for t in texts] if texts else []


@dataclass
class _TurnState:
    """Mutable per-turn accumulator for stream consumption."""

    turn_id: str
    final_answer: str | None = None
    last_unphased: str | None = None
    usage: TokenUsage | None = None
    completed: TurnCompletedNotification | None = None


@final
@dataclass(frozen=True)
class CodexAgent(Spawnable[[str], str], ToolProvider):
    """Immutable template for an OpenAI-Codex-backed thread.

    Carries the configuration used to launch the ``codex app-server``
    subprocess and start (or resume) its conversation thread, plus the
    display metadata needed to expose the resulting thread as a Strands
    tool. Picklable and safe to share across runtimes: session identity is
    a string (``resume_thread_id``), so a template can travel to any host
    that has Codex auth and an equivalent working directory.

    Implements:
        Spawnable, strands.tools.ToolProvider.

    Immutable: Yes.
    """

    config: CodexConfig | None = None
    """Launch configuration forwarded to the SDK client (binary path, cwd,
    env, config overrides); ``None`` uses the bundled runtime's defaults."""

    model: str | None = None
    """Model id for the conversation thread; ``None`` uses the Codex default."""

    sandbox: Sandbox | None = None
    """Filesystem access preset (``read_only`` / ``workspace_write`` /
    ``full_access``); ``None`` uses the Codex default."""

    approval_mode: ApprovalMode = ApprovalMode.auto_review
    """How escalated permission requests are resolved. ``auto_review`` (the
    default) lets Codex's own reviewer arbitrate; ``deny_all`` refuses them.
    The public SDK exposes no human-in-the-loop callback, so approvals are
    not routed through ``ctx.on_interrupt``."""

    effort: ReasoningEffort | None = None
    """Per-turn reasoning effort override; ``None`` uses the model default."""

    summary: ReasoningSummary | None = None
    """Per-turn reasoning summary override; ``None`` uses the model default."""

    output_schema: dict[str, object] | None = None
    """JSON Schema constraining each turn's final assistant message."""

    personality: Personality | None = None
    """Assistant personality; ``None`` uses the Codex default."""

    base_instructions: str | None = None
    """Replacement base instructions for the thread."""

    developer_instructions: str | None = None
    """Additional developer instructions for the thread."""

    cwd: str | None = None
    """Working directory for the conversation thread; ``None`` uses the
    app-server process's own working directory."""

    resume_thread_id: str | None = None
    """Resume this stored Codex thread instead of starting a fresh one.
    ``fork()`` returns templates carrying this field."""

    name: str = "codex"
    """Name used for telemetry and when exposed as a Strands tool."""

    description: str = "Send a prompt to a Codex agent and receive its final answer."
    """Description used when exposed as a Strands tool."""

    post_conditions: tuple[PostCondition, ...] = ()
    """Validators run against each cycle's result. On failure the thread
    feeds the failure messages back as the next user turn and re-runs, up to
    ``max_attempts``. Empty (default) disables the retry loop — behaviour is
    then a single query."""

    max_attempts: int = 10
    """Maximum number of cycles to satisfy ``post_conditions``. Ignored when
    ``post_conditions`` is empty — the loop short-circuits after the first
    query, so the default single-query behaviour is unchanged unless
    ``post_conditions`` is set."""

    @property
    def input_shape(self) -> InputShape:
        """Every CodexAgent thread accepts a single string prompt."""
        return InputShape.STR_PROMPT

    @override
    def to_thread(self) -> CodexAgentThread:
        """Produce a fresh ``CodexAgentThread`` bound to this template.

        The returned thread owns its own ``AsyncCodex`` client; the
        ``codex app-server`` subprocess is not spawned until the first cycle
        runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        return CodexAgentThread(self)

    @override
    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Expose this template as a Strands tool.

        The returned tool takes one ``prompt: str`` argument; each invocation
        spawns a private ``CodexAgentThread``, runs a single cycle, and tears
        it down.

        Args:
            kwargs: Ignored; present for protocol compatibility.

        Returns:
            A single-element list containing the ``AgentTool``.
        """
        from ..runtime.coordinator import InMemoryCoordinator
        from ..runtime.worker import LocalWorker

        template = self

        @_strands_tool(name=self.name, description=self.description)
        async def _invoke(prompt: str) -> str:
            coord = InMemoryCoordinator()
            worker = LocalWorker(coord)
            handle = await worker.spawn_locally(template)
            try:
                return await handle.run(prompt)
            finally:
                await handle.terminate_now()

        return [_invoke]

    @override
    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        return None

    @override
    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        return None


@final
class CodexAgentThread(Thread[[str], str]):
    """Live Codex-backed thread that owns one ``AsyncCodex`` client.

    Connects lazily on the first cycle: launches ``codex app-server`` and
    starts (or resumes) one conversation thread, both kept for the thread's
    lifetime. The app-server owns conversation history; ai_functions observes
    the notification stream and re-emits each element as a ai_functions
    event — pure observability, not a source of truth. The module docstring
    describes the full mapping.

    Unlike the Claude and Kiro backends, this thread supports:

    - real ``fork()`` — Codex forks the stored conversation server-side;
    - mid-turn ``notify()`` — the message is steered into the in-flight turn;
    - cooperative cancel that interrupts the in-flight turn rather than
      waiting for the cycle boundary.

    Implements:
        Thread[[str], str].

    Lifecycle:
        CREATED → CONNECTED → CLOSED.
    """

    __slots__ = (
        "_template",
        "_codex",
        "_thread",
        "_tool_server",
        "_connected",
        "_connect_lock",
        "_active_ctx",
        "_active_turn",
        "_inject_buffer",
        "_pending_steers",
    )

    def __init__(self, template: CodexAgent) -> None:
        self._template: CodexAgent = template
        self._codex: AsyncCodex | None = None
        self._thread: AsyncThread | None = None
        # Started at connect time so Codex can call the runtime tools over
        # HTTP MCP; lives exactly as long as the connection.
        self._tool_server: CoordinatorToolServer | None = None
        self._connected: bool = False
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        # Populated for the duration of each cycle; the dispatcher serialises
        # cycles, so reads from notify() during a cycle are safe.
        self._active_ctx: ThreadContext | None = None
        # The in-flight turn's handle, for steering and interruption.
        self._active_turn: AsyncTurnHandle | None = None
        # Pending side-channel messages delivered via ``notify`` while idle;
        # prepended to the next outgoing user turn.
        self._inject_buffer: list[str] = []
        # In-flight steer tasks, held so the loop keeps a strong reference and
        # ``teardown`` can cancel them.
        self._pending_steers: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``CodexAgent`` template."""
        return self._template.name

    async def notify(self, text: str) -> None:
        """Deliver ``text`` mid-turn via steering, or buffer it for the next turn.

        With a turn in flight, the message is steered into it as additional
        user input (scheduled as a background task so this call does not
        block; a failed steer falls back to the buffer). Otherwise it sits in
        the inject buffer and is prepended to the next ``execute`` prompt.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - No new cycle is started by this call.
            - A message that cannot be steered is buffered for the next turn,
              unless :meth:`teardown` runs first.
        """
        handle = self._active_turn
        ctx = self._active_ctx
        if handle is None or ctx is None:
            self._inject_buffer.append(text)
            return

        async def _steer() -> None:
            try:
                _ = await handle.steer(text)
            except Exception:
                logger.exception("steering into turn %s failed; buffering for the next turn", handle.id)
                self._inject_buffer.append(text)
                return
            ctx.on_event(MessageUserEvent(text=text))

        task = asyncio.create_task(_steer())
        self._pending_steers.add(task)
        task.add_done_callback(self._pending_steers.discard)

    async def execute(self, ctx: ThreadContext, prompt: str) -> str:
        """Send ``prompt`` to the Codex thread and return its string result.

        Runs the shared post-condition loop: the inject buffer is drained
        into the outgoing turn, failures are fed back as the next user turn,
        and the notification stream of each turn is re-emitted per the
        mapping table in the module docstring.

        Args:
            ctx: Freshly built per-cycle context; never reused across cycles.
            prompt: User prompt forwarded to the Codex thread.

        Returns:
            The turn's final assistant answer, or the empty string if the
            turn produced no text.

        Emits:
            - MESSAGE_USER — per drained inject-buffer entry, per steered
              message, plus one for ``prompt``.
            - MESSAGE_ASSISTANT_START / _TOKEN / _COMPLETE — per agent message.
            - MESSAGE_ASSISTANT_THINKING — per reasoning entry or delta.
            - TOOL_CALL / TOOL_RESULT — per command, file-change, MCP,
              dynamic-tool, or web-search item.
            - TOKEN_USAGE — exactly one per turn.
            - CustomEvent — codex_error / codex_plan / codex_item_* /
              codex_* passthroughs.

        Raises:
            asyncio.CancelledError: ``ctx.cancel_signal`` was set — at the
                cycle boundary, or mid-turn (the turn is interrupted first).
            AIFunctionError: The turn failed, or post-conditions were not
                satisfied within ``max_attempts`` attempts.
        """
        if ctx.cancel_signal.is_set():
            raise asyncio.CancelledError
        await ctx.coordinator.wait_until_unpaused(ctx.thread_id)
        self._active_ctx = ctx
        try:
            await self._ensure_connected(ctx)

            async def _send_turn(combined: str) -> str:
                return await self._run_turn(ctx, combined)

            return await run_post_condition_loop(
                ctx,
                prompt,
                thread_name=self.name,
                post_conditions=self._template.post_conditions,
                max_attempts=self._template.max_attempts,
                inject_buffer=self._inject_buffer,
                send_turn=_send_turn,
            )
        finally:
            self._active_ctx = None

    async def fork(self) -> Spawnable[[str], str]:
        """Fork the stored Codex conversation into a new template.

        Codex forks the transcript server-side; the returned template resumes
        the forked thread, so ``Coordinator.fork`` (which seeds the new
        ai_functions event log from the source's) yields a divergent
        continuation on both sides of the boundary.

        Returns:
            A ``CodexAgent`` carrying ``resume_thread_id`` of the fork — or
            this thread's own template when no session exists yet, since a
            never-connected thread's entire state is its template.
        """
        if self._codex is None or self._thread is None:
            return self._template
        forked = await self._codex.thread_fork(self._thread.id)
        return dataclasses.replace(self._template, resume_thread_id=forked.id)

    async def teardown(self) -> None:
        """Close the SDK client and release the ``codex app-server`` subprocess.

        Ensures:
            - Any running ``AsyncCodex`` client is closed.
            - The runtime tool server is stopped and its token revoked.
            - In-flight steers are cancelled and awaited.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        # Settle the steers before clearing the buffer: a steer failing after
        # the clear would re-append to it on a torn-down thread.
        steers = tuple(self._pending_steers)
        self._pending_steers.clear()
        for task in steers:
            _ = task.cancel()
        if steers:
            _ = await asyncio.gather(*steers, return_exceptions=True)
        self._inject_buffer.clear()
        codex = self._codex
        tool_server = self._tool_server
        self._codex = None
        self._thread = None
        self._tool_server = None
        self._active_turn = None
        self._connected = False
        # Close the client (killing the app-server) before stopping the tool
        # server, so no live Codex process outlasts its tool endpoint.
        if codex is not None:
            await codex.close()
        if tool_server is not None:
            await tool_server.stop()

    def serialize_result(self, result: str) -> str:
        """Return ``result`` unchanged; Codex results are already strings."""
        return result

    def deserialize_result(self, payload: str) -> str:
        """Return ``payload`` unchanged; Codex results are already strings."""
        return payload

    @property
    def template(self) -> CodexAgent:
        """The template this thread was created from."""
        return self._template

    @property
    def is_connected(self) -> bool:
        """Whether the app-server is running with a live conversation thread."""
        return self._connected

    @property
    def codex_thread_id(self) -> str | None:
        """The Codex-side conversation thread id once connected, or ``None``."""
        return None if self._thread is None else self._thread.id

    # ── Internals ──

    async def _ensure_connected(self, ctx: ThreadContext) -> None:
        """Lazily launch the app-server and start (or resume) the thread.

        Also starts this thread's :class:`CoordinatorToolServer` and injects
        its capability URL into the app-server launch config, so the Codex
        agent can call ``list_threads`` / ``send_message`` over HTTP MCP.
        """
        if self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            template = self._template
            tool_server = CoordinatorToolServer()
            await tool_server.start()
            try:
                reg = tool_server.register(ctx.coordinator, ctx.thread_id)
                codex = AsyncCodex(config=_config_with_runtime_tools(template.config, reg))
            except BaseException:
                await tool_server.stop()
                raise
            try:
                if template.resume_thread_id is not None:
                    thread = await codex.thread_resume(
                        template.resume_thread_id,
                        approval_mode=template.approval_mode,
                        base_instructions=template.base_instructions,
                        cwd=template.cwd,
                        developer_instructions=template.developer_instructions,
                        model=template.model,
                        personality=template.personality,
                        sandbox=template.sandbox,
                    )
                else:
                    thread = await codex.thread_start(
                        approval_mode=template.approval_mode,
                        base_instructions=template.base_instructions,
                        cwd=template.cwd,
                        developer_instructions=template.developer_instructions,
                        model=template.model,
                        personality=template.personality,
                        sandbox=template.sandbox,
                    )
            except BaseException:
                await codex.close()
                await tool_server.stop()
                raise
            self._codex = codex
            self._thread = thread
            self._tool_server = tool_server
            self._connected = True

    async def _run_turn(self, ctx: ThreadContext, combined: str) -> str:
        """Start one turn, consume its stream, and return the final answer."""
        assert self._thread is not None
        template = self._template
        handle = await self._thread.turn(
            combined,
            effort=template.effort,
            output_schema=cast("Any", template.output_schema),  # pyright: ignore[reportExplicitAny]
            summary=template.summary,
        )
        state = _TurnState(turn_id=handle.id)
        self._active_turn = handle
        watcher = asyncio.create_task(self._interrupt_on_cancel(ctx, handle))
        try:
            async for notification in handle.stream():
                self._emit_notification(notification, ctx, state)
        finally:
            self._active_turn = None
            _ = watcher.cancel()

        if state.usage is not None:
            ctx.on_event(TokenUsageEvent(token_usage=state.usage))

        completed = state.completed
        if completed is not None:
            status = completed.turn.status
            if status == TurnStatus.interrupted:
                raise asyncio.CancelledError
            if status == TurnStatus.failed:
                error = completed.turn.error
                message = error.message if error is not None and error.message else f"turn {status.value}"
                raise AIFunctionError(message, function_name=self.name)

        if state.final_answer is not None:
            return state.final_answer
        return state.last_unphased or ""

    async def _interrupt_on_cancel(self, ctx: ThreadContext, handle: AsyncTurnHandle) -> None:
        """Interrupt the in-flight turn when the cycle's cancel signal fires.

        The interrupted turn completes with ``TurnStatus.interrupted``, which
        ``_run_turn`` translates into ``asyncio.CancelledError`` — so cancel
        takes effect mid-turn instead of waiting for the turn to finish.
        """
        await ctx.cancel_signal.wait()
        try:
            _ = await handle.interrupt()
        except Exception:  # noqa: BLE001 - the turn may have just completed
            pass

    # ── Notification-to-event mapping ──

    def _emit_notification(self, notification: Notification, ctx: ThreadContext, state: _TurnState) -> None:
        """Translate one stream notification into ai_functions events."""
        payload = notification.payload
        if isinstance(payload, ItemStartedNotification):
            self._emit_item_started(_item_root(payload.item), ctx)
            return
        if isinstance(payload, ItemCompletedNotification):
            self._emit_item_completed(_item_root(payload.item), ctx, state)
            return
        if isinstance(payload, AgentMessageDeltaNotification):
            ctx.on_event(
                MessageAssistantTokenEvent(
                    message_id=MessageId(payload.item_id),
                    text=payload.delta,
                    complete=False,
                ),
            )
            return
        if isinstance(payload, ReasoningTextDeltaNotification | ReasoningSummaryTextDeltaNotification):
            ctx.on_event(
                MessageAssistantThinkingEvent(
                    message_id=MessageId(payload.item_id),
                    text=payload.delta,
                    complete=False,
                ),
            )
            return
        if isinstance(payload, ThreadTokenUsageUpdatedNotification):
            if payload.turn_id == state.turn_id:
                delta = _breakdown_to_token_usage(payload.token_usage)
                state.usage = delta if state.usage is None else state.usage + delta
            return
        if isinstance(payload, TurnCompletedNotification):
            if payload.turn.id == state.turn_id:
                state.completed = payload
            return
        if isinstance(payload, ErrorNotification):
            ctx.on_event(
                CustomEvent(
                    kind="codex_error",
                    payload={"message": payload.error.message, "will_retry": payload.will_retry},
                ),
            )
            return
        if notification.method == "turn/started":
            return  # the dispatcher owns lifecycle events
        kind = "codex_plan" if "plan" in notification.method else f"codex_{notification.method.replace('/', '_')}"
        ctx.on_event(CustomEvent(kind=kind, payload=_payload_dict(payload)))

    def _emit_item_started(self, item: object, ctx: ThreadContext) -> None:
        """Emit the opening event for one thread item."""
        if isinstance(item, AgentMessageThreadItem):
            ctx.on_event(MessageAssistantStartEvent(message_id=MessageId(item.id)))
            return
        call = _tool_call_for(item)
        if call is not None:
            ctx.on_event(call)

    def _emit_item_completed(self, item: object, ctx: ThreadContext, state: _TurnState) -> None:
        """Emit the closing event for one thread item and track the answer."""
        if isinstance(item, UserMessageThreadItem):
            return  # echo of our own input; MESSAGE_USER was emitted at send time
        if isinstance(item, AgentMessageThreadItem):
            ctx.on_event(
                MessageAssistantCompleteEvent(
                    message_id=MessageId(item.id),
                    content=cast("Any", [{"text": item.text}]),  # pyright: ignore[reportExplicitAny]
                ),
            )
            if item.phase == MessagePhase.final_answer:
                state.final_answer = item.text
            elif item.phase is None:
                state.last_unphased = item.text
            return
        if isinstance(item, ReasoningThreadItem):
            for entry in [*(item.summary or []), *(item.content or [])]:
                if entry:
                    ctx.on_event(
                        MessageAssistantThinkingEvent(
                            message_id=MessageId(item.id),
                            text=entry,
                            complete=True,
                        ),
                    )
            return
        if isinstance(item, PlanThreadItem):
            ctx.on_event(CustomEvent(kind="codex_plan", payload={"id": item.id, "text": item.text}))
            return
        result = _tool_result_for(item)
        if result is not None:
            ctx.on_event(result)
            return
        item_type = _item_type(item)
        if item_type not in _MAPPED_ITEM_TYPES:
            ctx.on_event(CustomEvent(kind=f"codex_item_{item_type}", payload=_payload_dict(item)))


def _breakdown_to_token_usage(usage: ThreadTokenUsage) -> TokenUsage:
    """Convert one ``last`` (per-model-call) breakdown to a ai_functions ``TokenUsage``.

    Codex counts cached tokens inside ``inputTokens`` (verified live:
    ``totalTokens == inputTokens + outputTokens`` with ``cachedInputTokens``
    a subset of ``inputTokens``), while ``TokenUsage`` totals
    ``input + cache_read + cache_write + output`` — so the cached share is
    subtracted out of ``input_tokens``. Codex reports no cache-write figure.
    """
    last = usage.last
    cached = last.cached_input_tokens
    return TokenUsage(
        input_tokens=max(0, last.input_tokens - cached),
        output_tokens=last.output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )


def _tool_call_for(item: object) -> ToolCallEvent | None:
    """Build the ``TOOL_CALL`` event for a tool-shaped thread item, if any."""
    if isinstance(item, CommandExecutionThreadItem):
        return ToolCallEvent(
            message_id=None,
            tool_use_id=item.id,
            tool_name="command_execution",
            arguments={"command": item.command, "cwd": str(item.cwd)},
        )
    if isinstance(item, FileChangeThreadItem):
        changes = [change.model_dump(mode="json", by_alias=True) for change in item.changes]
        return ToolCallEvent(
            message_id=None,
            tool_use_id=item.id,
            tool_name="file_change",
            arguments={"changes": changes},
        )
    if isinstance(item, McpToolCallThreadItem):
        raw = item.arguments
        arguments = {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {"arguments": raw}
        return ToolCallEvent(
            message_id=None,
            tool_use_id=item.id,
            tool_name=f"{item.server}.{item.tool}",
            arguments=cast("dict[str, object]", arguments),
        )
    if isinstance(item, DynamicToolCallThreadItem):
        raw = item.arguments
        arguments = {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {"arguments": raw}
        return ToolCallEvent(
            message_id=None,
            tool_use_id=item.id,
            tool_name=item.tool,
            arguments=cast("dict[str, object]", arguments),
        )
    if isinstance(item, WebSearchThreadItem):
        return ToolCallEvent(
            message_id=None,
            tool_use_id=item.id,
            tool_name="web_search",
            arguments={"query": item.query},
        )
    return None


def _tool_result_for(item: object) -> ToolResultEvent | None:
    """Build the ``TOOL_RESULT`` event for a completed tool-shaped item, if any."""
    if isinstance(item, CommandExecutionThreadItem):
        failed = _status_value(item.status) in ("failed", "declined") or (
            item.exit_code is not None and item.exit_code != 0
        )
        return ToolResultEvent(
            message_id=None,
            tool_use_id=item.id,
            status=cast("Any", "error" if failed else "success"),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", _tool_result_text(item.aggregated_output)),  # pyright: ignore[reportExplicitAny]
        )
    if isinstance(item, FileChangeThreadItem):
        return ToolResultEvent(
            message_id=None,
            tool_use_id=item.id,
            status=cast("Any", "error" if _status_value(item.status) == "failed" else "success"),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", _tool_result_text(_status_value(item.status))),  # pyright: ignore[reportExplicitAny]
        )
    if isinstance(item, McpToolCallThreadItem):
        failed = _status_value(item.status) == "failed" or item.error is not None
        body = item.result.model_dump_json(by_alias=True) if item.result is not None else None
        error_text = item.error.message if item.error is not None else None
        return ToolResultEvent(
            message_id=None,
            tool_use_id=item.id,
            status=cast("Any", "error" if failed else "success"),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", _tool_result_text(error_text, body)),  # pyright: ignore[reportExplicitAny]
        )
    if isinstance(item, DynamicToolCallThreadItem):
        failed = _status_value(item.status) == "failed" or item.success is False
        parts = [c.model_dump_json(by_alias=True) for c in (item.content_items or [])]
        return ToolResultEvent(
            message_id=None,
            tool_use_id=item.id,
            status=cast("Any", "error" if failed else "success"),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", _tool_result_text(*parts)),  # pyright: ignore[reportExplicitAny]
        )
    if isinstance(item, WebSearchThreadItem):
        action = item.action.model_dump_json(by_alias=True) if item.action is not None else None
        return ToolResultEvent(
            message_id=None,
            tool_use_id=item.id,
            status=cast("Any", "success"),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", _tool_result_text(action or item.query)),  # pyright: ignore[reportExplicitAny]
        )
    return None
