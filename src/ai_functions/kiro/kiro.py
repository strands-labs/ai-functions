"""``KiroAgent`` template and ``KiroAgentThread`` — ACP-backed Kiro Agent thread.

A ``KiroAgentThread`` drives a ``kiro-cli acp`` subprocess over the Agent Client
Protocol (ACP, JSON-RPC over stdio). It holds a *persistent ACP session* and
receives *structured* ``session/update`` notifications, which it re-emits as
ai_functions events for observability. The ACP agent owns the conversation
transcript; ai_functions observes the update stream — pure observability, not a
source of truth.

The thread supports an optional post-condition retry loop: when the template
carries ``post_conditions``, each turn's result is validated and, on failure,
the failure messages are fed back as the next user turn until the result passes
or ``max_attempts`` is reached. Tool-approval requests are auto-approved or
routed through ``ctx.on_interrupt`` (see ``auto_approve``).

ACP-to-event mapping
────────────────────

Each ``session/update`` notification maps to a fixed set of ai_functions events.
A fresh ``message_id`` is minted per turn and shared by every event of that
turn. Lifecycle events (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``,
``RESULT``) are emitted by the runtime dispatcher, never by the thread.

- ``agent_message_chunk``: one ``MESSAGE_ASSISTANT_TOKEN`` (``complete=False``)
  per text chunk; the chunks are accumulated and emitted once as
  ``MESSAGE_ASSISTANT_COMPLETE`` when the prompt turn ends.
- ``agent_thought_chunk``: one ``MESSAGE_ASSISTANT_THINKING`` (``complete=False``).
- ``tool_call`` (``ToolCallStart``): one ``TOOL_CALL`` with the tool title as
  name and ``raw_input`` as arguments.
- ``tool_call_update`` (``ToolCallProgress``) reaching ``completed``/``failed``:
  one ``TOOL_RESULT`` whose content comes from the update's display ``content``
  blocks, falling back to a JSON-encoded ``raw_output`` when those are empty
  (Kiro often reports the structured result only in ``raw_output``).
- Token accounting: one ``TOKEN_USAGE`` per turn when the ``session/prompt``
  response carries a ``usage`` payload (``kiro-cli acp`` may omit it, in which
  case no usage event is emitted).
- Other update variants (plans, available-commands, mode/config changes) are
  currently ignored.

Permission requests (``session/request_permission``) are auto-approved when the
template's ``auto_approve`` is set (the default, for non-interactive runs);
otherwise they are routed through ``ctx.on_interrupt``.

Invariants:
    I2 — every emitted event goes through ``Coordinator.append_event``.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Hashable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, cast, final, override

from strands.interrupt import Interrupt
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.content import ContentBlock
from strands.types.interrupt import InterruptResponseContent
from strands.types.tools import AgentTool, ToolResultContent, ToolResultStatus

from ..ai_thread.errors import AIFunctionError
from ..ai_thread.postcondition import PostCondition
from ..protocols import Spawnable, Thread
from ..types import (
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
    import acp
    from acp import (
        Client as _AcpClient,
    )
    from acp import (
        ClientSideConnection,
        RequestPermissionResponse,
    )
    from acp.schema import (
        AgentMessageChunk,
        AgentThoughtChunk,
        AllowedOutcome,
        ClientCapabilities,
        DeniedOutcome,
        PermissionOption,
        ToolCallProgress,
        ToolCallStart,
        ToolCallUpdate,
        Usage,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "KiroAgent requires the optional 'kiro' extra (the Agent Client Protocol "
        "client). Install it with:\n    pip install 'strands-ai-functions[kiro]'",
    ) from exc

if TYPE_CHECKING:
    from acp.schema import (
        AgentPlanUpdate,
        AvailableCommandsUpdate,
        ConfigOptionUpdate,
        CurrentModeUpdate,
        SessionInfoUpdate,
        UsageUpdate,
        UserMessageChunk,
    )


def _new_message_id() -> MessageId:
    return MessageId(f"msg-{uuid.uuid4().hex}")


def _usage_to_token_usage(usage: Usage | None) -> TokenUsage | None:
    """Extract a ai_functions ``TokenUsage`` from an ACP ``Usage`` payload.

    Returns ``None`` when ``usage`` is ``None`` so callers can skip emission.
    ``kiro-cli acp`` may not populate usage on the prompt response.
    """
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cached_read_tokens or 0,
        cache_write_tokens=usage.cached_write_tokens or 0,
    )


def _content_block_text(block: object) -> str:
    """Return the text of an ACP content block, or ``""`` for non-text blocks."""
    if getattr(block, "type", None) == "text":
        text = getattr(block, "text", None)
        return text if isinstance(text, str) else ""
    return ""


def _tool_arguments(raw_input: object) -> dict[str, object]:
    """Coerce an ACP tool ``raw_input`` into a string-keyed arguments dict."""
    if isinstance(raw_input, dict):
        return {str(k): v for k, v in cast("dict[object, object]", raw_input).items()}
    return {}


def _tool_result_content(content: object, raw_output: object) -> list[object]:
    """Flatten an ACP tool result into Strands-shape content blocks.

    Prefers the display ``content`` blocks; when the agent leaves those empty
    (it often reports the structured result only in ``raw_output``), falls back
    to a JSON-encoded ``raw_output`` text block so the ``TOOL_RESULT`` event
    still carries the tool's output.
    """
    blocks: list[object] = []
    if isinstance(content, list):
        for item in cast("list[object]", content):
            inner = getattr(item, "content", None)
            text = _content_block_text(inner) if inner is not None else ""
            if text:
                blocks.append({"text": text})
    if not blocks and raw_output is not None:
        blocks.append({"text": json.dumps(raw_output, default=str)})
    return blocks


@dataclass
class _Turn:
    """Mutable per-prompt accumulator shared with the ACP client callback."""

    message_id: MessageId
    text_parts: list[str] = field(default_factory=list[str])
    started: bool = False
    """Whether ``MESSAGE_ASSISTANT_START`` has been emitted for this turn."""


@final
@dataclass(frozen=True)
class KiroAgent(Spawnable[[str], str], ToolProvider):
    """Immutable template for a Kiro-Agent-backed (ACP) thread.

    Carries the configuration used to spawn the ``kiro-cli acp`` subprocess
    plus the display metadata needed to expose the resulting thread as a
    Strands tool, and the post-condition retry policy. Picklable and safe to
    share across runtimes.

    Implements:
        Spawnable, strands.tools.ToolProvider.

    Immutable: Yes.
    """

    executable: str = "kiro-cli"
    """Name or path of the Kiro CLI binary; invoked as ``<executable> acp``."""

    agent: str | None = None
    """Optional Kiro agent profile, passed as ``kiro-cli acp --agent`` when set."""

    model: str | None = None
    """Optional model id, passed as ``kiro-cli acp --model`` when set."""

    cwd: str | None = None
    """Working directory for the ACP session; ``None`` uses the process cwd.

    ACP requires an absolute path; the value is resolved with ``os.path.abspath``.
    """

    auto_approve: bool = True
    """When True, permission requests are auto-approved without involving
    ``ctx.on_interrupt`` — required for non-interactive autonomous runs. When
    False, requests are routed through ``ctx.on_interrupt``."""

    name: str = "kiro"
    """Name used for telemetry and when exposed as a Strands tool."""

    description: str = "Send a prompt to a Kiro agent and receive its final answer."
    """Description used when exposed as a Strands tool."""

    post_conditions: tuple[PostCondition, ...] = ()
    """Validators run against each cycle's result. On failure the thread feeds
    the failure messages back as the next user turn and re-runs, up to
    ``max_attempts``. Empty (default) disables the retry loop — behaviour is
    then a single query."""

    max_attempts: int = 10
    """Maximum number of cycles to satisfy ``post_conditions``. Ignored when
    ``post_conditions`` is empty — the loop short-circuits after the first
    query, so the default single-query behaviour is unchanged unless
    ``post_conditions`` is set."""

    @property
    def input_shape(self) -> InputShape:
        """Every KiroAgent thread accepts a single string prompt."""
        return InputShape.STR_PROMPT

    @override
    def to_thread(self) -> KiroAgentThread:
        """Produce a fresh ``KiroAgentThread`` bound to this template.

        The returned thread owns its own ACP session; the ``kiro-cli acp``
        subprocess is not spawned until the first cycle runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        return KiroAgentThread(self)

    @override
    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Expose this template as a Strands tool.

        The returned tool takes one ``prompt: str`` argument; each invocation
        spawns a private ``KiroAgentThread``, runs a single cycle, and tears
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
class KiroAgentThread(Thread[[str], str]):
    """Live Kiro-Agent-backed thread that owns one persistent ACP session.

    Connects the ``kiro-cli acp`` subprocess lazily on the first cycle and
    keeps the session alive for the thread's lifetime. The ACP agent owns
    conversation history; ai_functions observes the ``session/update`` stream
    and re-emits each element as a ai_functions event — pure observability, not
    a source of truth. The module docstring describes the full mapping.

    Implements:
        Thread[[str], str].

    Lifecycle:
        CREATED → CONNECTED → CLOSED.
    """

    __slots__ = (
        "_template",
        "_stack",
        "_conn",
        "_session_id",
        "_connected",
        "_connect_lock",
        "_active_ctx",
        "_turn",
        "_inject_buffer",
    )

    def __init__(self, template: KiroAgent) -> None:
        self._template: KiroAgent = template
        self._stack: AsyncExitStack | None = None
        self._conn: ClientSideConnection | None = None
        self._session_id: str | None = None
        self._connected: bool = False
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        # Populated for the duration of each cycle so the ACP client callback
        # can reach the live ctx; the dispatcher serialises cycles.
        self._active_ctx: ThreadContext | None = None
        # Current prompt's accumulator, swapped per turn.
        self._turn: _Turn | None = None
        # Pending side-channel messages delivered via ``notify``; prepended to
        # the next outgoing user turn (the ACP session owns history, so
        # injecting mid-turn is not supported).
        self._inject_buffer: list[str] = []

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``KiroAgent`` template."""
        return self._template.name

    async def notify(self, text: str) -> None:
        """Buffer ``text`` to be prepended to the next outgoing user turn.

        The ACP session owns the conversation turn boundary; ai_functions
        cannot inject text mid-stream. The message sits in the inject buffer
        and is prepended to the next ``execute`` prompt.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - ``text`` is appended to the thread-local inject buffer.
            - No ACP prompt is issued by this call.
        """
        self._inject_buffer.append(text)

    async def execute(self, ctx: ThreadContext, prompt: str) -> str:
        """Send ``prompt`` to the ACP session and return its string result.

        Drains the inject buffer, prepending any pending messages to the
        outgoing turn. Each drained entry and ``prompt`` are emitted as
        individual ``MESSAGE_USER`` events; the combined text is then sent as
        a single ACP ``session/prompt``. The streamed ``session/update``
        notifications are re-emitted per the mapping table in the module
        docstring.

        When the template carries ``post_conditions``, the result is validated
        after each turn; on failure the failure messages are fed back as the
        next user turn and the prompt re-runs, up to ``max_attempts`` (the
        original ``prompt`` is sent only on the first attempt — the ACP session
        owns history, so retries ride the feedback turn).

        Args:
            ctx: Freshly built per-cycle context; never reused across cycles.
            prompt: User prompt forwarded to the Kiro agent session.

        Returns:
            The accumulated assistant text for the (final) turn, or the empty
            string if the agent produced no text.

        Emits:
            - MESSAGE_USER — one per drained inject-buffer entry, plus one for ``prompt``.
            - MESSAGE_ASSISTANT_START — per turn.
            - MESSAGE_ASSISTANT_TOKEN — per agent message chunk.
            - MESSAGE_ASSISTANT_THINKING — per agent thought chunk.
            - MESSAGE_ASSISTANT_COMPLETE — per turn.
            - TOOL_CALL — per tool-call start.
            - TOOL_RESULT — per completed/failed tool call.
            - TOKEN_USAGE — once per turn when the prompt response carries usage.

        Raises:
            asyncio.CancelledError: ``ctx.cancel_signal`` was set at the cycle
                boundary.
            AIFunctionError: Post-conditions were not satisfied within
                ``max_attempts`` attempts.

        Concurrency:
            Pause and cancel signals are honoured at cycle boundaries only;
            to stop mid-turn use ``Coordinator.cancel`` (maps to ACP
            ``session/cancel``).
        """
        if ctx.cancel_signal.is_set():
            raise asyncio.CancelledError
        await ctx.coordinator.wait_until_unpaused(ctx.thread_id)
        self._active_ctx = ctx
        try:
            await self._ensure_connected()

            post_conditions = self._template.post_conditions
            max_attempts = self._template.max_attempts if post_conditions else 1

            result = ""
            for attempt in range(max(1, max_attempts)):
                # Drain inject buffer: each pending message is emitted as its
                # own MESSAGE_USER event, then prepended to the outgoing turn.
                # On retries the buffer holds the post-condition failure
                # feedback appended below; the original prompt is sent only on
                # the first attempt — the ACP session owns history, so retries
                # ride the feedback turn.
                pending = list(self._inject_buffer)
                self._inject_buffer.clear()

                parts: list[str] = []
                for injected in pending:
                    ctx.on_event(MessageUserEvent(text=injected))
                    parts.append(injected)
                if attempt == 0:
                    ctx.on_event(MessageUserEvent(text=prompt))
                    parts.append(prompt)
                combined = "\n\n".join(parts)

                result = await self._run_turn(ctx, combined)

                if not post_conditions:
                    return result

                errors = await self._validate_result(result, post_conditions)
                if not errors:
                    return result

                failures = "\n".join(f"- {e}" for e in errors)
                self._inject_buffer.append(
                    f"[{self.name}] Post-condition failures (attempt {attempt + 1}/{max_attempts}):\n{failures}",
                )

            raise AIFunctionError(
                f"Post-conditions not satisfied after {max_attempts} attempt(s)",
                function_name=self.name,
            )
        finally:
            self._active_ctx = None

    async def fork(self) -> Spawnable[[str], str]:
        """Not supported.

        Returns:
            Never returns; always raises.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "KiroAgentThread does not support forking; the underlying ACP session cannot be cloned.",
        )

    async def teardown(self) -> None:
        """Close the ACP session and release the subprocess.

        Ensures:
            - The ACP session and its ``kiro-cli acp`` subprocess are closed.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        self._inject_buffer.clear()
        stack = self._stack
        self._stack = None
        self._conn = None
        self._session_id = None
        self._connected = False
        if stack is not None:
            await stack.aclose()

    def serialize_result(self, result: str) -> str:
        """Return ``result`` unchanged; Kiro agent results are already strings."""
        return result

    def deserialize_result(self, payload: str) -> str:
        """Return ``payload`` unchanged; Kiro agent results are already strings."""
        return payload

    @property
    def template(self) -> KiroAgent:
        """The template this thread was created from."""
        return self._template

    @property
    def is_connected(self) -> bool:
        """Whether the underlying ACP session is currently connected."""
        return self._connected

    @property
    def session_id(self) -> str | None:
        """The ACP session id once connected, or ``None``."""
        return self._session_id

    # ── Internals ──

    def _resolve_cwd(self) -> str:
        """Return the absolute working directory for the ACP session."""
        return os.path.abspath(self._template.cwd or os.getcwd())

    async def _ensure_connected(self) -> None:
        """Lazily spawn ``kiro-cli acp`` and open an ACP session on first cycle."""
        if self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            cwd = self._resolve_cwd()
            stack = AsyncExitStack()
            client = _KiroAcpClient(self)
            extra_args: list[str] = []
            if self._template.agent is not None:
                extra_args += ["--agent", self._template.agent]
            if self._template.model is not None:
                extra_args += ["--model", self._template.model]
            conn, _proc = await stack.enter_async_context(
                acp.spawn_agent_process(client, self._template.executable, "acp", *extra_args, cwd=cwd),
            )
            await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
            )
            session = await conn.new_session(cwd=cwd, mcp_servers=[])
            self._stack = stack
            self._conn = conn
            self._session_id = session.session_id
            self._connected = True

    async def _run_turn(self, ctx: ThreadContext, prompt: str) -> str:
        """Send one ACP prompt, stream its updates, and return the turn's text.

        ``MESSAGE_ASSISTANT_START`` is emitted lazily by the update handler when
        the first child event of the turn arrives, and the matching
        ``MESSAGE_ASSISTANT_COMPLETE`` is emitted here in ``finally`` — so a
        ``session/prompt`` that fails mid-turn never leaves a dangling START.
        """
        assert self._conn is not None
        assert self._session_id is not None
        message_id = _new_message_id()
        turn = _Turn(message_id=message_id)
        self._turn = turn
        response = None
        try:
            response = await self._conn.prompt(
                prompt=[acp.text_block(prompt)],
                session_id=self._session_id,
            )
        finally:
            self._turn = None
            if turn.started:
                ctx.on_event(
                    MessageAssistantCompleteEvent(
                        message_id=message_id,
                        content=[cast("ContentBlock", {"text": "".join(turn.text_parts)})],
                    ),
                )
        usage = _usage_to_token_usage(response.usage)
        if usage is not None:
            ctx.on_event(TokenUsageEvent(token_usage=usage))
        return "".join(turn.text_parts)

    async def _validate_result(
        self,
        result: str,
        post_conditions: tuple[PostCondition, ...],
    ) -> list[str]:
        """Evaluate every post-condition against ``result`` in parallel.

        A condition returning ``None``/``passed`` passes; ``passed=False``
        contributes its message; a raised exception is treated as failure with
        the exception text. ``KiroAgentThread`` takes a single string prompt, so
        there are no bound keyword arguments to offer condition callables.

        Args:
            result: The candidate result string for the turn.
            post_conditions: Validators to run.

        Returns:
            Failure messages; empty when all conditions pass.
        """

        async def _run_one(cond: PostCondition) -> str | None:
            try:
                cond_result = cond(result)
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
            except Exception as exc:  # noqa: BLE001 - any failure is a failed condition
                return str(exc)
            if cond_result is None or cond_result.passed:
                return None
            return cond_result.message

        outcomes = await asyncio.gather(*(_run_one(c) for c in post_conditions))
        return [msg for msg in outcomes if msg is not None]

    # ── ACP client callbacks (invoked from the connection's read loop) ──

    def _handle_update(self, update: object) -> None:
        """Translate one ACP ``session/update`` notification to ai_functions events."""
        ctx = self._active_ctx
        turn = self._turn
        if ctx is None or turn is None:
            return
        message_id = turn.message_id
        if isinstance(update, AgentMessageChunk):
            text = _content_block_text(update.content)
            if text:
                self._ensure_started(ctx, turn)
                turn.text_parts.append(text)
                ctx.on_event(MessageAssistantTokenEvent(message_id=message_id, text=text, complete=False))
        elif isinstance(update, AgentThoughtChunk):
            text = _content_block_text(update.content)
            if text:
                self._ensure_started(ctx, turn)
                ctx.on_event(MessageAssistantThinkingEvent(message_id=message_id, text=text, complete=False))
        elif isinstance(update, ToolCallStart):
            self._ensure_started(ctx, turn)
            ctx.on_event(
                ToolCallEvent(
                    message_id=message_id,
                    tool_use_id=update.tool_call_id,
                    tool_name=update.title,
                    arguments=_tool_arguments(update.raw_input),
                ),
            )
        elif isinstance(update, ToolCallProgress):
            if update.status in ("completed", "failed"):
                self._ensure_started(ctx, turn)
                status = "error" if update.status == "failed" else "success"
                result_content = _tool_result_content(update.content, update.raw_output)
                ctx.on_event(
                    ToolResultEvent(
                        message_id=message_id,
                        tool_use_id=update.tool_call_id,
                        status=cast("ToolResultStatus", status),
                        content=cast("list[ToolResultContent]", result_content),
                    ),
                )

    @staticmethod
    def _ensure_started(ctx: ThreadContext, turn: _Turn) -> None:
        """Emit ``MESSAGE_ASSISTANT_START`` once, before the turn's first child event."""
        if not turn.started:
            turn.started = True
            ctx.on_event(MessageAssistantStartEvent(message_id=turn.message_id))

    async def _handle_permission(
        self,
        options: list[PermissionOption],
        tool_call: ToolCallUpdate,
    ) -> RequestPermissionResponse:
        """Resolve an ACP permission request to an allow/deny outcome."""
        if self._template.auto_approve:
            return _allow_or_deny(options)
        ctx = self._active_ctx
        if ctx is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        interrupt = Interrupt(
            id=tool_call.tool_call_id or f"kiro-approval-{uuid.uuid4().hex}",
            name="kiro_tool_approval",
            reason={
                "tool_name": tool_call.title,
                "tool_call_id": tool_call.tool_call_id,
                "options": [{"option_id": o.option_id, "kind": o.kind, "name": o.name} for o in options],
            },
        )
        responses: list[InterruptResponseContent] = await ctx.on_interrupt([interrupt])
        if not responses:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        decision = responses[0].get("interruptResponse", {}).get("response")
        if _is_allow(decision):
            return _allow_or_deny(options)
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _allow_or_deny(options: list[PermissionOption]) -> RequestPermissionResponse:
    """Select the first allow-style option, or deny when none is offered."""
    for option in options:
        if option.kind in ("allow_once", "allow_always"):
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=option.option_id),
            )
    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _is_allow(decision: object) -> bool:
    """Interpret an ``ctx.on_interrupt`` decision as allow (True) or deny (False)."""
    if isinstance(decision, str):
        return decision.lower() in ("allow", "allow_once", "allow_always", "approve", "yes")
    if isinstance(decision, dict):
        behavior = cast("dict[str, object]", decision).get("behavior")
        return behavior == "allow"
    return False


@final
class _KiroAcpClient(_AcpClient):
    """ACP ``Client`` bound to a ``KiroAgentThread``.

    Receives ``session/update`` notifications and ``session/request_permission``
    calls from the ``kiro-cli acp`` agent and forwards them to the owning
    thread. File-system and terminal capabilities are not advertised, so those
    client methods are never invoked; they raise ``method not found`` defensively.
    """

    __slots__ = ("_thread",)

    def __init__(self, thread: KiroAgentThread) -> None:
        self._thread: KiroAgentThread = thread

    @override
    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: object,
    ) -> None:
        """Forward one session update to the owning thread for event mapping."""
        self._thread._handle_update(update)  # pyright: ignore[reportPrivateUsage]

    @override
    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: object,
    ) -> RequestPermissionResponse:
        """Resolve a permission request via the owning thread."""
        return await self._thread._handle_permission(options, tool_call)  # pyright: ignore[reportPrivateUsage]

    @override
    async def read_text_file(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``fs`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="fs/read_text_file not supported")

    @override
    async def write_text_file(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``fs`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="fs/write_text_file not supported")

    @override
    async def create_terminal(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``terminal`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="terminal/create not supported")

    @override
    async def terminal_output(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``terminal`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="terminal/output not supported")

    @override
    async def release_terminal(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``terminal`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="terminal/release not supported")

    @override
    async def wait_for_terminal_exit(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``terminal`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="terminal/wait_for_exit not supported")

    @override
    async def kill_terminal(self, *args: object, **kwargs: object) -> NoReturn:
        """Unsupported; ``terminal`` capability is not advertised."""
        raise acp.exceptions.RequestError(code=-32601, message="terminal/kill not supported")

    @override
    def on_connect(self, conn: object) -> None:
        """No-op connection hook; the thread holds the connection directly."""
        return None

    @override
    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Unsupported custom extension method."""
        raise acp.exceptions.RequestError(code=-32601, message=f"ext method {method!r} not supported")

    @override
    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        """Ignore custom extension notifications."""
        return None
