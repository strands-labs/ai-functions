"""``ClaudeAgent`` template and ``ClaudeAgentThread`` — Claude Agent SDK-backed thread.

A ``ClaudeAgentThread`` drives a ``claude_agent_sdk.ClaudeSDKClient`` subprocess.
The SDK owns the conversation transcript; ai_functions observes its message stream and
re-emits each element through ``Coordinator.append_event`` as an observability
shadow. I7/I9 do not apply (the SDK, not the runtime, owns history); the thread
drains injected messages at work boundaries, supports pause, and routes
tool-approval requests through ``ctx.on_interrupt``.

SDK-to-event mapping
────────────────────

Every element of ``ClaudeSDKClient.receive_response()`` maps to a fixed set of
ai_functions events. No new built-in event kinds are introduced — content blocks are
re-packed into the Strands-format shapes the existing events expect. Lifecycle
events (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``, ``RESULT``) are
emitted by the runtime dispatcher, never by the thread.

- ``UserMessage`` with textual content: ignored (the thread already emitted
  ``MESSAGE_USER`` when it sent the prompt).
- ``UserMessage`` containing ``ToolResultBlock`` entries: one ``TOOL_RESULT``
  per block.
- ``AssistantMessage``: one ``MESSAGE_ASSISTANT_START`` with a fresh
  ``message_id``, one ``TOOL_CALL`` per ``ToolUseBlock``, one
  ``MESSAGE_ASSISTANT_THINKING`` per ``ThinkingBlock`` (``complete=True``),
  then one ``MESSAGE_ASSISTANT_COMPLETE`` whose ``content`` is the block list
  re-encoded as Strands-format ``ContentBlock`` dicts. The same ``message_id``
  ties these together.
- ``StreamEvent`` (only when ``options.include_partial_messages`` is set): one
  ``MESSAGE_ASSISTANT_TOKEN`` per text delta, one
  ``MESSAGE_ASSISTANT_THINKING`` per ``reasoningText`` delta, both with
  ``complete=False``.
- Token accounting: exactly one ``TOKEN_USAGE`` per turn, drawn from the
  richest available source (``ResultMessage.usage`` preferred over
  per-``AssistantMessage`` totals) so usage is never double-counted.
- ``RateLimitEvent``: ``CustomEvent(kind="claude_rate_limit", payload=...)``.
  Observability only — does not currently drive ``Coordinator.pause_signal``.
- ``SystemMessage`` variants (``TaskStartedMessage``, ``TaskProgressMessage``,
  ``TaskNotificationMessage``, ``MirrorErrorMessage``, …):
  ``CustomEvent(kind=f"claude_system_{subtype}", payload=...)``.

Invariants:
    I2 — every emitted event goes through ``Coordinator.append_event``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast, final, override

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    McpServerConfig,
    Message,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    RateLimitEvent,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from strands.interrupt import Interrupt
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.interrupt import InterruptResponseContent
from strands.types.tools import AgentTool

from ..ai_thread.errors import AIFunctionError
from ..ai_thread.postcondition import PostCondition
from ..protocols import Spawnable, Thread
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


def _new_message_id() -> MessageId:
    return MessageId(f"msg-{uuid.uuid4().hex}")


def _usage_to_token_usage(usage: dict[str, object] | None) -> TokenUsage | None:
    """Extract a ai_functions ``TokenUsage`` from an SDK ``usage`` dict.

    The Anthropic API returns usage keys ``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens``, ``cache_creation_input_tokens``. Values are
    ints; missing keys default to 0. Returns ``None`` when ``usage`` is
    ``None`` so callers can skip emission.
    """
    if not usage:
        return None

    def _int(key: str) -> int:
        v = usage.get(key)
        return v if isinstance(v, int) else 0

    return TokenUsage(
        input_tokens=_int("input_tokens"),
        output_tokens=_int("output_tokens"),
        cache_read_tokens=_int("cache_read_input_tokens"),
        cache_write_tokens=_int("cache_creation_input_tokens"),
    )


def _sdk_block_to_strands_content(block: object) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Re-encode one SDK content block as a Strands-format ``ContentBlock`` dict.

    The SDK uses dataclasses (``TextBlock``, ``ThinkingBlock``, ``ToolUseBlock``,
    ``ToolResultBlock``, ``ServerToolUseBlock``, ``ServerToolResultBlock``);
    Strands uses a flat TypedDict with ``text`` / ``toolUse`` / ``toolResult`` /
    ``reasoningContent`` keys. We map the fields the user-visible events
    actually carry; anything unrecognized falls back to a short ``text``
    stringification so the resulting message list remains well-formed.
    """
    if isinstance(block, TextBlock):
        return {"text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "reasoningContent": {
                "reasoningText": {"text": block.thinking, "signature": block.signature},
            },
        }
    if isinstance(block, ToolUseBlock):
        return {
            "toolUse": {
                "toolUseId": block.id,
                "name": block.name,
                "input": block.input,
            },
        }
    if isinstance(block, ToolResultBlock):
        return {"toolResult": _tool_result_dict(block)}
    if isinstance(block, ServerToolUseBlock):
        return {
            "toolUse": {
                "toolUseId": block.id,
                "name": block.name,
                "input": block.input,
            },
        }
    if isinstance(block, ServerToolResultBlock):
        return {
            "toolResult": {
                "toolUseId": block.tool_use_id,
                "status": "success",
                "content": [dict(block.content)],
            },
        }
    return {"text": repr(block)}


def _tool_result_dict(block: ToolResultBlock) -> dict[str, object]:
    """Pack a ``ToolResultBlock`` into a Strands-shape ``toolResult`` dict."""
    raw = block.content
    content: list[object]
    if raw is None:
        content = []
    elif isinstance(raw, str):
        content = [{"text": raw}]
    else:
        content = [dict(item) for item in raw]
    status = "error" if block.is_error else "success"
    return {
        "toolUseId": block.tool_use_id,
        "status": status,
        "content": content,
    }


def _tool_result_event_from_block(
    block: ToolResultBlock | ServerToolResultBlock,
    message_id: MessageId | None,
) -> ToolResultEvent:
    """Build a ai_functions ``ToolResultEvent`` from an SDK tool-result block."""
    if isinstance(block, ToolResultBlock):
        packed = _tool_result_dict(block)
        return ToolResultEvent(
            message_id=message_id,
            tool_use_id=block.tool_use_id,
            status=cast("Any", packed["status"]),  # pyright: ignore[reportExplicitAny]
            content=cast("Any", packed["content"]),  # pyright: ignore[reportExplicitAny]
        )
    return ToolResultEvent(
        message_id=message_id,
        tool_use_id=block.tool_use_id,
        status=cast("Any", "success"),  # pyright: ignore[reportExplicitAny]
        content=cast("Any", [dict(block.content)]),  # pyright: ignore[reportExplicitAny]
    )


def _rate_limit_payload(event: RateLimitEvent) -> dict[str, object]:
    info = event.rate_limit_info
    return {
        "status": info.status,
        "resets_at": info.resets_at,
        "rate_limit_type": info.rate_limit_type,
        "utilization": info.utilization,
        "overage_status": info.overage_status,
        "overage_resets_at": info.overage_resets_at,
        "overage_disabled_reason": info.overage_disabled_reason,
        "raw": info.raw,
        "uuid": event.uuid,
        "session_id": event.session_id,
    }


def _system_message_payload(message: SystemMessage) -> dict[str, object]:
    """Flatten a ``SystemMessage`` (including subclass fields) to a payload dict."""
    payload: dict[str, object] = {"subtype": message.subtype, "data": message.data}
    # Subclass-specific fields (TaskStartedMessage, TaskProgressMessage, …) —
    # harvest them generically via dataclass fields so we don't have to branch.
    if dataclasses.is_dataclass(message) and not isinstance(message, type):
        for field in dataclasses.fields(message):
            if field.name in ("subtype", "data"):
                continue
            payload[field.name] = getattr(message, field.name)
    return payload


@final
@dataclass(frozen=True)
class ClaudeAgent(Spawnable[[str], str], ToolProvider):
    """Immutable template for a Claude-Agent-backed thread.

    Carries the ``ClaudeAgentOptions`` used to spawn the underlying ``claude``
    subprocess plus the display metadata needed to expose the resulting thread
    as a Strands tool. Picklable and safe to share across runtimes.

    Implements:
        Spawnable, strands.tools.ToolProvider.

    Immutable: Yes.
    """

    options: ClaudeAgentOptions | None = None
    """Verbatim options forwarded to ``ClaudeSDKClient``; ``None`` uses SDK defaults."""

    name: str = "claude_code"
    """Name used for telemetry and when exposed as a Strands tool."""

    description: str = "Send a prompt to a Claude Agent and receive its final answer."
    """Description used when exposed as a Strands tool."""

    post_conditions: tuple[PostCondition, ...] = ()
    """Validators run against each cycle's result. On failure the thread
    feeds the failure messages back as the next user turn and re-runs, up to
    ``max_attempts``. Empty (default) disables the retry loop — behaviour is
    then a single query. Mirrors ``ThreadConfig.post_conditions`` so
    post-conditions are a Thread-level capability shared with ``AIThread``."""

    max_attempts: int = 10
    """Maximum number of cycles to satisfy ``post_conditions`` (mirrors
    ``ThreadConfig.max_attempts``). Ignored when ``post_conditions`` is empty —
    the loop short-circuits after the first query, so the default single-query
    behaviour is unchanged unless ``post_conditions`` is set."""

    @property
    def input_shape(self) -> InputShape:
        """Every ClaudeAgent thread accepts a single string prompt."""
        return InputShape.STR_PROMPT

    @override
    def to_thread(self) -> ClaudeAgentThread:
        """Produce a fresh ``ClaudeAgentThread`` bound to this template.

        The returned thread holds its own ``ClaudeSDKClient``; the subprocess
        is not spawned until the first cycle runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        return ClaudeAgentThread(self)

    @override
    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Expose this template as a Strands tool.

        The returned tool takes one ``prompt: str`` argument; each invocation
        spawns a private ``ClaudeAgentThread``, runs a single cycle, and tears
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
class ClaudeAgentThread(Thread[[str], str]):
    """Live Claude-Agent-backed thread that owns one ``ClaudeSDKClient``.

    Connects the SDK client lazily on the first cycle and keeps it alive for
    the thread's lifetime. The SDK owns conversation history; ai_functions observes
    the SDK's message stream and re-emits each element as a ai_functions event —
    pure observability, not a source of truth. The module docstring describes
    the full SDK-to-event mapping.

    Implements:
        Thread[[str], str].

    Lifecycle:
        CREATED → CONNECTED → CLOSED.
    """

    __slots__ = (
        "_template",
        "_client",
        "_connected",
        "_active_ctx",
        "_connect_lock",
        "_inject_buffer",
    )

    def __init__(self, template: ClaudeAgent) -> None:
        self._template: ClaudeAgent = template
        self._client: ClaudeSDKClient | None = None
        self._connected: bool = False
        # ``_active_ctx`` is populated for the duration of each cycle so the
        # permission shim installed on the SDK client can reach
        # ``ctx.on_interrupt``. The SDK's ``can_use_tool`` callback is invoked
        # from inside ``receive_response`` on the same event loop; reading this
        # attribute from there is safe because the dispatcher serialises cycles.
        self._active_ctx: ThreadContext | None = None
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        # Pending side-channel messages delivered via ``notify``.
        # Prepended to the outgoing user turn on the next ``execute``; Claude Agent
        # owns the conversation, so injecting mid-turn is not supported.
        self._inject_buffer: list[str] = []

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``ClaudeAgent`` template."""
        return self._template.name

    async def notify(self, text: str) -> None:
        """Buffer ``text`` to be prepended to the next outgoing user turn.

        The CLI owns the conversation turn boundary; ai_functions cannot inject
        text mid-stream. The message sits in the inject buffer and is
        prepended to the next ``execute`` prompt.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - ``text`` is appended to the thread-local inject buffer.
            - No SDK query is issued by this call.
        """
        self._inject_buffer.append(text)

    async def execute(self, ctx: ThreadContext, prompt: str) -> str:
        """Send ``prompt`` to the Claude Agent session and return its string result.

        Drains the inject buffer, prepending any pending messages to the
        outgoing turn. Each drained entry and ``prompt`` are emitted as
        individual ``MESSAGE_USER`` events; the combined text is then
        sent as a single SDK query. The response is streamed and re-emitted
        per the mapping table in the module docstring.

        Args:
            ctx: Freshly built per-cycle context; never reused across cycles.
            prompt: User prompt forwarded to the Claude Agent session.

        Returns:
            The ``ResultMessage.result`` string from the Claude Agent stream, or the
            empty string if the CLI produced no textual result.

        Emits:
            - MESSAGE_USER — one per drained inject-buffer entry, plus one for ``prompt``.
            - MESSAGE_ASSISTANT_START — per assistant turn.
            - MESSAGE_ASSISTANT_TOKEN — per text delta (partial streaming only).
            - MESSAGE_ASSISTANT_THINKING — per reasoning block or delta.
            - MESSAGE_ASSISTANT_COMPLETE — per assistant turn.
            - TOOL_CALL — per ``ToolUseBlock``.
            - TOOL_RESULT — per ``ToolResultBlock``.
            - TOKEN_USAGE — exactly one per turn.

        Tool-approval requests are routed through ``ctx.on_interrupt``, not
        emitted as events.

        Raises:
            asyncio.CancelledError: ``ctx.cancel_signal`` was set at the cycle
                boundary.

        Concurrency:
            Pause and cancel signals are honoured at cycle boundaries only;
            to stop mid-turn use ``Coordinator.cancel`` (maps to
            ``ClaudeSDKClient.interrupt``).
        """
        if ctx.cancel_signal.is_set():
            raise asyncio.CancelledError
        await ctx.coordinator.wait_until_unpaused(ctx.thread_id)
        # Set ``_active_ctx`` before connecting so the runtime MCP tools
        # registered at connect time can resolve a ctx on the first
        # cycle as well as later ones.
        self._active_ctx = ctx
        try:
            await self._ensure_connected()

            post_conditions = self._template.post_conditions
            max_attempts = self._template.max_attempts if post_conditions else 1

            result = ""
            for attempt in range(max(1, max_attempts)):
                # Drain inject buffer: each pending message is emitted as its
                # own MESSAGE_USER event, then prepended to the outgoing turn.
                # On the first attempt the buffer holds any caller-supplied
                # side-channel messages; on retries it holds the
                # post-condition failure feedback appended below. The original
                # ``prompt`` is only sent on the first attempt — the SDK owns
                # the conversation history, so retries ride the feedback turn
                # (mirrors ``AIThread``: failures are injected as the next
                # user turn, the task stays in context).
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

                assert self._client is not None
                await self._client.query(combined)
                result = await self._consume_stream(ctx)

                if not post_conditions:
                    return result

                errors = await self._validate_result(result, post_conditions)
                if not errors:
                    return result

                # Feed failures back as the next user turn and retry.
                failures = "\n".join(f"- {e}" for e in errors)
                self._inject_buffer.append(
                    f"[{self.name}] Post-condition failures (attempt {attempt + 1}/{max_attempts}):\n{failures}"
                )

            raise AIFunctionError(
                f"Post-conditions not satisfied after {max_attempts} attempt(s)",
                function_name=self.name,
            )
        finally:
            self._active_ctx = None

    async def _validate_result(
        self,
        result: str,
        post_conditions: tuple[PostCondition, ...],
    ) -> list[str]:
        """Evaluate every post-condition against ``result`` in parallel.

        Mirrors ``AIThread._validate_result``: a condition returning
        ``None``/``passed`` passes; ``passed=False`` contributes its message;
        a raised exception is treated as failure with the exception text.
        ``ClaudeAgentThread`` takes a single string prompt, so there are no
        bound keyword arguments to offer condition callables.

        Args:
            result: The candidate result string from the Claude Agent stream.
            post_conditions: Validators to run.

        Returns:
            Failure messages; empty when all conditions pass.
        """

        async def _run_one(cond: PostCondition) -> str | None:
            try:
                cond_result = cond(result)
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
            except Exception as exc:
                return str(exc)
            if cond_result is None or cond_result.passed:
                return None
            return cond_result.message

        outcomes = await asyncio.gather(*(_run_one(c) for c in post_conditions))
        return [msg for msg in outcomes if msg is not None]

    async def fork(self) -> Spawnable[[str], str]:
        """Not supported.

        Returns:
            Never returns; always raises.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ClaudeAgentThread does not support forking; the underlying SDK session cannot be cloned."
        )

    async def teardown(self) -> None:
        """Disconnect the SDK client and release the subprocess.

        Ensures:
            - Any connected ``ClaudeSDKClient`` is disconnected.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        self._inject_buffer.clear()
        if not self._connected or self._client is None:
            return
        try:
            await self._client.disconnect()
        finally:
            self._connected = False
            self._client = None

    def serialize_result(self, result: str) -> str:
        """Return ``result`` unchanged; Claude Agent results are already strings."""
        return result

    def deserialize_result(self, payload: str) -> str:
        """Return ``payload`` unchanged; Claude Agent results are already strings."""
        return payload

    @property
    def template(self) -> ClaudeAgent:
        """The template this thread was created from."""
        return self._template

    @property
    def is_connected(self) -> bool:
        """Whether the underlying ``ClaudeSDKClient`` is currently connected."""
        return self._connected

    # ── Internals ──

    async def _ensure_connected(self) -> None:
        """Lazily connect the SDK client on first cycle."""
        if self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            options = self._template.options or ClaudeAgentOptions()
            if options.can_use_tool is None:
                options = replace(options, can_use_tool=self._default_can_use_tool)
            options = self._wire_runtime_tools(options)
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._connected = True

    def _wire_runtime_tools(self, options: ClaudeAgentOptions) -> ClaudeAgentOptions:
        """Add the ai_functions runtime MCP server and its tool names to ``options``.

        The MCP server resolves the live cycle's ``ThreadContext`` via
        ``self._active_ctx`` at tool-invocation time. Connecting once
        and reusing the server across cycles is correct because the
        dispatcher serialises cycles.

        Args:
            options: The ``ClaudeAgentOptions`` to patch.

        Returns:
            A new ``ClaudeAgentOptions`` with the runtime MCP server
            merged into ``mcp_servers`` and the two tool names appended
            to ``allowed_tools``.

        Raises:
            ValueError: ``options.mcp_servers`` already contains the
                reserved key ``_ai_functions_runtime``.
        """
        from .coordinator_tools import (
            _RUNTIME_SERVER_NAME,  # pyright: ignore[reportPrivateUsage]  # package-internal
            _runtime_mcp_server_with_provider,  # pyright: ignore[reportPrivateUsage]  # package-internal
        )

        existing_servers = options.mcp_servers
        if isinstance(existing_servers, dict) and _RUNTIME_SERVER_NAME in existing_servers:
            raise ValueError(
                f"ClaudeAgentOptions.mcp_servers may not register the reserved "
                f"key {_RUNTIME_SERVER_NAME!r}; ai_functions uses it for the runtime "
                "MCP server.",
            )
        runtime_server = _runtime_mcp_server_with_provider(lambda: self._active_ctx)
        merged_servers: dict[str, McpServerConfig] = {
            **(existing_servers if isinstance(existing_servers, dict) else {}),
            _RUNTIME_SERVER_NAME: runtime_server,
        }
        merged_allowed = list(options.allowed_tools) + [
            f"mcp__{_RUNTIME_SERVER_NAME}__list_threads",
            f"mcp__{_RUNTIME_SERVER_NAME}__send_message",
        ]
        return replace(options, mcp_servers=merged_servers, allowed_tools=merged_allowed)

    async def _consume_stream(self, ctx: ThreadContext) -> str:
        """Iterate ``receive_response`` and translate each message to events."""
        assert self._client is not None
        result_text: str = ""
        result_usage: TokenUsage | None = None
        per_turn_usage: TokenUsage | None = None
        async for message in self._client.receive_response():
            turn_usage = self._emit_events_for(message, ctx)
            if turn_usage is not None:
                per_turn_usage = turn_usage if per_turn_usage is None else per_turn_usage + turn_usage
            if isinstance(message, ResultMessage):
                result_text = message.result or ""
                result_usage = _usage_to_token_usage(message.usage)
                break

        # Exactly one TOKEN_USAGE per turn: prefer the richer ResultMessage
        # totals; fall back to the sum of per-assistant-message usages.
        final_usage = result_usage if result_usage is not None else per_turn_usage
        if final_usage is not None:
            ctx.on_event(TokenUsageEvent(token_usage=final_usage))

        return result_text

    def _emit_events_for(
        self,
        message: Message,
        ctx: ThreadContext,
    ) -> TokenUsage | None:
        """Translate one SDK message to ai_functions events; return per-message usage."""
        if isinstance(message, AssistantMessage):
            return self._emit_assistant_message(message, ctx)
        if isinstance(message, UserMessage):
            self._emit_user_message(message, ctx)
            return None
        if isinstance(message, StreamEvent):
            self._emit_stream_event(message, ctx)
            return None
        if isinstance(message, ResultMessage):
            # Token accounting handled by caller; no per-message events.
            return None
        if isinstance(message, RateLimitEvent):
            ctx.on_event(
                CustomEvent(kind="claude_rate_limit", payload=_rate_limit_payload(message)),
            )
            return None
        ctx.on_event(
            CustomEvent(
                kind=f"claude_system_{message.subtype}",
                payload=_system_message_payload(message),
            ),
        )
        return None

    def _emit_assistant_message(
        self,
        message: AssistantMessage,
        ctx: ThreadContext,
    ) -> TokenUsage | None:
        """Emit the full event span for one assistant turn."""
        message_id = _new_message_id()
        ctx.on_event(MessageAssistantStartEvent(message_id=message_id))
        for block in message.content:
            if isinstance(block, ToolUseBlock | ServerToolUseBlock):
                arguments: dict[str, object] = {str(k): v for k, v in cast("dict[object, object]", block.input).items()}
                ctx.on_event(
                    ToolCallEvent(
                        message_id=message_id,
                        tool_use_id=block.id,
                        tool_name=block.name,
                        arguments=arguments,
                    ),
                )
            elif isinstance(block, ThinkingBlock):
                ctx.on_event(
                    MessageAssistantThinkingEvent(
                        message_id=message_id,
                        text=block.thinking,
                        complete=True,
                    ),
                )
        content_blocks = [_sdk_block_to_strands_content(b) for b in message.content]
        ctx.on_event(
            MessageAssistantCompleteEvent(
                message_id=message_id,
                content=cast("Any", content_blocks),  # pyright: ignore[reportExplicitAny]
            ),
        )
        return _usage_to_token_usage(message.usage)

    def _emit_user_message(self, message: UserMessage, ctx: ThreadContext) -> None:
        """Emit ``TOOL_RESULT`` events for tool-result blocks; ignore text echoes."""
        if isinstance(message.content, str):
            return
        for block in message.content:
            if isinstance(block, ToolResultBlock | ServerToolResultBlock):
                ctx.on_event(_tool_result_event_from_block(block, message_id=None))

    def _emit_stream_event(self, message: StreamEvent, ctx: ThreadContext) -> None:
        """Map a partial-message stream event to a token / thinking fragment."""
        event = cast("dict[str, object]", message.event)
        if event.get("type") != "content_block_delta":
            return
        delta_raw = event.get("delta")
        if not isinstance(delta_raw, dict):
            return
        delta = cast("dict[str, object]", delta_raw)
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                ctx.on_event(
                    MessageAssistantTokenEvent(text=text, complete=False),
                )
        elif delta_type == "thinking_delta":
            thinking = delta.get("thinking")
            if isinstance(thinking, str) and thinking:
                ctx.on_event(
                    MessageAssistantThinkingEvent(text=thinking, complete=False),
                )

    async def _default_can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],  # pyright: ignore[reportExplicitAny]
        context: ToolPermissionContext,
    ) -> PermissionResult:
        """Route Claude Agent permission requests through ``ctx.on_interrupt``.

        Synthesises a ``strands.interrupt.Interrupt`` carrying the tool-call
        context, awaits ``ctx.on_interrupt``, and parses the returned
        ``InterruptResponseContent`` into a ``PermissionResultAllow`` or
        ``PermissionResultDeny``.
        """
        ctx = self._active_ctx
        if ctx is None:
            # No active cycle (should not happen — ``can_use_tool`` is invoked
            # from within ``receive_response``). Default-deny rather than
            # crash.
            return PermissionResultDeny(
                message="Claude Agent tool approval requested with no active ai_functions cycle.",
                interrupt=False,
            )

        interrupt_id = context.tool_use_id or f"claude-agent-approval-{uuid.uuid4().hex}"
        interrupt = Interrupt(
            id=interrupt_id,
            name="claude_agent_tool_approval",
            reason={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_use_id": context.tool_use_id,
                "agent_id": context.agent_id,
            },
        )
        responses: list[InterruptResponseContent] = await ctx.on_interrupt([interrupt])
        if not responses:
            return PermissionResultDeny(
                message="No response from ai_functions approval handler.",
                interrupt=False,
            )
        response = responses[0].get("interruptResponse", {}).get("response")
        return _parse_permission_response(response)


def _parse_permission_response(response: object) -> PermissionResult:
    """Parse an ``InterruptResponse.response`` into a ``PermissionResult``.

    Accepts already-typed ``PermissionResult`` instances, ``dict`` payloads
    matching the SDK shape (``{"behavior": "allow"|"deny", ...}``), or a
    plain ``"allow"``/``"deny"`` string as a fallback. Anything else is
    treated as a deny.
    """
    if isinstance(response, (PermissionResultAllow, PermissionResultDeny)):
        return response
    if isinstance(response, dict):
        typed = cast("dict[str, object]", response)
        behavior = typed.get("behavior")
        if behavior == "allow":
            updated_input_raw = typed.get("updated_input")
            updated_permissions_raw = typed.get("updated_permissions")
            return PermissionResultAllow(
                updated_input=cast("dict[str, Any] | None", updated_input_raw),  # pyright: ignore[reportExplicitAny]
                updated_permissions=cast("list[Any] | None", updated_permissions_raw),  # pyright: ignore[reportExplicitAny]
            )
        if behavior == "deny":
            message_raw = typed.get("message", "")
            return PermissionResultDeny(
                message=str(message_raw) if message_raw else "",
                interrupt=bool(typed.get("interrupt", False)),
            )
    if response == "allow":
        return PermissionResultAllow()
    if response == "deny":
        return PermissionResultDeny(message="", interrupt=False)
    return PermissionResultDeny(
        message=f"Unrecognized approval response: {response!r}",
        interrupt=False,
    )
