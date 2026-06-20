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

from dataclasses import dataclass
from typing import Hashable, Sequence, final, override

from claude_agent_sdk import ClaudeAgentOptions
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..ai_thread.postcondition import PostCondition
from ..protocols import Spawnable, Thread
from ..types import ThreadContext


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

    description: str = (
        "Send a prompt to a Claude Agent and receive its final answer."
    )
    """Description used when exposed as a Strands tool."""

    post_conditions: tuple[PostCondition, ...] = ()
    """Validators run against each cycle's result. On failure the thread
    feeds the failure messages back as the next user turn and re-runs, up to
    ``max_attempts``. Empty (default) disables the retry loop. Mirrors
    ``ThreadConfig.post_conditions``."""

    max_attempts: int = 10
    """Maximum number of cycles to satisfy ``post_conditions`` (mirrors
    ``ThreadConfig.max_attempts``). Ignored when ``post_conditions`` is empty."""

    @override
    def to_thread(self) -> ClaudeAgentThread:
        """Produce a fresh ``ClaudeAgentThread`` bound to this template.

        The returned thread holds its own ``ClaudeSDKClient``; the subprocess
        is not spawned until the first cycle runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        ...

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
        ...

    @override
    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...

    @override
    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...


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

    def __init__(self, template: ClaudeAgent) -> None: ...

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
        ...

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
        ...

    async def fork(self) -> Spawnable[[str], str]:
        """Not supported.

        Returns:
            Never returns; always raises.

        Raises:
            NotImplementedError: Always.
        """
        ...

    async def teardown(self) -> None:
        """Disconnect the SDK client and release the subprocess.

        Ensures:
            - Any connected ``ClaudeSDKClient`` is disconnected.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        ...

    @property
    def template(self) -> ClaudeAgent:
        """The template this thread was created from."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the underlying ``ClaudeSDKClient`` is currently connected."""
        ...
