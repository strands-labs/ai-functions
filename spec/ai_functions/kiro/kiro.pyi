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

from dataclasses import dataclass
from typing import Hashable, Sequence, final, override

from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..ai_thread.postcondition import PostCondition
from ..protocols import Spawnable, Thread
from ..types import ThreadContext


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

    description: str = (
        "Send a prompt to a Kiro agent and receive its final answer."
    )
    """Description used when exposed as a Strands tool."""

    post_conditions: tuple[PostCondition, ...] = ()
    """Validators run against each cycle's result. On failure the thread feeds
    the failure messages back as the next user turn and re-runs, up to
    ``max_attempts``. Empty (default) disables the retry loop."""

    max_attempts: int = 10
    """Maximum number of cycles to satisfy ``post_conditions``. Ignored when
    ``post_conditions`` is empty."""

    @override
    def to_thread(self) -> KiroAgentThread:
        """Produce a fresh ``KiroAgentThread`` bound to this template.

        The returned thread owns its own ACP session; the ``kiro-cli acp``
        subprocess is not spawned until the first cycle runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        ...

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

    def __init__(self, template: KiroAgent) -> None: ...

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
        ...

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
        """Close the ACP session and release the subprocess.

        Ensures:
            - The ACP session and its ``kiro-cli acp`` subprocess are closed.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        ...

    @property
    def template(self) -> KiroAgent:
        """The template this thread was created from."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the underlying ACP session is currently connected."""
        ...

    @property
    def session_id(self) -> str | None:
        """The ACP session id once connected, or ``None``."""
        ...
