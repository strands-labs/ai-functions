"""``CodexAgent`` template and ``CodexAgentThread`` — OpenAI Codex-backed thread.

A ``CodexAgentThread`` drives a ``codex app-server`` subprocess through the
OpenAI Codex Python SDK. The app-server owns the conversation transcript;
ai_functions observes its notification stream and re-emits each element as a
ai_functions event — pure observability, not a source of truth. The
implementation module's docstring carries the full Codex-to-event mapping.

Distinguishing capabilities relative to the Claude and Kiro backends:

- ``fork()`` is real — Codex forks the stored conversation server-side and
  the returned template resumes the fork.
- ``notify()`` reaches a *running* turn by steering it, not only the next one.
- Cooperative cancel interrupts the in-flight turn rather than waiting for
  the cycle boundary.
- The template is meaningfully picklable: session identity is a string
  (``resume_thread_id``).

Approvals: the public SDK exposes only ``ApprovalMode.auto_review`` /
``deny_all`` — there is no human-in-the-loop callback, so approvals are not
routed through ``ctx.on_interrupt``.

Requires the ``codex`` extra (``openai-codex``, which bundles the runtime).
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import final

from openai_codex import ApprovalMode, Sandbox
from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import Personality, ReasoningEffort, ReasoningSummary
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..ai_thread.postcondition import PostCondition
from ..protocols import Spawnable, Thread
from ..types import InputShape, ThreadContext

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

    approval_mode: ApprovalMode = ...
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

    description: str = ...
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
        ...

    def to_thread(self) -> CodexAgentThread:
        """Produce a fresh ``CodexAgentThread`` bound to this template.

        The returned thread owns its own ``AsyncCodex`` client; the
        ``codex app-server`` subprocess is not spawned until the first cycle
        runs.

        Ensures:
            - Successive calls return independent instances with no shared state.
            - No subprocess is started by this call.
        """
        ...

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
        ...

    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...

    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...

@final
class CodexAgentThread(Thread[[str], str]):
    """Live Codex-backed thread that owns one ``AsyncCodex`` client.

    Connects lazily on the first cycle: launches ``codex app-server`` and
    starts (or resumes) one conversation thread, both kept for the thread's
    lifetime. The app-server owns conversation history; ai_functions observes
    the notification stream and re-emits each element as a ai_functions
    event — pure observability, not a source of truth.

    Implements:
        Thread[[str], str].

    Lifecycle:
        CREATED → CONNECTED → CLOSED.
    """

    def __init__(self, template: CodexAgent) -> None: ...
    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``CodexAgent`` template."""
        ...

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
        ...

    async def execute(self, ctx: ThreadContext, prompt: str) -> str:
        """Send ``prompt`` to the Codex thread and return its string result.

        Runs the shared post-condition loop: the inject buffer is drained
        into the outgoing turn, failures are fed back as the next user turn,
        and the notification stream of each turn is re-emitted per the
        mapping table in the implementation module's docstring.

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
        ...

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
        ...

    async def teardown(self) -> None:
        """Close the SDK client and release the ``codex app-server`` subprocess.

        Ensures:
            - Any running ``AsyncCodex`` client is closed.
            - In-flight steers are cancelled and awaited.
            - Pending inject-buffer entries are dropped.

        Concurrency:
            Idempotent; tearing down a never-connected thread is a no-op.
        """
        ...

    def serialize_result(self, result: str) -> str:
        """Return ``result`` unchanged; Codex results are already strings."""
        ...

    def deserialize_result(self, payload: str) -> str:
        """Return ``payload`` unchanged; Codex results are already strings."""
        ...

    @property
    def template(self) -> CodexAgent:
        """The template this thread was created from."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the app-server is running with a live conversation thread."""
        ...

    @property
    def codex_thread_id(self) -> str | None:
        """The Codex-side conversation thread id once connected, or ``None``."""
        ...
