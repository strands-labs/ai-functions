"""Event hierarchy emitted through ``Coordinator.append_event``.

``Event`` is a discriminated-union hierarchy rooted at ``BaseEvent``. Each
concrete subclass carries a distinct ``kind`` string and only the payload
fields that apply to that kind.

System events have ``kind`` values drawn from ``EventKind`` (a ``StrEnum``).
User-defined events use any other string: subclass ``CustomEvent``, set
``kind`` to a stable application-level identifier, and add application fields
that do not redefine event metadata. The default ``Event`` union parses
unknown kinds as generic ``CustomEvent`` instances; consumers that need their
own subclass apply its model or a union containing it explicitly.

Filtering is uniform for both system and custom events — pass any ``kind``
string (``EventKind`` member or plain string) to ``Coordinator.on(kinds=...)``
or ``Coordinator.get_events(kinds=...)``.

Invariants:
    I2.
"""

from __future__ import annotations

import enum
import time
from typing import Annotated, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from strands.types.content import ContentBlock
from strands.types.tools import ToolResultContent, ToolResultStatus

from .ids import EventId, MessageId, ThreadId


class EventKind(enum.StrEnum):
    """Discriminator values for built-in event variants.

    Any string outside this set is valid as a user-defined ``kind``;
    pydantic will route it to ``CustomEvent``.
    """

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    MESSAGE_USER = "message_user"
    MESSAGE_ASSISTANT_START = "message_assistant_start"
    MESSAGE_ASSISTANT_TOKEN = "message_assistant_token"
    MESSAGE_ASSISTANT_THINKING = "message_assistant_thinking"
    MESSAGE_ASSISTANT_COMPLETE = "message_assistant_complete"

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    APPROVAL_REQUEST = "approval_request"
    APPROVAL_DECIDED = "approval_decided"

    SESSION_CREATED = "session_created"
    SESSION_RESET = "session_reset"
    CONTEXT_SUMMARIZED = "context_summarized"

    TOKEN_USAGE = "token_usage"

    RESULT = "result"

    PARAMETER_RECALLED = "parameter_recalled"

    THREAD_SPAWNED = "thread_spawned"
    TRACE_DELEGATION = "trace_delegation"


class TokenUsage(BaseModel):
    """Per-call token accounting reported by an executor."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Return the componentwise sum of two ``TokenUsage`` values.

        Args:
            other: The ``TokenUsage`` to add.

        Returns:
            A new ``TokenUsage`` whose fields are the sums of ``self`` and
            ``other``.
        """
        ...


def _new_id() -> EventId: ...


class BaseEvent(BaseModel):
    """Fields shared by every event appended through ``Coordinator.append_event``.

    Concrete variants set ``kind`` to a distinct string value; downstream
    consumers dispatch on ``kind`` and narrow to the corresponding
    subclass.

    Invariants:
        I2.
    """

    model_config = ConfigDict(frozen=True)

    id: EventId = Field(default_factory=_new_id)
    """Stable event id; never reused. Used as ``since_id`` cursor by
    ``get_events``."""

    timestamp: float = Field(default_factory=time.time)
    """Wall-clock time of emission (``time.time()``)."""

    thread_id: ThreadId | None = None
    """Thread this event belongs to; both session id and subscription key.

    Optional at construction: the worker's ``_route_event`` stamps the
    routing thread id onto every event that flows through the worker
    gate, so callers can omit it and rely on the runtime to fill it in.
    Once an event has been persisted by a ``Coordinator`` (i.e. is
    readable via ``get_events``), this field is guaranteed to be set.
    ``Coordinator.append_event`` rejects unrouted events.
    """

    thread_name: str | None = None
    """Display name for UIs; never consulted for correctness."""

    message_id: MessageId | None = None
    """Ties the fragments of a single assistant turn together."""


# ── Thread lifecycle ──


class StartedEvent(BaseEvent):
    """Dispatcher has begun a cycle on this thread."""

    kind: Literal[EventKind.STARTED] = EventKind.STARTED


class CompletedEvent(BaseEvent):
    """Cycle finished normally with a typed result."""

    kind: Literal[EventKind.COMPLETED] = EventKind.COMPLETED


class FailedEvent(BaseEvent):
    """Cycle raised an uncaught exception."""

    kind: Literal[EventKind.FAILED] = EventKind.FAILED
    error: str
    """``repr`` of the raised exception."""


class CancelledEvent(BaseEvent):
    """Cycle terminated cooperatively or was torn down."""

    kind: Literal[EventKind.CANCELLED] = EventKind.CANCELLED


# ── Conversation content ──


class MessageUserEvent(BaseEvent):
    """A user turn was appended to the conversation."""

    kind: Literal[EventKind.MESSAGE_USER] = EventKind.MESSAGE_USER
    text: str


class MessageAssistantStartEvent(BaseEvent):
    """Assistant turn began; subsequent events share its ``message_id``."""

    kind: Literal[EventKind.MESSAGE_ASSISTANT_START] = EventKind.MESSAGE_ASSISTANT_START


class MessageAssistantTokenEvent(BaseEvent):
    """One streamed text chunk from the assistant.

    Emitted once per ``data`` callback from Strands (i.e. once per
    ``ContentBlockDeltaEvent`` with a text delta). The ``complete`` field
    mirrors Strands' ``complete`` flag; when ``True`` this is the last
    chunk of the current content block.
    """

    kind: Literal[EventKind.MESSAGE_ASSISTANT_TOKEN] = EventKind.MESSAGE_ASSISTANT_TOKEN
    text: str
    complete: bool = False


class MessageAssistantThinkingEvent(BaseEvent):
    """One streamed reasoning/thinking chunk from the assistant.

    Emitted when the model returns a ``reasoningText`` delta via the
    Strands callback handler (extended thinking / CoT models).
    """

    kind: Literal[EventKind.MESSAGE_ASSISTANT_THINKING] = EventKind.MESSAGE_ASSISTANT_THINKING
    text: str
    complete: bool = False


class MessageAssistantCompleteEvent(BaseEvent):
    """Assistant turn finished; closes the span opened by its ``message_id``.

    ``content`` holds the full Strands-format content block list for the
    turn, as returned by ``AfterModelCallEvent.stop_response.message``.
    """

    kind: Literal[EventKind.MESSAGE_ASSISTANT_COMPLETE] = EventKind.MESSAGE_ASSISTANT_COMPLETE
    content: list[ContentBlock] = Field(default_factory=list[ContentBlock])
    """Strands content blocks for the completed turn."""


# ── Tool activity ──


class ToolCallEvent(BaseEvent):
    """A tool call is about to run, with its resolved arguments.

    Emitted once per tool invocation just before the tool executes. The
    full arguments dict is already materialized by the time Strands fires
    ``BeforeToolCallEvent``. ``toolUse`` blocks in the preceding assistant
    turn (``MessageAssistantCompleteEvent.content``) are what the
    reconstructed message history relies on; this event is the
    observability companion that tools and UIs subscribe to.
    """

    kind: Literal[EventKind.TOOL_CALL] = EventKind.TOOL_CALL
    tool_use_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolResultEvent(BaseEvent):
    """A tool call finished — either successfully or with an error.

    Both outcomes share this event kind; ``status`` discriminates. The
    field layout mirrors Strands'
    :class:`strands.types.tools.ToolResult` (``{toolUseId, status,
    content}``) at the top level of this event so consumers get typed
    attribute access instead of string-keyed dict lookups.

    On reconstruction, these three fields are packed back into a
    ``ToolResult`` dict and splatted into a ``{"toolResult": ...}``
    ``ContentBlock`` — the same shape Strands itself produces in its
    event loop — so the cache prefix (I9) stays byte-identical.
    """

    kind: Literal[EventKind.TOOL_RESULT] = EventKind.TOOL_RESULT
    tool_use_id: str
    """Id of the ``toolUse`` block this result corresponds to."""
    status: ToolResultStatus
    """``"success"`` or ``"error"``, matching Strands' ``ToolResult.status``."""
    content: list[ToolResultContent] = Field(default_factory=list[ToolResultContent])
    """Strands-format content blocks produced by the tool."""


# ── Approvals ──


class ApprovalRequestEvent(BaseEvent):
    """Executor raised an interrupt requesting human approval for a tool call."""

    kind: Literal[EventKind.APPROVAL_REQUEST] = EventKind.APPROVAL_REQUEST
    approval_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ApprovalDecidedEvent(BaseEvent):
    """A pending approval was resolved with a policy decision."""

    kind: Literal[EventKind.APPROVAL_DECIDED] = EventKind.APPROVAL_DECIDED
    approval_id: str
    decision: str


# ── Session management ──


class SessionCreatedEvent(BaseEvent):
    """A new thread / session was registered with the orchestrator."""

    kind: Literal[EventKind.SESSION_CREATED] = EventKind.SESSION_CREATED


class SessionResetEvent(BaseEvent):
    """Session state for this thread was reset by the orchestrator."""

    kind: Literal[EventKind.SESSION_RESET] = EventKind.SESSION_RESET


class ContextSummarizedEvent(BaseEvent):
    """Boundary in the event log: compacted history replaces everything before.

    Emitted by the thread when its cumulative history is compacted
    (either proactively or reactively on a
    ``ContextWindowOverflowException``). The event marks a
    cache-invalidation point for I9: on the next reconstruction, every
    event strictly before this one (in append order) is dropped from the
    rendered history and ``new_history`` is rendered in its place.
    Events strictly after render normally.

    The payload is a synthetic sequence of conversational events — the
    same event shapes the thread would normally emit during a cycle —
    constructed by the :class:`SummarizationStrategy`.

    Invariants:
        - I9 — this event is the canonical cache-invalidation marker.
        - I10 — the strategy is responsible for producing a history whose
          ``toolUse`` / ``toolResult`` pairs are legal; the reconstruction
          healer is a safety net, not the primary contract.
    """

    kind: Literal[EventKind.CONTEXT_SUMMARIZED] = EventKind.CONTEXT_SUMMARIZED

    new_history: list[RenderableEvent] = Field(default_factory=list["RenderableEvent"])
    """The compacted history that replaces every event before this one.

    Rendered by ``reconstruct_messages`` in place of the preceding event
    sequence. The events are interpreted exactly as if they had been
    emitted during a normal cycle.

    Requires:
        - The first event (when present) has ``role == "user"`` semantics
          in the rendered view — most providers reject a message sequence
          that starts with an assistant turn.
        - Every ``toolUse`` block reachable through these events has a
          matching ``toolResult`` event later in ``new_history`` or gets
          healed by :func:`reconstruct_messages` (I10).
    """


# ── Resource usage ──


class TokenUsageEvent(BaseEvent):
    """Token usage reported for one model call."""

    kind: Literal[EventKind.TOKEN_USAGE] = EventKind.TOKEN_USAGE
    token_usage: TokenUsage
    model: str | None = None
    """The id of the model that spent ``token_usage``, as the thread's provider names it; ``None`` when unknown.

    A price table prices the usage by this id (:mod:`~ai_functions.runtime.prices`).
    """


# ── Result ──


class ResultEvent(BaseEvent):
    """The serialized result of a completed cycle.

    Emitted by the runtime dispatcher immediately after any successful
    cycle, before the matching ``CompletedEvent``. The payload is produced
    by ``Thread.serialize_result`` and can be recovered by calling
    ``Thread.deserialize_result(event.payload)``.

    Invariants:
        I5 — lifecycle events (``STARTED``, ``COMPLETED``, ``CANCELLED``,
        ``FAILED``, ``RESULT``) are emitted only by the runtime dispatcher
        that drives ``Thread.execute``/``resume``. ``Thread``
        implementations must not emit them directly.
    """

    kind: Literal[EventKind.RESULT] = EventKind.RESULT
    payload: str
    """Thread-serialized result; opaque to the runtime and coordinator."""


# ── Memory / optimization ──


class ParameterRecalledEvent(BaseEvent):
    """A memory parameter was recalled into a thread's execution.

    Emitted directly by ``MemoryBackend.recall`` / ``query`` / ``search`` the
    moment the read happens, stamped with the caller-supplied ``thread_id``.
    Because ``append_event`` creates a thread's log on demand, this event may
    be appended *before* the named thread spawns; the event is simply waiting
    in the log when the cycle starts.

    Consumed only by :func:`build_graph`, which matches ``backend_id`` back to
    a live backend to rebuild a ``ParameterNode``. It is **not** a
    :data:`RenderableEvent`: ``reconstruct_messages`` filters it out, so it
    never contributes a message and never shifts a summarization boundary or
    invalidates the prompt cache (I9 — non-renderable events are inert to
    reconstruction).

    Invariants:
        - I9 — non-renderable; inert to message reconstruction.
    """

    kind: Literal[EventKind.PARAMETER_RECALLED] = EventKind.PARAMETER_RECALLED
    name: str
    """Parameter name (supports nested ``a/b/c`` paths)."""
    value: object = None
    """Serialized recalled value; deserialized via ``backend.deserialize_value``."""
    derivation: Literal["full", "query", "search"] = "full"
    """How the value was produced: full recall, LLM query, or top-k search."""
    requires_grad: bool = True
    """Whether the optimizer may propagate feedback into this parameter."""
    backend_id: str = ""
    """``"ClassName:actor_id"`` identifying the originating backend."""
    description: str = ""
    """Human-readable description carried from the schema field."""
    meta: dict[str, object] = Field(default_factory=dict[str, object])
    """Backend-specific data (e.g. query text, top_k, scores)."""


class ThreadSpawnedEvent(BaseEvent):
    """A child thread was spawned from this one, recording the parent→child edge.

    Emitted into the **parent's** log by ``Coordinator.spawn`` whenever a
    ``parent_id`` is set, so the edge outlives the child's teardown (which drops
    the child's ``ThreadInfo`` but not its event log). Consumed only by
    :func:`build_graph`, which recurses into ``child_thread_id``. Not a
    :data:`RenderableEvent`.

    Invariants:
        - I9 — non-renderable; inert to message reconstruction.
    """

    kind: Literal[EventKind.THREAD_SPAWNED] = EventKind.THREAD_SPAWNED
    child_thread_id: ThreadId
    """Id of the spawned child thread; the key ``build_graph`` recurses on."""


class TraceDelegationEvent(BaseEvent):
    """This thread's conversation is delegated to one of its children.

    Emitted by a supervisor thread that runs no model itself but selects a child
    that does (an economic search, a retry wrapper). At graph-build time
    :func:`build_graph` splices the named child's messages onto this node, so
    the backward pass reads the real conversation, not the supervisor's
    telemetry. Not a :data:`RenderableEvent`.

    Invariants:
        - I9 — non-renderable; inert to single-thread message reconstruction.
    """

    kind: Literal[EventKind.TRACE_DELEGATION] = EventKind.TRACE_DELEGATION
    child_thread_id: ThreadId
    """Id of the child whose messages this node adopts at graph-build time."""


# ── User-defined extension ──


class CustomEvent(BaseEvent):
    """Catch-all event for user-defined ``kind`` values.

    Carries the same routing fields as every other event (``id``,
    ``timestamp``, ``thread_id``, ``thread_name``, ``message_id``), so routing
    survives the wire: a subscription filtered by ``thread_id`` matches a
    custom event that arrived over a ``CoordinatorClient``, ``get_events``
    returns it under its own thread, and ``Coordinator.append_event`` accepts
    it from a client that stamped the id itself.

    A ``mode="before"`` validator reshapes flat input dicts: all keys other
    than declared model fields land inside ``payload``. A ``model_serializer``
    flattens on the way out, emitting the declared fields next to the
    payload's entries, so the wire format round-trips through pydantic.

    A top-level key that names a declared field binds to that field, so
    ``CustomEvent(kind="k", thread_id=tid)`` routes rather than filling
    ``payload``. Payload entries must not shadow any declared field or field
    alias, including ``kind`` and ``payload`` itself. Conflicts raise an error;
    rename application fields (for example, ``item_id``) or nest them under an
    application key (for example, ``payload={"item": {"id": ...}}``). The wire
    representation stays flat; the serializer never nests conflicting keys.

    Subclasses may declare typed application fields and specialize ``kind``.
    They must not redefine ``BaseEvent`` fields or alias application fields
    onto the routing fields, ``kind``, or ``payload``. These names retain their
    framework meanings, including when the model is serialized with aliases.

    ``BaseEvent`` is frozen, so an instance is immutable; build a routed copy
    with ``model_copy(update={"thread_id": ...})``.

    If ``payload`` is provided explicitly, any extra top-level keys are
    merged into it (extras take precedence over explicit-payload entries
    that have the same key). Known declared fields on subclasses (if any)
    are preserved as-is and not swept into ``payload``.

    Invariants:
        I2.
    """

    kind: str
    payload: dict[str, object] = Field(default_factory=dict[str, object])


# ── Discriminated union ──

SystemEvent = Annotated[
    StartedEvent | CompletedEvent | FailedEvent | CancelledEvent | MessageUserEvent |
    MessageAssistantStartEvent | MessageAssistantTokenEvent | MessageAssistantThinkingEvent |
    MessageAssistantCompleteEvent |
    ToolCallEvent | ToolResultEvent |
    ApprovalRequestEvent | ApprovalDecidedEvent | SessionCreatedEvent | SessionResetEvent | ContextSummarizedEvent |
    TokenUsageEvent | ResultEvent | ParameterRecalledEvent | ThreadSpawnedEvent | TraceDelegationEvent,
    Field(discriminator="kind")
]

Event = Annotated[SystemEvent | SerializeAsAny[CustomEvent], Field(union_mode="left_to_right")]
"""Tagged union of every built-in event variant plus the ``CustomEvent`` fallback.

Pydantic tries union members left-to-right: the discriminated ``SystemEvent``
union is a single fast-path attempt on ``kind``, and only a kind that matches
no built-in variant falls through to ``CustomEvent``, whose
``model_validator(mode="before")`` reshapes the raw dict into
``{kind, routing fields, payload}`` — every key outside ``CustomEvent``'s
declared fields lands in ``payload``, so an unknown kind never loses data
and never loses its routing.

``union_mode="left_to_right"`` is what keeps a built-in kind out of
``CustomEvent``: smart mode tie-breaks two matching members by the number of
fields set, and ``CustomEvent`` declares every routing field, so it would
outscore the concrete variant a built-in kind belongs to and swallow it.

Users can write their own union to add correct parsing of their own
event types.

``SerializeAsAny`` makes the custom branch serialize the concrete model's fields
and serializers, including inside ``Event``-typed fields and containers. RPC
parameters and session logs inherit this policy without per-call flags. Input
validation still uses the declared union: a peer that does not know the subclass
receives a generic ``CustomEvent`` with application fields collected in ``payload``.
"""


# ── Renderable subset ──

RenderableEvent = (
    MessageUserEvent
    | MessageAssistantCompleteEvent
    | ToolCallEvent
    | ToolResultEvent
)
"""Events that may appear inside a ``ContextSummarizedEvent.new_history``.

Restricted to the event kinds that carry conversational content:

- :class:`MessageUserEvent`, :class:`MessageAssistantCompleteEvent` —
  turns.
- :class:`ToolCallEvent`, :class:`ToolResultEvent` — tool activity.
  ``ToolCallEvent`` is observability-only but included for completeness:
  a strategy may replay it to preserve the original timeline in a UI,
  even though it contributes no message.

Explicitly excluded:

- Lifecycle events (``STARTED`` / ``COMPLETED`` / ``FAILED`` /
  ``CANCELLED``), ``ResultEvent``, ``TokenUsageEvent``, session events,
  approval events.
- Streaming-fragment events — aggregated by
  ``MessageAssistantCompleteEvent``.
- ``ContextSummarizedEvent`` itself.
- ``CustomEvent`` — inert to reconstruction.
"""


def is_renderable_event(event: Event) -> TypeGuard[RenderableEvent]:
    """Return whether ``event`` is a :data:`RenderableEvent`.

    Self-maintaining: the concrete types are derived from
    :data:`RenderableEvent` via :func:`typing.get_args`, so widening the
    union automatically widens this guard.

    Args:
        event: Any :data:`Event` instance.

    Returns:
        ``True`` iff ``event`` is an instance of one of the union's
        concrete variants.
    """
    ...
