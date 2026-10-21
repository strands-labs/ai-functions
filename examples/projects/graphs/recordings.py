"""Read workflow recordings tolerantly and derive the viewer's models.

The graph runtime writes each run as plain files under ``<root>/<slug>/runs/<run-id>/``
(``run.json``, ``manifest.json``, ``events.jsonl``, ``result.json``). This module is the
read-only other half of that contract, in three layers:

- **Schema** — pydantic models for what the producer writes. Every field is defaulted
  and unknown keys are ignored, so recordings from older or newer runtimes still load.
- **Parse** — folds a run's raw event log into per-thread views and condition tickets.
- **Board** — discovers every run under a folder and caches parses between scans.

Nothing here writes to disk or talks to a running workflow.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

RUNTIME_VERSION = 4
"""Runtime version this reader is written against; older packages get a warning flag."""

PENDING = "pending"
"""Status of a step the manifest declares but the log has not reached: drawn, not yet started."""

UNRUN = "unrun"
"""Status of a package that has a ``workflow.py`` but no recorded run yet."""

BLOCKED = "blocked"
"""Status of a step whose pre-conditions failed, so its agent never started."""

SKIPPED = "skipped"
"""Status of a step a finished run never reached, because a dependency did not pass."""


# ── Schema: what the producer writes ──────────────────────────────────────────


class EventKind(enum.StrEnum):
    """Event kinds the workflow runtime adds on top of the library's own.

    The runtime emits these as ``CustomEvent``s with distinct ``aif_*`` kinds rather than
    reusing library lifecycle kinds, which only the dispatcher may emit.
    """

    PRECONDITION = "aif_precondition"
    POSTCONDITION = "aif_postcondition"
    STEP_STARTED = "aif_step_started"
    STEP_FINISHED = "aif_step_finished"


class ConditionSpec(BaseModel):
    """A condition as the manifest declares it, including its source.

    The runtime introspects each condition callable when a run starts, so these fields
    describe the code that actually ran: its docstring, signature and source text.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = "?"
    kind: str = "code"
    description: str = Field(default="", description="First docstring line")
    docstring: str = Field(default="", description="Full docstring")
    signature: str = Field(default="", description="`name(args) -> return` as introspected")
    source: str = Field(default="", description="The condition's own source, for the detail view")


class StepSpec(BaseModel):
    """A step as the manifest declares it.

    One entry of ``manifest.json``'s ``steps`` list: a graph node, its prompt, its
    conditions and the names of the steps it depends on (its incoming edges).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = "?"
    prompt: str = ""
    thread_id: str = ""
    backend: str = Field(default="", description="What ran the step: kiro, claude, codex, ai_thread or ai_function")
    input_type: str = Field(default="", description="What the thread takes; a Kiro thread takes a prompt")
    output_type: str = Field(default="", description="What the thread returns")
    pre_conditions: list[ConditionSpec] = Field(default_factory=list)
    post_conditions: list[ConditionSpec] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    max_attempts: int = 0


class Manifest(BaseModel):
    """A run's graph snapshot, or the package manifest for legacy recordings.

    Runtime version 4 writes ``runs/<run-id>/manifest.json`` per run, so later edits to
    a graph do not change how earlier runs display. Older runs fall back to the shared
    ``<slug>/manifest.json``, which reflects only the latest run.
    """

    model_config = ConfigDict(extra="ignore")

    runtime_version: int = 0
    slug: str = ""
    task: str = ""
    objective: str = ""
    workspace: str = ""
    backend: str = Field(default="", description="The run's default backend")
    steps: list[StepSpec] = Field(default_factory=list)

    @property
    def steps_by_name(self) -> dict[str, StepSpec]:
        """Steps keyed by name, for joining against parsed threads."""
        return {step.name: step for step in self.steps}


class RunMeta(BaseModel):
    """``<root>/<slug>/runs/<run-id>/run.json`` — one run's metadata.

    Written once with status ``running`` when the run starts and rewritten when it
    closes, so a ``running`` status with no finish time means live or interrupted.
    """

    model_config = ConfigDict(extra="ignore")

    runtime_version: int = 0
    run_id: str = ""
    slug: str = ""
    task: str = ""
    session: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    status: str = "running"
    backend: str = Field(default="", description="The run's default backend; empty for runs older than it")

    @property
    def stale_runtime(self) -> bool:
        """Whether this run was written by a runtime older than the reader expects."""
        return self.runtime_version < RUNTIME_VERSION


# ── Parse: raw events to a view model ─────────────────────────────────────────

# The library's native post-condition feedback turn, used as a fallback when a
# condition was not wrapped by the runtime's ``observed``.
_FEEDBACK = re.compile(
    r"^\[(?P<step>[^\]]+)\] Post-condition failures \(attempt (?P<attempt>\d+)/(?P<total>\d+)\):",
)

_ASSISTANT_FRAGMENTS = {
    "message_assistant_start",
    "message_assistant_token",
    "message_assistant_thinking",
}


class ConditionCell(BaseModel):
    """One condition evaluated on one attempt: a cell of the matrix view.

    Built from an ``aif_precondition`` / ``aif_postcondition`` event, or inferred from
    the library's native feedback turn when the condition was not wrapped by the runtime.
    """

    condition: str
    attempt: int
    passed: bool
    message: str | None = None
    kind: str = "code"
    timestamp: float = 0.0
    inferred: bool = Field(
        default=False,
        description="True when recovered from a native feedback turn rather than an emitted event",
    )


class TimelineEntry(BaseModel):
    """One mark on a thread's trajectory lane.

    A user turn, an assistant turn, a tool call or result, or any library event kind
    this module does not otherwise interpret.
    """

    kind: str
    timestamp: float
    label: str = ""
    detail: str = ""
    message_id: str | None = None
    attempt: int | None = None


class ToolActivity(BaseModel):
    """One tool call paired with its result — what the agent actually *did*.

    Created from a ``tool_call`` event and completed in place when the matching
    ``tool_result`` arrives; until then its ``status`` stays ``pending``.
    """

    tool: str
    timestamp: float
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(default="", description="Kiro's own __tool_use_purpose, when it sends one")
    status: str = Field(default="pending", description="pending until its result arrives")
    output: str = ""
    duration: float | None = None
    attempt: int = Field(default=1, description="Which post-condition attempt this call belongs to")


class ThreadView(BaseModel):
    """Everything the viewer shows for one thread (one workflow step).

    Accumulated event by event in :func:`parse_run`; it carries no manifest data, which
    :func:`thread_cards` joins in afterwards.
    """

    thread_id: str
    name: str
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "running"
    pre_conditions: list[ConditionCell] = Field(default_factory=list)
    post_conditions: list[ConditionCell] = Field(default_factory=list)
    tools: list[ToolActivity] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    attempts: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    result: str | None = None


class RunView(BaseModel):
    """A whole parsed run: its threads, plus roll-ups a headline uses.

    The output of :func:`parse_run`. Timestamps span the first and last event seen,
    across all threads.
    """

    run_id: str
    threads: list[ThreadView] = Field(default_factory=list)
    event_count: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    usage_reported: bool = Field(
        default=False,
        description="False when the agent reported no usage, so the viewer can say so instead of showing zeros",
    )


class ConditionTicket(BaseModel):
    """One condition's standing on a thread: the board's unit of verification.

    Collapses every evaluation of one condition into its latest verdict, joined with
    the condition's declaration so the viewer can show its source on demand.
    """

    name: str
    kind: str = "code"
    stage: str = Field(default="post", description="'pre' or 'post'")
    passed: bool = Field(description="Whether its most recent evaluation passed")
    attempts: int = Field(default=1, description="How many attempts it took, or has taken so far")
    message: str | None = Field(default=None, description="Latest failure message, if any")
    description: str = ""
    docstring: str = ""
    signature: str = ""
    source: str = Field(default="", description="The condition's source, shown when a ticket is opened")
    inferred: bool = False


class ThreadCard(BaseModel):
    """A single AI thread as the board shows it, flattened across runs.

    A :class:`ThreadView` joined with its manifest :class:`StepSpec` and run metadata.
    The viewer receives these as plain dicts via ``model_dump``.
    """

    thread_id: str
    name: str
    run_id: str
    slug: str
    session: str | None = None
    task: str = ""
    status: str = "running"
    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 0
    tool_calls: int = 0
    event_count: int = 0
    tickets: list[ConditionTicket] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    prompt: str = Field(default="", description="The prompt this thread was given")
    backend: str = Field(default="", description="What ran this thread, from the manifest")
    input_type: str = ""
    output_type: str = ""

    @property
    def verified(self) -> bool:
        """Whether every ticket on this thread is currently passing."""
        return all(ticket.passed for ticket in self.tickets)


def parse_run(run_id: str, events: list[dict[str, Any]]) -> RunView:
    """Fold raw events into a :class:`RunView`, one lane per thread.

    Events are applied in log order. A thread is created the first time its id appears,
    and its display name follows the latest ``thread_name`` seen.

    Args:
        run_id: Id of the run the events belong to.
        events: Decoded ``events.jsonl`` lines, in the order they were written.

    Returns:
        The run's threads, sorted by start time and then name, with run-level timestamps
        and whether any token usage was reported.
    """
    view = RunView(run_id=run_id, event_count=len(events))
    threads: dict[str, ThreadView] = {}
    # tool_use_id → tool name, so a result can name the call it answers.
    tools: dict[str, str] = {}

    # Pass 1: route each event to its thread and fold it in.
    for event in events:
        if not event.get("thread_id"):
            # Agent-level chatter with no thread — Claude Code's ``init`` and
            # ``commands_changed`` system messages arrive this way. It belongs to no
            # step, so it must not become a phantom thread on the board.
            continue
        thread_id = str(event["thread_id"])
        thread = threads.setdefault(
            thread_id,
            ThreadView(thread_id=thread_id, name=str(event.get("thread_name") or thread_id)),
        )
        if event.get("thread_name"):
            thread.name = str(event["thread_name"])
        # Widen the run's time span to cover this event.
        timestamp = float(event.get("timestamp") or 0.0)
        view.started_at = timestamp if view.started_at is None else min(view.started_at, timestamp)
        view.finished_at = timestamp if view.finished_at is None else max(view.finished_at, timestamp)
        _fold(thread, str(event.get("kind") or ""), event, timestamp, view, tools)

    # Pass 2: per-thread roll-ups that need the whole log.
    for thread in threads.values():
        # The feedback-turn fallback is for conditions that were never wrapped by
        # the runtime. When a thread has real condition events, the inferred cells
        # are the same failures seen a second time through prose — drop them, or
        # every failure shows up twice, once named after its own message.
        if any(not cell.inferred for cell in thread.post_conditions):
            thread.post_conditions = [c for c in thread.post_conditions if not c.inferred]
        thread.attempts = max([cell.attempt for cell in thread.post_conditions] or [0])
        # A failed pre-condition stops the step before its agent starts, so no lifecycle
        # event ever replaces the default status; say why it never ran instead.
        if thread.started_at is None and any(not cell.passed for cell in thread.pre_conditions):
            thread.status = BLOCKED
    view.threads = sorted(threads.values(), key=lambda t: (t.started_at or 0.0, t.name))
    return view


def thread_cards(
    view: RunView,
    *,
    slug: str,
    task: str = "",
    session: str | None = None,
    steps: dict[str, StepSpec] | None = None,
) -> list[ThreadCard]:
    """Flatten a parsed run into one card per thread, with condition tickets.

    Threads the log has reached are joined with their manifest step; declared steps the
    log has not reached yet become ``pending`` placeholders, so the whole graph is drawn
    from the first event.

    Args:
        view: The parsed run.
        slug: Workflow package name the run belongs to.
        task: The run's task text, copied onto each card.
        session: Label of the terminal or process that started the run.
        steps: Manifest steps keyed by name; empty for runs with no manifest.

    Returns:
        One card per thread, in manifest declaration order when a manifest is given.
    """
    declared = steps or {}
    # Cards for threads that appear in the log.
    cards: list[ThreadCard] = []
    seen: set[str] = set()
    for thread in view.threads:
        spec = declared.get(thread.name) or StepSpec(name=thread.name)
        seen.add(thread.name)
        cards.append(_card(thread, spec, view.run_id, slug=slug, task=task, session=session))
    # Steps the manifest declares but the log has not reached yet: pending, so the
    # whole graph is on screen from the first event, not thread by thread.
    for name, spec in declared.items():
        if name not in seen:
            placeholder = ThreadView(thread_id=spec.thread_id or f"th-{name}", name=name, status=PENDING)
            cards.append(_card(placeholder, spec, view.run_id, slug=slug, task=task, session=session))
    if declared:  # manifest order is the workflow's declared order
        order = {name: index for index, name in enumerate(declared)}
        cards.sort(key=lambda card: order.get(card.name, len(order)))
    return cards


def _card(thread: ThreadView, spec: StepSpec, run_id: str, *, slug: str, task: str, session: str | None) -> ThreadCard:
    """Join one parsed thread with what the manifest declares about its step.

    Args:
        thread: What the event log says happened on the thread.
        spec: What the manifest declares about the step; defaulted when undeclared.
        run_id: Id of the run the thread belongs to.
        slug: Workflow package name.
        task: The run's task text.
        session: Label of the terminal or process that started the run.

    Returns:
        The board card for the thread.
    """
    return ThreadCard(
        thread_id=thread.thread_id,
        name=thread.name,
        run_id=run_id,
        slug=slug,
        session=session,
        task=task,
        status=thread.status,
        started_at=thread.started_at,
        finished_at=thread.finished_at,
        attempts=thread.attempts,
        tool_calls=thread.tool_calls,
        event_count=len(thread.timeline),
        depends_on=list(spec.depends_on),
        prompt=spec.prompt,
        backend=spec.backend,
        input_type=spec.input_type,
        output_type=spec.output_type,
        tickets=_tickets(thread, spec),
    )


def _tickets(thread: ThreadView, spec: StepSpec) -> list[ConditionTicket]:
    """Collapse a thread's condition cells into one ticket per condition.

    A ticket's verdict is its latest evaluation (highest attempt, then latest time).
    Declared conditions that were never evaluated still get a ticket, marked
    ``not evaluated`` with zero attempts.

    Args:
        thread: The thread whose condition cells are collapsed.
        spec: The step's declaration, supplying descriptions and source.

    Returns:
        Pre-condition tickets first, then post-condition tickets.
    """
    tickets: list[ConditionTicket] = []
    stages = (
        ("pre", thread.pre_conditions, spec.pre_conditions),
        ("post", thread.post_conditions, spec.post_conditions),
    )
    for stage, cells, specs in stages:
        # Index declarations and evaluations by condition name.
        declared = {spec_item.name: spec_item for spec_item in specs}
        by_name: dict[str, list[ConditionCell]] = {}
        for cell in cells:
            by_name.setdefault(cell.condition, []).append(cell)
        # Evaluated conditions retain event order; pending declarations follow.
        for name in dict.fromkeys((*by_name, *declared)):
            group = by_name.get(name, [])
            latest = max(group, key=lambda c: (c.attempt, c.timestamp), default=None)
            declaration = declared.get(name, ConditionSpec())
            # Only a failing latest evaluation carries a message.
            message: str | None = "not evaluated"
            if latest is not None:
                message = latest.message if not latest.passed else None
            tickets.append(
                ConditionTicket(
                    name=name,
                    kind=latest.kind if latest else declaration.kind,
                    stage=stage,
                    passed=latest.passed if latest else False,
                    attempts=(max(c.attempt for c in group) or 1) if group else 0,
                    message=message,
                    description=declaration.description,
                    docstring=declaration.docstring,
                    signature=declaration.signature,
                    source=declaration.source,
                    inferred=latest.inferred if latest else False,
                ),
            )
    return tickets


def _fold(
    thread: ThreadView,
    kind: str,
    event: dict[str, Any],
    ts: float,
    view: RunView,
    tools: dict[str, str],
) -> None:
    """Apply one event to a thread's view. One branch per event family.

    Field names follow the library's event models exactly (``tool_name``,
    ``tool_use_id``, ``token_usage``, ``payload``); a guessed name reads as silently
    empty rather than failing.

    Args:
        thread: The thread the event belongs to; updated in place.
        kind: The event's ``kind`` string.
        event: The decoded event.
        ts: The event's timestamp.
        view: The whole run; updated in place when usage is reported.
        tools: Map of ``tool_use_id`` to tool name, shared across the run.
    """
    if kind in _ASSISTANT_FRAGMENTS:
        return  # Fragments are represented by their MESSAGE_ASSISTANT_COMPLETE aggregate.

    # Lifecycle: emitted only by the library's dispatcher.
    if kind == "started":
        thread.started_at = thread.started_at or ts
        thread.status = "running"
    elif kind in {"completed", "failed", "cancelled"}:
        thread.finished_at = ts
        thread.status = {"completed": "passed", "failed": "failed", "cancelled": "cancelled"}[kind]
    elif kind == "result":
        # ``ResultEvent.payload`` — the thread-serialized result, not a "result" key.
        thread.result = _text(event, "payload")
    elif kind == "message_user":
        # A user turn may be the library's post-condition feedback; recover failures from it.
        text = _text(event, "text") or ""
        cell = _feedback_cell(text, ts)
        if cell is not None:
            thread.post_conditions.extend(cell)
        thread.timeline.append(
            TimelineEntry(kind="user", timestamp=ts, label="user turn", detail=text[:2000]),
        )
    elif kind == "message_assistant_complete":
        # ``MessageAssistantCompleteEvent`` carries Strands ``content`` blocks and no
        # ``text`` field at all — pydantic silently drops a ``text=`` kwarg, so
        # reading one here would show every assistant turn as empty.
        thread.timeline.append(
            TimelineEntry(
                kind="assistant",
                timestamp=ts,
                label="assistant",
                detail=_content_text(event.get("content")) or (_text(event, "text") or ""),
                message_id=_opt_str(event.get("message_id")),
            ),
        )
    elif kind == "tool_call":
        # Open a pending activity and remember the call's name for its result.
        thread.tool_calls += 1
        name = str(event.get("tool_name") or "tool")
        if use_id := _opt_str(event.get("tool_use_id")):
            tools[use_id] = name
        arguments = event.get("arguments")
        typed_arguments = cast("dict[str, Any]", arguments) if isinstance(arguments, dict) else {}
        thread.tools.append(
            ToolActivity(
                tool=name,
                timestamp=ts,
                # Kiro packs its own rationale into the arguments; lift it out so the
                # pane can lead with *why* rather than with a wall of parameters.
                purpose=str(typed_arguments.get("__tool_use_purpose") or ""),
                arguments={k: v for k, v in typed_arguments.items() if k != "__tool_use_purpose"},
                # Calls after N evaluated attempts belong to attempt N + 1.
                attempt=max(1, len({c.attempt for c in thread.post_conditions}) + 1) if thread.post_conditions else 1,
            ),
        )
        thread.timeline.append(
            TimelineEntry(
                kind="tool_call",
                timestamp=ts,
                label=name,
                detail=_json_ish(event.get("arguments")),
                message_id=_opt_str(event.get("message_id")),
            ),
        )
    elif kind == "tool_result":
        # A ``ToolResultEvent`` names no tool — it carries ``tool_use_id``, so the
        # tool's name comes from the call it answers.
        use_id = _opt_str(event.get("tool_use_id")) or ""
        status = str(event.get("status") or "")
        label = tools.get(use_id, "result")
        output = _content_text(event.get("content"))
        # Attach to the most recent still-pending call of the same tool: results can
        # arrive out of order when several calls are in flight at once.
        for activity in reversed(thread.tools):
            if activity.status == "pending" and activity.tool == label:
                activity.status = status or "success"
                activity.output = output
                activity.duration = round(ts - activity.timestamp, 3)
                break
        thread.timeline.append(
            TimelineEntry(
                kind="tool_result",
                timestamp=ts,
                label=f"{label} ({status})" if status and status != "success" else label,
                detail=output,
                message_id=_opt_str(event.get("message_id")),
            ),
        )
    elif kind == "token_usage":
        # ``TokenUsageEvent.token_usage`` — reading a "usage" key silently yields zeros.
        usage = event.get("token_usage")
        thread.input_tokens += int(_num(usage, "input_tokens"))
        thread.output_tokens += int(_num(usage, "output_tokens"))
        view.usage_reported = True
    # The runtime's own aif_* events: condition verdicts and step boundaries.
    elif kind == EventKind.PRECONDITION:
        thread.pre_conditions.append(_workflow_cell(event, ts, attempt=0))
    elif kind == EventKind.POSTCONDITION:
        thread.post_conditions.append(_workflow_cell(event, ts, attempt=int(_num(event, "attempt", 1))))
    elif kind == EventKind.STEP_STARTED:
        thread.started_at = thread.started_at or ts
    elif kind.startswith("aif_"):
        return  # Other runtime events (e.g. step finished) carry nothing the view needs.
    else:
        # Unknown library events still show on the trajectory, labelled by kind.
        thread.timeline.append(TimelineEntry(kind=kind, timestamp=ts, label=kind))


def _workflow_cell(event: dict[str, Any], ts: float, *, attempt: int) -> ConditionCell:
    """Build a matrix cell from one of the runtime's own condition events.

    Args:
        event: A decoded ``aif_precondition`` or ``aif_postcondition`` event.
        ts: The event's timestamp.
        attempt: Attempt number; ``0`` for pre-conditions, which run once.

    Returns:
        The evaluated cell.
    """
    return ConditionCell(
        condition=str(event.get("condition") or "?"),
        attempt=attempt,
        passed=bool(event.get("passed")),
        message=_opt_str(event.get("message")),
        kind=str(event.get("condition_kind") or "code"),
        timestamp=ts,
    )


def _feedback_cell(text: str, ts: float) -> list[ConditionCell] | None:
    """Recover condition failures from the library's native feedback turn.

    The library sends failures back as ``[<step>] Post-condition failures (attempt n/m):``
    followed by one ``- <message>`` line per failure. The feedback names no condition,
    so each cell is named after the start of its message and marked ``inferred``.

    Args:
        text: The text of a ``message_user`` event.
        ts: The event's timestamp.

    Returns:
        One failed cell per failure line, or ``None`` when ``text`` is not feedback.
    """
    match = _FEEDBACK.match(text)
    if match is None:
        return None
    attempt = int(match.group("attempt"))
    # Each "- " bullet line is one failed condition's message.
    messages = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
    return [
        ConditionCell(
            condition=message[:60] or "post-condition",
            attempt=attempt,
            passed=False,
            message=message,
            timestamp=ts,
            inferred=True,
        )
        for message in messages
    ]


def _text(event: dict[str, Any], *keys: str) -> str | None:
    """Return the first string-ish value among ``keys``.

    Args:
        event: A decoded event.
        *keys: Field names to try, in order.

    Returns:
        The first non-empty string, or scalar rendered as one; ``None`` if none is found.
        Dicts and lists are skipped rather than rendered.
    """
    for key in keys:
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, dict | list):
            return str(value)
    return None


def _content_text(content: object) -> str:
    """Flatten Strands-format tool-result content blocks into readable text.

    Args:
        content: A string, a single block, or a list of blocks.

    Returns:
        The blocks' text joined by newlines, capped at 2000 characters.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:2000]
    if not isinstance(content, list):
        return _block_text(content)
    parts = [_block_text(block) for block in cast("list[object]", content)]
    return "\n".join(part for part in parts if part)[:2000]


def _block_text(block: object) -> str:
    """Return the text of one content block, or a compact rendering when it carries none.

    Keys are matched case-insensitively because Kiro sends ``Text`` where Strands
    sends ``text``.

    Args:
        block: One content block: a string, a dict, or anything else.

    Returns:
        The block's text, its nested items' text, or a JSON rendering as a last resort.
    """
    if isinstance(block, str):
        return _unwrap_json(block)
    if not isinstance(block, dict):
        return str(block)
    typed = cast("dict[str, object]", block)
    # Prefer a text field, then nested items or content, then a JSON rendering.
    for key, value in typed.items():
        if key.lower() == "text" and isinstance(value, str):
            return _unwrap_json(value)
    for key, value in typed.items():
        if key.lower() in {"items", "content"} and isinstance(value, list):
            nested = [_block_text(item) for item in cast("list[object]", value)]
            joined = "\n".join(part for part in nested if part)
            if joined:
                return joined
    # Kiro returns command results as a single ``{"Json": {...}}`` block; unwrap the
    # sole key rather than showing the reader a wrapper it does not care about.
    if len(typed) == 1:
        (inner,) = typed.values()
        if isinstance(inner, dict | list):
            return _json_ish(inner)
    return _json_ish(typed)


def _unwrap_json(text: str) -> str:
    """Unwrap a text block whose body is itself a JSON payload.

    Kiro returns tool results as a text block holding JSON-encoded output, so the
    payload is decoded and flattened again.

    Args:
        text: A text block's body.

    Returns:
        The payload's flattened text, or ``text`` unchanged when it is not JSON.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return text
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    inner = _block_text(payload)
    return inner or text


def _json_ish(value: object) -> str:
    """Render a payload as readable JSON, falling back to ``repr``.

    Args:
        value: Any payload.

    Returns:
        Indented JSON (or the string itself), capped at 2000 characters; empty for ``None``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:2000]
    try:
        return json.dumps(value, indent=2, default=str)[:2000]
    except (TypeError, ValueError):
        return repr(value)[:2000]


def _num(source: object, key: str, default: float = 0.0) -> float:
    """Read a numeric field defensively; missing or non-numeric yields ``default``.

    Args:
        source: The mapping to read from; anything else yields ``default``.
        key: The field name.
        default: Value used when the field is absent or not a number.

    Returns:
        The field as a float.
    """
    if not isinstance(source, dict):
        return default
    value = source.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _opt_str(value: object) -> str | None:
    """Coerce to ``str`` unless absent.

    Args:
        value: Any value.

    Returns:
        ``None`` for ``None``; otherwise ``str(value)``.
    """
    return None if value is None else str(value)


# ── Board: discovery and aggregation over a folder ────────────────────────────


def default_root() -> Path:
    """Return the folder to scan when none is given: ``$AIF_HOME`` or ``./.aif``.

    Returns:
        ``$AIF_HOME`` resolved to an absolute path when set; otherwise ``.aif`` under the
        current working directory.
    """
    override = os.environ.get("AIF_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / ".aif"


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read an events log, skipping a line still being written.

    The runtime appends to the log while the viewer reads it, so the last line may be
    partial. Reading stops at the first line that does not decode; the next scan picks
    it up once complete.

    Args:
        path: The run's ``events.jsonl``.

    Returns:
        The decoded events in log order; empty when the file does not exist.
    """
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("%s: skipping a partially written line", path)
            break
    return events


def _read_manifest(path: Path) -> Manifest:
    """Return a manifest, using empty defaults when it is missing or unreadable.

    Args:
        path: A package or run ``manifest.json``.

    Returns:
        The parsed manifest, or an empty one.
    """
    if path.exists():
        try:
            return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("%s: unreadable manifest", path)
    return Manifest()


@dataclass
class Run:
    """One recorded run on disk: its directory, metadata, and workflow manifest.

    Built by :meth:`Board._load`; the event log is read separately, on demand.
    """

    directory: Path
    meta: RunMeta
    manifest: Manifest

    @property
    def events_path(self) -> Path:
        """Path of this run's event log."""
        return self.directory / "events.jsonl"

    @property
    def package_dir(self) -> Path:
        """The ``<root>/<slug>/`` package holding this run's prompts and conditions."""
        return self.directory.parent.parent

    @property
    def signature(self) -> tuple[int, int]:
        """Cheap change token for the log: ``(size, mtime_ns)``.

        The log is append-only, so a changed size or mtime means new events; ``(0, 0)``
        when the log does not exist yet.
        """
        try:
            stat = self.events_path.stat()
        except OSError:
            return (0, 0)
        return (stat.st_size, stat.st_mtime_ns)


@dataclass
class _Cached:
    """A parsed run plus the log signature it was parsed from.

    The signature lets :meth:`Board._parse` skip re-parsing a log that has not grown.
    """

    signature: tuple[int, int]
    view: RunView
    cards: list[ThreadCard]


@dataclass
class Board:
    """Aggregate every run under ``root`` into thread cards.

    Discovery is a filesystem glob on each call, so runs recorded concurrently from any
    number of terminals appear without coordination. Parses are cached per run and
    reused while its log signature is unchanged.
    """

    root: Path = field(default_factory=default_root)
    _cache: dict[str, _Cached] = field(default_factory=dict)

    def runs(self) -> list[Run]:
        """Return every run found on disk, newest first.

        Returns:
            Runs matching ``<root>/*/runs/*/run.json``, sorted by start time, newest
            first. Runs whose metadata cannot be read are skipped.
        """
        found: list[Run] = []
        for meta_path in sorted(self.root.glob("*/runs/*/run.json")):
            run = self._load(meta_path)
            if run is not None:
                found.append(run)
        return sorted(found, key=lambda r: r.meta.started_at, reverse=True)

    def cards(self) -> list[ThreadCard]:
        """Return every thread across every run, most recently started first.

        Returns:
            Thread cards from all runs, sorted by start time; pending cards sort last.
        """
        cards = [card for run in self.runs() for card in self._parse(run).cards]
        return sorted(cards, key=lambda c: c.started_at or 0.0, reverse=True)

    def workflows(self) -> list[dict[str, Any]]:
        """Return each run with its threads attached, running ones first.

        Packages with a ``workflow.py`` but no recorded run are included as ``unrun``
        groups, so a freshly generated graph is visible before it executes.

        Returns:
            One dict per run: its metadata, objective, package and run paths, whether its
            runtime is stale, and its thread cards. Running runs first, then newest first.
        """
        groups: list[dict[str, Any]] = []
        for run in self.runs():
            cards = self._parse(run).cards
            groups.append(
                {
                    "run": run.meta.model_dump(),
                    "objective": run.manifest.objective,
                    "stale_runtime": run.meta.stale_runtime,
                    "package": self._relative(run.package_dir),
                    "run_dir": self._relative(run.directory),
                    "threads": [card.model_dump() for card in cards],
                },
            )
        groups.extend(self._unrun())
        return sorted(
            groups,
            key=lambda g: (g["run"]["status"] == "running", g["run"]["started_at"] or 0.0),
            reverse=True,
        )

    def _unrun(self) -> list[dict[str, Any]]:
        """Return a group per package that has a ``workflow.py`` but no run yet.

        Returns:
            Groups shaped like :meth:`graphs` entries, with a synthetic ``unrun:<slug>``
            run id, the script's mtime as the start time, and pending cards for any steps
            in the package manifest.
        """
        groups: list[dict[str, Any]] = []
        for workflow in sorted(self.root.glob("*/workflow.py")):
            package = workflow.parent
            if any(package.glob("runs/*/run.json")):
                continue  # Already shown through its recorded runs.
            manifest = _read_manifest(package / "manifest.json")
            # Synthesise run metadata so the package sorts and renders like a run.
            meta = RunMeta(
                runtime_version=manifest.runtime_version or RUNTIME_VERSION,
                run_id=f"unrun:{package.name}",
                slug=package.name,
                task=manifest.task,
                started_at=workflow.stat().st_mtime,
                status=UNRUN,
                backend=manifest.backend,
            )
            cards = thread_cards(
                RunView(run_id=meta.run_id), slug=meta.slug, task=meta.task, steps=manifest.steps_by_name
            )
            groups.append(
                {
                    "run": meta.model_dump(),
                    "objective": manifest.objective,
                    "stale_runtime": False,
                    "package": self._relative(package),
                    "run_dir": "",
                    "threads": [card.model_dump() for card in cards],
                },
            )
        return groups

    def run(self, run_id: str) -> Run | None:
        """Return one run by id, or ``None`` when it is not on disk.

        Args:
            run_id: The run's id, as recorded in ``run.json``.

        Returns:
            The run, rediscovered from disk on each call.
        """
        return next((r for r in self.runs() if r.meta.run_id == run_id), None)

    def view(self, run_id: str) -> tuple[Run, RunView] | None:
        """Return one run's full parsed view, for a detail panel.

        Args:
            run_id: The run's id.

        Returns:
            The run and its parsed view, or ``None`` when the run is not on disk.
        """
        run = self.run(run_id)
        return None if run is None else (run, self._parse(run).view)

    def _relative(self, path: Path) -> str:
        """Render a path relative to the folder holding ``root``, for display.

        Args:
            path: A package or run directory.

        Returns:
            The path relative to the project, or absolute when it lies outside it.
        """
        project = self.root.parent
        try:
            return str(path.relative_to(project))
        except ValueError:
            return str(path)

    def _load(self, meta_path: Path) -> Run | None:
        """Read a run's snapshot, falling back to the package only for legacy runs.

        Args:
            meta_path: The run's ``run.json``.

        Returns:
            The run, or ``None`` when its metadata cannot be read or parsed.
        """
        try:
            meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("%s: unreadable run metadata", meta_path)
            return None
        # Prefer the run's own snapshot; pre-v4 runs only have the package manifest.
        directory = meta_path.parent
        manifest_path = directory / "manifest.json"
        if meta.runtime_version < 4 and not manifest_path.exists():
            manifest_path = directory.parent.parent / "manifest.json"
        manifest = _read_manifest(manifest_path)
        # Fill identity fields from the directory layout when run.json omits them.
        if not meta.run_id:
            meta = meta.model_copy(update={"run_id": directory.name})
        if not meta.slug:
            meta = meta.model_copy(update={"slug": directory.parent.parent.name})
        return Run(directory=directory, meta=meta, manifest=manifest)

    def _parse(self, run: Run) -> _Cached:
        """Parse a run's log, reusing the cached parse when it has not grown.

        Args:
            run: The run to parse.

        Returns:
            The parsed view and cards, from cache when the log is unchanged.
        """
        signature = run.signature
        cached = self._cache.get(run.meta.run_id)
        if cached is not None and cached.signature == signature:
            return cached

        # Parse the log and join it with the run's manifest.
        view = parse_run(run.meta.run_id, read_events(run.events_path))
        cards = thread_cards(
            view,
            slug=run.meta.slug,
            task=run.meta.task,
            session=run.meta.session,
            steps=run.manifest.steps_by_name,
        )
        # A thread with no terminal event in a still-running run is live; the run's
        # own metadata is the only place that distinguishes live from interrupted.
        if run.meta.status == "running":
            for card in cards:
                if card.finished_at is None and card.status not in (PENDING, BLOCKED):
                    card.status = "running"
        else:
            # Once the run has finished, a step it never reached was skipped, not pending.
            for card in cards:
                if card.status == PENDING:
                    card.status = SKIPPED

        fresh = _Cached(signature=signature, view=view, cards=cards)
        self._cache[run.meta.run_id] = fresh
        return fresh


__all__ = [
    "BLOCKED",
    "PENDING",
    "RUNTIME_VERSION",
    "SKIPPED",
    "UNRUN",
    "Board",
    "ConditionCell",
    "ConditionSpec",
    "ConditionTicket",
    "EventKind",
    "Manifest",
    "Run",
    "RunMeta",
    "RunView",
    "StepSpec",
    "ThreadCard",
    "ThreadView",
    "TimelineEntry",
    "ToolActivity",
    "default_root",
    "parse_run",
    "read_events",
    "thread_cards",
]
