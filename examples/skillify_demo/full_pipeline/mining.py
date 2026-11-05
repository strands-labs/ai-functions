"""Mine reusable drafts from recorded experience.

Trajectory types, event conversion, subtask segmentation, and preserve/prune/
repair extraction feed thematic grouping and quality-gated merging. Drafts
retain their source provenance; repair mode runs only on calls labeled failed.
Rejected merges retry with feedback, then fall back to the strongest constituent.
Injected model callbacks use the same validation as the built-in templates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ai_functions.ai_thread import ai_function
from ai_functions.ai_thread.postcondition import PostConditionResult
from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.experimental.skills.skill_library import Skill, SkillLibrary
from ai_functions.types.events import Event

from .quality import Judge, _empty_library, _tier1_aggregate

# --- Trajectories, episodes, and drafts ---


@dataclass
class ToolCall:
    """One tool invocation requested by an assistant turn."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of one tool invocation.

    Attributes:
        tool_call_id: Id of the :class:`ToolCall` this result answers.
        status: ``"success"`` or ``"error"`` for a Bedrock/Strands-sourced
            result; always ``"success"`` for an OpenAI-sourced result (that
            wire shape has no status concept -- see ``mining.py``).
        content: Flattened text content of the result.
    """

    tool_call_id: str
    status: str = "success"
    content: str = ""


@dataclass
class Message:
    """A single role+content turn, with any attached tool activity.

    See the module docstring for the design rationale.
    """

    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Trajectory:
    """An ordered run of messages, identified for provenance tracking."""

    id: str
    messages: list[Message] = field(default_factory=list)


@dataclass
class Episode:
    """A contiguous message slice of a trajectory, optionally labeled.

    ``start``/``end`` are a half-open index range into the owning
    trajectory's ``messages`` (``trajectory.messages[start:end]``, Python
    slice semantics) rather than a copied message list, so episode
    builders can hand back index pairs cheaply and a consumer
    always re-reads from the trajectory of record.

    These indices are only valid against the exact ``Trajectory.messages``
    list they were computed from -- they are not portable across two
    ``Trajectory`` objects built from "the same" underlying conversation.
    In particular, a trajectory built via :func:`trajectories_from_events`
    can have a different message count than the raw conversation, because
    ``reconstruct_messages``'s I10 healing pass may insert synthetic
    tool-result messages. Always slice the same ``Trajectory`` an
    ``Episode`` was built from.
    """

    trajectory_id: str
    start: int
    end: int
    label: str | None = None


@dataclass
class Provenance:
    """Explains how a skill draft or group came to exist."""

    trajectory_ids: list[str] = field(default_factory=list)
    constituent_ids: list[str] = field(default_factory=list)
    strategy: str | None = None
    group_label: str | None = None


@dataclass
class SkillDraft:
    """A candidate skill extracted from one or more episodes."""

    name: str
    description: str
    body: str
    strategy: str
    provenance: Provenance


@dataclass
class SkillGroup:
    """A cluster of related skill drafts, ready for merging."""

    label: str
    members: list[SkillDraft] = field(default_factory=list)


# --- Recorded events to trajectories ---


def trajectories_from_events(events: list[Event], *, trajectory_id: str) -> Trajectory:
    """Reconstruct a :class:`Trajectory` from a coordinator event log.

    Args:
        events: A thread's event log in append order, e.g. from
            ``await coordinator.get_events(thread_id)``. Passed straight
            through to ``reconstruct_messages``, so its ordering and
            ``ContextSummarizedEvent`` requirements apply here too.
        trajectory_id: Id to stamp onto the returned ``Trajectory`` --
            typically the originating thread id, but this function does
            not read one off the events itself (several events carry an
            optional ``thread_id``; callers know which one is authoritative
            for their use case).

    Returns:
        A ``Trajectory`` whose ``messages`` are this log's conversation at
        message grain: one ``Message`` per user turn, per assistant turn
        (with any ``toolUse`` blocks attached as that message's
        ``tool_calls``), and per collapsed run of tool results (attached
        as that message's ``tool_results``) -- exactly the message list
        ``reconstruct_messages`` produces, remapped onto this package's
        ``Message`` model.

        Caveat for downstream ``Episode`` indices (see ``mining.py``):
        ``reconstruct_messages``'s I10 healing pass may insert synthetic
        tool-result messages for dangling ``toolUse`` blocks, so the
        message count here can differ from the raw conversation's. An
        ``Episode``'s ``start``/``end`` are only valid against the exact
        ``Trajectory.messages`` list they were computed from.
    """
    strands_messages = reconstruct_messages(list(events))
    messages = [_message_from_bedrock_blocks(m["role"], m["content"]) for m in strands_messages]
    return Trajectory(id=trajectory_id, messages=messages)


def _message_from_bedrock_blocks(role: str, content: Sequence[Mapping[str, Any]]) -> Message:
    """Flatten a Bedrock/Strands content-block list (as ``reconstruct_messages`` produces) into a :class:`Message`."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(str(block["text"]))
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            raw_input = tool_use.get("input")
            arguments = raw_input if isinstance(raw_input, dict) else {"value": raw_input}
            tool_calls.append(
                ToolCall(id=str(tool_use.get("toolUseId", "")), name=str(tool_use.get("name", "")), arguments=arguments)
            )
        elif "toolResult" in block:
            tool_result = block["toolResult"]
            tool_results.append(
                ToolResult(
                    tool_call_id=str(tool_result.get("toolUseId", "")),
                    status=str(tool_result.get("status", "success")),
                    content=_flatten_tool_result_content(tool_result.get("content", [])),
                )
            )
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            reasoning_text = reasoning.get("reasoningText", {}).get("text") if isinstance(reasoning, dict) else None
            if reasoning_text:
                text_parts.append(str(reasoning_text))
        # else: non-text content (image/document/video/audio/citations/...)
        # is intentionally dropped -- see the Message model docstring in
        # mining.py.

    return Message(role=role, content="\n".join(text_parts), tool_calls=tool_calls, tool_results=tool_results)


def _flatten_tool_result_content(blocks: Any) -> str:  # noqa: ANN401
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(str(block["text"]))
        elif "json" in block:
            parts.append(str(block["json"]))
    return "\n".join(parts)


# --- Subtask segmentation ---


class EpisodeBuilder(Protocol):
    """Segments a :class:`Trajectory` into a list of :class:`Episode` slices."""

    async def build(self, trajectory: Trajectory) -> list[Episode]:
        """Return this trajectory's episodes, in trajectory order.

        Args:
            trajectory: The trajectory to segment.

        Returns:
            ``Episode``s whose ``start``/``end`` are only valid against
            ``trajectory.messages`` (see ``Episode``'s docstring for
            the index-validity caveat) -- an empty list for an empty
            trajectory.
        """
        ...


# ---------------------------------------------------------------------------
# Compact per-message summary -- the Subtask prompt input
# ---------------------------------------------------------------------------

# Preferred argument keys for a tool call's "target" in the
# "{msg_index}: {tool_name} {target}" summary line: path/command/text-shaped
# arguments are the most informative signal for what a call touched.
# Checked in this order; the first present key wins.
_TARGET_KEYS = ("path", "file_path", "command", "text", "query")

# Cap on a rendered target's length, keeping the per-message summary line
# genuinely compact.
_TARGET_MAX_LEN = 40


def _target_for(call: ToolCall) -> str:
    """Pick a short "target" string for one tool call.

    Checks ``call.arguments`` for the first key present from
    ``_TARGET_KEYS`` (``path``, ``file_path``, ``command``, ``text``,
    ``query``, in that order); a path-shaped value (containing ``/``) is
    reduced to its basename. When none of those keys are present but the
    call has *some* argument, falls back to the first argument's value (by
    insertion order) -- better than an empty target for an unfamiliar tool.
    ``""`` when the call has no arguments at all. Every candidate is
    stringified and truncated to :data:`_TARGET_MAX_LEN` characters.

    Args:
        call: The tool call to summarize.

    Returns:
        The rendered target string (possibly empty).
    """
    value: Any = None  # noqa: ANN401
    for key in _TARGET_KEYS:
        if key in call.arguments:
            value = call.arguments[key]
            break
    else:
        if call.arguments:
            value = next(iter(call.arguments.values()))

    if value is None:
        return ""
    text = str(value)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text[:_TARGET_MAX_LEN]


def _compact_summary(messages: list[Message]) -> str:
    """Render the compact ``"{msg_index}: {tool_name} {target}"`` summary.

    One line per message that carries at least one tool call -- messages
    with no tool activity (plain user/assistant text turns) are omitted
    entirely, keeping the summary "compact" (a full per-message listing
    would mostly be blank lines). When a message carries more than one tool call (parallel tool use),
    only the *first* call is summarized -- the message index alone already
    anchors a cut point at that message, and spelling out every parallel
    call would blow past "compact" for little segmentation benefit.

    Args:
        messages: The trajectory's messages, in order.

    Returns:
        The newline-joined summary; ``""`` when no message has a tool call.
    """
    lines: list[str] = []
    for i, msg in enumerate(messages):
        if not msg.tool_calls:
            continue
        call = msg.tool_calls[0]
        lines.append(f"{i}: {call.name} {_target_for(call)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subtask -- LLM-backed segmentation
# ---------------------------------------------------------------------------


class CutPoints(BaseModel):
    """Typed LLM output: interior cut-point message indices for a trajectory.

    A valid ``cuts`` list is strictly increasing and every entry satisfies
    ``0 < cut < message_count`` -- see :func:`_check_cuts`, this output
    type's post-condition. ``0`` and ``message_count`` are always implicit
    boundaries (the trajectory's own start/end), so a builder combines them
    with ``cuts`` to slice consecutive ``Episode``s; ``cuts == []`` means
    "one whole-trajectory episode."
    """

    cuts: list[int] = Field(
        default_factory=list,
        description=(
            "Interior message-index cut points, strictly increasing, each "
            "strictly between 0 and the trajectory's message count. Empty "
            "means the whole trajectory is one episode."
        ),
    )


def _check_cuts(result: CutPoints, message_count: int) -> PostConditionResult:
    """Post-condition: ``result.cuts`` is strictly increasing and in bounds.

    Wired onto both LLM segmentation templates via
    ``ThreadConfig.post_conditions`` (mirrors ``memory/json_backend.py``'s
    ``_check_valid_python`` wiring onto ``_consolidate_procedural``).
    ``message_count`` is not a fixed constant: ``AIThread._validate_result``
    supplies it automatically because it names a parameter
    (``message_count``) that matches one of the owning prompt function's own
    bound argument names (``_subtask_segment`` takes
    ``(summary: str, message_count: int)``).

    Args:
        result: The candidate ``CutPoints`` to validate.
        message_count: The trajectory's message count for this call.

    Returns:
        ``PostConditionResult(passed=True)`` when every cut satisfies
        ``0 < cut < message_count`` and ``cuts`` is strictly increasing;
        otherwise ``PostConditionResult(passed=False, message=...)`` naming
        the violation and the offending list.
    """
    cuts = result.cuts
    out_of_bounds = [c for c in cuts if not (0 < c < message_count)]
    if out_of_bounds:
        return PostConditionResult(
            passed=False,
            message=(
                f"cut points must satisfy 0 < cut < message_count ({message_count}); "
                f"out of bounds: {out_of_bounds} (full list: {cuts})"
            ),
        )
    if any(a >= b for a, b in zip(cuts, cuts[1:], strict=False)):
        return PostConditionResult(
            passed=False,
            message=f"cut points must be strictly increasing; got {cuts}",
        )
    return PostConditionResult(passed=True)


@ai_function[CutPoints](post_conditions=[_check_cuts])
def _subtask_segment(summary: str, message_count: int) -> str:
    """Build the prompt asking for SUBTASK cut points over a tool-call summary."""
    return (
        "You are segmenting an agent execution trajectory into coherent SUBTASK "
        'units -- complete problem-solving cycles like "migrate the SQS client" '
        'or "fix the failing build."\n\n'
        "## Input\n"
        "A compact per-message summary of tool calls, one line per message that "
        "invoked a tool, in the format `msg_index: tool_name target`.\n\n"
        "## Task\n"
        "Identify natural SUBTASK boundaries. Split at goal transitions, not at "
        "every tool call -- a build/test attempt followed by fixing the errors it "
        "surfaced is ONE subtask, not two.\n\n"
        "## Response\n"
        "Return the interior cut-point message indices, strictly increasing, each "
        f"strictly between 0 and {message_count} (the total message count). Return "
        "an empty list if the whole trajectory is one subtask.\n\n"
        f"## Tool calls ({message_count} messages total)\n{summary}"
    )


SegmentFn = Callable[[str, int], Awaitable[CutPoints]]
"""Call shape shared by the real segmentation templates and any injected fake.

``(summary, message_count) -> CutPoints``, ``async``. ``_subtask_segment``
(an ``AIFunction`` instance) satisfies this by its normal ``__call__``; a test fake need only match the
same shape to be a drop-in replacement -- see this module's docstring on the
injection seam.
"""


async def _episodes_from_cuts(trajectory: Trajectory, segment_fn: SegmentFn) -> list[Episode]:
    """:class:`Subtask`'s ``build()`` body.

    Empty trajectory: returns ``[]`` without calling ``segment_fn``. No
    tool-call messages (so :func:`_compact_summary` is empty): returns a
    single whole-trajectory episode, again without calling ``segment_fn`` --
    there is nothing tool-shaped to hand the LLM, so this short-circuit
    saves the call rather than sending it an empty summary. Otherwise calls
    ``segment_fn(summary, message_count)`` once, re-validates the returned
    cuts with :func:`_check_cuts` (defense-in-depth: ``segment_fn`` is a
    public constructor parameter, so a caller-supplied fake or a future
    non-``@ai_function`` implementation gets no free pass just because the
    real template's post-condition would have caught it -- see
    :func:`_check_cuts`'s docstring and this module's docstring for why the
    real ``PostCondition`` machinery alone cannot be relied on here), and
    slices consecutive episodes from ``[0, *cuts, message_count]``.

    Args:
        trajectory: The trajectory to segment.
        segment_fn: The (real or fake) segmentation callable.

    Returns:
        The resulting episodes, in order.

    Raises:
        ValueError: ``segment_fn`` returned cuts that are out of bounds or
            not strictly increasing (per :func:`_check_cuts`) -- a
            contract-violating ``segment_fn`` is a programming error the
            caller must see, not something to silently paper over (e.g. an
            out-of-bounds negative cut would otherwise silently become a
            wrong-location slice via Python's negative-index wraparound,
            not a harmless empty episode).
    """
    messages = trajectory.messages
    if not messages:
        return []

    summary = _compact_summary(messages)
    if not summary:
        return [Episode(trajectory_id=trajectory.id, start=0, end=len(messages))]

    result = await segment_fn(summary, len(messages))
    message_count = len(messages)
    check = _check_cuts(result, message_count=message_count)
    if check is not None and not check.passed:
        raise ValueError(
            f"segment_fn returned invalid cuts {result.cuts!r} for message_count={message_count}: {check.message}"
        )
    boundaries = [0, *result.cuts, message_count]
    return [
        Episode(trajectory_id=trajectory.id, start=s, end=e) for s, e in zip(boundaries, boundaries[1:], strict=False)
    ]


@dataclass(frozen=True, slots=True)
class Subtask:
    """LLM segmentation into coherent SUBTASK units.

    Args:
        segment_fn: Segmentation callable; ``None`` (default) binds the
            real module-level ``@ai_function`` template
            (``_subtask_segment``, contracted by :func:`_check_cuts`). Inject
            a fake matching :data:`SegmentFn` in tests -- see this module's
            docstring on the injection seam.
    """

    segment_fn: SegmentFn | None = None

    async def build(self, trajectory: Trajectory) -> list[Episode]:
        """Segment ``trajectory`` into subtask episodes.

        Args:
            trajectory: The trajectory to segment.

        Returns:
            See :func:`_episodes_from_cuts`.
        """
        fn = self.segment_fn if self.segment_fn is not None else _subtask_segment
        return await _episodes_from_cuts(trajectory, fn)


# --- Preserve, prune, and repair extraction ---

MODES = ("preserve", "prune", "repair")
"""Every extraction mode this module knows how to run."""


class ExtractedSkill(BaseModel):
    """Typed extraction result: a candidate skill, or explicitly not-skill-worthy.

    ``worth_extracting=False`` is how "no coherent skill exists in this
    episode" is expressed -- a typed boolean discriminator rather than a
    magic sentinel string a caller would have to pattern-match. When ``worth_extracting`` is
    ``False``, ``name``/``description``/``body`` are not meaningful and are
    ignored by :func:`extract_skills` regardless of their content -- see
    :func:`_check_extracted_skill`, which only requires them non-empty in
    the ``True`` case.
    """

    worth_extracting: bool = Field(description="Whether this episode yields a coherent, non-trivial skill.")
    name: str = Field(default="", description="Short skill name; required when worth_extracting is True.")
    description: str = Field(
        default="", description="One-line skill description; required when worth_extracting is True."
    )
    body: str = Field(
        default="", description="The skill's full instructional body; required when worth_extracting is True."
    )


def _check_extracted_skill(result: ExtractedSkill) -> PostConditionResult:
    """Post-condition: a ``worth_extracting=True`` result carries real content.

    Wired onto every mode's ``@ai_function`` template via
    ``ThreadConfig.post_conditions`` (mirrors ``mining.py``'s
    ``_check_cuts`` wiring). A ``worth_extracting=False`` result always
    passes -- there is nothing to validate about a "no skill here" answer.

    Args:
        result: The candidate ``ExtractedSkill`` to validate.

    Returns:
        ``PostConditionResult(passed=True)`` when ``worth_extracting`` is
        ``False``, or ``True`` with non-blank ``name``/``description``/
        ``body``; otherwise ``PostConditionResult(passed=False,
        message=...)`` naming the blank field(s).
    """
    if not result.worth_extracting:
        return PostConditionResult(passed=True)
    blank = [
        field
        for field, value in (("name", result.name), ("description", result.description), ("body", result.body))
        if not value.strip()
    ]
    if blank:
        return PostConditionResult(
            passed=False,
            message=f"worth_extracting=True but the following fields are blank: {blank}",
        )
    return PostConditionResult(passed=True)


def _render_episode(trajectory: Trajectory, episode: Episode) -> str:
    """Render an episode's message slice as readable text for an extraction prompt.

    One block per message in ``trajectory.messages[episode.start:episode.end]``
    (Python slice semantics; see ``mining.py``'s ``Episode`` docstring for the
    index-validity caveat this relies on): the role, its flattened content,
    and a compact line per tool call/result so the prompt sees both what the
    agent said and what it did.

    Args:
        trajectory: The trajectory ``episode`` was cut from. Must be the
            exact ``Trajectory`` object (or an equal-length rebuild) whose
            messages ``episode.start``/``end`` index into.
        episode: The message slice to render.

    Returns:
        The rendered text, one message per paragraph, in order.
    """
    lines: list[str] = []
    for msg in trajectory.messages[episode.start : episode.end]:
        lines.append(_render_message(msg))
    return "\n\n".join(lines)


def _render_message(msg: Message) -> str:
    parts = [f"{msg.role.upper()}: {msg.content}".rstrip()]
    for call in msg.tool_calls:
        parts.append(f"  [tool_call] {call.name}({call.arguments})")
    for result in msg.tool_results:
        parts.append(f"  [tool_result:{result.status}] {result.content}")
    return "\n".join(parts)


ExtractFn = Callable[[str, str], Awaitable[ExtractedSkill]]
"""Call shape shared by the real per-mode templates and any injected fake.

``(episode_text, mode) -> ExtractedSkill``, ``async``. ``mode`` is one of
:data:`MODES`. A single injection seam covers all three modes (unlike
``mining.py``'s per-builder ``segment_fn``, extraction has one entry point
-- :func:`extract_skills` -- rather than one class per mode), so a test fake
need only branch on ``mode`` to stand in for all three real templates at
once. See this module's docstring on the injection seam.
"""


@ai_function[ExtractedSkill](post_conditions=[_check_extracted_skill])
def _preserve_extract(episode_text: str) -> str:
    """Build the prompt asking for a PRESERVE-mode skill draft."""
    return (
        "You are extracting a reusable SKILL from a successful agent episode -- "
        "a self-contained instruction capturing transferable knowledge an agent "
        "can apply in future scenarios (from CostCraft, Yuan et al. 2026's "
        '"preserve" extraction mode).\n\n'
        "## Task\n"
        "Identify the behaviors that led to success in this episode and distill "
        "them into a concise, reusable skill: a name, a one-line description, "
        "and a full instructional body. If no coherent, non-trivial skill "
        "exists in this episode, set worth_extracting to false.\n\n"
        f"## Episode\n{episode_text}"
    )


@ai_function[ExtractedSkill](post_conditions=[_check_extracted_skill])
def _prune_extract(episode_text: str) -> str:
    """Build the prompt asking for a PRUNE-mode skill draft."""
    return (
        "You are extracting a PRUNE rule from an agent episode -- a concise "
        "instruction about what to AVOID in future runs, identifying wasteful "
        "or redundant steps that were expensive but did not affect the outcome "
        '(CostCraft\'s "prune" extraction mode).\n\n'
        "## Task\n"
        "Identify any wasteful or redundant steps in this episode and distill "
        "them into a concise skill -- name, one-line description, and full "
        "instructional body -- describing what to avoid. If no waste exists, "
        "set worth_extracting to false.\n\n"
        f"## Episode\n{episode_text}"
    )


@ai_function[ExtractedSkill](post_conditions=[_check_extracted_skill])
def _repair_extract(episode_text: str) -> str:
    """Build the prompt asking for a REPAIR-mode skill draft."""
    return (
        "You are extracting a REPAIR rule from a FAILED agent episode -- "
        "identify the root cause of the failure and distill a concise "
        "instruction that would prevent it in future runs (CostCraft's "
        '"repair" extraction mode).\n\n'
        "## Task\n"
        "Identify the root cause of failure in this episode and distill it "
        "into a concise skill -- name, one-line description, and full "
        "instructional body -- describing the fix. If the failure cause is "
        "unclear, set worth_extracting to false.\n\n"
        f"## Episode (FAILED)\n{episode_text}"
    )


_PerModeTemplate = Callable[[str], Awaitable[ExtractedSkill]]
"""Call shape of each real per-mode template: ``(episode_text) -> ExtractedSkill``."""

_REAL_TEMPLATES: dict[str, _PerModeTemplate] = {
    "preserve": _preserve_extract,
    "prune": _prune_extract,
    "repair": _repair_extract,
}


async def _default_extract(episode_text: str, mode: str) -> ExtractedSkill:
    """Default ``extract_fn``: dispatch to the real per-mode ``@ai_function`` template.

    Construction of the three module-level templates above does no I/O (pure
    bookkeeping, matching ``mining.py``'s ``_subtask_segment`` rationale)
    -- only calling this dispatcher for a given ``mode`` actually reaches an
    LLM.

    Args:
        episode_text: The rendered episode text (see :func:`_render_episode`).
        mode: One of :data:`MODES`.

    Returns:
        The template's typed result.
    """
    return await _REAL_TEMPLATES[mode](episode_text)


def _episode_id(episode: Episode) -> str:
    """A stable, human-readable identity for an episode, for provenance tracking.

    ``Episode`` carries no id of its own (see ``mining.py``) -- this module's
    own convention, ``"{trajectory_id}:{start}-{end}"``, uniquely identifies
    an episode within the ``episodes`` list passed to :func:`extract_skills`
    (two distinct episodes never share a ``(trajectory_id, start, end)``
    triple within one well-formed call).
    """
    return f"{episode.trajectory_id}:{episode.start}-{episode.end}"


async def extract_skills(
    trajectories: Sequence[Trajectory],
    episodes: Sequence[Episode],
    *,
    modes: tuple[str, ...] = ("preserve", "prune"),
    extract_fn: ExtractFn | None = None,
    failed_trajectory_ids: frozenset[str] = frozenset(),
) -> list[SkillDraft]:
    """Extract candidate skill drafts from ``episodes``, one call per (episode, mode).

    Args:
        trajectories: The trajectories ``episodes`` were cut from -- source
            of the message content an episode's ``start``/``end`` index
            into (see this module's docstring on the episode-text-plumbing
            design). Every ``episode.trajectory_id`` must name a trajectory
            present here.
        episodes: The episodes to extract from, in order.
        modes: Which extraction modes to run, in order, over every episode
            (subject to the repair gate below). Must be a non-empty subset
            of :data:`MODES`.
        extract_fn: Extraction callable; ``None`` (default) binds
            :func:`_default_extract`, which dispatches to the real
            module-level ``@ai_function`` template for the requested mode.
            Inject a fake matching :data:`ExtractFn` in tests -- no LLM is
            constructed or reachable from that code path.
        failed_trajectory_ids: Trajectory ids to treat as carrying a failure
            label (see this module's docstring on the failure-label design).
            ``"repair"`` mode only runs for episodes whose ``trajectory_id``
            is in this set; ``"preserve"``/``"prune"`` run over every
            episode in ``modes`` unconditionally, regardless of this set.
            A *partial* mismatch against ``trajectories`` is permitted (the
            documented superset use case -- a caller may pass a set scoped
            to a broader run than this particular call); a *total*
            mismatch is rejected, see ``Raises`` below.

    Returns:
        One ``SkillDraft`` per (episode, mode) pair whose result had
        ``worth_extracting=True``, in ``episodes`` order then ``modes``
        order -- a mode that returned ``worth_extracting=False``, or a
        repair call skipped by the failure gate, contributes nothing.
        ``strategy`` on each draft is the mode name; ``provenance``
        carries the source trajectory id and this module's synthetic
        episode id (:func:`_episode_id`).

    Raises:
        ValueError: ``modes`` is empty, names a mode outside :data:`MODES`,
            an episode's ``trajectory_id`` has no matching entry in
            ``trajectories``, ``extract_fn`` returned an ``ExtractedSkill``
            violating :func:`_check_extracted_skill` (defense-in-depth --
            see this module's docstring; a contract-violating
            ``extract_fn`` is a programming error the caller must see, not
            something to silently paper over), or ``"repair"`` is
            requested with a non-empty ``failed_trajectory_ids`` that
            matches *none* of a non-empty ``trajectories``' ids -- a total
            mismatch like this is structurally indistinguishable from "no
            failures" (repair silently extracts nothing either way), so
            it is treated as a likely caller typo and rejected rather than
            silently accepted; a *partial* mismatch (at least one id
            matches) is not an error, and neither is an empty
            ``trajectories`` (nothing to mistype against -- the call
            trivially returns ``[]``).
    """
    if not modes:
        raise ValueError("modes must be non-empty")
    unknown_modes = [m for m in modes if m not in MODES]
    if unknown_modes:
        raise ValueError(f"unknown extraction mode(s) {unknown_modes}; expected a subset of {MODES}")

    by_id: dict[str, Trajectory] = {t.id: t for t in trajectories}
    missing = sorted({e.trajectory_id for e in episodes} - by_id.keys())
    if missing:
        raise ValueError(f"episode(s) reference trajectory id(s) not present in `trajectories`: {missing}")

    # `by_id` must be non-empty for this to mean anything: with no trajectories
    # at all there is no id set to mistype against (the call trivially extracts
    # nothing either way), so a total "mismatch" there is not evidence of a typo.
    if "repair" in modes and failed_trajectory_ids and by_id and failed_trajectory_ids.isdisjoint(by_id.keys()):
        raise ValueError(
            f"failed_trajectory_ids {sorted(failed_trajectory_ids)} match none of the "
            f"available trajectory ids {sorted(by_id.keys())} -- repair mode would silently "
            "extract nothing, indistinguishable from 'no failures' (likely a typo in "
            "failed_trajectory_ids); a partial match (at least one shared id) is fine"
        )

    fn = extract_fn if extract_fn is not None else _default_extract

    drafts: list[SkillDraft] = []
    for episode in episodes:
        episode_text = _render_episode(by_id[episode.trajectory_id], episode)
        for mode in modes:
            if mode == "repair" and episode.trajectory_id not in failed_trajectory_ids:
                continue
            result = await fn(episode_text, mode)
            check = _check_extracted_skill(result)
            if check is not None and not check.passed:
                raise ValueError(
                    f"extract_fn returned an invalid ExtractedSkill for episode "
                    f"{_episode_id(episode)!r} mode {mode!r}: {check.message}"
                )
            if not result.worth_extracting:
                continue
            drafts.append(
                SkillDraft(
                    name=result.name,
                    description=result.description,
                    body=result.body,
                    strategy=mode,
                    provenance=Provenance(
                        trajectory_ids=[episode.trajectory_id],
                        constituent_ids=[_episode_id(episode)],
                        strategy=mode,
                    ),
                )
            )
    return drafts


# --- Group related drafts ---

UNGROUPED_LABEL = "ungrouped"
"""Label for the combined bucket every singleton draft is folded into."""

# Cap on a rendered draft summary line's description length, keeping the
# per-draft summary line genuinely compact.
_SUMMARY_MAX_LEN = 100


class ThemeGroups(BaseModel):
    """Typed LLM output: a mapping of group label to member draft indices.

    A valid ``groups`` mapping, checked by :func:`_check_groups`, partitions
    ``range(draft_count)`` exactly: every index appears in exactly one
    label's list, no index repeats, and no index is out of range. Labels are
    caller-chosen text (a theme name like ``"DynamoDB Migration"`` for
    :class:`LLMThematicGrouper`) -- this type itself does not require them
    to be "thematic."
    """

    groups: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Group label -> member draft indices. Every draft index must appear in exactly one group.",
    )


def _check_groups(result: ThemeGroups, draft_count: int) -> PostConditionResult:
    """Post-condition: ``result.groups`` partitions ``range(draft_count)`` exactly.

    Wired onto both real ``@ai_function`` templates via
    ``ThreadConfig.post_conditions`` (mirrors ``mining.py``'s
    ``_check_cuts`` wiring). ``draft_count`` is supplied by
    ``AIThread._validate_result`` from the owning prompt function's own
    bound arguments, the same mechanism ``_check_cuts`` relies on for
    ``message_count`` -- both real templates below take
    ``(summary: str, draft_count: int)``.

    Args:
        result: The candidate ``ThemeGroups`` to validate.
        draft_count: The number of drafts this call was grouping.

    Returns:
        ``PostConditionResult(passed=True)`` when every index in
        ``range(draft_count)`` appears in exactly one group's list;
        otherwise ``PostConditionResult(passed=False, message=...)`` naming
        the out-of-range, duplicated, and/or missing indices.
    """
    seen: dict[int, str] = {}
    out_of_range: list[int] = []
    duplicates: list[int] = []
    for label, indices in result.groups.items():
        for i in indices:
            if not (0 <= i < draft_count):
                out_of_range.append(i)
                continue
            if i in seen:
                duplicates.append(i)
            else:
                seen[i] = label
    missing = [i for i in range(draft_count) if i not in seen]
    if out_of_range or duplicates or missing:
        return PostConditionResult(
            passed=False,
            message=(
                f"groups must partition range({draft_count}) exactly: "
                f"out_of_range={out_of_range}, duplicated={duplicates}, missing={missing}"
            ),
        )
    return PostConditionResult(passed=True)


def _draft_summary(index: int, draft: SkillDraft) -> str:
    """One compact summary line for a draft, for a grouping prompt."""
    return f"[{index}] {draft.name}: {draft.description[:_SUMMARY_MAX_LEN]}"


def _drafts_summary(drafts: list[SkillDraft]) -> str:
    return "\n".join(_draft_summary(i, d) for i, d in enumerate(drafts))


GroupFn = Callable[[str, int], Awaitable[ThemeGroups]]
"""Call shape shared by the real per-variant templates and any injected fake.

``(summary, draft_count) -> ThemeGroups``, ``async`` -- matches
``mining.py``'s ``SegmentFn`` shape (a compact text summary plus a count,
in, a typed result out). See this module's docstring on the injection seam.
"""


@ai_function[ThemeGroups](post_conditions=[_check_groups])
def _thematic_group(summary: str, draft_count: int) -> str:
    """Build the prompt asking for named THEMATIC clusters."""
    return (
        "You are grouping candidate skills into thematic clusters (no "
        "similarity threshold, no embeddings -- group by topic). Each cluster "
        "should contain skills about the same specific topic, with a "
        'human-interpretable label (e.g. "DynamoDB client migration", "SQS '
        'queue setup", "build error debugging").\n\n'
        "## Task\n"
        "Assign every skill below to exactly one cluster. A skill with no "
        "close relative to any other skill may form its own single-member "
        "cluster -- every index must still appear in exactly one group.\n\n"
        "## Response\n"
        "Return a JSON object mapping cluster label to the list of member "
        f"skill indices, covering every index from 0 to {draft_count - 1} "
        "exactly once.\n\n"
        f"## Skills ({draft_count} total)\n{summary}"
    )


@dataclass(frozen=True, slots=True)
class LLMThematicGrouper:
    """Groups drafts by asking an LLM to name themes and assign drafts to them.

    Args:
        group_fn: Grouping callable; ``None`` (default) binds the real
            module-level ``@ai_function`` template (``_thematic_group``,
            contracted by :func:`_check_groups`). Inject a fake matching
            :data:`GroupFn` in tests -- see this module's docstring on the
            injection seam.
    """

    group_fn: GroupFn | None = None

    async def group(self, drafts: list[SkillDraft]) -> list[SkillGroup]:
        """Group ``drafts`` into themed clusters (see :func:`_group_drafts`)."""
        fn = self.group_fn if self.group_fn is not None else _thematic_group
        return await _group_drafts(drafts, fn)


class Grouper(Protocol):
    """Structural contract a grouper satisfies (:class:`LLMThematicGrouper` does).

    Matches ``mining.py``'s ``EpisodeBuilder`` protocol style: a caller
    types a grouping-strategy parameter as ``Grouper`` without naming a
    concrete implementation, and any future grouper (e.g. a future
    embedding-based one) satisfies this contract structurally, with no
    inheritance required.
    """

    async def group(self, drafts: list[SkillDraft]) -> list[SkillGroup]:
        """Group ``drafts`` into ``SkillGroup``s under the membership contract.

        Args:
            drafts: The candidate skill drafts to cluster.

        Returns:
            ``SkillGroup``s such that every input draft appears in exactly
            one group's ``members``; singletons are folded into one
            combined ``"ungrouped"`` group (see module docstring).
        """
        ...


async def _group_drafts(drafts: list[SkillDraft], group_fn: GroupFn) -> list[SkillGroup]:
    """:class:`LLMThematicGrouper`'s ``group()`` body.

    Empty ``drafts``: returns ``[]`` without calling ``group_fn``. Exactly
    one draft: returns a single ``"ungrouped"`` group holding it, again
    without calling ``group_fn`` -- there is nothing to cluster, so this
    short-circuit saves the call (mirrors ``mining.py``'s
    ``_episodes_from_cuts`` empty/no-tool-call short-circuits). Otherwise
    calls ``group_fn(summary, len(drafts))`` once, re-validates the
    returned groups with :func:`_check_groups` (defense-in-depth -- see this
    module's docstring), and materializes ``SkillGroup``s: every group with
    two or more members becomes its own ``SkillGroup``; every group left
    with exactly one member (by the LLM's own clustering choice) is folded
    into one combined ``"ungrouped"`` ``SkillGroup``, appended last, only
    when at least one singleton exists. Caveat: if ``group_fn`` itself
    returns a *multi*-member group already labeled ``"ungrouped"``, that
    group is kept as-is (its label is caller-chosen text, not reserved) --
    the returned list can then contain two ``SkillGroup``s labeled
    ``"ungrouped"``, one from the LLM's own labeling and one from this
    function's singleton fold. Not deduplicated/merged; a caller that
    cares should merge same-labeled groups itself.

    Args:
        drafts: The drafts to group.
        group_fn: The (real or fake) grouping callable.

    Returns:
        The resulting groups. Every input draft appears in exactly one
        group's ``members`` (module-docstring contract).

    Raises:
        ValueError: ``group_fn`` returned groups that do not partition
            ``range(len(drafts))`` exactly (per :func:`_check_groups`) -- a
            contract-violating ``group_fn`` is a programming error the
            caller must see, not something to silently paper over (e.g. a
            dropped index would otherwise silently discard a draft, and a
            duplicated index would silently double-count one across two
            groups).
    """
    if not drafts:
        return []
    if len(drafts) == 1:
        return [_ungrouped_group(drafts)]

    summary = _drafts_summary(drafts)
    result = await group_fn(summary, len(drafts))
    check = _check_groups(result, draft_count=len(drafts))
    if check is not None and not check.passed:
        raise ValueError(
            f"group_fn returned invalid groups {result.groups!r} for {len(drafts)} drafts: {check.message}"
        )

    groups: list[SkillGroup] = []
    singletons: list[SkillDraft] = []
    for label, indices in result.groups.items():
        members = [drafts[i] for i in indices]
        if len(members) == 1:
            singletons.extend(members)
        else:
            groups.append(SkillGroup(label=label, members=members))
    if singletons:
        groups.append(_ungrouped_group(singletons))
    return groups


def _ungrouped_group(drafts: list[SkillDraft]) -> SkillGroup:
    return SkillGroup(label=UNGROUPED_LABEL, members=list(drafts))


# --- Merge behind a quality gate ---

DEFAULT_MAX_ATTEMPTS = 3
"""Default cap on propose->gate retries per group before falling back."""


# ---------------------------------------------------------------------------
# GateResult / MergeGate protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of one ``MergeGate.accept`` call.

    Attributes:
        accepted: Whether the merged candidate clears the gate.
        rationale: Free-form human-readable explanation.
        scores: Numeric scores backing the decision. By convention (see
            this module's docstring on "best constituent"), every gate here
            populates ``"merged"`` with the merged candidate's own score and
            ``f"constituent#{i}"`` for each ``constituents[i]`` -- callers
            (notably :func:`_fallback_draft`) rely on this to identify the
            best-scoring constituent without recomputing anything. A future
            ``MergeGate`` that cannot score constituents individually may
            omit some/all ``constituent#i`` entries; :func:`_fallback_draft`
            degrades to the first constituent when none are present.
    """

    accepted: bool
    rationale: str
    scores: Mapping[str, float | None]


class MergeGate(Protocol):
    """Decides whether a merged candidate replaces its constituents."""

    async def accept(self, merged: Skill, constituents: list[Skill]) -> GateResult:
        """Score ``merged`` against ``constituents`` and decide accept/reject.

        Args:
            merged: The candidate merged skill.
            constituents: The skills ``merged`` proposes to replace.

        Returns:
            A :class:`GateResult`; see its docstring for the ``scores``
            convention every gate in this module follows.
        """
        ...


# ---------------------------------------------------------------------------
# MetricGate -- the default gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricGate:
    """Default gate.

    Accepts iff the merged candidate's Tier-1 aggregate is >= the best
    constituent's.

    Args:
        judge: Shared judge for ``CompletenessMetric``/``DistinctivenessMetric``;
            ``None`` (default) lets each metric build its own lazy
            ``AIFunctionJudge`` -- tests inject ``StubJudge`` here directly.
        library: Library constituents/merged are measured for distinctiveness
            against. ``None`` (default) uses an empty library, so
            ``DistinctivenessMetric`` scores every candidate 1.0 (its own
            "nothing to compare against" convention -- see ``quality.py``).
    """

    judge: Judge | None = None
    library: SkillLibrary | None = None

    async def accept(self, merged: Skill, constituents: list[Skill]) -> GateResult:
        """Score ``merged`` and every constituent via :func:`_tier1_aggregate`.

        Accepts iff ``merged``'s aggregate is not ``None`` and is >= the
        best constituent's aggregate (a constituent whose own aggregate is
        ``None`` cannot win the "best" comparison; if *every* constituent's
        aggregate is ``None``, ``merged`` is accepted as long as it has one
        -- there is nothing for it to lose to).
        """
        lib = self.library if self.library is not None else _empty_library()
        merged_score, _ = await _tier1_aggregate(merged, lib, self.judge)

        constituent_scores: list[float | None] = []
        for c in constituents:
            score, _ = await _tier1_aggregate(c, lib, self.judge)
            constituent_scores.append(score)

        best_score = max((s for s in constituent_scores if s is not None), default=None)

        if merged_score is None:
            accepted = False
        elif best_score is None:
            accepted = True
        else:
            accepted = merged_score >= best_score

        scores: dict[str, float | None] = {"merged": merged_score}
        for i, s in enumerate(constituent_scores):
            scores[f"constituent#{i}"] = s

        rationale = (
            f"Tier-1 aggregate (completeness+distinctiveness+format_alignment mean): "
            f"merged={merged_score!r} vs best_constituent={best_score!r} -> "
            f"{'accepted' if accepted else 'rejected'}"
        )
        return GateResult(accepted=accepted, rationale=rationale, scores=scores)


# ---------------------------------------------------------------------------
# Merge candidate: pydantic (LLM-facing) + the real @ai_function template
# ---------------------------------------------------------------------------


class MergedSkillCandidate(BaseModel):
    """Typed LLM output: a single skill draft folding every constituent together."""

    name: str = Field(description="Short skill name for the merged skill.")
    description: str = Field(description="One-line description of the merged skill.")
    body: str = Field(description="The merged skill's full instructional body.")


def _check_merged_candidate(result: MergedSkillCandidate) -> PostConditionResult:
    """Post-condition / defensive re-check: a merge candidate carries real content.

    Wired onto the real template via ``ThreadConfig.post_conditions``
    (mirrors ``mining.py``'s ``_check_extracted_skill``), and re-run inside
    :func:`_merge_with_gate` against *any* ``merge_fn`` result (real or
    fake) -- see this module's docstring on defensive validation.
    """
    blank = [
        field_name
        for field_name, value in (("name", result.name), ("description", result.description), ("body", result.body))
        if not value.strip()
    ]
    if blank:
        return PostConditionResult(
            passed=False,
            message=f"merge candidate has blank field(s): {blank}",
        )
    return PostConditionResult(passed=True)


def _render_constituent(index: int, draft: SkillDraft) -> str:
    return f"[{index}] {draft.name}: {draft.description}\n{draft.body}"


def _render_constituents(members: list[SkillDraft]) -> str:
    return "\n\n".join(_render_constituent(i, d) for i, d in enumerate(members))


@ai_function[MergedSkillCandidate](post_conditions=[_check_merged_candidate])
def _merge_template(constituents_text: str) -> str:
    """Build the prompt asking for one merged skill folding every constituent together."""
    return (
        "You are merging a group of related candidate skills into ONE coherent, "
        "non-redundant skill that preserves every constituent's distinct, useful "
        "content.\n\n"
        "## Task\n"
        "Read every constituent skill below and produce a single merged skill -- "
        "a name, a one-line description, and a full instructional body -- that a "
        "competent agent could follow in place of ALL of them.\n\n"
        f"## Constituent skills\n{constituents_text}"
    )


MergeFn = Callable[[str], Awaitable[MergedSkillCandidate]]
"""Call shape shared by the real template and any injected fake.

``(constituents_text) -> MergedSkillCandidate``, ``async``. See this
module's docstring on the injection seam and its defensive-validation
consequence.
"""


async def _default_merge(constituents_text: str) -> MergedSkillCandidate:
    """Default ``merge_fn``: dispatch to the real module-level ``@ai_function`` template."""
    return await _merge_template(constituents_text)


# ---------------------------------------------------------------------------
# Skill <-> SkillDraft/MergedSkillCandidate conversion (in-memory, no disk I/O)
# ---------------------------------------------------------------------------

_NAME_RE_SUB = "abcdefghijklmnopqrstuvwxyz0123456789-"
_NAME_MAX_LEN = 64
"""Longest slug ``SkillFrontmatterSpec.name_pattern`` accepts (``quality.py``)."""


def _slugify_name(name: str) -> str:
    """Best-effort slug matching ``SkillFrontmatterSpec.name_pattern`` (``quality.py``).

    Not required to be perfect -- ``FormatAlignmentMetric`` only lowers one
    component's score on a mismatch, it never raises -- but a reasonable
    slug keeps Tier-1 format-alignment scoring meaningful for merge
    candidates and constituents alike. The pattern caps a name at 64
    characters, so a long name is truncated to :data:`_NAME_MAX_LEN` with any
    boundary hyphen stripped (same rule as ``discovery.py``'s own
    ``_slugify_name``, deliberately duplicated rather than imported).
    """
    lowered = name.strip().lower()
    slug = "".join(ch if ch in _NAME_RE_SUB else "-" for ch in lowered).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug[:_NAME_MAX_LEN].rstrip("-")
    return slug or "skill"


def _skill_from_parts(name: str, description: str, body: str) -> Skill:
    """Build an in-memory ``Skill`` (no disk I/O) from name/description/body.

    Used to score a ``SkillDraft`` constituent or a ``MergedSkillCandidate``
    through the Tier-1 metrics, which only read ``Skill`` attributes -- no
    real ``SKILL.md`` file is required to exist on disk for scoring.
    """
    slug = _slugify_name(name)
    raw = f"---\nname: {slug}\ndescription: {description}\n---\n\n{body}\n"
    return Skill(
        path=Path(f"<merge-candidate:{slug}>"),
        frontmatter={"name": slug, "description": description},
        body=body,
        raw=raw,
        resources=[],
    )


def _skill_from_draft(draft: SkillDraft) -> Skill:
    return _skill_from_parts(draft.name, draft.description, draft.body)


def _skill_from_candidate(candidate: MergedSkillCandidate) -> Skill:
    return _skill_from_parts(candidate.name, candidate.description, candidate.body)


def _draft_from_candidate(candidate: MergedSkillCandidate, group: SkillGroup) -> SkillDraft:
    trajectory_ids = sorted({tid for m in group.members for tid in m.provenance.trajectory_ids})
    constituent_ids = sorted({cid for m in group.members for cid in m.provenance.constituent_ids})
    return SkillDraft(
        name=candidate.name,
        description=candidate.description,
        body=candidate.body,
        strategy="merge",
        provenance=Provenance(
            trajectory_ids=trajectory_ids,
            constituent_ids=constituent_ids,
            strategy="merge",
            group_label=group.label,
        ),
    )


# ---------------------------------------------------------------------------
# Fallback selection -- shared by every gate via GateResult.scores
# ---------------------------------------------------------------------------


def _best_constituent_index(scores: Mapping[str, float | None], n: int) -> int:
    """Pick the index of the best-scoring ``constituent#i`` entry in ``scores``.

    Falls back to index 0 when no ``constituent#i`` key is present (a
    non-conforming gate) -- a conservative, deterministic default rather
    than raising, since exhausting attempts must never be a hard failure.
    """
    best_index = 0
    best_score = float("-inf")
    found = False
    for key, value in scores.items():
        if not key.startswith("constituent#"):
            continue
        try:
            index = int(key.removeprefix("constituent#"))
        except ValueError:
            continue
        if not (0 <= index < n):
            continue
        found = True
        candidate_score = value if value is not None else float("-inf")
        if candidate_score > best_score:
            best_score = candidate_score
            best_index = index
    return best_index if found else 0


def _fallback_draft(group: SkillGroup, last_result: GateResult | None) -> SkillDraft:
    """Return the best constituent unmerged — the fallback when no merge is accepted.

    "Best" per the last ``GateResult`` computed during the retry loop (see
    :func:`_best_constituent_index`) -- the same aggregate every gate in
    this module uses, so no extra scoring happens here.
    """
    scores = last_result.scores if last_result is not None else {}
    index = _best_constituent_index(scores, len(group.members))
    return group.members[index]


# ---------------------------------------------------------------------------
# merge_group -- the public entry point
# ---------------------------------------------------------------------------


def _render_retry_input(base_text: str, last_result: GateResult | None) -> str:
    """Append the prior rejection's rationale to ``base_text`` for a retry attempt.

    ``last_result`` is ``None`` on the first attempt (no prior rejection to
    report), in which case ``base_text`` is returned unchanged -- see
    :func:`_merge_with_gate`'s "retries carry rejection context" note.
    """
    if last_result is None:
        return base_text
    return (
        f"{base_text}\n\n"
        f"Previous merge attempt was rejected by the quality gate:\n"
        f"{last_result.rationale}\n"
        f"Produce an improved merge that addresses this."
    )


async def _merge_with_gate(
    group: SkillGroup,
    gate: MergeGate,
    fn: MergeFn,
    max_attempts: int,
) -> SkillDraft:
    """Explicit propose->gate->retry loop; falls back to the best constituent.

    See this module's docstring for why this loop lives here rather than as
    a literal post-condition on the merge ``@ai_function`` template, and for
    why each retry's input carries the prior rejection's rationale (see
    :func:`_render_retry_input`).

    Raises:
        ValueError: ``fn`` returned a ``MergedSkillCandidate`` violating
            :func:`_check_merged_candidate` -- a contract-violating
            ``merge_fn`` is a programming error the caller must see, not
            something folded into an ordinary gate rejection.
        Exception: Any exception raised by ``fn`` or ``gate.accept`` -- see
            :func:`merge_group`'s ``Raises`` section. Only a *rejection*
            (``GateResult(accepted=False, ...)``) is treated as ordinary
            business and folded into the fallback; an infrastructure fault
            (a raised exception from either callable) propagates unchanged.
    """
    constituent_skills = [_skill_from_draft(d) for d in group.members]
    base_text = _render_constituents(group.members)

    last_result: GateResult | None = None
    for _attempt in range(max_attempts):
        retry_input = _render_retry_input(base_text, last_result)
        candidate = await fn(retry_input)
        check = _check_merged_candidate(candidate)
        if check is not None and not check.passed:
            raise ValueError(
                f"merge_fn returned an invalid MergedSkillCandidate for group {group.label!r}: {check.message}"
            )
        merged_skill = _skill_from_candidate(candidate)
        last_result = await gate.accept(merged_skill, constituent_skills)
        if last_result.accepted:
            return _draft_from_candidate(candidate, group)

    return _fallback_draft(group, last_result)


async def merge_group(
    group: SkillGroup,
    *,
    gate: MergeGate,
    merge_fn: MergeFn | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> list[SkillDraft]:
    """Merge ``group``'s members into one candidate, gated by ``gate``.

    Args:
        group: The group to merge. Passed through unmerged (see below) when
            it is the ``"ungrouped"`` bucket or has at most one member.
        gate: Decides accept-or-fallback for each merge attempt. ``None``
            is not accepted -- callers wanting the default policy construct
            ``MetricGate()`` explicitly.
        merge_fn: Merge callable; ``None`` (default) dispatches to the real
            module-level ``@ai_function`` template (:func:`_default_merge`).
            Inject a fake matching :data:`MergeFn` in tests -- see this
            module's docstring on the injection seam.
        max_attempts: Propose->gate retries before falling back to the best
            constituent (see :func:`_fallback_draft`). Must be >= 1 --
            validated before the empty/singleton/``"ungrouped"`` passthrough,
            so an invalid value always raises regardless of the group's shape.

    Returns:
        ``group.members`` unchanged (singleton/``"ungrouped"``), or a
        single-element list holding either the accepted merged
        ``SkillDraft`` or -- when every attempt was rejected -- the best
        constituent's own ``SkillDraft``, unmerged. Never empty for a
        non-empty input group; never raises on gate *rejection*.

    Raises:
        ValueError: ``max_attempts < 1``, or ``merge_fn`` (real or fake)
            returned a ``MergedSkillCandidate`` failing
            :func:`_check_merged_candidate` (defense-in-depth; see this
            module's docstring).
        Exception: Whatever ``merge_fn`` or ``gate.accept`` themselves raise
            propagates unchanged -- deliberately **not** caught and folded
            into the fallback. The "fallback, not hard failure" rule (see
            the module docstring and this function's ``Returns``) covers
            gate *rejection* only, i.e. an ordinary
            ``GateResult(accepted=False, ...)`` verdict -- that is business
            as usual and retried/falls back. A raised exception is a
            different thing: an infrastructure fault (a network error, a
            programmer error in a caller-supplied ``merge_fn``/``gate``,
            ``AIFunctionError`` from the real template exhausting its own
            shape-validation retries, ...) that the caller needs to see,
            not one this function is positioned to recover from silently.
    """
    # Argument validation comes first: an invalid `max_attempts` is a caller
    # bug, and must not be silently tolerated just because *this* group
    # happens to take the empty/singleton/ungrouped passthrough below.
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if not group.members:
        return []
    if group.label == UNGROUPED_LABEL or len(group.members) <= 1:
        return list(group.members)

    fn = merge_fn if merge_fn is not None else _default_merge
    draft = await _merge_with_gate(group, gate, fn, max_attempts)
    return [draft]
