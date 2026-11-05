"""ScienceWorld evaluation: task selection, episode runs, and result analysis.

The fixed train/test variations, shared player prompt, run records, evaluator,
and with/without reports live together. main.py owns the learning workflow;
this module supplies its tasks and measures the frozen library on held-out
variations without changing the prompt used during learning.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import statistics
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scienceworld_env import ScienceWorldSession, Variation

from ai_functions import ai_function
from ai_functions.experimental.skills import observe_skills, skill_event
from ai_functions.experimental.skills.observation import trace_observed
from ai_functions.experimental.skills.skill_library import (
    SkillDeliveryError,
    SkillLibraryBackend,
    SkillToolbox,
    backend_from_dirs,
)
from ai_functions.types import ThreadId
from ai_functions.types.events import (
    CompletedEvent,
    Event,
    MessageAssistantCompleteEvent,
    ResultEvent,
    StartedEvent,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_logger = logging.getLogger(__name__)

# --- Fixed tasks ---

# Haiku, not Sonnet: skills can only show a gain where the agent has room to
# improve, and Sonnet 4.6 solved 7 of 8 boil train variations perfectly with no
# skills at all. The claim this supports is "mined skills help an agent with
# headroom", not "skills help frontier models" -- report it that way.
DEFAULT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

TASK = "measure-melting-point-unknown-substance"
"""The ScienceWorld task family. Every id below names a variation of it."""

# Chosen once, by a measured no-skill pilot of Claude Haiku 4.5 at a 60-step
# budget: each candidate variation was played several times with no skills,
# and a variation was kept only where that baseline landed strictly between
# floor and ceiling -- a task already solved has nothing to gain, and one
# never solved has nothing to show. The kept train variations alternate the
# family's two failure modes, stalling partway (a partial score) and
# committing to a wrong answer (-100), so that a prefix of the list carries
# both; that is why the order below is not sorted. The pilot is not part of
# the demo: its outcome is these two lists. A different model or step budget
# wants its own selection, and these ids are the only thing to change.
TRAFFIC_VARIATIONS = [9, 11, 82, 39, 12, 119, 93, 87, 134, 68, 50, 103]
"""Train-split variations the function is simply called on (mining and gate material)."""

EVAL_VARIATIONS = [283, 243, 254, 249, 275, 296, 255, 225, 260, 299, 272, 265, 229, 233, 298]
"""Test-split variations the final library is measured on. Disjoint from the above by split."""


# --- Shared player prompt ---

# The player's prompt template, shared with ``main.py`` so the evaluation's
# arms and the decorated function play with the same words. Its indentation is part of the text the model sees
# (it began as a nested function's docstring), so it is kept byte for byte.
PLAY_PROMPT = """You are an agent acting in the ScienceWorld text environment.

        Your task:
        {task_description}

        Opening observation:
        {opening_observation}

        {skills}

        Act by calling `sw_act` with one short imperative action at a time, reading
        each observation before choosing the next action. Explore to find what you
        need before trying to use it. Work until the task is complete or you can
        make no further progress, then reply with a one-paragraph summary of what
        you did. Do not ask for confirmation; you are running unattended.
        """

SKILLS_SLOT = "        {skills}\n\n"
"""The line ``PLAY_PROMPT`` shows the skill index on. ``@skillify`` puts the index in the system prompt instead."""


# --- Run records and event traces ---

OracleFn = Callable[[Path, Any], "OracleResult"]


@dataclass
class OracleResult:
    """A benchmark oracle's verdict on one run."""

    passed: bool
    score: float  # [0, 1]
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that ``score`` is within ``[0, 1]``."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")


@dataclass
class TraceEntry:
    """One step in a typed agent transcript."""

    entry_type: Literal["thought", "action", "observation", "error"]
    timestamp: str  # ISO-8601
    content: str
    tool_name: str | None = None
    action_type: Literal["read", "write", "execute", "search", "communicate", "reason", "other"] | None = None


@dataclass
class TokenCounts:
    """Input/output token usage for one agent run."""

    input_tokens: int
    output_tokens: int


@dataclass
class RichExecutionTrace:
    """What one run produced: its output, oracle result, transcript and token use."""

    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    oracle_result: OracleResult | None = None
    transcript: list[TraceEntry] = field(default_factory=list)
    token_counts: TokenCounts | None = None


@dataclass(frozen=True)
class RunSpec:
    """One task to run under one or more arms, optionally repeated.

    Attributes:
        task_id: Stable identifier for this task; names the spawned threads.
        prompt: Prompt sent to the agent thread.
        workspace: The task's workspace directory.
        skills: Either a plain list of skill directories or a
            ``SkillLibraryBackend``, installed for the ``"skill_on"`` arm only.
        oracle_fn: Optional ``(run_dir, output) -> OracleResult``.
        repeats: Number of times to run each arm.
        arms: Which arms to run, in the order results are returned. Only
            the literal ``"skill_on"`` arm gets skills.
    """

    task_id: str
    prompt: str
    workspace: Path
    skills: Sequence[Path] | SkillLibraryBackend
    oracle_fn: OracleFn | None = None
    repeats: int = 1
    arms: tuple[str, ...] = ("skill_on", "skill_off")


@dataclass(frozen=True)
class RunResult:
    """One completed (spec, arm, repeat) run.

    Attributes:
        prompt: The originating ``RunSpec.prompt``.
        thread_id: The runtime-assigned id of the thread this run executed
            on (distinct from ``thread_name``, the human-readable label).
    """

    task_id: str
    arm: str
    repeat: int
    trace: RichExecutionTrace
    run_dir: Path
    thread_name: str
    prompt: str
    thread_id: ThreadId


# ── Skill-invocation contract (spec V1) ─────────────────────────────────────
# The single place that knows which tool invokes a skill and how to read the
# slug out of its arguments.

_SKILL_TOOL_NAMES = ("Skill",)


def _extract_skill_slug(arguments: dict[str, object]) -> str | None:
    """Return the skill slug from a ``Skill`` tool call's arguments, or ``None``.

    A missing key or a non-string value (both technically malformed calls a
    misbehaving or future model variant could still produce) are treated
    alike: skipped, not raised.
    """
    slug = arguments.get("skill")
    return slug if isinstance(slug, str) else None


# ── Tool name -> TraceEntry.action_type map (spec 4.4, binding, exact) ─────

_ActionType = Literal["read", "write", "execute", "search", "communicate", "reason", "other"]

_ACTION_TYPE_BY_TOOL: dict[str, _ActionType] = {
    "Read": "read",
    "LS": "read",
    "WebFetch": "read",
    "Glob": "search",
    "Grep": "search",
    "WebSearch": "search",
    "Write": "write",
    "Edit": "write",
    "MultiEdit": "write",
    "NotebookEdit": "write",
    "Bash": "execute",
    "Task": "communicate",
}


def _action_type_for(tool_name: str) -> _ActionType:
    """Map a tool name onto ``TraceEntry.action_type``; unmapped -> ``"other"``."""
    return _ACTION_TYPE_BY_TOOL.get(tool_name, "other")


def _identity(payload: str) -> Any:  # noqa: ANN401
    return payload


def _iso_timestamp(ts: float) -> str:
    """Render a ``BaseEvent.timestamp`` epoch float as ISO-8601 (UTC)."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _tool_call_content(tool_name: str, arguments: dict[str, object]) -> str:
    """One-line ``TraceEntry.content`` for a tool call.

    A ``Skill`` call with a valid slug renders as the bare slug (no other
    argument keys) so ``InvocationTimingMetric``'s
    ``_find_skill_invocations`` -- which reads the slug as the first
    whitespace-delimited token of ``content`` -- finds it directly. Every
    other call (including a malformed ``Skill`` call whose slug did not
    extract) falls back to a JSON rendering of its arguments.
    """
    if tool_name in _SKILL_TOOL_NAMES:
        slug = _extract_skill_slug(arguments)
        if slug is not None:
            return slug
    try:
        return json.dumps(arguments, default=str)[:300]
    except (TypeError, ValueError):
        return str(arguments)[:300]


def _tool_result_content(content: Sequence[Any]) -> str:
    """Flatten a ``ToolResultEvent.content`` block list into display text."""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if text is not None:
            parts.append(str(text))
            continue
        payload = block.get("json")
        if payload is not None:
            parts.append(str(payload))
    return " ".join(parts)[:300]


def _thought_entries(event: MessageAssistantCompleteEvent) -> list[TraceEntry]:
    """One ``TraceEntry(entry_type="thought")`` per text/reasoning block."""
    entries: list[TraceEntry] = []
    ts = _iso_timestamp(event.timestamp)
    for block in event.content:
        text = block.get("text")
        if text:
            entries.append(TraceEntry(entry_type="thought", timestamp=ts, content=str(text), action_type="reason"))
            continue
        reasoning = block.get("reasoningContent")
        reasoning_text = reasoning.get("reasoningText", {}).get("text") if isinstance(reasoning, dict) else None
        if reasoning_text:
            entries.append(
                TraceEntry(entry_type="thought", timestamp=ts, content=str(reasoning_text), action_type="reason")
            )
    return entries


def trace_from_events(
    events: Sequence[Event],
    *,
    deserialize_result: Callable[[str], Any] | None = None,
    oracle_result: OracleResult | None = None,
) -> RichExecutionTrace:
    """Flatten a coordinator event log into a ``RichExecutionTrace``.

    Args:
        events: A thread's event log, e.g. from
            ``await coordinator.get_events(thread_id)``. Any iteration order
            other than append order is not supported -- ``STARTED``/
            ``COMPLETED`` pairing and "first"/"last" event selection assume
            the log's natural order.
        deserialize_result: Applied to the (last) ``RESULT`` event's
            ``payload`` to produce ``output``. Defaults to the identity
            function, matching ``ExecutionTrace.output``'s permissive
            ``Any`` contract when no thread-specific deserializer is given.
        oracle_result: Passed through verbatim onto the returned trace.

    Returns:
        A populated ``RichExecutionTrace``. See the module docstring for the
        exact per-event-kind field mapping.
    """
    deserialize = deserialize_result or _identity

    transcript: list[TraceEntry] = []
    skills_used: list[str] = []
    seen_skills: set[str] = set()

    sum_input = 0
    sum_output = 0
    sum_cache_read = 0
    sum_cache_write = 0

    started_ts: float | None = None
    completed_ts: float | None = None
    output_payload: str | None = None

    for event in events:
        if isinstance(event, StartedEvent):
            if started_ts is None:
                started_ts = event.timestamp
        elif isinstance(event, CompletedEvent):
            if completed_ts is None:
                completed_ts = event.timestamp
        elif isinstance(event, MessageAssistantCompleteEvent):
            transcript.extend(_thought_entries(event))
        elif isinstance(event, ToolCallEvent):
            is_skill_call = event.tool_name in _SKILL_TOOL_NAMES
            if is_skill_call:
                slug = _extract_skill_slug(event.arguments)
                if slug is not None and slug not in seen_skills:
                    skills_used.append(slug)
                    seen_skills.add(slug)
            # Normalize any alternate skill-tool name onto the canonical
            # "Skill" so every downstream consumer of TraceEntry.tool_name
            # can match on the literal "Skill" without knowing about _SKILL_TOOL_NAMES. A
            # no-op today (_SKILL_TOOL_NAMES == ("Skill",)); guards against
            # a future contract change (spec V1) touching only this module.
            tool_name = "Skill" if is_skill_call else event.tool_name
            transcript.append(
                TraceEntry(
                    entry_type="action",
                    timestamp=_iso_timestamp(event.timestamp),
                    content=_tool_call_content(event.tool_name, event.arguments),
                    tool_name=tool_name,
                    action_type=_action_type_for(event.tool_name),
                )
            )
        elif isinstance(event, ToolResultEvent):
            transcript.append(
                TraceEntry(
                    entry_type="error" if event.status == "error" else "observation",
                    timestamp=_iso_timestamp(event.timestamp),
                    content=_tool_result_content(event.content),
                )
            )
        elif isinstance(event, TokenUsageEvent):
            usage = event.token_usage
            sum_input += usage.input_tokens
            sum_output += usage.output_tokens
            sum_cache_read += usage.cache_read_tokens
            sum_cache_write += usage.cache_write_tokens
        elif isinstance(event, ResultEvent):
            output_payload = event.payload

    latency_seconds = completed_ts - started_ts if started_ts is not None and completed_ts is not None else None
    # A negative delta means the log isn't a single clean run (e.g. a leftover
    # COMPLETED from a prior cycle precedes this cycle's STARTED) -- treat it
    # as unknown rather than fabricate a measurement by clamping to 0.
    if latency_seconds is not None and latency_seconds < 0:
        latency_seconds = None

    metadata: dict[str, Any] = {
        "skills_used": skills_used,
        "total_tokens": sum_input + sum_output + sum_cache_read + sum_cache_write,
    }
    if latency_seconds is not None:
        metadata["latency_seconds"] = latency_seconds

    output = deserialize(output_payload) if output_payload is not None else None

    return RichExecutionTrace(
        output=output,
        metadata=metadata,
        transcript=transcript,
        token_counts=TokenCounts(input_tokens=sum_input, output_tokens=sum_output),
        oracle_result=oracle_result,
    )


# --- Play held-out episodes ---


def _make_player(session: ScienceWorldSession, model: str | None, extra_tools: Sequence[object] = ()):
    """Build the AI Function for one episode, with that episode's tools bound.

    The function is created per run rather than at import time because its
    tools close over a specific ``ScienceWorldSession`` -- two concurrent
    runs must not share an environment.
    """

    # coordinator_tools_enabled=False: this episode has no peer threads to
    # discover or message, and offering list_threads/send_message alongside
    # sw_act only gives the model irrelevant options to spend turns on.
    def play(task_description: str, opening_observation: str, skills: str) -> str: ...

    play.__doc__ = PLAY_PROMPT
    return ai_function(model=model, tools=[*session.make_tools(), *extra_tools], coordinator_tools_enabled=False)(play)


def _backend_for(skills: Sequence[Path] | SkillLibraryBackend, run_label: str) -> SkillLibraryBackend | None:
    """Turn either form ``RunSpec.skills`` allows into a backend, or ``None`` when it is empty.

    ``RunSpec.skills`` is ``Sequence[Path] | SkillLibraryBackend``; both forms
    are honoured, so a with-skill arm never silently runs with no skill. A list
    of skill directories is copied into a fresh root, one subdirectory per
    skill, which is the layout ``SkillLibraryBackend`` reads.

    Raises:
        TypeError: ``skills`` is neither form -- a contract violation, never
            silently read as "no skills".
        FileNotFoundError: a listed directory does not exist.
    """
    if isinstance(skills, SkillLibraryBackend):
        return skills
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError(f"RunSpec.skills must be a SkillLibraryBackend or a sequence of paths, got {type(skills)!r}")
    dirs = [Path(d) for d in skills]
    if not dirs:
        return None
    return backend_from_dirs(dirs, Path(tempfile.mkdtemp(prefix=f"skills-{run_label}-")), actor_id=f"eval-{run_label}")


class AIFunctionEvalDriver:
    """Runs ``RunSpec``s as AI Function episodes.

    Args:
        variations: Maps ``RunSpec.task_id`` to the ScienceWorld variation
            that task is. The driver owns environment construction, so a
            spec carries only an id and a prompt.
        model: Model id passed to every ``@ai_function``. ``None`` uses the
            library default.
        max_concurrency: Upper bound on in-flight episodes. Each episode
            holds a JVM-backed environment, so this is a memory bound as
            much as a rate limit.
        run_root: Parent for per-run directories. ``RunSpec.workspace`` is
            unused -- an AI Function episode has no workspace to copy --
            but ``RunResult.run_dir`` is part of the contract, so each run
            still gets a real directory for artifacts.
    """

    def __init__(
        self,
        variations: dict[str, Variation],
        *,
        model: str | None = None,
        max_concurrency: int = 3,
        run_root: Path | None = None,
        step_limit: int = 150,
        max_attempts: int = 4,
        retry_backoff: int = 20,
        observation: dict[str, Any] | None = None,
    ) -> None:
        self.variations = variations
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._run_root = run_root
        self._step_limit = step_limit
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff
        self._observation = observation or {}

    async def run(self, specs: Sequence[RunSpec]) -> list[RunResult]:
        """Run every (spec, arm, repeat) combination, bounded by ``max_concurrency``."""
        jobs = [
            self._run_one(spec, arm, repeat) for spec in specs for arm in spec.arms for repeat in range(spec.repeats)
        ]
        return list(await asyncio.gather(*jobs))

    async def _run_one(self, spec: RunSpec, arm: str, repeat: int) -> RunResult:
        """One run, retried on transient infrastructure failures.

        Without this, a single failed episode -- e.g. Bedrock returning
        ``ServiceUnavailableException`` after its own retries -- aborts the
        whole batch and with it a multi-hour unattended run. Each attempt
        starts a fresh environment session, because a half-played episode
        cannot be resumed.

        :class:`SkillDeliveryError` is never retried -- that one means the
        harness is broken, and it should stop the run loudly.
        """
        with observe_skills(
            **self._observation,
            run_id=f"evaluation-{uuid.uuid4().hex}",
            phase="evaluation",
            function="play",
            task_id=spec.task_id,
            arm=arm,
            repeat=repeat,
        ):
            try:
                return await self._run_observed(spec, arm, repeat)
            except BaseException as exc:
                skill_event("run.finished", status="failed", error=f"{type(exc).__name__}: {exc}")
                raise

    async def _run_observed(self, spec: RunSpec, arm: str, repeat: int) -> RunResult:
        last: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._attempt(spec, arm, repeat)
            except SkillDeliveryError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient infra failures are the norm here
                last = exc
                if attempt == self._max_attempts:
                    break
                delay = self._retry_backoff * 2 ** (attempt - 1)
                _logger.warning(
                    "%s/%s/%d failed (attempt %d/%d): %s: %s -- retrying in %ds",
                    spec.task_id,
                    arm,
                    repeat,
                    attempt,
                    self._max_attempts,
                    type(exc).__name__,
                    str(exc)[:160],
                    delay,
                )
                await asyncio.sleep(delay)
        assert last is not None
        _logger.error("%s/%s/%d failed after %d attempts", spec.task_id, arm, repeat, self._max_attempts)
        raise last

    async def _attempt(self, spec: RunSpec, arm: str, repeat: int) -> RunResult:
        async with self._semaphore:
            variation = self.variations[spec.task_id]
            session = ScienceWorldSession(variation=variation, step_limit=self._step_limit)
            opening = await asyncio.to_thread(session.start)

            backend = _backend_for(spec.skills, f"{spec.task_id}-{arm}-{repeat}") if arm == "skill_on" else None
            if arm == "skill_on" and backend is None:
                _logger.warning(
                    "%s/%s/%d: skill_on run with no skills supplied -- identical to skill_off",
                    spec.task_id,
                    arm,
                    repeat,
                )
            toolbox = SkillToolbox(backend) if backend is not None else None
            if arm == "skill_on" and backend is not None and not (toolbox and toolbox.available):
                raise SkillDeliveryError(
                    f"{spec.task_id}/{arm}/{repeat}: skill_on run was given a library with no skills in it; "
                    "refusing to run a with-skill arm that exposes nothing"
                )
            skills_text = toolbox.index_text() if toolbox else ""
            skill_event(
                "run.started",
                skill_root=backend.root if backend else None,
                inputs={"task_description": variation.description, "opening_observation": opening},
                trace_delivery="live",
            )

            play = _make_player(session, self.model, extra_tools=[toolbox.make_tool()] if toolbox else ())
            try:
                result = await trace_observed(
                    play,
                    task_description=variation.description,
                    opening_observation=opening,
                    skills=skills_text,
                )
            except Exception:
                _logger.exception("run %s/%s/%d failed", spec.task_id, arm, repeat)
                raise
            finally:
                session.close()

            events = await result.coordinator.get_events(result.thread_id)
            skill_event("run.events", events=events)
            skill_event("run.finished", status="completed", outcome=max(0.0, session.score) / 100.0)
            trace = trace_from_events(
                events,
                oracle_result=OracleResult(
                    # ScienceWorld scores 0-100; normalize so a metric's
                    # 0-1 convention holds and the gate's min_gain reads as
                    # a fraction rather than a raw point count.
                    passed=session.score >= 100.0,
                    score=max(0.0, session.score) / 100.0,
                    details={
                        "raw_score": session.score,
                        "steps": session.steps,
                        # What the treatment was: the skills this run could see, and the ones
                        # the agent chose to load. A with-skill run must expose a non-empty
                        # index; loading is the agent's decision, so an empty `skills_loaded`
                        # is a result (invocation rate), not a delivery failure.
                        "skills_available": list(toolbox.available) if toolbox else [],
                        "skills_loaded": list(toolbox.loaded) if toolbox else [],
                    },
                ),
            )

            run_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{spec.task_id}-{arm}-{repeat}-",
                    dir=str(self._run_root) if self._run_root else None,
                )
            )
            return RunResult(
                task_id=spec.task_id,
                arm=arm,
                repeat=repeat,
                trace=trace,
                run_dir=run_dir,
                thread_name=f"{spec.task_id}/{arm}/{repeat}",
                prompt=spec.prompt,
                thread_id=result.thread_id,
            )


# --- With/without analysis ---


def _spec(variation: Variation, arms: tuple[str, ...], repeats: int) -> RunSpec:
    """A ``RunSpec`` for one variation.

    ``workspace`` is required by the dataclass but unused by this driver --
    an AI Function episode has no directory to copy -- so it points at the
    example directory rather than inventing a temp tree per task.
    """
    return RunSpec(
        task_id=variation.task_id,
        prompt=variation.description,
        workspace=Path(__file__).parent,
        skills=[],
        arms=arms,
        repeats=repeats,
    )


def _paired_permutation(
    rows: list[dict],
    key: str,
    *,
    clamp_floor: float | None,
    greater_is_better: bool,
    iterations: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """Pooled with-skill minus baseline on ``key``, with a within-task permutation p-value.

    Arm labels are shuffled only among runs of the same task, so the null
    distribution keeps every task's difficulty fixed. The p-value is
    one-sided in the direction the metric counts as better.
    """
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"skill_on": [], "skill_off": []})
    for row in rows:
        value = row.get(key)
        if value is None or row["arm"] not in ("skill_on", "skill_off"):
            continue
        value = float(value)
        if clamp_floor is not None:
            value = max(clamp_floor, value)
        by_task[row["task_id"]][row["arm"]].append(value)
    by_task = {t: arms for t, arms in by_task.items() if arms["skill_on"] and arms["skill_off"]}
    if not by_task:
        return {"diff": float("nan"), "on": float("nan"), "off": float("nan"), "p": float("nan"), "n_tasks": 0}

    def pooled(groups: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
        on = [v for arms in groups.values() for v in arms["skill_on"]]
        off = [v for arms in groups.values() for v in arms["skill_off"]]
        return statistics.fmean(on), statistics.fmean(off)

    on_mean, off_mean = pooled(by_task)
    observed = on_mean - off_mean
    sign = 1.0 if greater_is_better else -1.0
    rng = random.Random(seed)
    at_least = 0
    for _ in range(iterations):
        shuffled = {}
        for task_id, arms in by_task.items():
            pool = arms["skill_on"] + arms["skill_off"]
            rng.shuffle(pool)
            n_on = len(arms["skill_on"])
            shuffled[task_id] = {"skill_on": pool[:n_on], "skill_off": pool[n_on:]}
        s_on, s_off = pooled(shuffled)
        if sign * (s_on - s_off) >= sign * observed:
            at_least += 1
    return {"diff": observed, "on": on_mean, "off": off_mean, "p": at_least / iterations, "n_tasks": len(by_task)}


def _report(title: str, rows: list[tuple[str, str]]) -> None:
    width = max((len(k) for k, _ in rows), default=0)
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}")
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    print("=" * 78)


def report_selection(traffic: Sequence[Variation], evaluation: Sequence[Variation]) -> None:
    """Print the "Task selection" block: exactly which variations this run uses."""
    _report(
        f"Task selection ({TASK}, fixed variation ids)",
        [
            ("traffic (train split)", ", ".join(str(v.variation) for v in traffic) or "- (--evaluate-only)"),
            ("evaluation (test split)", f"{len(evaluation)} tasks: " + ", ".join(str(v.variation) for v in evaluation)),
        ],
    )


def report_evaluation(
    eval_results: Sequence[RunResult], evaluation: Sequence[Variation], out_root: Path, *, repeats: int
) -> dict:
    """Write ``eval_results.json`` and ``analysis.json`` and print the with/without tables and headline.

    The pre-registered analysis: fixed before any with-skill result
    existed. Returns what ``analysis.json`` holds.
    """
    # Persist every run before summarizing, so the distribution behind the
    # headline can be examined afterwards rather than inferred from a median.
    rows = [
        {
            "task_id": r.task_id,
            "arm": r.arm,
            "repeat": r.repeat,
            "score": (r.trace.oracle_result.details or {}).get("raw_score") if r.trace.oracle_result else None,
            "steps": (r.trace.oracle_result.details or {}).get("steps") if r.trace.oracle_result else None,
            # The primary metric. ScienceWorld pays partial credit -- this family
            # gives 79 of 100 for the mechanics alone, and the final answer action
            # is what earns the rest -- so a mean score hides whether the task was
            # actually finished: a mean near 90 can hide a completion rate near 70%.
            "success": 1.0
            if r.trace.oracle_result and (r.trace.oracle_result.details or {}).get("raw_score", 0) >= 100
            else 0.0,
            # What the treatment was, so it is checkable rather than assumed: the
            # skills this run could see, and the ones the agent chose to load
            # through the Skill tool.
            "skills_available": (r.trace.oracle_result.details or {}).get("skills_available")
            if r.trace.oracle_result
            else None,
            "skills_loaded": (r.trace.oracle_result.details or {}).get("skills_loaded")
            if r.trace.oracle_result
            else None,
            "input_tokens": r.trace.token_counts.input_tokens if r.trace.token_counts else None,
            "output_tokens": r.trace.token_counts.output_tokens if r.trace.token_counts else None,
        }
        for r in eval_results
    ]
    (out_root / "eval_results.json").write_text(json.dumps(rows, indent=2))
    _logger.info("wrote %d per-run rows to %s", len(rows), out_root / "eval_results.json")

    # Outcome per task, per arm, across repeats -- median and range, because
    # a single mean over n=5 hides exactly the variance the repeats bought.
    print("\nEvaluation (ScienceWorld score, 0-100)\n" + "-" * 78)
    print(f"{'task':<16}{'baseline':>22}{'with skills':>22}{'delta':>10}")
    deltas: list[float] = []
    for variation in evaluation:
        by_arm: dict[str, list[float]] = {"skill_off": [], "skill_on": []}
        for result in eval_results:
            if result.task_id != variation.task_id or result.trace.oracle_result is None:
                continue
            by_arm[result.arm].append(result.trace.oracle_result.details.get("raw_score", 0.0))
        if not by_arm["skill_on"] or not by_arm["skill_off"]:
            continue
        off, on = by_arm["skill_off"], by_arm["skill_on"]
        delta = statistics.median(on) - statistics.median(off)
        deltas.append(delta)
        print(
            f"{variation.task_id:<16}"
            f"{f'{statistics.median(off):.0f} [{min(off):.0f}-{max(off):.0f}]':>22}"
            f"{f'{statistics.median(on):.0f} [{min(on):.0f}-{max(on):.0f}]':>22}"
            f"{delta:>+10.1f}"
        )

    # Steps are continuous where the ScienceWorld score is nearly binary
    # (complete, or stalled at partial credit), so this axis can register a
    # difference the outcome column rounds away.
    print("\nEfficiency (environment steps; lower is better)\n" + "-" * 78)
    print(f"{'task':<16}{'baseline':>22}{'with skills':>22}{'delta':>10}")
    for variation in evaluation:
        by_arm: dict[str, list[float]] = {"skill_off": [], "skill_on": []}
        for row in rows:
            if row["task_id"] == variation.task_id and row["steps"] is not None:
                by_arm[row["arm"]].append(float(row["steps"]))
        if not by_arm["skill_on"] or not by_arm["skill_off"]:
            continue
        off, on = by_arm["skill_off"], by_arm["skill_on"]
        print(
            f"{variation.task_id:<16}"
            f"{f'{statistics.median(off):.0f} [{min(off):.0f}-{max(off):.0f}]':>22}"
            f"{f'{statistics.median(on):.0f} [{min(on):.0f}-{max(on):.0f}]':>22}"
            f"{statistics.median(on) - statistics.median(off):>+10.1f}"
        )

    # --- Pre-registered analysis ---------------------------------------------
    # Fixed before any with-skill result existed. Primary: the mean paired
    # difference in score over every (task, repeat), with negative scores
    # clamped to 0 -- ScienceWorld scores a task-destroying action at -100,
    # and one such run would otherwise swing a pooled mean by itself. The
    # p-value comes from shuffling arm labels *within* each task, so task
    # difficulty never masquerades as a skill effect. Secondary: steps.
    primary = _paired_permutation(rows, "success", clamp_floor=None, greater_is_better=True)
    score_mean = _paired_permutation(rows, "score", clamp_floor=0.0, greater_is_better=True)
    secondary = _paired_permutation(rows, "steps", clamp_floor=None, greater_is_better=False)
    on_rows = [row for row in rows if row["arm"] == "skill_on"]
    loads = [s for row in on_rows for s in (row["skills_loaded"] or [])]
    invocation = {
        # Delivery: every with-skill run must have *seen* a non-empty index.
        "skill_on_runs": len(on_rows),
        "skill_on_runs_with_index": sum(1 for row in on_rows if row["skills_available"]),
        # Behaviour: loading is the agent's choice, so this is a result.
        "skill_on_runs_that_loaded_a_skill": sum(1 for row in on_rows if row["skills_loaded"]),
        "mean_loads_per_skill_on_run": (len(loads) / len(on_rows)) if on_rows else 0.0,
        "loads_by_skill": dict(collections.Counter(loads).most_common()),
    }
    analysis = {
        "primary_success_rate": primary,
        "secondary_steps": secondary,
        "reported_score_mean": score_mean,
        "invocation": invocation,
    }
    (out_root / "analysis.json").write_text(json.dumps(analysis, indent=2))

    if deltas:
        _report(
            "Headline",
            [
                (
                    "PRIMARY  success rate (task completed)",
                    f"{100 * primary['on']:.1f}% with skills vs {100 * primary['off']:.1f}% baseline "
                    f"({100 * primary['diff']:+.1f} pts, one-sided p = {primary['p']:.3f})",
                ),
                (
                    "         mean score (partial credit)",
                    f"{score_mean['on']:.1f} / {score_mean['off']:.1f}  ({score_mean['diff']:+.1f}, "
                    f"p = {score_mean['p']:.3f}) -- reported, not the primary",
                ),
                ("SECONDARY mean steps saved", f"{-secondary['diff']:+.1f}  (one-sided p = {secondary['p']:.3f})"),
                (
                    "skill index delivered",
                    f"{invocation['skill_on_runs_with_index']} / {invocation['skill_on_runs']} with-skill runs",
                ),
                (
                    "agent loaded >=1 skill",
                    f"{invocation['skill_on_runs_that_loaded_a_skill']} / "
                    f"{invocation['skill_on_runs']} with-skill runs",
                ),
                ("tasks", str(len(deltas))),
                ("repeats per arm", str(repeats)),
                ("median of per-task deltas", f"{statistics.median(deltas):+.1f}"),
                ("mean of per-task deltas", f"{statistics.fmean(deltas):+.1f}"),
                (
                    "improved / regressed / tied",
                    f"{sum(d > 0 for d in deltas)} / {sum(d < 0 for d in deltas)} / {sum(d == 0 for d in deltas)}",
                ),
            ],
        )
    else:
        print("\nNo paired evaluation results to report.")
    return analysis
