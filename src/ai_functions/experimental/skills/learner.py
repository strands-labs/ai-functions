"""``ReplayDriver`` and ``SkillLearner`` -- replaying recorded calls, and the rounds behind ``@skillify``.

Two parts:

  * **``ReplayDriver``** re-runs recorded calls of an AI Function as gate
    trials. ``@skillify`` decides whether a candidate skill helps by
    replaying recent calls with and without it. This driver satisfies the
    learner's private ``_EvalDriver`` protocol (below): a ``_RunSpec``'s
    ``task_id`` is a recorded call id, its arguments come from the
    ``LibraryStore``, and the ``"skill_on"`` arm runs with ``spec.skills``
    shown through the index and ``Skill`` tool. Every other arm runs the
    function with no skills.

    It calls the undecorated AI Function directly, so a replay is never
    recorded as a call and never counts toward the next round.

    Outcome: ``score(arguments, result)`` when given, else ``1.0`` for a
    call that returned (its post-conditions passed) and ``0.0`` for one
    that raised ``AIFunctionError``. Any other exception is an
    infrastructure failure: it is retried, then raised, which fails the
    round and leaves the library as it was.

    With an ``environment``, every run -- each arm, each repeat, each
    attempt -- builds its own :class:`Episode` from the recorded arguments,
    runs with the episode's tools added, takes its outcome from
    ``episode.outcome(result)`` and always closes it. A factory that raises
    is an infrastructure failure.

  * **``SkillLearner``** turns the calls recorded since the last round into,
    possibly, a new library version:

    1. **Trigger.** ``should_induce`` (default :func:`_should_induce`) looks
       at the window's outcomes. No failure and no spread means nothing to
       mine: the round skips mining, and is declined unless it has staged
       revisions to gate.
    2. **Mine.** Scored calls go to the :class:`~.discovery.Miner` as
       :class:`RecordedCall` objects; an unscored call (an infrastructure
       error, a failing ``score``/``environment.outcome``) still counts
       toward the window but is never mined. Calls below ``pass_at`` have
       ``passed=False``.
    3. **Deduplicate.** A candidate named like an installed skill, or whose
       directory maps to the same parameter name as an installed skill or an
       earlier candidate (``foo_bar``/``foo-bar``), is dropped; one named for
       the skill index is rejected as ``"invalid"``. Revisions staged in
       ``pending/`` (e.g. by an optimizer writing through
       :class:`~.skill_library.SkillLibraryBackend`) are taken here and join
       the candidates first, each as ``Candidate(skill, replaces=slug)``; one
       whose skill is no longer installed is discarded as ``"stale"``, one
       that does not load as ``"invalid"``. A staged revision never triggers
       a round by itself, but every round that runs gates it.
    4. **Choose trial inputs.** Replayable, scored calls from *outside* the
       window, failed ones first. When there are none (always so in the first
       round) it uses the window's own and says so (``trials_overlap_mining``).
    5. **Gate.** The :class:`~.evaluation.Gate` returns one verdict per
       candidate. It replays trial inputs through ``replay(skill_dirs,
       calls, repeats)``, which the learner builds over its replay driver.
       The default (:class:`~.evaluation.CompletionGate`) measures each
       candidate added to the current library against the current library
       alone, judged on completion, after a budget check that drops
       candidates from the end of the list first, then trial inputs.
    6. **Publish.** The current skills, with admitted revisions swapped in,
       plus the admitted new ones, capped at ``max_skills`` by gain, become a
       new version. Revisions do not count against ``max_skills``.

    A round that raises leaves ``CURRENT`` unchanged, and so does a round
    that finds the library changed under it (e.g. by a rollback) before it
    publishes: it is reported as failed and publishes nothing, and the
    revisions it took are staged again. A round that
    runs -- declined, failed or not -- consumes its window and appends a
    :class:`RoundReport` to ``history.jsonl``, so a failing round cannot
    re-trigger on every call. Two kinds of round append no history and
    consume nothing, leaving the window pending for the next round: a
    cancelled one, and one refused because another round is already running
    in this process or another process holds the library's round lock (that
    refusal is returned as a ``"failed"`` report but not persisted).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.types import no_thread_scope
from ai_functions.types.events import Event

from .discovery import Miner, PromptMiner
from .evaluation import Candidate, CompletionGate, Gate, Replay, Verdict
from .observation import _scope, skill_event, trace_observed
from .skill_library import (
    INDEX,
    CallRecord,
    LibraryStore,
    Skill,
    SkillDeliveryError,
    SkillLibraryBackend,
    _param_name,
    backend_from_dirs,
    load_skill,
    with_skills,
)

if TYPE_CHECKING:
    from strands.models import Model

__all__ = [
    "Environment",
    "Episode",
    "RecordedCall",
    "ReplayDriver",
    "RoundReport",
    "Score",
    "SkillLearner",
    "SkillifyConfig",
    "bind_arguments",
    "close_episode",
    "episode_outcome",
    "open_episode",
    "outcome_of",
    "with_episode_tools",
]

_logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ── ReplayDriver: re-run recorded calls of an AI Function as gate trials ──


@dataclass(frozen=True)
class _RunSpec:
    """A recorded call to replay under each of ``arms``, ``repeats`` times; ``"skill_on"`` is shown ``skills``."""

    task_id: str
    skills: Sequence[Path]
    repeats: int = 1
    arms: tuple[str, ...] = ("skill_on", "skill_off")


@dataclass(frozen=True)
class _RunResult:
    """One run's outcome in ``[0, 1]`` (``None`` when unscored) and whether it reached ``pass_at``."""

    outcome: float | None
    passed: bool


Score = Callable[[Mapping[str, Any], Any], float]
"""``(arguments bound by name, result) -> outcome in [0, 1]``."""


class Episode(Protocol):
    """One run's stateful environment, built from that call's arguments.

    Attributes:
        tools: Tools bound to this run's environment; added to the run's tools.
    """

    tools: Sequence[Any]

    def outcome(self, result: Any) -> float:  # noqa: ANN401
        """The run's outcome in ``[0, 1]``, read from the environment's state."""
        ...

    def close(self) -> None:
        """Release the environment. Called exactly once per episode, however the run ended."""
        ...


Environment = Callable[[Mapping[str, Any]], Episode | Awaitable[Episode]]
"""``(arguments bound by name) -> Episode``, or an awaitable of one."""


def bind_arguments(fn: AIFunction[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The call's arguments keyed by parameter name."""
    return dict(inspect.signature(fn.prompt_fn).bind(*args, **kwargs).arguments)


def outcome_of(
    fn: AIFunction[..., Any],
    score: Score | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    result: Any,  # noqa: ANN401
) -> float | None:
    """Outcome of a call that returned; ``None`` when ``score`` raised or left ``[0, 1]``."""
    if score is None:
        return 1.0
    return _checked("score", lambda: score(bind_arguments(fn, args, kwargs), result))


def _checked(source: str, compute: Callable[[], float]) -> float | None:
    """``compute()`` as a float in ``[0, 1]``, or ``None`` (logged) when it raised or left that range."""
    try:
        value = float(compute())
    except Exception:  # noqa: BLE001 - a broken scorer must never fail the call
        _logger.warning("skillify: %s raised; the call is left unscored", source, exc_info=True)
        return None
    if not 0.0 <= value <= 1.0:
        _logger.warning("skillify: %s returned %r, outside [0, 1]; the call is left unscored", source, value)
        return None
    return value


async def open_episode(environment: Environment, arguments: Mapping[str, Any]) -> Episode:
    """Call ``environment(arguments)``, awaiting the result if it is awaitable. Exceptions propagate."""
    episode = environment(arguments)
    if inspect.isawaitable(episode):
        episode = await episode
    return episode


def with_episode_tools[**P, T](fn: AIFunction[P, T], episode: Episode | None) -> AIFunction[P, T]:
    """``fn`` with the episode's tools appended, for this run only; ``fn`` itself when there is no episode."""
    if episode is None:
        return fn
    return fn.replace(tools=(*fn.config.tools, *episode.tools))


def episode_outcome(episode: Episode, result: Any) -> float | None:  # noqa: ANN401
    """``episode.outcome(result)``; ``None`` when it raised or left ``[0, 1]``, as for ``score``."""
    return _checked("environment outcome", lambda: episode.outcome(result))


def close_episode(episode: Episode) -> None:
    """``episode.close()``; a raising close is logged, never raised."""
    try:
        episode.close()
    except Exception:  # noqa: BLE001 - teardown must not mask the run's own result or exception
        _logger.warning("skillify: episode close failed", exc_info=True)


class ReplayDriver:
    """Satisfies the learner's private ``_EvalDriver`` protocol by replaying recorded calls of ``fn``.

    Args:
        fn: The undecorated AI Function.
        store: Where the recorded arguments live.
        score: Outcome override; ``None`` uses post-conditions.
        pass_at: Outcome at or above which a run counts as passed.
        work_root: Parent of each run's skill copy and ``run_dir``.
        max_concurrency: Replays in flight.
        max_attempts: Tries per run on infrastructure errors.
        retry_backoff: Seconds before the first retry; doubles each time.
        environment: Builds a fresh ``Episode`` per run from the recorded
            arguments; its outcome replaces ``score``/post-conditions.
    """

    def __init__(
        self,
        fn: AIFunction[..., Any],
        store: LibraryStore,
        *,
        score: Score | None,
        pass_at: float,
        work_root: Path,
        max_concurrency: int = 4,
        max_attempts: int = 3,
        retry_backoff: float = 5.0,
        environment: Environment | None = None,
    ) -> None:
        self._fn = fn
        self._environment = environment
        self._store = store
        self._score = score
        self._pass_at = pass_at
        self._work_root = Path(work_root)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff

    async def run(self, specs: Sequence[_RunSpec]) -> list[_RunResult]:
        """Every (spec, arm, repeat), in spec, arm, repeat order."""
        jobs = [
            self._run_one(spec, arm, repeat) for spec in specs for arm in spec.arms for repeat in range(spec.repeats)
        ]
        return list(await asyncio.gather(*jobs))

    async def _run_one(self, spec: _RunSpec, arm: str, repeat: int) -> _RunResult:
        with _scope(
            run_id=f"replay-{uuid.uuid4().hex}",
            phase="replay",
            function=self._fn.name,
            library=str(self._store.root.resolve()),
            source_call_id=spec.task_id,
            arm=arm,
            repeat=repeat,
        ):
            return await self._run_observed(spec, arm, repeat)

    async def _run_observed(self, spec: _RunSpec, arm: str, repeat: int) -> _RunResult:
        label = f"{spec.task_id}-{arm}-{repeat}"
        # A baseline and every candidate share (task_id, arm, repeat) in a
        # shared-baseline trial; a unique directory keeps their files apart.
        run_dir = self._work_root / f"{label}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True)
        dirs = [Path(d) for d in spec.skills] if arm == "skill_on" else []
        args, kwargs = self._store.load_args(spec.task_id)

        last: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            backend = backend_from_dirs(dirs, run_dir / f"skills-{attempt}", actor_id=f"replay-{label}")
            skill_event("run.started", skill_dirs=dirs, attempt=attempt)
            try:
                async with self._semaphore:
                    outcome = await self._attempt(backend, args, kwargs)
            except SkillDeliveryError:
                skill_event("run.finished", status="failed", error="SkillDeliveryError")
                raise
            except asyncio.CancelledError:
                skill_event("run.finished", status="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - infrastructure; retried below
                last = exc
                skill_event("run.finished", status="failed", error=f"{type(exc).__name__}: {exc}")
                if attempt < self._max_attempts:
                    delay = self._retry_backoff * 2 ** (attempt - 1)
                    _logger.warning("replay %s failed (attempt %d): %s; retrying in %.0fs", label, attempt, exc, delay)
                    await asyncio.sleep(delay)
                continue
            skill_event("run.finished", status="completed", outcome=outcome)
            return _RunResult(outcome, outcome is not None and outcome >= self._pass_at)
        assert last is not None
        raise last

    async def _attempt(
        self, backend: SkillLibraryBackend | None, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> float | None:
        """One try of one run: its own episode (if any), the call, the outcome, then close.

        A factory that raises propagates, so the caller treats it as an
        infrastructure failure; ``AIFunctionError`` from the call is outcome 0.0.
        """
        episode = None
        if self._environment is not None:
            episode = await open_episode(self._environment, bind_arguments(self._fn, args, kwargs))
        try:
            variant, _ = with_skills(with_episode_tools(self._fn, episode), backend)
            try:
                result = await trace_observed(variant, *args, **kwargs)
            except AIFunctionError:
                return 0.0
            if episode is not None:
                return episode_outcome(episode, result.value)
            return outcome_of(self._fn, self._score, args, kwargs, result.value)
        finally:
            if episode is not None:
                close_episode(episode)


# ── SkillLearner: the rounds behind ``@skillify`` ──────────────────────────


class _EvalDriver(Protocol):
    """The slice of an eval driver the learner's replays need.

    Structural, not a concrete class, so tests inject a fake driver (no
    ``ClaudeAgent``, no subprocess) satisfying only ``run``.
    """

    async def run(self, specs: Sequence[_RunSpec]) -> list[_RunResult]:
        """Run every (spec, arm, repeat) triple and return one ``_RunResult`` each."""
        ...


def _should_induce(results: Sequence[_RunResult]) -> bool:
    """Mine only when these runs carry something to learn from.

    Three runs is the floor: below that there is no cross-trajectory
    evidence, and a skill mined from one or two runs is a transcript, not a
    generalization. Above it, one of two signals has to be present --

      * **a failure**, which is repair material, or
      * **spread** between the best and worst score, which means some run
        did something the others didn't.

    A batch where every run passed with an identical score is the
    at-ceiling case: nothing failed and nothing distinguishes one run from
    another, so there is no differential to mine and induction is declined.
    Unscored runs carry no evidence either way, so they
    are mined rather than silently skipped.
    """
    if len(results) < 3:
        return False
    scored = [r for r in results if r.outcome is not None]
    if not scored:
        return True
    if any(not r.passed for r in scored):
        return True
    scores = [r.outcome for r in scored if r.outcome is not None]
    return min(scores) < max(scores)


@dataclass(frozen=True)
class SkillifyConfig:
    """Every ``@skillify`` setting. See :func:`~.skillify.skillify` for each field.

    Raises:
        ValueError: ``replay`` is not ``True``; both ``score`` and
            ``environment`` are set; ``gate`` is set together with a
            non-default ``min_gain``/``trial_repeats``/``budget_calls``
            (they configure the default gate only); or a numeric field is
            out of range.
    """

    library: Path
    replay: bool | None = None
    every: int = 20
    score: Score | None = None
    pass_at: float = 1.0
    trial_inputs: int = 8
    trial_repeats: int = 2
    min_gain: float = 0.02
    max_skills: int = 20
    budget_calls: int = 200
    keep_versions: int = 10
    max_concurrency: int = 4
    environment: Environment | None = None
    miner: Miner | None = None
    gate: Gate | None = None

    def __post_init__(self) -> None:
        """Normalize ``library`` to a ``Path`` and validate every field."""
        object.__setattr__(self, "library", Path(self.library))
        if self.replay is not True:
            raise ValueError(
                "skillify requires replay=True: skills are gated by re-running recorded calls, which re-runs "
                "the function's tools. Pass replay=True only if they are safe to call again."
            )
        if self.score is not None and self.environment is not None:
            raise ValueError(
                "pass score= or environment=, not both: with an environment the outcome is episode.outcome(result)"
            )
        if self.gate is not None:
            defaults = {f.name: f.default for f in fields(self)}
            if any(getattr(self, name) != defaults[name] for name in ("min_gain", "trial_repeats", "budget_calls")):
                raise ValueError("pass either gate= or min_gain=/trial_repeats=/budget_calls=, not both")
        for name in ("every", "trial_inputs", "trial_repeats", "max_skills", "budget_calls", "max_concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not 0.0 < self.pass_at <= 1.0:
            raise ValueError(f"pass_at must be in (0, 1], got {self.pass_at}")


@dataclass
class RoundReport:
    """What one round saw and did. Persisted as one line of ``history.jsonl``."""

    round: int
    started_at: str
    finished_at: str | None = None
    status: Literal["declined", "admitted", "none_admitted", "failed"] = "declined"
    window_calls: int = 0
    window_success_rate: float | None = None
    window_end: int = -1
    mined: list[str] = field(default_factory=list)
    admitted: dict[str, dict[str, Any]] = field(default_factory=dict)
    revised: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropped_duplicates: list[str] = field(default_factory=list)
    version_before: int = 0
    version_after: int = 0
    extra_calls: int = 0
    budget_cuts: list[str] = field(default_factory=list)
    trials_overlap_mining: bool = False
    mined_from: int = 0
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The history-line form, tagged ``"type": "round"``."""
        return {"type": "round", **asdict(self)}

    @classmethod
    def from_json(cls, entry: Mapping[str, Any]) -> RoundReport:
        """Inverse of :meth:`to_json`."""
        return cls(**{k: v for k, v in entry.items() if k != "type"})


@dataclass(frozen=True)
class RecordedCall:
    """What a miner and a gate see of one recorded call.

    Attributes:
        call_id: The call's id in the library's records.
        outcome: The call's outcome in ``[0, 1]``; ``None`` when unscored
            (such a call is never mined or replayed).
        passed: Whether ``outcome`` reached ``pass_at``.
        events: The call's event log.
    """

    call_id: str
    outcome: float | None
    passed: bool
    events: list[Event]


_REPORTED = frozenset({"stage", "order", "extra_calls", "budget_cuts"})
"""Verdict-score keys a gate may set for the round report; kept out of the recorded scores."""


def _as_run_results(records: Sequence[CallRecord], pass_at: float) -> list[_RunResult]:
    """Present scored records in the shape ``should_induce`` predicates take."""
    return [_RunResult(r.outcome, (r.outcome or 0.0) >= pass_at) for r in records]


def _verdict_json(verdict: Verdict) -> dict[str, Any]:
    stage = verdict.scores.get("stage")
    return {
        "stage": stage if isinstance(stage, str) else "trial",
        "rationale": verdict.reason,
        "scores": {k: v for k, v in verdict.scores.items() if k not in _REPORTED},
    }


def _key(verdict: Verdict) -> str:
    """The report key for a verdict: the revised skill's slug, or the new skill's name."""
    return verdict.candidate.replaces or verdict.candidate.skill.name


def _in_decision_order(verdicts: Sequence[Verdict]) -> list[Verdict]:
    """``verdicts`` in the order the gate decided them, when it says so in ``scores["order"]``; else as given."""

    def key(item: tuple[int, Verdict]) -> float:
        order = item[1].scores.get("order")
        return float(item[0]) if order is None else float(order)

    return [v for _, v in sorted(enumerate(verdicts), key=key)]


class SkillLearner:
    """Records calls and runs rounds for one decorated function.

    Args:
        store: The library's on-disk state.
        config: The ``@skillify`` settings; ``config.miner`` and
            ``config.gate`` replace the default miner and gate.
        driver_factory: ``work_root -> _EvalDriver`` the round's replays run
            on (``@skillify`` passes a ``ReplayDriver`` factory).
        should_induce: Trigger over the window; defaults to
            :func:`_should_induce`.
        model: The model the default ``PromptMiner`` runs on (``@skillify``
            passes the decorated function's); unused with ``config.miner``.
    """

    def __init__(
        self,
        store: LibraryStore,
        config: SkillifyConfig,
        *,
        driver_factory: Callable[[Path], _EvalDriver],
        should_induce: Callable[[Sequence[_RunResult]], bool] | None = None,
        model: str | Model | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self._driver_factory = driver_factory
        self._miner: Miner = config.miner or PromptMiner(model)
        self._gate: Gate = config.gate or CompletionGate(
            min_gain=config.min_gain,
            repeats=config.trial_repeats,
            budget_calls=config.budget_calls,
            pass_at=config.pass_at,
        )
        self._replay: Replay = self._replay_calls
        self._runs_root = store.root / "runs"
        self._should_induce = should_induce or _should_induce
        self._pending = len(store.new_records())
        self._task: asyncio.Task[RoundReport] | None = None
        self._running = False

    @property
    def pending(self) -> int:
        """Calls recorded since the last round."""
        return self._pending

    def record(self, **kwargs: Any) -> CallRecord:  # noqa: ANN401
        """Persist one call (``LibraryStore.append_record`` arguments)."""
        record = self.store.append_record(**kwargs)
        self._pending += 1
        return record

    def maybe_start_round(self) -> asyncio.Task[RoundReport] | None:
        """Start a background round if ``every`` calls are pending and none is running.

        Creates the task under ``no_thread_scope()``: a task copies the calling
        context, and the triggering call's own thread is torn down once that
        call returns, so a round inheriting its scope would find its replays
        attributed to an already-gone thread.
        """
        if self._task is not None and not self._task.done():
            return None
        if self._pending < self.config.every:
            return None
        with no_thread_scope():
            self._task = asyncio.get_running_loop().create_task(self.run_round())
        self._task.add_done_callback(self._log_task_failure)
        return self._task

    def _log_task_failure(self, task: asyncio.Task[RoundReport]) -> None:
        """Log a background round's exception so it is never silently lost."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _logger.error("skillify: background round failed", exc_info=exc)

    async def refresh(self) -> RoundReport:
        """Wait for any running round, then run one now regardless of ``every``.

        Waits both for the background round ``maybe_start_round`` started and
        for one started by awaiting ``run_round()`` directly. A round held by
        another process is not waited for: this one is then refused as usual.

        Runs under ``no_thread_scope()`` for the same reason as
        ``maybe_start_round``: a caller may invoke this from inside its own
        ambient thread scope (e.g. as a tool), and a round must not attribute
        its replays to that unrelated, still-running thread. Always runs its
        own round after awaiting any round already in flight, rather than
        returning that one's report: whether the in-flight round already
        covered everything pending depends on timing (it may have just
        finished on an older window, or new calls may have landed while it
        ran), so only a fresh round run right now can promise its report
        reflects the current window.
        """
        if self._task is not None and not self._task.done():
            # ``wait`` rather than ``await``: the background round being
            # cancelled must not cancel this call, while this call being
            # cancelled still propagates.
            await asyncio.wait({self._task})
        # A round started by awaiting ``run_round()`` directly is not
        # ``self._task``; without waiting for it too, this round would be
        # refused as "already running".
        while self._running:
            await asyncio.sleep(0.05)
        with no_thread_scope():
            return await self.run_round()

    async def run_round(self) -> RoundReport:
        """One full round. Raises only on cancellation; every other failure is reported and persisted."""
        with _scope(
            library=str(self.store.root.resolve()),
            round_id=uuid.uuid4().hex,
            run_id=None,
            version=None,
            phase="learning",
        ):
            return await self._run_round()

    async def _run_round(self) -> RoundReport:
        if self._running:
            return RoundReport(
                round=self.store.next_round(),
                started_at=_now(),
                finished_at=_now(),
                status="failed",
                error="a round is already running in this process",
                version_before=self.store.current_version(),
                version_after=self.store.current_version(),
            )
        self._running = True
        try:
            with self.store.round_lock() as held:
                if not held:
                    return RoundReport(
                        round=self.store.next_round(),
                        started_at=_now(),
                        finished_at=_now(),
                        status="failed",
                        error="another process holds this library's round lock",
                        version_before=self.store.current_version(),
                        version_after=self.store.current_version(),
                    )
                window = self.store.new_records()
                current = self.store.current_version()
                report = RoundReport(
                    round=self.store.next_round(),
                    started_at=_now(),
                    window_calls=len(window),
                    window_end=max((r.seq for r in window), default=self.store.last_window_end()),
                    version_before=current,
                    version_after=current,
                )
                round_dir = self.store.round_dir(report.round)
                skill_event("round.started", report=report, window=window)
                try:
                    await self._round(report, window, round_dir)
                except asyncio.CancelledError:
                    skill_event("round.cancelled", round=report.round)
                    # A cancelled round must not be mistaken for a completed one: no history
                    # entry is appended, so the window stays pending for the next round.
                    _logger.warning(
                        "skillify: round %d was cancelled; its window stays pending for the next round",
                        report.round,
                    )
                    shutil.rmtree(round_dir, ignore_errors=True)
                    self._pending = len(self.store.new_records())
                    raise
                except Exception as exc:  # noqa: BLE001 - a round never breaks the caller
                    _logger.exception("skillify: round %d failed; library left at version %d", report.round, current)
                    report.status = "failed"
                    report.error = f"{type(exc).__name__}: {exc}"
                    report.version_after = self.store.current_version()
                report.finished_at = _now()
                self.store.append_history(report.to_json())
                self._pending = len(self.store.new_records())
                shutil.rmtree(round_dir, ignore_errors=True)
                return report
        finally:
            self._running = False

    async def _round(self, report: RoundReport, window: list[CallRecord], round_dir: Path) -> None:
        cfg = self.config
        scored = [r for r in window if r.outcome is not None]
        if scored:
            report.window_success_rate = sum(r.outcome >= cfg.pass_at for r in scored) / len(scored)  # type: ignore[operator]
        induce = self._should_induce(_as_run_results(scored, cfg.pass_at))
        skill_event(
            "induction.decided",
            round=report.round,
            induce=induce,
            scored_calls=len(scored),
            success_rate=report.window_success_rate,
        )
        candidates: list[Skill] = []
        if induce:
            # Only scored records are mined: an unscored call (an infrastructure error,
            # a failing score/environment.outcome) still counts toward the window and
            # consumes it, but is never handed to the miner, so it can never be mined
            # as -- or look like -- a success.
            calls = [
                self._recorded(r, events) for r in scored if (events := self.store.load_events(r.call_id)) is not None
            ]
            report.mined_from = len(calls)
            skill_event("mining.started", call_ids=[c.call_id for c in calls])
            candidates = await self._miner(calls, round_dir / "candidates")
            report.mined = [c.name for c in candidates]
            skill_event("mining.finished", skills=candidates)

        current_dirs = self.store.skill_dirs()
        installed = {d.name for d in current_dirs}
        candidates = self._drop_duplicates(report, candidates, current_dirs)
        revisions: list[Candidate] = []
        try:
            revisions += self._take_revisions(report, installed, round_dir / "pending")
            if not induce and not revisions:
                report.status = "declined"
                return
            await self._gate_and_publish(report, window, round_dir, current_dirs, revisions, candidates)
        except BaseException:  # a failed or cancelled round never gated them
            self._restage(revisions)
            raise

    def _drop_duplicates(self, report: RoundReport, candidates: list[Skill], current_dirs: list[Path]) -> list[Skill]:
        """``candidates`` that can join the library.

        A candidate named like an installed skill, or whose directory maps to the same
        parameter name as an installed skill or an earlier candidate (``foo_bar`` and
        ``foo-bar``), is a duplicate; one whose directory maps to the index's reserved
        name is rejected as ``"invalid"``.
        """
        installed = {d.name for d in current_dirs}
        taken = {_param_name(d.name) for d in current_dirs}
        kept: list[Skill] = []
        for candidate in candidates:
            param = _param_name(candidate.path.name)
            if param == INDEX:
                rationale = f"{candidate.path.name!r} is the name reserved for the skill index"
                report.rejected[candidate.name] = {"stage": "invalid", "rationale": rationale, "scores": {}}
                skill_event("candidate.excluded", skill=candidate, stage="invalid", reason=rationale)
            elif candidate.name in installed or param in taken:
                report.dropped_duplicates.append(candidate.name)
                skill_event("candidate.excluded", skill=candidate, stage="duplicate", reason="Already represented")
            else:
                taken.add(param)
                kept.append(candidate)
        return kept

    def _restage(self, revisions: Sequence[Candidate]) -> None:
        """Put each revision back in ``pending/``, unless a newer one was staged meanwhile."""
        for candidate in revisions:
            slug = candidate.replaces or ""
            if not (self.store.pending_dir / slug).exists():
                self.store.stage_revision(slug, candidate.skill.path)

    def _take_revisions(self, report: RoundReport, installed: set[str], dest: Path) -> list[Candidate]:
        """Staged revisions of installed skills as candidates.

        Each one discarded instead is reported: ``"stale"`` when its skill is not
        installed, ``"invalid"`` when its ``SKILL.md`` does not load.
        """
        revisions: list[Candidate] = []
        for slug, path in self.store.take_pending(dest):
            if slug not in installed:
                rationale = f"revision of {slug!r}, which version {report.version_before} does not install; discarded"
                report.rejected[slug] = {"stage": "stale", "rationale": rationale, "scores": {}}
                continue
            try:
                skill = load_skill(path)
            except Exception as exc:  # noqa: BLE001 - one malformed revision must not cost the others
                _logger.warning("skillify: round %d: revision of %r does not load: %s", report.round, slug, exc)
                report.rejected[slug] = {"stage": "invalid", "rationale": f"{type(exc).__name__}: {exc}", "scores": {}}
                continue
            revisions.append(Candidate(skill, replaces=slug))
            skill_event("candidate.revised", skill=skill, replaces=slug)
        return revisions

    async def _gate_and_publish(
        self,
        report: RoundReport,
        window: list[CallRecord],
        round_dir: Path,
        current_dirs: list[Path],
        revisions: list[Candidate],
        candidates: Sequence[Skill],
    ) -> None:
        cfg = self.config
        gated = [*revisions, *(Candidate(c) for c in candidates)]
        if not gated:
            report.status = "none_admitted"
            return

        trial_records, report.trials_overlap_mining = self._trial_records(window)
        trials = [self._recorded(r, self.store.load_events(r.call_id) or []) for r in trial_records]
        self._runs_root = round_dir / "runs"
        skill_event(
            "gate.started",
            candidates=gated,
            trial_ids=[c.call_id for c in trials],
            trials_overlap_mining=report.trials_overlap_mining,
        )
        verdicts = await self._gate(gated, current_dirs, trials, self._replay)
        if len(verdicts) != len(gated):
            raise RuntimeError(f"gate returned {len(verdicts)} verdicts for {len(gated)} candidates")
        if sorted(map(id, (v.candidate for v in verdicts))) != sorted(map(id, gated)):
            raise RuntimeError("gate returned verdicts that are not one per candidate it was given")
        skill_event("gate.finished", verdicts=verdicts)
        for verdict in verdicts:
            cuts = verdict.scores.get("budget_cuts")
            if isinstance(cuts, list):
                report.budget_cuts.extend(str(c) for c in cuts)
            report.extra_calls += int(verdict.scores.get("extra_calls") or 0)
        verdicts = _in_decision_order(verdicts)

        for verdict in verdicts:
            if not verdict.admitted:
                report.rejected[_key(verdict)] = _verdict_json(verdict)
        revised = [v for v in verdicts if v.admitted and v.candidate.replaces is not None]
        admitted = sorted(
            (v for v in verdicts if v.admitted and v.candidate.replaces is None),
            key=lambda v: v.gain or 0.0,
            reverse=True,
        )
        room = max(cfg.max_skills - len(current_dirs), 0)
        keep, cut = admitted[:room], admitted[room:]
        for verdict in cut:
            report.rejected[_key(verdict)] = {**_verdict_json(verdict), "stage": "max_skills"}
        if not keep and not revised:
            report.status = "none_admitted"
            return
        # ``publish_if_current`` checks ``CURRENT`` and writes the new version under
        # one lock, so a rollback (in this process or another) landing between the
        # round's read of ``version_before`` and this publish can never be
        # overwritten: either it lands first and this call sees the mismatch below,
        # or this call publishes first and the rollback, taking the same lock, wins
        # afterwards. ``current_dirs`` is the library as the round found it;
        # publishing it plus the admitted skills would otherwise silently undo
        # whatever changed CURRENT since -- the verdicts were measured against a
        # library that is no longer current, so none is admitted.
        swapped = {v.candidate.replaces for v in revised}
        published = self.store.publish_if_current(
            report.version_before,
            [
                *(d for d in current_dirs if d.name not in swapped),
                *(v.candidate.skill.path for v in revised),
                *(v.candidate.skill.path for v in keep),
            ],
        )
        if published is None:
            now = self.store.current_version()
            for verdict in [*revised, *keep]:
                report.rejected[_key(verdict)] = {**_verdict_json(verdict), "stage": "stale"}
            report.status = "failed"
            report.error = (
                f"the library changed during the round (version {report.version_before} -> {now}, "
                "e.g. a rollback); nothing was published"
            )
            report.version_after = now
            _logger.warning("skillify: round %d: %s", report.round, report.error)
            self._restage(revisions)  # measured against a library that is no longer current
            return
        for verdict in revised:
            report.revised[_key(verdict)] = _verdict_json(verdict)
        for verdict in keep:
            report.admitted[_key(verdict)] = _verdict_json(verdict)
        report.version_after = published
        report.status = "admitted"

    def _trial_records(self, window: list[CallRecord]) -> tuple[list[CallRecord], bool]:
        in_window = {r.call_id for r in window}
        pool = [
            r for r in self.store.records() if r.replayable and r.outcome is not None and r.call_id not in in_window
        ]
        overlap = False
        if not pool:
            pool = [r for r in window if r.replayable and r.outcome is not None]
            overlap = True
        pool.sort(key=lambda r: ((r.outcome or 0.0) >= self.config.pass_at, -r.seq))
        return pool[: self.config.trial_inputs], overlap

    def _recorded(self, record: CallRecord, events: list[Event]) -> RecordedCall:
        """``record`` with its event log as a :class:`RecordedCall`."""
        return RecordedCall(
            call_id=record.call_id,
            outcome=record.outcome,
            passed=record.outcome is not None and record.outcome >= self.config.pass_at,
            events=events,
        )

    async def _replay_calls(
        self, skill_dirs: Sequence[Path], calls: Sequence[RecordedCall], repeats: int
    ) -> list[float | None]:
        """The round's :data:`~.evaluation.Replay`: re-run ``calls`` with ``skill_dirs`` installed.

        One run per (call, repeat), in that order, on a driver rooted in the
        round's directory; an empty ``skill_dirs`` runs with no skills.
        """
        driver = self._driver_factory(self._runs_root)
        arms = ("skill_on",) if skill_dirs else ("skill_off",)
        specs = [_RunSpec(task_id=c.call_id, skills=list(skill_dirs), repeats=repeats, arms=arms) for c in calls]
        return [r.outcome for r in await driver.run(specs)]
