"""``ReplayDriver`` and ``SkillLearner`` -- replaying recorded calls, and the rounds behind ``@skillify``.

Two parts:

  * **``ReplayDriver``** re-runs recorded calls of an AI Function as gate
    trials. ``@skillify`` decides whether a candidate skill helps by
    replaying recent calls with and without it. This driver satisfies the
    learner's private ``_EvalDriver`` protocol (below): a ``RunSpec``'s
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

    Invariants:
        R1 — a replay calls the undecorated function; it is never recorded.
        R2 — only the ``"skill_on"`` arm sees skills.
        R3 — ``AIFunctionError`` is outcome 0.0; other exceptions are retried then raised.
        R4 — with an environment, a replay builds a fresh episode from the recorded arguments;
        no episode is shared between runs, and its ``close()`` always runs.

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
       ``pending/`` (an optimizer writing through ``SkillLibraryBackend``) are
       taken here and join the candidates first, each as
       ``Candidate(skill, replaces=slug)``; one whose skill is no longer
       installed is discarded as ``"stale"``, one that does not load as
       ``"invalid"``. A staged revision never triggers a round by itself, but
       every round that runs gates it.
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
    6. **Publish.** The current skills, with admitted revisions swapped in
       (``RoundReport.revised``), plus the admitted new ones, capped at
       ``max_skills`` by gain, become a new version. Revisions do not count
       against ``max_skills``.

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

    Invariants:
        L1 — one round per library at a time — a synchronous flag in-process, file lock across processes.
        L2 — a failed round leaves CURRENT unchanged.
        L3 — every round that runs, whatever its status, consumes its window and appends one
        RoundReport; a cancelled round, and one refused because another round is running (in
        this process, or holding the lock in another), append nothing and consume nothing.
        L4 — trial inputs come from outside the mining window unless none exist (reported).
        L5 — a candidate is admitted only if the gate admits it (the default: it improves on the current library).
        L6 — a gate that returns a verdict count other than the candidate count fails the round.
        L7 — every staged revision a round takes is reported (revised, or rejected: trial, stale, invalid); one
        that never reached a verdict because the round failed, or whose round found the library changed
        before publishing, is staged again (unless a newer one was staged meanwhile).
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from strands.models import Model

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.types.events import Event

from .discovery import Miner
from .evaluation import Gate
from .skill_library import CallRecord, LibraryStore

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

# ── ReplayDriver ─────────────────────────────────────────────────────────────

Score = Callable[[Mapping[str, Any], Any], float]

class Episode(Protocol):
    """One run's stateful environment, built from that call's arguments."""

    tools: Sequence[Any]
    def outcome(self, result: Any) -> float:
        """The run's outcome in ``[0, 1]``, read from the environment's state."""
        ...
    def close(self) -> None:
        """Release the environment. Called exactly once per episode, however the run ended."""
        ...

Environment = Callable[[Mapping[str, Any]], Episode | Awaitable[Episode]]

def bind_arguments(fn: AIFunction[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]: ...
def outcome_of(
    fn: AIFunction[..., Any],
    score: Score | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    result: Any,
) -> float | None: ...
async def open_episode(environment: Environment, arguments: Mapping[str, Any]) -> Episode: ...
def with_episode_tools[**P, T](fn: AIFunction[P, T], episode: Episode | None) -> AIFunction[P, T]: ...
def episode_outcome(episode: Episode, result: Any) -> float | None: ...
def close_episode(episode: Episode) -> None: ...

class ReplayDriver:
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
    ) -> None: ...
    async def run(self, specs: Sequence[Any]) -> list[Any]: ...

# ── SkillLearner ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillifyConfig:
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
        ...

@dataclass
class RoundReport:
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
    def to_json(self) -> dict[str, Any]: ...
    @classmethod
    def from_json(cls, entry: Mapping[str, Any]) -> RoundReport: ...

@dataclass(frozen=True)
class RecordedCall:
    call_id: str
    outcome: float | None
    passed: bool
    events: list[Event]

class SkillLearner:
    store: LibraryStore
    config: SkillifyConfig
    def __init__(
        self,
        store: LibraryStore,
        config: SkillifyConfig,
        *,
        driver_factory: Callable[[Path], Any],
        should_induce: Callable[[Sequence[Any]], bool] | None = None,
        model: str | Model | None = None,
    ) -> None: ...
    @property
    def pending(self) -> int: ...
    def record(self, **kwargs: Any) -> CallRecord: ...
    def maybe_start_round(self) -> asyncio.Task[RoundReport] | None: ...
    async def refresh(self) -> RoundReport: ...
    async def run_round(self) -> RoundReport: ...
