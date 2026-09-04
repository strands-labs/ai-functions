"""Belief providers: where per-(task, candidate) estimates come from and how they learn.

A :class:`Beliefs` implementation owns three verbs:

- ``estimate`` — produce an :class:`~.search.Estimate` per candidate for one
  task, read by the economic function's :class:`~.search.Search` loop.
- ``update`` — fold one freshly booked :class:`~.types.AttemptRecord` in,
  online, at run time (the provisional booking).
- ``settle`` — revise previously booked records when downstream feedback
  assigns their true score (the settlement; see invariant E2).

Two learning channels feed a provider, both driven by the optimizer's backward
pass over the reconstructed graph. Outcome records are numeric data and flow
through ``update``/``settle``: the economic run emits its routing decision as a
grad-enabled parameter hosted by the economic function itself (its
``ParameterHost`` methods), and the score the backward pass assigns that
parameter settles the run's records. Textual judgment flows through ordinary
memory-parameter optimization: a provider that learns from text
(``LLMForecaster``) recalls its notes grad-enabled inside the economic run, so
the backward pass refines feedback against that run's trace and consolidates it
into the notes' backend like any other parameter.

The dollar worth of success is declared once, on the economic function, and
passed into ``estimate`` per call — a ``Beliefs`` instance holds no value
configuration of its own, which is what lets one instance back several
economic functions with different values.

Invariants:
    E2 — records are revisable: implementations store per-record
    contributions (or sufficient statistics keyed by record id) such that
    ``settle`` replaces a record's effect rather than double-counting it.

    E4 — ``estimate`` must return an estimate for every candidate it is
    given; a missing label is an error, never silently defaulted.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ai_functions.ai_thread.ai_function import AIFunction, ai_function
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.memory.frozen import Frozen
from ai_functions.types import TokenUsage

from .search import Bernoulli, Estimate
from .types import AttemptRecord, RecordId, TaskView

if TYPE_CHECKING:
    from ai_functions.memory.base import MemoryBackend

    from .types import Candidate

logger = logging.getLogger(__name__)

# Output tokens assumed for a candidate's cost before any attempt is observed.
# Only used to seed the cost estimate; replaced by the observed mean once the
# candidate has run at least once.
_COST_PRIOR_OUTPUT_TOKENS = 500


def _check_score(score: float, candidate: str) -> None:
    """Raise if ``score`` is outside ``[0, 1]`` — the range the Beta posterior needs."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"score must be in [0, 1], got {score} for candidate {candidate!r}; "
            "return a normalized score (e.g. F1), not a dollar amount"
        )


class Beliefs(ABC):
    """Estimator plus learner for a set of candidates.

    Stateful and shareable: one instance may back several economic
    functions, and persistence across processes is delegated to a memory
    backend where the implementation supports one.
    """

    @abstractmethod
    async def estimate(
        self,
        task: TaskView,
        candidates: list[Candidate],
        value: float | None,
        history: list[AttemptRecord],
    ) -> dict[str, Estimate]:
        """Estimate each candidate's economics for one task.

        Args:
            task: The task being attempted (prompt and structured arguments).
            candidates: Candidates to estimate, including labels, prices,
                and descriptions.
            value: Dollars a fully passing result is worth — the scale for
                reward distributions under ``@routed``. ``None`` under
                ``merge``, where the value is a callable and no constant
                scale exists.
            history: Records already booked for *this* task's search, newest
                last; non-empty only on re-estimation rounds.

        Returns:
            An estimate per ``Candidate.label``, covering every candidate (E4).

        Raises:
            AIFunctionError: The estimate could not be produced, or requires
                the constant value scale and ``value`` is ``None``.
        """
        ...

    def update(self, record: AttemptRecord) -> None:
        """Fold one provisional record in, online.

        Called by the economic function immediately after each attempt with
        ``local_score`` set. Default implementation is a no-op for
        estimators that do not learn.
        """
        del record

    def settle(self, record_id: RecordId, score: float) -> None:
        """Replace a booked record's score with its downstream-corrected value.

        Called by settlement when backward feedback reaches the economic
        function's decision node. Default implementation is a no-op.
        """
        del record_id, score

    def stats(self) -> dict[str, str]:
        """Return a human-readable summary line per candidate label.

        The introspection hook examples and dashboards read; also what
        :class:`LLMForecaster` interpolates into its prompt. Default
        implementation returns an empty dict for estimators with no learned
        state.
        """
        return {}

    @classmethod
    def fixed(cls, estimates: dict[str, Estimate]) -> Beliefs:
        """Constant estimates, independent of task and history.

        The zero-cost provider for known workloads, tests, and examples.

        Args:
            estimates: The estimate returned for each label, every time.

        Returns:
            A ``Beliefs`` whose ``estimate`` always returns ``estimates``
            and whose learning verbs are no-ops.
        """
        return _FixedBeliefs(dict(estimates))


class _FixedBeliefs(Beliefs):
    """Constant estimates; learning verbs are no-ops. See :meth:`Beliefs.fixed`."""

    def __init__(self, estimates: dict[str, Estimate]) -> None:
        self._estimates = estimates

    async def estimate(
        self, task: TaskView, candidates: list[Candidate], value: float | None, history: list[AttemptRecord]
    ) -> dict[str, Estimate]:
        """Return the fixed estimate for each candidate; every label must be present (E4)."""
        del task, value, history
        missing = [c.label for c in candidates if c.label not in self._estimates]
        if missing:
            raise KeyError(f"fixed beliefs have no estimate for {missing}")
        return {c.label: self._estimates[c.label] for c in candidates}


# ── Routing memory schema ─────────────────────────────────────────


class ObservedAttempt(BaseModel):
    """One attempt's outcome, as persisted routing statistics.

    The durable form of a booked :class:`~.types.AttemptRecord`'s learnable
    content: what ran, what it cost, how much work it took, and the score it
    (currently) settles at. Machine-managed by :class:`EmpiricalBeliefs`.
    """

    record_id: str
    candidate: str
    score: float
    cost: float
    turns: int = 0
    output_tokens: int = 0


_CASEBOOK_DESCRIPTION = (
    "A list of example task classes for a router deciding which candidate model to "
    "pick. Empty means a fresh casebook: begin one. Each entry describes one class "
    "of tasks and records, per candidate tried "
    "on it: #turns, #output_tokens per turn, the downstream score in [0, 1], and one "
    "line of qualitative feedback (why it scored that way). "
    "When new feedback describes a task class similar to an existing entry, UPDATE "
    "that entry to cover both cases and merge the stats (e.g. average the numbers, "
    "note the range) instead of adding a near-duplicate. "
    "*DO NOT* write instructions on how to perform tasks — the model executing the "
    "task will never see this text."
)


class RoutingMemory(BaseModel):
    """Learned state of one routed function, as a field in a user memory schema.

    Add one field of this type per ``@routed`` function and point the
    function's :class:`LLMForecaster` at it via ``memory_key``::

        class MyMemory(BaseModel):
            researcher_routing: RoutingMemory = Field(default_factory=RoutingMemory)


        beliefs = LLMForecaster(memory=backend, memory_key="researcher_routing")

    ``notes`` is the textual channel — a casebook the optimizer's backward
    pass consolidates downstream feedback into, recalled grad-enabled on every
    estimation. ``stats`` is the numeric channel — the observed attempt
    records :class:`EmpiricalBeliefs` persists and reloads across processes;
    frozen because it is machine-managed, never a gradient target.
    """

    notes: str = Field(default="", description=_CASEBOOK_DESCRIPTION)
    stats: Frozen[list[ObservedAttempt]] = Field(
        default_factory=list,
        description="Observed attempt statistics, machine-managed by the router. Do not edit.",
    )


# ── EmpiricalBeliefs ──────────────────────────────────────────────


@dataclass
class _Contribution:
    """One record's contribution to a candidate's posterior, kept for settlement."""

    label: str
    score: float
    cost: float
    turns: int = 0
    output_tokens: int = 0


class EmpiricalBeliefs(Beliefs):
    """Population-level statistics per candidate; no task-dependence, no LLM.

    Maintains a Beta posterior over each candidate's success score and the
    mean observed cost, folded from the record stream. Contributions are
    stored per record id, so settlement replaces a record's effect exactly
    rather than double-counting it (E2). ``estimate`` ignores the task and
    returns each candidate's posterior mean as a :class:`~.search.Bernoulli`
    at the call's ``value``, priced at the candidate's mean observed cost
    (or a prior derived from its token prices before any attempt).

    Args:
        memory: Backend persisting the statistics across processes; ``None``
            keeps them in-memory only. Requires ``stats_key``.
        stats_key: Parameter path of a ``list[ObservedAttempt]`` field in
            ``memory`` (e.g. ``"researcher_routing/stats"`` for a
            :class:`RoutingMemory` field). Loaded at construction; rewritten
            after every ``update``/``settle``.
        prior_successes: Beta prior pseudo-successes per candidate.
        prior_failures: Beta prior pseudo-failures per candidate.
        decay: Per-record exponential down-weighting of old evidence, in
            ``(0, 1]``; 1.0 = never forget. With ``decay < 1`` the posterior
            is recomputed from stored contributions weighted by recency.

    Raises:
        ValueError: ``decay`` outside ``(0, 1]``, a negative prior, or
            ``memory`` given without ``stats_key`` (or vice versa).
    """

    def __init__(
        self,
        memory: MemoryBackend | None = None,
        stats_key: str | None = None,
        prior_successes: float = 1.0,
        prior_failures: float = 1.0,
        decay: float = 1.0,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if prior_successes < 0 or prior_failures < 0:
            raise ValueError("priors must be non-negative")
        if (memory is None) != (stats_key is None):
            raise ValueError("memory and stats_key must be given together")
        self._memory = memory
        self._stats_key = stats_key
        self._prior_a = prior_successes
        self._prior_b = prior_failures
        self._decay = decay
        # Insertion-ordered contributions, newest last — recency drives decay.
        self._contribs: dict[RecordId, _Contribution] = {}
        self._load_stats()

    def _load_stats(self) -> None:
        """Rebuild the contributions from the persisted statistics, if configured."""
        if self._memory is None or self._stats_key is None:
            return
        try:
            raw = self._memory.fetch(self._stats_key)
        except Exception:  # noqa: BLE001 — a missing/malformed field disables persistence, not routing
            logger.warning("EmpiricalBeliefs: could not load stats from %r", self._stats_key, exc_info=True)
            return
        for item in raw or []:
            rec = ObservedAttempt.model_validate(item) if isinstance(item, dict) else item
            self._contribs[RecordId(rec.record_id)] = _Contribution(
                label=rec.candidate,
                score=rec.score,
                cost=rec.cost,
                turns=rec.turns,
                output_tokens=rec.output_tokens,
            )

    def _persist_stats(self) -> None:
        """Write the current contributions back to memory, if configured."""
        if self._memory is None or self._stats_key is None:
            return
        rows = [
            ObservedAttempt(
                record_id=str(rid),
                candidate=c.label,
                score=c.score,
                cost=c.cost,
                turns=c.turns,
                output_tokens=c.output_tokens,
            )
            for rid, c in self._contribs.items()
        ]
        try:
            self._memory.save(self._stats_key, rows)
        except Exception:  # noqa: BLE001 — persistence is best-effort; routing continues in-memory
            logger.warning("EmpiricalBeliefs: could not persist stats to %r", self._stats_key, exc_info=True)

    def _posterior(self, label: str) -> tuple[float, float, float | None]:
        """Return ``(alpha, beta, mean_cost)`` for a label from stored contributions.

        Recomputed from scratch so settlement (which mutates a stored
        contribution in place) is always reflected exactly. Decay weights the
        i-th most recent contribution by ``decay**i``.
        """
        alpha, beta = self._prior_a, self._prior_b
        cost_num = cost_den = 0.0
        label_contribs = [c for c in self._contribs.values() if c.label == label]
        n = len(label_contribs)
        for i, c in enumerate(label_contribs):
            w = self._decay ** (n - 1 - i)  # newest (last) gets weight 1
            alpha += w * c.score
            beta += w * (1.0 - c.score)
            cost_num += w * c.cost
            cost_den += w
        mean_cost = cost_num / cost_den if cost_den > 0 else None
        return alpha, beta, mean_cost

    async def estimate(
        self, task: TaskView, candidates: list[Candidate], value: float | None, history: list[AttemptRecord]
    ) -> dict[str, Estimate]:
        """Return each candidate's posterior mean as a Bernoulli at ``value``.

        The Bernoulli value carries the reward scale (``value``); the posterior
        mean ``p`` is the expected score in ``[0, 1]``, so the expected reward
        is ``value * p``.
        """
        del task, history
        if value is None:
            raise AIFunctionError("EmpiricalBeliefs requires a constant value scale (dollars a pass is worth)")
        out: dict[str, Estimate] = {}
        for c in candidates:
            alpha, beta, mean_cost = self._posterior(c.label)
            p = alpha / (alpha + beta)
            cost = (
                mean_cost
                if mean_cost is not None
                else c.prices.cost_of(TokenUsage(output_tokens=_COST_PRIOR_OUTPUT_TOKENS))
            )
            out[c.label] = Estimate(dist=Bernoulli(p=p, value=value), cost=cost)
        return out

    def _turn_stats(self, label: str) -> tuple[float, float] | None:
        """Decay-weighted ``(mean_turns, mean_output_tokens_per_turn)`` for a label.

        Computed over contributions with a measured turn count (``turns > 0``;
        records booked without a run, e.g. via ``Decision.report``, carry no
        turn data). ``None`` when no such contribution exists.
        """
        measured = [c for c in self._contribs.values() if c.label == label and c.turns > 0]
        if not measured:
            return None
        n = len(measured)
        w_sum = t_num = o_num = 0.0
        for i, c in enumerate(measured):
            w = self._decay ** (n - 1 - i)  # newest (last) gets weight 1
            w_sum += w
            t_num += w * c.turns
            o_num += w * (c.output_tokens / c.turns)
        return t_num / w_sum, o_num / w_sum

    def update(self, record: AttemptRecord) -> None:
        """Fold a record's score, cost, and turn shape into its candidate's posterior.

        The score must be in ``[0, 1]`` — the Beta posterior folds ``score`` into
        ``alpha`` and ``1 - score`` into ``beta``, which is only coherent on that
        range. Validated here so an out-of-range score fails clearly rather than
        later at ``Bernoulli`` construction.
        """
        score = record.settled_score if record.settled_score is not None else record.local_score
        _check_score(score, record.candidate)
        self._contribs[record.id] = _Contribution(
            label=record.candidate,
            score=score,
            cost=record.cost,
            turns=record.turns,
            output_tokens=record.usage.output_tokens,
        )
        self._persist_stats()

    def settle(self, record_id: RecordId, score: float) -> None:
        """Overwrite a stored contribution's score with its settled value (E2)."""
        contrib = self._contribs.get(record_id)
        if contrib is None:
            logger.debug("settle: no contribution for record %s; ignoring", record_id)
            return
        _check_score(score, contrib.label)
        contrib.score = score
        self._persist_stats()

    def stats(self) -> dict[str, str]:
        """Return a ``"NN% pass over K attempts, avg cost $X, avg N turns ..."`` line per candidate.

        The rate is the observed (decay-weighted) pass fraction — evidence
        only. The routing estimate additionally smooths it with the Beta
        prior (see ``estimate``).
        """
        out: dict[str, str] = {}
        labels = {c.label for c in self._contribs.values()}
        for label in sorted(labels):
            alpha, beta, mean_cost = self._posterior(label)
            n = sum(1 for c in self._contribs.values() if c.label == label)
            # Subtracting the priors from the posterior leaves the weighted
            # evidence: successes / attempts, decay- and settlement-aware.
            p = (alpha - self._prior_a) / (alpha + beta - self._prior_a - self._prior_b)
            cost_str = f", avg cost ${mean_cost:.4f}" if mean_cost is not None else ""
            turn_str = ""
            turn_stats = self._turn_stats(label)
            if turn_stats is not None:
                mean_turns, mean_out_per_turn = turn_stats
                turn_str = f", avg {mean_turns:.1f} turns at {mean_out_per_turn:.0f} output tokens/turn"
            out[label] = f"{p:.0%} pass over {n} attempts{cost_str}{turn_str}"
        return out


# ── LLMForecaster ─────────────────────────────────────────────────


class LLMForecaster(Beliefs):
    """Task-dependent beliefs: an LLM adjusts a numeric base rate per task.

    Layers input-dependent judgment on top of a contained numeric provider.
    The forecast call sees, per candidate: the base provider's statistics
    (:meth:`Beliefs.stats`), the candidate's description, and the accumulated
    routing casebook recalled from memory. Its output is a pass probability
    and turn-shape prediction per candidate, anchored on the base rate.

    Learned state lives in one :class:`RoutingMemory` field of the user's
    memory schema, named by ``memory_key``: the casebook under
    ``<memory_key>/notes`` (textual channel, optimized by the backward pass),
    the observed attempt statistics under ``<memory_key>/stats`` (numeric
    channel, persisted by the default :class:`EmpiricalBeliefs` base).

    The forecast call's own cost is not currently priced: its token usage
    lands in the run's event log, but no ``Prices`` are declared for the
    forecasting model, so booked records and :func:`~.function.spend` cover
    attempts only.

    Args:
        base: Numeric provider supplying population statistics; owns the
            record stream. Defaults to an :class:`EmpiricalBeliefs` that
            persists its statistics under ``<memory_key>/stats`` when
            ``memory`` is given (in-memory otherwise). Pass an explicit base
            to control its priors/decay or to share one across functions.
        forecast_fn: The forecasting brain — any ``AIFunction`` producing a
            per-candidate ``ForecastResult``; defaults to the module's
            :data:`forecast`. Swap or ``.replace(model=...)`` freely.
        memory: Backend persisting the learned state; when ``None`` the
            forecaster runs stateless (nothing recalled or written).
        memory_key: Name of the :class:`RoutingMemory` field in ``memory``'s
            schema; required when ``memory`` is given.

    Raises:
        ValueError: ``memory`` given without a ``memory_key``.
    """

    def __init__(
        self,
        base: EmpiricalBeliefs | None = None,
        forecast_fn: AIFunction | None = None,
        memory: MemoryBackend | None = None,
        memory_key: str | None = None,
    ) -> None:
        if memory is not None and not memory_key:
            raise ValueError("`memory_key` must be specified when using `memory`")
        if base is None:
            base = (
                EmpiricalBeliefs(memory=memory, stats_key=f"{memory_key}/stats")
                if memory is not None
                else EmpiricalBeliefs()
            )
        self._base = base
        self._forecast_fn = forecast_fn if forecast_fn is not None else forecast
        self._memory = memory
        self._notes_key = f"{memory_key}/notes" if memory_key else None

    async def estimate(
        self, task: TaskView, candidates: list[Candidate], value: float | None, history: list[AttemptRecord]
    ) -> dict[str, Estimate]:
        """Forecast a pass probability per candidate, anchored on base statistics.

        Raises:
            AIFunctionError: ``value`` is ``None`` — the forecast is a pass
                probability priced at the constant value scale. Checked before
                the forecast call so no tokens are spent on an unusable estimate.
        """
        if value is None:
            raise AIFunctionError("LLMForecaster requires a constant value scale (dollars a pass is worth)")
        stats = self._base.stats()
        notes = await self._recall_notes()

        forecaster = self._forecast_with_check(candidates)
        result = await forecaster(
            query=task.prompt,
            candidate_table=_format_candidate_table(candidates, stats),
            notes=notes,
        )

        prompt_tokens = _approx_tokens(task.prompt)
        out: dict[str, Estimate] = {}
        base_estimates = await self._base.estimate(task, candidates, value, history)
        for c in candidates:
            forecast_est = result.estimates.get(c.label)
            if forecast_est is None:
                # E4 safety net: fall back to the base estimate for this label.
                out[c.label] = base_estimates[c.label]
                continue
            p = min(max(forecast_est.pass_percentage / 100.0, 0.0), 1.0)
            cost = _approx_cost(
                output_price=c.prices.output,
                prompt_tokens=prompt_tokens,
                turns=max(forecast_est.turns, 1),
                output_tokens_per_turn=forecast_est.output_tokens_per_turn,
            )
            out[c.label] = Estimate(dist=Bernoulli(p=p, value=value), cost=cost)
        return out

    def _forecast_with_check(self, candidates: list[Candidate]) -> AIFunction:
        """Attach a post-condition asserting the forecast covers every candidate (E4)."""
        from ...ai_thread.postcondition import PostConditionResult

        labels = {c.label for c in candidates}

        def _covers_all(result: ForecastResult, **kwargs: object) -> PostConditionResult | None:
            del kwargs
            missing = labels - result.estimates.keys()
            if missing:
                return PostConditionResult(passed=False, message=f"Missing estimates for: {sorted(missing)}")
            return None

        return self._forecast_fn.replace(post_conditions=[_covers_all])

    async def _recall_notes(self) -> str:
        """Recall the reflection notes from memory, or ``""`` when none/stateless.

        Recalled grad-enabled under the ambient thread scope (the economic
        run's thread), so the recall event lands in that run's log and the
        notes surface as a routable ``ParameterNode`` on the economic node.
        That is the forecaster's textual learning channel: the backward pass
        refines downstream feedback against the run's trace and consolidates
        it into the notes like any other memory parameter.
        """
        if self._memory is None or self._notes_key is None:
            return ""
        try:
            view = await self._memory.recall(self._notes_key)
            return str(view.value or "")
        except Exception:  # noqa: BLE001 — a missing/unreadable note is non-fatal
            logger.debug("LLMForecaster: could not recall notes %r", self._notes_key, exc_info=True)
            return ""

    def update(self, record: AttemptRecord) -> None:
        """Delegate numeric learning to the base provider."""
        self._base.update(record)

    def settle(self, record_id: RecordId, score: float) -> None:
        """Delegate settlement to the base provider (E2)."""
        self._base.settle(record_id, score)

    def stats(self) -> dict[str, str]:
        """Delegate statistics to the base provider."""
        return self._base.stats()


def _approx_tokens(text: str) -> int:
    """Crude token count for a prompt: one token per 4 characters."""
    return max(len(text) // 4, 1)


def _approx_cost(
    output_price: float,
    prompt_tokens: int,
    turns: int,
    output_tokens_per_turn: int,
) -> float:
    """Approximate an attempt's dollar cost from its predicted turn shape.

    Assumes prompt caching, so input tokens are either a cache write (first
    time a token enters the context) or a cache read (every later turn re-
    sending it), never priced at the plain input rate. Rates are tied to the
    output price by the fixed ratios

        output = 4 x cache_write = 50 x cache_read

    rather than read from ``Prices`` — the estimate needs one anchor (the
    output rate, which callers always supply) and the ratios are stable
    across current models, whereas per-model cache rates are often left at
    their zero default.

    Per-turn context growth is approximated as one turn's output (the tool
    round-trip); with ``T`` turns, per-turn output ``o``, and prompt ``P``:

    - written once:  ``P + (T - 1) * o``
    - read back:     ``sum_{k=2..T} (P + (k - 1) * o)``
    - generated:     ``T * o``
    """
    write_price = output_price / 4.0
    read_price = output_price / 50.0
    o = output_tokens_per_turn
    written = prompt_tokens + (turns - 1) * o
    read = (turns - 1) * prompt_tokens + o * (turns - 1) * turns / 2.0
    generated = turns * o
    return (written * write_price + read * read_price + generated * output_price) / 1e6


def _format_candidate_table(candidates: list[Candidate], stats: dict[str, str]) -> str:
    """Format candidates as a YAML block for the forecaster, with learned stats."""
    lines = ["candidates:"]
    for c in candidates:
        lines.append(f"  - label: {c.label}")
        if c.description:
            lines.append(f"    description: {c.description}")
        lines.append(f"    cost_per_million_output_tokens: {c.prices.output:.2f}")
        if c.label in stats:
            lines.append(f"    track_record: {stats[c.label]}")
    return "\n".join(lines)


# ── Default forecaster brain and its schema ───────────────────────


class CandidateForecast(BaseModel):
    """Forecaster prediction for a single candidate."""

    pass_percentage: int = Field(ge=0, le=100, description="Likelihood of completing the task, 0-100")
    turns: int = Field(
        ge=1,
        description="Expected model calls to finish the task: 1 for a direct answer, "
        "plus one per expected tool-call round",
    )
    output_tokens_per_turn: int = Field(ge=0, description="Expected output tokens generated per turn")


class ForecastResult(BaseModel):
    """Forecaster output: per-candidate predictions, keyed by candidate label."""

    estimates: dict[str, CandidateForecast]


# coordinator_tools_enabled=False: the forecaster is internal machinery and
# must not try to discover or message its parent (which is blocked awaiting
# this very forecast — that would deadlock the search).
@ai_function[ForecastResult](max_attempts=3, coordinator_tools_enabled=False)
def forecast(query: str, candidate_table: str, notes: str = ""):
    """For each candidate listed below, estimate its chance and cost of completing this task.

    Anchor on each candidate's track_record when present (it reports the observed
    pass rate, turn count, and output tokens per turn); adjust up or down for how
    well this specific task suits it. Return, per candidate:

    1. pass_percentage (integer, 0-100)
    2. turns (integer, >= 1): expected model calls — 1 for a direct answer, plus
       one per expected tool-call round (e.g. a task needing 2 searches before
       answering is 3 turns). A weaker model may need more turns on the same task.
    3. output_tokens_per_turn (integer): expected output tokens generated per turn.

    {candidate_table}

    Routing notes from past feedback:
    {notes}

    Task:
    ```
    {query}
    ```

    Output your estimates immediately. Do NOT reason beforehand.
    """
