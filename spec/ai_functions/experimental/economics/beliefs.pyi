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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...memory.frozen import Frozen
from .search import Estimate
from .types import AttemptRecord, RecordId, TaskView

if TYPE_CHECKING:
    from ...ai_thread.ai_function import AIFunction
    from ...memory.base import MemoryBackend
    from .types import Candidate


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

    notes: str = ...
    stats: Frozen[list[ObservedAttempt]] = ...


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
        ...

    def settle(self, record_id: RecordId, score: float) -> None:
        """Replace a booked record's score with its downstream-corrected value.

        Called by settlement when backward feedback reaches the economic
        function's decision node. Default implementation is a no-op.
        """
        ...

    def stats(self) -> dict[str, str]:
        """Return a human-readable summary line per candidate label.

        The introspection hook examples and dashboards read; also what
        :class:`LLMForecaster` interpolates into its prompt. Default
        implementation returns an empty dict for estimators with no learned
        state.
        """
        ...

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
        ...


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
    ) -> None: ...


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
        forecast_fn: AIFunction[..., object] | None = None,
        memory: MemoryBackend | None = None,
        memory_key: str | None = None,
    ) -> None: ...
