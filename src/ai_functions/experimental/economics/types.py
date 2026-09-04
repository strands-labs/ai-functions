"""Data types for the economics module.

Every quantity in this module is denominated in dollars. Rewards, costs,
budgets, and reservation prices share one currency, so no exchange-rate
parameter (a ``lambda_``) exists anywhere in the API: an attempt is worth
making exactly when its expected reward exceeds its expected cost.

Invariants:
    E1 — dollars are the only unit. Any layer that introduces a second unit
    (scores, token counts, probabilities) must convert at its own boundary.

    E2 — an ``AttemptRecord`` is revisable: ``local_score`` is booked at run
    time from the candidate's own post-conditions (or a ``scorer``'s
    ``[0, 1]`` value, when given). A ``settled_score`` is written when
    downstream feedback arrives; consumers must treat ``settled_score`` as
    authoritative when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NewType

from pydantic import BaseModel, ConfigDict

from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.types import TokenUsage

if TYPE_CHECKING:
    from strands.models import Model

    from ai_functions.ai_thread.ai_function import AIFunction

RecordId = NewType("RecordId", str)
"""Stable identifier of one ``AttemptRecord``; the settlement cursor."""


def _default_label(model: object) -> str:
    """Best-effort display label for a ``Model | str`` value.

    Strings label themselves. For a Strands ``Model``, reads ``model_id``
    from ``get_config()``. Returns ``""`` when no label can be derived — the
    caller must then supply one explicitly.
    """
    if isinstance(model, str):
        return model
    get_config = getattr(model, "get_config", None)
    if get_config is None:
        return ""
    config = get_config()
    model_id = config.get("model_id") if isinstance(config, dict) else None
    return str(model_id) if model_id else ""


class Prices(BaseModel):
    """Token prices for one model, in dollars per million tokens.

    ``input`` and ``output`` are required: the module's premise is
    caller-supplied prices, and a silently free model would win every
    routing decision. Cache rates default to 0 for setups without caching.
    """

    model_config = ConfigDict(frozen=True)

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost_of(self, usage: TokenUsage) -> float:
        """Price ``usage`` at these rates.

        Args:
            usage: Token counts to price.

        Returns:
            Total cost in dollars.
        """
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / 1e6


@dataclass(frozen=True)
class PricedModel:
    """A model together with the prices the *caller* pays for it.

    The library ships no price table: negotiated contracts, provisioned
    throughput, and regional rates make any built-in default wrong for
    someone. Prices are therefore always user-supplied, once, in this
    pairing — examples define their own illustrative collection.

    Args:
        model: Strands model instance or model ID string; same convention
            as ``ThreadConfig.model``.
        prices: What the caller pays for this model's tokens.
        label: Short key for candidates built from this model; derived from
            the model id when empty.
        description: Optional context surfaced to belief estimators
            (e.g. "fast and cheap; weak at multi-step reasoning").

    Raises:
        ValueError: No label given and none can be derived from ``model``.
    """

    model: Model | str
    prices: Prices
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Derive ``label`` from the model when not set explicitly."""
        if not self.label:
            derived = _default_label(self.model)
            if not derived:
                raise ValueError(
                    "PricedModel has no label and none could be derived from the model; "
                    "set label= explicitly (required for Model instances that expose no model_id)"
                )
            object.__setattr__(self, "label", derived)


@dataclass(frozen=True)
class Candidate[**P, T]:
    """One way of attempting a task: a fully configured function plus its prices.

    A candidate is *any* ``AIFunction`` — a different model, a different
    thinking budget, a different prompt, or a non-LLM heuristic wrapped as a
    function. Variants are built with ``fn.replace(...)``, not with per-
    candidate model config.

    Args:
        label: Unique key for this candidate within an ``EconomicFunction``.
            Beliefs, records, and cost attribution are keyed by it.
        fn: The function to run for an attempt. Its post-conditions define
            *local* success; what success is worth is the economic
            function's ``value``.
        prices: Token prices used to convert this candidate's measured usage
            into dollars.
        description: Optional context surfaced to belief estimators.
    """

    label: str
    fn: AIFunction[P, T]
    prices: Prices
    description: str = ""


class TaskView(BaseModel):
    """What a belief estimator sees of one task.

    Carries both the rendered prompt and the original call arguments, so
    estimators can read structured task features instead of re-parsing
    prompt text.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str
    """The prompt as the candidate's function would render it."""
    arguments: dict[str, Any]
    """Bound call arguments (keyword form), before prompt interpolation."""


@dataclass(frozen=True)
class Ranking:
    """One candidate's rank entry in a decision round."""

    label: str
    reservation_price: float
    """Dollars; the reward in hand at which trying this candidate is exactly
    break-even. Sequential search under ``ReservationPricePolicy`` tries
    candidates in descending order of this."""
    net_value: float
    """Dollars; the myopic ``E[reward] - cost`` of a single committed attempt.
    The criterion under ``Greedy`` (the default policy) — the right metric
    for one-shot routing, which ``plan`` + manual execution is."""


class AttemptRecord(BaseModel):
    """Outcome of one attempt; the join point of event log, beliefs, and backward.

    Booked provisionally at run time and revised (settled) when downstream
    feedback arrives — see invariant E2. Beliefs implementations must store
    statistics in a form that supports replacing a record's contribution,
    not just appending it.
    """

    id: RecordId
    task: TaskView
    candidate: str
    """``Candidate.label`` of the candidate that ran."""
    cost: float
    """Dollars actually spent, measured from the attempt's event log."""
    usage: TokenUsage = TokenUsage()
    turns: int = 0
    """Model calls the attempt took (1 + one per tool round), measured from
    the attempt's event log. 0 on records booked without a measured run
    (e.g. ``Decision.report``)."""
    reward: float
    """Dollars this attempt was booked at: ``value * local_score`` (the full
    ``value`` for a perfect pass, proportionally less for a graded partial),
    0.0 on fail."""
    local_score: float
    """The attempt's score in ``[0, 1]``, booked at run time: the
    post-condition pass/fail (``1.0``/``0.0``) by default, or a caller
    ``scorer``'s value. A ``settled_score``, written from downstream
    feedback, is authoritative when present (E2)."""
    settled_score: float | None = None
    """Downstream-corrected score in ``[0, 1]``, written by settlement.
    ``None`` until feedback reaches this record. Authoritative when present."""


class EconomicsError(AIFunctionError):
    """Base of the economic function's failure modes; carries the run's bookkeeping.

    Args:
        message: Human-readable explanation.
        function_name: Name of the economic function that raised.
        records: The ``AttemptRecord``s booked before the failure, in order —
            a failed run still reports what it tried and spent.
    """

    def __init__(
        self,
        message: str,
        function_name: str = "",
        records: list[AttemptRecord] | None = None,
    ) -> None:
        self.records: list[AttemptRecord] = records if records is not None else []
        super().__init__(message, function_name=function_name)


class Abstained(EconomicsError):
    """No candidate was worth running: every net value was negative at decision time.

    Raised by ``EconomicFunction.__call__`` instead of knowingly spending
    more than the expected reward. Callers that want to anticipate (or
    handle) abstention without an exception use ``plan()``, whose
    ``Decision.candidate`` is ``None`` in the same condition.
    """


class BudgetExceeded(EconomicsError):
    """An attempt would charge more than the remaining budget allows."""


class CandidatesExhausted(EconomicsError):
    """Every profitable candidate was tried and none produced a passing result."""
