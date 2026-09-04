"""``@routed`` — the entry point to the economics module.

Stacks on ``@ai_function`` and returns an
:class:`~.function.EconomicFunction` that is called exactly like the function
it wraps. It routes each call to the candidate worth running across a set of
priced models, samples independent attempts while another is expected to pay
for itself, and returns the best result by score — declining
(:class:`~.types.Abstained`) when no candidate's expected reward covers its
cost. Attempts are independent; cumulative work that accumulates across passes
belongs to an agentic orchestrator, not this decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .beliefs import Beliefs, EmpiricalBeliefs
from .function import EconomicFunction, Scorer
from .search import Policy, ReservationPricePolicy
from .types import Candidate, PricedModel

if TYPE_CHECKING:
    from ai_functions.ai_thread.ai_function import AIFunction


def _candidates_from(
    fn: AIFunction,
    models: list[PricedModel] | None,
    candidates: list[Candidate] | None,
) -> dict[str, Candidate]:
    """Build the label→candidate mapping from ``models`` or explicit ``candidates``.

    Exactly one source must be given. Raises on duplicate labels.
    """
    if (models is None) == (candidates is None):
        raise ValueError("provide exactly one of models= or candidates=")
    built = (
        [
            Candidate(label=m.label, fn=fn.replace(model=m.model), prices=m.prices, description=m.description)
            for m in models
        ]
        if models is not None
        else list(candidates or [])
    )
    out: dict[str, Candidate] = {}
    for c in built:
        if c.label in out:
            raise ValueError(f"duplicate candidate label {c.label!r}")
        out[c.label] = c
    return out


def routed[**P, T](
    *,
    value: float,
    models: list[PricedModel] | None = None,
    candidates: list[Candidate[P, T]] | None = None,
    scorer: Scorer | None = None,
    beliefs: Beliefs | None = None,
    budget: float | None = None,
    policy: Policy | None = None,
    max_tries: int | None = 1,
) -> Callable[[AIFunction[P, T]], EconomicFunction[P, T]]:
    """Route each call to the candidate worth running; sample and keep the best.

    One call = one search over independent attempts: try the candidate with the
    highest expected reward, escalate/re-sample while another attempt is
    expected to pay for itself, keep the highest-scoring result, and decline
    (:class:`~.types.Abstained`) when no candidate's expected reward covers its
    cost.

    Each attempt is scored in ``[0, 1]`` — the post-condition pass/fail by
    default (a pass is ``1.0``), or a caller-supplied ``scorer`` for
    partial credit — and its reward is ``value * score``. Attempts are
    independent: the score of one does not depend on another, which is what
    makes the per-candidate statistics well-defined.

    Args:
        value: Dollars a fully-successful (score ``1.0``) result is worth — a
            positive constant. The scale estimates are built at.
        models: Priced models to build candidates from, one per entry.
            Exactly one of ``models`` and ``candidates`` must be given.
        candidates: Explicit candidates, for variants beyond model swaps.
        scorer: Grade a passing result in ``[0, 1]`` (e.g. an F1
            score). ``None`` scores every pass ``1.0`` (binary pass/fail). An
            out-of-range score raises a ``ValueError``.
        beliefs: Estimate/learn provider. Defaults to a fresh
            :class:`~.beliefs.EmpiricalBeliefs`.
        budget: Hard dollar cap per call.
        policy: Search policy; defaults to ``ReservationPricePolicy``
            (Weitzman: sample while a candidate's reservation price beats the
            best reward in hand). Pass ``Greedy()`` for one-shot highest-net-
            value routing that stops at the first passing attempt.
        max_tries: Attempts per candidate per call; ``None`` = unbounded
            (requires ``budget``).

    Returns:
        A decorator producing the configured ``EconomicFunction``.

    Raises:
        ValueError: A callable or non-positive ``value``; both or neither of
            ``models``/``candidates`` given; duplicate labels; or
            ``max_tries=None`` without ``budget``.
    """
    if callable(value):
        raise ValueError("value must be a constant dollar amount; grade partial success with scorer= instead")
    if value <= 0:
        raise ValueError(f"value must be positive dollars, got {value}")

    def _decorate(fn: AIFunction[P, T]) -> EconomicFunction[P, T]:
        return EconomicFunction(
            candidates=_candidates_from(fn, models, candidates),
            value=value,
            scorer=scorer,
            beliefs=beliefs if beliefs is not None else EmpiricalBeliefs(),
            budget=budget,
            policy=policy if policy is not None else ReservationPricePolicy(),
            max_tries=max_tries,
        )

    return _decorate
