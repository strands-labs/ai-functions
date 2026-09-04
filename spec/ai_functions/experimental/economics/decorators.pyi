"""``@routed`` — the entry point to the economics module.

Stacks on ``@ai_function`` and returns an
:class:`~.function.EconomicFunction` that is called exactly like the function
it wraps. It routes each call to the candidate worth running across a set of
priced models over *independent* attempts, samples independent attempts while
another is expected to pay for itself, and returns the best result by score —
declining (:class:`~.types.Abstained`) when no candidate's expected reward
covers its cost. Attempts are independent; cumulative work that accumulates
across passes belongs to an agentic orchestrator, not this decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .beliefs import Beliefs
from .function import EconomicFunction, Scorer
from .search import Policy
from .types import Candidate, PricedModel

if TYPE_CHECKING:
    from ...ai_thread.ai_function import AIFunction


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
    ...
