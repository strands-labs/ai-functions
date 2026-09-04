"""Economic semantics for AI functions: value, cost, and optimal effort.

Post-conditions give a function correctness semantics; this module adds the
economics — what success is worth, what attempts cost, and therefore which
model to route to, when to escalate to a fallback, and when to stop.
Everything is denominated in dollars (E1), so the stopping rule is one
sentence: stop when no remaining attempt is expected to pay for itself.

``@routed`` is the entry point: it routes each call across a set of priced
models over *independent* attempts, samples while another attempt is expected
to pay for itself, and keeps the best result by score (the post-condition
pass/fail by default, or a caller-supplied ``[0, 1]`` grader). It constructs an
:class:`EconomicFunction`, which mirrors the calling surface of ``AIFunction``
and adds ``plan()``.

The top level exports the decorator path. The pure search core —
:class:`~ai_functions.experimental.economics.search.Search`,
:class:`~ai_functions.experimental.economics.search.Estimate`, the reward distributions,
and the :class:`~ai_functions.experimental.economics.search.Policy` implementations —
lives in :mod:`ai_functions.experimental.economics.search` for power users.
"""

from __future__ import annotations

from .beliefs import (
    Beliefs,
    EmpiricalBeliefs,
    LLMForecaster,
    ObservedAttempt,
    RoutingMemory,
)
from .decorators import routed
from .function import (
    ATTEMPT_EVENT,
    DECISION_EVENT,
    Decision,
    EconomicFunction,
    Scorer,
    Value,
    attempts,
    decisions,
    spend,
)
from .types import (
    Abstained,
    AttemptRecord,
    BudgetExceeded,
    Candidate,
    CandidatesExhausted,
    EconomicsError,
    PricedModel,
    Prices,
    Ranking,
    RecordId,
    TaskView,
)

__all__ = [
    "ATTEMPT_EVENT",
    "Abstained",
    "AttemptRecord",
    "Beliefs",
    "BudgetExceeded",
    "Candidate",
    "CandidatesExhausted",
    "DECISION_EVENT",
    "Decision",
    "EconomicFunction",
    "EconomicsError",
    "EmpiricalBeliefs",
    "LLMForecaster",
    "ObservedAttempt",
    "PricedModel",
    "Prices",
    "Ranking",
    "RecordId",
    "RoutingMemory",
    "Scorer",
    "TaskView",
    "Value",
    "attempts",
    "decisions",
    "routed",
    "spend",
]
