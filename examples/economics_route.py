"""Custom beliefs: routing from task structure.

The beliefs decide what's worth trying. The default ``EmpiricalBeliefs``
learns population pass rates; this example replaces them with a custom
``RatioBeliefs`` that reads the feature governing 3-SAT hardness (the
clause/variable ratio) directly from the call arguments — so routing is
task-aware from the first call, no learning needed.

Three instances of increasing difficulty hit the three possible outcomes:
the easy one routes to the cheap model, the medium one to the strong
model, and for the hardest the function abstains (no candidate's expected
reward covers its cost). The example uses ``plan()`` to preview each
decision without spending, then executes the chosen candidate itself.
"""

import asyncio
import logging
import math

from _economics_utils import HAIKU, SONNET, check_sat, format_instance, make_instance
from _utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.economics import (
    AttemptRecord,
    Beliefs,
    Candidate,
    TaskView,
    routed,
)
from ai_functions.experimental.economics.search import Bernoulli, Estimate, Greedy

VALUE = 0.05  # a verified solution is worth 5 cents

# Per-candidate skill: the clause/variable ratio at which success crosses
# 50%. Illustrative, not measured — the default learned beliefs would
# estimate this from outcomes instead.
SKILL_MIDPOINT = {HAIKU.label: 3.2, SONNET.label: 4.2}


class RatioBeliefs(Beliefs):
    """Task-dependent estimates from the SAT clause/variable ratio.

    ``estimate`` receives the structured call arguments, not just prompt
    text, so the governing feature is read directly from ``task.arguments``.
    """

    async def estimate(
        self, task: TaskView, candidates: list[Candidate], value: float, history: list[AttemptRecord]
    ) -> dict[str, Estimate]:
        ratio = (task.arguments["clauses"].count("\n") + 1) / task.arguments["n_vars"]
        out: dict[str, Estimate] = {}
        for c in candidates:
            p = 1.0 / (1.0 + math.exp(2.0 * (ratio - SKILL_MIDPOINT[c.label])))
            est_tokens = 300 + 400 * ratio  # harder instances burn more reasoning tokens
            out[c.label] = Estimate(dist=Bernoulli(p=p, value=value), cost=c.prices.output * est_tokens / 1e6)
        return out


class Assignment(BaseModel):
    values: list[bool] = Field(description="values[i] is the truth value of variable x(i+1)")


# Greedy ranks by net value (E[reward] - cost): the pick is whichever candidate
# the custom beliefs say is most profitable for this task, and it stops once a
# result is in hand — the right rule for one-shot routing (the decorator's
# default is ReservationPricePolicy, which prices in fallback options).
@routed(models=[HAIKU, SONNET], value=VALUE, beliefs=RatioBeliefs(), policy=Greedy())
@ai_function[Assignment](post_conditions=[check_sat])
def solve(clauses: str, n_vars: int):
    """Find a satisfying assignment for this 3-SAT formula over variables x1..x{n_vars}.
    Work through the clauses carefully and check your assignment before answering.

    {clauses}"""


TASKS = [
    ("easy", 8, 2.6, 100),
    ("medium", 12, 3.6, 200),
    ("hard", 16, 4.4, 300),
]


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("Planning: one decision per task, abstain when hopeless")

    lines = []
    for label, n_vars, ratio, seed in TASKS:
        formula = format_instance(make_instance(n_vars=n_vars, seed=seed, ratio=ratio))

        decision = await solve.plan(clauses=formula, n_vars=n_vars)

        # Under Greedy the pick is the highest net value, so rank the display
        # by net value too (explain() orders by reservation price by default).
        by_nv = sorted(decision.ranking, key=lambda r: r.net_value, reverse=True)
        ranking = "  ".join(f"{r.label} nv=${r.net_value:+.4f}" for r in by_nv)
        lines.append(f"{label:<8} ({n_vars} vars, ratio {ratio})   {ranking}")

        if decision.candidate is None:
            lines.append(f"{'':8} → abstain: no candidate expects to cover its cost")
            lines.append("")
            continue

        # Execute the decision ourselves; report() feeds the outcome to the
        # beliefs, same as a normal call does automatically after each attempt.
        try:
            result = await decision.candidate.fn(clauses=formula, n_vars=n_vars)
            outcome = "solved: " + "".join("1" if v else "0" for v in result.values)
        except Exception:
            result, outcome = None, "failed"
        decision.report(result)
        lines.append(f"{'':8} → {decision.candidate.label}: {outcome}")
        lines.append("")

    display("Decisions", "\n".join(lines), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
