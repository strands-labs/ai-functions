"""Beliefs that converge: the function learns which candidate earns its keep.

No estimates are given anywhere. ``@routed`` defaults to learning beliefs:
the first calls explore from a uniform prior, every attempt's outcome
(pass/fail against the verifier, cost from the event log) updates the
per-candidate statistics, so each candidate's pass rate and cost estimate
sharpen as evidence accumulates.

The function is called in a plain loop, like any async function. The only
extra machinery is read-only: ``solve.beliefs.stats()`` snapshots what has
been learned so far.
"""

import asyncio
import logging

from _economics_utils import HAIKU, SONNET, check_sat, format_instance, make_instance
from _utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.economics import Abstained, CandidatesExhausted, routed

# A stream of same-difficulty instances — the setting where population-level
# statistics are the right thing to learn. At this ratio the cheap model
# clears some instances and misses others, while the strong model solves most
# instances
TASKS = [(10, 3.6, seed) for seed in range(100, 106)]


class Assignment(BaseModel):
    values: list[bool] = Field(description="values[i] is the truth value of variable x(i+1)")


# value is the dollar worth of a solved instance. It must exceed a model's
# per-attempt cost or the function correctly declines to play (E5); set well
# above an attempt actually costs so the search keeps trying and its beliefs
# can sharpen.
@routed(models=[HAIKU, SONNET], value=0.50)
@ai_function[Assignment](post_conditions=[check_sat])
def solve(clauses: str, n_vars: int):
    """Find a satisfying assignment for this 3-SAT formula over variables x1..x{n_vars}.
    Work through the clauses carefully and check your assignment before answering.

    {clauses}"""


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("Learning beliefs: no estimates given, statistics accumulate")

    lines = []
    for i, (n_vars, ratio, seed) in enumerate(TASKS, 1):
        clauses = make_instance(n_vars=n_vars, seed=seed, ratio=ratio)
        try:
            await solve(clauses=format_instance(clauses), n_vars=n_vars)
            outcome = "solved"
        except Abstained:
            outcome = "declined (no candidate worth its cost)"
        except CandidatesExhausted:
            outcome = "unsolved (every candidate tried)"
        lines.append(f"task {i:>2}  {outcome}")

        if i in (3, len(TASKS)):
            lines.append("")
            lines.append(f"  beliefs after {i} tasks:")
            for arm_label, summary in solve.beliefs.stats().items():
                lines.append(f"    {arm_label:<16} {summary}")
            lines.append("")

    display("Routing under learned beliefs", "\n".join(lines), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
