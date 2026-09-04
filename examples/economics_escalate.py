"""Cost-aware model routing with escalation — one decorator, one value for what success is worth.

``@routed`` turns an ``@ai_function`` into an economic function over several
models. The one required number is ``value``: what a verified success is
worth, in dollars. Given that, each call tries the candidate whose *reservation
index* is highest (here, the cheap model at first), escalates when the verifier
rejects the answer, and stops when no attempt is worth its cost.

A batch of 3-SAT instances — mostly easy, one hard (see ``_economics_utils.py``
for the difficulty knob) — against a straight-to-strong baseline shows the
savings in dollars.
"""

import asyncio
import logging

from _economics_utils import HAIKU, SONNET, check_sat, format_instance, make_instance
from _utils import console, display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.economics import EconomicsError, EmpiricalBeliefs, attempts, routed

# (n_vars, clause/variable ratio, seed): easy majority, one hard instance.
INSTANCES_SPEC = [
    (6, 2.6, 100),
    (7, 2.6, 200),
    (8, 2.6, 300),
    (9, 2.6, 400),
    (10, 2.6, 500),
    (11, 2.6, 600),
    (15, 4.3, 700),
]
INSTANCES = [(n, make_instance(n_vars=n, seed=seed, ratio=ratio)) for n, ratio, seed in INSTANCES_SPEC]


class Assignment(BaseModel):
    values: list[bool] = Field(description="values[i] is the truth value of variable x(i+1)")


# The whole @routed setup: which priced models compete, and what success
# is worth. Beliefs start uniform and learn from every call. `check_sat`
# verifies against the call's own `clauses`/`n_vars` arguments, so one
# function serves every instance. value is the dollar worth of a solved
# instance — set above the strong model's per-attempt cost so escalating to
# it is worthwhile (below it, the function would rightly decline to escalate).
@routed(models=[HAIKU, SONNET], value=0.50)
@ai_function[Assignment](post_conditions=[check_sat])
def solve(clauses: str, n_vars: int):
    """Find a satisfying assignment for this 3-SAT formula over variables x1..x{n_vars}.
    Work through the clauses carefully and check your assignment before answering.

    {clauses}"""


# Baseline for comparison: same function, strong model only, with its OWN
# fresh beliefs (replace() would otherwise share solve's).
solve_strong = solve.replace(
    candidates={"sonnet": solve.candidates["sonnet"]},
    beliefs=EmpiricalBeliefs(),
)


async def run_batch(label: str, solver) -> tuple[list[str], float]:
    """Solve every instance; report each candidate trail and the dollar total.

    ``trace`` runs the function like a plain call but keeps the run's event
    log reachable so ``attempts`` can report what happened; a failed run
    reports the same records via ``EconomicsError.records``.
    """
    console.print(f"[dim]running {label}…[/dim]")
    rows, total = [], 0.0
    for n_vars, clauses in INSTANCES:
        try:
            records = await attempts(await solver.trace(clauses=format_instance(clauses), n_vars=n_vars))
        except EconomicsError as exc:
            records = exc.records
        cost = sum(r.cost for r in records)
        total += cost
        trail = " → ".join(f"{r.candidate} {'✓' if r.local_score > 0 else '✗'}" for r in records) or "(declined)"
        rows.append(f"3-SAT {n_vars:>2} vars / {len(clauses):>2} clauses   {trail:<26} ${cost:.4f}")
    return rows, total


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("Routing with escalation vs. straight to sonnet")

    routed_rows, routed_total = await run_batch("routed (haiku → sonnet)", solve)
    baseline_rows, baseline_total = await run_batch("baseline (straight to sonnet)", solve_strong)

    lines = ["routed (haiku → sonnet):", *routed_rows, "", "baseline (straight to sonnet):", *baseline_rows, ""]
    lines.append(f"routed total    ${routed_total:.4f}")
    lines.append(f"baseline total  ${baseline_total:.4f}")
    saved = (1.0 - routed_total / baseline_total) * 100 if baseline_total else 0.0
    lines.append(f"saved           {saved:.0f}%")

    display("Cost comparison", "\n".join(lines), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
