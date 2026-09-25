"""Generate a verified compiled function from a formal contract written in Python.

The goal is to compute the maximum payout given a certain budget, subject to tricky
rules regarding fees and rounding.

The contract only states that the result should be the largest affordable
payout given the rules. `verified.ai_compile` has the agent write an algorithm to find
it and prove it meets the contract for every input; after that, calls run native code
and no model is involved.

Demonstrates:
- Contracts written in Python (`@project.function`, `@project.proposition`)
- `verified.ai_compile`: generate verified implementation from a formal specification
"""

import asyncio
import logging

from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.dsl import assume, every, implies
from ai_functions.experimental.verified.lean import LeanProject

# A core-Lean project; the decorators below translate each Python function to Lean when it is defined.
project = LeanProject()


# @project.proposition: a proposition. Each `assert` is a conjunct, `assume` a hypothesis.
@project.proposition
def valid_inputs(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


# @project.function: a computable Lean definition, still an ordinary Python function too.
@project.function
def fees(payout: int, fixed_fee_cents: int, fee_bps: int) -> int:
    """The fixed fee plus the percentage fee rounded up to a cent; nothing for a zero payout."""
    return fixed_fee_cents + (payout * fee_bps + 9_999) // 10_000 if payout > 0 else 0


# A payout is affordable if it respects the limit and the balance covers it plus its fees.
@project.proposition
def affordable(payout: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    assert 0 <= payout <= payout_limit_cents
    assert payout + fees(payout, fixed_fee_cents, fee_bps) <= balance_cents


# The contract, result first: on valid inputs, the result is the largest affordable payout.
@project.proposition
def best_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    """The result is affordable, and no affordable payout is larger."""
    assume(valid_inputs(balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents))
    assert affordable(result, balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents)
    # `every(int)` quantifies over all integers; Lean must prove it, Python cannot enumerate it.
    assert all(
        implies(affordable(p, balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents), p <= result)
        for p in every(int)
    )


# The body is left empty: a model writes the implementation in Lean and proves `best_payout` for all inputs.
@verified.ai_compile(model=models.large, contract=best_payout, max_attempts=5)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit.

    For a positive payout p, charge a fixed fee plus ceil(p * fee_bps / 10000)
    cents. A zero payout incurs no fee. Include fees in the balance constraint.
    """


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    async with scope(on_event=print_event):
        print("Preparing Lean and compiling max_payout...", flush=True)
        await max_payout.compile()

    for balance, fixed_fee, fee_bps, limit in [
        (10_000, 30, 290, 20_000),
        (10_000, 30, 290, 5_000),
        (100, 30, 290, 1_000),
        (31, 30, 290, 1_000),
        (0, 30, 290, 1_000),
    ]:
        # Calls the verified native implementation; no model call once compiled.
        payout = await max_payout(balance, fixed_fee, fee_bps, limit)
        # `fees` is the same definition the contract uses, run here as plain Python.
        charged = fees(payout, fixed_fee, fee_bps)
        remaining = balance - payout - charged
        print(f"balance={balance}, limit={limit}: payout={payout}, fees={charged}, remaining={remaining}")

    print(f"Specification, implementation, and proof: {max_payout.artifact_dir}/Verified.lean")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
