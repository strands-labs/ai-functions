"""Synthesize the largest payout that fits a balance after fees.

Amounts are integer cents. The percentage fee is charged on the payout and
rounded up to a whole cent; the fixed fee is charged only for a nonzero payout.

    AWS_PROFILE=my-profile hatch run python examples/verified_payout.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def payout_inputs(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    """Define the inputs the synthesized function must handle.

    The proof covers every combination admitted here, with no upper bound on the
    balance, the fixed fee, or the limit: integers are arbitrary precision, so
    values beyond machine-word range are in scope. `fee_bps` is a rate in basis
    points, so 0 is a free payout and 10_000 is 100% of the payout.

    Inputs outside this domain are rejected before any synthesis or native call,
    which is why the implementation never has to define behavior for them.
    """
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


def maximum_safe_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    """State what makes a payout correct: it must fit, and it must be the largest that fits.

    Three requirements, in order below:

    1. The payout is nonnegative and respects both the limit and the balance.
    2. A positive payout leaves enough money for the fixed fee and the
       rounded-up percentage fee.
    3. Unless the limit binds, one more cent would not fit.

    Requirement 3 is what rules out answers that are safe but too small; without
    it, returning 0 for every input would satisfy the contract. Because fees
    increase monotonically with the payout, rejecting the next cent establishes
    that no larger permitted payout fits either.

    This is a specification, not a recipe. It says which answers are acceptable
    and leaves the formula or search to the synthesizer, so reviewing it does
    not mean reviewing an implementation.
    """
    assert 0 <= result <= payout_limit_cents
    assert result <= balance_cents
    if result > 0:
        # Whole cents left for fees. Negative means the fixed fee alone does not
        # fit, and the assertion below then fails, which is the intent.
        fee_budget = balance_cents - result - fixed_fee_cents
        # ceil(result * fee_bps / 10_000) <= fee_budget, written without division
        # so the contract stays exact integer arithmetic and never rounds.
        assert result * fee_bps <= fee_budget * 10_000
    if result < payout_limit_cents:
        # Maximality: the same affordability test must fail one cent higher.
        next_payout = result + 1
        next_fee_budget = balance_cents - next_payout - fixed_fee_cents
        assert next_payout * fee_bps > next_fee_budget * 10_000


@verified_ai_compile(
    pre_conditions=[payout_inputs],
    post_conditions=[maximum_safe_payout],
    max_attempts=5,
)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit.

    For a positive payout p, charge a fixed fee plus ceil(p * fee_bps / 10000)
    cents. A zero payout incurs no fee. Include fees in the balance constraint.
    """


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
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
            payout = max_payout.run_sync(balance, fixed_fee, fee_bps, limit)
            fees = fixed_fee + (payout * fee_bps + 9_999) // 10_000 if payout else 0
            print(
                f"balance={balance}, limit={limit}: payout={payout}, fees={fees}, remaining={balance - payout - fees}"
            )
        if arguments.show_artifacts and max_payout.artifact_dir is not None:
            for path in sorted(max_payout.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
