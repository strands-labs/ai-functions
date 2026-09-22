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
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


def maximum_safe_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    assert 0 <= result <= payout_limit_cents
    assert result <= balance_cents
    if result > 0:
        fee_budget = balance_cents - result - fixed_fee_cents
        assert result * fee_bps <= fee_budget * 10_000
    if result < payout_limit_cents:
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
