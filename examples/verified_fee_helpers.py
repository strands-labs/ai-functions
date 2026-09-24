"""The largest payout after fees, with the fee written as an ordinary Python helper.

Same policy as verified_payout.py, stated the way a payments service computes
fees: a fixed fee plus a percentage rounded up with integer division. The
helper becomes a named Lean definition in the specification.

    AWS_PROFILE=my-profile hatch run python examples/verified_fee_helpers.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def fee(payout, fixed_fee_cents, fee_bps):
    """Fees for a payout: none for zero, else the fixed fee plus the percentage rounded up."""
    if payout == 0:
        return 0
    return fixed_fee_cents + (payout * fee_bps + 9_999) // 10_000


def payout_inputs(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    """Nonnegative amounts, and a rate between 0% and 100% in basis points."""
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


def maximum_safe_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    """The payout and its fees fit the balance, and one more cent would not."""
    assert 0 <= result <= payout_limit_cents
    assert result + fee(result, fixed_fee_cents, fee_bps) <= balance_cents
    if result < payout_limit_cents:
        assert result + 1 + fee(result + 1, fixed_fee_cents, fee_bps) > balance_cents


@verified_ai_compile(
    pre_conditions=[payout_inputs],
    post_conditions=[maximum_safe_payout],
    max_attempts=5,
)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling max_payout...", flush=True)
            await max_payout.compile()
        cases = [(10_000, 30, 290, 20_000), (100, 30, 290, 1_000), (31, 30, 290, 1_000)]
        for balance, fixed_fee, fee_bps, limit in cases:
            payout = max_payout.run_sync(balance, fixed_fee, fee_bps, limit)
            print(f"balance={balance}, limit={limit}: payout={payout}, fees={fee(payout, fixed_fee, fee_bps)}")
        if arguments.show_artifacts and max_payout.artifact_dir is not None:
            for path in sorted(max_payout.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
