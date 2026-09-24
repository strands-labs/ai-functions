"""The smallest level loan payment under the lender's own rounding.

Each period the lender charges interest on the outstanding balance, rounded half
up to the cent, and the borrower pays the same amount every period. What is the
smallest payment that clears the loan on schedule? The textbook annuity formula,
rounded to the cent, ignores the per-period rounding; on 998 of 2,000 sampled
loans it leaves cents owed after the final payment.

The contract states the answer's defining property, with the lender's policy
written once as a Python helper: the payment clears the loan, and one cent less
would not. The final balance falls as the payment rises, so exactly one payment
has that property. The requested implementation bisects over payments and
simulates each one on natural numbers, stopping once the loan is paid off; the
proof relates that simulation to balance_after, which runs every period on
integers.

    AWS_PROFILE=my-profile hatch run python examples/verified_loan_payment.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def balance_after(principal_cents, rate_bps, payment_cents, periods):
    """The lender's policy: each period, interest rounded half up to the cent, then the payment."""
    balance = principal_cents
    for _ in range(periods):
        balance = balance + (balance * rate_bps + 5_000) // 10_000 - payment_cents
    return balance


def loan_terms(principal_cents: int, rate_bps: int, periods: int):
    """A nonnegative principal, a periodic rate from 0% to 100% in basis points, and 1 to 1,200 periods."""
    assert principal_cents >= 0
    assert 0 <= rate_bps <= 10_000
    assert 1 <= periods <= 1_200


def smallest_payment(result: int, principal_cents: int, rate_bps: int, periods: int):
    """result clears the loan in the given periods, and one cent less would not."""
    assert result >= 0
    assert balance_after(principal_cents, rate_bps, result, periods) <= 0
    assert result == 0 or balance_after(principal_cents, rate_bps, result - 1, periods) > 0


@verified_ai_compile(pre_conditions=[loan_terms], post_conditions=[smallest_payment], max_attempts=5)
def level_payment(principal_cents: int, rate_bps: int, periods: int) -> int:
    """Return the smallest level payment in cents that clears the loan under balance_after.

    Bisect the payment between 0 and 2 * principal_cents, which always clears the loan.
    Requirement: the implementation must not call balance_after, or the loop it unfolds to;
    both run every period on Int and are too slow. Simulate each candidate payment with a local
    recursive function on Nat that stops as soon as the balance is paid off (a paid-off balance
    stays paid off), and prove it agrees with balance_after. Lean keeps Nat unboxed below 2^63
    but Int only within 32 bits, and balance * rate_bps passes 2^31 for a $500,000 loan at 50 bps.
    """


def textbook_payment(principal_cents: int, rate_bps: int, periods: int) -> int:
    """The plausible-but-wrong answer, for contrast: the annuity formula most code uses.

    Rounding the formula's result to the nearest cent ignores the per-period rounding
    in balance_after, so the payment can be a cent short and leave money owed. The
    contracts, the synthesis, and the proof never use this function.
    """
    rate = rate_bps / 10_000
    if rate == 0:
        return round(principal_cents / periods)
    return round(principal_cents * rate / (1 - (1 + rate) ** -periods))


async def main() -> None:
    """Compile with the agent event feed, then compare with the textbook formula."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling level_payment...", flush=True)
            await level_payment.compile()
        # A 30-year mortgage and a 5-year car loan at 6% a year, where the rounded
        # textbook payment is a cent short, and a mortgage where it happens to agree.
        for principal, rate, periods in [(30_000_000, 50, 360), (3_000_000, 50, 60), (25_000_000, 50, 360)]:
            exact = level_payment.run_sync(principal, rate, periods)
            guess = textbook_payment(principal, rate, periods)
            owed = balance_after(principal, rate, guess, periods)
            if guess == exact:
                textbook = "the textbook formula agrees"
            elif owed > 0:
                textbook = f"the textbook formula's {guess} leaves {owed} cents owed"
            else:
                textbook = f"the textbook formula's {guess} is more than needed"
            print(f"${principal / 100:,.2f} at {rate} bps for {periods} periods: {exact} cents; {textbook}")
        if arguments.show_artifacts and level_payment.artifact_dir is not None:
            for path in sorted(level_payment.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
