"""The first transaction that overdraws an account, specified with prefix sums.

The contract restates the question directly: every balance before the answer
is nonnegative, and the balance right after it is negative. Written that way it
re-adds a prefix for every position, which is quadratic. The implementation can
keep one running balance, and the proof shows the two agree on every input.
The balance helper becomes a named Lean definition in the specification.

    AWS_PROFILE=my-profile hatch run python examples/verified_first_overdraft.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def balance_after(opening_balance, transactions, count):
    """The balance once the first count transactions have posted."""
    return opening_balance + sum(transactions[:count])


def opening_in_credit(opening_balance: int, transactions: list[int]):
    """The account starts with a nonnegative balance."""
    assert opening_balance >= 0


def is_first_overdraft(result: int, opening_balance: int, transactions: list[int]):
    """result is the first position whose posting leaves the balance negative, or -1 if none does."""
    assert -1 <= result < len(transactions)
    checked = len(transactions) if result == -1 else result
    assert all(balance_after(opening_balance, transactions, k + 1) >= 0 for k in range(checked))
    if result >= 0:
        assert balance_after(opening_balance, transactions, result + 1) < 0


@verified_ai_compile(pre_conditions=[opening_in_credit], post_conditions=[is_first_overdraft], max_attempts=5)
def first_overdraft(opening_balance: int, transactions: list[int]) -> int:
    """Return the index of the first transaction that takes the balance below zero, or -1.

    Make a single pass that carries the running balance and the first overdraft found so far.
    Re-adding a prefix for every position is quadratic; do not do that.
    """


async def main() -> None:
    """Compile with the agent event feed, then scan a few statements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling first_overdraft...", flush=True)
            await first_overdraft.compile()
        for opening, transactions in [(100, [-30, -50, -40, 90]), (0, [10, -10, 5]), (50, []), (0, [-1])]:
            print(f"opening {opening}, {transactions}: {first_overdraft.run_sync(opening, transactions)}")
        if arguments.show_artifacts and first_overdraft.artifact_dir is not None:
            for path in sorted(first_overdraft.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
