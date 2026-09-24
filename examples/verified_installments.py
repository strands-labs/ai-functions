"""Split an amount into installments that add up to the cent.

$100.00 in three installments is 3334, 3333, 3333 cents, not 3333 three times.
The contract describes any acceptable split: the right number of parts, an
exact total, sizes within one cent of each other, larger parts first. Those
four properties leave exactly one answer, and the contract never says how to
compute it. The result is a list, checked by index.

    AWS_PROFILE=my-profile hatch run python examples/verified_installments.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def valid_split(amount_cents: int, parts: int):
    """A nonnegative amount and at least one installment."""
    assert amount_cents >= 0
    assert parts >= 1


def even_installments(result: list[int], amount_cents: int, parts: int):
    """Exactly parts installments, adding up to the amount, within a cent of each other, largest first."""
    assert len(result) == parts
    assert sum(result) == amount_cents
    assert max(result) - min(result) <= 1
    assert all(result[i] >= result[i + 1] for i in range(parts - 1))


@verified_ai_compile(pre_conditions=[valid_split], post_conditions=[even_installments], max_attempts=5)
def installments(amount_cents: int, parts: int) -> list[int]:
    """Split the amount into parts installments that differ by at most one cent, largest first.

    Every installment gets amount_cents // parts; the first amount_cents % parts get one more cent.
    """


async def main() -> None:
    """Compile with the agent event feed, then split a few amounts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling installments...", flush=True)
            await installments.compile()
        for amount, parts in [(10_000, 3), (100, 7), (5, 8), (0, 2), (10**20 + 5, 4)]:
            split = installments.run_sync(amount, parts)
            print(f"{amount} in {parts}: {split} (sum {sum(split)})")
        if arguments.show_artifacts and installments.artifact_dir is not None:
            for path in sorted(installments.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
