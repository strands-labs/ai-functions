"""Banker's rounding of a fraction, specified without dividing.

Python's round() sends exact halves to the even neighbor: round(2.5) == 2 and
round(3.5) == 4. Currency code that rounds a rate or a split by hand often gets
this wrong. The contract below never divides: it scales the error by twice the
denominator, so it stays exact for any integers, and it states the tie rule
separately. The implementation is free to divide however it likes.

    AWS_PROFILE=my-profile hatch run python examples/verified_round_half_even.py --show-artifacts
"""

import argparse
import asyncio
import logging
from fractions import Fraction

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def positive_denominator(numerator: int, denominator: int):
    """A fraction with a positive denominator; negate both to flip a sign."""
    assert denominator > 0


def nearest_with_ties_to_even(result: int, numerator: int, denominator: int):
    """result is at most half a unit from numerator/denominator; exactly half means even."""
    # 2 * denominator * (numerator / denominator - result), in integers.
    error = 2 * (numerator - result * denominator)
    assert abs(error) <= denominator
    if abs(error) == denominator:
        assert result % 2 == 0


@verified_ai_compile(pre_conditions=[positive_denominator], post_conditions=[nearest_with_ties_to_even], max_attempts=5)
def round_half_even(numerator: int, denominator: int) -> int:
    """Round numerator/denominator to the nearest integer, sending exact halves to the even neighbor.

    Compare twice the remainder of floor division with the denominator.
    """


async def main() -> None:
    """Compile with the agent event feed, then compare with Python's round()."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling round_half_even...", flush=True)
            await round_half_even.compile()
        for numerator, denominator in [(5, 2), (7, 2), (-5, 2), (2, 3), (10**30 + 1, 2), (-7, 4)]:
            value = round_half_even.run_sync(numerator, denominator)
            print(f"{numerator}/{denominator} -> {value} (Python round: {round(Fraction(numerator, denominator))})")
        if arguments.show_artifacts and round_half_even.artifact_dir is not None:
            for path in sorted(round_half_even.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
