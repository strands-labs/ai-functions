"""The Luhn check digit on payment card numbers, specified by what it guarantees.

The Luhn rule is written once, as two small Python helpers. The contract then
says only that the result is a digit and that appending it makes the number
pass. It does not say how to find the digit. Appending shifts every other
digit's doubling, so the direct computation is not the rule itself; the proof
has to connect them.

    AWS_PROFILE=my-profile hatch run python examples/verified_luhn_check_digit.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def luhn_value(digit, position):
    """A digit's contribution: every second digit from the right is doubled, less 9 above 9."""
    if position % 2 == 1:
        doubled = 2 * digit
        return doubled - 9 if doubled > 9 else doubled
    return digit


def luhn_total(digits):
    """The Luhn sum, counting positions from the rightmost digit."""
    return sum(luhn_value(digit, position) for position, digit in enumerate(reversed(digits)))


def decimal_digits(digits: list[int]):
    """Every entry is a decimal digit."""
    assert all(0 <= digit <= 9 for digit in digits)


def makes_valid_number(result: int, digits: list[int]):
    """result is a digit, and the number followed by it passes the Luhn check."""
    assert 0 <= result <= 9
    assert luhn_total(digits + [result]) % 10 == 0


@verified_ai_compile(pre_conditions=[decimal_digits], post_conditions=[makes_valid_number], max_attempts=5)
def check_digit(digits: list[int]) -> int:
    """Return the Luhn check digit for the payload digits.

    After the check digit is appended, the payload's rightmost digit is the first one doubled.
    """


async def main() -> None:
    """Compile with the agent event feed, then complete a few numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling check_digit...", flush=True)
            await check_digit.compile()
        for payload in ("7992739871", "453201511283036", "0", ""):
            digit = check_digit.run_sync([int(character) for character in payload])
            print(f"{payload or '(empty)'} -> {payload}{digit}")
        if arguments.show_artifacts and check_digit.artifact_dir is not None:
            for path in sorted(check_digit.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
