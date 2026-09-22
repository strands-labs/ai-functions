"""Generate a sorted-table insertion lookup from quantified Python contracts.

    AWS_PROFILE=my-profile hatch run python examples/verified_lower_bound.py --show-artifacts

The contract states where the result belongs. It contains no search algorithm.
The native function is verified once for every sorted integer list and key.
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def sorted_values(values: list[int]):
    assert values == sorted(values)


def insertion_position(result: int, values: list[int], key: int):
    assert 0 <= result <= len(values)
    assert all(value < key for value in values[:result])
    assert all(value >= key for value in values[result:])


@verified_ai_compile(
    pre_conditions=[sorted_values],
    post_conditions=[insertion_position],
    max_attempts=5,
)
def lower_bound(values: list[int], key: int) -> int:
    """Return the first insertion index that preserves sorted order, including duplicates and missing keys."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-artifacts", action="store_true", help="Print the generated specification, code, and proof"
    )
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling lower_bound...", flush=True)
            await lower_bound.compile()
        for values, key in [([], 4), ([1, 3, 3, 8], 3), ([1, 3, 3, 8], 4), ([1, 3, 3, 8], 10), ([-9, -2, 7], -5)]:
            print(f"lower_bound({values}, {key}) = {lower_bound.run_sync(values, key)}")
        if arguments.show_artifacts and lower_bound.artifact_dir is not None:
            for path in sorted(lower_bound.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
