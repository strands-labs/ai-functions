"""The position of the first largest value, specified with indexing, range(), and quantifiers.

The contracts say where the answer is, not how to find it: the value there is
at least every value in the list, and strictly greater than every value
before it. Indexing out of range would raise in Python, so the translated
contract requires the index to be in range wherever it indexes.

    AWS_PROFILE=my-profile hatch run python examples/verified_first_maximum.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def nonempty(values: list[int]):
    """A maximum exists only for a nonempty list."""
    assert len(values) > 0


def first_maximum(result: int, values: list[int]):
    """values[result] is a maximum, and no earlier value reaches it."""
    assert 0 <= result < len(values)
    assert all(value <= values[result] for value in values)
    assert all(values[i] < values[result] for i in range(result))


@verified_ai_compile(pre_conditions=[nonempty], post_conditions=[first_maximum], max_attempts=5)
def first_maximum_index(values: list[int]) -> int:
    """Return the index of the first occurrence of the largest value."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling first_maximum_index...", flush=True)
            await first_maximum_index.compile()
        for values in ([3, 9, 2, 9], [5], [-4, -1, -1, -7], [2**70, 0, 2**70]):
            print(f"{values}: {first_maximum_index.run_sync(values)}")
        if arguments.show_artifacts and first_maximum_index.artifact_dir is not None:
            for path in sorted(first_maximum_index.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
