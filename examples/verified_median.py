"""Synthesize a median algorithm from ordering properties, not a reference algorithm.

Run with an authenticated model profile, for example:
    AWS_PROFILE=my-profile hatch run python examples/verified_median.py

Add --show-artifacts to print the generated specification, code, and proof path.
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def is_median(result: int, a: int, b: int, c: int):
    """The median is an input with at least two inputs on either side (including ties)."""
    assert result == a or result == b or result == c
    assert (a <= result and b <= result) or (a <= result and c <= result) or (b <= result and c <= result)
    assert (a >= result and b >= result) or (a >= result and c >= result) or (b >= result and c >= result)


@verified_ai_compile(
    model="global.anthropic.claude-opus-5",
    post_conditions=[is_median],
    max_attempts=5,
)
def median_of_three(a: int, b: int, c: int) -> int:
    """Return the median of three arbitrary integers, including duplicates and negative values."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the generated source file for inspection")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling median_of_three...", flush=True)
            await median_of_three.compile()
        for values in [(9, 1, 5), (9, 9, -4), (-10, -3, -7), (2**100, -(2**100), 42)]:
            result = median_of_three.run_sync(*values)
            print(f"median{values} = {result}")
        if arguments.show_artifacts:
            directory = median_of_three.artifact_dir
            if directory is not None:
                for path in sorted(directory.glob("*.lean")):
                    print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
