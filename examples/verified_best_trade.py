"""The best single trade in a price series, specified over every pair of days.

The contract compares every purchase day with every later sale day, which is
quadratic: no pair gains more than the result, and some pair gains exactly
the result unless no trade gains at all. A linear, one-pass implementation
meets it on every input, and the proof has to show why.

    AWS_PROFILE=my-profile hatch run python examples/verified_best_trade.py --show-artifacts
"""

import argparse
import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def best_single_trade(result: int, prices: list[int]):
    """No purchase and later sale gains more than result, and one gains exactly result unless it is 0."""
    assert result >= 0
    assert all(
        all(prices[sell] - prices[buy] <= result for sell in range(buy + 1, len(prices))) for buy in range(len(prices))
    )
    assert result == 0 or any(
        any(prices[sell] - prices[buy] == result for sell in range(buy + 1, len(prices))) for buy in range(len(prices))
    )


@verified_ai_compile(post_conditions=[best_single_trade], max_attempts=5)
def best_trade(prices: list[int]) -> int:
    """Return the largest gain from buying on one day and selling on a later day, or 0 if none gains.

    Make a single pass that carries the lowest price seen so far and the best gain so far.
    Comparing every pair of days is quadratic; do not do that.
    """


async def main() -> None:
    """Compile with the agent event feed, then evaluate a few series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-artifacts", action="store_true", help="Print the specification, code, and proof path")
    arguments = parser.parse_args()
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling best_trade...", flush=True)
            await best_trade.compile()
        for prices in ([7, 1, 5, 3, 6, 4], [7, 6, 4, 3, 1], [], [2**70, 1, 2**71]):
            print(f"{prices}: {best_trade.run_sync(prices)}")
        if arguments.show_artifacts and best_trade.artifact_dir is not None:
            for path in sorted(best_trade.artifact_dir.glob("*.lean")):
                print(f"Specification, implementation, and proof: {path}")
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
