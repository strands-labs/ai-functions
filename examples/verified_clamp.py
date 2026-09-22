"""A verified native function defined entirely through Python contracts.

Requires CPython 3.12+. The first compilation prepares Lean, builds the bridge,
and synthesizes and checks one implementation; subsequent calls reuse it.

The default model uses Amazon Bedrock. Select your AWS credentials with
AWS_PROFILE when running the example.
"""

import asyncio
import logging

from ai_functions import scope
from ai_functions.ai_thread import AIFunctionError
from ai_functions.cli import print_event
from ai_functions.experimental.verified_compile import verified_ai_compile


def valid_bounds(lo: int, hi: int):
    assert lo <= hi


def check_clamp(result: int, x: int, lo: int, hi: int):
    assert lo <= result <= hi
    if x < lo:
        assert result == lo
    elif x > hi:
        assert result == hi
    else:
        assert result == x


@verified_ai_compile(pre_conditions=[valid_bounds], post_conditions=[check_clamp], max_attempts=3)
def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x to the inclusive interval [lo, hi]."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""
    try:
        async with scope(on_event=print_event):
            print("Preparing Lean and compiling clamp...", flush=True)
            await clamp.compile()
        print(clamp.run_sync(12, 0, 10))
        print(clamp.run_sync(-5, 0, 10))
    except AIFunctionError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
