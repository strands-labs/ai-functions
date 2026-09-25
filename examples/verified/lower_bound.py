"""Compiles a function given it formal specification.

The contract (`lean/Lookup.lean`) says where a value should be inserted.
The agent writes an algorithm to solve the task and proves its correctness
for every sorted list and value; the example then checks it against
Python's `bisect` and times both.

Demonstrates:
- `verified.ai_compile` with a contract from an existing Lean project
- A recursive implementation proved for all inputs
"""

import asyncio
import bisect
import logging
import random
from pathlib import Path

from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject(Path(__file__).parent / "lean", imports=["Lookup"])


@verified.ai_compile(model=models.large, contract=project.symbols.Lookup.Contract, max_attempts=5)
def lower_bound(values: list[int], value: int) -> int:
    """Return the first insertion index that preserves sorted order, including duplicates and missing values."""


async def main() -> None:
    """Compile with the agent event feed, then exercise the native function."""

    async with scope(on_event=print_event):
        print("Preparing Lean and compiling lower_bound...", flush=True)
        await lower_bound.compile()

    for data, value in [([], 4), ([1, 3, 3, 8], 3), ([1, 3, 3, 8], 4), ([1, 3, 3, 8], 10), ([-9, -2, 7], -5)]:
        print(f"lower_bound({data}, {value}) =", await lower_bound(data, value))

    # Cross-check against the standard library on random sorted lists with duplicates and missing keys.
    rng = random.Random(0)
    cases, n = [], 5_000
    for _ in range(n):
        values = sorted(rng.choices(range(-50, 50), k=rng.randrange(40)))
        cases.append((values, rng.randrange(-60, 60)))
    native = [lower_bound.run_sync(values, key) for values, key in cases]  # compiled: a direct native call
    reference = [bisect.bisect_left(values, key) for values, key in cases]
    assert native == reference, "the verified function disagrees with bisect.bisect_left"
    print("Compiled code matches bisect.bisect_left on all examples")

    print(f"Specification, implementation, and proof: {lower_bound.artifact_dir}/Verified.lean")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
