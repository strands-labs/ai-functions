"""Measure warm FFI and public-call latency using verified identity functions.

Run after an ordinary package installation. Model responses are fixed; proof
checking, compilation, value conversion, and native execution are real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import resource
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental.lean.toolchain import DEFAULT_LEAN_TOOLCHAIN
from ai_functions.experimental.verified_compile import verified_ai_compile
from ai_functions.experimental.verified_compile.compiler import Candidate
from ai_functions.testing import ScriptedModel, Turn


def integer(value: int) -> int:
    """Return the integer unchanged."""


def sequence(value: list[int]) -> list[int]:
    """Return the sequence unchanged."""


def unchanged(result: Any, value: Any) -> None:
    """Specify identity without an executable function body."""
    assert result == value


def latency(call: Callable[[], object], count: int) -> float:
    """Return median microseconds per call across five warmed batches."""
    for _ in range(5):
        call()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(count):
            call()
        samples.append((time.perf_counter_ns() - start) / count / 1000)
    return statistics.median(samples)


async def main() -> None:
    """Compile two functions and report conversion and validation costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verified-benchmark.json"))
    parser.add_argument("--cache-dir", type=Path)
    options = parser.parse_args()
    candidate = Candidate(implementation="v0", proof="by intro v0 h; simp [pre, post, implementation]")
    functions = []
    for function in (integer, sequence):
        model = ScriptedModel([Turn(tool_calls=(("Candidate", candidate.model_dump()),))])
        wrapped = verified_ai_compile(
            post_conditions=[unchanged], model=model, max_attempts=0, cache_dir=options.cache_dir
        )(function)
        async with scope(on_event=print_event):
            print(f"Preparing Lean and compiling {function.__name__}...", flush=True)
            await wrapped.compile()
        functions.append(wrapped)

    cases = [
        ("small_int", functions[0], 12345, 3000),
        ("20001_bit_int", functions[0], 2**20000 + 7, 100),
        ("list_10", functions[1], list(range(10)), 1000),
        ("list_1000", functions[1], list(range(1000)), 100),
        ("list_10000", functions[1], list(range(10000)), 15),
    ]
    results = {}
    for name, function, value, count in cases:
        bound = function._spec.bind(value)
        results[name] = {
            "ffi_microseconds": latency(
                lambda function=function, bound=bound: function._artifact.invoke(function._spec, bound), count
            ),
            "function_microseconds": latency(lambda function=function, value=value: function.run_sync(value), count),
        }
        print(name, results[name], flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "python": sys.version,
        "toolchain": DEFAULT_LEAN_TOOLCHAIN,
        "cases": results,
        "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
    }
    options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
