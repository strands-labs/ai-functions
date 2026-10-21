"""Conditions for a generated workflow — a starting point, not a finished file.

Copied from the aif-workflow skill as a starting point. Replace the paths, the
`expected_*` helpers and the conditions themselves; keep the shape.

Two rules the shape encodes:

  1. **Expected values come from the input**, computed here — never read back from the
     agent's own output, or the check is circular.
  2. **`checks` handles plumbing; the comparison stays visible** in each condition's
     body, because the UX shows this source so a human can audit what was verified.
"""

from __future__ import annotations

import os
from pathlib import Path

import checks
from ai_functions.ai_thread.postcondition import PostConditionResult


def _default_workspace() -> Path:
    """This package's workspace under the project's `data/` directory.

    Derived from this file's location — `.aif/<slug>/conditions.py` sits two levels below
    the project root — so the workflow runs from anywhere with no absolute path baked in.

    **Set this to the directory the user actually named.** Leaving a placeholder here is
    how a run ends up blocked on `<somewhere>/input.csv is missing`.
    """
    package = Path(__file__).resolve().parent
    return package.parents[1] / "data" / package.name


WORKSPACE = Path(os.environ.get("AIF_WORKSPACE") or _default_workspace()).expanduser().resolve()
INPUT = WORKSPACE / "input.csv"
OUTPUT = WORKSPACE / "output.json"

input_unchanged = checks.file_unchanged(INPUT, because="It is the source data")
"""Built at import time so the digest predates the run. See checks.file_unchanged."""


# ── Expected values, computed from the input ───────────────────────────────────


def expected_total() -> float:
    """What a correct answer is, derived from INPUT rather than from the agent.

    Returns:
        The sum of the input's ``amount`` column, rounded to cents.
    """
    rows = checks.read_csv(INPUT) or []
    return round(sum(float(row["amount"]) for row in rows), 2)


# ── Pre-conditions: gates, checked before the thread spawns ────────────────────


def input_present() -> PostConditionResult | None:
    """The input the step needs is present."""
    if not INPUT.is_file():
        return PostConditionResult(passed=False, message=f"{INPUT} is missing")
    return None


# ── Post-conditions: checked against the workspace after each attempt ──────────


def output_runs_cleanly(result: str) -> PostConditionResult | None:
    """The script the step wrote exists and exits zero."""
    del result  # Check the world, not the agent's account of it.
    return checks.must_exit_zero(WORKSPACE / "compute.py")


def output_holds_the_right_total(result: str) -> PostConditionResult | None:
    """The output reports the true total."""
    del result
    # Read the file the step wrote, not the agent's reply.
    data = checks.read_json(OUTPUT)
    if data is None:
        return PostConditionResult(passed=False, message=f"{OUTPUT.name} is missing or unparseable; write it.")
    # Compare against the value computed from the input.
    wanted = expected_total()
    if not checks.close(data.get("total"), wanted):
        return PostConditionResult(
            passed=False,
            message=f"{OUTPUT.name} total is {data.get('total')!r}; it should be {wanted}.",
        )
    return None


def report_mentions_the_total(result: str) -> PostConditionResult | None:
    """The report states the total as $-formatted currency."""
    del result
    wanted = checks.currency(expected_total())
    if checks.missing_strings(OUTPUT, [wanted]):
        return PostConditionResult(
            passed=False,
            message=f"{OUTPUT.name} must state the total as `{wanted}` — dollar sign, separators, 2 decimals.",
        )
    return None
