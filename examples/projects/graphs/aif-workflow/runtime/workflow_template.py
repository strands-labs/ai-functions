"""<one line: what this workflow does>.

Copied from the aif-workflow skill. Edit only the marked block below — the wiring
around it is the same in every workflow, and `run_workflow` owns everything else.

    uv run python .aif/<slug>/workflow.py
"""

from __future__ import annotations

import asyncio
import logging

import conditions as c
from runtime import Step, prompt_section, run_workflow

logger = logging.getLogger("aif.workflow")


async def main() -> None:
    """Run the workflow and log each step's outcome."""
    # ── EDIT THIS BLOCK ───────────────────────────────────────────────────────
    # One Step per agent thread. Prompts live in prompt.md, conditions in
    # conditions.py; this list only wires them together. Steps with no unmet
    # `depends_on` run concurrently. Use `{step_name}` in a prompt to interpolate
    # that step's result.
    # Names below match conditions_template.py — rename both together.
    task = "the user's request, verbatim"
    objective = "what success means, in checkable terms"
    steps = [
        Step(
            name="first_step",
            prompt=prompt_section("first_step"),
            pre_conditions=(c.input_present,),
            post_conditions=(
                (c.output_runs_cleanly, "code"),
                (c.output_holds_the_right_total, "code"),
            ),
            max_attempts=4,
        ),
        Step(
            name="second_step",
            prompt=prompt_section("second_step"),
            post_conditions=((c.report_mentions_the_total, "code"),),
            depends_on=("first_step",),
            max_attempts=4,
        ),
    ]
    # ── END EDIT BLOCK ────────────────────────────────────────────────────────

    # One agent per step, conditions checked, the whole run recorded under runs/.
    results = await run_workflow(
        task=task,
        objective=objective,
        steps=steps,
        workspace=c.WORKSPACE,
    )
    # Report how each step ended; `error` says why a failed step gave up.
    for name, outcome in results.items():
        logger.info("%s: %s", name, outcome.status)
        if outcome.error:
            logger.info("  %s", outcome.error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
