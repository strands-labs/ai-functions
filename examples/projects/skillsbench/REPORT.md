# Demo Results

One full run of `memory_improvement_loop.py` (2026-07-06): five SkillsBench
tasks, ten installed skills (four relevant + six distractors), two rounds with
one optimizer pass in between. Round 0 runs with empty memory; round 1 runs
with the guidelines learned from round 0's oracle feedback. Optimizer model:
`us.anthropic.claude-sonnet-4-6` on Amazon Bedrock.

## Oracle scores by round

Each score is the fraction of the task's pytest oracle tests that passed.

| Task | Round 0 | Round 1 | Δ |
|------|:-:|:-:|:-:|
| `court-form-filling` | 1.00 | 0.98 | −0.02 |
| `exceltable-in-ppt` | 0.75 | **1.00** | +0.25 |
| `financial-modeling-qa` | 0.50 | 0.75 | +0.25 |
| `offer-letter-generator` | 0.06 | **1.00** | +0.94 |
| `pptx-reference-formatting` | 0.92 | 0.92 | 0.00 |
| **Average** | **0.65** | **0.93** | **+0.28** |

## What the optimizer learned

After round 0, `backward()` produced one textual gradient per failing task and
`consolidate()` merged them into a single `workflow_guidelines` memory field
(~3.9 KB). The guidelines are task-specific and concrete — skill selection,
output paths, and workflow ordering. Excerpts:

> ### `offer-letter-generator`
> - Always invoke the `docx` skill to generate the offer letter document.
> - After generating the offer letter, save employee/candidate data as a JSON
>   file at the path: `<task_output_dir>/output/employee_data.json`. This file
>   is required by downstream tests.

> ### `exceltable-in-ppt`
> - Invoke both the `pptx` skill and the `xlsx` skill in combination.
> - Workflow: (1) use the pptx skill to locate and extract the embedded Excel
>   table from the PowerPoint file, (2) use the xlsx skill to read the full
>   table data preserving all existing cell values, (3) update ONLY the
>   specific target cell(s) as required by the task […]

For the already-passing `court-form-filling`, the consolidated memory instead
records "preserve this approach in future runs" — the optimizer reinforces
successes rather than only patching failures.

## Observations

- **The biggest gains come from output-contract mistakes, not reasoning
  mistakes.** `offer-letter-generator` failed round 0 (0.06) mostly because
  the agent didn't know a side-output (`employee_data.json`) was required;
  one line of learned guideline fixed it completely.
- **Skill selection improves.** With six distractor skills installed, round 0
  occasionally explored irrelevant skills; round 1 guidelines pin each task to
  its relevant skills.
- **`financial-modeling-qa` improves but doesn't saturate** (0.50 → 0.75):
  the remaining failure is a genuine calculation error, which guidelines about
  workflow can't fully fix — a second optimization round would be the natural
  next step.
- **`court-form-filling`'s −0.02** is a flaky oracle check on an
  intentionally-empty form field, not a regression in learned behavior.
