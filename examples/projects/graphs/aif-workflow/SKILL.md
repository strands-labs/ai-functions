---
name: aif-workflow
description: Discover, reuse, or dynamically create AI-Functions graphs of agent threads with checked pre- and post-conditions and recorded execution. Use when the user says "aif_workflow", "aif workflow", "run this as a Kiro workflow", "run this as an AI workflow", or asks for a graph of agents with conditions, observability, and control over execution. Reuses compatible packages under .aif/ or generates a new one, then executes it and records each run.
---

# aif_workflow

Turn a task into a **generated graph of agent threads using AI-Functions**.
Pre- and post-conditions check the work, and recorded events make each step
observable. Prompts, dependencies, conditions, and retry limits control execution.
Each workflow lives under `.aif/` and carries its own runtime.
Discover compatible workflows before generating a new graph for a repeated task.

This skill lives in the repository's `examples/projects/graphs` example. From that
folder, install it with
`uv run python graphs.py skills install aif-workflow --claude|--kiro|--codex --project /path/to/project`.
Read the project's `CLAUDE.md` or `AGENTS.md` first if it has one. The library's own
documentation is the source of truth for event field names: grep the installed package
(`python -c "import ai_functions, pathlib; print(pathlib.Path(ai_functions.__file__).parent)"`),
and read `types/events.py` there before assuming a field name.

## What you produce

```
.aif/<slug>/
├── prompt.md         one `## <step_name>` section per step, holding its prompt
├── conditions.py     the pre- and post-condition callables — no orchestration
├── workflow.py       declares Steps and calls run_workflow — no condition logic
├── runtime.py        COPIED VERBATIM, never written by hand or edited
├── checks.py         COPIED VERBATIM — condition plumbing you import, never re-derive
├── manifest.json     written by the runtime on each run
└── runs/<run-id>/    manifest.json, run.json, events.jsonl, result.json — per run
```

Only two of these are yours to write: `prompt.md` and `conditions.py`. `runtime.py` and
`checks.py` are copied; `workflow.py` starts as a copied template where you edit one
marked block.

Then you **run it** and report the run id.

## Procedure

### 0. Discover and reuse

Look in the target project's `.aif/` directory, even in a new conversation.
If the user names a workflow, inspect that package first. Otherwise list
`.aif/*/workflow.py` and read package manifests to shortlist candidates by
objective, workspace, inputs, expected outputs, and backend.

Read the shortlisted packages' `workflow.py`, `prompt.md`, and `conditions.py`
as text. These files define what a rerun will do; the package manifest describes
the last run and may be out of date. Packages without a manifest can still be
candidates: inspect their source. Do not import or execute candidate code or
conditions during discovery.

Reuse a package when its current prompts and checks satisfy the request in
the same workspace. Resolve relative paths against the intended project and
check input paths, schemas, hardcoded expected values, dependencies, and effective
backend/model choices, including environment defaults. Changed input contents
are compatible only when the workflow and its checks support them. A similar
name or a previous passing run alone is insufficient.

Read recent `runs/*/run.json` and, when relevant, `result.json` for execution
evidence. A failed run can be retried if its cause is understood and the graph
still fits. Do not start overlapping runs that write the same files; check
whether a run marked `running` is still active before proceeding.

- **One compatible candidate:** report its path and why it fits, then go
  directly to **5. Run it** using the existing script and required environment.
- **Several compatible candidates:** respect an explicit user selection;
  otherwise ask which to use when their behavior differs.
- **No compatible candidate:** continue to **1. Decompose**. Use a new slug
  when an existing graph needs changed prompts, checks, paths, or configuration.
  If a specifically requested workflow is incompatible, explain the mismatch
  before substituting another.

Reuse leaves package source files unchanged. Do not copy templates over them,
skip conditions because an earlier run passed, or treat discovery as permission
for work outside the user's request. Every execution creates fresh threads,
checks conditions again, and adds a run directory to the same workflow's history.
Report whether the graph was reused or generated along with the new run id.

Older copied runtimes remain usable, but versions before 4 lack per-run graph
snapshots. Report that limitation when relevant; do not silently replace their
runtime during reuse. Historical snapshots contain graph metadata and condition
source, not copies of workspace inputs or outputs.

### 1. Decompose

Restate the task as a one-paragraph `objective` that says what *success* means in
checkable terms. Pick a short kebab-case `slug`.

Make one step per separately *verifiable* piece of work: a step earns its place by
producing something its own conditions can check, not by being a separate
instruction. A step is one agent thread with one prompt and its own conditions.
Work that does not depend on other work becomes its own step, so it runs in
parallel. Do not add steps that only pass data along. Most tasks need 1–6 steps;
split further only when each part has its own checkable output. When the user names
the steps or their dependencies, build exactly that graph, however many steps it
has, and still give every step its own code-checkable post-condition.

For each step decide: its prompt, its pre-conditions, its post-conditions, and
which steps it depends on. Steps with no unmet dependencies run concurrently, so
declare `depends_on` only where a real ordering or data need exists.

### 2. Copy the scaffolding — do not write it

```bash
S=<the runtime/ folder beside this SKILL.md>     # e.g. .claude/skills/aif-workflow/runtime
mkdir -p .aif/<slug>
cp $S/runtime.py .aif/<slug>/runtime.py                      # verbatim, never edit
cp $S/checks.py .aif/<slug>/checks.py                        # verbatim, never edit
cp $S/workflow_template.py .aif/<slug>/workflow.py           # edit one marked block
cp $S/conditions_template.py .aif/<slug>/conditions.py       # your starting point
```

`runtime.py` and `checks.py` go in byte for byte. Never edit them, never re-derive
them, never write your own orchestration or your own subprocess handling.

The runtime already handles what fails silently when rewritten: terminating agent
threads, recording every event, routing condition events to the right
thread, and awaiting async conditions (a sync wrapper makes every AI judge pass).

`checks.py` covers the condition plumbing that has produced real bugs when hand-rolled
— a cached file read grading attempt 2 against attempt 1's content, an `==` on floats
failing a correct answer by a fraction of a cent. **Import it; do not reimplement it:**

```python
import checks

checks.must_exit_zero(WORKSPACE / "stats.py")   # runs it, stderr in the message
checks.read_json(path) / checks.read_csv(path)  # defensive, never cached
checks.close(value, wanted)                     # float compare within a cent
checks.currency(1234.5)                         # "$1,234.50"
checks.file_unchanged(TESTS)                    # negative condition, digest taken now
checks.missing_strings(REPORT, [wanted])        # which strings are absent
checks.undocumented_functions(path)             # AST walks: docstrings, hints, length
```

If the scaffolding genuinely cannot express what a task needs, stop and say so instead
of working around it.

### 3. Write the conditions — the part that matters

**Point `WORKSPACE` at the directory the user named.** The template defaults to
`data/<slug>/`, which is almost certainly not what they meant — a wrong workspace shows up
as a pre-condition blocking the step with `<path>/<file> is missing`, and no agent session
runs at all. If the user did not name a directory, ask before generating anything.

**At least one post-condition per step must be checkable by code.** A judge alone
is not verification. Prefer, in order: a file exists and parses; a command exits
zero; a value is in range; a required substring is present. Reach for an
`@ai_function` judge only for genuinely qualitative properties, and pair it with a
code check.

A condition takes the result **string** and returns `PostConditionResult`, `None`
to pass, or raises.

**Everything about a condition is displayed, so write it for a reader.** The runtime
introspects each callable and records three things in `manifest.json`, so a reader can
see what was checked without opening the code:

| What it shows | Where it comes from | So |
|---|---|---|
| the ticket label and one-line summary | the **first docstring line** | make it a statement of what must be true, present tense: `"stats.py exists and exits zero."` |
| the signature | **type hints** on the parameters and return | annotate them; `(result: str) -> PostConditionResult \| None` tells the reader what it consumes |
| the full body | `inspect.getsource` | it will be read, so keep it short and obvious — no cleverness |

Type hints and a docstring are therefore not optional decoration here; they are the
condition's user interface. The same goes for the step prompts, which are recorded
alongside them.

The failure `message` is fed back to the agent verbatim as its next turn, so write
messages as *instructions for the fix*:

```python
# Good: the agent can act on this.
return PostConditionResult(passed=False, message=f"{PATH} has no `--json` flag; add it to the argparse setup.")
# Useless: nothing to act on.
return PostConditionResult(passed=False, message="wrong")
```

Read files and run commands from the condition — the result string is the agent's
*claim*, and the point is to check the world instead of trusting it. One assertion
per condition, so a failure names one fix.

**Use `checks` for plumbing, but keep the comparison in your own body.** The condition's
source is recorded for a human to audit, so hiding the check behind a helper defeats it:

```python
# ✅ plumbing borrowed, assertion visible
data = checks.read_json(METRICS) or {}
if not checks.close(data.get("total_revenue"), expected_total()):
    return PostConditionResult(passed=False, message=f"total_revenue is {data.get('total_revenue')!r}; ...")

# ❌ the source panel now shows a call, not the check
return checks.json_field_equals(METRICS, "total_revenue", expected_total())
```

**Conditions that look like verification but are not.** Every one of these passes
while telling the user nothing; do not write them:

| Not verification | Why | Instead |
|---|---|---|
| `assert result` / `len(result) > 50` | checks that the agent said *something* | check the artifact it claims to have produced |
| `"success" in result.lower()` | the agent grades itself | run the thing; check the exit code or the output |
| `REPORT.md exists and is non-empty` *as the only check* | any file passes | check its *content* against something computed independently |
| a judge asking "is this good?" with no code check beside it | unfalsifiable | pin one mechanical property too |
| recomputing the answer *from the agent's output* | circular | recompute from the source data |

**Where the expected value comes from matters more than the assertion.** Compute it
from the inputs — parse the CSV yourself, hash the file before the run, capture the
original function's output before the refactor — so a confident wrong answer fails.

**Negative conditions are often the important ones.** "the test file is unchanged",
"the public signature is identical", "no control id outside the list" — these are
what stop a task being satisfied by deleting the thing that was failing. Add one
whenever the cheap way out would technically pass.

**If the task admits no code-checkable condition, say so before generating
anything.** Ask the user for the missing piece: the command that proves it works, the
list of things to check, the value to compare against. A workflow whose conditions
cannot fail is theatre, and shipping one is worse than declining.

```python
"""Conditions for the <slug> workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ai_functions.ai_thread.postcondition import PostConditionResult

WORKSPACE = Path("<absolute path to the directory being worked on>")
TARGET = WORKSPACE / "<file the step must produce or change>"


def target_dir_exists() -> PostConditionResult | None:
    """The directory the step will edit is present."""
    if not TARGET.parent.is_dir():
        return PostConditionResult(passed=False, message=f"{TARGET.parent} does not exist")
    return None


def flag_written(result: str) -> PostConditionResult | None:
    """The target file exists and declares the --json flag."""
    del result  # Check the world, not the agent's claim about it.
    if not TARGET.is_file():
        return PostConditionResult(passed=False, message=f"{TARGET} was not created; write it.")
    if "--json" not in TARGET.read_text(encoding="utf-8"):
        return PostConditionResult(passed=False, message=f"{TARGET} has no `--json` flag; add it.")
    return None


def tests_pass(result: str) -> PostConditionResult | None:
    """The test suite for the change exits zero."""
    del result
    proc = subprocess.run(["python", "-m", "pytest", "-q", str(TARGET)], capture_output=True, text=True)
    if proc.returncode != 0:
        return PostConditionResult(passed=False, message=f"pytest failed:\n{proc.stdout[-1500:]}")
    return None
```

### 4. Write `workflow.py` — wiring only

```python
"""Generated by the aif_workflow skill: <one line on what this workflow does>."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import conditions as c
from runtime import Step, prompt_section, run_workflow

WORKSPACE = Path("<absolute path the agent works in>")


async def main() -> None:
    """Run the workflow and log each step's outcome."""
    steps = [
        Step(
            name="implement",
            prompt=prompt_section("implement"),
            pre_conditions=(c.target_dir_exists,),
            post_conditions=((c.flag_written, "code"), (c.tests_pass, "code")),
            max_attempts=4,
        ),
    ]
    results = await run_workflow(
        task="<the user's request, verbatim>",
        objective="<what success means, in checkable terms>",
        steps=steps,
        workspace=WORKSPACE,
    )
    for name, outcome in results.items():
        logging.getLogger("aif").info("%s: %s", name, outcome.status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
```

`workflow.py` and `conditions.py` sit beside `runtime.py`, so plain
`import conditions` / `from runtime import ...` work when run as a script. Thread
ids are derived from step names by the runtime — do not invent them. Use
`{step_name}` in a prompt to interpolate an upstream step's result.

### 5. Run it

From the project root, run the workflow in the project's own environment:

```bash
uv run python .aif/<slug>/workflow.py
```

That environment needs `strands-ai-functions` with the extras for the backends the
graph uses: `kiro`, `claude-code` or `codex`, plus the `strands-agents-tools` package
for `ai_thread`. If the run fails with `ModuleNotFoundError` for `ai_functions` or a
backend package, do not add dependencies to the user's project without asking.
Run it in a temporary environment instead, and tell the user how to add the package
permanently (`uv add 'strands-ai-functions[<extras>]'`):

```bash
uv run --with 'strands-ai-functions[<extras>]' python .aif/<slug>/workflow.py
```

Replace `<extras>` with the extras the graph's backends need, comma-separated (for
example `kiro,codex`), and add `--with strands-agents-tools` for `ai_thread`. If the user gives a different run command, use theirs.

Report: the run id, the path of the run directory, each step's status, and — when a step
needed more than one attempt — which post-condition failed first and what the feedback
said. That retry story is the interesting part; do not bury it.

Then stop. Recording is done by the runtime; whatever reads those files afterwards is not
this skill's concern — do not start servers or other processes. The user watches runs
in their own terminal, from the project root, with the example's viewer:
`uv run --project <examples/projects/graphs> python <examples/projects/graphs>/graphs.py view`.

### 6. Backends — which agent runs each step

Every step's prompt runs on one of four agents. All take a prompt and return text, so the
generated files are identical whichever one is used; only what gets spawned differs.

| Backend | Agent | Needs |
|---|---|---|
| `kiro` (default) | `kiro-cli` over ACP, unattended | `strands-ai-functions[kiro]`, `kiro-cli` on `PATH` |
| `claude` | Claude Code via the Claude Agent SDK, `bypassPermissions` | `strands-ai-functions[claude-code]`, Claude Code auth |
| `codex` | OpenAI Codex via its SDK, `workspace_write` sandbox | `strands-ai-functions[codex]`, Codex auth |
| `ai_thread` | the library's own Strands agent with file and shell tools | `strands-agents-tools` (or `tools=` on the Step) and model credentials, Bedrock by default |

Pick one, most specific wins:

```python
Step(name="review", prompt=..., backend="claude")      # this step only — steps in one workflow may mix
run_workflow(..., backend="codex")                       # every step without its own
AIF_BACKEND=ai_thread uv run python .aif/<slug>/workflow.py   # the environment; no file changes
```

`Step(model=...)` or `AIF_MODEL` picks the model for any backend. If the user names an
agent ("do this with Codex"), set it on `run_workflow`; if they name it for one step,
set it on that `Step`. Otherwise leave it unset — the environment decides, and the run
records which backend ran each step in `manifest.json`.

Every backend runs **unattended**: Kiro with `auto_approve`, Claude Code with
`bypassPermissions`, Codex with a write sandbox, the AI thread with tool consent
bypassed. Rule 2 below applies to all four alike. Kiro reports no token usage; the
others do, and the viewer shows whichever it finds.

## Rules

1. **Copy `runtime.py`; never hand-roll orchestration.** No direct `kiro-cli`
   calls, no agent threads constructed by hand, no coordinator setup in generated code.
2. **Every backend edits files without asking.** Point `WORKSPACE` at a directory
   the user has agreed can be modified. If the task would have the agent edit
   `.claude/`, `.kiro/`, `.agents/`, or `.venv/`, stop and ask.
3. **No condition may mutate the workspace.** Conditions observe; steps change
   things. A condition with side effects makes retries incoherent.
4. **Keep generated code short and boring.** It is read aloud during demos. No
   cleverness, no helpers used once, no defensive layers.
5. **Never edit the canonical runtime from this skill.** If a workflow needs
   something it does not offer, say so and stop.
6. **Regenerating is fine, editing history is not.** Re-running adds a new run
   directory; never delete or rewrite an existing one.
7. **Reuse without overwriting.** A compatible existing package can be run from
   a new conversation. When generating a different graph, choose an unused slug
   such as `<slug>-2`; never overwrite an existing package or edit it while running.

## When a run fails

Report it plainly — a workflow that stops after `max_attempts` did its job: it
refused to return an unverified result. Say which condition held, what the agent
tried, and what you would change: the prompt, the condition's message, or the
attempt budget. Do not loosen a condition to make a run pass unless the user asks;
that converts verification into theatre.
