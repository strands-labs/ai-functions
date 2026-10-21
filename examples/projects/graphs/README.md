# Graphs: Verified Agent Workflows from Your Coding Assistant

Describe a task to your coding assistant, and the `aif-workflow` skill builds a
graph of agents to do it with AI Functions. This experimental example shows what
that gives you:

- **A skill for the assistant you already use** — Install `aif-workflow` into Claude
  Code, Codex, or Kiro CLI and describe a task in plain language. Your assistant
  turns it into a graph of agents and runs it. There is no new tool to learn and no
  orchestration code to write.
- **Built on AI Functions, saved as reusable graphs** — Each step is an agent thread
  on an AI Functions coordinator, and steps that don't depend on each other run in
  parallel. Every graph is saved as a small, readable Python package under `.aif/`.
  You can rerun it, edit it, or have the skill find and reuse it the next time a
  similar task comes up.
- **Any agent can run a step** — Steps run on Strands (the library's native
  `AIThread`), Kiro, Claude Code, or Codex, and one graph can mix them. Every thread
  follows the same protocol, so any other agent wrapped as a thread can be hosted the
  same way.
- **The writer and the runners are separate** — The assistant that writes the graph
  and the agents that execute its steps can be different. For example, Claude Code
  can write a graph whose steps run on Kiro.
- **Self-contained packages** — Each package carries its own copy of the runtime. It
  needs only the AI Functions package and the chosen backend to run, and later
  changes to this example don't change how it behaves.
- **Checked, not trusted** — Each step declares pre-conditions (is it ready to run?)
  and post-conditions (did it do the work?). At least one check per step is plain
  code that compares the result against values computed from the inputs. When a
  check fails, its message goes back to the agent as the next instruction, until the
  step passes or uses up its attempts.
- **Every run is recorded and inspectable** — All prompts, tool calls, and condition
  results are written to disk as the run happens. A read-only terminal viewer shows
  the graph, each check and its source code, and each attempt's tool calls, both
  while the run is live and after it finishes.

## What a Generated Graph Looks Like

The [Quick Start](#quick-start) asks for a regional sales report from a ten-row CSV
and leaves the shape of the graph to the agent. For that task, one run built this
13-step graph; yours may be split differently. Each region's rows are split out,
then summarised and checked in parallel, and the results join into a report, an
audit, and a final summary:

```text
ingest ─▶ north ─┬▶ north_stats ─┐
       ├▶ south ─┼▶ south_stats ─┼▶ report ─┐
       └▶ east ──┼▶ east_stats  ─┘          ├▶ publish
                 ├▶ north_checks ┐          │
                 ├▶ south_checks ┼▶ audit ──┘
                 └▶ east_checks ─┘
```

The generated `workflow.py` only wires the steps together. Prompts live in
`prompt.md`, and checks live in `conditions.py`. An abridged excerpt:

```python
steps = [
    Step(
        name="ingest",
        prompt=prompt_section("ingest"),
        pre_conditions=(c.file_present(c.DATA),),
        post_conditions=((c.rows_match_csv, "code"), (c.rows_have_revenue, "code"), unchanged),
        backend="kiro",
    ),
]
for region in c.REGIONS:
    ...  # <region>, then <region>_stats on kiro and <region>_checks on claude
steps += [
    Step(
        name="report",
        prompt=prompt_section("report"),
        post_conditions=((c.report_lines_correct, "code"), (c.report_total_correct, "code"), unchanged),
        depends_on=tuple(f"{region}_stats" for region in c.REGIONS),
        backend="kiro",
    ),
    ...  # audit, then publish
]
await run_workflow(task=task, objective=objective, steps=steps, workspace=c.WORKSPACE)
```

A condition reads the workspace rather than trusting the agent's reply. Its expected
values come from `region_facts()`, a helper in the same file that computes each
region's figures from `data.csv` itself, never from another step's output:

```python
def report_lines_correct(result: str) -> PostConditionResult | None:
    """REPORT.md has a line per region with its revenue, units and top product."""
    del result
    lines = checks.read_text(REPORT).lower().splitlines()
    for region in REGIONS:
        facts = region_facts(region)
        parts = [region, checks.currency(facts.revenue), f"{facts.units} units", facts.top_product]
        if not any(all(part.lower() in line for part in parts) for line in lines):
            return PostConditionResult(
                passed=False, message=f"REPORT.md needs one line for {region} containing all of {parts}."
            )
    return None
```

`run_workflow` comes from the copied `runtime.py`. It hosts each step on an
`InMemoryCoordinator` and `LocalWorker`, runs steps whose dependencies have passed
in parallel, and checks pre-conditions before spawning each one. It records every
event, including each post-condition evaluation, through one `coordinator.on(...)`
subscriber.

Here is the viewer after a real run of this graph. It took about 100 seconds, with
Claude Code running the three `_checks` steps and Kiro running the rest. Recorded
runs are listed on the left, and the graph is in the middle, with each node's
backend and checks. On the right are the selected node's prompt, its condition
tickets, and the source of the selected check:

![The graphs viewer after a real regional-report run: the run's 13 threads on the left; in the middle, ingest fans out to north, south and east, each splits into stats on kiro and checks on claude, which join into report and audit, then publish, with every condition passing; on the right, the report node's prompt, condition tickets, and the source of report_lines_correct.](assets/viewer.png)

## Quick Start

These steps add the skill to a project of your own, here called `~/my-project`, and
run a first graph there. You need Python 3.12 or later,
[uv](https://docs.astral.sh/uv/getting-started/installation/), git, and one of the
coding assistants in step 4.

### 1. Set up the agents that run the steps

Each step of a graph runs on an agent, called its **backend**. Choose one or more
backends, and make sure each one is installed and signed in on this machine,
following its own guide:

| Backend | Runs each step on | What you need | Check it | Package extra |
|---|---|---|---|---|
| `kiro` | [Kiro CLI](https://kiro.dev), via `KiroAgent` | `kiro-cli` installed, on your `PATH`, and signed in | `kiro-cli doctor` | `kiro` |
| `claude` | [Claude Code](https://docs.anthropic.com/en/docs/claude-code), via `ClaudeAgent` | A Claude Code sign-in or an `ANTHROPIC_API_KEY`; the package extra includes the Claude Code runtime | `claude --version` | `claude-code` |
| `codex` | [Codex](https://developers.openai.com/codex), via `CodexAgent` | A Codex sign-in or an `OPENAI_API_KEY`; the package extra includes the Codex runtime | `codex --version` | `codex` |
| `ai_thread` | AI Functions' own `AIThread` (Strands), with file and shell tools | Credentials for a [model provider](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/); Amazon Bedrock by default | `aws sts get-caller-identity` | none; add the `strands-agents-tools` package |

Note the backends you choose: steps 2 and 5 refer to them.

### 2. Add AI Functions to your project

Generated graphs run in your project's Python environment, so add the AI Functions
package there, with the line for each backend you chose in step 1:

```bash
cd ~/my-project
uv init --bare --python 3.12                    # only if there is no pyproject.toml yet

uv add 'strands-ai-functions[kiro]'             # kiro
uv add 'strands-ai-functions[claude-code]'      # claude
uv add 'strands-ai-functions[codex]'            # codex
uv add strands-ai-functions strands-agents-tools   # ai_thread
```

Running more than one of these adds each backend's extra alongside the others. If
you'd rather not add a `pyproject.toml` to your project, skip this step: the skill
then runs each graph with `uv run --with 'strands-ai-functions[…]'`, which uses a
temporary environment and leaves your project unchanged.

### 3. Get the example

The skill and the viewer come from this example, so clone the repository once,
anywhere you like:

```bash
git clone https://github.com/strands-labs/ai-functions.git ~/ai-functions
export GRAPHS=~/ai-functions/examples/projects/graphs   # used by the commands below
```

Commands that use `--project "$GRAPHS"` run the example's own tools in the example's
own environment, which uv creates the first time. They don't change your project's
environment.

### 4. Add the skill to your coding assistant

From your project, run the line for the assistant you use:

```bash
cd ~/my-project
uv run --project "$GRAPHS" python "$GRAPHS/graphs.py" skills install aif-workflow --claude   # Claude Code
uv run --project "$GRAPHS" python "$GRAPHS/graphs.py" skills install aif-workflow --codex    # Codex
uv run --project "$GRAPHS" python "$GRAPHS/graphs.py" skills install aif-workflow --kiro     # Kiro CLI
```

| Assistant | The skill is copied to | Then start the assistant with |
|---|---|---|
| Claude Code | `.claude/skills/aif-workflow/` | `claude` |
| Codex | `.agents/skills/aif-workflow/` | `codex` |
| Kiro CLI | `.kiro/skills/aif-workflow/` | `kiro-cli chat --agent ai-functions` |

The skill holds the assistant's instructions and the runtime that every generated
graph copies. Installing it changes nothing outside `~/my-project`. For Kiro, the
installer also creates `.kiro/agents/ai-functions.json` to load the skill, which is
why Kiro starts with `--agent ai-functions`.

The assistant you chat with is the one that **writes** the graph. It doesn't need to
be one of the backends that **run** it.

### 5. Ask for a graph

Copy the bundled sample data into a folder the agents can work in:

```bash
mkdir -p data/regional-report
cp "$GRAPHS/aif-workflow/examples/regional-report/data.csv" data/regional-report/
```

Then start your assistant here, with the command from the table in step 4.

Paste this request, replacing `[backend]` with a backend from step 1. It says what
to produce and check, and leaves the graph to the agent. To mix backends, name one
per kind of work instead, for example "Run the data checks on codex and everything
else on ai_thread."

> Use aif_workflow to build and run a graph named regional-report over
> data/regional-report/data.csv (sales rows; revenue is units × unit_price),
> keeping data.csv unchanged. For each region, work out its rows, units, revenue and
> top product, and check its data: date range, all units positive, and one price
> per product. Then write REPORT.md with the per-region figures and the total
> revenue, AUDIT.md with the data checks, and summary.json with total_revenue,
> per-region revenue and checks_passed. Run every step on [backend], and let
> independent work run in parallel.

The assistant writes the graph to `~/my-project/.aif/regional-report/` and runs it.
It may ask permission to write files or run commands. The agents running the steps
work unattended and edit files without asking, so only name a workspace they're
allowed to change. The assistant decides how many steps to use. The run in the
screenshot above used 13 agent sessions, up to six at a time, and took a few
minutes.

When it finishes, `data/regional-report/` holds every step's output. Compare
`REPORT.md` and `summary.json` with these figures; all checks should pass:

| Region | Rows | Units | Revenue | Top product | Dates |
|---|---|---|---|---|---|
| north | 4 | 26 | $1,167.29 | gizmo | 2026-01-04 to 2026-01-22 |
| south | 3 | 23 | $353.18 | gizmo | 2026-01-05 to 2026-01-19 |
| east | 3 | 67 | $709.73 | widget | 2026-01-08 to 2026-01-28 |
| **Total** | **10** | **116** | **$2,230.20** | | |

### 6. Watch the run

In a second terminal, from `~/my-project`:

```bash
export GRAPHS=~/ai-functions/examples/projects/graphs
uv run --project "$GRAPHS" python "$GRAPHS/graphs.py" view
```

The viewer shows every graph under `./.aif` and refreshes about every 1.5 seconds.
Select a node to see its prompt, its condition tickets and their source, and its
tool calls grouped by attempt. Use the arrow keys to move, `Tab` to switch panes,
`r` to rescan, and `q` to quit. The viewer only reads recordings: it doesn't start,
pause, or stop runs.

### 7. Run it again

Ask your assistant, in this conversation or a new one:

> Use aif_workflow to run the regional report again in data/regional-report.
> Discover and reuse a compatible saved workflow, keep the input unchanged, and
> report the new run directory.

The skill looks through the graphs saved under `.aif/` and reuses one only when its
prompts, checks, workspace, and backend fit the request. If none fits, it generates
a new graph and leaves the old one unchanged. A reused graph runs on fresh agents
and its checks run again; an earlier pass doesn't skip them. You can also rerun a
saved graph yourself, from `~/my-project`:

```bash
uv run python .aif/regional-report/workflow.py
# or, if you skipped step 2 (list your backends' extras, e.g. [kiro,codex]):
uv run --with 'strands-ai-functions[kiro]' python .aif/regional-report/workflow.py
```

Each run is recorded separately, and the viewer lists it with its run id, backend,
and session.

## Reference

### Generated files

```text
.aif/<slug>/
  prompt.md          one section per step
  conditions.py      checks derived from the task and the input data
  workflow.py        steps, dependencies, and the run_workflow() call
  runtime.py         copied orchestration and recording; never edited
  checks.py          copied condition helpers; never edited
  manifest.json      the latest run's graph, prompts, and condition source
  runs/<run-id>/
    manifest.json    this run's graph snapshot
    run.json         status, task, session, and timings
    events.jsonl     append-only event log
    result.json      each step's outcome
```

The per-run manifest means later edits to a graph don't change how earlier runs are
displayed. Snapshots keep graph metadata, not workspace files, so input and output
files may change on a rerun. Packages written by older runtimes don't have per-run
snapshots, but the viewer still reads them.

### Control

`Step` exposes `depends_on`, `pre_conditions`, `post_conditions`, `max_attempts`,
`backend`, and `model`. A step's own backend takes priority over the run's, and the
run's takes priority over `AIF_BACKEND`. `Step(model=...)` or `AIF_MODEL` selects the
model. AI judges can add to a step's code checks but cannot replace them.

### Updating the skill

Reinstalling leaves identical files alone and reports any files that differ;
review those before passing `--force`. Packages that were already generated keep
their own runtime copy, so reinstalling doesn't upgrade them.
