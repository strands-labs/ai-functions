# Learn and visualize skills with `@skillify`

This demo shows how an AI Function can build and use a library of reusable
procedures while solving ScienceWorld tasks. `@skillify` records the function's
calls, mines candidate procedures, evaluates their effect, and stores accepted
skills as `SKILL.md` files. The model receives a skill index and can load a
procedure's instructions through the `Skill` tool.

Use this example to try skill learning, inspect model trajectories and skill
source lines, and replay recorded model activity. For the reusable API, see
the [skills guide](../../docs/skills.md); for the shared viewer, see
[skills_viz](../skills_viz/README.md).

## Quick start

You need:

- Linux, macOS, or another Unix-like system; the skills library uses `fcntl` file locks.
- Python 3.12 or newer and `uv`.
- A JVM with `java` available on `PATH`; ScienceWorld includes its simulator JAR.
- Amazon Bedrock credentials with access to the configured models. See the
  [AI Functions setup guide](../../README.md#getting-started) for model-provider setup.

From the repository root, install the demo and check Java:

```bash
uv sync --package ai-functions-skillify-demo-example
java -version
cd examples/skillify_demo
```

Run the small preset and capture its skill-learning events:

```bash
uv run --project ../.. --package ai-functions-skillify-demo-example \
  visualize.py capture -- --smoke --out ./out-smoke
```

Then open the recording:

```bash
python3.12 visualize.py view --data ./out-smoke/inspection --port 0
```

Open the local address printed by the viewer. Keep that terminal running while
browsing; press Ctrl+C to stop it. The viewer can also follow the recording from
a second terminal while capture is running.

The smoke preset uses four training tasks, two evaluation tasks, and a 12-step
action budget. It checks the workflow but may produce no accepted skills. Use
the full configuration below to give the learner more evidence. Each capture
needs a fresh `--out` directory.

## Run learning and evaluation

The following commands run from `examples/skillify_demo/`.

Record the full workflow: 24 training calls, learning rounds, and evaluation on
15 held-out tasks with two runs per arm:

```bash
uv run --project ../.. --package ai-functions-skillify-demo-example \
  visualize.py capture -- --out ./out-skillify
```

Evaluate an existing library with four runs per arm, the evaluation setting used
in the results below:

```bash
uv run --project ../.. --package ai-functions-skillify-demo-example \
  visualize.py capture -- \
  --evaluate-only ./out-skillify/library --repeats 4 --out ./out-evaluation
```

`--evaluate-only` freezes the library's current version and measures it with and
without skills. It does not mine new candidates. `main.py` implements the demo
workflow; `visualize.py capture --` runs that workflow with recording enabled.
All demo options go after `--`.

| Option | Default | Purpose |
| --- | --- | --- |
| `--out` | `./out-skillify` | Directory for the library, reports, and recording |
| `--smoke` | off | Small preset; explicit options override its settings |
| `--evaluate-only LIBRARY` | — | Evaluate an existing skillify library |
| `--model` | Claude Haiku 4.5 | Model used by the player |
| `--step-limit` | 60 | Environment actions allowed per episode |
| `--traffic-tasks` / `--traffic-repeats` | 12 / 2 | Training tasks and calls per task |
| `--every` | 12 | New recorded calls between learning rounds |
| `--trial-inputs` / `--trial-repeats` | 4 / 2 | Recorded inputs and runs per arm for candidate trials |
| `--min-gain` | 0.02 | Minimum completion-rate gain for admission |
| `--budget-calls` | 200 | Maximum additional replay calls per learning round |
| `--eval-tasks` / `--repeats` | 15 / 2 | Evaluation tasks and runs per arm |
| `--concurrency` | 10 | Concurrency limit for calls, replays, and evaluation |

Reduce `--concurrency` if the simulator uses too much memory or the model provider
throttles requests. Training calls and background replays can overlap, so the
number of open environments can exceed this limit. `--model` configures the
player and its replays; the discovery and quality functions in `full_pipeline/`
use their own AI Function model configurations. For all options, run:

```bash
uv run --project ../.. --package ai-functions-skillify-demo-example main.py --help
```

## Add skill learning to your own AI Function

The reusable API is `ai_functions.experimental.skills`. It is installed with
this demo. To use the repository's library in another Python environment,
install the checkout there with `python -m pip install -e .` from the repository
root. The import name is `ai_functions`; the distribution name is
`strands-ai-functions`.

Given an existing `@ai_function` named `solve` and a collection of `tasks`, wrap
it once, then call it normally from your async application:

```python
from ai_functions.experimental.skills import skillify

skilled_solve = skillify(
    library="./skills/solve",
    replay=True,
    every=20,
)(solve)

for task in tasks:
    result = await skilled_solve(task=task)

report = await skilled_solve.refresh_skills()
print(report.status, list(report.admitted))
print([skill.name for skill in skilled_solve.skills])
```

You can also stack `@skillify(...)` directly above `@ai_function(...)`.
`library` holds the versioned procedures and recorded calls. `every` controls
background learning; `refresh_skills()` waits for any active round and requests
a round immediately. `history()` returns the admission and rejection reports.

Supply meaningful post-conditions on the AI Function or a
`score(arguments, result)` callback returning a value in `[0, 1]`. Without a
score callback, successful returns count as success and exhausted
post-conditions count as failure. Mining needs at least three scored calls and
evidence of failures or score variation; a round can therefore be declined.

`replay=True` confirms that the function's tools can safely be run again during
candidate evaluation. For stateful tasks, pass an `environment=` factory that
creates a fresh episode from the call arguments. The ScienceWorld factory in
[scienceworld_env.py](scienceworld_env.py) demonstrates this pattern.

The default components are `PromptMiner` and `CompletionGate`. You can supply
your own `miner=` and `gate=` through the same API. This demo uses
`FullDiscoveryMiner` and `PrefilteredGate(CompletionGate(...))`; the reported
results use those components. See the [skills guide](../../docs/skills.md) for
scoring, environment, miner, gate, and library-management interfaces.

## Inspect skills and recorded runs

Open a captured run to browse skill text, procedures, line numbers, candidate
scores, library versions, and model/tool trajectories:

```bash
python3.12 visualize.py view --data ./out-skillify/inspection --port 0
```

A successful `Skill` tool response confirms delivery of an exact source revision.
Action-to-instruction highlights are labeled **inferred**: matching text suggests
a connection, but does not prove that the model followed a particular line.

To inspect a written Markdown file without running a model:

```bash
python3.12 visualize.py view \
  --data ./out-document \
  --import-skill /path/to/SKILL.md \
  --port 0
```

The input may be any `.md` file or a directory containing `SKILL.md`. Use a fresh
import destination. Document mode shows source and metadata; execution and
learning history require a captured run.

Viewing uses saved model responses and tool results. It needs no model
credentials or running ScienceWorld environment. See the
[shared viewer guide](../skills_viz/README.md) for setup without demo
dependencies, portable exports, and custom capture.

## How the demo learns

1. Calls run with the current library and record their arguments, trajectories,
   and outcomes. An empty library adds no skill index or `Skill` tool.
2. A background round examines the scored calls. `FullDiscoveryMiner` builds
   episodes, extracts procedures, groups related drafts, merges them, and
   formats candidate `SKILL.md` files.
3. `PrefilteredGate` checks completeness, distinctiveness, and formatting.
   Candidates that pass go to `CompletionGate`, which compares task completion
   with and without each candidate against the current library.
4. Candidates meeting `--min-gain` are published in a new library version.
   Rejected candidates retain their scores and reasons in the round report.
5. The final version is evaluated on held-out tasks under both arms.

Each training call and gate replay uses a fresh ScienceWorld session. The
factory verifies the opening observation against the recorded input; a mismatch
raises `ReplayDriftError`. During execution, the model sees the skill index and
decides which full procedures to load. Reports distinguish skills that were
available from skills that were actually loaded.

## Tasks and results

The task family is `measure-melting-point-unknown-substance`. The default model
is Claude Haiku 4.5 on Bedrock, identified by
`global.anthropic.claude-haiku-4-5-20251001-v1:0`, with a **60-step action budget**.
The fixed variations target tasks with room for improvement under that model
and budget. Their constants are in [evaluation.py](evaluation.py).

| Split | Variation IDs |
| --- | --- |
| Training, 12 tasks | 9, 11, 82, 39, 12, 119, 93, 87, 134, 68, 50, 103 |
| Held-out evaluation, 15 tasks | 283, 243, 254, 249, 275, 296, 255, 225, 260, 299, 272, 265, 229, 233, 298 |

The sets use ScienceWorld's disjoint train and test splits. Task-count flags take
a prefix of these lists. Changing the model, action budget, or task selection
changes the evaluation conditions.

The reported learning results use 24 training calls:

| Round | Calls in window | Task success | Mined | Admitted | Library |
| --- | --- | --- | --- | --- | --- |
| 1 | 12 | 16.7% | 23 | 13 | v1 |
| 2 | 12 | 58.3% | 16 | 3 | v2, 16 skills |

Of the 23 rejected candidates, 12 reduced completion in gate replays and 11
produced no improvement. The gate evaluates training calls; its gains are
separate from the held-out results below.

The final library was evaluated on 15 held-out tasks with four runs per arm:

| Metric | With skills | Without skills | Difference | One-sided p |
| --- | --- | --- | --- | --- |
| **Task success rate** | **81.7%** | 46.7% | **+35.0 points** | **<0.001** |
| Mean score | 81.7 | 63.6 | +18.1 | 0.007 |
| Environment steps | 28.0 | 43.9 | −15.9 (−36%) | <0.001 |

| Outcome, 60 runs per arm | Without skills | With skills |
| --- | --- | --- |
| Finished the task | 28 | **49** |
| Did the work but never answered | 13 | **0** |
| Answered wrongly | 19 | **11** |

Ten tasks improved, two regressed, and three tied. All 60 with-skill executions
loaded at least one procedure. The most frequently loaded skills were
`determine-melting-point-stove-thermometer` (57),
`examine-after-focus-for-substance-properties` (56), and
`batch-time-advance-with-stopwatch` (23).

Task success means a score of 100. Mean score clamps negative scores to zero.
Significance uses a permutation test that shuffles arm labels within each task.
These results describe the stated task set, model, budget, and discovery
pipeline; generated procedures and outcomes can vary between executions.

## Outputs

All output paths are relative to `--out`.

| Path | Contents |
| --- | --- |
| `library/` | Published skill versions, call records, and round history |
| `library/versions/<version>/<skill>/SKILL.md` | A stored procedure's source |
| `skillify_summary.json` | Configuration, round reports, final library, and evaluation summary |
| `eval_results.json` | Per-execution scores, success, steps, tokens, and available/loaded skills |
| `analysis.json` | Aggregate comparisons and skill-use statistics |
| `inspection/skills.sqlite3` | Source snapshots and lifecycle events from `visualize.py capture` |

## Code guide

| File | Responsibility |
| --- | --- |
| [main.py](main.py) | Decorated player, training calls, learning rounds, evaluation, and CLI |
| [visualize.py](visualize.py) | Capture and viewer entry points |
| [scienceworld_env.py](scienceworld_env.py) | Simulator tools and fresh-episode factory |
| [evaluation.py](evaluation.py) | Tasks, player prompt, evaluation driver, and result analysis |
| [full_pipeline/](full_pipeline/) | Custom discovery, formatting, and quality filtering |
