# Skill visualization and recording

Inspect reusable procedures stored as Markdown, or record how a `@skillify` AI
Function learns and uses them. The viewer shows exact skill source lines,
learning decisions, library versions, and model/tool trajectories. The same
recording can be replayed interactively or exported as a portable JSON archive.

A **skill** here is a procedure stored in a `SKILL.md` file, optionally with YAML
metadata and supporting files. `@skillify` wraps an AI Function to learn and
maintain a library of these procedures. See the
[skills API guide](../../docs/skills.md) for creating a skillified function, or
the [ScienceWorld demo](../skillify_demo/README.md) for a complete application.

## Requirements and setup

Use Python 3.12 or newer and a checkout of this repository. These tools live
under `examples/skills_viz/`; the commands below run from the **repository root**
and set `PYTHONPATH=examples` so Python can import them.

| Task | Requirements |
| --- | --- |
| View a Markdown file or saved recording | Python and a web browser |
| Record an AI Function | The `ai_functions` package, plus the model credentials and tools required by your application |

Use your application's Python environment for capture. To install the library
from this checkout into that environment:

```bash
python3.12 -m pip install -e .
```

Document viewing and recorded playback work without installing the agent
package. If viewing Markdown in a separate environment, install `pyyaml` for
full parsing of nested or multiline YAML metadata; the exact source remains
visible without it:

```bash
python3.12 -m pip install pyyaml
```

## Quick start: open a written skill

Point the viewer at a Markdown file:

```bash
PYTHONPATH=examples python3.12 -m skills_viz.viewer \
  --data ./out-document \
  --import-skill /path/to/my-skill.md \
  --port 0
```

Replace the input path with your file. The input may be any `.md` file, a
`SKILL.md`, or a directory containing `SKILL.md`. Frontmatter is optional. For
example, a skill file can contain:

```markdown
---
name: check-report
description: Check a report against its source data.
---

## Instructions
1. Check each reported total against the source data.
2. Identify discrepancies and cite the affected source rows.
3. Summarize the checks and any unresolved issues.
```

Open the local address printed by the command. `--port 0` selects an available
port; use an explicit port if you want a fixed address. Keep the terminal
running while browsing, and press Ctrl+C to stop the server.

Document mode opens the procedure as numbered blocks, with a source-line link
below each step. Nested instructions and conditions remain inside their parent
block. Switch to **Markdown source** to inspect the complete document, including
metadata and text outside the procedure. It reads the original file without
modifying it. Execution, loading, and learning history require a recording.

Imports require an empty destination. To reopen an imported document, omit the
import option:

```bash
PYTHONPATH=examples python3.12 -m skills_viz.viewer \
  --data ./out-document --port 0
```

An import is a snapshot. To inspect edits to the original file, import it into
a fresh `--data` directory.

## Record your own skillified function

The recorder works with any local `@skillify` AI Function. In your async
application, surround the calls and learning rounds you want to capture:

```python
from ai_functions.experimental.skills import observe_skills
from skills_viz.capture import Recorder

# solve is your existing @skillify function; tasks is your workload.
recorder = Recorder("./out-inspection")
with observe_skills(recorder):
    for task in tasks:
        result = await solve(task=task)
    report = await solve.refresh_skills()
```

Run that application with the shared tools on its import path:

```bash
PYTHONPATH=examples python3.12 /path/to/your_application.py
```

Replace `solve`, its arguments, and `tasks` with your application's equivalents.
The example belongs inside an async function or another context supporting
`await`. `refresh_skills()` waits for an active round and requests a round; keep
the observation context open until the work you want to record has finished.
A round may decline to mine or reject all candidates, so a recording does not
guarantee a nonempty skill library.

Capture writes `out-inspection/skills.sqlite3`. It preserves complete model
messages and tool results, skill source revisions, candidate decisions, and
library publications. Sources are copied before temporary candidate directories
or retained versions are removed. Multiple functions and libraries can share
one recorder.

Capture follows child async tasks in the same process. Distributed workers need
capture configured in their own execution contexts; callbacks are not
serialized or aggregated across processes. Model credentials and task tools are
needed for capture, but not for inspecting or exporting the saved recording.

## Browse a recording

```bash
PYTHONPATH=examples python3.12 -m skills_viz.viewer \
  --data ./out-inspection --port 0
```

The main controls are:

- **Procedures:** search skills and filter by library, version, or decision status.
- **Procedure blocks / Markdown source:** read one block per step, or inspect
  the complete source. Each block links to its original line range.
  **Evidence & metadata** shows scores, reasons, provenance, resources, and
  confirmed loads.
- **Model uses skills:** select an execution and inspect its recorded messages,
  tool arguments, and results. Load links open the source revision delivered.
- **Learning log:** inspect induction, filtering, and publication events.
- **Replay / seek:** move through recorded milestones or select a recording
  position. Replay advances through lifecycle events and model/tool activity,
  with two seconds per milestone at 1× speed. **Follow live** displays new events
  as capture appends them.

Only events and source revisions available at the selected recording position
are shown. Candidate text remains inspectable after rejection. When you use a
custom miner or gate, its overall boundaries are captured automatically; expose
its internal stages as described below.

### What the evidence means

A **confirmed load** requires a successful `Skill` tool result matching the
captured document body. It confirms delivery of that source revision to the
agent.

Blue load indicators identify document delivery. They do not indicate that
individual steps were applied. An amber block labeled **Possible step use ·
inferred** identifies a possible association with a recorded action. The
procedure panel and action panel use the same block number, text, line range,
and color for that association.

An **inferred** action-to-instruction link means that a later tool action shares
words with an instruction block in a loaded skill. The highlight is a navigation
aid, not proof of the model's internal reasoning, execution of a Markdown line,
or a causal improvement from that action.

Gray panels labeled **Environment output** show observations from recorded tool
execution. They describe the task environment and do not identify skill steps.

Line numbers refer to the complete captured file, including YAML frontmatter.
Draft blocks use line numbers in the draft body and are labeled **Draft**;
formatted skills use line numbers in the captured Markdown file.
Document inspection shows source only. A recording can show execution and
learning evidence only when those events were captured.

## Share recordings and inspect saved libraries

Click **Export recording** in the viewer to download a self-contained JSON
archive. To open that file on another machine:

```bash
PYTHONPATH=examples python3.12 -m skills_viz.viewer \
  --data ./out-imported \
  --import-json /path/to/skills-recording.json \
  --port 0
```

The export includes captured skill text and events, so the receiving machine
needs neither the original source paths nor model credentials.

You can also import a skillify library directory containing `versions/`,
`records/`, and round history:

```bash
PYTHONPATH=examples python3.12 -m skills_viz.viewer \
  --data ./out-library-inspection \
  --import-library /path/to/skillify-library \
  --port 0
```

Library import recovers retained skill versions, completed calls, and round
reports. Discarded source revisions or unrecorded intermediate stages cannot be
reconstructed; the viewer identifies this limitation. It does not unpickle saved
arguments or re-execute calls. Each import uses an empty destination, and import
options are mutually exclusive.

Archives include prompts, tool results, and skill text. Review those contents
before sharing them. The viewer binds to `127.0.0.1`, serves fixed read-only
routes, and uses no remote scripts or fonts.

## Expose custom pipeline stages

Custom miners and gates can add stage details through the same observation API:

```python
from ai_functions.experimental.skills import skill_event

skill_event("stage.started", stage="extract")
drafts = await extract(...)
skill_event("stage.finished", stage="extract", drafts=drafts)
```

Use this inside your async pipeline with its own `extract` implementation. The
surrounding function, library, run, and round labels are attached automatically.
`observe_skills(function="name", ...)` can add context labels while inheriting
the current recorder. Observer callbacks receive copied, JSON-compatible data;
exceptions are logged without interrupting the task. Callbacks run synchronously
and should return promptly.

Custom evaluation drivers can use `trace_observed` from
`ai_functions.experimental.skills.observation` within a labeled run scope to
capture live and failed executions. The
[ScienceWorld evaluation driver](../skillify_demo/evaluation.py) is an example
of recording run boundaries, source availability, outcomes, and traces.

## Outputs and code guide

| Path | Contents |
| --- | --- |
| `<data>/skills.sqlite3` | Append-only observation journal with source snapshots |
| `skills-recording.json` | Portable archive downloaded from the viewer |

| File | Responsibility |
| --- | --- |
| [capture.py](capture.py) | Recorder, imports/exports, source maps, and replay state |
| [viewer.py](viewer.py) / [index.html](index.html) | Local web server and interactive UI |
| [Core observation API](../../src/ai_functions/experimental/skills/observation.py) | Capture scope and lifecycle callbacks used by `@skillify` |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named skills_viz` | Run from the repository root with `PYTHONPATH=examples`, or add the absolute `examples/` path to your application's import path. |
| Empty recording or missing rounds | Check the `--data` path and keep the observation context open until the calls and rounds finish. |
| No accepted skills | Inspect the learning log: a round may decline to mine, reject candidates, or exhaust its replay budget. |
| Import requires an empty archive | Choose a new `--data` directory; omit the import flag when reopening an existing archive. |

For the full command-line options, use `python3.12 -m skills_viz.viewer --help`
with `PYTHONPATH=examples` set.
