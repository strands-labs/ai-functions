# SkillsBench Improvement Loop

A self-improving agent demo: a `ClaudeAgent` solves real
[SkillsBench](https://github.com/benchflow-ai/skillsbench) tasks, gets graded
by each task's pytest oracle, and the `TextGradOptimizer` turns oracle
feedback into textual gradients that update a persistent `JSONMemoryBackend`.
A second round then runs with the learned memory — task scores improve without
any change to prompts or code.

## How it works

```
SkillsBenchSource (5 tasks, 10 skills: 4 relevant + 6 distractors)
      │
      ▼
ClaudeAgent  (one thread per task, run in parallel)
      │
      ▼
Oracle  (pytest test_outputs.py against sandbox output files)
      │
      ▼
TextGradOptimizer.backward()    (one call per failing task)
TextGradOptimizer.consolidate() (merge gradients → memory update)
      │
      ▼
JSONMemoryBackend  (workflow_guidelines updated for round 1)
```

The mental model is PyTorch autograd:

- `recall()` ≈ reading a learnable weight into the forward pass
- `backward()` ≈ `loss.backward()` — accumulates textual gradients
- `consolidate()` ≈ `optimizer.step()` — writes improvements to memory

Because tasks run through `ClaudeAgent` rather than an `@ai_function`, the
demo builds a small synthetic computation graph per task (a `ParameterNode`
for the memory field plus a `ThreadNode`) so the optimizer has a graph to
backpropagate through.

### Tasks and skills

| Task | Relevant skills |
|------|-----------------|
| `exceltable-in-ppt` | `pptx`, `xlsx` |
| `financial-modeling-qa` | `pdf`, `xlsx` |
| `offer-letter-generator` | `docx` |
| `pptx-reference-formatting` | `pptx` |
| `court-form-filling` | `pdf` |

Six distractor skills (`pdf-editing`, `text-parser`, `image-ocr`, `marker`,
`fuzzy-match`, `data-reconciliation`) are installed alongside the relevant
ones, so the agent must also learn *which* skills to use.

## Setup

1. **SkillsBench checkout** (data dependency — tasks are read from disk):

   ```bash
   git clone https://github.com/benchflow-ai/skillsbench.git
   ```

2. **Python dependencies** — from the repository root:

   ```bash
   uv sync --package ai-functions-skillsbench-example
   ```

   (or `pip install -e ".[claude-code]" pytest openpyxl pandas python-docx
   rapidfuzz python-pptx pypdf rich` into your environment).

3. **`pdftotext`** — the `court-form-filling` oracle shells out to it:

   ```bash
   sudo apt-get install poppler-utils   # or: brew install poppler
   ```

4. **Credentials** — the agent runs through the Claude Code CLI
   (authenticated `claude` on your PATH), and the optimizer calls Amazon
   Bedrock (`us.anthropic.claude-sonnet-4-6`), so AWS credentials with
   Bedrock access must be present in the environment.

## Usage

```bash
python -u examples/skillsbench/memory_improvement_loop.py \
    --skillsbench /path/to/skillsbench \
    --output-dir /tmp/aifunc-improvement-demo
```

The run takes roughly half an hour: round 0 (no guidelines), one
backward/consolidate pass, round 1 (learned guidelines), and a
round-over-round score comparison at the end. The learned memory persists at
`<output-dir>/memory.json`; installed skills live under
`<output-dir>/.claude/skills/`.

See [REPORT.md](REPORT.md) for results from a full run.

## Files

- `memory_improvement_loop.py` — the improvement loop (agent, optimizer, memory)
- `task_source.py` — SkillsBench task loading, sandboxing, and pytest oracle
- `REPORT.md` — demo results from a full two-round run
