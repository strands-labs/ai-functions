# Examples

## Run your first example

From the repository root:

```bash
cd examples
export STRANDS_TOOL_CONSOLE_MODE="enabled"
uv run getting_started/one_shot.py
```

Use Python 3.12 or later and configure model credentials as described in
[Getting started](../README.md#getting-started). Unless noted otherwise, the commands below run
from `examples/`.

By default the examples use pre-configured `BedrockModel` of various sizes:

```python
from ai_functions import ai_function
from example_helpers import models

@ai_function(model=models.medium)
def answer(question: str) -> str:
    """Answer: {question}"""
```
Edit [models.py](example_helpers/models.py) to change the model
configuration, credentials, or use a different model or provider.
The Claude Code and Kiro integrations use their external runtimes.

## A short learning path

1. [One-shot execution](getting_started/one_shot.py): define a function, get a
   typed result, and add a post-condition.
2. [Async translation](getting_started/async_translation.py): run independent
   AI calls concurrently and validate their output.
3. [Meeting summaries](validation/meeting_summary.py): combine Python checks
   and an AI post-condition to guide retries.
4. [Receipt workflow](workflows/receipt_tool.py): expose an AI function as a
   tool that another agent can call.
5. [Multi-turn conversation](threads/multi_turn.py): keep conversation history
   across calls, then try [saving and resuming](threads/session_resume.py).
6. [Memory tools](memory/tools.py): let an agent read and write memory, then try
   [optimization](memory/optimization.py) to improve a workflow from feedback.
7. [Verified leave requests](verified/hr.py): prove that a decision follows a
   formal policy, then try [verified compilation](verified/payout.py) to generate
   a native function with a proof of correctness.

Configure model credentials before starting. The verified examples also set up
Lean automatically; see [Verified](#verified) for native compilation setup.

## I want to…

| Goal                                                         | Start here |
|--------------------------------------------------------------| --- |
| Define an AI function or return structured data              | [Getting started](#getting-started) |
| Check answers and retry when they fail                       | [Validation](#validation) |
| Let an agent execute Python and return native objects        | [Code execution](#code-execution) |
| Combine functions or delegate work through tools             | [Workflows](#workflows) |
| Keep state, resume a session, or host cooperating agents     | [Threads](#threads) |
| Remember information and improve results from feedback       | [Memory](#memory) |
| Choose models according to cost and expected success         | [Economics](#economics) |
| Use Claude Code or Kiro as a thread                          | [Integrations](#integrations) |
| Verify the agent's work or compile a verified implementation | [Verified](#verified) |
| Explore a larger learning loop or benchmark                  | [Projects](#projects) |

## Getting started

| Example | What it teaches | Run |
| --- | --- | --- |
| [One shot](getting_started/one_shot.py) | Primitive and structured results, post-conditions, sync and async calls. | `uv run getting_started/one_shot.py` |
| [Sentiment](getting_started/sentiment.py) | Compose classification and summarization with a retrying post-condition. | `uv run getting_started/sentiment.py` |
| [Async translation](getting_started/async_translation.py) | Translate concurrently with `asyncio.gather` and an AI validator. | `uv run getting_started/async_translation.py` |

## Validation

These examples check output with post-conditions and retry when checks fail.

| Example | What it teaches | Run |
| --- | --- | --- |
| [Meeting summary](validation/meeting_summary.py) | Combine length checks and an AI style check. | `uv run validation/meeting_summary.py` |
| [Contract analyzer](validation/contract_analyzer.py) | Validate coverage, source excerpts, and risk fields in structured output. | `uv run validation/contract_analyzer.py` |
| [SQL generator](validation/sql_generator.py) | Validate SQL syntax and schema references using `sqlglot`. | `uv run --with sqlglot validation/sql_generator.py` |

## Code execution

These examples enable local Python execution. Their inputs and sample data are
included; the shared environment supplies SymPy and pandas.

| Example | What it teaches | Run |
| --- | --- | --- |
| [Symbolic integral](code_execution/symbolic_integral.py) | Return a native SymPy expression from generated Python. | `uv run code_execution/symbolic_integral.py` |
| [Universal loader](code_execution/universal_loader.py) | Load CSV, JSON, and SQLite into a validated DataFrame. | `uv run code_execution/universal_loader.py` |

## Workflows

Compose functions with Python control flow, expose them as tools, or implement
your own workflow thread.

| Example | What it teaches | Run | Extra setup |
| --- | --- | --- | --- |
| [Receipt tool](workflows/receipt_tool.py) | Give a Strands agent an AI function as a parsing tool. | `uv run workflows/receipt_tool.py` | None; sample receipts are included. |
| [Stock report](workflows/stock_report.py) | Research in parallel, pass DataFrames between functions, and produce HTML. | `uv run workflows/stock_report.py` | `EXA_API_KEY` or `TAVILY_API_KEY`; network access for market data. |
| [Research team](workflows/research_team.py) | Delegate research, planning, and critique through tools. | `uv run workflows/research_team.py` | `EXA_API_KEY` or `TAVILY_API_KEY`. |
| [Custom workflow](workflows/custom_workflow.py) | Implement a plain Python workflow as a spawnable thread. | `uv run workflows/custom_workflow.py` | None. |

## Threads

Keep conversation state, manage thread lifecycles, and connect workers.

| Example | What it teaches | Run |
| --- | --- | --- |
| [Multi-turn conversation](threads/multi_turn.py) | Reuse conversation history across calls. | `uv run threads/multi_turn.py` |
| [Session resume](threads/session_resume.py) | Save two threads and restore their histories and IDs. | `uv run threads/session_resume.py --session-dir /tmp/ai-functions-session` |
| [Summarization](threads/summarization.py) | Compact a long conversation proactively. | `uv run threads/summarization.py` |
| [Coordinator basics](threads/coordinator_basics.py) | Spawn child threads and observe their lifecycle events. | `uv run threads/coordinator_basics.py` |
| [Two local workers](threads/two_workers_local.py) | Discover peers and send messages through a shared coordinator. | `uv run threads/two_workers_local.py` |
| [Two remote workers](threads/two_workers_remote.py) | Use the same thread interactions over WebSockets. | `uv run threads/two_workers_remote.py` |
| [CLI hosting](threads/serve_cli.py) | Host a persistent agent that other processes can discover and drive. | Start `uv run ai-functions server`, then `uv run threads/serve_cli.py` in another terminal. |

The remote-worker example starts its own local server on port 9901. The CLI
hosting example's [module docstring](threads/serve_cli.py) shows the commands for
submitting prompts and attaching an interactive client.

## Memory

Start with memory tools or list search, then try optimization and procedural
memory. Local examples create temporary memory stores.

| Example | What it teaches | Run | Extra setup |
| --- | --- | --- | --- |
| [Memory tools](memory/tools.py) | Give an agent tools to read and write its memory. | `uv run memory/tools.py` | None. |
| [List search](memory/list_search.py) | Retrieve relevant list entries and consolidate feedback. | `uv run memory/list_search.py` | None. |
| [Optimization](memory/optimization.py) | Backpropagate feedback through a multi-function workflow. | `uv run memory/optimization.py` | None. |
| [Procedural memory](memory/procedural.py) | Learn reusable Python helpers from execution feedback. | `uv run memory/procedural.py` | None. |
| [AgentCore memory](memory/agentcore.py) | Use a persistent AWS backend for the optimization loop. | `uv run memory/agentcore.py` | AWS credentials with AgentCore Memory permissions; configure its region as described in the script. |

For a larger code-generation learning loop, see [SciPy learning](projects/scipy_learning.py).

## Economics

Route among priced candidates while learning which attempts are worth their
cost. The prices are illustrative; edit
[the priced presets](example_helpers/economics.py) for your own setup.
Sample tasks are included, and the workflow uses a canned search tool.

| Example | What it teaches | Run |
| --- | --- | --- |
| [Routing basics](economics/routing_basics.py) | Escalate between models and compare cost with a single-model baseline. | `uv run economics/routing_basics.py` |
| [Custom routing](economics/custom_routing.py) | Predict success from task structure and preview a routing decision. | `uv run economics/custom_routing.py` |
| [Learning](economics/learning.py) | Improve routing beliefs from observed outcomes. | `uv run economics/learning.py` |
| [Graded search](economics/graded_search.py) | Calibrate candidates and keep the best report while further attempts pay off. | `uv run economics/graded_search.py` |
| [Workflow](economics/workflow.py) | Update two routed stages using downstream feedback. | `uv run economics/workflow.py` |

The workflow persists its learned state in `forecast_memory.json` in the working
directory.

## Integrations

These examples run external agent sessions as threads. Run them from
`examples/`, like the other examples; the agents inspect that working directory.

| Example | What it teaches | Run | Extra setup |
| --- | --- | --- | --- |
| [Claude Code](integrations/claude_code.py) | Observe and drive a Claude Agent session through the thread API. | `uv run integrations/claude_code.py` | An authenticated Claude Code runtime. |
| [Kiro](integrations/kiro.py) | Observe and drive a Kiro session through the thread API. | `uv run integrations/kiro.py` | An authenticated Kiro CLI with ACP support. |

## Verified

Use formal contracts to prove an answer or synthesize a compiled function.
The library manages the Lean toolchain; the first run may download and build it.
Compilation uses `leanc` from that Lean installation. On macOS, install the
Xcode Command Line Tools to provide the system SDK used during compilation. See
[the verified guide](../docs/verified.md#getting-started) for setup.

| Example | What it teaches                                                                    | Run |
| --- |------------------------------------------------------------------------------------| --- |
| [Leave requests](verified/hr.py) | Combine interpreted requests, employee records, and a proof of policy compliance.  | `uv run verified/hr.py` |
| [Policy questions](verified/hr_generic_question.py) | Infer how to formalize a question against a policy and provide a certified answer. | `uv run verified/hr_generic_question.py` |
| [Access decisions](verified/access.py) | Prove an authorization decision from facts fetched through tools.                  | `uv run verified/access.py` |
| [Ledger audit](verified/audit.py) | Prove that a paginated audit found every incorrect fee.                            | `uv run verified/audit.py` |
| [Independent set](verified/mis.py) | Combine a solver with a correctness certificate.                                   | `uv run verified/mis.py` |
| [Lower bound](verified/lower_bound.py) | Compile a binary-search-style function from a Lean contract.                       | `uv run verified/lower_bound.py` |
| [Payout](verified/payout.py) | Compile a verified function from a contract written in Python.                     | `uv run verified/payout.py` |

## Projects

Larger demonstrations with multiple training steps, datasets, or extra setup.

| Project | What it teaches | Run | Extra setup |
| --- | --- | --- | --- |
| [SciPy learning](projects/scipy_learning.py) | Train on bundled problems and compare code generation before and after learning. | `uv run projects/scipy_learning.py` | None beyond the shared environment; runs several model calls. |
| [Looped threads](projects/looped_threads/README.md) | Repeatedly edit, execute, and evaluate a house-price predictor. | `(cd projects/looped_threads && uv run main.py)` | Its own dependencies; downloads the dataset on first use. |
| [SkillsBench](projects/skillsbench/README.md) | Improve an external agent's workflow memory using benchmark feedback. | See its README for the setup and command with `--skillsbench`. | SkillsBench checkout, Claude Code, additional Python packages, and `pdftotext`. |
