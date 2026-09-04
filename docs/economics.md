# Economics-aware Execution

Post-conditions give an AI Function correctness semantics: a result either passes verification or it does not. The `ai_functions.experimental.economics` module adds the economics: what a result is worth in dollars, what each candidate model's tokens cost, and therefore which model to try, whether to switch after a failure, and when to stop. Everything shares one currency, so one rule governs every attempt: **an attempt is worth making only when it's expected to yield more than it costs**.

This enables:

- **Cost-aware model routing** — to maximize profit across multiple attempts, each call starts with the model whose *reservation index* [(Weitzman, 1979)](https://www.jstor.org/stable/1910412) is the highest, optionally switches models when verification fails, and declines tasks that are not worth attempting at all. Per-model estimates of pass rate and cost (used for computing reservation indices) sharpen with use.
- **Task-aware routing with an LLM forecaster** — a lightweight AI function reads each task and predicts, per candidate, its chance of success and its cost, so easy tasks route to cheap models and hard tasks route to strong models within the same function. 
- **Routing that can improve with feedback** — routing decisions plug into the library's [optimization loop](tutorial.md#memory-and-optimization): feedback on a workflow's final output propagates back to the routing decisions that produced it, corrects their statistics, and distills task-routing notes for future calls.
- **Adaptive stopping for graded tasks** — when results have graded scores (a review that finds four defects beats one that finds two), a `scorer` scores the result in `[0, 1]`, and the search keeps sampling while another attempt is expected to yield more than it costs, keeping the best result.

Note: the module is experimental — the API may change in future releases.

## Contents

- [Routing with `@routed`](#routing-with-routed)
- [Task-aware routing that learns: `LLMForecaster`](#task-aware-routing-that-learns-llmforecaster)
- [How a search ends](#how-a-search-ends)
- [Tasks with continuous scores](#tasks-with-continuous-scores)
- [Customizing the search](#customizing-the-search)
- [Going further](#going-further)
- [Examples](#examples)

## Routing with `@routed`

`@routed` stacks on `@ai_function` and searches across candidate models to maximize the expected net gain of a call (the value of the best result minus the cost of all attempts) given your dollar value for a successful outcome and the prices of the candidate models:

```python
from ai_functions import ai_function
from ai_functions.experimental.economics import PricedModel, Prices, routed

# The candidate models, priced at what you pay (dollars per million tokens).
HAIKU = PricedModel(model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                    prices=Prices(input=1.00, output=5.00))
SONNET = PricedModel(model="global.anthropic.claude-sonnet-4-6",
                     prices=Prices(input=3.00, output=15.00))


# A solved instance is worth 50 cents to us; check_sat is an ordinary
# post-condition that verifies the assignment against the clauses.
@routed(models=[HAIKU, SONNET], value=0.50)
@ai_function(post_conditions=[check_sat])
def solve(clauses: str, n_vars: int) -> Assignment:
    """Find a satisfying assignment for this 3-SAT formula over variables x1..x{n_vars}.

    {clauses}"""


result = await solve(clauses=clauses, n_vars=8)   # called like any AI Function
```

A routed AI Function consists of three key components, and each is doing a specific job:

- **`value`** is what a fully-successful result is worth to you, in dollars. It is a positive constant, and the reward side of every decision: a model is only worth trying if `value` times its expected score exceeds its expected cost.
- **`models`** are the candidates, each a `PricedModel` pairing a model with the per-token prices you pay for it (`Prices` also accepts `cache_read`/`cache_write` rates, and an optional `description` that estimators can read).
- **The post-conditions define success.** They are the verifier that decides whether an attempt passed.

A call then works as follows: the expected score and cost are estimated per candidate; candidates are ordered by reservation index; the search tries the top candidate, keeps the best result while any candidate's reservation index beats the best reward in hand, and abstains (raising `Abstained` rather than knowingly wasting money) when no candidate's index is positive (expected yield is lower than cost). This is the default `ReservationPricePolicy`; see [Customizing the search](#customizing-the-search) to swap it for `Greedy` routing.

Routing learns by default. Out of the box, estimates come from `EmpiricalBeliefs`: per-candidate pass rates and average costs, starting from a uniform prior and updated after every attempt. Over a batch of calls the cheap model keeps the tasks it handles and the strong model inherits the ones it doesn't. You can watch this happen:

```python
print(solve.beliefs.stats())
# {'haiku': '40% pass over 15 attempts, avg cost $0.0142, ...',
#  'sonnet': '78% pass over 9 attempts, avg cost $0.0840, ...'}
```

Optional knobs bound and shape the search: `budget` is a hard dollar cap per call (distinct from `value`: `value` drives choices, `budget` bounds spend); `max_tries` (default 1) caps independent attempts per candidate; `scorer` grades partial success ([Tasks with continuous scores](#tasks-with-continuous-scores)); `policy` swaps the ordering-and-stopping rule ([Customizing the search](#customizing-the-search)).

See `examples/economics_escalate.py` for a runnable comparison: SAT instances routed cheap-first vs. straight to the strong model, with the dollar savings printed. `examples/economics_learning.py` shows the beliefs updated over a batch.

## Task-aware routing that learns: `LLMForecaster`

Population statistics treat every call the same. For query-specific estimates, pass `beliefs=LLMForecaster(...)`: a lightweight AI Function reads each task and predicts, per candidate, its chance of passing and its expected cost, anchored on the learned statistics. Its state lives in your memory schema, one `RoutingMemory` field per routed function, so everything it learns persists across processes:

```python
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.economics import LLMForecaster, RoutingMemory, routed
from ai_functions.memory import JSONMemoryBackend
from ai_functions.optimizer import TextGradOptimizer


class Memory(BaseModel):
    research_routing: RoutingMemory = Field(default_factory=RoutingMemory)


memory = JSONMemoryBackend(schema=Memory, actor_id="demo", path="memory.json")


@routed(
    models=[HAIKU, SONNET],
    value=0.05,  # good sources are worth 5 cents
    beliefs=LLMForecaster(memory=memory, memory_key="research_routing"),
    budget=0.10,
)
@ai_function(tools=[web_search], post_conditions=[cited])
def research(query: str) -> Sources:
    """Research this topic on the web and return the key findings with sources:

    {query}"""
```

This is also where routing connects to the library's optimization loop. The post-conditions are only *local* checks (here: findings carry source URLs); whether the research was actually *useful* is decided downstream, by whoever consumes it. Run the function under `trace()` and that judgment can flow back:

```python
run = await research.trace(query="What changed in the EU AI Act's GPAI obligations in 2025?")

optimizer = TextGradOptimizer()
await optimizer.step(
    run,
    "Too shallow: this needed primary sources (the Act's text, Commission guidance).",
    backends=[memory, research],   # an economic function hosts its own routing parameters
)
```

One `optimizer.step` teaches the router two things at once. The *numeric* channel updates the statistics: the routed model's attempts in that run are re-scored by the downstream feedback, so a model whose results pass local checks but don't hold up downstream sees its pass rate sink anyway. The *text* channel distills the feedback into the forecaster's casebook (`research_routing/notes` in your memory), steering future task-dependent routing — "regulatory comparisons: haiku's sources too thin, route strong." Feedback given on a *downstream* output propagates to the routed stages that fed it, exactly as in [Memory and optimization](tutorial.md#memory-and-optimization); `examples/economics_workflow.py` runs the full loop on a two-stage pipeline, both stages routed, settled by one line of feedback on the final report.

## How a search ends

A search stops when no remaining attempt is expected to pay for itself. If an attempt passed by then, the call returns the best result. When none did, the failure is one of three typed exceptions, each carrying the attempt trail in `.records`:

- **`Abstained`** — nothing was tried: no candidate's expected reward covered its cost. This is a feature, not a failure mode: a router that cannot say "not worth it" silently overpays on hopeless tasks. It tells you the task looks unprofitable *before* money is spent — raise `value`, or stop sending this class of task here.
- **`CandidatesExhausted`** — every profitable candidate ran (up to its `max_tries`) and no attempt passed the post-conditions.
- **`BudgetExceeded`** — a candidate would still be worth trying, but its expected cost no longer fits the remaining budget: the cap, not the economics, stopped the search.

## Tasks with continuous scores

By default an attempt is scored **1.0 if its post-conditions pass, 0.0 if not**, and its reward is `value * score`: the full `value` for a pass, nothing for a fail. That is the pass/fail case above.

Some tasks are *graded*: a review that finds four real defects is worth more than one that finds two. Pass a `scorer` (any function returning a score in `[0, 1]`) and the reward becomes `value * score`, pricing partial success.

```python
def f1_score(report: Report) -> float:
    """Score in [0, 1]: F1 of the report's defects against ground truth."""
    ...  # precision/recall harmonic mean; 1.0 only for an exact match


# A perfect report is worth 10 cents; reward = 0.10 * f1_score.
@routed(models=[HAIKU, SONNET], value=0.10, scorer=f1_score, budget=0.10)
@ai_function(post_conditions=[at_least_one])
def review(source: str) -> Report:
    """You are reviewing this C module. Report any real bugs you find ..."""
```

The `scorer` must return a score in `[0, 1]`, otherwise an out-of-range error will be raised. If you have a raw count or an unbounded metric, normalize it first (e.g. divide by a target) and let `value` carry the scale.

See `examples/economics_graded.py`: two models graded by F1, each calibrated with a few attempts to build an empirical reward distribution, then searched with the Pandora's Box rule.

## Customizing the search

**Policies.** How candidates are ordered and when the search stops is a pluggable `policy=`. `@routed` defaults to `ReservationPricePolicy`, which orders candidates by their *reservation price*, following the [Pandora's Box rule (Weitzman, 1979)](https://www.jstor.org/stable/1910412): it prices in the option to sample again and continues while some candidate's reservation price beats the best reward in hand. This is an adaptive and sequential version of best-of-N sampling, with the index setting both the order and stopping criterion. Pass `policy=Greedy()` for routing that maximizes expected profit per attempt and stop once a positive reward is in hand; `Exhaustive` tries everything the budget allows, cheapest first.

**Custom beliefs.** When you know the feature that governs task difficulty, you can skip learning it: subclass `Beliefs` and compute the estimates directly. `estimate` receives a `TaskView` carrying both the rendered prompt and the structured call `arguments`:

```python
from ai_functions.experimental.economics import Beliefs
from ai_functions.experimental.economics.search import Bernoulli, Estimate


class RatioBeliefs(Beliefs):
    """3-SAT hardness is governed by the clause/variable ratio; read it
    from the call's own arguments instead of learning it from outcomes."""

    async def estimate(self, task, candidates, value, history):
        ratio = (task.arguments["clauses"].count("\n") + 1) / task.arguments["n_vars"]
        return {
            c.label: Estimate(
                dist=Bernoulli(p=self._pass_probability(c.label, ratio), value=value),
                cost=self._expected_cost(c, ratio),
            )
            for c in candidates
        }
```

`estimate` must return an estimate for every candidate it is given, and the `value` it receives is the constant `value` (the reward scale), so an estimate's expected reward is `value` times the expected score. For known workloads and tests, `Beliefs.fixed({label: Estimate(...)})` returns constant estimates with no learning. See `examples/economics_route.py` for the complete `RatioBeliefs`.

**Custom candidates.** When model swaps are not enough, pass `candidates=[Candidate(label=..., fn=..., prices=...)]` instead of `models=`: a candidate is *any* `AIFunction` plus its prices — a different thinking budget, a different prompt, or a non-LLM heuristic wrapped as a function. Build variants with `fn.replace(...)`.

## Going further

**Inspecting a run.** `await fn.trace(...)` runs like a plain call but keeps the event log, so the per-attempt economics can be read back afterwards:

```python
from ai_functions.experimental.economics import attempts, decisions, spend

run = await review.trace(source=BUGGY_C)

for r in await attempts(run):        # one AttemptRecord per attempt, in order
    print(r.candidate, r.reward, r.cost, r.local_score)  # local_score in [0, 1], reward in dollars

await decisions(run)                 # the ranked estimation rounds, in order
await spend(run)                     # total dollars booked by the run and its subtree
```

Every attempt runs as a child thread and emits durable events, so `spend` gives dollar accounting across a whole tree of economic calls from the event log alone. A failed run carries the same records on the exception's `.records`.

**Deciding without executing.** `await fn.plan(...)` runs one estimation round and returns a `Decision` without attempting anything: the candidate the search would try first (`None` means it would abstain — the no-exception way to anticipate `Abstained`) and the full ranking, for dashboards and debugging. A caller that takes the decision and executes it *itself* closes the learning loop with `decision.report(result, cost)`; without it, the beliefs never see the outcome. See `examples/economics_route.py`.

**Persisting the plain statistics.** `LLMForecaster` persists everything it learns through its `RoutingMemory` field. To persist the default statistics without a forecaster, construct them with a backend directly: `EmpiricalBeliefs(memory=backend, stats_key="research_routing/stats")` reloads at construction and rewrites after every update, so a new process resumes routing where the last one left off.

## Examples

| Example                 | Shows                                                                                                                       |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `economics_escalate.py` | `@routed` basics: cheap-first escalation on a SAT batch, dollar savings vs. a straight-to-strong baseline                   |
| `economics_learning.py` | `EmpiricalBeliefs` converging over a batch: exploration from a uniform prior, routing sharpening with evidence              |
| `economics_graded.py`   | Graded routing with `scorer`: two models graded by F1 with per-arm calibrated beliefs, searched with the Pandora's Box rule |
| `economics_route.py`    | A custom task-dependent `Beliefs`; `plan()` previews each decision                                                          |
| `economics_workflow.py` | Two routed stages in a pipeline; `LLMForecaster`, persistence, and feedback settling both stages via `optimizer.step`       |

Run any of them from the `examples/` folder with `uv run economics_<name>.py`.
