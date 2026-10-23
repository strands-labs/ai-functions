# Teams

An AI Function answers one request on one thread. The `ai_functions.experimental.teams` module puts one lead AI Function over a few live members. The lead holds every member as a typed tool, and it decides whom to ask and what to ask them. Hooks add everything else: extra tools, a rewritten request, a review of the answer, a check before every ask.

This enables:

- **A lead that delegates through typed tools.** Each member is a tool named after it. A member keeps its own conversation across the lead's asks, so a follow-up question can refer to an earlier answer.
- **Members from functions and methods.** `FunctionMember` wraps an `AIFunction` and `MethodMember` wraps one `@ai_method` of a `MethodAgent`. Any object with a name and `spawn`, `ask` and `retire` is a member too.
- **Reviewed answers.** A hook's `on_answer` returns `Accept()` or `Revise(feedback=...)`. A revised answer goes back through every reviewer before it ships, and every reviewer's budget of revisions is bounded.
- **A transcript the team writes.** Every member call and every revision is a typed record, written by the code that carried it rather than reported by the model.
- **Budgets in tokens and dollars.** `Team.run(budget=...)` charges the run everything it spends, prices tokens by the model that spent them, and reserves each member ask before it runs.

Note: the module is experimental, and its API may change in future releases.

## Contents

- [A lead, members and a run](#a-lead-members-and-a-run)
- [Members](#members)
- [Hooks](#hooks)
- [Reviewing the answer: `Accept` and `Revise`](#reviewing-the-answer-accept-and-revise)
- [The transcript](#the-transcript)
- [Budgets and deadlines](#budgets-and-deadlines)
- [Cancelling a run](#cancelling-a-run)
- [Errors](#errors)
- [Examples](#examples)

## A lead, members and a run

```python
from ai_functions import ai_function
from ai_functions.experimental.teams import FunctionMember, Team


@ai_function
def research(topic: str) -> str:
    """Report what you know about {topic}, in three sentences."""


@ai_function(system_prompt="Answer the question. Ask the scout before you decide.")
def answer(question: str) -> str:
    """{question}"""


team = Team(answer, [FunctionMember(research, name="scout")])
run = await team.run("Why are there two tides a day?")
print(run.answer)
```

The lead is an ordinary AI Function. `Team(lead, members, hooks=())` builds the team, and `await team.run(request)` does the rest. It spawns the lead's thread, spawns each member as a child of it, runs the lead with every member as a tool, walks the hooks over the answer, and retires everybody in a `finally`, even when something raised. The request lands in the lead's first positional parameter.

With no `coordinator`, a run stands up a private `InMemoryCoordinator` and `LocalWorker` and closes them when it ends, which is what a script wants. Pass a coordinator of your own to put the run's threads in a larger runtime's registry and event log, and `parent_id=` to make the lead a child of a thread you already hold. The whole run is then one subtree under that parent.

`run` returns a `TeamRun`:

| Field                          | What it holds                                                                                        |
|--------------------------------|------------------------------------------------------------------------------------------------------|
| `answer`                       | The lead's final answer, in the lead's output type                                                   |
| `transcript`                   | The typed records of the run, in order (see [The transcript](#the-transcript))                      |
| `hooks_data`                   | What hooks put in `Workspace.data`                                                                   |
| `blocked`, `blocked_by`        | Whether a mandatory reviewer ran out of revisions without approving, and which ones                 |
| `cancelled`, `cancel_reason`   | Whether the run was stopped while live, and the first reason given                                   |
| `cost`                         | The run's measured spend as a `Cost`: its tokens, and the dollars of the usage the price table priced |
| `run_id`, `lead_id`            | The run's identity and the lead's thread id                                                           |

Empty fields drop out of `model_dump()`, so a bare run dumps its answer, its identity and its cost.

A `Team` instance runs one run at a time. Members hold live handles and hooks keep per-run state, so a second `run` while the first is in flight raises `WiringError` before anything spawns. Build two `Team` instances to run in parallel. Sequential runs on one instance are fine.

## Members

A member is something with a name that spawns, answers one request at a time, and retires. Two adapters cover the common shapes.

`FunctionMember(function, *, name=None, parameter=None, ask_estimate=None, **overrides)` wraps an `AIFunction`. `name` is the member's name on the team and its tool name on the lead's side, and defaults to `function.name`. Pass one when two members share a function. Tool names allow only letters, digits, `_` and `-`, so any other character is mapped to `_` on the tool, while the transcript keeps the real name. `parameter` names the keyword the request arrives as. It defaults to the function's first positional parameter, and the function's other parameters keep their defaults. `overrides` are `ThreadConfig` overrides for the member's thread, such as `model=`.

`MethodMember(agent, method, *, name=None, parameter=None, ask_estimate=None, **overrides)` puts one `@ai_method` of a `MethodAgent` on a live thread. Its name defaults to the method's tool name, `{owner}_{method}`, so the lead's tool, the coordinator's thread name and every error agree. One agent can contribute several members, one per method.

```python
from ai_functions.experimental.teams import MethodMember

team = Team(postmortem, [MethodMember(analyst, "timeline"), MethodMember(analyst, "causes")])
```

The lead sees each member as a tool that takes one `request: str`. The tool's description tells the lead that the member doesn't see its conversation, so the request has to say everything the member needs. A member that exposes a `description` attribute (the `Described` protocol) gets that text appended to its tool's description, so the lead can choose between members by what they do.

A member's answer reaches the lead the way any Strands tool result does. Text passes through as is, and anything else is formatted as its JSON. The transcript records the same text.

### Writing your own member

Any object that satisfies the `Member` protocol joins a team:

```python
class Member(Protocol):
    name: str

    async def spawn(self, coordinator: Coordinator, *, parent_id: ThreadId | None = None) -> Spawned: ...
    async def ask(self, request: str) -> Any: ...
    async def retire(self) -> None: ...
```

`spawn` returns anything with an `id`; the core reads nothing else from it. `retire` must be idempotent, because the core calls it on every path out of a run. A member that also exposes `equip(config_hook)` (the `Equippable` protocol) receives the tools hooks hand to members. A member without it is skipped rather than refused, and the transcript records an `unequipped` note naming the member and the hooks whose tools it goes without.

`current_call()` returns the `CallContext` of the ask in flight on the current task, with the run id, the member's name and the ask's ordinal. A member that does work of its own during an ask can name that work after the ask that caused it. A `Team.run` started inside a member's ask picks the context up on its own and re-runs the outer team's `before_ask` checks before each of its own asks, so stopping the outer run also stops the inner one.

### The team is the only connection between its threads

A thread on a coordinator normally gets the runtime's `list_threads` and `send_message` tools. A team turns both off on every thread it spawns: the lead, and every member that `FunctionMember` or `MethodMember` spawns. Both adapters refuse `coordinator_tools_enabled=True`. So the lead reaches a member only through the member's tool, and every ask it makes is recorded on the transcript and runs under the run's checks, deadline and budget. Members share what they know only through tools a hook hands them. A member you write yourself owns its spawn and should turn those tools off the same way.

## Hooks

A hook is any object that implements one or more of these methods. Every method is optional, and the core skips the ones a hook doesn't have.

| Method                                      | Runs                                                                                    |
|---------------------------------------------|-----------------------------------------------------------------------------------------|
| `on_assemble(work)`                         | Once the members are live, before the lead runs                                         |
| `on_request(work, request) -> str`          | Before the lead's first cycle; returns the request the lead sees, folded in hook order  |
| `tools_for_lead(work, ctx) -> tools`        | On every lead cycle; returns extra tools for that cycle                                 |
| `tools_for_member(work, member, ctx)`       | On every cycle of each equippable member; returns extra tools for that cycle            |
| `before_ask(work, member)`                  | Before every member ask, the lead's and a hook's; raise `StopRun` to stop the run      |
| `on_answer(work, answer) -> Accept/Revise`  | On each answer the lead gives (see below)                                               |
| `on_teardown(work)`                         | When the run is over, successfully or not, while the members are still live            |
| `on_closed(work)`                           | When everything is retired; `work.usage` is the final total and `work.fault` is set     |

`tools_for_lead` and `tools_for_member` are synchronous, because the runtime builds a cycle's tools synchronously. The other methods may be sync or async, and the core awaits whichever it finds. `Team(hooks=...)` is typed `PartialTeamHook`, any object with at least one of these methods, so a type checker flags an object that has none of them.

Every method receives the run's `Workspace`. It carries the request, the live `members`, the lead's handle, the `transcript`, a `data` dictionary that the run publishes as `TeamRun.hooks_data`, and the run's `run_id`. A workspace is one run and compares by identity, so a hook instance that serves many runs keeps per-run state keyed by the workspace. A hook that asks a member a question of its own calls `await work.ask(member, request)`, which runs under the same `before_ask` checks and cancellation as the lead's asks and returns the typed answer. The ask's tokens count toward the run's cost, but it reserves nothing against a budget, has no deadline, and leaves the transcript to the hook.

A model-fixable failure in a hook's tool should come back as `"error: ..."` text, so the model can recover. A failure in persistence should raise.

## Reviewing the answer: `Accept` and `Revise`

```python
import re

from ai_functions.experimental.teams import Accept, Revise, Workspace


class HouseStyle:
    """Send the update back until it states the restore time the way the status page prints it."""

    def on_answer(self, work: Workspace, answer: str) -> Accept | Revise:
        if re.search(r"Resolved at \d{2}:\d{2} UTC", answer):
            return Accept()
        return Revise(feedback='End with the restore time as "Resolved at HH:MM UTC".', cap=2)
```

`on_answer` returns exactly one of two verdicts. `Accept()` passes the answer on. `Revise(feedback=..., cap=2, mandatory=False)` re-runs the lead with the feedback. `cap` is how many revisions this hook may ask for in the run, read from its latest verdict. Anything else returned from `on_answer` raises `WiringError` naming the hook, so a hook that forgets to return can't pass for an approval.

Reviewers run in hook order. After any `Revise`, the walk starts again at the first reviewer, so a revised answer goes back through every reviewer that had already accepted the earlier draft. An answer ships only after one full pass in which every reviewer accepts it or has run out of revisions. Each restart spends one revision from some hook's cap, so the loop is bounded by the sum of the caps. A hook that runs out of revisions lets the answer through, and the transcript records a `revise_cap` entry for it once.

`mandatory=True` turns that pass-through into a blocked result. When the final pass includes a mandatory reviewer that ran out of revisions without approving, `run` returns `TeamRun(blocked=True, blocked_by=(...))` with the last answer still attached. A mandatory reviewer that ran out on one draft and then accepted the next one approved what ships, so it blocks nothing. Hooks are labeled by their class name, with a `#n` suffix when two hooks share a class.

A revision runs the lead again on its live thread, with `Your answer was reviewed and needs revision.` and the feedback as its first argument. The lead keeps its history, so it can revise its own draft with what its members already told it. Its docstring is rendered again with the feedback in place of the request, though, so keep standing instructions (whom to ask, what form to answer in) in the lead's `system_prompt` and keep the docstring to the request. Otherwise the lead reads its instructions again and may ask its members the same questions a second time.

## The transcript

`TeamRun.transcript` lists what happened in the run as frozen typed records:

| Record        | `kind`        | Written when                                    | Fields                                                  |
|---------------|---------------|-------------------------------------------------|---------------------------------------------------------|
| `MemberCall`  | `member`      | The lead asked a member a question              | `member`, `request`, `answer`, `call_id`                |
| `Revision`    | `revise`      | A hook sent the answer back                     | `hook`, `round`, `feedback`                             |
| `RevisionCap` | `revise_cap`  | A hook ran out of revisions                     | `hook`, `rounds`                                        |
| `Note`        | a hook's own  | A hook appended something it wants remembered   | `kind`, `data`                                          |

The core writes the first three, and hooks append `Note` entries. A `Note` can't use the core's three kinds (`CORE_TRANSCRIPT_KINDS`), so a reader filtering on `kind` never mistakes a hook's entry for the core's. `MemberCall.call_id` is `run_id:member:n`, where `n` counts that member's asks from one. `MemberCall.answer` is the text the lead read. When the ask raised, it's `error:` followed by the exception, and the lead sees a failed tool call.

## Budgets and deadlines

```python
from ai_functions.runtime import Budget, Cost

team = Team(
    status_update,
    [
        MethodMember(analyst, "timeline", ask_estimate=Cost(dollars=0.01)),
        FunctionMember(check_claim, name="skeptic", ask_estimate=Cost(dollars=0.01)),
    ],
)
run = await team.run("Payment errors on checkout this morning.", budget=Budget(Cost(dollars=0.50)))
print(run.cost)
```

`ai_functions.runtime.Budget` is a cost bound in tokens, dollars or both, with reservations, and `Budget.child()` nests one budget under another. Under `Team.run(budget=...)` the run holds one charge on the budget for its measured spend: the lead's tokens, every member's, and any hook's asks. The charge is refreshed around every ask and closed at the same figure `TeamRun.cost` reports. Each ask the lead makes also reserves the member's estimate before it runs, so a budget the run has already spent refuses the next ask. A hook's own asks are charged with the run but reserve nothing.

A member's estimate is its `ask_estimate` until it has measured an ask of its own, then the costliest of its recent measured asks. A budget's `on_unknown` policy decides what happens to a member with no estimate. The default, `"refuse"`, rejects such a member with a `WiringError` before anything spawns. `"estimate"` holds the budget's `default_estimate` instead. A refused reservation asks the member nothing and reaches the lead as `error:` text, so the lead can finish with what it has.

Dollars come from the coordinator's price table, `ai_functions.runtime.PRICES` by default, which holds list prices for current Claude models and applies the Bedrock regional premium to geographic inference profiles. Each token usage event names the model that spent it, and the table prices it by that id. To price other models or negotiated rates, pass `InMemoryCoordinator(prices=PRICES.with_prices({...}))` and run the team on that coordinator. Usage the table can't price adds no dollars, so `TeamRun.cost.dollars` sums only the priced usage, and it's `None` when nothing was priced. That usage isn't treated as free, though: once a run has spent any, every later ask under a budget that bounds dollars is refused, with an error naming the model.

`Team(ask_timeout_s=...)` sets a deadline, in seconds, on every member ask. A member that carries an `ask_timeout_s` attribute of its own is held to that instead. Past the deadline the member is retired and the lead reads `error:` text that says so. A retired member isn't asked again in that run: a later call to its tool asks nothing and returns `error:` text. `None`, the default, bounds nothing.

## Cancelling a run

`team.cancel(reason)` stops the instance's live run from the same event loop and returns whether a run was live. The lead is interrupted, every member is retired, the teardown hooks run, and `run` returns `TeamRun(cancelled=True, cancel_reason=reason)`. The first reason wins. A `before_ask` hook that raises `StopRun(reason)` stops the run the same way, before the ask it was checking, and the lead's model isn't called again. Cancelling the caller's own task still propagates as usual.

## Errors

| Error                             | Raised when                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `WiringError`                     | two members share a tool name, a hook returns something other than a verdict, a run is already in flight on the instance, a member can't take hook tools, a member has no estimate under a refusing budget, or a member adapter can't be wired |
| `StopRun`                         | a `before_ask` hook raises it to stop the run; `run` returns a cancelled `TeamRun` rather than propagating it                |
| `BudgetExhausted`, `UnknownCost`  | a reservation is refused; inside a team run they reach the lead as `error:` text                                          |

`WiringError` subclasses `TeamError`, an `AIFunctionError` whose `function_name` names the lead or member that raised it. `StopRun` subclasses `asyncio.CancelledError`. `BudgetExhausted` and `UnknownCost` subclass `ai_functions.runtime.BudgetError`.

## Examples

| Example                 | Shows                                                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `teams/status_desk.py`  | A lead over a method member and a function member, a hook that sends the first draft back, and a run under a dollar budget |

Run it from the `examples/` folder with `uv run teams/status_desk.py`. The [examples guide](../examples/README.md#teams) lists all commands and setup requirements.
