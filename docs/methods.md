# Method Agents

An AI Function is a typed Python function that an agent evaluates. Its docstring is the prompt and its signature is the contract. The `ai_functions.experimental.methods` module brings the same idea to methods. You write a class, mark its methods with `@ai_method`, and each method becomes an AI Function whose prompt can read the object's state through `self`. Python drops `self` from a bound method's signature, so the model only fills in the method's own parameters.

This enables:

- **Typed capabilities over private state.** An object holds what its prompts need, such as notes, a policy, or a gate. Callers pass only the method's arguments.
- **Methods as tools.** `agents()` hands every method to another agent as a typed tool, with the method's parameters, types and defaults in the tool schema.
- **One live thread per method.** `spawn()` puts a method on a thread whose calls share history.
- **Gated proposals.** `GatedProposer` runs a gate as the method's post-condition, so every proposal is checked and a rejected one goes back to the model with the gate's report.
- **Memory the optimizer can update.** `Recall` fetches memory on every call and passes it in as an argument, which is where `TextGradOptimizer` looks for parameters to update.

Note: the module is experimental, and its API may change in future releases.

## Contents

- [Methods as AI Functions](#methods-as-ai-functions)
- [Choosing the model](#choosing-the-model)
- [Methods as tools: `agents()`](#methods-as-tools-agents)
- [Live threads: `spawn()`](#live-threads-spawn)
- [Gated proposals: `GatedProposer`](#gated-proposals-gatedproposer)
- [Memory as a call argument: `Recall`](#memory-as-a-call-argument-recall)
- [Errors](#errors)
- [Examples](#examples)

## Methods as AI Functions

```python
from pydantic import BaseModel, Field

from ai_functions.experimental.methods import MethodAgent, ai_method


class Finding(BaseModel):
    answer: str
    evidence: list[str] = Field(description="The note lines the answer rests on.")


class Analyst(MethodAgent):
    name = "analyst"

    def __init__(self, service: str, notes: str) -> None:
        self.service = service
        self.notes = notes

    @ai_method(Finding, description="Name the most likely causes of a symptom")
    def causes(self, symptom: str, max_causes: int = 2) -> Finding:
        """You're the analyst for the {self.service} service. Using only these notes:

        {self.notes}
        Name at most {max_causes} likely causes of this symptom: {symptom}
        """


analyst = Analyst("checkout", notes)
finding = await analyst.compiled("causes")(symptom="errors on /pay", max_causes=1)
```

`@ai_method` takes the output type first, then any `ThreadConfig` field that `@ai_function` accepts, such as `description`, `name`, `post_conditions`, `max_attempts`, `tools` or `model`. The decorator records these settings and returns the method unchanged, so inheritance and `super()` work as usual. `ai_methods()` lists the decorated methods on a class, including those on base classes.

The docstring is the prompt template. It's rendered with `self` and the call's bound arguments, with defaults filled in, so `{self.notes}` and `{max_causes}` both work, and it renders exactly as an `@ai_function` docstring does. If the method body returns a string or a t-string, that is the prompt and the docstring is skipped. A method whose body returns `None` and that has no docstring raises `RenderError` when it's called.

`compiled(name)` builds an ordinary `AIFunction` for one method. Awaiting it runs one call on a fresh thread, and `trace`, `replace` and `render_prompt` work as they do on any AI Function. Keyword arguments to `compiled` override the decorator's settings, as `AIFunction.replace` applies them, so keys that aren't `ThreadConfig` fields, such as `hooks=`, go to the Strands `Agent`. If your module uses `from __future__ import annotations`, define the types of the method's parameters at module level, because those annotations are resolved against the module's globals.

## Choosing the model

A method's model comes from the most specific setting. The `model=` passed to `compiled`, `agents` or `spawn` comes first, then the `model=` given to `@ai_method`, then the agent's `model` attribute, which you can set on the class or on one instance. When none of these is set, the method runs on the Strands default model, exactly as a plain `@ai_function` does.

```python
class Analyst(MethodAgent):
    model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # every method on this class

    @ai_method(Finding, model="global.anthropic.claude-sonnet-4-6")  # this method only
    def causes(self, symptom: str) -> Finding:
        """Name the likely causes of {symptom}."""


analyst.compiled("causes", model=other_model)  # this call only
```

## Methods as tools: `agents()`

```python
from ai_functions import ai_function

analyst = Analyst("checkout", notes)


@ai_function(tools=analyst.agents())
def postmortem(incident: str) -> str:
    """Write a short postmortem for this incident: {incident}

    Ask the analyst tools for the timeline and the likely cause.
    """
```

`agents()` returns one compiled `AIFunction` per `@ai_method`, and an AI Function given as a tool keeps its own signature. The calling agent sees each method's parameters with their types, defaults and `Literal` choices. The object's state stays on the object. Keyword arguments to `agents()` apply to every method.

Each tool is named `{owner}_{method}`, so the analyst's `causes` method becomes the `analyst_causes` tool. The owner is the agent's `name` attribute, which you can set on the class or on an instance. When it's unset, the owner is the class name in lowercase. `name=`, given to `@ai_method` or to `compiled`, replaces the whole tool name. The compiled `AIFunction`, its threads and its errors all use the same name. Bedrock and the Strands tool-use check accept only letters, digits, `_` and `-` in a tool name, so a name you choose should use only those characters.

## Live threads: `spawn()`

Each call to a compiled method runs on a fresh thread. `spawn(name, coordinator)` puts one method on a live thread and returns a `MethodThread`, and every `run` on it adds to the same conversation. Here the analyst has a `timeline(question: str)` method:

```python
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

coord = InMemoryCoordinator()
worker = await LocalWorker(coord).register()

thread = await analyst.spawn("timeline", coord)
await thread.run(question="When did latency on /pay start climbing?")
await thread.run(question="How many minutes after that was the rollback?")  # sees the first turn
await thread.retire()
```

`run` takes the method's own arguments, so a wrong keyword raises `TypeError`. You always pass the coordinator, so the thread joins your registry and event log. Pass `parent_id=` to attach it to a parent thread. A thread hosts one method, so an agent with three live methods has three threads. To give one method another's context, spawn it with `seed_from=other.id`, which copies the other thread's history into the new one. `seed_events=` and `thread_id=` resume a thread saved by a `SessionStore`, as they do for any AI Function thread.

`MethodThread` also has `notify(text)`, which adds a message to the history without starting a cycle, `fork()`, which branches the conversation, and `handle`, the underlying `ThreadHandle`. `retire()` tears the thread down. It's safe to call twice, or after something else has already ended the thread, and if the teardown itself fails, the thread stays live so you can retry. After `retire()`, every other operation raises `RetiredThreadError` with the method's name.

## Gated proposals: `GatedProposer`

A `GatedProposer` is a `MethodAgent` whose proposals are judged by a gate. The gate runs as the propose method's post-condition, so every proposal is checked before the call returns.

```python
from dataclasses import dataclass

from ai_functions.experimental.methods import GatedProposer, ai_method


@dataclass(frozen=True)
class CapCheck:
    percent: int
    cap: int

    @property
    def ok(self) -> bool:
        return self.percent <= self.cap

    def report_text(self) -> str:
        return f"A {self.percent}% discount is over the {self.cap}% cap for this account tier."


class RetentionOffer(GatedProposer):
    name = "retention"
    REASK = "Propose a smaller discount."

    def __init__(self, cap: int) -> None:
        super().__init__(lambda percent: CapCheck(percent, cap))

    def candidate_of(self, response: Discount) -> int:
        return response.percent

    @ai_method(Discount, description="Propose a retention discount", max_attempts=3)
    def propose(self, case: str) -> Discount:
        """A customer wants to cancel. Propose the discount to offer them.

        {case}
        """


offer = RetentionOffer(cap=10)
discount = await offer.gated()(case=case)
print(offer.rejected)  # every verdict the gate turned away, in order
```

A gate is any callable that takes a candidate and returns a `Verdict`. A verdict is any object with an `ok` property and a `report_text()` method. `Gate` and `Verdict` are protocols, so any type with these members fits, including types you already have. `candidate_of` picks the part of the response the gate judges, and it returns the whole response by default.

`gated(method="propose")` compiles the method with the gate attached. The gate goes first in `post_conditions`, then the method's own `post_conditions`, then any `post_conditions=` you pass. When the gate rejects a proposal, it raises `GateRejectionError` with the report followed by `REASK`. The model reads that message on its next attempt, and the verdict is added to `rejected`. If the gate itself raises, or returns a verdict that can't be read, the call ends with `GateFaultError` instead of retrying. Its message says the gate broke, and nothing is added to `rejected`. `GateFaultError` is a `PostConditionError`, the core error a post-condition raises to end the call rather than send feedback to the model. `gated()` raises `WiringError` before any model call when a post-condition's first parameter has the same name as one of the method's parameters, because the runtime would pass that argument twice. This path needs a synchronous gate, so `gated()` also raises `WiringError` for an async gate. An async gate works with `judge` and `propose_k`.

### Several proposals at once: `propose_k`

```python
admitted = await offer.propose_k(3, coord, case=case)  # [(percent, verdict), ...] for each admitted branch
```

`propose_k(k, coordinator, ...)` spawns one thread for the method, forks it into `k` branches, and asks each branch for one proposal. Each proposal goes straight to the gate with no retries, and the method's own `post_conditions` don't run on the branches. The call returns the admitted `(candidate, verdict)` pairs in branch order and adds the rest to `rejected`. `seed=[{...}, ...]` runs cycles on the first thread before it's forked, so every branch starts from that history. Every branch is retired when `propose_k` returns or raises. The method's other arguments are passed by keyword, so a method with a parameter named `method` or `seed` can't be beamed and raises `WiringError`.

The branches are sibling threads on your coordinator. `propose_k` spawns them without the coordinator's `list_threads` and `send_message` tools, so a branch can't find or message its siblings, and each proposal is an independent draw. The method's own `tools=` still reach every branch.

## Memory as a call argument: `Recall`

`TextGradOptimizer` finds the parameters it can update by scanning the arguments of a traced call. So memory has to arrive as an argument for the optimizer to update it. `Recalled` and `Recall` declare memory on the method's signature and pass it in as an argument on every call.

```python
from typing import Annotated

from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer
from ai_functions.experimental.methods import MethodAgent, Recall, Recalled, ai_method


class Choice(BaseModel):
    transition: str


class Playbook(BaseModel):
    guidance: list[str] = Field(default_factory=list, description="Lessons about deciding, one per entry.")


class Navigator(MethodAgent):
    name = "navigator"

    @ai_method(Choice, description="Choose the next transition")
    def choose(self, advice: Annotated[list[str], Recalled("guidance", k=2)], state: str) -> Choice:
        """Guidance retrieved for this decision: {advice}

        You are in {state}. Choose the next transition.
        """


memory = JSONMemoryBackend(Playbook, actor_id="navigator", path="playbook.json")
result = await Recall(Navigator(), memory).trace("choose", state="PENDING", queries={"advice": "the permit is pending"})

await TextGradOptimizer().step(result, "The choice ignored the missing document.", backends=[memory])
```

`Recalled(source, k=None)` marks a parameter as coming from the memory field `source`. With `k=None`, the whole field is recalled. With `k=2`, the field is searched and the top two entries are passed in. A search needs a query on every call, passed through `queries=`.

`Recall(agent, backend).trace(method, ...)` fetches each marked parameter fresh, passes it to the method as a keyword argument, and traces the call. It returns the same `Result` that `AIFunction.trace` does, ready for `optimizer.step`. After a search, the optimizer applies its feedback to the entries that call retrieved. See [Memory and optimization](tutorial.md#memory-and-optimization) for how the graph and the optimizer work.

The marked parameter stays in the tool schema, so a peer agent that calls the method as a tool supplies the value itself. A value you pass to `trace` by keyword is used as is, and nothing is retrieved for it. `trace` raises `RecallWiringError` before any retrieval or model call when a search has no query, a query names a parameter that won't be retrieved, a positional argument would land on a marked parameter, or the method has a parameter named `queries` or `overrides`. It raises `CompilationError`, also before any retrieval, when the method isn't an `@ai_method`.

To train a value that lives in the process instead of in a memory backend, mark the parameter `Trainable`; see [Local parameters](params.md).

## Errors

Every error in the module subclasses `MethodError`, an `AIFunctionError` whose `function_name` names the agent or method that raised it.

| Error                | Raised when                                                                          |
|----------------------|--------------------------------------------------------------------------------------|
| `CompilationError`   | `compiled` or `Recall.trace` gets a name that isn't an `@ai_method`                  |
| `RenderError`        | a method has no docstring and its body returns `None`                                |
| `RetiredThreadError` | a `MethodThread` is used after `retire()`                                            |
| `GateRejectionError` | the gate turns a proposal away; the message is the model's retry prompt              |
| `GateFaultError`     | the gate, `candidate_of`, or the verdict raises, or `admits` gets an awaitable       |
| `WiringError`        | a post-condition would get an argument twice, or a gate or beam can't be wired       |
| `RecallMarkerError`  | a `Recalled` marker has `k` below 1, or one annotation carries two different markers |
| `RecallWiringError`  | a `Recall.trace` call can't be wired, as listed above                                |

## Examples

| Example                     | Shows                                                                                                            |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|
| `methods/analyst_team.py`   | An analyst whose methods read notes from `self`, called once, handed to an orchestrator, and spawned on a thread |
| `methods/gated_proposer.py` | A discount proposer whose gate holds a policy cap, with a rejected candidate, `gated()` retries, and a beam      |

Run either of them from the `examples/` folder, for example
`uv run methods/analyst_team.py`. The [examples guide](../examples/README.md#methods)
lists all commands and setup requirements.
