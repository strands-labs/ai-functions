# Local Parameters

The `ai_functions.experimental.params` module makes a value on a function's signature trainable. Mark a parameter `Trainable`, and its default becomes the initial value of a parameter that `TextGradOptimizer` can update from feedback. The value lives in a `Local` host in your process, so no memory backend is needed. Each parameter carries a version, a rule for combining feedback, and a policy for what happens when two learners write to it at once.

```python
from typing import Annotated

from ai_functions import TextGradOptimizer, ai_function
from ai_functions.experimental.params import Bind, Trainable


@ai_function
def reply(ticket: str, guidelines: Annotated[str, Trainable()] = "Thank the customer.") -> str:
    """Follow these guidelines: {guidelines}

    Write the reply to this support ticket: {ticket}
    """


bound = Bind(reply)
result = await bound.trace(ticket=ticket)
await TextGradOptimizer().step(result, "End every reply with the case reference.", backends=[bound.local])

guidelines, version = bound.local.load("guidelines")   # the default plus the new guideline, at version 1
```

## Contents

- [Declaring a trainable parameter](#declaring-a-trainable-parameter)
- [Training it: `Bind` and `TextGradOptimizer`](#training-it-bind-and-textgradoptimizer)
- [Methods](#methods)
- [How feedback combines: reducers](#how-feedback-combines-reducers)
- [Versions and concurrent learners](#versions-and-concurrent-learners)
- [Sharing a parameter across stages](#sharing-a-parameter-across-stages)
- [Saving and restoring](#saving-and-restoring)
- [Search optimizers: `components` and `apply`](#search-optimizers-components-and-apply)
- [Local parameters or recalled memory](#local-parameters-or-recalled-memory)
- [Errors](#errors)
- [Examples](#examples)

## Declaring a trainable parameter

`Trainable` is `Annotated` metadata. `Annotated[str, Trainable()]` marks a parameter as a local trainable value, and the parameter's Python default is its initial value. `Trainable[str]` is a short spelling of the same annotation. Type checkers read `Trainable[str]` as the marker class and reject its default, so code that is type-checked writes the `Annotated` form.

Every knob is optional:

| Knob          | Default     | What it does                                                                                   |
|---------------|-------------|------------------------------------------------------------------------------------------------|
| `reducer`     | inferred    | How several pieces of feedback combine into the next value; see [reducers](#how-feedback-combines-reducers) |
| `conflict`    | `"refuse"`  | What a write against a stale version does; see [versions](#versions-and-concurrent-learners)   |
| `description` | `""`        | What the parameter is for, shown to the optimizer's backward model with the reducer's guidance |

```python
style: Annotated[str, Trainable(conflict="merge", description="How the report should read.")] = "Plain prose."
```

The parameter stays in the function's signature and its tool schema. A caller, or a peer agent calling the function as a tool, can still pass a value, and a value passed this way is used as is.

## Training it: `Bind` and `TextGradOptimizer`

`Bind(fn)` reads the function's `Trainable` markers and declares each parameter in a `Local` host from its default. `bound.trace(...)` fills each marked parameter the caller did not pass with a fresh `ParameterView` of its current value, then traces the call. It returns the same `Result` that `AIFunction.trace` does, ready for `optimizer.step`.

The host is the backend: `optimizer.step(result, feedback, backends=[bound.local])` routes the backward pass's feedback to each parameter and asks the host to fold it in. The host combines feedback with the parameter's reducer and makes no model call. Each write that changes the value bumps the version by one, and `bound.local.load(name)` returns the value and its version. `bound.local.history(name)` returns one value per version, oldest first.

`bound.trace` forwards the target's arguments unchanged, and a positional argument can't land on a marked parameter. `overrides=` passes `ThreadConfig` overrides, such as `{"model": model}`, to the traced call. A marked parameter can't be positional-only, can't be named `overrides`, and needs a default; `Bind` refuses each case with `ParamWiringError` before any model call.

## Methods

`Trainable` works the same way on an `@ai_method` parameter. `Bind` takes a `MethodAgent`, and the first argument of `trace` is the method name, as with `Recall.trace`:

```python
class Navigator(MethodAgent):
    @ai_method(Choice, description="Choose the next transition")
    def choose(self, state: str, playbook: Annotated[str, Trainable()] = "Prefer the shortest path.") -> Choice:
        """Playbook: {playbook}

        You are in {state}. Choose the next transition.
        """


bound = Bind(Navigator())
result = await bound.trace("choose", state="PENDING")
```

`bound.bound("choose")` lists the marked parameters of one method. A parameter can be marked `Trainable` or `Recalled`, not both; `Bind` refuses an annotation that carries both.

## How feedback combines: reducers

A reducer turns the current value and the feedback routed to it into the next value. It is a pure function: no model call, no I/O, and equal inputs give equal outputs. When no `reducer` is given, the initial value's shape picks one:

| Initial value                        | Reducer       | What it does                                                                                  |
|--------------------------------------|---------------|-----------------------------------------------------------------------------------------------|
| `str` or `list[str]`                 | `ORDERED`     | Appends each piece of feedback as one block of lines (or one list item), in order, skipping verbatim repeats |
| a number, or a sequence of numbers   | `NUMERIC_ADD` | Parses each piece of feedback as a number (or a JSON array of numbers) and adds it, pointwise for vectors |
| anything else                        | none          | The declaration is refused until you pass `reducer=`                                           |

`ORDERED` keeps the current text verbatim and never rewrites it. `NUMERIC_ADD` refuses a vector delta of the wrong length. A reducer refuses an input it can't combine with `ReducerError`; it never coerces or drops one.

Each reducer has a one-sentence `guidance` that the host appends to the parameter's description, so the backward model knows what shape of feedback the parameter accepts. TextGrad's backward model writes feedback to a parameter as `- add:`, `- update:` and `- delete:` bullets. `ORDERED` and `NUMERIC_ADD` remove the `- add:` marker before combining, and they refuse a `- update:` or `- delete:` line, because appending and adding can't apply it.

Any callable `(current, feedback) -> new` can be a reducer; pass it as `reducer=` and it's wrapped in a `CallableReducer`, whose guidance is the first line of the function's docstring. A `CallableReducer` receives the feedback text unchanged, markers included, so a rule that rewrites or deletes belongs there:

```python
from collections.abc import Sequence

from ai_functions.types.graph import GradFeedback


def latest(current: str, feedback: Sequence[GradFeedback]) -> str:
    """The last feedback replaces the value."""
    return feedback[-1].text if feedback else current


tone: Annotated[str, Trainable(reducer=latest)] = "Friendly."
```

## Versions and concurrent learners

Every view records the version it read. When `TextGradOptimizer` consolidates, it passes the oldest version the graph read back to the host as `expected_version`. If another learner wrote to the parameter in between, the parameter's `conflict` policy decides what happens:

| Policy                | A write against a stale version                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `"refuse"` (default)  | raises `VersionConflictError`, naming the parameter and both versions, and changes nothing        |
| `"merge"`             | folds the stale feedback onto the current value                                                  |
| `"last_writer_wins"`  | rebuilds from the version the writer read and drops the write made in between                     |

A version the host never had is refused under every policy, and so is a version from before a [restore](#saving-and-restoring), whose value the host no longer holds. A write that leaves the value unchanged is not a new version.

The optimizer hands a version only to a host that has `load(name) -> (value, version)`, which is the `VersionedParameterHost` protocol in `ai_functions.types.graph`. Memory backends don't have `load`, so they're called as before.

## Sharing a parameter across stages

Several binders can share one host, so the stages of a pipeline learn one parameter together:

```python
local = Local()
research_step = Bind(research, local)
report_step = Bind(write, local)
```

A parameter declared twice with the same initial value, reducer and policy is one parameter. Declared twice with a different initial value, reducer or policy, it's refused with `ParamWiringError` naming both declarations.

## Saving and restoring

`local.snapshot()` returns one frozen `Param` record per parameter, holding the value, the version, the reducer's name and the conflict policy. A `Param` is a Pydantic model, so it round-trips as JSON. To keep what a program learned between runs, declare the parameters again in the new process (with `Bind` or `Local.declare`), then `restore` the stored records:

```python
saved = {name: param.model_dump(mode="json") for name, param in bound.local.snapshot().items()}

fresh = Bind(reply)
fresh.local.restore({name: Param.model_validate(record) for name, record in saved.items()})
```

`restore` checks every record before writing any. It refuses a name that isn't declared, a reducer or policy that disagrees with the declaration, a value that doesn't validate as the declared type, and a parameter that this host has already written. After a restore, `history` starts at the restored version.

## Search optimizers: `components` and `apply`

An optimizer that searches over candidates instead of following gradients can read and write the whole host at once. `local.components()` returns every parameter as one `dict[str, str]` candidate, with text as is and any other value as canonical JSON. `local.apply(candidate)` writes a candidate back and bumps the version of each parameter whose value changed. It refuses an unknown name, or a text that doesn't parse as its parameter's type, before anything changes. `bound.components()` is the same mapping for a binder's host.

## Local parameters or recalled memory

Both `Trainable` and [`Recalled`](methods.md#memory-as-a-call-argument-recall) make a parameter something the optimizer can update. Choose by where the value should live:

| | `Trainable` with `Bind` | `Recalled` with `Recall` |
|---|---|---|
| Where the value lives | a `Local` host in the process | a `MemoryBackend` field |
| Initial value | the signature's default | the backend's stored value |
| How feedback is applied | the parameter's reducer, with no model call | the backend's consolidation |
| Versions and conflict policy | yes | no |
| Kept between runs | with `snapshot` and `restore` | by the backend |

Use `Trainable` for a value that belongs to the code that uses it, such as a prompt fragment, a threshold or a weight vector. Use `Recalled` for knowledge shared across processes or users, or retrieved by search.

## Errors

Every error in the module subclasses `ParamsError`, an `AIFunctionError`.

| Error                  | Raised when                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------|
| `ParamWiringError`     | a marker, declaration, bind, `restore` or `apply` is refused, before any model call or write         |
| `VersionConflictError` | a write targets a version the parameter has moved past under `"refuse"`, or a version the host doesn't hold |
| `ReducerError`         | a reducer can't combine the value with a piece of feedback, including a `- update:` or `- delete:` line |

## Examples

| Example                           | Shows                                                                                                   |
|-----------------------------------|---------------------------------------------------------------------------------------------------------|
| `params/trainable_guidelines.py`  | A support reply whose guidelines learn a house rule from one piece of feedback, then persist with `snapshot` and `restore` |

Run it from the `examples/` folder with `uv run params/trainable_guidelines.py`. The [examples guide](../examples/README.md#params) lists all commands and setup requirements.
