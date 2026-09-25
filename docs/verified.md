# Verified AI Functions

Suppose an employee writes to HR: "Could I take next Thursday and Friday off?" An AI Function can read the message, look up the employee's records and answer "approved". But is that the *right* answer? It is only if the agent read the request correctly, looked up the right records, and applied every relevant rule of the leave policy. A post-condition could check this only by doing the whole derivation again.

The `ai_functions.experimental.verified` module has the agent show its work instead. You write the policy in [Lean](https://lean-lang.org), a language for machine-checked logic, or in a restricted subset of Python. The agent returns its answer together with a proof that the answer follows from the policy, and the library returns the answer only after Lean has checked that proof. Each answer comes with a **certificate** listing what the proof relies on: the records returned by your tools, and the agent's *judgments*, such as how it read the message. A proof cannot check a judgment, so the certificate records it for a human to review.

The module offers two decorators, which differ in *when* the agent runs:

- **`verified.ai_function`** runs the agent on every call, like an ordinary AI Function. Each result comes with a proof about that specific result, and the agent can use tools to gather the facts the proof relies on.
- **`verified.ai_compile`** runs the agent once, to write an implementation together with a proof that it is correct for *every* input. After that, calls run compiled native code and make no model calls at all.

This tutorial starts from a leave-request example, introducing `verified.ai_function`, contracts, tools, judgments and certificates as they appear in it, then extends it to open questions about the policy. A short Python DSL example shows how to write contracts in Python, followed by a complete `verified.ai_compile` example. The remaining sections cover the components in detail and explain how to choose an approach.

Note: this module is experimental; its API may change in future releases.

## Contents

- [Getting started](#getting-started)
- [A first example: deciding leave requests](#a-first-example-deciding-leave-requests)
- [Open questions about a policy](#open-questions-about-a-policy)
- [Python DSL for policies](#python-dsl-for-policies)
- [Verified compilation](#verified-compilation)
- [Verified tools](#verified-tools)
- [Judgments](#judgments)
- [Certificates](#certificates)
- [Contracts and Lean projects](#contracts-and-lean-projects)
- [Contracts in Python](#contracts-in-python)
- [Helping the agent prove](#helping-the-agent-prove)
- [Compilation details](#compilation-details)
- [Choosing an approach](#choosing-an-approach)
- [Examples](#examples)

## Getting started

Install the library and configure model credentials as described in the main tutorial's [Getting started](tutorial.md#getting-started). The library manages Lean for you: on first use, it finds a matching toolchain or downloads one, then builds your Lean project. This makes the first call noticeably slower than later ones. Verified compilation uses `leanc` from that Lean installation; on macOS, install the Xcode Command Line Tools to provide the system SDK. See [types, platforms, and limits](#types-platforms-and-limits) for the native execution requirements. Two environment variables control Lean setup:

- `AI_FUNCTIONS_LEAN_CACHE_DIR` sets where the toolchain and project builds are cached (by default, the platform's AI Functions cache directory).
- `AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE` selects where Lean comes from: `auto` (the default) uses a matching installed Lean if there is one and downloads otherwise, `system` requires an installed Lean and never downloads, and `managed` uses only the library's own cache.

The examples live in `examples/verified/`, with their Lean theories in `examples/verified/lean/`, and run like the other examples (see the [README](../README.md#examples)). Set `STRANDS_TOOL_CONSOLE_MODE=enabled` to watch every Lean request the agent makes.

## A first example: deciding leave requests

The HR example (`examples/verified/hr.py`) decides employees' leave requests according to a company policy written in Lean:

```python
from pathlib import Path

from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

# The Lean project holding the leave policy, and its declarations.
project = LeanProject(Path(__file__).parent / "lean", imports=["HR"])
HR = project.symbols.HR


# A tool that looks up a record; its result becomes a fact the proof can use.
@verified.tool(HR.vacationDaysLeft)
def vacation_days_left(employee: str) -> int:
    """Vacation days the employee has left this year."""
    return RECORDS[employee]["vacation_left"]

# ... `tenure_months` and `sick_days_taken` are defined the same way.


@verified.ai_function(
    contract=HR.DecisionCorrect,                                # what a correct decision is
    tools=[tenure_months, vacation_days_left, sick_days_taken],   # where the records come from
    judgments=[HR.requestedLeave],                              # what the agent must read from the message
)
def decide(employee: str, message: str) -> bool:
    """Decide the leave request employee {employee} sent: "{message}"

    First judge the requested leave (its kind and number of working days) with
    lean_judge on HR.requestedLeave at H.message. Then decide it, with a proof.
    """


# Returns the decision, and the certificate of what it relies on.
approved, certificate = await decide.with_certificate(
    "sam", "Could I take next Thursday and Friday off? We're planning a small family trip."
)
```

`decide` is called like any AI Function (`await decide(employee, message)`, or `decide.run_sync(...)` in scripts); `with_certificate` also returns the certificate. For two requests, the example prints the decision and what it rests on:

```
sam: Could I take next Thursday and Friday off? We're planning a small family trip.
Decision: approved
  request read as: HR.Leave.vacation 2  (The message asks for next Thursday and Friday off for a family trip, i.e. two working days of vacation leave.)
  record: HR.tenureMonths "sam" = 14
  record: HR.vacationDaysLeft "sam" = 9

jo: I have the flu and must stay home from today, Tuesday, through Friday.
Decision: denied
  request read as: HR.Leave.sick 4  (The message reports illness (flu) and absence from Tuesday through Friday inclusive, which is 4 working days of sick leave.)
  record: HR.sickDaysTaken "jo" = 8
```

Each decision is proved to follow from the policy. The certificate lists what the proof relies on:

- **Tool results.** The proof trusts the records the tools returned. Each one is listed with the call that produced it, so it can be audited.
- **Judgments.** The agent read Sam's message as `HR.Leave.vacation 2`, and the certificate records that reading with the agent's justification, for a human to review.

The policy lives in `lean/HR.lean`. It defines when a request is approved, in terms of the employee's records:

```lean
namespace HR

/-- A leave of some kind, lasting some number of working days. -/
inductive Leave where
  | vacation (days : Nat)
  | sick (days : Nat)

/-- A request is approved when:
* vacation: the employee has worked here at least 6 months and has the days left;
* sick leave: the year's sick days stay within 10. -/
def isApproved (employee : String) : Leave → Prop
  | .vacation days => 6 ≤ tenureMonths employee ∧ days ≤ vacationDaysLeft employee
  | .sick days => sickDaysTaken employee + days ≤ 10

/-- **Contract.** The decision is `true` exactly when the requested leave is approved. -/
def DecisionCorrect (approved : Bool) (employee message : String) : Prop :=
  approved = true ↔ isApproved employee (requestedLeave message)

end HR
```

### Open questions about a policy

The first example restricts the agent to a fixed kind of question: which leave is requested, and is it approved. Employees ask many others ("Do I have enough vacation left for a full week off?"), and the policy should not need a function for each. Instead, the question itself can be the judgment: the agent restates it as a proposition in the language of the policy. The theory declares `asks employee message : Prop`, and the contract requires the answer to be `true` exactly when that proposition holds. `examples/verified/hr_generic_question.py` does this, with a richer policy (`lean/LeavePolicy.lean`) that states only rules, such as how vacation days are earned and when requests made together are approved. The same function answers:

| Question | Formalized as | Answer |
|---|---|---|
| Can I take Thursday and Friday off, and will I be paid? | `isLeaveApproved "sam" (.vacation 2) ∧ isPaidLeave "sam" (.vacation 2)` | yes |
| Do I have enough vacation left for a full week off? | `isLeaveApproved "sam" (.vacation 5)` | yes |
| Can I take both a day off and a full week off with my current vacation balance? | `areApproved "sam" 0 [.vacation 1, .vacation 5]` | yes |
| Could I take two weeks off right now? | `isLeaveApproved "sam" (.vacation 10)` | no |

This flexibility moves more of the answer into the judgment. In the first example, the reviewer checks a leave kind and a day count against the message. Here, the reviewer checks that a whole proposition means what the employee asked: the answer is proved, but it is the answer to the question as formalized. A formalization that reads the question differently yields a correct proof of the wrong question.

## Python DSL for policies

Policies can also be written in a restricted Python DSL. This small contract states when a decision follows the vacation policy: approve exactly when the employee has at least six months of tenure and enough days left.

```python
from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject()


@project.proposition
def vacation_decision(approved: bool, tenure_months: int, days: int, days_left: int) -> bool:
    assert approved == (tenure_months >= 6 and 0 <= days <= days_left)


@verified.ai_function(contract=vacation_decision)
def decide_vacation(tenure_months: int, days: int, days_left: int) -> bool:
    """Decide a request for {days} days with {days_left} days left
    and {tenure_months} months of tenure, following the vacation policy.
    """


# The contract also runs as Python, checking a proposed decision.
print(vacation_decision(True, 14, 2, 9))  # True: this approval is correct
print(vacation_decision(True, 3, 2, 9))   # False: this approval violates the policy
```

The decorator registers a Lean proposition named `vacation_decision`, available as `project.symbols.vacation_decision` for use in Lean contracts and proofs. Its first parameter, `approved`, is the proposed result; the remaining parameters match `decide_vacation`'s inputs. The same definition remains callable in Python: it returns `False` if an assertion fails and `True` if they all hold. Define it in a `.py` file, since the decorator reads its source.

Python DSL contracts work with both `verified.ai_function` and `verified.ai_compile`. `@project.function` similarly translates a function with a single `return` expression into a Lean definition while keeping it callable in Python. The [reference below](#contracts-in-python) covers the supported syntax and types.

## Verified compilation

A verified AI Function pays for an agent, and a proof, on every call. For a pure computation with a precise contract, `verified.ai_compile` can generate an implementation proved to satisfy that contract for every input, compile it, and run it as native code.

The payout example (`examples/verified/payout.py`) computes a transfer: send as much money as possible while the balance still covers a fixed fee and a percentage fee rounded up to a whole cent, within a payout limit. The right formula is not obvious: the fee depends on the amount being computed, and rounding can make one more cent unaffordable. With a $100.00 balance, a $0.30 fixed fee and a 2.90% rate, $96.89 fits exactly and $96.90 overdraws the balance. Rather than deriving the formula, you state what a correct answer is (`best_payout`) and let the agent find an implementation:

```python
from ai_functions.experimental import verified
from ai_functions.experimental.verified.dsl import assume, every, implies
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject()


@project.proposition
def valid_inputs(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


# Specification of how fees are computed.
@project.function
def fees(payout: int, fixed_fee_cents: int, fee_bps: int) -> int:
    """The fixed fee plus the percentage fee rounded up to a cent; nothing for a zero payout."""
    return fixed_fee_cents + (payout * fee_bps + 9_999) // 10_000 if payout > 0 else 0


# The payout stays within the limit and leaves enough funds to cover fees.
@project.proposition
def affordable(payout: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    assert 0 <= payout <= payout_limit_cents
    assert payout + fees(payout, fixed_fee_cents, fee_bps) <= balance_cents


# The contract: on valid inputs, the result is the largest affordable payout.
@project.proposition
def best_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> bool:
    assume(valid_inputs(balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents))
    assert affordable(result, balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents)
    assert all(
        implies(affordable(p, balance_cents, fixed_fee_cents, fee_bps, payout_limit_cents), p <= result)
        for p in every(int)
    )


# Define the function through both an informal English description and a formal contract.
@verified.ai_compile(contract=best_payout, max_attempts=5)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit.

    For a positive payout p, charge a fixed fee plus ceil(p * fee_bps / 10000)
    cents. A zero payout incurs no fee. Include fees in the balance constraint.
    """


# Generate and compile an implementation.
# This call is optional: the function also compiles on first use.
await max_payout.compile()

# Call the compiled function.
print(await max_payout(10_000, 30, 290, 20_000))  # 9689
print(max_payout.run_sync(31, 30, 290, 1_000))    # 0
```

The last `assert` in `best_payout` makes the result the *largest* affordable payout: without it, returning `0` for every valid input would satisfy the contract. `assume(valid_inputs(...))` limits the guarantee to valid inputs, and `every(int)` states that no affordable integer payout is larger.

The body of `max_payout` is left empty, and the docstring is the agent's guidance. The agent writes an implementation in Lean and proves that it returns the largest affordable payout for every valid input, including zero balances and fees larger than the funds. Once the proof is accepted, the implementation is compiled to a native library, and every later call runs it directly, with no model call. The docstring can also contain additional informal requirements. The agent will try to satisfy them, but the proof establishes only the formal contract.

## Verified tools

Agents often need results from tool calls to complete a task. To use those results in a proof, the theory must describe what they mean. The simplest way is to treat a tool as an observation of an opaque function. For example, this tool:

```python
@verified.tool(HR.vacationDaysLeft)
def vacation_days_left(employee: str) -> int:
    """Vacation days the employee has left this year."""
    return RECORDS[employee]["vacation_left"]
```

observes a declaration in the `HR` namespace:

```lean
/-- Vacation days the employee has left this year. Looked up with a tool. -/
opaque vacationDaysLeft : (employee : String) → Nat
```

This declaration gives Lean a name and type for the record without giving it a value. An **observation tool** supplies the value when called. `verified.tool(symbol)` connects the Lean declaration to the Python tool.

The agent calls the tool like any other. In addition to returning a result, the call records a fact the proof can use:

```lean
/-- The tool observed that Sam has 9 vacation days left. -/
axiom H.vacation_days_left1_spec : HR.vacationDaysLeft "sam" = 9
```

The fact is trusted as an assumption; Lean does not verify the underlying record. The final certificate reports every tool result the proof depends on.

### Tools that return properties of their results

In the example above, the tool returns a value. Some tools can also vouch for properties of that value. In `mis.py`, the tool is an exact solver for the maximum independent set of small graphs. Its answer is useful to the proof because the tool also guarantees that the returned size is optimal:

```python
from ai_functions.experimental.verified.function import Certified
from ai_functions.experimental.verified.lean.types import encode


@verified.tool(MIS.misSolver, stem="mis")
def mis_solver(graph: Graph) -> Certified:
    """Exact maximum-independent-set solver for graphs of at most 15 vertices."""
    ...  # refuse graphs above the cap, then solve
    size = exact_mis(verts, edges)
    lean_graph = encode(graph, MIS.misSolver.info.parameters[0].type)
    return Certified(size, guarantees=lambda value: f"MIS.MaxIndependentSize {lean_graph} {value}")
```

`guarantees=` is a Lean proposition, a list of them, or a function that receives the returned value as Lean source and returns them. Each guarantee becomes one more fact (`H.mis1_contract1`), and is trusted as stated: it appears in the certificate as an assumption, just like the value itself. Only add guarantees your tool actually provides. The final `Certificate` records which of those claims the checked proof used.

## Judgments

Some steps need interpretation rather than lookup: what an employee's message asks, which clause of a contract a complaint refers to, what a free-text form means. An opaque function lets the theory describe how to use an interpretation without supplying it. In the HR theory, the decision contract applies the policy to the leave requested in the message:

```lean
/-- The kind of leave and number of working days requested in the message. -/
opaque requestedLeave : (message : String) → Leave

/-- Approve exactly when the policy allows the interpreted request. -/
def DecisionCorrect (approved : Bool) (employee message : String) : Prop :=
  approved = true ↔ isApproved employee (requestedLeave message)
```

Proofs can reason symbolically about `requestedLeave` without knowing its value. To supply a concrete interpretation of a message, the agent calls `lean_judge` with the symbol, its arguments, the interpreted value, and a justification. The tool records that reading as a new fact:

```lean
/-- The agent interprets the message as a request for two days of vacation. -/
axiom J.requestedLeave1 :
  HR.requestedLeave "Could I take next Thursday and Friday off?" = HR.Leave.vacation 2
```

If the proof uses this fact, the certificate identifies it as a model judgment and includes the model's justification.

A judged symbol must have explicit parameters of type `Int`, `Nat`, `Bool`, `String`, or lists and products of those types. Its result may be one of those types, a supported concrete type from your theory such as `HR.Leave`, or a proposition, as in [open questions about a policy](#open-questions-about-a-policy). The result type must be fixed: polymorphic signatures and result types that depend on argument values are not supported. With a proposition, the agent states the question in terms your policy understands, and a reviewer checks that it captures the original question.

A few rules keep judgments reviewable:

- Only the symbols you list in `@verified.ai_function(..., judgments=[...])` can be judged.
- Judgment arguments must reduce to closed values, such as a string literal or the input `H.message`.
- A judgment cannot be revised later in the same call: the agent must build its proof on the first one.
- Every judgment the proof uses is in the certificate. Judgments recorded but not used are not.

A proof that uses a judgment is only as good as the judgment, so keep judgments narrow: use them for the interpretation step, and keep everything else (policy, arithmetic, data) in the theory and in observations. Tell the agent how to interpret, in the docstring, as the HR example does ("its kind and number of working days") or in the comments of the theory. Inputs that are not a matter of interpretation, such as who is asking, belong in the function's arguments, not in the judgment.

## Certificates

Every result of `verified.ai_function` has a checked proof. Use `with_certificate` to return its `Certificate` alongside the answer:

```python
approved, certificate = await decide.with_certificate("sam", message)
certificate.summary()  # print the answer, the goal, and the facts the proof used
certificate.write("data/hr.lean")  # save the complete Lean file
```

A certificate has:

- `answer`: the proved result, and `goal`: the Lean statement that was proved about it.
- `proof`: the agent's proof, and `artifact`: a complete Lean file (the inputs, the recorded facts, the agent's definitions and lemmas, and the final theorem) that you can check again in the same project.
- `axioms`: every assumption the proof depends on. These are Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`) plus the recorded facts the proof actually used. Any entry beyond the standard three is something the result is conditional on.
- `observations`: every tool result and judgment recorded during the call, by name, and `used`: the subset the proof depends on. Each entry is an `Observation` with its `kind` (`"tool"` or `"judgment"`), `symbol`, `arguments` and `value` as Lean text, and `origin`: the tool call, or the judgment's justification.

`used` is what a reviewer reads. The HR example prints it this way:

```python
for fact in certificate.used.values():
    if fact.kind == "judgment":
        print(f"  request read as: {fact.value}  ({fact.origin})")
    else:
        print(f"  record: {fact.symbol} {' '.join(fact.arguments)} = {fact.value}")
```

For an audit trail, store the artifact with the result. Lean can check the saved file again using the same project and toolchain; `certificate.project_fingerprint` identifies the project used for the proof.

### What the guarantee means

Before returning a result, the library rechecks the proof from scratch and confirms that the returned value is exactly the one proved. The agent cannot add arbitrary assumptions or leave a proof step unfinished. Beyond Lean's standard logical axioms, the proof may rely on the observations and judgments authorized by your tools and `judgments=` declarations; the certificate lists the ones it uses.

The guarantee is conditional on those facts and on the contract you wrote. In the HR example, it establishes that the decision follows from the policy, the observed records, and the stated interpretation. Reviewing the records' source, the interpretation, and the policy itself remains your responsibility.

## Contracts and Lean projects

Every contract lives in a `LeanProject`, which fixes the Lean library available to the contract and to the agent's proofs. There are three ways to create one:

```python
from ai_functions.experimental.verified.lean import LeanProject

# Core Lean only.
project = LeanProject()

# An existing Lake project (a folder with lakefile.toml or lakefile.lean and lean-toolchain),
# importing the listed modules.
project = LeanProject("path/to/lake/project", imports=["HR"])

# A Lake project shipped inside a Python package, as a package resource folder.
project = LeanProject.from_package("my_app", "lean", imports=["HR"])
```

The source folder of a Lake project is never modified: builds happen in the cache. Only what the project imports is available; in particular, if your proofs need [Mathlib](https://leanprover-community.github.io/mathlib4_docs/), add it as a dependency of your Lake project. A Lake project is the natural home for a theory like `HR.lean`, with doc comments, lemmas and a lakefile, which you can build and test with Lean's own tools. For a few lines, `project.add` registers Lean source inline instead:

```python
project = LeanProject()
project.add("def Contract (result x : Int) : Prop := result = x + 1")
# You can now use the symbol.
project.symbols.Contract
```

`project.symbols` names any declaration of the project, whether added inline or imported: `project.symbols.HR.DecisionCorrect` and `project.symbols["HR.DecisionCorrect"]` are the same declaration, and a namespace can be kept in a variable, as in `HR = project.symbols.HR`. Declarations must be registered before the project is first used, which in practice means at module level.

Python DSL definitions, imported Lean modules, and `project.add` blocks can share a project. Inline Lean can refer to DSL definitions registered before it, including `vacation_decision` from the [DSL example](#python-dsl-for-policies). A verified function's contract, observation tools, and judgments must all belong to the same `LeanProject` instance. The toolchain and build-cache settings are described in [Getting started](#getting-started).

### Contract signatures

A contract is a Lean proposition with explicit parameters: the proposed **result first**, followed by the inputs it constrains. For `verified.ai_function`, inputs are matched by name. For example, `HR.DecisionCorrect (approved : Bool) (employee message : String)` fits `def decide(employee: str, message: str) -> bool`. For `verified.ai_compile`, the input types and order must match the function's signature.

A Python DSL proposition's `-> bool` annotation describes whether its conditions hold. When the proposition is used as a contract, its first parameter describes the verified function's result. In `vacation_decision`, `approved` is that result, and the assertion equates it with the policy's approval condition. This rules out both incorrect approvals and incorrect denials.

## Contracts in Python

The [Python DSL example](#python-dsl-for-policies) introduced a policy written as assertions. Two decorators of `LeanProject` translate Python definitions into Lean:

- `@project.function` translates a function whose body is a single `return` expression into a Lean definition.
- `@project.proposition` translates either a single `return` expression or a *proposition block* into a Lean proposition. In a block, each `assert` is one conjunct, and local assignments, `if`/`elif`/`else`, and `for` loops are allowed. Every proposition has a Python `bool` return annotation; its Lean result is `Prop`.

Define these functions in a `.py` file rather than an interactive session, since the library reads their source. Supported value types are `int` (Lean `Int`), `bool` (Lean `Bool`), and `list[int]` (Lean `List Int`), and every parameter and the result must be annotated. Expressions support arithmetic (`+`, `-`, `*`, and `//` and `%` with Python's rounding), comparisons, `and`/`or`/`not`, conditional expressions, `len`, `sum`, `range` with one or two bounds, `all`/`any` over ranges and lists, and calls to other definitions of the same project, defined above the functions that use them. Anything outside this subset is reported when the decorator runs, not at call time.

Three helpers from `ai_functions.experimental.verified.dsl` express logical conditions:

- `assume(condition)` makes the rest of a proposition block conditional on `condition`. It is allowed only at the top level of the block; use `if condition:` to guard assertions inside a branch or loop.
- `implies(a, b)` states that `a` implies `b`.
- `every(int)` denotes all integers for an unbounded quantifier, as in `all(... for p in every(int))`. It is useful for specifications such as the payout's optimality requirement.

The decorated definitions remain callable in Python. A function evaluates its return expression; a proposition with a single `return` evaluates that expression too. A proposition block returns `False` when an `assert` fails and `True` when an `assume` fails or the block runs to its end. Reaching `every(...)` during Python evaluation raises `NotComputable`, since Python cannot enumerate all integers. In particular, a false assumption makes the remaining claim hold vacuously; it does not reject the inputs.

### Calling Lean declarations from Python contracts

To call a declaration of a Lake project from a Python contract, adapt it with `@project.external`. The decorated function gives its signature and, optionally, a Python implementation used when the contract runs as Python; the signature is checked against the Lean declaration when the project is first used:

```python
# `Rates.feeCents : Int → Int → Int` is defined in the project's Lake sources.
@project.external(project.symbols.Rates.feeCents)
def fee_cents(payout: int, fee_bps: int) -> int:
    return (payout * fee_bps + 9_999) // 10_000
```

A Python contract can be passed as `contract=` to either decorator, just like a Lean symbol.

## Helping the agent prove

The HR proofs are short: rewrite with the recorded facts, and the rest is arithmetic. Other contracts need real work. In `audit.py`, nothing in the contract computes, so the agent must write a per-row fee check, a filter and a balance scan, and prove each equivalent to the definitions (`FeeOk`, `Solvent`), including a loop invariant for the scan. In `mis.py`, it must find a reduction of a graph too large for the solver and prove each step sound. A few things make such tasks feasible.

**The agent reads your theory.** The agent is shown the Lean source of the modules that define your contract, tools and judgments, doc comments included. Write the theory for two readers, the reviewer and the agent: say what each opaque function returns, and how definitions that do not compute should be handled. Lemmas in the theory are the most effective help. `lean/Access.lean` ships `hasAccess_delegated` and `not_hasAccess_succ`, which reduce a delegation chain to one hop at a time, and `allow_of_policy` and `deny_of_policy`, one for each way the decision can go. A lemma you prove once is one the agent does not have to find on every call.

**Name things for the agent.** In docstrings, refer to the function's inputs as the agent sees them: each input is a definition `H.<parameter>` (`H.employee`, `H.message`), and facts are named as described above (`H.vacation_days_left1_spec`, `J.requestedLeave1`). The agent adds its own definitions and lemmas under the prefix `A.`.

After a failed submission, Lean's error is shown to the agent, which tries again; `max_attempts=` bounds the retries. For `verified.ai_function`, `timeout=` limits individual Lean operations. Exhausting the retry budget raises `ContractNotProved`; the function does not return an unproved answer.

## Compilation details

The contract's parameters after the result correspond to the function's parameters in order. The signature is checked against the contract before any model call, and a mismatch raises `ContractError`. Contracts can come from a Lake project as well: `lower_bound.py` compiles a sorted-list search from a Lean contract that says where the key belongs, not how to find it:

```lean
def Contract (position : Int) (values : List Int) (key : Int) : Prop :=
  values.Pairwise (· ≤ ·) →
    0 ≤ position ∧ position ≤ Int.ofNat values.length ∧
    (∀ value ∈ values.take position.toNat, value < key) ∧
    (∀ value ∈ values.drop position.toNat, key ≤ value)
```

```python
project = LeanProject(Path(__file__).parent / "lean", imports=["Lookup"])


@verified.ai_compile(contract=project.symbols.Lookup.Contract, max_attempts=5)
def lower_bound(values: list[int], key: int) -> int:
    """Return the first insertion index that preserves sorted order, including duplicates and missing keys."""
```

### Compiling and calling

`compile()` (or `compile_sync()`) runs the agent and checks the proof. Call it at startup when the function should be ready before it handles requests; otherwise, the first call compiles it. After that, `await max_payout(...)` and `max_payout.run_sync(...)` call the native code, and argument types are checked on every call.

A compiled function is cached across processes, keyed by the contract, the project, the signature, the docstring, the Lean toolchain and the library version. A cache hit makes no model call, so a service restart, or a second process using the same function, reuses the verified implementation; changing any of these compiles a new one. `max_payout.artifact_dir` points to the compiled function's files, including `Verified.lean`, which holds the specification, the implementation and its proof. Treat these files as read-only.

`verified.ai_compile` takes these options:

- `contract=`: the result-first contract (required).
- `model=`: the model that writes the implementation and proof; by default, a Claude model on Amazon Bedrock.
- `max_attempts=`: the number of retries after the first rejected submission (default 10).
- `compile_timeout=`: the time limit, in seconds, for each compilation and proof-checking step.
- `cache_dir=`: where compiled functions are stored.

When no implementation is accepted after the initial attempt and the allowed retries, compilation raises `SynthesisError`; a failure to build the Lean code or the native library raises `CompilerError`. No unverified implementation is ever returned or used as a fallback. `ContractError`, `SynthesisError`, and `CompilerError` are importable from `ai_functions.experimental.verified`.

### Types, platforms, and limits

Native functions take and return `int`, `bool`, `float` and `list[int]`. Integers keep Python's arbitrary precision (`max_payout(2**100, ...)` works), and floats follow Lean's IEEE 754 double-precision semantics, so a float contract must say what happens with NaN and infinities. Verified compilation requires CPython 3.12 or later (the standard build with the GIL) on macOS or Linux and uses the Lean installation's `leanc`; on macOS, the Xcode Command Line Tools provide the required system SDK.

Keep two limits in mind:

- **An assumption is not an input check.** `assume(valid_inputs(...))` means the proof says nothing about inputs that violate it, and the compiled function does not reject them: `max_payout(-5, 30, 290, 1_000)` returns *some* integer, with no guarantee. If callers can pass such inputs, check them before the call.
- **Imported modules are for contracts and proofs.** The implementation can use Lean's standard library and definitions registered inline with `@project.function` or `project.add`, but not definitions imported from a Lake project; proofs can use both.

## Choosing an approach

The three options trade flexibility for strength of guarantee and cost:

- **`ai_function` with post-conditions** is the default. It works with any return type, including Pydantic models and native Python objects, and costs one agent run per call plus a retry for each failed check. Its guarantee is exactly what your post-conditions check. Use it when checking a result is easier than producing it.
- **`verified.ai_function`** costs one agent run per call, with additional time spent finding and checking a proof, plus the work of writing the contract in Python or Lean. It returns `int`, `bool`, `str`, lists and tuples of those, or a `LeanTerm`. Use it when you need a checked proof that a policy was applied to data from tools, when an interpretation step should be recorded as an explicit assumption, or when you need a certificate as an audit trail.
- **`verified.ai_compile`** costs model calls once per contract and signature, and nothing afterwards: calls run native code with no model call and negligible latency. It supports `int`, `bool`, `float` and `list[int]`, and cannot use tools or judgments, since its implementation must work for every input without asking anyone. Use it for pure computations over these types that are called often, where you can state what a correct answer is but would rather not derive and prove the algorithm yourself.

## Examples

The examples in `examples/verified/` are complete and runnable, with their Lean theories in `examples/verified/lean/`:

- `hr.py`: deciding leave requests, with the requested leave as a judgment and the decision proved from the policy and the records.
- `hr_generic_question.py`: answering open yes/no questions about a richer leave policy, with the question's formalization as the judgment.
- `access.py`: a data-export decision proved from complete directory and grant records; the allow and the deny are both proved, the deny because the only live delegation chain is one hop too long.
- `audit.py`: auditing a paginated settlement ledger, where the agent writes the checking programs and proves them equivalent to a specification that does not compute; the flag list is sound and complete, and the balance never goes negative.
- `mis.py`: a maximum independent set of a graph too large for a size-capped solver, whose results carry guarantees via `Certified`; the agent reduces the graph and uses the solver on the pieces.
- `payout.py`: a compiled function computing the largest affordable payout, with its contract written in Python.
- `lower_bound.py`: a compiled sorted-list search, from a contract written in Lean.

Run them like the other examples, from the `examples/` folder (for example `uv run verified/hr.py`), and set `STRANDS_TOOL_CONSOLE_MODE=enabled` to see every Lean request the agent makes.
