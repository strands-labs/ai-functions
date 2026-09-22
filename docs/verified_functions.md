# Verified native functions

Use `verified_ai_compile` for pure calculations governed by precise policies:
which outputs are allowed, which limits must hold, and what makes an answer
optimal. Write Python contracts that check a proposed result. The system
synthesizes an implementation and checks a proof that it satisfies those
contracts for every valid input. Subsequent calls reuse the compiled native
function without contacting a model.

A useful contract should be easier to review than the code that finds the
answer. Repeating a reference implementation inside an assertion provides little
benefit. The example below specifies feasibility and maximality; it does not
calculate the result.

## Example: the largest payout that fits after fees

A payout service has a balance, a fixed processing fee, a percentage fee, and a
limit on the amount it can send. All amounts are integer cents. The percentage
fee is charged on the payout and rounded **up** to a whole cent. A zero payout
incurs no fee.

The policy has three requirements:

1. The payout is nonnegative and respects the supplied balance and payout limit.
2. A positive payout leaves enough money to pay both fees.
3. Unless the payout limit is reached, increasing the payout by one cent would
   exceed the balance.

The third requirement matters: returning zero for every input would satisfy
many safety-only contracts. Here, an unnecessarily small payout is also wrong.
Because fees increase monotonically with the payout, rejecting the next cent
establishes that no larger permitted payout fits.

```python
from ai_functions.experimental.verified_compile import verified_ai_compile


def payout_inputs(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    assert balance_cents >= 0
    assert fixed_fee_cents >= 0
    assert 0 <= fee_bps <= 10_000
    assert payout_limit_cents >= 0


def maximum_safe_payout(result: int, balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int):
    assert 0 <= result <= payout_limit_cents
    assert result <= balance_cents
    if result > 0:
        fee_budget = balance_cents - result - fixed_fee_cents
        assert result * fee_bps <= fee_budget * 10_000
    if result < payout_limit_cents:
        next_payout = result + 1
        next_fee_budget = balance_cents - next_payout - fixed_fee_cents
        assert next_payout * fee_bps > next_fee_budget * 10_000


@verified_ai_compile(
    pre_conditions=[payout_inputs],
    post_conditions=[maximum_safe_payout],
    max_attempts=5,
)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit.

    For a positive payout p, charge a fixed fee plus ceil(p * fee_bps / 10000)
    cents. A zero payout incurs no fee. Include fees in the balance constraint.
    """


payout = max_payout.run_sync(
    balance_cents=10_000,
    fixed_fee_cents=30,
    fee_bps=290,               # 2.90% of the payout, rounded up.
    payout_limit_cents=20_000,
)
assert payout == 9_689
```

The fee checks use multiplication and inequalities. If a proposed payout has
`fee_budget` whole cents available, its rounded-up percentage fee fits exactly
when `payout * fee_bps <= fee_budget * 10_000`. This avoids floating-point
rounding in the contract. It does not tell the synthesizer how to choose the
payout; that requires deriving a formula or a search procedure and proving it
meets both requirements.

For a $100.00 balance, a $0.30 fixed fee, and a 2.90% percentage fee, the answer is
$96.89. The percentage fee rounds up to $2.81, so the payout and fees use exactly
$100.00. Sending $96.90 would make the percentage fee $2.82 and the total $100.02.
A superficially plausible extra cent would violate the policy.

The same contracts cover the less obvious boundaries:

| Balance | Payout limit | Payout | Total fees | Reason |
| --- | --- | --- | --- | --- |
| $100.00 | $200.00 | $96.89 | $3.11 | Largest amount that fits after fees |
| $100.00 | $50.00 | $50.00 | $1.75 | The payout limit binds |
| $1.00 | $10.00 | $0.68 | $0.32 | Percentage-fee rounding still matters |
| $0.31 | $10.00 | $0.00 | $0.00 | Even a one-cent payout would cost $0.32 |
| $0.00 | $10.00 | $0.00 | $0.00 | No funds; no payout or fee |

All rows use the same $0.30 fixed fee and 2.90% rate. The proof covers every
combination admitted by `payout_inputs`, including zero rates, zero limits,
fees larger than the balance, and integers beyond machine-word precision.
These examples illustrate the policy; they are not the scope of verification.

The [runnable payout example](../examples/verified_payout.py) generates the
implementation and proof and prints these cases. The value of the feature is a
checked relationship between the implementation and a reviewable policy. The
contracts can be longer than a particular implementation; the system also
constructs and checks the proof that the implementation meets them.

This guarantee covers the calculation for the supplied values. The service
still owns balance reads, fee configuration, and atomic payment execution. A
proof cannot correct a missing policy rule or stale input data.

## Install and run

Use standard CPython 3.12–3.14 on macOS 15+ or Linux, on x86-64 or ARM64:

```bash
pip install strands-ai-functions
```

The first compilation downloads Lean if needed and builds the native runtime
locally. Python development headers and C build tools are required; on macOS,
install the Xcode Command Line Tools.

To set up Lean ahead of time, using the payout contracts above:

```python
from ai_functions.experimental.lean import LeanConfig

lean = LeanConfig()
lean.setup()

@verified_ai_compile(
    pre_conditions=[payout_inputs],
    post_conditions=[maximum_safe_payout],
    lean_config=lean,
    offline=True,  # Disable tool downloads; synthesis still uses the model.
)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit."""

max_payout.compile_sync()
```

By default, synthesis uses Claude through Amazon Bedrock. Run the payout example
with an authenticated AWS profile:

```bash
STRANDS_TOOL_CONSOLE_MODE=enabled hatch run python examples/verified_payout.py
```

The examples print agent events and compilation progress, including verification
failures and retries. Pass `model=` to `verified_ai_compile` to choose another model
or provider (see also [Getting started](tutorial.md#getting-started)):

```python
from strands.models.openai import OpenAIModel

model = OpenAIModel(client_args={"api_key": "<KEY>"}, model_id="gpt-4o")

@verified_ai_compile(
    pre_conditions=[payout_inputs],
    post_conditions=[maximum_safe_payout],
    model=model,
)
def max_payout(balance_cents: int, fixed_fee_cents: int, fee_bps: int, payout_limit_cents: int) -> int:
    """Return the largest affordable payout in cents, subject to the payout limit."""
```

## Contract semantics

The contracts define correctness. The docstring provides synthesis guidance;
the decorated Python body is not executed. Contracts are translated
deterministically without a model and without executing the validator functions.

Preconditions receive matching input arguments by name. Postconditions receive
the result as their first positional argument and matching inputs by name, as
with `ai_function`. Unannotated validator parameters inherit the decorated
function's types. At least one postcondition is required.

Assertions and early returns retain their control-flow meaning. A successful
`None` return passes, while an assertion failure or supported explicit `raise`
fails. You can also return the existing result object:

```python
from ai_functions.ai_thread import PostConditionResult


def within_payout_limit(result: int, payout_limit_cents: int):
    return PostConditionResult(
        passed=0 <= result <= payout_limit_cents,
        message="Payout must be nonnegative and within its limit",
    )
```

A bare Boolean return is not supported. Contracts remain enforced when Python
runs with `-O`.

## Compile explicitly or on first use

```python
await max_payout.compile()  # Prepare the implementation at service startup.
payout = await max_payout(10_000, 30, 290, 20_000)

max_payout.compile_sync()   # Equivalent preparation from synchronous code.
payout = max_payout.run_sync(10_000, 30, 290, 20_000)
```

Both compile methods are idempotent and return the decorated function. Without
an explicit compile call, the first valid function call performs compilation.
`max_payout.is_compiled` reports whether this object has resolved a compiled
artifact. Compilation proves one reusable function; it is not specialized to
the first balance, fee, or limit.

Preconditions are checked before every invocation. Invalid inputs do not
initiate synthesis. Concurrent calls coordinate compilation through a file
lock. Verified artifacts are reused across objects and Python processes using
the same compatible runtime installation. Cache keys include the contracts,
types, captured constants, guidance, compiler/translator version, platform,
and runtime installation. Corrupted or incomplete entries are rebuilt.

## Inspect generated artifacts

After compiling, `max_payout.artifact_dir` returns the directory containing the
verified source and native binary. Before compilation it returns `None`; merely
reading this property never starts synthesis.

```python
max_payout.compile_sync()
print(max_payout.artifact_dir)
```

The directory's `Verified<hash>.lean` file contains `pre` and `post` (the
translated specification), `implementation`, and `implementation_correct` (the
proof). The same directory also contains generated C, compiled artifacts, and
their integrity manifest. Treat these cache files as read-only. No knowledge of
the internal compiler language is needed to define or call the Python function.

The example can print the source path directly:

```bash
hatch run python examples/verified_payout.py --show-artifacts
```

## Supported contract types

- Function inputs and outputs can be `int`, `bool`, `float`, or `list[int]`.
  Integers retain arbitrary precision. Values must have the declared type;
  implicit integer/Boolean/float conversions are not performed.
- Validators are ordinary synchronous functions available in Python source
  files. Local assignments, `if`/`elif`/`else`, conditional expressions, early
  returns, assertions, and simple explicit failures are supported.
- Expressions support `+`, `-`, `*`, comparisons, and Boolean logic. Lists
  support equality, membership, concatenation, `len`, `sorted`, and slices
  without a step. A shallow snapshot of list references prevents caller
  mutation during validation or compilation from changing the verified input;
  the integer values are not copied.
- Bounded `all` and `any` generators over integer lists and slices are supported,
  including filters and nested quantifier expressions. Empty-domain and
  short-circuit behavior follow Python. General indexing, stepped slices,
  arbitrary calls, mutation, async validators, and AI validators are rejected.
- Captured numeric constants are frozen when the decorator is applied.
  Reapply the decorator to create a new specification after changing a captured
  constant. Mutable captured state is rejected.

Use constant result messages. Simple f-strings are supported on assertions;
integer/list interpolation in a `PostConditionResult` message is rejected
because it can raise under Python's integer-to-decimal conversion limit.

### Quantified search contracts

The [lower-bound example](../examples/verified_lower_bound.py) specifies a sorted
table lookup using properties of the returned position:

```python
def sorted_values(values: list[int]):
    assert values == sorted(values)


def insertion_position(result: int, values: list[int], key: int):
    assert 0 <= result <= len(values)
    assert all(value < key for value in values[:result])
    assert all(value >= key for value in values[result:])
```

For integer lists, equality with `sorted(values)` translates to the equivalent
pairwise ordering property. The returned position is specified independently of
the search algorithm. Functional verification does not establish logarithmic
complexity. List copying and runtime precondition validation also have a cost.

### Floating-point behavior

Python `float` maps to native IEEE-754 binary64 with a corresponding logical
model. Comparisons use IEEE equality: NaN is unequal to itself and positive and
negative zero compare equal. `math.isfinite`, `math.isnan`, and `math.isinf` are
supported in contracts. Use float literals such as `0.0` in float comparisons;
mixed integer/float comparisons are rejected rather than rounded silently.

Addition, subtraction, multiplication, and negation use binary64 semantics.
Native compilation disables multiply/add contraction to preserve separate
rounding steps. Division and transcendental functions in Python contracts are
not yet supported.

Finite values, infinities, subnormals, and signed zero cross the boundary without
decimal conversion. The pinned floating-point model/runtime canonicalizes NaNs;
preserving a NaN's payload or sign bits is not part of this interface. Contracts
cannot inspect raw floating-point bits.

## Failures and configuration

Setup errors and native build failures fail directly. Exhausted synthesis raises
the existing `AIFunctionError` base type, with a function-oriented message and
optional internal `diagnostics` for debugging. A failed candidate is never
installed or executed, and there is no fallback to an unverified implementation.

`compile_timeout` sets the timeout in seconds for each compiler/checker stage
(default 120). Cancellation terminates compiler subprocesses and releases cache
locks. `cache_dir` can select a different artifact cache, including for CI.

Verification establishes the written contracts. Their deterministic translation
and the native compiler/runtime are trusted implementation components. Keep the
contracts strong enough to specify the behavior the application needs.
