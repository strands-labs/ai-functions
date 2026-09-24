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

A call checks its argument types and runs the native code. By default it does
not evaluate the contracts. The proof covers every input that satisfies the
preconditions. An input that violates one still returns a value, because Lean
functions are total, and the proof says nothing about that value. For example,
a floor-division function implemented as `Int.fdiv v0 v1` returns 0 for a zero
divisor, where Python would raise `ZeroDivisionError`.

- `check_pre_conditions=True` evaluates the preconditions before every call.
  Invalid inputs raise `ContractError` and do not initiate synthesis.
- `check_post_conditions=True` evaluates the postconditions on every native
  result, as a check on the trusted native compiler, runtime, and value
  conversion for the inputs actually used. A failure raises `CompilerError`.

Both checks run the translated contracts in a Python interpreter. For the loan
example's $250,000, 360-period loan, the postcondition check takes about 26
times as long as the native call.

Concurrent calls coordinate compilation through a file lock. Verified artifacts
are reused across objects and Python processes using the same compatible runtime
installation. Cache keys include the contracts, types, captured constants,
guidance, compiler/translator version, platform, and runtime installation.
Corrupted or incomplete entries are rebuilt.

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
  files. Local assignments (including `x += 1` and `a, b = b, a`),
  `if`/`elif`/`else`, conditional expressions, early returns, assertions, and
  simple explicit failures are supported.
- Expressions support `+`, `-`, `*`, `//`, `%`, `**` with a non-negative integer
  literal exponent, float `/`, comparisons, and Boolean logic, plus `abs`, `min`,
  `max`, `sum`, and `math.sqrt`. Lists support equality, membership,
  concatenation, indexing (including negative indices), `len`, `sorted`,
  `list`, `reversed`, `.count()`, `.index()`, list comprehensions, and slices
  without a step. A shallow snapshot of list references prevents caller
  mutation during validation or compilation from changing the verified input;
  the integer values are not copied.
- Loops, comprehensions, and bounded `all`/`any` iterate over integer lists,
  slices, `range()` (with any nonzero step), `zip()` of two, `enumerate()`
  (optionally with a start), and `reversed()`, with filters and nested
  quantifiers. Empty-domain and short-circuit behavior follow Python.
- Calls to other functions in the same subset are supported and appear in the
  specification as named Lean definitions. Recursive helpers are rejected.
- `for` loops that update variables initialized before the loop are supported
  and become `List.foldl` definitions. Loop bodies contain assignments and `if`
  statements; `break`, `continue`, `return`, assertions inside the loop, and
  `while` are rejected.
- Operations that raise in Python (`//` or `%` by zero, float `/` by zero,
  `math.sqrt` of a negative, out-of-range indexing, `min`/`max` of an empty
  list, `.index()` of a missing value, a zero `range()` step) make the contract
  fail exactly where Python would raise. Quantifiers require every element to be
  defined, which is stricter than Python stopping at the first false element. A
  loop whose raising operation depends on a branch over the loop's own
  variables requires that operation to be defined on every iteration. Stricter
  is sound: a stricter precondition narrows the inputs the proof covers (and,
  with `check_pre_conditions=True`, rejects more of them), and a stricter
  postcondition asks the proof for more.
- Stepped slices, `while`, arbitrary method calls, mutation, float `//` and `%`,
  float `sum` (CPython 3.12 sums floats with compensated summation), async
  validators, and AI validators are rejected.
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
complexity. List copying also has a cost, and so does runtime precondition
validation with `check_pre_conditions=True`.

### Helpers, loops, and indexing

These examples state a property and leave the algorithm to synthesis:

| Example | Contract uses | What the proof connects |
| --- | --- | --- |
| [Banker's rounding](../examples/verified_round_half_even.py) | `abs`, `%`, no division | a tie rule stated in integers, and a floor-division implementation |
| [Installment splits](../examples/verified_installments.py) | `sum`, `max`, `min`, indexing, `range()`, a list result | four properties of the split, and the unique split they allow |
| [First overdraft](../examples/verified_first_overdraft.py) | a helper, slices, `sum`, `range()` | a quadratic prefix-sum statement, and a single pass |
| [Luhn check digit](../examples/verified_luhn_check_digit.py) | two helpers, `enumerate()`, `reversed()`, `%` | "appending it makes the number valid", and a direct computation |
| [Best single trade](../examples/verified_best_trade.py) | nested quantifiers over `range()`, indexing | a comparison of every pair of days, and a single pass |
| [Fee helper](../examples/verified_fee_helpers.py) | a helper, `//` | the payout policy as its own driver computes fees |
| [First maximum](../examples/verified_first_maximum.py) | indexing, `range()`, quantifiers | where the answer is, and how to find it |
| [Level loan payment](../examples/verified_loan_payment.py) | a helper containing a loop, `//` | "one cent less would not clear the loan", and a bisection over a `Nat` simulation that stops once the loan is paid off |

A synthesized implementation can be less efficient than the one a docstring
suggests; the proof covers the result, not the running time. State a performance
requirement in the docstring as a requirement, as the loan example does. Lean
keeps `Nat` values below 2^63 unboxed but `Int` values only within 32 bits, so an
implementation that computes with `Nat` avoids big-integer arithmetic for values
such as cents times basis points.

### Floating-point behavior

Python `float` maps to native IEEE-754 binary64 with a corresponding logical
model. Comparisons use IEEE equality: NaN is unequal to itself and positive and
negative zero compare equal. `math.isfinite`, `math.isnan`, and `math.isinf` are
supported in contracts. Use float literals such as `0.0` in float comparisons;
mixed integer/float comparisons are rejected rather than rounded silently.

Addition, subtraction, multiplication, division, negation, and `math.sqrt` use
binary64 semantics. Native compilation disables multiply/add contraction to
preserve separate rounding steps. Lean's kernel evaluates binary64 arithmetic
on concrete values but not `Float.sqrt` of an ordinary value, so proofs about
square roots are limited to implementations that mirror the specification.
Other transcendental functions are not yet supported.

Finite values, infinities, subnormals, and signed zero cross the boundary without
decimal conversion. The pinned floating-point model/runtime canonicalizes NaNs;
preserving a NaN's payload or sign bits is not part of this interface. Contracts
cannot inspect raw floating-point bits.

## How synthesis works

Each synthesis attempt gives the model two tools, with 24 calls per attempt
across both:

- `test_implementation` runs a candidate implementation, without a proof, on up
  to 100 sampled inputs that satisfy the preconditions, and reports the first
  input where a postcondition fails. It runs in a separate Lean process after
  the same lexical filter as the proof check, and takes seconds.
- `check_lean` runs the full check: the Lean kernel, `leanchecker`, the axiom
  audit, and the native build.

The model is told to test an implementation before it writes a proof. The
candidate it returns is checked again before it is installed, so no tool result
counts as verification.

Sampled inputs come from fixed values per type, plus the integer literals in the
contracts and their neighbors, which are the usual boundaries. A draw that fails
a precondition is discarded. When no draw satisfies the preconditions, the test
reports that nothing ran. The sample only guides the model; the proof covers
every input that satisfies the preconditions.

## Failures and configuration

Setup errors and native build failures fail directly. Exhausted synthesis raises
the existing `AIFunctionError` base type, with a function-oriented message and
optional internal `diagnostics` for debugging. A failed candidate is never
installed or called through the native bridge, and there is no fallback to an
unverified implementation.

`compile_timeout` sets the timeout in seconds for each compiler/checker stage
(default 120). Cancellation terminates compiler subprocesses and releases cache
locks. `cache_dir` can select a different artifact cache, including for CI.

Verification establishes the written contracts. Their deterministic translation
and the native compiler/runtime are trusted implementation components. Keep the
contracts strong enough to specify the behavior the application needs.
`check_post_conditions=True` re-checks results against the same translation, so
it guards the compiler, runtime, and value conversion, but not a mistranslation
of the Python contract.
