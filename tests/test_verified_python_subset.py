"""Arithmetic, sequences, helpers, and loops in contracts.

Every construct is checked three ways on the same inputs: running the Python
validator, the translator's evaluator (which runs precondition checks), and
Lean evaluating the emitted specification. The translation is trusted, so any
disagreement means a proof would certify something Python does not do.
"""

from __future__ import annotations

import inspect
import math
import random
import re
import subprocess

import pytest

from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.verified_compile.contracts import Expr, Specification, specification
from ai_functions.experimental.verified_compile.errors import ContractError

# --- helpers that contracts call; module level so their source is available ---


def fee(amount, bps):
    return (amount * bps + 9_999) // 10_000


def positive(x):
    return x > 0


def checked_half(x):
    assert x % 2 == 0
    return x // 2


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def total(xs):
    t = 0
    for x in xs:
        t += x
    return t


def recursive(x):
    return recursive(x - 1) if x > 0 else 0


# --- decorated signatures ---


def two_ints(a: int, b: int) -> int:
    """Two integers."""


def two_floats(x: float, y: float) -> float:
    """Two floats."""


def one_float(x: float) -> float:
    """One float."""


def one_int(x: int) -> int:
    """One integer."""


def a_list(xs: list[int]) -> int:
    """A list."""


def a_list_and_index(xs: list[int], i: int) -> int:
    """A list and an index."""


def a_list_and_key(xs: list[int], k: int) -> int:
    """A list and a key."""


def an_amount(amount: int, bps: int) -> int:
    """An amount and a rate."""


# --- Phase 1: arithmetic and builtins ---


def floor_division(result, a, b):
    assert result == a // b
    assert a % b == a - b * (a // b)


def float_division(result, x, y):
    assert result <= x / y or math.isnan(x / y)


def powers(result, a, b):
    assert result == a**3 - 2 * a**2 + b**0


def scalar_builtins(result, a, b):
    assert result == max(abs(a), b, min(a, b, 0))


def float_builtins(result, x, y):
    assert result == max(x, abs(y)) or result == min(y, x)


def square_root(result, x):
    assert math.sqrt(x) >= result


def list_builtins(result, xs):
    assert result == sum(xs) - max(xs) + min(x * 2 for x in xs if x > 0)


def augmented(result, a, b):
    t = a
    t += b
    t *= 2
    t //= 3
    first, second = t, b
    first, second = second, first
    assert result == first - second


# --- Phase 2: sequences ---


def indexing(result, xs, i):
    assert result == xs[i] + xs[-1]


def adjacent_by_index(result, xs):
    assert all(xs[k] <= xs[k + 1] for k in range(len(xs) - 1)) == (result == 1)


def adjacent_by_zip(result, xs):
    assert all(a <= b for a, b in zip(xs, xs[1:])) or result == 0  # noqa: B905 - strict= is not in the subset


def enumerated(result, xs):
    assert all(x >= i for i, x in enumerate(xs, 1)) or any(i == result for i, x in enumerate(xs) if x == 0)


def comprehension_values(result, xs):
    assert result == len([x for x in xs if x % 2 == 0]) + sum(x * x for x in xs)


def list_methods(result, xs, k):
    assert result >= xs.count(k)
    assert xs.index(k) <= result


def reversed_and_list(result, xs):
    assert list(reversed(xs)) == sorted(xs) or result == sum(reversed(xs))


def stepped_range(result, xs):
    assert all(xs[k] >= result for k in range(0, len(xs), 2))
    assert all(xs[k] != result for k in range(len(xs) - 1, -1, -1))


# --- Phase 3: helpers and loops ---


def with_helpers(result, amount, bps):
    assert positive(result) or amount <= 0
    assert result == amount - fee(amount, bps)


def with_checked_helper(result, x):
    assert result == checked_half(x) + sign(x)


def loop_accumulators(result, xs):
    acc = 0
    count = 0
    for x in xs:
        acc += x
        if x > 0:
            count += 1
    assert result == acc + count


def loop_search(result, xs, k):
    found = False
    position = -1
    for i, x in enumerate(xs):
        if x == k and not found:
            found = True
            position = i
    assert result == position
    assert found or position == -1


def loop_guarded_division(result, xs):
    acc = 0
    for x in xs:
        if x != 0:
            acc += 100 // x
    assert result == acc


def loop_unguarded_division(result, xs):
    acc = 1
    for x in xs:
        acc = acc + 10 // x
    assert result == acc


def loop_indexed_maximum(result, xs):
    best = 0
    for k in range(len(xs)):
        if xs[k] > best:
            best = xs[k]
    assert result == best


def loop_building_list(result, xs):
    evens = []
    for x in xs:
        if x % 2 == 0:
            evens = evens + [x]
    assert result == len(evens)


def helper_with_loop(result, xs):
    assert result == total(xs) * 2


def result_message(result, a, b):
    return PostConditionResult(passed=result > a // 3, message="result must exceed a third of a")


INTS = [-7, -3, -2, -1, 0, 1, 2, 3, 5, 10, 2**70, -(2**70)]
FLOATS = [0.0, -0.0, 1.0, -1.0, 2.5, -4.0, 0.25, 1e308, math.inf, -math.inf, math.nan]
LISTS = [[], [0], [5], [1, 2, 3], [3, 1, 2], [-2, 0, 2], [4, 4, 4], [0, 1, 0, 2], [-5, 7, -9, 11, 0, 3]]
SMALL = [-3, -1, 0, 1, 2, 4]

# (label, function, validators, input pools, natural result or None)
CASES = [
    ("floor division", two_ints, [floor_division], {"a": INTS, "b": INTS}, lambda a, b: a // b),
    ("float division", two_floats, [float_division], {"x": FLOATS, "y": FLOATS}, lambda x, y: x / y),
    ("powers", two_ints, [powers], {"a": INTS, "b": INTS}, lambda a, b: a**3 - 2 * a**2 + 1),
    ("scalar builtins", two_ints, [scalar_builtins], {"a": INTS, "b": INTS}, lambda a, b: max(abs(a), b, min(a, b, 0))),
    ("float builtins", two_floats, [float_builtins], {"x": FLOATS, "y": FLOATS}, lambda x, y: max(x, abs(y))),
    ("square root", one_float, [square_root], {"x": FLOATS}, lambda x: math.sqrt(x)),
    ("list builtins", a_list, [list_builtins], {"xs": LISTS}, None),
    ("augmented", two_ints, [augmented], {"a": INTS, "b": INTS}, lambda a, b: b - (2 * (a + b)) // 3),
    ("indexing", a_list_and_index, [indexing], {"xs": LISTS, "i": list(range(-6, 7))}, None),
    ("adjacent by index", a_list, [adjacent_by_index], {"xs": LISTS}, None),
    ("adjacent by zip", a_list, [adjacent_by_zip], {"xs": LISTS}, None),
    ("enumerated", a_list, [enumerated], {"xs": LISTS}, None),
    ("comprehension values", a_list, [comprehension_values], {"xs": LISTS}, None),
    ("list methods", a_list_and_key, [list_methods], {"xs": LISTS, "k": SMALL}, lambda xs, k: xs.count(k)),
    ("reversed and list", a_list, [reversed_and_list], {"xs": LISTS}, lambda xs: sum(xs)),
    ("stepped range", a_list, [stepped_range], {"xs": LISTS}, None),
    ("helpers", an_amount, [with_helpers], {"amount": INTS, "bps": [0, 1, 290, 10_000]}, None),
    ("checked helper", one_int, [with_checked_helper], {"x": INTS}, None),
    ("loop accumulators", a_list, [loop_accumulators], {"xs": LISTS}, None),
    ("loop search", a_list_and_key, [loop_search], {"xs": LISTS, "k": SMALL}, None),
    ("loop guarded division", a_list, [loop_guarded_division], {"xs": LISTS}, None),
    ("loop unguarded division", a_list, [loop_unguarded_division], {"xs": LISTS}, None),
    ("loop indexed maximum", a_list, [loop_indexed_maximum], {"xs": LISTS}, lambda xs: max([0, *xs])),
    ("loop building list", a_list, [loop_building_list], {"xs": LISTS}, lambda xs: len([x for x in xs if x % 2 == 0])),
    ("helper with loop", a_list, [helper_with_loop], {"xs": LISTS}, lambda xs: 2 * sum(xs)),
    ("result message", two_ints, [result_message], {"a": INTS, "b": INTS}, None),
]


def _run_python(validator, result, inputs):
    """Whether the Python validator completes successfully."""
    names = list(inspect.signature(validator).parameters)[1:]
    try:
        outcome = validator(result, **{name: inputs[name] for name in names})
    except Exception:
        return False
    return outcome is None or outcome.passed


def _samples(function, pools, natural, count=40, seed=0):
    """Deterministic inputs and results, half of them the natural answer."""
    generator = random.Random(seed)
    kind = inspect.signature(function).return_annotation
    results = FLOATS if kind in (float, "float") else INTS + list(range(-4, 12))
    samples = []
    for index in range(count):
        inputs = {name: generator.choice(pool) for name, pool in pools.items()}
        result = generator.choice(results)
        if natural is not None and index % 2 == 0:
            try:
                result = natural(**inputs)
            except Exception:
                pass
        samples.append((inputs, result))
    return samples


def _values(spec: Specification, inputs, result):
    values = {f"v{i}": inputs[name] for i, (name, _) in enumerate(spec.parameters)}
    return {**values, "r": result}


def _lean_value(value, kind) -> str:
    if kind is list:
        return Expr("list", list, tuple(Expr("literal", int, value=v) for v in value)).lean()
    if kind is bool:
        return "true" if value else "false"
    return Expr("literal", kind, value=value).lean()


@pytest.mark.parametrize(("label", "function", "validators", "pools", "natural"), CASES, ids=[c[0] for c in CASES])
def test_translation_evaluates_like_python(label, function, validators, pools, natural):
    spec = specification(function, [], validators)
    for inputs, result in _samples(function, pools, natural):
        values = _values(spec, inputs, result)
        for validator, contract in zip(validators, spec.post, strict=True):
            expected = _run_python(validator, result, inputs)
            assert bool(contract.predicate.evaluate(values)) == expected, f"{label}: {inputs} result={result}"


def test_emitted_lean_evaluates_like_the_translation(native_runtime, tmp_path):
    blocks, expected = [], []
    for index, (label, function, validators, pools, natural) in enumerate(CASES):
        spec = specification(function, [], validators)
        samples = _samples(function, pools, natural, count=24, seed=1)
        declarations = (spec.helpers() + spec.declarations()).replace("public ", "")
        evaluations = []
        for position, contract in enumerate(spec.post):
            calls = []
            for inputs, result in samples:
                arguments = [_lean_value(result, spec.output_type)]
                arguments += [_lean_value(inputs[name], kind) for name, kind in spec.parameters]
                calls.append(f"decide (post{position} {' '.join(arguments)})")
                evaluated = bool(contract.predicate.evaluate(_values(spec, inputs, result)))
                assert evaluated == _run_python(validators[position], result, inputs), f"{label}: {inputs} {result}"
                expected.append((label, inputs, result, evaluated))
            evaluations.append(f"#eval [{', '.join(calls)}]")
        blocks.append(f"namespace Case{index}\n{declarations}\n" + "\n".join(evaluations) + f"\nend Case{index}\n")
    source = tmp_path / "Differential.lean"
    source.write_text("set_option linter.unusedVariables false\nset_option maxRecDepth 8192\n" + "\n".join(blocks))
    run = subprocess.run(
        [str(native_runtime.toolchain.lean), str(source)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env=native_runtime.toolchain.environment(tmp_path),
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    # Lean wraps long lists across lines, so read the Boolean tokens rather than split on commas.
    printed = [token == "true" for token in re.findall(r"\b(true|false)\b", run.stdout)]
    assert len(printed) == len(expected), run.stdout[-2000:]
    mismatches = [
        (label, inputs, result)
        for (label, inputs, result, want), got in zip(expected, printed, strict=True)
        if want != got
    ]
    assert not mismatches, mismatches[:10]


def test_partial_operations_carry_python_definedness():
    spec = specification(two_ints, [], [floor_division])
    assert "(v1 ≠ (0x0 : Int))" in spec.post[0].predicate.prop()
    assert spec.post[0].predicate.evaluate({"v0": 7, "v1": 0, "r": 0}) is False
    guarded = specification(a_list, [], [loop_guarded_division])
    # The guard on the loop element keeps the division defined, so no condition remains.
    assert "∀" not in guarded.post[0].predicate.prop()
    unguarded = specification(a_list, [], [loop_unguarded_division])
    assert "∀ q0 ∈ v0" in unguarded.post[0].predicate.prop()


def test_helpers_and_loops_become_named_definitions():
    spec = specification(an_amount, [], [with_helpers])
    names = [d.name for d in spec.definitions]
    assert names == ["positive", "fee"]
    assert spec.declarations().splitlines()[0].startswith("abbrev positive (x : Int) : Prop")
    assert "def fee (amount : Int) (bps : Int) : Int := (Int.fdiv" in spec.declarations()
    assert {"positive", "fee"} <= spec.definition_names
    loop = specification(a_list_and_key, [], [loop_search]).definitions[0]
    assert loop.name == "loop0" and loop.result_type == "Bool × Int"
    assert "List.foldl" in loop.declarations()[0] and "List.zipIdx v0" in loop.declarations()[0]
    helper = specification(one_int, [], [with_checked_helper]).definitions[0]
    assert helper.names == ("checked_half", "checked_half_defined")


def test_only_the_helpers_a_specification_uses_are_emitted():
    assert specification(two_ints, [], [floor_division]).helpers() == ""
    assert "pythonAt" in specification(a_list_and_index, [], [indexing]).helpers()
    assert "pythonRange" in specification(a_list, [], [stepped_range]).helpers()
    assert "pythonRange" not in specification(a_list, [], [adjacent_by_index]).helpers()


def _rejected_true_division(result, a, b):
    assert result == a / b


def _rejected_float_floor(result, x, y):
    assert result == x // y


def _rejected_len_of_zip(result, xs):
    assert result == len(zip(xs, xs))  # noqa: B905 - strict= is not in the contract subset


def _rejected_range_value(result, xs):
    assert xs == range(3)


def _rejected_single_target(result, xs):
    assert all(pair for pair in zip(xs, xs))  # noqa: B905 - strict= is not in the contract subset


def _rejected_recursion(result, a, b):
    assert result == recursive(a)


def _rejected_assert_in_loop(result, xs):
    acc = 0
    for x in xs:
        assert x > 0
        acc += x


def _rejected_uninitialized(result, xs):
    for x in xs:
        acc = x  # noqa: F841 - deliberately never initialized before the loop


def _rejected_state_dependent(result, xs):
    acc = 1
    for _ in xs:
        acc = 10 // acc


def _rejected_list_mutation(result, xs):
    ys = xs
    ys += [1]


def _rejected_message(result, a, b):
    assert result > 0, f"{a // b}"


def _rejected_float_sum(result, xs):
    assert result == sum(0.5 for x in xs)


@pytest.mark.parametrize(
    ("function", "validator", "message"),
    [
        (two_ints, _rejected_true_division, "use // for integer division"),
        (two_floats, _rejected_float_floor, "require int operands"),
        (a_list, _rejected_len_of_zip, "iterates over"),
        (a_list, _rejected_range_value, "iterates over"),
        (a_list, _rejected_single_target, "two-name tuple target"),
        (two_ints, _rejected_recursion, "recursive"),
        (a_list, _rejected_assert_in_loop, "assert all"),
        (a_list, _rejected_uninitialized, "initialized before it"),
        (a_list, _rejected_state_dependent, "depends on a variable the loop updates"),
        (a_list, _rejected_list_mutation, "mutates a list"),
        (two_ints, _rejected_message, "must not contain operations that can raise"),
        (a_list, _rejected_float_sum, "must produce integers"),
    ],
)
def test_constructs_outside_the_subset_are_rejected(function, validator, message):
    with pytest.raises(ContractError, match=message):
        specification(function, [], [validator])


def test_helper_translation_never_executes_the_helper():
    effects = []

    def helper(x):
        effects.append(x)
        return x

    def contract(result, x):
        assert result == helper(x)

    with pytest.raises(ContractError, match="Expr is not supported"):
        specification(one_int, [], [contract])
    assert effects == []


# --- end to end: kernel-checked canned candidates, compiled and called natively ---


def floor_div(a: int, b: int) -> int:
    """Floor division."""


def nonzero_divisor(b):
    assert b != 0


def is_floor_division(result, a, b):
    assert result == a // b


def element(xs: list[int], i: int) -> int:
    """Python indexing."""


def index_in_range(xs, i):
    assert -len(xs) <= i < len(xs)


def is_element(result, xs, i):
    assert result == xs[i]


def net(amount: int, bps: int) -> int:
    """Amount after the rounded-up fee."""


def is_net(result, amount, bps):
    assert result == amount - fee(amount, bps)


def running_total(xs: list[int]) -> int:
    """Sum of the list."""


def is_running_total(result, xs):
    acc = 0
    for x in xs:
        acc += x
    assert result == acc


LOOP_PROOF = """by
  intro v0 h
  simp only [post, post0, loop0, implementation]
  have key : ∀ (l : List Int) (a : Int), List.foldl (fun (s0 : Int) (q0 : Int) => s0 + q0) a l = a + l.sum := by
    intro l
    induction l with
    | nil => intro a; simp
    | cons x t ih => intro a; simp only [List.foldl_cons, List.sum_cons, ih]; omega
  rw [key]; simp"""


def _compiled(function, pre, post, implementation, proof, cache, **options):
    from ai_functions.experimental.verified_compile import verified_ai_compile
    from ai_functions.experimental.verified_compile.compiler import Candidate
    from ai_functions.testing import ScriptedModel, Turn

    candidate = Candidate(implementation=implementation, proof=proof)
    llm = ScriptedModel([Turn(tool_calls=(("Candidate", candidate.model_dump()),))])
    return verified_ai_compile(
        pre_conditions=pre, post_conditions=post, model=llm, cache_dir=cache, max_attempts=0, **options
    )(function)


async def test_floor_division_runs_natively_like_python(tmp_path, native_runtime):
    proof = "by intro v0 v1 h; simp_all [pre, pre0, post, post0, implementation]"
    fn = _compiled(floor_div, [nonzero_divisor], [is_floor_division], "Int.fdiv v0 v1", proof, tmp_path)
    for a in (-7, 7, 0, 2**80, -(2**80)):
        for b in (-3, -2, 2, 3):
            assert await fn(a, b) == a // b
    # Python raises here. The proof does not cover a zero divisor, and Lean's total
    # division returns 0, so only the optional precondition check reports it.
    assert fn.run_sync(1, 0) == 0
    checked = _compiled(
        floor_div, [nonzero_divisor], [is_floor_division], "Int.fdiv v0 v1", proof, tmp_path, check_pre_conditions=True
    )
    with pytest.raises(ContractError, match="nonzero_divisor"):
        checked.run_sync(1, 0)


async def test_negative_indexing_runs_natively_like_python(tmp_path, native_runtime):
    proof = "by intro v0 v1 h; simp_all [pre, pre0, post, post0, implementation]"
    fn = _compiled(element, [index_in_range], [is_element], "pythonAt v0 v1", proof, tmp_path)
    for xs in ([1, 2, 3], [-(2**70), 5]):
        for i in range(-len(xs), len(xs)):
            assert await fn(xs, i) == xs[i]
    # Python raises IndexError here; pythonAt returns its default, 0.
    assert fn.run_sync([1, 2, 3], 3) == 0
    checked = _compiled(
        element, [index_in_range], [is_element], "pythonAt v0 v1", proof, tmp_path, check_pre_conditions=True
    )
    with pytest.raises(ContractError, match="index_in_range"):
        checked.run_sync([1, 2, 3], 3)


async def test_helper_definitions_run_natively_like_python(tmp_path, native_runtime):
    proof = "by intro v0 v1 h; simp [post, post0, implementation, fee]"
    fn = _compiled(net, [], [is_net], "v0 - Int.fdiv (v0 * v1 + 9999) 10000", proof, tmp_path)
    for amount in (0, 1, 9_689, 10_000, -500, 2**70):
        for bps in (0, 1, 290, 10_000):
            assert await fn(amount, bps) == amount - fee(amount, bps)


async def test_loop_definitions_run_natively_like_python(tmp_path, native_runtime):
    fn = _compiled(running_total, [], [is_running_total], "List.sum v0", LOOP_PROOF, tmp_path)
    for xs in ([], [5], [1, -2, 3], [2**70, -(2**70), 7]):
        assert await fn(xs) == sum(xs)
