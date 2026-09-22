"""Semantic checks for the deterministic Python-contract translator."""

from __future__ import annotations

import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.verified_compile.contracts import specification
from ai_functions.experimental.verified_compile.errors import ContractError

_LIMIT = 7


def _function(x: int, lo: int = -10, hi: int = 10) -> int:
    """Clamp x to the inclusive interval."""


def _bounds(lo, hi):
    assert lo <= hi


def _clamp(result, x, lo, hi):
    assert lo <= result <= hi
    if x < lo:
        assert result == lo
    elif x > hi:
        assert result == hi
    else:
        assert result == x


def _branching(result, x):
    square = x * x
    if x < 0:
        return PostConditionResult(passed=result == square, message="expected square")
    if x == 0:
        raise ValueError("zero is rejected")
    assert result == x + 1, f"expected successor of {x}"
    return None


def _python_passes(validator, result, **kwargs):
    try:
        outcome = validator(result, **kwargs)
        return outcome is None or outcome.passed
    except Exception:
        return False


@given(st.integers(), st.integers(), st.integers(), st.integers())
def test_clamp_translation_matches_python(result, x, lo, hi):
    spec = specification(_function, [_bounds], [_clamp])
    values = {"v0": x, "v1": lo, "v2": hi, "r": result}
    assert spec.pre[0].predicate.evaluate(values) == (lo <= hi)
    assert spec.post[0].predicate.evaluate(values) == _python_passes(_clamp, result, x=x, lo=lo, hi=hi)


@given(st.integers(), st.integers())
def test_assignments_branches_and_early_returns_match_python(result, x):
    spec = specification(_function, [], [_branching])
    assert spec.post[0].predicate.evaluate({"v0": x, "r": result}) == _python_passes(_branching, result, x=x)


def test_binding_preserves_names_defaults_and_preconditions():
    spec = specification(_function, [_bounds], [_clamp])
    assert spec.bind(x=20) == {"v0": 20, "v1": -10, "v2": 10}
    with pytest.raises(ContractError, match="Precondition '_bounds' failed"):
        spec.bind(2, 10, -10)
    with pytest.raises(TypeError, match="must be int"):
        spec.bind(True)
    with pytest.raises(TypeError):
        spec.bind()


def test_postcondition_input_assertion_is_not_a_precondition():
    def contract(result, x):
        assert x >= 0
        assert result >= x

    spec = specification(_function, [], [contract])
    assert spec.pre == ()
    assert spec.post[0].predicate.evaluate({"v0": -1, "r": 5}) is False


def test_translation_never_executes_the_validator():
    effects = []

    def unsupported(result):
        effects.append(result)
        assert result > 0

    with pytest.raises(ContractError, match="Expr is not supported"):
        specification(_function, [], [unsupported])
    assert effects == []


def test_unsupported_source_is_rejected_even_after_a_return():
    def unsupported(result):
        return None
        print(result)  # noqa: B018 — deliberate unreachable unsupported syntax

    with pytest.raises(ContractError, match="Expr is not supported"):
        specification(_function, [], [unsupported])


def test_partial_arithmetic_and_arbitrary_calls_are_rejected():
    def division(result, x):
        assert result == x // 2

    def call(result, x):
        assert result == abs(x)

    for validator in (division, call):
        with pytest.raises(ContractError, match="not supported"):
            specification(_function, [], [validator])


def test_mutable_globals_and_async_validators_are_rejected():
    state = [1]

    def mutable(result):
        assert result == state[0]

    async def asynchronous(result):
        assert result == 1

    for validator in (mutable, asynchronous):
        with pytest.raises(ContractError):
            specification(_function, [], [validator])


def test_a_bare_boolean_return_is_not_a_validator():
    def unsupported(result):
        return result > 0

    with pytest.raises(ContractError, match="return None or PostConditionResult"):
        specification(_function, [], [unsupported])


def test_integer_formatting_cannot_weaken_a_successful_result_contract():
    def unsupported(result):
        return PostConditionResult(passed=True, message=f"{result}")

    with pytest.raises(ContractError, match="digit limit"):
        specification(_function, [], [unsupported])


def test_immutable_constants_are_snapshotted_into_the_specification():
    bound = 10

    def contract(result):
        assert result >= bound

    spec = specification(_function, [], [contract])
    bound = 20
    assert spec.post[0].predicate.evaluate({"r": 15})
    assert specification(_function, [], [contract]).post[0].predicate.evaluate({"r": 15}) is False


def test_uninitialized_local_does_not_resolve_to_global():
    def contract(result):
        assert result == _LIMIT  # noqa: F823 — deliberate uninitialized local
        _LIMIT = 0

    with pytest.raises(ContractError, match="'_LIMIT' is not a parameter"):
        specification(_function, [], [contract])


def test_large_captured_constants_do_not_use_python_decimal_conversion():
    large = 2**20000

    def contract(result):
        assert result == large

    spec = specification(_function, [], [contract])
    assert "0x" in spec.post[0].predicate.lean()
    assert spec.post[0].predicate.evaluate({"r": large}) is True
    assert spec.post[0].predicate.evaluate({"r": large + 1}) is False


@given(st.booleans(), st.booleans(), st.booleans())
def test_boolean_predicates_match_python(result, a, b):
    def function(a: bool, b: bool) -> bool:
        pass

    def contract(result, a, b):
        expected = a and not b
        return PostConditionResult(passed=result == expected, message="incorrect Boolean result")

    spec = specification(function, [], [contract])
    assert spec.post[0].predicate.evaluate({"v0": a, "v1": b, "r": result}) == _python_passes(
        contract, result, a=a, b=b
    )


def test_contracts_remain_enforced_under_python_optimization(tmp_path):
    script = tmp_path / "optimized.py"
    script.write_text("""from ai_functions.experimental.verified_compile.contracts import specification
from ai_functions.experimental.verified_compile.errors import ContractError
def function(x: int) -> int:
    pass
def pre(x):
    assert x > 0
def post(result, x):
    assert result == x
pre(-1)  # Python -O erases this ordinary assertion.
try:
    specification(function, [pre], [post]).bind(-1)
except ContractError:
    print("contract enforced")
else:
    raise RuntimeError("contract was erased")
""")
    result = subprocess.run([sys.executable, "-O", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "contract enforced"
