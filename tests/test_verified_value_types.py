"""Python semantics for list quantifiers, slices, and IEEE-754 contracts."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_functions.experimental.verified_compile.contracts import specification
from ai_functions.experimental.verified_compile.errors import ContractError


def list_function(values: list[int], bound: int) -> bool:
    pass


def filtered_contract(result, values, bound):
    expected = all(value < bound for value in values if value >= 0)
    assert result == expected


def nested_contract(result, values, bound):
    assert result == all(any(other >= value for other in values if other <= bound) for value in values)


@given(st.lists(st.integers(), max_size=30), st.integers(), st.booleans())
def test_quantifiers_preserve_filtering_empty_domains_and_nested_bindings(values, bound, result):
    filtered = specification(list_function, [], [filtered_contract])
    assert filtered.post[0].predicate.evaluate({"v0": values, "v1": bound, "r": result}) == (
        result == all(value < bound for value in values if value >= 0)
    )
    nested = specification(list_function, [], [nested_contract])
    assert nested.post[0].predicate.evaluate({"v0": values, "v1": bound, "r": result}) == (
        result == all(any(other >= value for other in values if other <= bound) for value in values)
    )


@given(st.lists(st.integers(), max_size=30), st.integers(), st.integers())
def test_slice_semantics_include_negative_and_out_of_range_bounds(values, start, stop):
    def function(values: list[int], start: int, stop: int) -> list[int]:
        pass

    def contract(result, values, start, stop):
        assert result == values[start:stop]

    spec = specification(function, [], [contract])
    assert spec.post[0].predicate.evaluate({"v0": values, "v1": start, "v2": stop, "r": values[start:stop]})


def test_list_arguments_are_snapshotted_and_type_checked():
    spec = specification(list_function, [], [filtered_contract])
    values = [1, 2]
    bound = spec.bind(values, 5)
    values.append(100)
    assert bound["v0"] == [1, 2]
    with pytest.raises(TypeError, match="only int"):
        spec.bind([1, True], 5)
    with pytest.raises(TypeError, match=r"list\[int\]"):
        spec.bind((1, 2), 5)


@given(st.lists(st.integers(), max_size=30))
def test_sortedness_is_a_property_equivalent_to_python_sorting(values):
    def ordered(values):
        assert values == sorted(values)

    spec = specification(list_function, [ordered], [filtered_contract])
    assert spec.pre[0].predicate.evaluate({"v0": values}) == (values == sorted(values))
    assert "List.Pairwise" in spec.pre[0].predicate.prop()


def test_shadowed_builtins_and_unbounded_domains_are_rejected():
    def all(values):
        raise RuntimeError("must not execute")

    def contract(result, values):
        assert result == all(value > 0 for value in values)

    # A function that shadows a builtin is a helper, not the builtin, and is never executed.
    with pytest.raises(ContractError, match="only inside the builtin"):
        specification(list_function, [], [contract])

    def local_shadow(result, values):
        any = 1
        assert any(value > 0 for value in values)

    with pytest.raises(ContractError, match="local values"):
        specification(list_function, [], [local_shadow])


def float_function(x: float, y: float) -> bool:
    pass


def float_contract(result, x, y):
    assert result == ((x == y) or (math.isnan(x) and math.isinf(y)))


@given(st.floats(width=64), st.floats(width=64), st.booleans())
def test_float_comparisons_keep_ieee_nan_semantics(x, y, result):
    spec = specification(float_function, [], [float_contract])
    expected = result == ((x == y) or (math.isnan(x) and math.isinf(y)))
    assert spec.post[0].predicate.evaluate({"v0": x, "v1": y, "r": result}) == expected
    assert "Float.beq" in spec.post[0].predicate.prop()


def test_mixed_numeric_comparison_is_not_silently_rounded():
    def contract(result, x):
        assert x >= 0

    with pytest.raises(ContractError, match="matching types"):
        specification(float_function, [], [contract])


def test_float_nan_and_signed_zero_truthiness():
    def contract(result, x):
        if x:
            assert result
        else:
            assert not result

    spec = specification(float_function, [], [contract])
    for value in (0.0, -0.0, math.nan, math.inf, -math.inf, 1.0):
        assert spec.post[0].predicate.evaluate({"v0": value, "r": bool(value)})
