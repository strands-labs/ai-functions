"""Tests for post-condition validation functions and PostConditionRunner.

This module tests:
- validate_post_condition_signature: Validates post-condition function signatures
- validate_post_condition_params: Validates post-condition parameters match AI function
- PostConditionRunner: Runs post-condition validators on AI function results
"""

import keyword

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_functions.types.ai_function import PostConditionResult
from ai_functions.validation.post_conditions import (
    PostConditionRunner,
    validate_post_condition_params,
    validate_post_condition_signature,
)

# Strategy for generating valid Python identifiers that are not reserved keywords
valid_python_identifier = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda x: not keyword.iskeyword(x) and x not in {"result", "self", "cls"}
)


# =============================================================================
# Unit Tests: validate_post_condition_signature
# =============================================================================


class TestValidatePostConditionSignature:
    """Unit tests for validate_post_condition_signature function."""

    def test_raises_for_no_parameters(self):
        """Test raises ValueError for function with no parameters."""

        def no_params():
            pass

        with pytest.raises(ValueError, match="must have at least one parameter"):
            validate_post_condition_signature(no_params)

    def test_raises_for_only_kwargs(self):
        """Test raises ValueError for function with only **kwargs."""

        def only_kwargs(**kwargs):
            pass

        with pytest.raises(ValueError, match="must have a positional parameter"):
            validate_post_condition_signature(only_kwargs)

    def test_raises_for_args_first(self):
        """Test raises ValueError for function with *args as first parameter."""

        def args_first(*args):
            pass

        with pytest.raises(ValueError, match="must have a named first parameter"):
            validate_post_condition_signature(args_first)

    def test_raises_for_keyword_only_first(self):
        """Test raises ValueError for function with keyword-only first parameter."""

        def keyword_only_first(*, result):
            pass

        with pytest.raises(ValueError, match="first parameter must be positional"):
            validate_post_condition_signature(keyword_only_first)

    def test_accepts_valid_signature(self):
        """Test accepts function with valid signature (positional first param)."""

        def valid(result):
            return True

        # Should not raise
        validate_post_condition_signature(valid)

    def test_accepts_valid_signature_with_additional_params(self):
        """Test accepts function with valid signature and additional parameters."""

        def valid_with_params(result, extra_param: str = "default"):
            return True

        # Should not raise
        validate_post_condition_signature(valid_with_params)

    def test_accepts_valid_signature_with_kwargs(self):
        """Test accepts function with valid first param and **kwargs."""

        def valid_with_kwargs(result, **kwargs):
            return True

        # Should not raise
        validate_post_condition_signature(valid_with_kwargs)


# =============================================================================
# Unit Tests: validate_post_condition_params
# =============================================================================


class TestValidatePostConditionParams:
    """Unit tests for validate_post_condition_params function."""

    def test_raises_for_params_not_in_ai_function(self):
        """Test raises ValueError for condition params not in AI function."""

        def condition(result, unknown_param):
            return True

        def ai_func(a: str, b: int) -> str:
            return f"{a}-{b}"

        with pytest.raises(ValueError, match="doesn't exist in AI function"):
            validate_post_condition_params(condition, ai_func)

    def test_accepts_kwargs_conditions(self):
        """Test accepts condition with **kwargs regardless of AI function params."""

        def condition_with_kwargs(result, **kwargs):
            return True

        def ai_func(a: str, b: int, c: float) -> str:
            return f"{a}-{b}-{c}"

        # Should not raise - **kwargs accepts any parameters
        validate_post_condition_params(condition_with_kwargs, ai_func)

    def test_accepts_matching_parameters(self):
        """Test accepts condition with parameters matching AI function."""

        def condition(result, a, b):
            return True

        def ai_func(a: str, b: int) -> str:
            return f"{a}-{b}"

        # Should not raise - parameters match
        validate_post_condition_params(condition, ai_func)

    def test_accepts_subset_of_parameters(self):
        """Test accepts condition with subset of AI function parameters."""

        def condition(result, a):
            return True

        def ai_func(a: str, b: int, c: float) -> str:
            return f"{a}-{b}-{c}"

        # Should not raise - 'a' exists in AI function
        validate_post_condition_params(condition, ai_func)

    def test_accepts_result_only_condition(self):
        """Test accepts condition with only result parameter."""

        def condition(result):
            return True

        def ai_func(a: str, b: int) -> str:
            return f"{a}-{b}"

        # Should not raise - no additional params to validate
        validate_post_condition_params(condition, ai_func)

    def test_accepts_args_in_condition(self):
        """Test accepts condition with *args (skips validation for *args)."""

        def condition(result, *args):
            return True

        def ai_func(a: str, b: int) -> str:
            return f"{a}-{b}"

        # Should not raise - *args is skipped in validation
        validate_post_condition_params(condition, ai_func)


# =============================================================================
# Property Tests: Post-Condition Signature Validation
# =============================================================================


class TestPostConditionSignatureProperties:
    """Property-based tests for post-condition signature validation."""

    @given(
        param_name=valid_python_identifier,
        has_default=st.booleans(),
        has_kwargs=st.booleans(),
    )
    @settings(max_examples=100)
    def test_valid_signature_acceptance(self, param_name, has_default, has_kwargs):
        """For any function with at least one positional parameter as the first
        parameter, `validate_post_condition_signature` should not raise an exception.
        """
        # Dynamically create a function with valid signature
        if has_default and has_kwargs:
            exec(f"def valid_func({param_name}='default', **kwargs): return True", globals())
        elif has_default:
            exec(f"def valid_func({param_name}='default'): return True", globals())
        elif has_kwargs:
            exec(f"def valid_func({param_name}, **kwargs): return True", globals())
        else:
            exec(f"def valid_func({param_name}): return True", globals())

        func = globals()["valid_func"]

        # Should not raise for any valid signature
        validate_post_condition_signature(func)


# =============================================================================
# Property Tests: Post-Condition Parameter Validation
# =============================================================================


class TestPostConditionParamsProperties:
    """Property-based tests for post-condition parameter validation."""

    @given(
        ai_param_names=st.lists(
            valid_python_identifier,
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_kwargs_accepts_any_parameters(self, ai_param_names):
        """For any post-condition function that uses **kwargs,
        `validate_post_condition_params` should accept it regardless of
        the AI function's parameters.
        """

        # Create a condition with **kwargs
        def condition_with_kwargs(result, **kwargs):
            return True

        # Create AI function with arbitrary parameters
        params_str = ", ".join(f"{name}: str" for name in ai_param_names)
        exec(f"def ai_func({params_str}) -> str: return 'result'", globals())
        ai_func = globals()["ai_func"]

        # Should not raise - **kwargs accepts any parameters
        validate_post_condition_params(condition_with_kwargs, ai_func)

    @given(
        ai_param_names=st.lists(
            valid_python_identifier,
            min_size=2,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_matching_parameters_accepted(self, ai_param_names):
        """For any post-condition function whose parameters (after the first)
        match the AI function's parameters, `validate_post_condition_params`
        should not raise an exception.
        """
        # Pick a subset of AI function params for the condition
        subset_size = len(ai_param_names) // 2 + 1
        condition_params = ai_param_names[:subset_size]

        # Create condition with matching parameters
        cond_params_str = ", ".join(condition_params)
        exec(f"def condition(result, {cond_params_str}): return True", globals())
        condition = globals()["condition"]

        # Create AI function with all parameters
        ai_params_str = ", ".join(f"{name}: str" for name in ai_param_names)
        exec(f"def ai_func({ai_params_str}) -> str: return 'result'", globals())
        ai_func = globals()["ai_func"]

        # Should not raise - all condition params exist in AI function
        validate_post_condition_params(condition, ai_func)


# =============================================================================
# Unit Tests: PostConditionRunner
# =============================================================================


class TestPostConditionRunner:
    """Unit tests for PostConditionRunner class."""

    @pytest.mark.asyncio
    async def test_returns_passed_for_post_condition_result_true(self):
        """Test returns passed PostConditionResult when condition returns passed=True."""

        def check(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].message is None

    @pytest.mark.asyncio
    async def test_returns_passed_for_none_return(self):
        """Test returns passed PostConditionResult when condition returns None."""

        def check(result):
            return None  # None means passed

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].message is None

    @pytest.mark.asyncio
    async def test_returns_passed_for_async_none_return(self):
        """Test returns passed PostConditionResult when async condition returns None."""

        async def check(result):
            return None  # None means passed

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].message is None

    @pytest.mark.asyncio
    async def test_returns_failed_result_for_failed_condition(self):
        """Test returns failed PostConditionResult when condition returns passed=False."""

        def check(result):
            return PostConditionResult(passed=False, message="Validation failed")

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].message == "Validation failed"

    @pytest.mark.asyncio
    async def test_uses_post_condition_result_directly(self):
        """Test uses PostConditionResult directly when returned by condition."""

        def check(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_returns_failed_result_with_error_message(self):
        """Test returns PostConditionResult with message when condition fails."""

        def check(result):
            return PostConditionResult(passed=False, message="Custom error message")

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].message == "Custom error message"

    @pytest.mark.asyncio
    async def test_handles_async_conditions(self):
        """Test handles async condition functions correctly."""

        async def async_check(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([async_check], "test_result")

        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_returns_failed_result_for_async_condition(self):
        """Test returns failed result for async condition that returns failed result."""

        async def async_check(result):
            return PostConditionResult(passed=False, message="Async validation failed")

        runner = PostConditionRunner()
        results = await runner.validate([async_check], "test_result")

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].message == "Async validation failed"

    @pytest.mark.asyncio
    async def test_captures_exceptions_as_failed_result(self):
        """Test captures exceptions raised by condition functions as failed results."""

        def check_raises(result):
            raise ValueError("Something went wrong")

        runner = PostConditionRunner()
        results = await runner.validate([check_raises], "test_result")

        assert len(results) == 1
        assert results[0].passed is False
        assert "ValueError" in results[0].message
        assert "Something went wrong" in results[0].message

    @pytest.mark.asyncio
    async def test_passes_all_args_to_kwargs_conditions(self):
        """Test passes all bound_args to conditions that accept **kwargs."""
        received_kwargs = {}

        def check_with_kwargs(result, **kwargs):
            received_kwargs.update(kwargs)
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        bound_args = {"a": "value_a", "b": 42, "c": [1, 2, 3]}

        await runner.validate([check_with_kwargs], "test_result", bound_args)

        assert received_kwargs == bound_args

    @pytest.mark.asyncio
    async def test_passes_only_matching_args(self):
        """Test passes only matching bound_args to conditions with specific params."""
        received_args = {}

        def check_with_specific_params(result, a, b):
            received_args["a"] = a
            received_args["b"] = b
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        bound_args = {"a": "value_a", "b": 42, "c": "should_not_be_passed"}

        await runner.validate([check_with_specific_params], "test_result", bound_args)

        assert received_args == {"a": "value_a", "b": 42}
        assert "c" not in received_args

    @pytest.mark.asyncio
    async def test_multiple_conditions_all_pass(self):
        """Test multiple conditions all passing."""

        def check1(result):
            return PostConditionResult(passed=True)

        def check2(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([check1, check2], "test_result")

        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_multiple_conditions_first_fails_continues_to_second(self):
        """Test multiple conditions where first fails - should continue and run all."""
        check2_called = False

        def check1(result):
            return PostConditionResult(passed=False, message="First check failed")

        def check2(result):
            nonlocal check2_called
            check2_called = True
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([check1, check2], "test_result")

        assert len(results) == 2
        assert results[0].passed is False
        assert results[0].message == "First check failed"
        assert results[1].passed is True
        assert check2_called is True

    @pytest.mark.asyncio
    async def test_function_name_stored_in_runner(self):
        """Test function_name is stored in PostConditionRunner for caller use."""

        def check(result):
            return PostConditionResult(passed=False, message="Check failed")

        runner = PostConditionRunner(function_name="my_ai_function")
        results = await runner.validate([check], "test_result")

        # Runner stores function_name for caller to use when building errors
        assert runner.function_name == "my_ai_function"
        assert len(results) == 1
        assert results[0].passed is False

    @pytest.mark.asyncio
    async def test_captures_type_error_for_non_post_condition_result(self):
        """Test captures TypeError as failed result when condition returns non-PostConditionResult."""

        def check(result):
            return True  # Returns bool instead of PostConditionResult or None

        runner = PostConditionRunner()
        results = await runner.validate([check], "test_result")

        assert len(results) == 1
        assert results[0].passed is False
        assert "must return PostConditionResult or None" in results[0].message

    @pytest.mark.asyncio
    async def test_infrastructure_failure_continues_other_conditions(self):
        """Test infrastructure failure in one condition doesn't stop others (return_exceptions=True)."""
        import unittest.mock as mock

        # Mock _check_condition to simulate infrastructure failure for second condition
        def check1(result):
            return PostConditionResult(passed=True)

        def check2(result):
            return PostConditionResult(passed=True)

        def check3(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()

        # Patch _check_condition to raise exception for the second condition
        original_check = runner._check_condition

        async def patched_check(condition, result, bound_args):
            if condition is check2:
                # Simulate infrastructure failure (not caught by _check_condition)
                raise RuntimeError("Infrastructure failure in validation layer")
            return await original_check(condition, result, bound_args)

        with mock.patch.object(runner, "_check_condition", side_effect=patched_check):
            results = await runner.validate([check1, check2, check3], "test_result")

        # All three conditions should return results
        assert len(results) == 3

        # First condition passes normally
        assert results[0].passed is True

        # Second condition fails due to infrastructure error
        assert results[1].passed is False
        assert "Infrastructure failure" in results[1].message
        assert "RuntimeError" in results[1].message

        # Third condition still runs and passes
        assert results[2].passed is True


# =============================================================================
# Property Tests: PostConditionRunner - Batch Validation
# =============================================================================


class TestPostConditionRunnerBatchValidationProperties:
    """Property-based tests for batch validation behavior.

    Feature: post-condition-batch-validation
    """

    @given(
        num_passing=st.integers(min_value=0, max_value=5),
        num_failing=st.integers(min_value=0, max_value=5),
        num_exception=st.integers(min_value=0, max_value=5),
        result_value=st.one_of(st.text(), st.integers(), st.booleans()),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_all_conditions_executed(self, num_passing: int, num_failing: int, num_exception: int, result_value):
        """Property 1: All Conditions Executed

        *For any* list of post-conditions (including conditions that pass, fail,
        or raise exceptions), `validate()` SHALL return a result for each condition
        in the input list, with the length of results equal to the length of conditions.

        **Validates: Requirements 1.1, 1.2, 1.3**

        Feature: post-condition-batch-validation, Property 1: All Conditions Executed
        """
        # Track which conditions were actually called
        called_conditions: list[str] = []

        # Create condition functions that track their execution
        conditions: list = []

        for i in range(num_passing):
            name = f"passing_{i}"

            def make_passing(n: str):
                def condition(result):
                    called_conditions.append(n)
                    return PostConditionResult(passed=True)

                condition.__name__ = n
                return condition

            conditions.append(make_passing(name))

        for i in range(num_failing):
            name = f"failing_{i}"

            def make_failing(n: str):
                def condition(result):
                    called_conditions.append(n)
                    return PostConditionResult(passed=False, message=f"Error from {n}")

                condition.__name__ = n
                return condition

            conditions.append(make_failing(name))

        for i in range(num_exception):
            name = f"exception_{i}"

            def make_exception(n: str):
                def condition(result):
                    called_conditions.append(n)
                    raise ValueError(f"Exception from {n}")

                condition.__name__ = n
                return condition

            conditions.append(make_exception(name))

        # Skip if no conditions (trivial case)
        if not conditions:
            return

        runner = PostConditionRunner()
        results = await runner.validate(conditions, result_value)

        # Property: len(results) == len(conditions)
        assert len(results) == len(conditions), f"Expected {len(conditions)} results, got {len(results)}"

        # Property: All conditions were called (validates 1.1, 1.2, 1.3)
        assert len(called_conditions) == len(conditions), (
            f"Expected {len(conditions)} conditions to be called, but only {len(called_conditions)} were called"
        )

        # Verify result types
        for result in results:
            assert isinstance(result, PostConditionResult)


# =============================================================================
# Property Tests: PostConditionRunner
# =============================================================================


class TestPostConditionRunnerProperties:
    """Property-based tests for PostConditionRunner."""

    @given(
        result_value=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.lists(st.integers()),
            st.dictionaries(st.text(min_size=1), st.integers()),
        ),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_passed_result_returns_passed(self, result_value):
        """For any condition function that returns PostConditionResult(passed=True),
        PostConditionRunner should return a PostConditionResult with passed=True.
        """

        def always_passes(result):
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        results = await runner.validate([always_passes], result_value)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].message is None

    @given(
        result_value=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.lists(st.integers()),
            st.dictionaries(st.text(min_size=1), st.integers()),
        ),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_failed_result_returns_failed(self, result_value):
        """For any condition function that returns PostConditionResult(passed=False),
        PostConditionRunner should return a list with a failed PostConditionResult.
        """

        def always_fails(result):
            return PostConditionResult(passed=False, message="Always fails")

        runner = PostConditionRunner()
        results = await runner.validate([always_fails], result_value)

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].message == "Always fails"

    @given(
        passed=st.booleans(),
        result_value=st.one_of(st.text(), st.integers()),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_async_condition_handling(self, passed, result_value):
        """For any async condition function, PostConditionRunner should await it
        and return the result correctly.
        """

        async def async_condition(result):
            if passed:
                return PostConditionResult(passed=True)
            return PostConditionResult(passed=False, message="Async condition failed")

        runner = PostConditionRunner()
        results = await runner.validate([async_condition], result_value)

        assert len(results) == 1
        assert results[0].passed == passed
        if not passed:
            assert results[0].message == "Async condition failed"

    @given(
        error_message=st.text(min_size=1, max_size=100),
        result_value=st.one_of(st.text(), st.integers()),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_exception_handling(self, error_message, result_value):
        """For any condition function that raises an exception, PostConditionRunner
        should return a failed PostConditionResult containing the exception details.
        """

        def raises_exception(result):
            raise ValueError(error_message)

        runner = PostConditionRunner()
        results = await runner.validate([raises_exception], result_value)

        assert len(results) == 1
        assert results[0].passed is False
        # The error message should contain the exception type and message
        assert "ValueError" in results[0].message
        assert error_message in results[0].message

    @given(
        bound_args=st.dictionaries(
            valid_python_identifier,
            st.one_of(st.text(), st.integers(), st.booleans()),
            min_size=1,
            max_size=5,
        ),
        result_value=st.one_of(st.text(), st.integers()),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_kwargs_arg_passing(self, bound_args, result_value):
        """For any condition function that accepts **kwargs and any set of bound_args,
        PostConditionRunner should pass all bound_args to the condition.
        """
        received_kwargs = {}

        def condition_with_kwargs(result, **kwargs):
            received_kwargs.update(kwargs)
            return PostConditionResult(passed=True)

        runner = PostConditionRunner()
        await runner.validate([condition_with_kwargs], result_value, bound_args)

        # All bound_args should be passed to the condition
        assert received_kwargs == bound_args

    @given(
        param_names=st.lists(
            valid_python_identifier,
            min_size=2,
            max_size=5,
            unique=True,
        ),
        result_value=st.one_of(st.text(), st.integers()),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_matching_arg_passing(self, param_names, result_value):
        """For any condition function with specific named parameters and any set of
        bound_args, PostConditionRunner should pass only the matching bound_args.
        """
        # Create bound_args with all param names
        bound_args = {name: f"value_{name}" for name in param_names}

        # Pick a subset of params for the condition (at least 1)
        subset_size = max(1, len(param_names) // 2)
        condition_params = param_names[:subset_size]
        extra_params = param_names[subset_size:]

        received_args = {}

        # Dynamically create a condition function with specific params
        params_str = ", ".join(condition_params)
        local_ns = {"received_args": received_args, "PostConditionResult": PostConditionResult}
        exec(
            f"def condition(result, {params_str}):\n"
            f"    for name in {condition_params!r}:\n"
            f"        received_args[name] = locals()[name]\n"
            f"    return PostConditionResult(passed=True)",
            local_ns,
        )
        condition = local_ns["condition"]

        runner = PostConditionRunner()
        await runner.validate([condition], result_value, bound_args)

        # Only matching params should be passed
        for param in condition_params:
            assert param in received_args
            assert received_args[param] == bound_args[param]

        # Extra params should NOT be passed
        for param in extra_params:
            assert param not in received_args
