"""Unit tests for async_utils module.

Tests for run_async function that executes async code from sync contexts.
"""

import asyncio
import contextvars

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_functions.utils._async import run_async


class TestRunAsync:
    """Tests for run_async function."""

    def test_executes_async_function_and_returns_result(self):
        """Test run_async executes async function and returns result."""

        async def async_func():
            return 42

        result = run_async(lambda: async_func())
        assert result == 42

    def test_executes_async_function_with_string_result(self):
        """Test run_async returns string result correctly."""

        async def async_func():
            return "hello world"

        result = run_async(lambda: async_func())
        assert result == "hello world"

    def test_executes_async_function_with_complex_result(self):
        """Test run_async returns complex data structures correctly."""

        async def async_func():
            return {"key": "value", "numbers": [1, 2, 3]}

        result = run_async(lambda: async_func())
        assert result == {"key": "value", "numbers": [1, 2, 3]}

    def test_executes_async_function_with_await(self):
        """Test run_async handles async functions that use await."""

        async def inner_async():
            await asyncio.sleep(0.001)
            return "awaited"

        async def outer_async():
            return await inner_async()

        result = run_async(lambda: outer_async())
        assert result == "awaited"

    def test_propagates_value_error(self):
        """Test run_async propagates ValueError exceptions."""

        async def async_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_async(lambda: async_func())

    def test_propagates_runtime_error(self):
        """Test run_async propagates RuntimeError exceptions."""

        async def async_func():
            raise RuntimeError("runtime failure")

        with pytest.raises(RuntimeError, match="runtime failure"):
            run_async(lambda: async_func())

    def test_propagates_custom_exception(self):
        """Test run_async propagates custom exceptions."""

        class CustomError(Exception):
            pass

        async def async_func():
            raise CustomError("custom error message")

        with pytest.raises(CustomError, match="custom error message"):
            run_async(lambda: async_func())

    def test_preserves_context_variables(self):
        """Test run_async preserves context variables across thread boundary."""
        ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_var")
        ctx_var.set("original_value")

        async def async_func():
            return ctx_var.get()

        result = run_async(lambda: async_func())
        assert result == "original_value"

    def test_preserves_multiple_context_variables(self):
        """Test run_async preserves multiple context variables."""
        ctx_var1: contextvars.ContextVar[str] = contextvars.ContextVar("var1")
        ctx_var2: contextvars.ContextVar[int] = contextvars.ContextVar("var2")
        ctx_var1.set("string_value")
        ctx_var2.set(123)

        async def async_func():
            return (ctx_var1.get(), ctx_var2.get())

        result = run_async(lambda: async_func())
        assert result == ("string_value", 123)

    def test_returns_none_when_async_returns_none(self):
        """Test run_async correctly returns None."""

        async def async_func():
            return None

        result = run_async(lambda: async_func())
        assert result is None


class TestRunAsyncProperty:
    """Property-based tests for run_async."""

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=100)
    def test_async_function_returns_same_integer(self, value):
        """For any integer, run_async should return the same value."""

        async def async_func():
            return value

        result = run_async(lambda: async_func())
        assert result == value

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_async_function_returns_same_string(self, value):
        """For any string, run_async should return the same value."""

        async def async_func():
            return value

        result = run_async(lambda: async_func())
        assert result == value


class TestRunAsyncExceptionProperty:
    """Property-based tests for exception propagation in run_async."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_propagates_exception_with_message(self, error_message):
        """For any error message, run_async should propagate the exception with that message."""

        async def async_func():
            raise ValueError(error_message)

        with pytest.raises(ValueError) as exc_info:
            run_async(lambda: async_func())

        assert str(exc_info.value) == error_message
