"""Tests for AIFunction core initialization.

Tests for AIFunction class initialization, async detection, and metadata preservation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from ai_functions.core import AIFunction
from ai_functions.types.ai_function import AIFunctionConfig, CodeExecutionMode
from ai_functions.types.errors import AIFunctionError, ValidationError
from ai_functions.utils._template import Interpolation, Template
from ai_functions.validation.post_conditions import PostConditionResult


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestAIFunctionInitialization:
    """Tests for AIFunction initialization."""

    def test_raises_for_missing_return_type(self):
        """Test AIFunction raises ValueError for function without return type."""

        def no_return_func(a: str):
            """Function without return type."""
            return a

        with pytest.raises(ValueError, match="must specify a return type"):
            AIFunction(no_return_func, AIFunctionConfig())

    def test_accepts_json_serializable_in_disabled_mode(self):
        """Test AIFunction accepts JSON serializable types (like str) in DISABLED mode."""

        def str_return_func() -> str:
            """Function returning string."""
            return "test"

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(str_return_func, config)
        # str is JSON serializable, so structured output is enabled
        assert ai_func._is_structured_output_enabled is True

    def test_detects_sync_function(self):
        """Test AIFunction correctly detects sync function (is_async=False)."""

        def sync_func() -> str:
            """Sync function."""
            return "test"

        ai_func = AIFunction(sync_func, AIFunctionConfig())
        assert ai_func.is_async is False

    def test_detects_async_function(self):
        """Test AIFunction correctly detects async function (is_async=True)."""

        async def async_func() -> str:
            """Async function."""
            return "test"

        ai_func = AIFunction(async_func, AIFunctionConfig())
        assert ai_func.is_async is True

    def test_preserves_function_name(self):
        """Test AIFunction preserves the wrapped function's __name__."""

        def my_custom_function() -> str:
            """Custom function."""
            return "test"

        ai_func = AIFunction(my_custom_function, AIFunctionConfig())
        assert ai_func.__name__ == "my_custom_function"

    def test_preserves_function_docstring(self):
        """Test AIFunction preserves the wrapped function's __doc__."""

        def documented_func() -> str:
            """This is my custom docstring."""
            return "test"

        ai_func = AIFunction(documented_func, AIFunctionConfig())
        assert ai_func.__doc__ == "This is my custom docstring."

    def test_preserves_metadata_combined(self):
        """Test AIFunction preserves both __name__ and __doc__ together."""

        def my_func() -> str:
            """My docstring"""
            return "test"

        ai_func = AIFunction(my_func, AIFunctionConfig())
        assert ai_func.__name__ == "my_func"
        assert ai_func.__doc__ == "My docstring"

    def test_accepts_pydantic_return_with_disabled_mode(self):
        """Test AIFunction accepts Pydantic return type with DISABLED mode."""

        def pydantic_func() -> SampleModel:
            """Function returning Pydantic model."""
            return SampleModel(name="test", value=42)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(pydantic_func, config)
        assert ai_func._is_pydantic_return is True

    def test_accepts_pydantic_return_with_local_mode(self):
        """Test AIFunction accepts Pydantic return type with LOCAL mode."""

        def pydantic_func() -> SampleModel:
            """Function returning Pydantic model."""
            return SampleModel(name="test", value=42)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.LOCAL)
        ai_func = AIFunction(pydantic_func, config)
        assert ai_func._is_pydantic_return is True

    def test_accepts_non_pydantic_return_with_local_mode(self):
        """Test AIFunction accepts non-Pydantic return type with LOCAL mode."""

        def str_func() -> str:
            """Function returning string."""
            return "test"

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.LOCAL)
        ai_func = AIFunction(str_func, config)
        assert ai_func._is_pydantic_return is False


class TestAIFunctionAsyncDetectionProperty:
    """Property-based tests for AIFunction async detection."""

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_sync_function_detected_as_not_async(self, func_name: str, docstring: str):
        """For any sync function, AIFunction should set is_async=False."""

        # Create a sync function dynamically
        def sync_func() -> str:
            return "test"

        sync_func.__name__ = func_name
        sync_func.__doc__ = docstring

        ai_func = AIFunction(sync_func, AIFunctionConfig())
        assert ai_func.is_async is False

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_async_function_detected_as_async(self, func_name: str, docstring: str):
        """For any async function, AIFunction should set is_async=True."""

        # Create an async function dynamically
        async def async_func() -> str:
            return "test"

        async_func.__name__ = func_name
        async_func.__doc__ = docstring

        ai_func = AIFunction(async_func, AIFunctionConfig())
        assert ai_func.is_async is True


class TestAIFunctionMetadataPreservationProperty:
    """Property-based tests for AIFunction metadata preservation."""

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_metadata_preserved_for_sync_function(self, func_name: str, docstring: str):
        """For any function with a name and docstring, AIFunction should preserve both."""

        # Create a sync function dynamically
        def sync_func() -> str:
            return "test"

        sync_func.__name__ = func_name
        sync_func.__doc__ = docstring

        ai_func = AIFunction(sync_func, AIFunctionConfig())

        assert ai_func.__name__ == func_name
        assert ai_func.__doc__ == docstring

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_metadata_preserved_for_async_function(self, func_name: str, docstring: str):
        """For any async function with a name and docstring, AIFunction should preserve both."""

        # Create an async function dynamically
        async def async_func() -> str:
            return "test"

        async_func.__name__ = func_name
        async_func.__doc__ = docstring

        ai_func = AIFunction(async_func, AIFunctionConfig())

        assert ai_func.__name__ == func_name
        assert ai_func.__doc__ == docstring


class TestAIFunctionArgumentBinding:
    """Tests for AIFunction argument binding.

    Tests that _get_bound_arguments correctly maps positional and keyword
    arguments to parameter names, and applies default values.
    """

    def test_binds_positional_arguments(self):
        """Test AIFunction binds positional arguments to parameter names correctly."""

        def func(a: str, b: int) -> str:
            """Function with two positional parameters."""
            return f"{a}-{b}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments("hello", 42)

        assert bound == {"a": "hello", "b": 42}

    def test_binds_keyword_arguments(self):
        """Test AIFunction binds keyword arguments to parameter names correctly."""

        def func(a: str, b: int) -> str:
            """Function with two parameters."""
            return f"{a}-{b}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(a="hello", b=42)

        assert bound == {"a": "hello", "b": 42}

    def test_applies_defaults(self):
        """Test AIFunction applies default values for missing arguments."""

        def func(a: str, b: int = 10) -> str:
            """Function with default parameter."""
            return f"{a}-{b}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments("hello")

        assert bound == {"a": "hello", "b": 10}

    def test_binds_mixed_positional_and_keyword(self):
        """Test AIFunction binds mixed positional and keyword arguments."""

        def func(a: str, b: int, c: float) -> str:
            """Function with three parameters."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments("hello", c=3.14, b=42)

        assert bound == {"a": "hello", "b": 42, "c": 3.14}

    def test_applies_multiple_defaults(self):
        """Test AIFunction applies multiple default values."""

        def func(a: str, b: int = 10, c: float = 2.5) -> str:
            """Function with multiple defaults."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments("hello")

        assert bound == {"a": "hello", "b": 10, "c": 2.5}

    def test_overrides_defaults_with_provided_values(self):
        """Test AIFunction overrides defaults when values are provided."""

        def func(a: str, b: int = 10, c: float = 2.5) -> str:
            """Function with defaults."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments("hello", b=99)

        assert bound == {"a": "hello", "b": 99, "c": 2.5}

    def test_binds_no_arguments_function(self):
        """Test AIFunction handles function with no parameters."""

        def func() -> str:
            """Function with no parameters."""
            return "test"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments()

        assert bound == {}

    def test_binds_single_argument(self):
        """Test AIFunction binds single argument correctly."""

        def func(x: int) -> int:
            """Function with single parameter."""
            return x * 2

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(42)

        assert bound == {"x": 42}


class TestAIFunctionArgumentBindingProperty:
    """Property-based tests for AIFunction argument binding.

    For any function with parameters and any valid combination of positional/keyword
    arguments with defaults, AIFunction should correctly bind all arguments to their
    parameter names.
    """

    @given(
        a_val=st.text(min_size=1, max_size=50),
        b_val=st.integers(min_value=-1000, max_value=1000),
        c_val=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_positional_arguments_bound_correctly(self, a_val: str, b_val: int, c_val: float):
        """For any positional arguments, AIFunction should map them to parameter names correctly."""

        def func(a: str, b: int, c: float) -> str:
            """Function with three parameters."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(a_val, b_val, c_val)

        assert bound["a"] == a_val
        assert bound["b"] == b_val
        assert bound["c"] == c_val

    @given(
        a_val=st.text(min_size=1, max_size=50),
        b_val=st.integers(min_value=-1000, max_value=1000),
        c_val=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_keyword_arguments_bound_correctly(self, a_val: str, b_val: int, c_val: float):
        """For any keyword arguments, AIFunction should map them to parameter names correctly."""

        def func(a: str, b: int, c: float) -> str:
            """Function with three parameters."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(a=a_val, b=b_val, c=c_val)

        assert bound["a"] == a_val
        assert bound["b"] == b_val
        assert bound["c"] == c_val

    @given(
        a_val=st.text(min_size=1, max_size=50),
        default_b=st.integers(min_value=-1000, max_value=1000),
        default_c=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_defaults_applied_correctly(self, a_val: str, default_b: int, default_c: float):
        """For any function with defaults, AIFunction should apply default values for missing arguments."""

        # Create function with the generated default values
        def make_func(b_default: int, c_default: float):
            def func(a: str, b: int = b_default, c: float = c_default) -> str:
                """Function with defaults."""
                return f"{a}-{b}-{c}"

            return func

        func = make_func(default_b, default_c)
        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(a_val)

        assert bound["a"] == a_val
        assert bound["b"] == default_b
        assert bound["c"] == default_c

    @given(
        a_val=st.text(min_size=1, max_size=50),
        b_val=st.integers(min_value=-1000, max_value=1000),
        c_val=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_mixed_positional_and_keyword_bound_correctly(self, a_val: str, b_val: int, c_val: float):
        """For any mix of positional and keyword arguments, AIFunction should bind all correctly."""

        def func(a: str, b: int, c: float) -> str:
            """Function with three parameters."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())
        # Pass first as positional, rest as keyword
        bound = ai_func._get_bound_arguments(a_val, c=c_val, b=b_val)

        assert bound["a"] == a_val
        assert bound["b"] == b_val
        assert bound["c"] == c_val

    @given(
        a_val=st.text(min_size=1, max_size=50),
        default_b=st.integers(min_value=-1000, max_value=1000),
        override_b=st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=100)
    def test_provided_values_override_defaults(self, a_val: str, default_b: int, override_b: int):
        """For any function with defaults, provided values should override defaults."""

        def make_func(b_default: int):
            def func(a: str, b: int = b_default) -> str:
                """Function with default."""
                return f"{a}-{b}"

            return func

        func = make_func(default_b)
        ai_func = AIFunction(func, AIFunctionConfig())
        bound = ai_func._get_bound_arguments(a_val, b=override_b)

        assert bound["a"] == a_val
        assert bound["b"] == override_b


class TestAIFunctionPromptBuilding:
    """Tests for AIFunction prompt building.

    Tests that _build_prompt correctly uses function return value as prompt,
    falls back to docstring when return is None, substitutes placeholders,
    and raises ValueError when both are empty.
    """

    @pytest.mark.asyncio
    async def test_uses_function_return_string_as_prompt(self):
        """Test AIFunction uses function return string as prompt."""

        def func(name: str) -> str:
            """This docstring should not be used."""
            return f"Hello, {name}! Please help me."

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments("Alice")
        prompt = await ai_func._build_prompt(bound_args)

        assert "Hello, Alice! Please help me." in prompt
        assert "docstring" not in prompt.lower()
        assert "FinalAnswer" in prompt  # Structured output instructions added

    @pytest.mark.asyncio
    async def test_falls_back_to_docstring_when_return_is_none(self):
        """Test AIFunction falls back to docstring when function returns None."""

        def func(name: str) -> str:
            """Greet the user named {name}."""
            return None  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments("Bob")
        prompt = await ai_func._build_prompt(bound_args)

        assert "Bob" in prompt
        assert "Greet" in prompt

    @pytest.mark.asyncio
    async def test_substitutes_placeholders_in_docstring(self):
        """Test AIFunction substitutes {param} placeholders in docstring."""

        def func(city: str, country: str) -> str:
            """Find information about {city} in {country}."""
            return None  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments("Paris", "France")
        prompt = await ai_func._build_prompt(bound_args)

        assert "Paris" in prompt
        assert "France" in prompt
        assert "{city}" not in prompt
        assert "{country}" not in prompt

    @pytest.mark.asyncio
    async def test_raises_valueerror_when_both_empty(self):
        """Test AIFunction raises ValueError when both function return and docstring are empty."""

        def func() -> str:
            return ""

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()

        with pytest.raises(ValueError, match="empty prompt"):
            await ai_func._build_prompt(bound_args)

    @pytest.mark.asyncio
    async def test_raises_valueerror_when_no_docstring_and_none_return(self):
        """Test AIFunction raises ValueError when no docstring and function returns None."""

        def func() -> str:
            return None  # type: ignore

        # Remove docstring
        func.__doc__ = None

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()

        with pytest.raises(ValueError):
            await ai_func._build_prompt(bound_args)

    @pytest.mark.asyncio
    async def test_uses_return_value_over_docstring(self):
        """Test AIFunction prefers return value over docstring when both available."""

        def func(x: int) -> str:
            """This is the docstring with {x}."""
            return f"Return value: {x}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(42)
        prompt = await ai_func._build_prompt(bound_args)

        assert "Return value: 42" in prompt
        assert "docstring" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_handles_async_function_return_value(self):
        """Test AIFunction handles async function return value as prompt."""

        async def func(message: str) -> str:
            """Docstring fallback."""
            return f"Async prompt: {message}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments("test")
        prompt = await ai_func._build_prompt(bound_args)

        assert "Async prompt: test" in prompt

    @pytest.mark.asyncio
    async def test_raises_exception_when_function_raises(self):
        """Test AIFunction propagates exception when function raises."""

        def func(value: int) -> str:
            """Process the value {value}."""
            raise RuntimeError("Function error")

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(123)

        with pytest.raises(RuntimeError, match="Function error"):
            await ai_func._build_prompt(bound_args)

    @pytest.mark.asyncio
    async def test_whitespace_only_prompt_raises_valueerror(self):
        """Test AIFunction raises ValueError when prompt is whitespace only."""

        def func() -> str:
            return "   \n\t  "

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()

        with pytest.raises(ValueError, match="empty prompt"):
            await ai_func._build_prompt(bound_args)


class TestAIFunctionPromptFromReturnValueProperty:
    """Property-based tests for AIFunction prompt from return value.

    For any function that returns a non-empty string, AIFunction should use
    that string as the prompt.
    """

    @given(
        prompt_text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        param_value=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_function_return_string_used_as_prompt(self, prompt_text: str, param_value: str):
        """For any function that returns a non-empty string, AIFunction should use that string as the prompt."""

        # Create a function that returns the prompt_text with param_value interpolated
        def func(param: str) -> str:
            """This docstring should be ignored."""
            return f"{prompt_text} - {param}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(param_value)
        prompt = await ai_func._build_prompt(bound_args)

        # The prompt should contain what the function returns (plus system instructions)
        # Strip to handle edge cases like leading spaces
        expected = f"{prompt_text} - {param_value}".strip()
        assert expected in prompt or prompt.startswith(expected)

    @given(prompt_text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_function_return_string_takes_precedence_over_docstring(self, prompt_text: str):
        """For any function that returns a non-empty string, the return value should take precedence over docstring."""
        docstring_marker = "DOCSTRING_MARKER_SHOULD_NOT_APPEAR"

        def func() -> str:
            return prompt_text

        func.__doc__ = f"This is the docstring with {docstring_marker}"

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()
        prompt = await ai_func._build_prompt(bound_args)

        # The prompt should contain the return value, not the docstring
        assert prompt_text.strip() in prompt or prompt.startswith(prompt_text)
        assert docstring_marker not in prompt


class TestAIFunctionPromptFallbackToDocstringProperty:
    """Property-based tests for AIFunction prompt fallback to docstring.

    For any function that returns None but has a docstring, AIFunction should
    use the docstring as the prompt.
    """

    @given(
        docstring_text=st.text(min_size=5, max_size=200).filter(lambda s: s.strip() and "{" not in s and "}" not in s)
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_docstring_used_when_function_returns_none(self, docstring_text: str):
        """For any function that returns None but has a docstring, AIFunction should use the docstring."""

        def func() -> str:
            return None  # type: ignore

        func.__doc__ = docstring_text

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()
        prompt = await ai_func._build_prompt(bound_args)

        # The prompt should contain the docstring content
        # Note: tstr may process the docstring, so we check for key content
        assert docstring_text.strip() in prompt or prompt.strip() == docstring_text.strip()

    @given(
        docstring_text=st.text(min_size=5, max_size=200).filter(lambda s: s.strip() and "{" not in s and "}" not in s)
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_exception_propagated_when_function_raises(self, docstring_text: str):
        """For any function that raises an exception, AIFunction should propagate it."""

        def func() -> str:
            raise RuntimeError("Intentional error")

        func.__doc__ = docstring_text

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()

        with pytest.raises(RuntimeError, match="Intentional error"):
            await ai_func._build_prompt(bound_args)


class TestAIFunctionDocstringPlaceholderSubstitutionProperty:
    """Property-based tests for AIFunction docstring placeholder substitution.

    For any docstring with {param} placeholders and matching bound arguments,
    AIFunction should substitute the argument values.
    """

    @given(
        param_a=st.text(min_size=1, max_size=50).filter(lambda s: s.strip() and "{" not in s and "}" not in s),
        param_b=st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_single_placeholder_substituted(self, param_a: str, param_b: int):
        """For any docstring with a single placeholder, AIFunction should substitute the argument value."""

        def func(name: str, count: int) -> str:
            """Process {name} with count {count}."""
            return None  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(param_a, param_b)
        prompt = await ai_func._build_prompt(bound_args)

        # The placeholders should be substituted with actual values
        assert param_a in prompt
        assert str(param_b) in prompt
        assert "{name}" not in prompt
        assert "{count}" not in prompt

    @given(
        value1=st.text(min_size=1, max_size=30).filter(lambda s: s.strip() and "{" not in s and "}" not in s),
        value2=st.text(min_size=1, max_size=30).filter(lambda s: s.strip() and "{" not in s and "}" not in s),
        value3=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_multiple_placeholders_substituted(self, value1: str, value2: str, value3: float):
        """For any docstring with multiple placeholders, AIFunction should substitute all argument values."""

        def func(first: str, second: str, third: float) -> str:
            """Combine {first} with {second} and {third}."""
            return None  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(value1, value2, value3)
        prompt = await ai_func._build_prompt(bound_args)

        # All placeholders should be substituted
        assert value1 in prompt
        assert value2 in prompt
        assert str(value3) in prompt
        assert "{first}" not in prompt
        assert "{second}" not in prompt
        assert "{third}" not in prompt

    @given(param_value=st.text(min_size=1, max_size=50).filter(lambda s: s.strip() and "{" not in s and "}" not in s))
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_repeated_placeholder_substituted(self, param_value: str):
        """For any docstring with repeated placeholders, AIFunction should substitute all occurrences."""

        def func(item: str) -> str:
            """First mention of {item}, second mention of {item}, third mention of {item}."""
            return None  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments(param_value)
        prompt = await ai_func._build_prompt(bound_args)

        # The placeholder should be substituted in all occurrences
        assert prompt.count(param_value) >= 3
        assert "{item}" not in prompt


class TestAIFunctionResultExtraction:
    """Tests for AIFunction result extraction.

    Tests that _extract_result correctly extracts results from agent responses
    based on return type and code execution mode.
    """

    def test_extracts_structured_output_for_pydantic_return_type(self):
        """Test AIFunction extracts structured_output for Pydantic return type."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="test", value=42)

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response with structured_output
        class MockResponse:
            structured_output = SampleModel(name="extracted", value=100)

        result = ai_func._extract_result(MockResponse(), {})

        assert isinstance(result, SampleModel)
        assert result.name == "extracted"
        assert result.value == 100

    def test_extracts_from_invocation_state_for_non_pydantic_return(self):
        """Test AIFunction extracts from invocation_state for non-Pydantic return type."""

        def func() -> str:
            """Return a string."""
            return "test"

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response without structured_output
        class MockResponse:
            pass

        # Create a result object with answer attribute (wrapping required for non-Pydantic)
        class ResultWithAnswer:
            answer = "extracted_string"

        invocation_state = {"python_executor_result": ResultWithAnswer()}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == "extracted_string"

    def test_extracts_answer_from_invocation_state_result(self):
        """Test AIFunction extracts answer attribute from invocation_state result."""

        def func() -> int:
            """Return an integer."""
            return 42

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response without structured_output
        class MockResponse:
            pass

        # Create a result object with answer attribute
        class ResultWithAnswer:
            answer = 999

        invocation_state = {"python_executor_result": ResultWithAnswer()}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == 999

    def test_raises_validation_error_when_no_result_available_pydantic(self):
        """Test AIFunction raises AIFunctionError when no result available for Pydantic return in DISABLED mode."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="test", value=42)

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response without structured_output
        class MockResponse:
            structured_output = None

        # In DISABLED mode, AIFunctionError is raised when no structured output is available
        with pytest.raises(AIFunctionError, match="did not produce a structured output"):
            ai_func._extract_result(MockResponse(), {})

    def test_raises_ai_function_error_when_no_result_available_non_pydantic(self):
        """Test AIFunction raises AIFunctionError when no result available for non-Pydantic return."""

        def func() -> str:
            """Return a string."""
            return "test"

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response without structured_output
        class MockResponse:
            pass

        # Empty invocation_state
        invocation_state = {}

        # Raises AIFunctionError when no structured output is available in DISABLED mode
        with pytest.raises(AIFunctionError, match="did not produce a structured output"):
            ai_func._extract_result(MockResponse(), invocation_state)

    def test_raises_ai_function_error_in_disabled_mode_without_structured_output(self):
        """Test AIFunction raises AIFunctionError in DISABLED mode without structured_output."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="test", value=42)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        # Create a mock response without structured_output
        class MockResponse:
            structured_output = None

        with pytest.raises(AIFunctionError, match="did not produce a structured output"):
            ai_func._extract_result(MockResponse(), {})

    def test_disabled_mode_extracts_structured_output_when_available(self):
        """Test AIFunction in DISABLED mode extracts structured_output when available."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="test", value=42)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        # Create a mock response with structured_output
        class MockResponse:
            structured_output = SampleModel(name="disabled_mode", value=200)

        result = ai_func._extract_result(MockResponse(), {})

        assert isinstance(result, SampleModel)
        assert result.name == "disabled_mode"
        assert result.value == 200

    def test_extracts_int_from_invocation_state(self):
        """Test AIFunction extracts integer from invocation_state."""

        def func() -> int:
            """Return an integer."""
            return 42

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap in an object with answer attribute
        class ResultWithAnswer:
            answer = 12345

        invocation_state = {"python_executor_result": ResultWithAnswer()}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == 12345

    def test_extracts_list_from_invocation_state(self):
        """Test AIFunction extracts list from invocation_state."""

        def func() -> list[str]:
            """Return a list of strings."""
            return ["a", "b", "c"]

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap in an object with answer attribute
        class ResultWithAnswer:
            answer = ["extracted", "list", "items"]

        invocation_state = {"python_executor_result": ResultWithAnswer()}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == ["extracted", "list", "items"]


class TestAIFunctionResultExtractionPydanticProperty:
    """Property-based tests for AIFunction result extraction for Pydantic types.

    For any AIFunction with a Pydantic return type and a response with structured_output,
    the result should be the structured_output value.
    """

    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        value=st.integers(min_value=-10000, max_value=10000),
    )
    @settings(max_examples=100)
    def test_pydantic_result_extraction_returns_structured_output(self, name: str, value: int):
        """For any Pydantic model in structured_output, AIFunction should return it directly."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="placeholder", value=0)

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response with structured_output containing the generated values
        expected_model = SampleModel(name=name, value=value)

        class MockResponse:
            structured_output = expected_model

        result = ai_func._extract_result(MockResponse(), {})

        # The result should be exactly the structured_output
        assert result is expected_model
        assert result.name == name
        assert result.value == value

    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        value=st.integers(min_value=-10000, max_value=10000),
    )
    @settings(max_examples=100)
    def test_pydantic_result_extraction_ignores_invocation_state(self, name: str, value: int):
        """For Pydantic return types, AIFunction should use structured_output, not invocation_state."""

        def func() -> SampleModel:
            """Return a sample model."""
            return SampleModel(name="placeholder", value=0)

        ai_func = AIFunction(func, AIFunctionConfig())

        # Create a mock response with structured_output
        expected_model = SampleModel(name=name, value=value)

        class MockResponse:
            structured_output = expected_model

        # Even with invocation_state containing a different result, structured_output should be used
        invocation_state = {"python_executor_result": SampleModel(name="wrong", value=-999)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        # The result should be the structured_output, not invocation_state
        assert result is expected_model
        assert result.name == name
        assert result.value == value


class TestAIFunctionResultExtractionNonPydanticProperty:
    """Property-based tests for AIFunction result extraction for non-Pydantic types.

    For any AIFunction with a non-Pydantic return type and an invocation_state with a python_executor_result,
    the result should be extracted from invocation_state.
    """

    @given(string_value=st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_non_pydantic_string_extraction_from_invocation_state(self, string_value: str):
        """For any string result in invocation_state, AIFunction should return it directly."""

        def func() -> str:
            """Return a string."""
            return "placeholder"

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap value in an object with answer attribute
        class ResultWithAnswer:
            def __init__(self, answer):
                self.answer = answer

        invocation_state = {"python_executor_result": ResultWithAnswer(string_value)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == string_value

    @given(int_value=st.integers(min_value=-100000, max_value=100000))
    @settings(max_examples=100)
    def test_non_pydantic_int_extraction_from_invocation_state(self, int_value: int):
        """For any integer result in invocation_state, AIFunction should return it directly."""

        def func() -> int:
            """Return an integer."""
            return 0

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap value in an object with answer attribute
        class ResultWithAnswer:
            def __init__(self, answer):
                self.answer = answer

        invocation_state = {"python_executor_result": ResultWithAnswer(int_value)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == int_value

    @given(float_value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_non_pydantic_float_extraction_from_invocation_state(self, float_value: float):
        """For any float result in invocation_state, AIFunction should return it directly."""

        def func() -> float:
            """Return a float."""
            return 0.0

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap value in an object with answer attribute
        class ResultWithAnswer:
            def __init__(self, answer):
                self.answer = answer

        invocation_state = {"python_executor_result": ResultWithAnswer(float_value)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == float_value

    @given(list_values=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_non_pydantic_list_extraction_from_invocation_state(self, list_values: list):
        """For any list result in invocation_state, AIFunction should return it directly."""

        def func() -> list[str]:
            """Return a list of strings."""
            return []

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Wrap value in an object with answer attribute
        class ResultWithAnswer:
            def __init__(self, answer):
                self.answer = answer

        invocation_state = {"python_executor_result": ResultWithAnswer(list_values)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == list_values

    @given(answer_value=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_non_pydantic_extracts_answer_attribute_from_result(self, answer_value: str):
        """For any result with answer attribute in invocation_state, AIFunction should extract the answer."""

        def func() -> str:
            """Return a string."""
            return "placeholder"

        ai_func = AIFunction(func, AIFunctionConfig())

        class MockResponse:
            pass

        # Create a result object with answer attribute
        class ResultWithAnswer:
            def __init__(self, answer):
                self.answer = answer

        invocation_state = {"python_executor_result": ResultWithAnswer(answer_value)}

        result = ai_func._extract_result(MockResponse(), invocation_state)

        assert result == answer_value


class TestAgentToolsConfiguration:
    """Tests for tools configuration when creating the agent."""

    def test_user_tools_included_in_agent(self):
        """User-provided tools should be included when creating an agent."""

        def custom_tool(x: int) -> int:
            """Custom tool."""
            return x * 2

        config = AIFunctionConfig(tools=[custom_tool])

        def func() -> str:
            """Test function."""
            return "test"

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            assert custom_tool in tools

    def test_python_executor_included_in_local_mode(self):
        """Python executor should be included when code_execution_mode is LOCAL."""

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.LOCAL)

        def func() -> str:
            """Test function."""
            return "test"

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            # Python executor should be in the tools list
            tool_names = [getattr(t, "__name__", str(t)) for t in tools]
            assert "python_executor" in tool_names

    def test_python_executor_not_included_in_disabled_mode(self):
        """Python executor should NOT be included when code_execution_mode is DISABLED."""

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)

        def func() -> SampleModel:
            """Test function."""
            return SampleModel(name="test", value=1)

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            # No tool should be named python_executor
            tool_names = [getattr(t, "__name__", str(t)) for t in tools]
            assert "python_executor" not in tool_names

    def test_multiple_user_tools_all_included(self):
        """Multiple user-provided tools should all be included."""

        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        def tool_c(z: float) -> float:
            """Tool C."""
            return z

        config = AIFunctionConfig(tools=[tool_a, tool_b, tool_c])

        def func() -> str:
            """Test function."""
            return "test"

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            assert tool_a in tools
            assert tool_b in tools
            assert tool_c in tools

    def test_no_user_tools_only_executor_in_local_mode(self):
        """With no user tools in LOCAL mode, only python executor should be present."""

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.LOCAL)

        def func() -> str:
            """Test function."""
            return "test"

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            # Should have only python_executor
            assert len(tools) == 1
            tool_names = [getattr(t, "__name__", str(t)) for t in tools]
            assert "python_executor" in tool_names

    def test_no_tools_in_disabled_mode_without_user_tools(self):
        """With no user tools in DISABLED mode, tools list should be empty."""

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)

        def func() -> SampleModel:
            """Test function."""
            return SampleModel(name="test", value=1)

        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent:
            mock_agent.return_value = MagicMock()
            ai_func._create_agent({})

            call_kwargs = mock_agent.call_args.kwargs
            tools = call_kwargs["tools"]

            assert len(tools) == 0


class TestAsyncFunctionCall:
    """Tests for async function __call__ path."""

    @pytest.mark.asyncio
    async def test_async_function_returns_coroutine(self):
        """Test that calling an async AIFunction returns a coroutine that can be awaited."""

        async def async_func(x: int) -> SampleModel:
            """Async function."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(async_func, config)

        with patch.object(ai_func, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (SampleModel(name="test", value=42), [])

            result = await ai_func(5)

            assert isinstance(result, SampleModel)
            assert result.name == "test"
            assert result.value == 42


class TestAwaitMethod:
    """Tests for __await__ method."""

    def test_await_without_call_raises_typeerror(self):
        """Test that awaiting AIFunction without calling raises TypeError."""

        def func() -> str:
            """Test function."""
            return "test"

        ai_func = AIFunction(func, AIFunctionConfig())

        with pytest.raises(TypeError, match="Cannot await .* without calling it"):
            ai_func.__await__()


class TestExecuteAsyncRetryLogic:
    """Tests for retry logic in _execute_async."""

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self):
        """Test that _execute_async retries on ValidationError."""

        def func() -> SampleModel:
            """Test function."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED, max_attempts=2)
        ai_func = AIFunction(func, config)

        call_count = 0

        async def mock_run_agent(bound_args, messages):
            nonlocal call_count
            call_count += 1
            return SampleModel(name="test", value=call_count), messages

        validation_call_count = 0

        async def mock_validate(result, bound_args):
            nonlocal validation_call_count
            validation_call_count += 1
            if validation_call_count == 1:
                raise ValidationError(function_name="func", validation_errors={"check": "First attempt failed"})

        with patch.object(ai_func, "_run_agent", side_effect=mock_run_agent):
            with patch.object(ai_func, "_validate_result", side_effect=mock_validate):
                await ai_func._execute_async({})

        assert call_count == 2
        assert validation_call_count == 2

    @pytest.mark.asyncio
    async def test_raises_validation_error_after_max_attempts(self):
        """Test that _execute_async raises ValidationError after max attempts."""

        def func() -> SampleModel:
            """Test function."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED, max_attempts=1)
        ai_func = AIFunction(func, config)

        async def mock_run_agent(bound_args, messages):
            return SampleModel(name="test", value=1), messages

        async def mock_validate(result, bound_args):
            raise ValidationError(function_name="func", validation_errors={"check": "Always fails"})

        with patch.object(ai_func, "_run_agent", side_effect=mock_run_agent):
            with patch.object(ai_func, "_validate_result", side_effect=mock_validate):
                with pytest.raises(ValidationError):
                    await ai_func._execute_async({})

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception_in_ai_function_error(self):
        """Test that unexpected exceptions are wrapped in AIFunctionError."""

        def func() -> SampleModel:
            """Test function."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        async def mock_run_agent(bound_args, messages):
            raise RuntimeError("Unexpected error")

        with patch.object(ai_func, "_run_agent", side_effect=mock_run_agent):
            with pytest.raises(AIFunctionError) as exc_info:
                await ai_func._execute_async({})

            assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reraises_ai_function_error_as_is(self):
        """Test that AIFunctionError is re-raised without wrapping."""

        def func() -> SampleModel:
            """Test function."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        original_error = AIFunctionError(message="Original error", function_name="func")

        async def mock_run_agent(bound_args, messages):
            raise original_error

        with patch.object(ai_func, "_run_agent", side_effect=mock_run_agent):
            with pytest.raises(AIFunctionError) as exc_info:
                await ai_func._execute_async({})

            assert exc_info.value is original_error


class TestRunAgent:
    """Tests for _run_agent method."""

    @pytest.mark.asyncio
    async def test_run_agent_builds_prompt_on_first_call(self):
        """Test that _run_agent builds prompt when messages is empty."""

        def func(x: int) -> SampleModel:
            """Process {x}."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value=MagicMock(structured_output=SampleModel(name="test", value=1)))
        mock_agent.messages = [{"role": "user", "content": [{"text": "test"}]}]

        with patch.object(ai_func, "_create_agent", return_value=mock_agent):
            with patch.object(ai_func, "_build_prompt", new_callable=AsyncMock) as mock_prompt:
                mock_prompt.return_value = "Test prompt"

                result, messages = await ai_func._run_agent({"x": 5}, [])

                mock_prompt.assert_called_once_with({"x": 5})
                mock_agent.invoke_async.assert_called_once()
                assert mock_agent.invoke_async.call_args[0][0] == "Test prompt"

    @pytest.mark.asyncio
    async def test_run_agent_skips_prompt_on_retry(self):
        """Test that _run_agent passes None prompt to agent when messages exist (retry)."""

        def func(x: int) -> SampleModel:
            """Process {x}."""
            return None

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value=MagicMock(structured_output=SampleModel(name="test", value=1)))
        mock_agent.messages = [{"role": "user", "content": [{"text": "test"}]}]

        existing_messages = [{"role": "user", "content": [{"text": "previous"}]}]

        with patch.object(ai_func, "_create_agent", return_value=mock_agent):
            with patch.object(ai_func, "_build_prompt", new_callable=AsyncMock) as mock_prompt:
                mock_prompt.return_value = "Test prompt"
                result, messages = await ai_func._run_agent({"x": 5}, existing_messages)

                # _build_prompt is still called to get prompt for invocation_state
                mock_prompt.assert_called_once()
                # But agent is invoked with None prompt
                mock_agent.invoke_async.assert_called_once()
                assert mock_agent.invoke_async.call_args[0][0] is None
                assert mock_agent.invoke_async.call_args[0][0] is None


class TestBuildPromptTemplateHandling:
    """Tests for Template return handling in _build_prompt."""

    @pytest.mark.asyncio
    async def test_handles_template_return_value(self):
        """Test that _build_prompt handles Template return value."""

        def func(name: str) -> str:
            """Test function."""
            return Template("Hello, ", Interpolation(name, "name"), "!")

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments("World")

        prompt = await ai_func._build_prompt(bound_args)

        assert "Hello" in prompt
        assert "World" in prompt

    @pytest.mark.asyncio
    async def test_raises_typeerror_for_invalid_return_type(self):
        """Test that _build_prompt raises TypeError for invalid return type."""

        def func() -> str:
            """Test function."""
            return 12345  # type: ignore

        ai_func = AIFunction(func, AIFunctionConfig())
        bound_args = ai_func._get_bound_arguments()

        with pytest.raises(TypeError, match="must return str, Template, or None"):
            await ai_func._build_prompt(bound_args)


class TestValidateResult:
    """Tests for _validate_result method."""

    @pytest.mark.asyncio
    async def test_validate_result_passes_when_no_post_conditions(self):
        """Test that _validate_result passes when no post_conditions configured."""

        def func() -> str:
            """Test function."""
            return "test"

        config = AIFunctionConfig(post_conditions=None)
        ai_func = AIFunction(func, config)

        await ai_func._validate_result("result", {})

    @pytest.mark.asyncio
    async def test_validate_result_passes_when_all_conditions_pass(self):
        """Test that _validate_result passes when all conditions pass."""

        def func() -> str:
            """Test function."""
            return "test"

        def check_not_empty(result: str) -> bool:
            """Check result is not empty."""
            return len(result) > 0

        config = AIFunctionConfig(post_conditions=[check_not_empty])
        ai_func = AIFunction(func, config)

        with patch.object(ai_func._post_runner, "validate", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = [PostConditionResult(passed=True)]

            await ai_func._validate_result("result", {})

    @pytest.mark.asyncio
    async def test_validate_result_raises_when_condition_fails(self):
        """Test that _validate_result raises ValidationError when condition fails."""

        def func() -> str:
            """Test function."""
            return "test"

        def check_length(result: str) -> bool:
            """Check result length."""
            return len(result) > 100

        config = AIFunctionConfig(post_conditions=[check_length])
        ai_func = AIFunction(func, config)

        with patch.object(ai_func._post_runner, "validate", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = [PostConditionResult(passed=False, message="Result too short")]

            with pytest.raises(ValidationError) as exc_info:
                await ai_func._validate_result("short", {})

            assert "check_length" in str(exc_info.value.validation_errors)


class TestCreateAgentWithMessages:
    """Tests for _create_agent with conversation history."""

    def test_create_agent_passes_messages_to_agent(self):
        """Test that _create_agent passes messages to Agent constructor."""

        def func() -> SampleModel:
            """Test function."""
            return SampleModel(name="test", value=1)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
        ]

        with patch("ai_functions.core.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            ai_func._create_agent({}, messages)

            call_kwargs = mock_agent_class.call_args.kwargs
            assert call_kwargs["messages"] == messages

    def test_create_agent_uses_empty_list_when_no_messages(self):
        """Test that _create_agent uses empty list when messages is None."""

        def func() -> SampleModel:
            """Test function."""
            return SampleModel(name="test", value=1)

        config = AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED)
        ai_func = AIFunction(func, config)

        with patch("ai_functions.core.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            ai_func._create_agent({}, None)

            call_kwargs = mock_agent_class.call_args.kwargs
            assert call_kwargs["messages"] == []


class TestSignatureProperty:
    """Tests for __signature__ property."""

    def test_signature_returns_wrapped_function_signature(self):
        """Test that __signature__ returns the wrapped function's signature."""

        def func(a: str, b: int, c: float = 3.14) -> str:
            """Test function."""
            return f"{a}-{b}-{c}"

        ai_func = AIFunction(func, AIFunctionConfig())

        sig = ai_func.__signature__
        params = list(sig.parameters.keys())

        assert params == ["a", "b", "c"]
        assert sig.parameters["c"].default == 3.14
