"""Tests for ai_function decorator.

Tests:
- Usage patterns (bare, empty parens, config object)
- Configuration passing via AIFunctionConfig
- Post-condition validation
- Metadata preservation
- Property-based tests
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from ai_functions.core import AIFunction
from ai_functions.decorator import ai_function
from ai_functions.types import AIFunctionConfig, CodeExecutionMode
from ai_functions.types.ai_function import AgentKwargs


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestDecoratorUsagePatterns:
    """Tests for different decorator usage patterns."""

    def test_bare_decorator(self):
        """@ai_function without parentheses wraps the function."""

        @ai_function
        def my_func() -> str:
            """Test docstring."""

        assert isinstance(my_func, AIFunction)

    def test_empty_parens(self):
        """@ai_function() with empty parentheses wraps the function."""

        @ai_function()
        def my_func() -> str:
            """Test docstring."""

        assert isinstance(my_func, AIFunction)

    def test_with_config_object(self):
        """@ai_function(config=...) uses the provided config."""
        config = AIFunctionConfig(max_attempts=7)

        @ai_function(config=config)
        def my_func() -> str:
            """Test docstring."""

        assert isinstance(my_func, AIFunction)
        # Config is modified to add defaults, so check key attributes instead
        assert my_func.config.max_attempts == config.max_attempts

    def test_with_function_parameters(self):
        """Decorator works with functions that have parameters and captures them correctly."""

        @ai_function
        def my_func(name: str, count: int, optional: str = "default") -> str:
            """Process {name} with count {count} and {optional}."""

        assert isinstance(my_func, AIFunction)
        # Verify function signature is preserved
        import inspect

        sig = inspect.signature(my_func.func)
        params = list(sig.parameters.keys())
        assert params == ["name", "count", "optional"]
        # Verify parameter types are preserved
        assert sig.parameters["name"].annotation is str
        assert sig.parameters["count"].annotation is int
        assert sig.parameters["optional"].default == "default"
        # Verify bound arguments work
        bound = my_func._get_bound_arguments("Alice", 42)
        assert bound == {"name": "Alice", "count": 42, "optional": "default"}
        bound_with_optional = my_func._get_bound_arguments("Bob", 10, "custom")
        assert bound_with_optional == {"name": "Bob", "count": 10, "optional": "custom"}

    def test_with_async_function(self):
        """Decorator works with async functions."""

        @ai_function
        async def my_func() -> str:
            """Test docstring."""

        assert isinstance(my_func, AIFunction)
        assert my_func.is_async is True


class TestConfigurationPassing:
    """Tests for configuration parameters being passed correctly via AIFunctionConfig."""

    def test_max_retries(self):
        """max_attempts is passed via config."""

        @ai_function(config=AIFunctionConfig(max_attempts=15))
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.max_attempts == 15

    def test_system_prompt(self):
        """system_prompt is passed via config."""

        @ai_function(config=AIFunctionConfig(system_prompt="You are helpful."))
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.system_prompt == "You are helpful."

    def test_tools(self):
        """tools list is passed via config."""

        def custom_tool(x: int) -> int:
            """A tool."""
            return x * 2

        @ai_function(config=AIFunctionConfig(tools=[custom_tool]))
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.tools == [custom_tool]

    def test_code_execution_mode(self):
        """code_execution_mode is passed via config."""

        @ai_function(config=AIFunctionConfig(code_execution_mode=CodeExecutionMode.DISABLED))
        def my_func() -> SampleModel:
            """Test docstring."""

        assert my_func.config.code_execution_mode == CodeExecutionMode.DISABLED

    def test_post_conditions(self):
        """post_conditions list is passed via config."""

        def check(result: str) -> bool:
            return len(result) > 0

        @ai_function(config=AIFunctionConfig(post_conditions=[check]))
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.post_conditions == [check]

    def test_multiple_params(self):
        """Multiple config params are all passed."""
        config = AIFunctionConfig(
            max_attempts=10,
            system_prompt="Custom",
            code_execution_mode=CodeExecutionMode.LOCAL,
        )

        @ai_function(config=config)
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.max_attempts == 10
        assert my_func.config.system_prompt == "Custom"
        assert my_func.config.code_execution_mode == CodeExecutionMode.LOCAL

    def test_config_object_fields(self):
        """All fields from config object are accessible."""

        def check(result: str) -> bool:
            return True

        def tool(x: int) -> int:
            return x

        config = AIFunctionConfig(
            system_prompt="Full config",
            tools=[tool],
            post_conditions=[check],
            code_execution_mode=CodeExecutionMode.LOCAL,
            max_attempts=20,
        )

        @ai_function(config=config)
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.system_prompt == "Full config"
        assert my_func.config.tools == [tool]
        assert my_func.config.post_conditions == [check]
        assert my_func.config.code_execution_mode == CodeExecutionMode.LOCAL
        assert my_func.config.max_attempts == 20

    def test_agent_kwargs(self):
        """agent_kwargs is passed via config."""
        config = AIFunctionConfig(agent_kwargs=AgentKwargs(agent_id="test_id"))

        @ai_function(config=config)
        def my_func() -> str:
            """Test docstring."""

        # Default tool_executor and conversation_manager are added, so check that user values are present
        assert my_func.config.agent_kwargs["agent_id"] == "test_id"


class TestPostConditionValidation:
    """Tests for post-condition signature validation."""

    def test_raises_for_no_parameters(self):
        """Raises ValueError for post-condition with no parameters."""

        def invalid():
            return True

        with pytest.raises(ValueError, match="must have at least one parameter"):

            @ai_function(config=AIFunctionConfig(post_conditions=[invalid]))
            def my_func() -> str:
                """Test docstring."""

    def test_raises_for_only_kwargs(self):
        """Raises ValueError for post-condition with only **kwargs."""

        def invalid(**kwargs):
            return True

        with pytest.raises(ValueError, match="must have a positional parameter"):

            @ai_function(config=AIFunctionConfig(post_conditions=[invalid]))
            def my_func() -> str:
                """Test docstring."""

    def test_raises_for_args_first(self):
        """Raises ValueError for post-condition with *args as first parameter."""

        def invalid(*args):
            return True

        with pytest.raises(ValueError, match="must have a named first parameter"):

            @ai_function(config=AIFunctionConfig(post_conditions=[invalid]))
            def my_func() -> str:
                """Test docstring."""

    def test_raises_for_keyword_only_first(self):
        """Raises ValueError for post-condition with keyword-only first parameter."""

        def invalid(*, result):
            return True

        with pytest.raises(ValueError, match="first parameter must be positional"):

            @ai_function(config=AIFunctionConfig(post_conditions=[invalid]))
            def my_func() -> str:
                """Test docstring."""

    def test_raises_for_param_mismatch(self):
        """Raises ValueError for post-condition with non-matching parameter names."""

        def invalid(result: str, nonexistent: int) -> bool:
            return True

        with pytest.raises(ValueError, match="doesn't exist in AI function"):

            @ai_function(config=AIFunctionConfig(post_conditions=[invalid]))
            def my_func(name: str) -> str:
                """Test docstring."""

    def test_valid_post_condition_accepted(self):
        """Valid post-condition is accepted."""

        def valid(result: str, name: str) -> bool:
            return True

        @ai_function(config=AIFunctionConfig(post_conditions=[valid]))
        def my_func(name: str) -> str:
            """Test docstring."""

        assert my_func.config.post_conditions == [valid]


class TestMetadataPreservation:
    """Tests for preserving wrapped function metadata."""

    def test_preserves_name(self):
        """Preserves __name__ attribute."""

        @ai_function
        def my_custom_name() -> str:
            """Test docstring."""

        assert my_custom_name.__name__ == "my_custom_name"

    def test_preserves_docstring(self):
        """Preserves __doc__ attribute."""

        @ai_function
        def my_func() -> str:
            """This is my docstring."""

        assert my_func.__doc__ == "This is my docstring."

    def test_preserves_with_config(self):
        """Preserves metadata when using config object."""
        config = AIFunctionConfig(max_attempts=5)

        @ai_function(config=config)
        def config_func() -> str:
            """Config docstring."""

        assert config_func.__name__ == "config_func"
        assert config_func.__doc__ == "Config docstring."

    def test_preserves_for_async(self):
        """Preserves metadata for async functions."""

        @ai_function
        async def async_func() -> str:
            """Async docstring."""

        assert async_func.__name__ == "async_func"
        assert async_func.__doc__ == "Async docstring."


class TestPropertyBasedConfigPassing:
    """Property-based tests for configuration passing via AIFunctionConfig."""

    @given(max_attempts=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_any_max_retries(self, max_attempts: int):
        """Any max_attempts value is passed via config."""

        @ai_function(config=AIFunctionConfig(max_attempts=max_attempts))
        def func() -> str:
            """Test."""

        assert func.config.max_attempts == max_attempts

    @given(system_prompt=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
    @settings(max_examples=50)
    def test_any_system_prompt(self, system_prompt: str):
        """Any system_prompt value is passed via config."""

        @ai_function(config=AIFunctionConfig(system_prompt=system_prompt))
        def func() -> str:
            """Test."""

        assert func.config.system_prompt == system_prompt

    @given(code_mode=st.sampled_from([CodeExecutionMode.LOCAL, CodeExecutionMode.DISABLED]))
    @settings(max_examples=10)
    def test_any_code_execution_mode(self, code_mode: CodeExecutionMode):
        """Any code_execution_mode value is passed via config."""

        @ai_function(config=AIFunctionConfig(code_execution_mode=code_mode))
        def func() -> SampleModel:
            """Test."""

        assert func.config.code_execution_mode == code_mode


class TestPropertyBasedMetadataPreservation:
    """Property-based tests for metadata preservation."""

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_any_name_and_docstring_preserved(self, func_name: str, docstring: str):
        """Any function name and docstring are preserved."""

        def func() -> str:
            pass

        func.__name__ = func_name
        func.__doc__ = docstring

        decorated = ai_function(func)

        assert decorated.__name__ == func_name
        assert decorated.__doc__ == docstring

    @given(
        func_name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        docstring=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        max_attempts=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=50)
    def test_preserved_with_any_config(self, func_name: str, docstring: str, max_attempts: int):
        """Metadata preserved regardless of config."""

        def func() -> str:
            pass

        func.__name__ = func_name
        func.__doc__ = docstring

        decorated = ai_function(config=AIFunctionConfig(max_attempts=max_attempts))(func)

        assert decorated.__name__ == func_name
        assert decorated.__doc__ == docstring


class TestInlineKwargsUsage:
    """Tests for inline kwargs usage patterns (without config object)."""

    def test_inline_config_kwargs(self):
        """@ai_function(max_attempts=10) passes config kwargs inline."""

        @ai_function(max_attempts=15, system_prompt="Be helpful")
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.max_attempts == 15
        assert my_func.config.system_prompt == "Be helpful"

    def test_inline_agent_kwargs(self):
        """@ai_function(temperature=0.7) passes agent kwargs."""

        @ai_function(agent_id="test_id")
        def my_func() -> str:
            """Test docstring."""

        # Default tool_executor and conversation_manager are added, so check that user values are present
        assert my_func.config.agent_kwargs["agent_id"] == "test_id"

    def test_mixed_inline_kwargs(self):
        """@ai_function with both config and agent kwargs inline."""

        @ai_function(max_attempts=20, system_prompt="Custom", agent_id="test_id")
        def my_func() -> str:
            """Test docstring."""

        # Config kwargs go to config
        assert my_func.config.max_attempts == 20
        assert my_func.config.system_prompt == "Custom"
        # Non-config kwargs go to agent_kwargs
        assert my_func.config.agent_kwargs["agent_id"] == "test_id"

    def test_config_with_override_kwargs(self):
        """@ai_function(config=..., max_attempts=20) overrides config values."""
        config = AIFunctionConfig(max_attempts=5, system_prompt="Original")

        @ai_function(config=config, max_attempts=20)
        def my_func() -> str:
            """Test docstring."""

        # max_attempts should be overridden
        assert my_func.config.max_attempts == 20
        # system_prompt should remain from config
        assert my_func.config.system_prompt == "Original"

    def test_config_with_agent_kwargs(self):
        """@ai_function(config=..., temperature=0.5) passes agent kwargs."""
        config = AIFunctionConfig(max_attempts=10)

        @ai_function(config=config, agent_id="test_id")
        def my_func() -> str:
            """Test docstring."""

        assert my_func.config.max_attempts == 10
        # Default tool_executor and conversation_manager are added, so check that user values are present
        assert my_func.config.agent_kwargs["agent_id"] == "test_id"

    def test_config_with_override_and_agent_kwargs(self):
        """@ai_function(config=..., max_attempts=20, agent_id="test_id_updated") combines all."""
        config = AIFunctionConfig(
            max_attempts=5, system_prompt="Original", agent_kwargs=AgentKwargs(agent_id="test_id")
        )

        @ai_function(config=config, max_attempts=20, agent_id="test_id_updated")
        def my_func() -> str:
            """Test docstring."""

        # Config override
        assert my_func.config.max_attempts == 20
        assert my_func.config.system_prompt == "Original"
        # Agent kwargs (with defaults)
        assert my_func.config.agent_kwargs["agent_id"] == "test_id_updated"

    def test_bare_decorator_has_empty_agent_kwargs(self):
        """@ai_function has default agent_kwargs (tool_executor and conversation_manager)."""

        @ai_function
        def my_func() -> str:
            """Test docstring."""

        # Default tool_executor and conversation_manager are added
        assert len(my_func.config.agent_kwargs) == 0
