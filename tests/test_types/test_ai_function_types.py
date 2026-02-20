"""Tests for AI function types.

Tests for CodeExecutionMode enum and PostConditionResult model.
"""

import dataclasses
from typing import get_type_hints

from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from ai_functions.types.ai_function import (
    AIFunctionConfig,
    AIFunctionKwargs,
    CodeExecutionMode,
    PostConditionResult,
)


class TestCodeExecutionMode:
    """Tests for CodeExecutionMode enum."""

    def test_has_local_value(self):
        """Test CodeExecutionMode has LOCAL value with correct string."""
        assert CodeExecutionMode.LOCAL.value == "local"

    def test_has_disabled_value(self):
        """Test CodeExecutionMode has DISABLED value with correct string."""
        assert CodeExecutionMode.DISABLED.value == "disabled"


class TestPostConditionResult:
    """Tests for PostConditionResult model."""

    def test_is_pydantic_model(self):
        """Test PostConditionResult is a Pydantic BaseModel."""
        assert issubclass(PostConditionResult, BaseModel)

    def test_stores_passed_status_true(self):
        """Test PostConditionResult stores passed=True correctly."""
        result = PostConditionResult(passed=True)
        assert result.passed is True

    def test_stores_passed_status_false_with_message(self):
        """Test PostConditionResult stores passed=False correctly when message is provided."""
        result = PostConditionResult(passed=False, message="validation failed")
        assert result.passed is False

    def test_requires_message_when_passed_is_false(self):
        """Test PostConditionResult requires message when passed=False."""
        import pytest

        with pytest.raises(ValueError, match="message is required when passed is False"):
            PostConditionResult(passed=False)

    def test_stores_message(self):
        """Test PostConditionResult stores message correctly."""
        result = PostConditionResult(passed=False, message="test error message")
        assert result.message == "test error message"

    def test_message_defaults_to_none(self):
        """Test PostConditionResult message defaults to None when not provided."""
        result = PostConditionResult(passed=True)
        assert result.message is None

    def test_stores_empty_message(self):
        """Test PostConditionResult stores empty message correctly."""
        result = PostConditionResult(passed=False, message="")
        assert result.message == ""


class TestPostConditionResultProperty:
    """Property-based tests for PostConditionResult."""

    @given(message=st.text(min_size=0, max_size=500))
    def test_message_storage(self, message: str):
        """For any message string, a PostConditionResult created with that
        message should store it correctly.
        """
        result = PostConditionResult(passed=False, message=message)
        assert result.message == message


class TestAIFunctionConfigKwargsSync:
    """Tests to ensure AIFunctionConfig and AIFunctionKwargs stay in sync."""

    def test_config_and_kwargs_have_same_fields(self):
        """Test that AIFunctionKwargs has exactly the same fields as AIFunctionConfig.

        This ensures that when AIFunctionConfig is updated, AIFunctionKwargs is also updated.
        """
        config_fields = {f.name for f in dataclasses.fields(AIFunctionConfig)}
        kwargs_fields = set(AIFunctionKwargs.__annotations__.keys())

        assert config_fields == kwargs_fields, (
            f"AIFunctionConfig and AIFunctionKwargs fields must match.\n"
            f"Missing in AIFunctionKwargs: {config_fields - kwargs_fields}\n"
            f"Extra in AIFunctionKwargs: {kwargs_fields - config_fields}"
        )

    def test_config_and_kwargs_have_same_types(self):
        """Test that AIFunctionKwargs has the same types as AIFunctionConfig.

        This ensures type consistency between the dataclass and TypedDict.
        """
        config_hints = get_type_hints(AIFunctionConfig)
        kwargs_hints = get_type_hints(AIFunctionKwargs)

        for field_name in config_hints:
            assert field_name in kwargs_hints, f"Field {field_name} missing in AIFunctionKwargs"
            assert config_hints[field_name] == kwargs_hints[field_name], (
                f"Type mismatch for field '{field_name}':\n"
                f"  AIFunctionConfig: {config_hints[field_name]}\n"
                f"  AIFunctionKwargs: {kwargs_hints[field_name]}"
            )
