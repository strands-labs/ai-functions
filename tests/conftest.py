"""Shared fixtures for comprehensive unit tests.

This module provides common fixtures used across all test modules.
"""

from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from strands import Agent

from ai_functions.types.ai_function import AIFunctionConfig, PostConditionResult


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class AnotherModel(BaseModel):
    """Another sample Pydantic model for testing."""

    title: str
    count: int
    active: bool = True


@pytest.fixture
def sample_pydantic_model():
    """Create a sample Pydantic model class for testing.

    Returns:
        The SampleModel class (not an instance)
    """
    return SampleModel


@pytest.fixture
def sample_pydantic_instance():
    """Create a sample Pydantic model instance for testing.

    Returns:
        An instance of SampleModel
    """
    return SampleModel(name="test", value=42)


@pytest.fixture
def sample_ai_function_config():
    """Create a sample AIFunctionConfig for testing.

    Returns:
        A default AIFunctionConfig instance
    """
    return AIFunctionConfig()


@pytest.fixture
def sample_post_condition():
    """Create a sample post-condition function for testing.

    Returns:
        A simple post-condition that checks result length
    """

    def check_result(result: str) -> bool:
        return len(result) > 0

    return check_result


@pytest.fixture
def sample_post_condition_with_kwargs():
    """Create a sample post-condition that accepts **kwargs.

    Returns:
        A post-condition that accepts any additional arguments
    """

    def check_result_with_context(result: str, **kwargs) -> bool:
        return len(result) > 0 and len(kwargs) >= 0

    return check_result_with_context


@pytest.fixture
def sample_post_condition_result_passed():
    """Create a passed PostConditionResult for testing.

    Returns:
        A PostConditionResult with passed=True
    """
    return PostConditionResult(passed=True)


@pytest.fixture
def sample_post_condition_result_failed():
    """Create a failed PostConditionResult for testing.

    Returns:
        A PostConditionResult with passed=False and a message
    """
    return PostConditionResult(passed=False, message="Test error message")


@pytest.fixture
def sample_sync_function():
    """Create a sample sync function for testing.

    Returns:
        A simple sync function with type hints
    """

    def sync_func(a: str, b: int = 10) -> str:
        """Sample sync function docstring."""
        return f"{a}-{b}"

    return sync_func


@pytest.fixture
def sample_async_function():
    """Create a sample async function for testing.

    Returns:
        A simple async function with type hints
    """

    async def async_func(a: str, b: int = 10) -> str:
        """Sample async function docstring."""
        return f"{a}-{b}"

    return async_func


@pytest.fixture
def sample_function_no_return_type():
    """Create a sample function without return type for testing.

    Returns:
        A function without return type annotation
    """

    def no_return_func(a: str):
        """Function without return type."""
        return a

    return no_return_func


@pytest.fixture
def sample_function_pydantic_return():
    """Create a sample function with Pydantic return type.

    Returns:
        A function that returns a Pydantic model
    """

    def pydantic_return_func(name: str, value: int) -> SampleModel:
        """Function returning Pydantic model."""
        return SampleModel(name=name, value=value)

    return pydantic_return_func


@pytest.fixture
def mock_summarization_agent():
    """Create a mock summarization agent for testing.

    Returns:
        A mock Agent instance that can be used for conversation summarization
    """
    mock_agent = Mock(spec=Agent)
    mock_agent.run = Mock(return_value="Mock summary of conversation")
    return mock_agent
