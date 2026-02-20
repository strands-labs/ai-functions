"""Tests for error types.

This module tests AIFunctionError and ValidationError classes.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from ai_functions.types.errors import AIFunctionError, ValidationError


class TestAIFunctionError:
    """Unit tests for AIFunctionError class."""

    def test_includes_function_name(self):
        """Test AIFunctionError includes function_name in formatted message."""
        error = AIFunctionError(message="test", function_name="my_func")
        error_str = str(error)
        assert "[my_func]" in error_str
        assert "test" in error_str

    def test_cause_accessible_via_dunder(self):
        """Test original exception is accessible via __cause__ when using 'from'."""
        original = ValueError("original error")
        try:
            raise AIFunctionError(message="test", function_name="test_func") from original
        except AIFunctionError as error:
            assert error.__cause__ is original

    def test_stores_attributes_correctly(self):
        """Test AIFunctionError stores all attributes."""
        error = AIFunctionError(
            message="my message",
            function_name="func_name",
        )
        assert error.message == "my message"
        assert error.function_name == "func_name"


class TestValidationError:
    """Unit tests for ValidationError class."""

    def test_inherits_from_ai_function_error(self):
        """Test ValidationError inherits from AIFunctionError."""
        assert issubclass(ValidationError, AIFunctionError)
        error = ValidationError(function_name="test_func", validation_errors={"check": "error1"})
        assert isinstance(error, AIFunctionError)
        assert isinstance(error, Exception)

    def test_stores_attributes_correctly(self):
        """Test ValidationError stores all attributes."""
        error = ValidationError(
            function_name="my_func",
            validation_errors={"check1": "err1", "check2": "err2"},
        )
        assert error.validation_errors == {"check1": "err1", "check2": "err2"}
        assert error.function_name == "my_func"

    def test_all_validation_errors_formatted(self):
        """Test ValidationError formats all errors correctly."""
        error = ValidationError(
            function_name="process",
            validation_errors={"check_format": "invalid format", "check_field": "missing field"},
        )
        expected = "[process] Validation failed:\n- check_format: invalid format\n- check_field: missing field"
        assert str(error) == expected


class TestAIFunctionErrorPropertyBased:
    """Property-based tests for AIFunctionError."""

    @settings(max_examples=100)
    @given(
        message=st.text(min_size=1, max_size=100),
        function_name=st.text(min_size=1, max_size=50),
    )
    def test_message_and_function_name_in_output(self, message: str, function_name: str):
        """For any message and function_name, both should appear in formatted output."""
        error = AIFunctionError(message=message, function_name=function_name)
        error_str = str(error)
        assert message in error_str
        assert function_name in error_str

    @settings(max_examples=100)
    @given(
        message=st.text(min_size=1, max_size=100),
        function_name=st.text(min_size=1, max_size=50),
    )
    def test_message_always_in_output(self, message: str, function_name: str):
        """For any message, it should appear in formatted output."""
        error = AIFunctionError(message=message, function_name=function_name)
        assert message in str(error)


class TestValidationErrorPropertyBased:
    """Property-based tests for ValidationError."""

    @settings(max_examples=100)
    @given(
        function_name=st.text(min_size=1, max_size=50),
        validation_errors=st.dictionaries(
            st.text(min_size=1, max_size=30),
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=5,
        ),
    )
    def test_function_name_in_output(self, function_name: str, validation_errors: dict):
        """For any function_name, it should appear in formatted output."""
        error = ValidationError(function_name=function_name, validation_errors=validation_errors)
        error_str = str(error)
        assert function_name in error_str

    @settings(max_examples=100)
    @given(
        function_name=st.text(min_size=1, max_size=50),
        validation_errors=st.dictionaries(
            st.text(min_size=1, max_size=30),
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=5,
        ),
    )
    def test_all_validation_errors_in_output(self, function_name: str, validation_errors: dict):
        """For any dict of validation_errors, all keys and values should appear in formatted output."""
        error = ValidationError(function_name=function_name, validation_errors=validation_errors)
        error_str = str(error)
        for name, msg in validation_errors.items():
            assert name in error_str
            assert msg in error_str
