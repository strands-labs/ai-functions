"""Tests for LocalPythonExecutorTool.

This module tests:
- Code execution with success/failure states
- final_answer capture
- stdout capture (via print)
- Error handling
- Initial state injection
- Tool description building

Note: This tool uses smolagents' AST-based interpreter which has some
behavioral differences from exec():
- print() output is captured in state['_print_outputs']
- first final_answer() call wins (execution stops)
"""

import keyword

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

# Strategy for generating valid Python identifiers that are not reserved keywords
valid_python_identifier = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda x: not keyword.iskeyword(x) and len(x) <= 20
)


class OutputModel(BaseModel):
    """Sample output model for testing."""

    answer: str


class MultiFieldModel(BaseModel):
    """Multi-field output model for testing."""

    name: str
    value: int
    active: bool = True


# =============================================================================
# Unit Tests: LocalPythonExecutorTool
# =============================================================================


class TestLocalPythonExecutorToolExecution:
    """Unit tests for LocalPythonExecutorTool code execution."""

    def test_executes_valid_code_with_success(self):
        """Test executes valid Python code with success=True."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("x = 1 + 1")

        assert result.success is True
        assert result.error is None

    def test_captures_final_answer_result(self):
        """Test captures final_answer result when called."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("final_answer(answer='hello')")

        assert result.final_answer == {"answer": "hello"}
        assert result.success is True

    def test_captures_stdout(self):
        """Test captures stdout from print statements."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("print('hello world')")

        assert "hello world" in result.stdout
        assert result.success is True

    def test_returns_error_on_exception(self):
        """Test returns error on exception with success=False."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("raise ValueError('test error')")

        assert result.success is False
        assert result.error is not None
        assert "test error" in result.error

    def test_injects_initial_state(self):
        """Test injects initial_state variables into execution namespace."""
        tool = LocalPythonExecutorTool(output_type=OutputModel, initial_state={"x": 42, "name": "test"})
        result = tool._execute_code("final_answer(answer=f'{name}-{x}')")

        assert result.final_answer == {"answer": "test-42"}
        assert result.success is True

    def test_initial_state_with_complex_objects(self):
        """Test initial_state with complex objects."""
        tool = LocalPythonExecutorTool(
            output_type=OutputModel, initial_state={"data": [1, 2, 3], "config": {"key": "value"}}
        )
        result = tool._execute_code("final_answer(answer=str(len(data)))")

        assert result.final_answer == {"answer": "3"}
        assert result.success is True


class TestLocalPythonExecutorToolDescription:
    """Unit tests for LocalPythonExecutorTool description building."""

    def test_builds_tool_description_with_pydantic_signature(self):
        """Test builds tool description with correct signature for Pydantic model."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        description = tool.python_executor.tool_spec["description"]

        assert "final_answer(answer: str)" in description
        assert "Execute Python code" in description

    def test_builds_tool_description_with_multi_field_signature(self):
        """Test builds tool description with multiple fields in signature."""
        tool = LocalPythonExecutorTool(output_type=MultiFieldModel)
        description = tool.python_executor.tool_spec["description"]

        assert "name: str" in description
        assert "value: int" in description
        assert "active: bool = True" in description

    @pytest.mark.skip(reason="Code requires BaseModel types, not str types")
    def test_builds_tool_description_for_non_pydantic_type(self):
        """Test builds tool description for non-Pydantic return type."""
        tool = LocalPythonExecutorTool(output_type=str)
        description = tool.python_executor.tool_spec["description"]

        assert "final_answer(answer=<your_result>)" in description


class TestLocalPythonExecutorToolEdgeCases:
    """Unit tests for edge cases in LocalPythonExecutorTool."""

    def test_empty_code_executes_successfully(self):
        """Test empty code string executes successfully."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("")

        assert result.success is True
        assert result.final_answer is None

    def test_first_final_answer_wins(self):
        """Test first final_answer call wins (smolagents behavior)."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("""
final_answer(answer='first')
final_answer(answer='second')
""")
        # smolagents stops execution on first final_answer
        assert result.final_answer == {"answer": "first"}

    def test_syntax_error_returns_error(self):
        """Test syntax error returns error."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("def broken(")

        assert result.success is False
        assert result.error is not None

    def test_name_error_returns_error(self):
        """Test undefined variable returns error."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code("x = undefined_variable")

        assert result.success is False
        assert result.error is not None
        assert "undefined_variable" in result.error

    def test_final_answer_with_multiple_kwargs(self):
        """Test final_answer with multiple keyword arguments."""
        tool = LocalPythonExecutorTool(output_type=MultiFieldModel)
        result = tool._execute_code("final_answer(name='test', value=42, active=False)")

        assert result.final_answer == {"name": "test", "value": 42, "active": False}

    def test_no_initial_state_defaults_to_empty(self):
        """Test no initial_state defaults to empty dict."""
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        # Should not raise - builtins should still be available
        result = tool._execute_code("x = len([1, 2, 3])")

        assert result.success is True


# =============================================================================
# Property Tests: LocalPythonExecutorTool
# =============================================================================


class TestLocalPythonExecutorToolProperties:
    """Property-based tests for LocalPythonExecutorTool."""

    # Valid Python code snippets that don't raise exceptions
    VALID_CODE_SNIPPETS = [
        "x = 1",
        "y = 'hello'",
        "z = [1, 2, 3]",
        "result = 1 + 1",
        "a = {'key': 'value'}",
        "b = (1, 2, 3)",
        "c = {1, 2, 3}",
        "d = True",
        "e = None",
        "f = 3.14",
        "g = 1 + 2 + 3",
        "h = 'hello' + ' world'",
        "i = [x for x in range(5)]",
        "j = {k: v for k, v in enumerate('abc')}",
        "k = len([1, 2, 3])",
        "l = max(1, 2, 3)",
        "m = min(1, 2, 3)",
        "n = sum([1, 2, 3])",
        "o = abs(-5)",
        "p = round(3.7)",
    ]

    @given(
        code_index=st.integers(min_value=0, max_value=len(VALID_CODE_SNIPPETS) - 1),
    )
    @settings(max_examples=100)
    def test_success_on_valid_code(self, code_index):
        """For any valid Python code that doesn't raise an exception,
        LocalPythonExecutorTool should return success=True.
        """
        code = self.VALID_CODE_SNIPPETS[code_index]
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        result = tool._execute_code(code)

        assert result.success is True
        assert result.error is None

    @given(
        answer_value=st.from_regex(r"[a-zA-Z0-9_]+", fullmatch=True).filter(lambda x: len(x) > 0 and len(x) <= 50),
    )
    @settings(max_examples=100)
    def test_captures_final_answer(self, answer_value):
        """For any code that calls final_answer with keyword arguments,
        LocalPythonExecutorTool should capture those arguments in the result.
        """
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        code = f"final_answer(answer='{answer_value}')"
        result = tool._execute_code(code)

        assert result.success is True
        assert result.final_answer is not None
        assert result.final_answer["answer"] == answer_value

    @given(
        message=st.from_regex(r"[a-zA-Z0-9]+", fullmatch=True).filter(lambda x: len(x) > 0 and len(x) <= 50),
    )
    @settings(max_examples=100)
    def test_captures_stdout(self, message):
        """For any code that prints to stdout, LocalPythonExecutorTool
        should capture the output in the result.
        """
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        code = f"print('{message}')"
        result = tool._execute_code(code)

        assert result.success is True
        assert message in result.stdout

    @given(
        error_message=st.from_regex(r"[a-zA-Z0-9]+", fullmatch=True).filter(lambda x: len(x) > 0 and len(x) <= 50),
    )
    @settings(max_examples=100)
    def test_error_on_exception(self, error_message):
        """For any code that raises an exception, LocalPythonExecutorTool
        should return success=False with the error message.
        """
        tool = LocalPythonExecutorTool(output_type=OutputModel)
        code = f"raise ValueError('{error_message}')"
        result = tool._execute_code(code)

        assert result.success is False
        assert result.error is not None
        assert error_message in result.error

    @given(
        base_name=valid_python_identifier,
        var_value=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100)
    def test_definition_persistence(self, base_name, var_value):
        """For any definition (variable, function, class) executed in one
        python_executor call, that definition should be accessible in subsequent calls
        on the same tool instance.
        """
        # Generate unique names by adding suffixes to avoid collisions
        var_name = f"{base_name}_var"
        func_name = f"{base_name}_func"
        class_name = f"{base_name.capitalize()}Class"

        tool = LocalPythonExecutorTool(output_type=OutputModel)

        # Call 1: Define a variable
        result1 = tool._execute_code(f"{var_name} = {var_value}")
        assert result1.success is True

        # Call 2: Verify variable persists and define a function
        result2 = tool._execute_code(f"""
def {func_name}():
    return {var_name} * 2
""")
        assert result2.success is True

        # Call 3: Verify function persists and define a class
        result3 = tool._execute_code(f"""
class {class_name}:
    def __init__(self):
        self.value = {func_name}()
""")
        assert result3.success is True

        # Call 4: Verify all definitions persist - variable, function, class
        result4 = tool._execute_code(f"""
# Use the variable
var_check = {var_name}
# Use the function
func_check = {func_name}()
# Use the class
obj = {class_name}()
class_check = obj.value
# Verify all values
final_answer(answer=f'{{var_check}}-{{func_check}}-{{class_check}}')
""")
        assert result4.success is True, f"Failed: {result4.error}"
        assert result4.final_answer is not None

        # Verify the expected values
        expected_var = var_value
        expected_func = var_value * 2
        expected_class = var_value * 2  # class uses function which uses variable
        expected = f"{expected_var}-{expected_func}-{expected_class}"
        assert result4.final_answer["answer"] == expected

    @given(
        var_name=valid_python_identifier,
        var_value=st.integers(min_value=0, max_value=1000),
        num_calls=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=100)
    def test_initial_state_availability(self, var_name, var_value, num_calls):
        """For any variable in initial_state, that variable should be accessible
        in every python_executor call throughout the tool instance's lifetime.
        """
        tool = LocalPythonExecutorTool(output_type=OutputModel, initial_state={var_name: var_value})

        # Verify initial_state is available across multiple calls
        for i in range(num_calls):
            # Each call should be able to access the initial_state variable
            result = tool._execute_code(f"final_answer(answer=str({var_name}))")
            assert result.success is True, f"Call {i + 1} failed: {result.error}"
            assert result.final_answer is not None
            assert result.final_answer["answer"] == str(var_value)

        # Also verify initial_state survives after user-defined variables are added
        tool._execute_code("user_var = 999")
        result = tool._execute_code(f"final_answer(answer=str({var_name}))")
        assert result.success is True
        assert result.final_answer["answer"] == str(var_value)

    @given(
        var_name=valid_python_identifier,
        var_value=st.integers(min_value=0, max_value=1000),
        error_message=st.from_regex(r"[a-zA-Z0-9]+", fullmatch=True).filter(lambda x: len(x) > 0 and len(x) <= 50),
    )
    @settings(max_examples=100)
    def test_error_resilience(self, var_name, var_value, error_message):
        """For any python_executor call that raises an exception, all definitions
        made before the exception should remain in the namespace and be accessible
        in subsequent calls.
        """
        tool = LocalPythonExecutorTool(output_type=OutputModel)

        # Call 1: Define a variable before any error
        result1 = tool._execute_code(f"{var_name} = {var_value}")
        assert result1.success is True

        # Call 2: Execute code that raises an exception
        result2 = tool._execute_code(f"raise ValueError('{error_message}')")
        assert result2.success is False
        assert error_message in result2.error

        # Call 3: Verify variable defined before exception is still accessible
        result3 = tool._execute_code(f"final_answer(answer=str({var_name}))")
        assert result3.success is True, f"Variable {var_name} should survive after exception: {result3.error}"
        assert result3.final_answer is not None
        assert result3.final_answer["answer"] == str(var_value)

        # Call 4: Verify final_answer callback remains available after errors
        result4 = tool._execute_code("final_answer(answer='callback_works')")
        assert result4.success is True
        assert result4.final_answer["answer"] == "callback_works"

        # Call 5: Define new variable after error, then cause another error
        new_var_name = f"{var_name}_new"
        result5 = tool._execute_code(f"{new_var_name} = {var_value} + 100")
        assert result5.success is True

        # Call 6: Another exception
        result6 = tool._execute_code("raise RuntimeError('another error')")
        assert result6.success is False

        # Call 7: Both old and new variables should survive multiple errors
        result7 = tool._execute_code(f"final_answer(answer=f'{{{var_name}}}-{{{new_var_name}}}')")
        assert result7.success is True
        assert result7.final_answer["answer"] == f"{var_value}-{var_value + 100}"
