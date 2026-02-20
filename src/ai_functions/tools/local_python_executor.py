"""Local Python executor tool for AI Functions.

This module provides a tool for executing Python code locally using
smolagents' LocalPythonExecutor for safer AST-based execution.
"""

import io
import os
import textwrap
from typing import Any

from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from smolagents.local_python_executor import LocalPythonExecutor
from strands import ToolContext, tool

from ..utils._type import generate_signature_from_model


class PythonExecuteResult(BaseModel):
    """Result from local Python code execution.

    Attributes:
        success: Whether the execution completed without errors
        final_answer: Dict of kwargs passed to final_answer() callback, or None
        stdout: Captured standard output
        error: Error message if execution failed, or None
    """

    success: bool
    final_answer: dict[str, Any] | None = None
    stdout: str = ""
    error: str | None = None

    def to_markdown(self) -> str:
        """Convert execution result to markdown format for display."""
        parts = []
        if self.error:
            parts.append("## ERROR")
            parts.append(self.error)
            parts.append(
                "Note: To fix the error you do not have to rewrite the full code. "
                "Code before the error has been executed, and variables assigned before the error "
                "are already in the state."
            )
        if self.stdout:
            parts.append("## STDOUT")
            parts.append(self.stdout)
        if self.final_answer:
            parts.append(f"## Final answer\n\n{self.final_answer.get('answer', self.final_answer)}")
        return "\n\n".join(parts)


def _display_code(content: str, title: str | None = None, line_numbers: bool = True) -> None:
    """Display python code with syntax highlighting if STRANDS_TOOL_CONSOLE_MODE is enabled.

    If STRANDS_TOOL_CONSOLE_MODE environment variable is set to "enabled",
    displays python code to stdout with syntax highlighting.
    """
    console = Console(file=io.StringIO()) if os.getenv("STRANDS_TOOL_CONSOLE_MODE") != "enabled" else Console()
    syntax = Syntax(content, lexer="python", theme="monokai", line_numbers=line_numbers)
    panel = Panel(syntax, title=title, border_style="blue", box=box.DOUBLE, expand=False, padding=(0, 0))
    console.print(panel)


class LocalPythonExecutorTool:
    """Strands tool that executes Python code using smolagents' LocalPythonExecutor.

    This tool provides a safer interface for executing Python code by using smolagents' AST-based interpreter.
    https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution#our-local-python-executor
    """

    def __init__(
        self,
        output_type: type[BaseModel],
        initial_state: dict[str, Any] | None = None,
        additional_authorized_imports: list[str] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the local Python executor tool.

        Args:
            output_type: The expected output type for final_answer
            initial_state: Variables to inject into the execution namespace
            additional_authorized_imports: List of modules allowed to import.
            executor_kwargs: Additional keyword arguments passed to LocalPythonExecutor.
        """
        assert issubclass(output_type, BaseModel)
        self._output_type = output_type
        self._final_answer: dict[str, Any] | None = None

        # Create the smolagents executor with final_answer as a static tool.
        # final_answer is the callback used to capture the final answer from execution env.
        self._code_executor = LocalPythonExecutor(
            additional_authorized_imports=additional_authorized_imports if additional_authorized_imports else [],
            additional_functions={"final_answer": self._set_execution_result},
            **(executor_kwargs or {}),
        )

        # Initialize tools (required for additional_functions to be callable)
        self._code_executor.send_tools({})

        # Inject initial state
        if initial_state:
            self._code_executor.send_variables(initial_state)

        # Update tool description with the correct final_answer signature
        self.python_executor.tool_spec["description"] = self._build_tool_description()

    def _build_tool_description(self) -> str:
        """Build the tool description with the correct final_answer signature."""
        signature = self._get_final_answer_signature()
        return textwrap.dedent(f"""\
            Execute Python code in a persistent environment.

            WHEN TO USE:
            - Tasks requiring computation, data processing, or Python object creation

            OUTPUT:
            - stdout/stderr visible only to you (assistant), not the end user

            PERSISTENT STATE:
            - Variables, imports, functions, and classes persist between calls
            - Build up state incrementally across multiple invocations

            RETURNING RESULTS:
            Return a result from code execution by calling the method: {signature}
            The function final_answer is already imported. All arguments must be keyword arguments.
            If final_answer is not called, no result is returned.
            """)

    def _get_final_answer_signature(self) -> str:
        """Generate the signature for final_answer based on output type."""
        return generate_signature_from_model(self._output_type)

    def _set_execution_result(self, *args: Any, **kwargs: Any) -> None:
        """Called by executed code to set the final result."""
        is_simple_wrapper = len(self._output_type.model_fields) == 1 and "answer" in self._output_type.model_fields
        # Allow calling final_answer(output) without kwargs if the output model is a simple wrapper
        if len(args) == 1 and len(kwargs) == 0 and is_simple_wrapper:
            kwargs["answer"] = args[0]
            args = ()
        # Raise an exception if the agent passed positional arguments
        if args:
            raise ValueError(
                f"final_answer only accepts keyword arguments "
                f"with the following signature: {self._get_final_answer_signature()}"
            )
        self._final_answer = kwargs

    def _execute_code(self, code: str) -> PythonExecuteResult:
        """Execute Python code using smolagents' LocalPythonExecutor.

        Args:
            code: Python code to execute

        Returns:
            PythonExecuteResult with keys:
                - success: bool indicating if execution completed without error
                - final_answer: Dict from final_answer callback if called, else None
                - stdout: str of captured print output
                - error: str error message if execution failed, else None
        """
        try:
            result = self._code_executor(code)

            return PythonExecuteResult(
                success=True,
                final_answer=self._final_answer,
                stdout=result.logs,
            )
        except Exception as e:
            return PythonExecuteResult(
                success=False,
                final_answer=None,
                error=str(e),
            )

    @tool(context=True)
    def python_executor(self, code: str, tool_context: ToolContext) -> str:
        """Execute Python code in the local process.

        Args:
            code: Python code to execute
            tool_context: Strands tool context for state management

        Returns:
            String representation of execution result

        Raises:
            ValueError: If final_answer validation fails
            RuntimeError: If code execution fails
        """
        _display_code(code, title="Python Executor Tool")

        # Reset result before execution
        self._final_answer = None
        result = self._execute_code(code)
        result_md = result.to_markdown()
        _display_code(result_md, title="Python Executor Result")

        # If final_answer was called, store the result in invocation_state and stop the agent
        if result.final_answer:
            try:
                tool_context.invocation_state["python_executor_result"] = self._output_type(**result.final_answer)
                tool_context.invocation_state["request_state"]["stop_event_loop"] = True
            except Exception as e:
                raise ValueError(f"Failed to construct output from final_answer: {e}") from e

        if result.success:
            return str(result_md)
        raise RuntimeError(f"Error executing code:\n{result_md}")
