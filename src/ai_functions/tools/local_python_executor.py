"""Local Python executor tool for AI Functions.

Executes agent-generated Python with smolagents' ``LocalPythonExecutor`` — an
AST-based interpreter that is far safer than raw ``exec`` (no arbitrary imports,
no attribute escapes; only an allowlist of modules). Used to run ``Procedural``
memory parameters: their code is injected into the execution namespace, and the
agent calls helpers and returns a typed result via the ``final_answer`` callback.

``smolagents`` is an optional dependency: importing this module is cheap, but
constructing :class:`LocalPythonExecutorTool` raises ``ImportError`` if the
package is not installed (``pip install strands-ai-functions[procedural]``).

The model returns its answer by calling ``final_answer(...)`` inside executed
code. The tool captures that, writes it into
``tool_context.invocation_state["request_state"]["python_executor_result"]``,
and requests the event loop to stop. The runtime then reads it from
``AgentResult.state`` (see ``AIThread._extract_result``). This ``request_state``
channel is the v2-Strands-correct location (verified to surface as
``AgentResult.state``); the predecessor wrote to top-level ``invocation_state``,
which does not surface in this Strands version.
"""

from __future__ import annotations

import inspect
import io
import os
import textwrap
from typing import Any

from pydantic import BaseModel
from strands import ToolContext, tool

# Modules the sandboxed interpreter may import. Pure-computation stdlib only —
# no os, sys, subprocess, socket, etc.
SAFE_BUILTINS = [
    "math", "cmath", "decimal", "fractions", "random", "statistics", "numbers",
    "collections", "heapq", "bisect", "array", "queue", "copy", "pprint", "enum",
    "dataclasses", "graphlib", "string", "re", "textwrap", "unicodedata",
    "difflib", "stringprep", "datetime", "calendar", "zoneinfo", "itertools",
    "functools", "operator", "typing", "types", "abc", "contextlib", "json",
    "base64", "binascii", "html", "hashlib",
]  # fmt: skip


def _generate_signature_from_model(model: type[BaseModel], func_name: str = "final_answer") -> str:
    """Build a ``final_answer(...)`` signature string from a pydantic model's fields."""
    params: list[inspect.Parameter] = []
    for field_name, field_info in model.model_fields.items():
        if field_info.is_required():
            params.append(
                inspect.Parameter(field_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=field_info.annotation)
            )
        else:
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=field_info.default,
                    annotation=field_info.annotation,
                )
            )
    return f"{func_name}{inspect.Signature(params)}"


class PythonExecuteResult(BaseModel):
    """Result of one ``python_executor`` call."""

    success: bool
    final_answer: dict[str, Any] | None = None
    stdout: str = ""
    error: str | None = None

    def to_markdown(self) -> str:
        """Render the result as markdown for the agent to read."""
        parts: list[str] = []
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
    """Pretty-print code/results when ``STRANDS_TOOL_CONSOLE_MODE=enabled``."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console(file=io.StringIO()) if os.getenv("STRANDS_TOOL_CONSOLE_MODE") != "enabled" else Console()
    syntax = Syntax(content, lexer="python", theme="monokai", line_numbers=line_numbers)
    console.print(Panel(syntax, title=title, border_style="blue", box=box.DOUBLE, expand=False, padding=(0, 0)))


class LocalPythonExecutorTool:
    """Strands tool wrapping smolagents' AST-based ``LocalPythonExecutor``.

    Args:
        output_type: Pydantic model the ``final_answer`` callback constructs.
        initial_state: Variables injected into the execution namespace (the
            cycle's bound arguments, including any ``Procedural`` code strings).
        additional_authorized_imports: Extra modules allowed beyond ``SAFE_BUILTINS``.
        executor_kwargs: Extra kwargs forwarded to ``LocalPythonExecutor``.

    Raises:
        ImportError: ``smolagents`` is not installed.
    """

    def __init__(
        self,
        output_type: type[BaseModel],
        initial_state: dict[str, Any] | None = None,
        initial_code: list[str] | None = None,
        additional_authorized_imports: list[str] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from smolagents.local_python_executor import LocalPythonExecutor
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "LocalPythonExecutorTool requires the 'smolagents' package. "
                "Install it with: pip install strands-ai-functions[procedural]"
            ) from exc

        assert issubclass(output_type, BaseModel)
        self._output_type = output_type
        self._final_answer: dict[str, Any] | None = None

        self._code_executor = LocalPythonExecutor(
            additional_authorized_imports=SAFE_BUILTINS + list(additional_authorized_imports or []),
            additional_functions={"final_answer": self._set_execution_result},
            **(executor_kwargs or {}),
        )
        self._code_executor.send_tools({})
        if initial_state:
            self._code_executor.send_variables(initial_state)
        # Execute procedural code blocks so their functions/classes are DEFINED
        # in the persistent namespace (the sandbox forbids exec(), so injecting
        # the source as a string variable would not make the helpers callable).
        # Surface setup failures: malformed/erroring recalled helper code means
        # the helpers silently would not exist, leaving the agent no diagnostic.
        for code in initial_code or []:
            if code and code.strip():
                setup = self._execute_code(code)
                if not setup.success:
                    raise ValueError(f"Failed to load procedural code into the executor namespace:\n{setup.error}")

        self.python_executor.tool_spec["description"] = self._build_tool_description()

    def _build_tool_description(self) -> str:
        signature = _generate_signature_from_model(self._output_type)
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

    def _set_execution_result(self, *args: Any, **kwargs: Any) -> None:
        """``final_answer`` callback invoked from inside executed code."""
        is_simple_wrapper = len(self._output_type.model_fields) == 1 and "answer" in self._output_type.model_fields
        if len(args) == 1 and len(kwargs) == 0 and is_simple_wrapper:
            kwargs["answer"] = args[0]
            args = ()
        if args:
            raise ValueError(
                f"final_answer only accepts keyword arguments with the signature: "
                f"{_generate_signature_from_model(self._output_type)}"
            )
        self._final_answer = kwargs

    def _execute_code(self, code: str) -> PythonExecuteResult:
        """Run ``code`` in the sandbox, capturing stdout / final_answer / errors."""
        try:
            result = self._code_executor(code)
            return PythonExecuteResult(success=True, final_answer=self._final_answer, stdout=result.logs)
        except Exception as e:  # noqa: BLE001 - report any execution error to the agent
            return PythonExecuteResult(success=False, final_answer=None, error=str(e))

    @tool(context=True)
    def python_executor(self, code: str, tool_context: ToolContext) -> str:
        """Execute Python code in the local sandboxed environment.

        Args:
            code: Python code to execute.
            tool_context: Strands-provided context for invocation state.

        Returns:
            A markdown rendering of stdout / final answer.

        Raises:
            ValueError: ``final_answer`` was called with an output the model cannot construct.
            RuntimeError: The code raised during execution.
        """
        _display_code(code, title="Python Executor Tool")
        self._final_answer = None
        result = self._execute_code(code)
        result_md = result.to_markdown()
        _display_code(result_md, title="Python Executor Result")

        # Distinguish "final_answer not called" (None) from "called but with an
        # empty/invalid payload" ({}). The latter must still attempt construction
        # so a missing required field surfaces as an error to the model, rather
        # than being silently dropped as if no answer were produced.
        if result.final_answer is not None:
            try:
                request_state = tool_context.invocation_state["request_state"]
                request_state["python_executor_result"] = self._output_type(**result.final_answer)
                request_state["stop_event_loop"] = True
            except Exception as e:  # noqa: BLE001 - surface construction failure to the agent
                raise ValueError(f"Failed to construct output from final_answer: {e}") from e

        if result.success:
            return str(result_md)
        raise RuntimeError(f"Error executing code:\n{result_md}")
