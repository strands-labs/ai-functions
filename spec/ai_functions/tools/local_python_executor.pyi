"""Local Python executor tool for AI Functions.

Executes agent-generated Python with smolagents' ``LocalPythonExecutor`` — an
AST-based interpreter that only permits an allowlist of imports and forbids
arbitrary escapes (no ``exec``, no attribute breakouts). Used to run
``Procedural`` memory: the code is defined in the execution namespace, the agent
calls the helpers, and returns a typed result via the ``final_answer`` callback.

``smolagents`` is an optional dependency: importing this module is cheap, but
constructing :class:`LocalPythonExecutorTool` raises ``ImportError`` if the
package is not installed (``pip install strands-ai-functions[procedural]``).
"""

from typing import Any

from pydantic import BaseModel
from strands.types.tools import AgentTool

SAFE_BUILTINS: list[str]
"""Stdlib modules the sandboxed interpreter may import (pure computation only)."""


class PythonExecuteResult(BaseModel):
    """Result of one ``python_executor`` call."""

    success: bool
    final_answer: dict[str, Any] | None
    stdout: str
    error: str | None

    def to_markdown(self) -> str:
        """Render the result (error / stdout / final answer) as markdown."""
        ...


class LocalPythonExecutorTool:
    """Strands tool wrapping smolagents' AST-based ``LocalPythonExecutor``.

    The agent returns its answer by calling ``final_answer(...)`` inside executed
    code; the tool writes that into
    ``invocation_state["request_state"]["python_executor_result"]`` and requests
    the event loop to stop, so the runtime can read it from ``AgentResult.state``.
    """

    python_executor: AgentTool
    """The Strands tool the agent calls to execute code (``@tool``-decorated)."""

    def __init__(
        self,
        output_type: type[BaseModel],
        initial_state: dict[str, Any] | None = None,
        initial_code: list[str] | None = None,
        additional_authorized_imports: list[str] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Build a sandboxed Python executor tool.

        Args:
            output_type: Pydantic model the ``final_answer`` callback constructs;
                its fields define the ``final_answer(...)`` signature.
            initial_state: Variables injected into the execution namespace as-is.
            initial_code: Python source blocks executed at setup so their
                functions/classes are defined (and callable) in the persistent
                namespace — used for ``Procedural`` parameter code, since the
                sandbox forbids ``exec`` of an injected source string.
            additional_authorized_imports: Modules allowed beyond ``SAFE_BUILTINS``.
            executor_kwargs: Extra kwargs forwarded to ``LocalPythonExecutor``.

        Raises:
            ImportError: ``smolagents`` is not installed.
        """
        ...
