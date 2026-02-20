"""Core types for AI Functions.

This module defines the core types used throughout the interface.
"""

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping, TypedDict, Union

from pydantic import BaseModel, Field
from strands.agent import ConversationManager
from strands.agent.state import AgentState
from strands.hooks import HookProvider
from strands.models import Model
from strands.session import SessionManager
from strands.tools import ToolProvider
from strands.tools.executors._executor import ToolExecutor
from strands.types.content import Messages
from strands.types.tools import JSONSchema
from strands.types.traces import AttributeValue
from typing_extensions import Unpack


class CodeExecutionMode(StrEnum):
    """Execution mode for Python code execution.

    Attributes:
        LOCAL: Execute code in the current process (default)
        DISABLED: No code execution - relies solely on structured output from the model
    """

    LOCAL = "local"
    DISABLED = "disabled"


class PostConditionResult(BaseModel):
    """Result from running a post-condition validator.

    Attributes:
        passed: Whether the condition passed
        message: Validation message (error message required when passed is False)
    """

    passed: bool = Field(description="Whether the condition passed")
    message: str | None = Field(default=None, description="Validation message")

    def model_post_init(self, __context: object) -> None:
        """Validate that message is present when passed is False."""
        if not self.passed and self.message is None:
            raise ValueError("message is required when passed is False")


# Type alias for post-condition callables
PostCondition = Callable[..., PostConditionResult | None]
"""A post-condition callable that validates AI function results.

Post-conditions receive the result as the first argument and optionally
bound_args as keyword arguments. They return either:
- ``PostConditionResult`` with passed status and message (message required when failed)
- ``None`` to indicate the condition passed (shorthand for ``PostConditionResult(passed=True)``)
- Raising an exception is treated as a failed condition with the exception message
"""


# Utility classes for typing
class AgentKwargs(TypedDict, total=False):
    """Typing information for the arguments that are specific to the strands.Agent constructor."""

    messages: Messages | None
    callback_handler: Callable[..., Any] | None
    conversation_manager: ConversationManager | None
    record_direct_tool_call: bool
    load_tools_from_directory: bool
    trace_attributes: Mapping[str, AttributeValue] | None
    agent_id: str | None
    state: AgentState | dict | None
    hooks: list[HookProvider] | None
    session_manager: SessionManager | None
    tool_executor: ToolExecutor | None


class AIFunctionKwargs(TypedDict, total=False):
    """Typing information for the fields of AIFunctionConfig (should always match AIFunctionConfig exactly)."""

    model: Model | str | None
    system_prompt: str | None
    tools: list[Union[str, dict[str, str], ToolProvider, Any]] | None
    post_conditions: list[PostCondition]
    max_attempts: int
    code_execution_mode: CodeExecutionMode | str
    code_executor_kwargs: dict[str, Any]
    code_executor_additional_imports: list[str]
    name: str | None
    description: str | None
    inputSchema: JSONSchema | None
    agent_kwargs: AgentKwargs


class AIFunctionMergedKwargs(AgentKwargs, AIFunctionKwargs):
    """Utility class to type the kwargs in the @ai_function decorator."""


# Actual config class for AIFunction
@dataclass
class AIFunctionConfig:
    """Configuration for an AI Function.

    This dataclass holds all parameters that can be passed to the @ai_function
    decorator to configure the behavior of the AI-enhanced function.

    Attributes:
        model: Strands model provider instance (e.g., BedrockModel, OpenAIModel) or model ID string
        system_prompt: System prompt for the AI agent
        tools: Additional tools to provide to the agent (can be tool functions,
               MCP server configs, ToolProvider instances, or any tool-like object)
        post_conditions: Validation functions to run after execution
        max_attempts: Maximum retry attempts for retryable errors
        code_execution_mode: How to execute Python code (local or remote)
        code_executor_kwargs: Keyword arguments passed to LocalPythonExecutorTool (e.g. timeout_seconds)
        code_executor_additional_imports: Additional authorized imports for the code executor
        agent_kwargs: Additional keyword arguments passed directly to the Strands Agent constructor
    """

    model: Model | str | None = None
    system_prompt: str | None = None
    tools: list[Union[str, dict[str, str], ToolProvider, Any]] | None = None
    post_conditions: list[PostCondition] = field(default_factory=list)
    max_attempts: int = 10
    code_execution_mode: CodeExecutionMode | str = CodeExecutionMode.DISABLED
    code_executor_kwargs: dict[str, Any] = field(default_factory=dict)
    code_executor_additional_imports: list[str] = field(default_factory=list)
    agent_kwargs: AgentKwargs = field(default_factory=AgentKwargs)
    name: str | None = None
    description: str | None = None
    inputSchema: JSONSchema | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        # validate code_execution_mode and convert to proper enum value
        if isinstance(self.code_execution_mode, str):
            self.code_execution_mode = CodeExecutionMode(self.code_execution_mode)


def split_config_and_agent_kwargs(**kwargs: Unpack[AIFunctionMergedKwargs]) -> tuple[AIFunctionKwargs, AgentKwargs]:
    """Split kwargs into AIFunctionConfig fields and agent kwargs."""
    _AI_FUNCTION_CONFIG_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(AIFunctionConfig))
    config_kwargs = {k: v for k, v in kwargs.items() if k in _AI_FUNCTION_CONFIG_FIELD_NAMES}
    agent_kwargs = {k: v for k, v in kwargs.items() if k not in _AI_FUNCTION_CONFIG_FIELD_NAMES}
    return AIFunctionKwargs(**config_kwargs), AgentKwargs(**agent_kwargs)  # type: ignore[typeddict-item]
