"""AIFunction wrapper class for AI Functions.

This module provides the core AIFunction class that wraps Python functions
with AI capabilities, handling:
- Parameter binding and validation
- Agent creation and execution
- Post-condition validation
- Retry logic
- Python code execution via configured executor
- Async execution support
"""

import functools
import inspect
import logging
import types
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, ParamSpec, Self, TypeVar, Unpack, get_type_hints

import pydantic
from pydantic import BaseModel, ConfigDict, Field, create_model
from strands import Agent, ToolContext
from strands.agent import AgentResult
from strands.tools.decorator import DecoratedFunctionTool, FunctionToolMetadata
from strands.tools.tool_provider import ToolProvider
from strands.types.content import Messages
from strands.types.tools import AgentTool
from tstr import Template, generate_template

from .tools.local_python_executor import SAFE_BUILTINS, LocalPythonExecutorTool
from .trace_context import get_context, trace_scope
from .types.ai_function import (
    AIFunctionConfig,
    AIFunctionMergedKwargs,
    CodeExecutionMode,
    split_config_and_agent_kwargs,
)
from .types.errors import AIFunctionError, ValidationError
from .types.graph import ParameterView, Result
from .utils._async import run_async
from .utils._console import print_in_box
from .utils._formatting import extract_signatures, truncate, unique_name
from .utils._graph import collect_nodes, unwrap_nodes
from .utils._template import render_template_with_indent
from .utils._type import (
    add_parameter_to_signature,
    generate_signature_from_model,
    is_json_serializable_type,
    is_pydantic_model,
)
from .validation.post_conditions import (
    PostConditionRunner,
    get_failed_results,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Template for validation error feedback message
_VALIDATION_ERROR_TEMPLATE = """[VALIDATION ERROR]
Your previous response failed validation with the following errors:
{error_messages}

Please try again and ensure your output satisfies all requirements."""


class AIFunction(Generic[P, R], ToolProvider):
    """Wrapper class that executes an AI-enhanced function.

    This class handles the full lifecycle of an AI function call:
    1. Bind arguments to function parameters
    2. Create and execute the Strands Agent
    3. Run post-conditions to validate outputs
    4. Handle retries on failure

    Supports both sync and async execution:
    - Sync functions: Call directly with `ai_func(...)`
    - Async functions: Call with `await ai_func(...)`

    Attributes:
        func: The wrapped Python function
        config: Configuration for the AI function
        __name__: Name of the wrapped function
        __doc__: Docstring of the wrapped function
        is_async: Whether the wrapped function is async
    """

    # Declare attributes set by functools.update_wrapper for mypy
    __name__: str
    __doc__: str | None
    func: Callable[P, R] | Callable[P, Awaitable[R]]

    def __init__(self, func: Callable[P, R] | Callable[P, Awaitable[R]], config: AIFunctionConfig):
        """Initialize the AIFunction wrapper.

        Args:
            func: The Python function to wrap
            config: Configuration for the AI function
        """
        super().__init__()

        self.func = func
        self.config = config

        # Detect if the wrapped function is async
        self.is_async = inspect.iscoroutinefunction(func)

        # Preserve function metadata (must happen before accessing __name__)
        functools.update_wrapper(self, func)

        # Get return type for structured output
        self._return_type = self._get_return_type()
        self._is_pydantic_return = is_pydantic_model(self._return_type)
        self._is_json_serializable = is_json_serializable_type(self._return_type)
        # structured output is always possible if the output is json serializable
        self._is_structured_output_enabled = self._is_json_serializable
        # all types that are not already pydantic are always wrapped in a pydantic type
        self._is_return_wrapped = not self._is_pydantic_return

        # Create structured output type for agent
        if self._is_return_wrapped:
            # Wrap type in a pydantic model
            self._structured_output_type = create_model(
                "FinalAnswer",
                __config__=ConfigDict(arbitrary_types_allowed=True),
                answer=(self._return_type, Field(description="The final answer to return.")),
            )
        else:
            assert issubclass(self._return_type, BaseModel)
            # already pydantic, can use as is
            self._structured_output_type = self._return_type

        # Create condition runner for post-conditions
        self._post_runner = PostConditionRunner(function_name=self.__name__)

        # Validate DISABLED mode requirements
        if self.config.code_execution_mode == CodeExecutionMode.DISABLED and not self._is_structured_output_enabled:
            raise ValueError(
                f"ai_function '{self.__name__}' uses code_execution_mode=DISABLED but has a "
                f"non-Pydantic return type '{self._return_type}'. DISABLED mode requires a "
                "Pydantic model return type."
            )

        # Creates tool specs to use the ai_function as a tool
        self._tool: AgentTool | None = None
        try:
            self._tool = self.as_tool()
        except pydantic.errors.PydanticSchemaGenerationError:
            # Inputs are not pydantic-serializable so we cannot use this function as a tool
            # (the agent would not be able to call it)
            pass

    @property
    def __signature__(self) -> inspect.Signature:
        """Return the signature of the wrapped function."""
        return inspect.signature(self.func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R | Awaitable[R]:
        """DEPRECATED: runs the AIFunction as either sync or async based on the original function type.

        New code uses SyncAIFunction and AsyncAIFunction for proper typing
        """
        if self.is_async:
            return self.call_async(*args, **kwargs)
        else:
            return self.call(*args, **kwargs)

    def replace(self, **kwargs: Unpack[AIFunctionMergedKwargs]) -> Self:
        """Create a new AIFunction with modified configuration.

        Args that match AIFunctionConfig fields are used to update the config directly.
        Args that don't match are merged into agent_kwargs.
        """
        import dataclasses

        # Split kwargs into config fields vs agent kwargs
        config_kwargs, agent_kwargs = split_config_and_agent_kwargs(**kwargs)

        # Merge agent_kwargs with existing ones
        if agent_kwargs:
            merged_agent_kwargs = self.config.agent_kwargs | agent_kwargs
            config_kwargs["agent_kwargs"] = merged_agent_kwargs

        config = dataclasses.replace(self.config, **config_kwargs)
        return type(self)(self.func, config)

    def as_tool(self) -> AgentTool:
        """Expose the ai_function as a strands tool so it can be used by other agents."""

        @functools.wraps(self.func)
        def tool_wrapper(*args: P.args, tool_context: ToolContext, **kwargs: P.kwargs) -> R:  # type: ignore[valid-type]
            result: Result[R] = self.trace(*args, **kwargs)
            # Set tool-id to track the association between tool calls in the agent trace
            # and tool_result nodes
            tool_id = tool_context.tool_use.get("toolUseId")
            result.tool_id = tool_id
            # Register the tool call result in the graph context of the parent
            get_context().tool_results.append(result)
            return result.value

        # functools.wraps makes it so that the function signature is the same as the wrapped (self.func)
        # so we need to manually add `tool_context` to ensure strands will provide it
        add_parameter_to_signature(tool_wrapper, "tool_context", annotation=ToolContext)
        tool_meta = FunctionToolMetadata(tool_wrapper, context_param="tool_context")

        tool_spec = tool_meta.extract_metadata()
        if self.config.name is not None:
            tool_spec["name"] = self.config.name
        if self.config.description is not None:
            tool_spec["description"] = self.config.description
        if self.config.inputSchema is not None:
            tool_spec["inputSchema"] = self.config.inputSchema

        tool_name = tool_spec.get("name", self.func.__name__)

        if not isinstance(tool_name, str):
            raise ValueError(f"Tool name must be a string, got {type(tool_name)}")

        return DecoratedFunctionTool(tool_name, tool_spec, tool_wrapper, tool_meta)

    async def load_tools(self, **kwargs: Any) -> Sequence[AgentTool]:
        """Provides the ai_function wrapper tool to agents requesting it."""
        if self._tool is None:
            raise ValueError(
                f"ai_function `{self.__name__}` cannot be used as tool. Perhaps its input types are not serializable?`"
            )
        return [self._tool]

    def add_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Add a consumer to this tool provider."""
        pass

    def remove_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Remove a consumer from this tool provider."""
        pass

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the AI function synchronously.

        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the AI function execution

        Raises:
            AIFunctionError: If execution fails after all retries
            ValidationError: If post conditions fail
        """
        return self.trace(*args, **kwargs).value

    async def call_async(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the AI function asynchronously.

        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the AI function execution

        Raises:
            AIFunctionError: If execution fails after all retries
            ValidationError: If post conditions fail
        """
        return (await self.trace_async(*args, **kwargs)).value

    def trace(self, *args: P.args, **kwargs: P.kwargs) -> Result[R]:
        """Execute the AI function synchronously and return the full Result node."""
        bound_args = self._get_bound_arguments(*args, **kwargs)
        # For sync functions, use run_async to execute the async core
        return run_async(lambda: self._execute_async(bound_args))

    async def trace_async(self, *args: P.args, **kwargs: P.kwargs) -> Result[R]:
        """Execute the AI function asynchronously and return the full Result node."""
        bound_args = self._get_bound_arguments(*args, **kwargs)
        # Run async and await
        return await self._execute_async(bound_args)

    def __await__(self) -> None:
        """Make the AI function awaitable when called without arguments.

        Note: This is called when the wrapper object itself is awaited without
        being called first. Normally users should call the function with args:
        `await ai_func(arg1, arg2)` which returns a coroutine from __call__.
        """
        raise TypeError(
            f"Cannot await {self.__name__} without calling it. "
            f"Use 'await {self.__name__}(args...)' to execute the function."
        )

    def __get__(self, obj: Any, objtype: Any = None) -> Self:
        """Support instance methods by implementing the descriptor protocol.

        When ``@ai_function`` is used as a decorator on a method defined in a class body,
        it captures the raw function before it becomes a bound method. For example::

            class MyClass:
                @ai_function
                def my_method(self):
                    ...

            instance = MyClass()
            instance.my_method()  # triggers __get__

        When ``instance.my_method`` is accessed, Python invokes this method via the
        descriptor protocol. We use it to wrap the bound method in a new
        ``AIFunction``, preserving the original configuration while ensuring
        ``self`` is correctly passed to the underlying function.
        """
        if obj is None:
            # Called on the class, return self
            return self
        # Return a new AIFunction that wraps the bound method instead
        bound_method = types.MethodType(self.func, obj)
        return type(self)(bound_method, self.config)

    def _get_bound_arguments(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Bind args/kwargs to function parameters and return as a dict.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Dictionary mapping parameter names to values
        """
        sig = inspect.signature(self.func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    async def _execute_async(self, bound_args: dict[str, Any]) -> Result[R]:
        """Async execution logic with retry handling for validation errors.

        This is the unified implementation that handles both sync and async execution.
        The retry loop, error handling, and validation are all consolidated here.

        Args:
            bound_args: Bound function arguments

        Returns:
            The result of the AI function

        Raises:
            AIFunctionError: If execution fails after all retries
            ValidationError: If post-conditions fail after all retries
        """
        result: Result[R] | None = None  # Conversation history

        # Open a new trace scope for this node
        with trace_scope():
            for attempt in range(self.config.max_attempts + 1):
                try:
                    # Create agent and execute with conversation history
                    result = await self._run_agent(bound_args, result)

                    # Run post-conditions after successful execution
                    await self._validate_result(result.value, bound_args)

                    return result

                except ValidationError as e:
                    if attempt < self.config.max_attempts:
                        # Build retry message with all error messages
                        error_messages = e.message if e.message else "Unknown validation error"
                        if result is not None:
                            result.agent.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"text": _VALIDATION_ERROR_TEMPLATE.format(error_messages=error_messages)}
                                    ],
                                }
                            )
                        logger.warning(
                            f"Validation failed for {self.__name__}, "
                            f"retrying (attempt {attempt + 1}/{self.config.max_attempts}). "
                            f"Error messages: {error_messages}"
                        )
                    else:
                        raise

                except AIFunctionError:
                    # Re-raise AIFunctionError subclasses as-is
                    raise

                except Exception as e:
                    # Wrap unexpected errors in AIFunctionError
                    raise AIFunctionError(
                        message=str(e),
                        function_name=self.__name__,
                    ) from e
            # This should not be reachable, but we need it for mypy
            raise ValidationError(
                function_name=self.__name__, validation_errors={"max_retries": "Max number of retries reached"}
            )

    async def _run_agent(
        self,
        bound_args: dict[str, Any],
        result: Result[R] | None = None,
    ) -> Result[R]:
        """Execute agent with conversation history and return a Result node.

        Handles both initial invocation and retry scenarios where the previous
        Result's agent (and its message history) is reused.

        Args:
            bound_args: Bound function arguments.
            result: Previous Result from a failed attempt, or None on first call.

        Returns:
            A new Result node wrapping the agent's output.
        """
        # Create agent with current conversation history
        agent = result.agent if result else self._create_agent(bound_args, [])
        messages = agent.messages

        # Build prompt - needed for invocation_state even on retries
        unwrapped_bound_args = unwrap_nodes(bound_args)
        prompt_for_state = await self._build_prompt(unwrapped_bound_args)
        prompt_for_state = self._add_prompt(prompt_for_state, bound_args)

        # Use prompt for first invocation, None for retries (agent doesn't need it again)
        if not messages:
            prompt = prompt_for_state
            logger.debug(f"Executing AI function '{self.__name__}' with prompt: '{truncate(prompt, max_length=200)}'")
        else:
            prompt = None
            logger.debug(f"Retrying AI function '{self.__name__}' with {len(messages)} messages in history")

        if prompt is not None:
            model_id = self._get_model_identifier()
            title = f"{self.__name__}: {model_id}" if model_id else self.__name__
            print_in_box(prompt, title=title)

        # Execute agent - always pass prompt to conversation manager for summarization
        invocation_state: dict[str, Any] = {"prompt": prompt_for_state, "tool_results": {}}

        # Invoke agent asynchronously
        agent_response = await agent.invoke_async(prompt, invocation_state=invocation_state)

        ctx = get_context()
        # Extract result
        value = self._extract_result(agent_response, invocation_state)
        new_result = Result(
            value=value,
            name=unique_name(self.__name__),
            func=self,
            agent=agent,
            tool_results=ctx.tool_results,
            inputs=ctx.inputs + collect_nodes(bound_args),
        )

        return new_result

    async def _build_prompt(self, bound_args: dict[str, Any]) -> str:
        """Build prompt from function return value or docstring.

        Handles both sync and async wrapped functions. Tries calling the function
        first to get a prompt, falls back to docstring if the function returns
        None. Exceptions raised by the wrapped function are propagated.

        Args:
            bound_args: Bound function arguments

        Returns:
            The rendered prompt string

        Raises:
            ValueError: If no prompt source is available or prompt is empty
            TypeError: If function returns a non-str/Template value
            Exception: Any exception raised by the wrapped function
        """
        # Try function return value first
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(**bound_args)  # type: ignore[call-arg]
        else:
            result = self.func(**bound_args)  # type: ignore[call-arg]

        if result is None:
            # Fall back to docstring with argument substitution
            if not self.func.__doc__:
                raise ValueError(f"ai_function '{self.__name__}' has no docstring to use as prompt")
            # Build template context: start with bound args, then add the
            # bound instance so ``{self.attr}`` works in method docstrings.
            template_context = dict(bound_args)
            if isinstance(self.func, types.MethodType):
                template_context["self"] = self.func.__self__
            result = generate_template(
                self.func.__doc__, globals=self.func.__globals__, context=template_context, use_eval=True
            )

        # Render the result
        if isinstance(result, str):
            prompt = result
        elif isinstance(result, Template):
            prompt = render_template_with_indent(result)
        else:
            raise TypeError(
                f"ai_function '{self.__name__}' must return str, Template, or None, got {type(result).__name__}"
            )

        # Validate prompt is not empty
        if not prompt.strip():
            raise ValueError(
                f"ai_function '{self.__name__}' produced an empty prompt. "
                "The function must return a non-empty string or have a non-empty docstring."
            )

        return prompt

    async def _validate_result(
        self,
        result: Any,
        bound_args: dict[str, Any],
    ) -> None:
        """Run post-condition validation.

        Calls validate() and receives list of results. Checks if any condition
        failed and raises ValidationError with all errors if so.

        Args:
            result: The result to validate
            bound_args: Original input arguments

        Raises:
            ValidationError: If any post-condition fails (with all errors)
        """
        if not self.config.post_conditions:
            return

        validation_results = await self._post_runner.validate(self.config.post_conditions, result, bound_args)

        # Check if any condition failed
        if not all(r.passed for r in validation_results):
            failed = get_failed_results(validation_results, self.config.post_conditions)
            raise ValidationError(
                function_name=self.__name__,
                validation_errors={name: r.message or "Unknown error" for name, r in failed},
            )

    def _get_model_identifier(self) -> str | None:
        """Return a human-readable identifier for the configured model."""
        model = self.config.model
        if model is None:
            return None
        if isinstance(model, str):
            return model
        # Model instance — try config dict, fall back to class name
        config = model.get_config()
        return config.get("model_id") or config.get("endpoint_name") or type(model).__name__

    def _get_return_type(self) -> type[Any]:
        """Get the return type from the function's type hints.

        Returns:
            The return type annotation

        Raises:
            ValueError: If no return type is specified
        """
        hints = get_type_hints(self.func)
        if "return" not in hints:
            raise ValueError(f"ai_function '{self.__name__}' must specify a return type")
        return_type: type[Any] = hints["return"]
        return return_type

    def _create_agent(
        self,
        bound_args: dict[str, Any],
        messages: Messages | None = None,
    ) -> Agent:
        """Create a Strands Agent with configured tools, model, and conversation history.

        Args:
            bound_args: Bound function arguments
            messages: Existing conversation history (for retry scenarios)

        Returns:
            Configured Strands Agent with conversation history
        """
        # Build tools list from user-provided tools
        tools = self.config.tools or []

        # Collect procedural parameters
        procedural_inputs = {
            k: v for k, v in bound_args.items() if isinstance(v, ParameterView) and v.procedural and v.value
        }

        # Raise if procedural parameters are provided but code execution is disabled
        if procedural_inputs and self.config.code_execution_mode == CodeExecutionMode.DISABLED:
            names = ", ".join(procedural_inputs.keys())
            raise ValueError(
                f"ai_function '{self.__name__}' received procedural parameter(s) [{names}] "
                f"but code_execution_mode is DISABLED."
            )

        # Add python_executor for LOCAL mode
        if self.config.code_execution_mode == CodeExecutionMode.LOCAL:
            # Exclude procedural parameters from initial_state — they are
            # executed as code, not injected as variables.
            initial_state = {k: unwrap_nodes(v) for k, v in bound_args.items() if k not in procedural_inputs}
            python_executor_tool = LocalPythonExecutorTool(
                output_type=self._structured_output_type,
                initial_state=initial_state,
                additional_authorized_imports=self.config.code_executor_additional_imports,
                executor_kwargs=self.config.code_executor_kwargs,
            )
            # Execute procedural parameter code in the python environment
            for name, param in procedural_inputs.items():
                exec_result = python_executor_tool._execute_code(param.value)
                if exec_result.success:
                    logger.debug(f"Executed procedural parameter '{name}' in python environment")
                else:
                    logger.warning(f"Procedural parameter '{name}' failed to execute: {exec_result.error}")
            # Inject ai_function tools as callables in the code execution environment
            ai_fn_tools = {t.__name__: t.call for t in tools if isinstance(t, AIFunction)}
            if ai_fn_tools:
                python_executor_tool._code_executor.send_variables(ai_fn_tools)
            tools = tools + [python_executor_tool.python_executor]
        else:
            python_executor_tool = None

        # Create system prompt
        system_prompt = self._create_system_prompt()

        # Create agent with or without structured output, passing conversation history
        # Note: tool_executor is already wrapped by decorator in agent_kwargs
        # Remove `messages` from agent_kwargs to avoid duplicate keyword argument
        agent_kwargs = self.config.agent_kwargs.copy()
        messages_from_kwargs = agent_kwargs.pop("messages", [])

        return Agent(
            model=self.config.model,
            system_prompt=system_prompt,
            tools=tools,
            structured_output_model=self._structured_output_type if self._is_structured_output_enabled else None,
            messages=messages if messages is not None else messages_from_kwargs,
            **agent_kwargs,  # type: ignore[misc, arg-type]
        )

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        base_prompt = self.config.system_prompt or "You are an expert assistant who can solve any task"
        return base_prompt

    def _final_answer_prompt(self) -> str:
        """Generate a prompt describing the possible ways to return an output."""
        is_code_execution_enabled = self.config.code_execution_mode != CodeExecutionMode.DISABLED
        parts = []
        if self._is_structured_output_enabled:
            parts.append(f"use the {self._structured_output_type.__name__} tool")
        if is_code_execution_enabled:
            final_answer_signature = generate_signature_from_model(self._structured_output_type)
            parts.append(f"call {final_answer_signature} from inside the python_executor tool")
        return f"\nIMPORTANT: To provide your final result, {' or '.join(parts)}."

    def _add_prompt(self, base_prompt: str, bound_args: dict[str, Any]) -> str:
        """Create the system prompt for the agent."""
        is_code_execution_enabled = self.config.code_execution_mode != CodeExecutionMode.DISABLED

        if is_code_execution_enabled:
            parts = [
                "",
                "You have access to a python execution environment.",
                "Use it if needed, but prefer using tool calls directly if the task can be accomplished "
                "without writing code.",
                f"The following modules are available for import: "
                f"{', '.join(SAFE_BUILTINS + self.config.code_executor_additional_imports)}.",
                "Modules not listed above are not available for security reasons. "
                "You cannot use the `os` module. You cannot use the `open(...)` builtin.",
            ]
            # Add a list of available variables and their representation
            procedural_parts: list[str] = []
            regular_parts: list[str] = []
            for k, v in bound_args.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, ParameterView) and v.procedural and v.value:
                    procedural_parts.append(f"\n```python\n{extract_signatures(v.value)}\n```")
                    continue
                if isinstance(v, ParameterView):
                    v = v.value
                v_repr = truncate(repr(v), max_length=200)
                regular_parts.append(f"\n - {k}: {v_repr}")
            if procedural_parts:
                parts.append(
                    "\nThe following functions have already been executed and are available "
                    "in the python environment's namespace. You can call them directly:"
                )
                parts.extend(procedural_parts)
            if regular_parts:
                parts.append(
                    "\nThe following variables are already available in the python environment (DO NOT redefine them):"
                )
                parts.extend(regular_parts)
            # Mention ai_function tools available as callables in the code env
            ai_fn_names = [t.__name__ for t in (self.config.tools or []) if isinstance(t, AIFunction)]
            if ai_fn_names:
                parts.append(
                    "\nThe following tools are also callable as functions in the python environment: "
                    + ", ".join(f"`{n}`" for n in ai_fn_names)
                )
            base_prompt = "<environment>" + "\n".join(parts) + "\n</environment>\n\n" + base_prompt
        # Add instructions on how to return the result
        base_prompt += "\n" + self._final_answer_prompt()

        return base_prompt.strip()

    def _extract_result(
        self,
        response: AgentResult,
        invocation_state: dict[str, Any],
    ) -> Any:
        """Extract the result from agent response.

        For DISABLED mode, only structured output is used.
        For Pydantic returns, uses structured output.
        For non-JSON returns, extracts from invocation state.

        Args:
            response: The agent response
            invocation_state: State from agent invocation

        Returns:
            The extracted result

        Raises:
            AIFunctionError: If no result was produced
        """

        def _maybe_unwrap(output: Any) -> Any:
            return output.answer if self._is_return_wrapped else output

        # Check if the model produced a structured_output
        if hasattr(response, "structured_output") and response.structured_output:
            return _maybe_unwrap(response.structured_output)

        # Check if the model answered using the python executor
        if "python_executor_result" in invocation_state:
            result = invocation_state["python_executor_result"]
            return _maybe_unwrap(result)

        # We did non manage to get an output, raise an appropriate error

        if self.config.code_execution_mode == CodeExecutionMode.DISABLED:
            # Strands already tries to force the model to produce a structured output, we cannot recover from this
            raise AIFunctionError(
                message=f"ai_function '{self.__name__}' did not produce a structured output.",
                function_name=self.__name__,
            )
        # Create a suggestion based on what output modes are available
        guidance = self._final_answer_prompt()
        # Raise ValidationError to trigger retry with guidance
        raise ValidationError(
            function_name=self.__name__,
            validation_errors={"missing_result": (f"No result was produced. Agent response: '{response}'. {guidance}")},
        )


# Wrapper for Sync function so that __call__ is sync by default (needed for proper typing)
class SyncAIFunction(AIFunction[P, R]):
    """AIFunction variant whose ``__call__`` is synchronous."""

    func: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the AI function synchronously."""
        return self.call(*args, **kwargs)


# Wrapper for Async function so that __call__ is async by default (needed for proper typing)
class AsyncAIFunction(AIFunction[P, R]):
    """AIFunction variant whose ``__call__`` is asynchronous."""

    func: Callable[P, Awaitable[R]]

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the AI function asynchronously."""
        return await self.call_async(*args, **kwargs)
