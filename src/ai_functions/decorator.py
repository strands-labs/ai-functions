"""AI Function decorator for enhancing functions with AI capabilities.

This module provides the @ai_function decorator that transforms regular Python
functions into AI-enhanced functions using the Strands Agents SDK.

The decorator supports:
- Bare usage: @ai_function
- Parameterized usage: @ai_function(config=...)
"""

import dataclasses
from typing import Callable, TypeVar, Unpack, overload

from .core import AIFunction
from .types.ai_function import AIFunctionConfig, AIFunctionMergedKwargs, split_config_and_agent_kwargs
from .validation.post_conditions import (
    validate_post_condition_params,
    validate_post_condition_signature,
)

# Type variable for the decorated function
F = TypeVar("F", bound=Callable)


# Overload: parameterized usage - @ai_function(), @ai_function(config=...), @ai_function(max_attempts=5)
@overload
def ai_function(
    func: None = None,
    *,
    config: AIFunctionConfig | None = None,
    **kwargs: Unpack[AIFunctionMergedKwargs],
) -> Callable[[F], AIFunction]: ...


# Overload: bare decorator usage - @ai_function
@overload
def ai_function(func: F) -> AIFunction: ...


def ai_function(
    func: Callable | None = None,
    *,
    config: AIFunctionConfig | None = None,
    **kwargs: Unpack[AIFunctionMergedKwargs],
) -> AIFunction | Callable[[Callable], AIFunction]:
    """Transform a Python function into an AI-powered function.

    The function's docstring serves as the prompt template with ``{param_name}``
    placeholders. The return type annotation defines the expected output structure.

    Args:
        func: Function to decorate.
        config: ``AIFunctionConfig`` object with model, tools, post_conditions, etc.
        **kwargs: Kwargs matching ``AIFunctionConfig`` fields (``max_attempts``,
            ``system_prompt``, etc.) override config values. Other kwargs (``temperature``,
            ``max_tokens``, etc.) are merged into ``config.agent_kwargs``.

    Returns:
        ``AIFunction`` wrapper callable like the original function.

    Raises:
        ValueError: Missing return type annotation.
        ValidationError: Post-condition failure.
        AIFunctionError: General execution failure.

    Example::

        @ai_function
        def summarize(text: str) -> str:
            '''Summarize: {text}'''

        @ai_function(max_attempts=5, temperature=0.7)
        def generate(prompt: str) -> str:
            '''Generate: {prompt}'''

        @ai_function(config=...)
        def generate(prompt: str) -> str:
            '''Generate: {prompt}'''
    """

    def decorator(fn: Callable) -> AIFunction:
        # Partition kwargs into config fields vs agent kwargs
        config_args, agent_args = split_config_and_agent_kwargs(**kwargs)

        # Merge agent_args into agent_kwargs if present
        base_config = config or AIFunctionConfig()

        # Apply merged agent_kwargs
        config_args["agent_kwargs"] = base_config.agent_kwargs | agent_args

        # Create resolved config (replace only if we have overrides)
        resolved_config = dataclasses.replace(base_config, **config_args) if config_args else base_config

        # Validate post-condition signatures and parameters
        for condition in resolved_config.post_conditions or []:
            validate_post_condition_signature(condition)
            validate_post_condition_params(condition, fn)

        # Create the AIFunction wrapper
        wrapper = AIFunction(func=fn, config=resolved_config)

        return wrapper

    # Handle bare decorator usage: @ai_function
    if func is not None:
        return decorator(func)

    # Handle parameterized decorator usage: @ai_function(...)
    return decorator
