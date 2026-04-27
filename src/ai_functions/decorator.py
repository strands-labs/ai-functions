"""AI Function decorator for enhancing functions with AI capabilities.

This module provides the @ai_function decorator that transforms regular Python
functions into AI-enhanced functions using the Strands Agents SDK.

The decorator supports:
- Bare usage: @ai_function
- Parameterized usage: @ai_function(config=...)
"""

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar, Unpack, cast, overload

from .core import AsyncAIFunction, SyncAIFunction
from .types.ai_function import AIFunctionConfig, AIFunctionMergedKwargs, split_config_and_agent_kwargs
from .validation.post_conditions import (
    validate_post_condition_params,
    validate_post_condition_signature,
)

P = ParamSpec("P")
R = TypeVar("R")


def _build(
    fn: Callable[P, R | Awaitable[R]],
    config: AIFunctionConfig | None,
    **kwargs: Unpack[AIFunctionMergedKwargs],
) -> SyncAIFunction[P, R] | AsyncAIFunction[P, R]:
    """Build a SyncAIFunction or AsyncAIFunction from a callable and config/kwargs."""
    config_args, agent_args = split_config_and_agent_kwargs(**kwargs)
    base_config = config or AIFunctionConfig()
    config_args["agent_kwargs"] = base_config.agent_kwargs | agent_args
    resolved_config = dataclasses.replace(base_config, **config_args) if config_args else base_config

    for condition in resolved_config.post_conditions or []:
        validate_post_condition_signature(condition)
        validate_post_condition_params(condition, fn)

    if inspect.iscoroutinefunction(fn):
        return AsyncAIFunction(func=cast(Callable[P, Awaitable[R]], fn), config=resolved_config)
    return SyncAIFunction(func=cast(Callable[P, R], fn), config=resolved_config)


class _Decorator:
    """Utility class to get the correct typing."""

    def __init__(self, config: AIFunctionConfig | None, **kwargs: Unpack[AIFunctionMergedKwargs]):
        self.config = config
        self.kwargs = kwargs

    @overload
    def __call__(self, fn: Callable[P, Awaitable[R]], /) -> AsyncAIFunction[P, R]: ...  # type: ignore[overload-overlap]
    @overload
    def __call__(self, fn: Callable[P, R], /) -> SyncAIFunction[P, R]: ...

    def __call__(self, fn: Callable[P, R | Awaitable[R]]) -> SyncAIFunction[P, R] | AsyncAIFunction[P, R]:
        return _build(fn, self.config, **self.kwargs)


# Bare decorator: @ai_function
@overload
def ai_function(func: Callable[P, Awaitable[R]], /) -> AsyncAIFunction[P, R]: ...  # type: ignore[overload-overlap]
@overload
def ai_function(func: Callable[P, R], /) -> SyncAIFunction[P, R]: ...


# Parameterized decorator: @ai_function(...), @ai_function(config=...), etc.
@overload
def ai_function(
    func: None = None,
    *,
    config: AIFunctionConfig | None = None,
    **kwargs: Unpack[AIFunctionMergedKwargs],
) -> _Decorator: ...


def ai_function(
    func: Callable | None = None,
    *,
    config: AIFunctionConfig | None = None,
    **kwargs: Unpack[AIFunctionMergedKwargs],
) -> SyncAIFunction | AsyncAIFunction | _Decorator:
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
    if func is not None:
        return _build(func, config, **kwargs)
    return _Decorator(config=config, **kwargs)
