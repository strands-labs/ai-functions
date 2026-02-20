"""Validation layer for AI Functions.

This module provides post-condition runners that validate
outputs of AI functions. Supports sync, async, and AI-powered validators.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable

from ..types.ai_function import PostConditionResult

logger = logging.getLogger(__name__)


def _get_callable_name(func: Callable) -> str:
    """Extract a human-readable name from a callable."""
    return getattr(func, "__name__", str(func))


def get_failed_results(
    results: list[PostConditionResult],
    conditions: list[Callable],
) -> list[tuple[str, PostConditionResult]]:
    """Get list of (condition_name, result) tuples for failed conditions.

    Args:
        results: List of PostConditionResult from validate()
        conditions: List of condition functions (same order as results)

    Returns:
        List of (condition_name, result) tuples for conditions that failed
    """
    return [
        (_get_callable_name(cond), result)
        for cond, result in zip(conditions, results, strict=True)
        if not result.passed
    ]


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """Check if a callable is async (coroutine function or has async __call__)."""
    if inspect.iscoroutinefunction(func):
        return True
    # Check for async __call__ method on callable objects
    # Use callable() check as required by linter, then inspect the method
    if callable(func):
        try:
            call_method = type(func).__call__
            return inspect.iscoroutinefunction(call_method)
        except AttributeError:
            pass
    return False


def validate_post_condition_signature(condition: Callable) -> None:
    """Validate that a callable can be used as a post-condition.

    Post-conditions must have:
    - At least one positional parameter as the first parameter
    - The first parameter receives the AI function's result

    Args:
        condition: The post-condition function to validate

    Raises:
        ValueError: If the condition has no parameters or only **kwargs
    """
    sig = inspect.signature(condition)
    params = list(sig.parameters.values())
    name = _get_callable_name(condition)

    if not params:
        raise ValueError(
            f"Post-condition '{name}' must have at least one parameter to receive the result. "
            f"Example: def {name}(result): ..."
        )

    # Check that first param can accept a positional argument
    first_param = params[0]
    if first_param.kind == inspect.Parameter.VAR_KEYWORD:
        raise ValueError(
            f"Post-condition '{name}' must have a positional parameter to receive the result. "
            f"Found only **kwargs. Example: def {name}(result, **kwargs): ..."
        )

    if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
        raise ValueError(
            f"Post-condition '{name}' must have a named first parameter, not *args. "
            f"Example: def {name}(result, *args): ..."
        )

    if first_param.kind == inspect.Parameter.KEYWORD_ONLY:
        raise ValueError(
            f"Post-condition '{name}' first parameter must be positional. "
            f"Found keyword-only '{first_param.name}'. Example: def {name}(result, *, other): ..."
        )


def validate_post_condition_params(
    condition: Callable,
    ai_func: Callable,
) -> None:
    """Validate that post-condition parameters match AI function parameters.

    Post-condition parameters (beyond the first, which receives the result) must either:
    - Match a parameter name in the AI function signature
    - Be **kwargs (which accepts any arguments)

    Args:
        condition: The post-condition function to validate
        ai_func: The AI function whose parameters to check against

    Raises:
        ValueError: If a post-condition parameter doesn't exist in the AI function
    """
    cond_sig = inspect.signature(condition)
    cond_params = list(cond_sig.parameters.values())

    ai_sig = inspect.signature(ai_func)
    ai_param_names = set(ai_sig.parameters.keys())

    cond_name = _get_callable_name(condition)
    ai_name = _get_callable_name(ai_func)

    # Check each parameter after 'result'
    for param in cond_params[1:]:  # Skip first param (result)
        # **kwargs accepts anything
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        # *args - skip validation (will receive nothing, but that's ok)
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        # Named parameter must exist in AI function
        if param.name not in ai_param_names:
            raise ValueError(
                f"Post-condition '{cond_name}' has parameter '{param.name}' "
                f"that doesn't exist in AI function '{ai_name}'. "
                f"Available parameters: {sorted(ai_param_names)}"
            )


class PostConditionRunner:
    """Runs post-condition validators on AI function results.

    Post-conditions are functions that validate the output of an AI function.
    They can be:
    - Synchronous functions: `def check(result) -> PostConditionResult | None`
    - Asynchronous functions: `async def check(result) -> PostConditionResult | None`
    - AI-powered validators: `@ai_function` decorated functions

    Validators can return:
    - ``PostConditionResult(passed=True)`` - validation passed
    - ``PostConditionResult(passed=False, message="...")`` - validation failed
    - ``None`` - shorthand for passed (equivalent to ``PostConditionResult(passed=True)``)
    - Raising an exception - treated as failed with exception message

    Validators can access original input arguments by accepting **kwargs.

    Example:
        >>> import asyncio
        >>> from ai_functions.types.ai_function import PostConditionResult
        >>> def check_length(result: str) -> PostConditionResult | None:
        ...     if len(result) > 10:
        ...         return None  # Passed
        ...     return PostConditionResult(passed=False, message="Result too short")
        >>>
        >>> runner = PostConditionRunner()
        >>> results = asyncio.run(runner.validate([check_length], "short"))
        >>> results[0].passed
        False
    """

    def __init__(self, function_name: str = ""):
        """Initialize the post-condition runner.

        Args:
            function_name: Name of the AI function being validated (for error context)
        """
        self.function_name = function_name

    async def validate(
        self,
        conditions: list[Callable],
        result: Any,
        bound_args: dict[str, Any] | None = None,
    ) -> list[PostConditionResult]:
        """Run all post-conditions and return results.

        Args:
            conditions: List of validator functions to run
            result: The result to validate
            bound_args: Original input arguments (passed to validators that accept them)

        Returns:
            List of PostConditionResult for each condition (pass or fail)
        """
        bound_args = bound_args or {}

        tasks = [self._check_condition(condition, result, bound_args) for condition in conditions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any infrastructure exceptions to PostConditionResult
        processed_results = []
        for i, result_or_exception in enumerate(results):
            if isinstance(result_or_exception, Exception):
                condition_name = _get_callable_name(conditions[i])
                logger.error(
                    f"Post-condition '{condition_name}' validation infrastructure failure: "
                    f"{type(result_or_exception).__name__}: {result_or_exception}",
                    exc_info=result_or_exception,
                )
                processed_results.append(
                    PostConditionResult(
                        passed=False,
                        message=f"Infrastructure failure: {type(result_or_exception).__name__}: {result_or_exception}",
                    )
                )
            else:
                processed_results.append(result_or_exception)

        return processed_results

    async def _check_condition(
        self,
        condition: Callable,
        result: Any,
        bound_args: dict[str, Any],
    ) -> PostConditionResult:
        """Run a single post-condition and capture the result.

        Args:
            condition: The condition function to run
            result: The result to validate
            bound_args: Original input arguments

        Returns:
            PostConditionResult with pass/fail status (None treated as passed)
        """
        name = _get_callable_name(condition)

        try:
            kwargs_to_pass = self._build_kwargs_for_condition(condition, bound_args)

            if _is_async_callable(condition):
                condition_output = await condition(result, **kwargs_to_pass)
            else:
                condition_output = await asyncio.to_thread(condition, result, **kwargs_to_pass)

            # None means condition passed
            if condition_output is None:
                return PostConditionResult(passed=True)

            if not isinstance(condition_output, PostConditionResult):
                raise TypeError(
                    f"Post-condition '{name}' must return PostConditionResult or None, "
                    f"got {type(condition_output).__name__}"
                )
            return condition_output

        except Exception as e:
            return PostConditionResult(
                passed=False,
                message=f"Condition raised exception: {type(e).__name__}: {e}",
            )

    def _build_kwargs_for_condition(
        self,
        condition: Callable,
        bound_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs to pass to a condition based on its signature.

        Args:
            condition: The condition function
            bound_args: Original input arguments from the AI function

        Returns:
            Dictionary of kwargs to pass to the condition
        """
        sig = inspect.signature(condition)
        params = sig.parameters
        param_names = list(params.keys())

        # Check if condition accepts **kwargs
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        # Get the first param name (receives the result) to exclude from kwargs
        first_param_name = param_names[0] if param_names else None

        # Filter out the first param name to avoid conflict with positional result argument
        filtered_bound_args = {k: v for k, v in bound_args.items() if k != first_param_name}

        if has_var_keyword:
            return filtered_bound_args.copy()

        # Only pass bound_args that match parameter names (skip first param which receives result)
        return {
            param_name: filtered_bound_args[param_name]
            for param_name in param_names[1:]
            if param_name in filtered_bound_args
        }
