"""Async utility functions for AI Functions.

This module provides utilities for running async code from sync contexts,
handling event loop management and context variable preservation.
"""

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function in a separate thread to avoid event loop conflicts.

    This utility handles the common pattern of running async code from sync contexts
    by using ThreadPoolExecutor to isolate the async execution. Context variables
    are preserved across the thread boundary using contextvars.copy_context().

    This approach is necessary because:
    1. asyncio.run() fails if called from within an existing event loop
    2. We need to preserve context variables (like logging context) across threads
    3. ThreadPoolExecutor provides clean isolation for the new event loop

    Args:
        async_func: A callable that returns an awaitable (typically a lambda)

    Returns:
        The result of the async function

    Example:
        >>> async def fetch_data():
        ...     return await some_async_operation()
        >>> result = run_async(lambda: fetch_data())
    """

    async def execute_async() -> T:
        return await async_func()

    def execute() -> T:
        return asyncio.run(execute_async())

    with ThreadPoolExecutor() as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, execute)
        return future.result()
