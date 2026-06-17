"""Blocking bridge between sync callers and ``async def`` code."""

from collections.abc import Awaitable, Callable


def run_blocking[T](coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``coro_factory()`` to completion and return its result.

    The factory is invoked exactly once, inside an event loop owned by
    this call. If the caller is not currently inside a running event
    loop, a loop is started via ``asyncio.run`` on the calling thread;
    otherwise a worker thread is used so the nested ``asyncio.run`` does
    not clash with the outer loop. In both cases the calling thread
    blocks until the coroutine finishes.

    ``coro_factory`` is a factory rather than a pre-built coroutine so
    the awaitable is constructed inside the target loop; pre-building in
    one loop and awaiting in another is a runtime error in ``asyncio``.

    Args:
        coro_factory: Zero-argument callable returning a fresh awaitable.

    Returns:
        The value awaited from the coroutine.

    Raises:
        BaseException: Any exception raised by the awaited coroutine is
            re-raised unchanged on the calling thread.

    Concurrency:
        Blocks the calling thread. Safe to call from both sync contexts
        and from within a running event loop (the work is dispatched to a
        worker thread in the latter case). ``contextvars`` are copied
        into the worker thread so framework-managed context is preserved.
    """
    ...
