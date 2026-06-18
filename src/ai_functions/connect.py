"""Connect to the coordinator discovered on this host.

:func:`connect` is the low-level entry point used by client-only code
— scripts that inspect, observe, or control threads hosted in other
processes. It resolves the coordinator URL via
:func:`ai_functions.discover_coordinator`, opens a
:class:`~ai_functions.network.CoordinatorClient`, and yields it through an
async context manager that closes the client on exit.

Scripts that want to *host* an agent (register a worker, keep it alive
across many cycles, answer messages) should use :func:`ai_functions.serve`
instead — it takes care of the worker lifecycle and the
wait-for-signal loop.

Example — list threads and message one of them::

    import asyncio
    import ai_functions


    async def main() -> None:
        async with ai_functions.connect() as client:
            for info in await client.list_threads():
                print(info.thread_id, info.thread_name, info.status)
            await client.notify(target_id, "status?")


    asyncio.run(main())
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from .discovery import discover_coordinator
from .network import CoordinatorClient


@asynccontextmanager
async def _connect(url: str | None) -> AsyncGenerator[CoordinatorClient]:
    resolved = url if url is not None else discover_coordinator().url
    client = await CoordinatorClient.connect(resolved)
    try:
        yield client
    finally:
        await client.close()


def connect(url: str | None = None) -> AbstractAsyncContextManager[CoordinatorClient]:
    """Open a :class:`CoordinatorClient` against the local coordinator.

    URL resolution precedence:

    1. ``url`` argument, if given.
    2. ``AI_FUNCTIONS_COORDINATOR_URL`` environment variable.
    3. The runtime file written by ``ai-functions server``
       (see :mod:`ai_functions.discovery`).

    The returned object is an async context manager; on exit the
    underlying :class:`CoordinatorClient` is closed even if the
    ``async with`` body raises.

    Args:
        url: Explicit coordinator URL; bypasses discovery when given.

    Returns:
        An async context manager whose ``__aenter__`` resolves to a
        connected :class:`CoordinatorClient`.

    Raises:
        NoCoordinatorError: ``url`` was not given, the env var was not
            set, and no live coordinator was discovered on this host.
        OSError: The connection could not be established.

    Ensures:
        - The yielded client is connected and ready for RPC use before
          the ``async with`` body begins.
        - The client is closed before the context manager returns
          control, regardless of whether the body raised.

    Concurrency:
        Each call creates an independent connection; callers may open
        many concurrent clients against the same coordinator.
    """
    return _connect(url)
