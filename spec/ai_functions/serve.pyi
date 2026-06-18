"""Host a persistent agent on the local coordinator.

:func:`serve` is the entry point for a script that IS an agent. It
registers the given :class:`~ai_functions.protocols.Spawnable` with the
coordinator and then blocks — keeping the thread visible in
``ai-functions ps`` and reachable via ``submit`` / ``notify`` from
other processes — until a termination signal arrives.

Contrast with :func:`ai_functions.connect`:

- ``connect`` yields a bare :class:`CoordinatorClient` for
  **client-only** scripts that observe or control threads hosted
  elsewhere. No worker is registered; nothing stays alive after the
  ``async with`` body exits.
- ``serve`` registers a local worker, spawns ``target``, and keeps the
  script running. Other processes can see and message the agent.

The typical shape is one ``serve`` call at the bottom of a script::

    import ai_functions

    helper = ai_function(str)(lambda q: f"...{q}...")

    if __name__ == "__main__":
        ai_functions.serve(helper, thread_name="helper")

Running ``python helper.py`` now blocks; another terminal can see the
agent via ``ai-functions ps`` and talk to it via ``ai-functions submit`` / ``ai_functions
attach``. Ctrl-C shuts the agent down cleanly: the thread is
terminated, its teardown runs, the worker is deregistered, the client
is closed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .handle import ThreadHandle
from .protocols import Spawnable


async def aserve[**P, T](
    target: Spawnable[P, T],
    *args: Any,  # pyright: ignore[reportExplicitAny]
    start: bool = False,
    url: str | None = None,
    thread_name: str | None = None,
    metadata: dict[str, object] | None = None,
    on_ready: Callable[[ThreadHandle[P, T]], None] | None = None,
    **kwargs: Any,  # pyright: ignore[reportExplicitAny]
) -> T | None:
    """Async variant of :func:`serve`.

    Performs the following, in order:

    1. Resolve the coordinator (``url`` > ``AI_FUNCTIONS_COORDINATOR_URL`` >
       runtime file) and open a :class:`CoordinatorClient`.
    2. Create a :class:`~ai_functions.runtime.LocalWorker` bound to the
       client and register it.
    3. Call ``worker.spawn_locally(target, thread_name=...,
       metadata=...)`` to produce a handle.
    4. If ``args`` / ``kwargs`` are given, OR ``start=True``, kick off
       exactly one cycle via ``handle.run(*args, **kwargs)``. The call
       is launched as a background task — ``aserve`` does NOT await
       the result before entering the wait loop, so the agent is
       reachable by other clients while the initial cycle runs.
    5. Install signal handlers for SIGINT / SIGTERM and await until
       one fires, OR the thread reaches a terminal
       :class:`ThreadStatus`, OR the coordinator connection drops,
       whichever comes first. A dropped connection is re-raised (see
       ``Raises``) rather than treated as a clean shutdown, so the host
       fails loudly instead of exiting silently.
    6. On exit, ``terminate`` the thread, await termination, then
       ``close`` the worker and the client — in that order — even if
       an exception propagated.

    The initial cycle's result, if one was started, is returned. If
    no initial cycle was requested (the common "wait for someone to
    message me" case), this returns ``None`` on clean shutdown.

    Args:
        target: Any :class:`~ai_functions.protocols.Spawnable` — an
            ``ai_function`` or a custom class implementing ``to_thread`` /
            ``input_shape``.
        args: Positional arguments for the optional initial cycle.
            Passing any positional arg implies ``start=True``.
        start: When ``True``, kick off one cycle at registration with
            the given ``args`` / ``kwargs``. When ``False`` (default)
            and no positional / keyword args were supplied, no cycle
            is started; the thread remains ``NOT_STARTED`` until a
            peer calls ``submit``.
        url: Explicit coordinator URL; bypasses discovery when given.
        thread_name: Human label recorded on the thread's
            :class:`ThreadInfo` for telemetry and CLI display.
        metadata: Application metadata attached to the thread.
        on_ready: Optional callback invoked once, with the thread's
            :class:`ThreadHandle`, immediately after the thread is
            registered and before the wait loop begins. Use it to
            announce the running thread (e.g. print its id). Exceptions
            raised by the callback are suppressed so a faulty hook
            cannot bring down the host.
        kwargs: Keyword arguments for the optional initial cycle.
            Passing any keyword arg not in the reserved set (``url``,
            ``thread_name``, ``metadata``, ``start``) implies
            ``start=True``.

    Returns:
        The result of the initial cycle if one was started and
        completed before shutdown; ``None`` otherwise. If the initial
        cycle raised, the exception is re-raised after teardown runs.

    Raises:
        NoCoordinatorError: No coordinator could be discovered.
        OSError: The connection could not be established.
        ConnectionClosedError: The coordinator connection dropped while
            hosting (detected by the status-poll loop). Re-raised after
            teardown so the host process exits non-zero rather than
            silently; a requested shutdown (SIGINT / SIGTERM) takes
            precedence and still exits cleanly.
        ValueError: Reserved keyword arguments conflict with the
            spawnable's own parameters (e.g. the spawnable expects a
            parameter named ``url``). Callers work around this by
            building the client themselves with :func:`connect` and
            driving ``LocalWorker`` directly.
        Exception: Any exception raised by the initial cycle, if one
            was started.

    Ensures:
        - The thread is registered before the wait loop begins; other
          processes can discover it the moment ``aserve`` yields to
          the loop.
        - On exit (signal, thread terminal status, or exception): the
          thread is terminated, the worker is closed, and the client
          is closed — in that order, under a ``try / finally`` that
          survives exceptions in any step.
        - Signal handlers are removed before returning so that a
          caller using ``aserve`` in a larger event loop (e.g. a test
          harness) does not leak handlers.

    Concurrency:
        ``aserve`` is intended to own its event loop. Running it
        concurrently with other long-lived tasks in the same loop is
        supported, but the signal handlers it installs are process-
        wide; the last ``aserve`` on the process wins.
    """
    ...


def serve[**P, T](
    target: Spawnable[P, T],
    *args: Any,  # pyright: ignore[reportExplicitAny]
    start: bool = False,
    url: str | None = None,
    thread_name: str | None = None,
    metadata: dict[str, object] | None = None,
    on_ready: Callable[[ThreadHandle[P, T]], None] | None = None,
    **kwargs: Any,  # pyright: ignore[reportExplicitAny]
) -> T | None:
    """Synchronous entry point for the "my script is an agent" pattern.

    Thin wrapper around :func:`aserve` that runs it via
    :func:`ai_functions.utils.run_blocking`, so the common case — a module
    whose ``__main__`` block hosts an agent — does not require the
    user to write ``asyncio.run`` themselves::

        if __name__ == "__main__":
            ai_functions.serve(my_agent, thread_name="helper")

    Args:
        target: Spawnable to host; see :func:`aserve`.
        args: Initial-cycle positional arguments.
        start: Force an initial cycle even with no args.
        url: Explicit coordinator URL.
        thread_name: Human label for telemetry.
        metadata: Application metadata.
        on_ready: Optional callback invoked with the thread's
            :class:`ThreadHandle` once it is registered; see
            :func:`aserve`.
        kwargs: Initial-cycle keyword arguments.

    Returns:
        The initial cycle's result, if any; ``None`` otherwise.

    Raises:
        NoCoordinatorError: No coordinator was discovered.
        OSError: The connection could not be established.
        ConnectionClosedError: The coordinator connection dropped while
            hosting; see :func:`aserve`.
        Exception: Any exception raised by the initial cycle.
    """
    ...
