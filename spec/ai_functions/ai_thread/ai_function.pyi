"""Immutable ``@ai_function`` decorator and ``AIFunction`` template."""

from __future__ import annotations

from typing import Callable, Hashable, Sequence, Unpack, final, overload, override

from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..protocols import Spawnable
from ..handle import ThreadHandle
from .ai_thread import AIThread
from .config import (
    ThreadConfig,
    ThreadMergedKwargs,
)


@final
class AIFunction[**P, T](ToolProvider, Spawnable[P, T]):
    """Immutable AI function template; factory for ``AIThread`` instances.

    Args:
        prompt_fn: User function that builds the prompt; ``None`` means
            use the function's docstring.
        output_type: The declared output type for this function.
        config: Default per-thread configuration.

    Implements:
        Spawnable, strands.tools.ToolProvider.
    """

    def __init__(
        self,
        prompt_fn: Callable[P, str | None],
        output_type: type[T],
        config: ThreadConfig,
    ) -> None: ...

    @property
    def name(self) -> str:
        """Name of the wrapped function."""
        ...

    @property
    def config(self) -> ThreadConfig:
        """The default ``ThreadConfig`` attached to this template."""
        ...

    @property
    def output_type(self) -> type[T]:
        """The declared output type for this function."""
        ...

    @property
    def prompt_fn(self) -> Callable[P, str | None]:
        """The user-provided prompt builder."""
        ...

    # ── Spawnable ──

    def to_thread(
        self,
        **config_overrides: Unpack[ThreadMergedKwargs],
    ) -> AIThread[P, T]:
        """Produce a fresh ``AIThread`` bound to this template.

        Args:
            config_overrides: Kwargs matching ``ThreadConfig`` fields update
                the config directly; others merge into ``agent_kwargs``.

        Returns:
            A new ``AIThread`` with its own buffer and resolved config.

        Ensures:
            - The returned thread is unbound.
            - Successive calls return independent instances with no
              shared state.
        """
        ...

    # ── One-shot execution (sugar) ──

    async def spawn(
        self,
        **config_overrides: Unpack[ThreadMergedKwargs],
    ) -> ThreadHandle[P, T]:
        """Spawn this template on a private in-process worker and return its handle.

        Convenience for scripts and examples that want a multi-turn
        session without standing up a coordinator and worker pair
        explicitly. The returned handle is backed by a coordinator owned
        by a fresh :class:`LocalWorker`; both stay alive for as long as
        the handle is reachable.

        Args:
            config_overrides: Kwargs matching ``ThreadConfig`` fields update
                the config directly; others merge into ``agent_kwargs``.

        Returns:
            A ``ThreadHandle`` in ``NOT_STARTED`` state bound to a
            private worker.

        Ensures:
            - Each call returns a handle on an independent worker with
              no shared state.
            - The caller owns the handle's lifecycle; teardown requires
              ``await handle.terminate()`` or
              ``await handle.terminate_now()``.

        Strategy:
            1. Merge ``config_overrides`` into this template
               (``self.replace``) if any.
            2. Construct a fresh :class:`InMemoryCoordinator` and
               :class:`LocalWorker`.
            3. ``await worker.spawn_locally(target)`` and return the handle.
        """
        ...

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one throwaway cycle standalone and return the result.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The typed cycle result.

        Strategy:
            1. ``handle = self.spawn()``.
            2. ``await handle.run(*args, **kwargs)``.
            3. ``await handle.terminate_now()``.
        """
        ...

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Blocking wrapper around ``__call__``.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The typed cycle result.

        Concurrency:
            Blocks the calling thread until the cycle completes.
        """
        ...

    # ── Template variants ──

    def replace(self, **kwargs: Unpack[ThreadMergedKwargs]) -> AIFunction[P, T]:
        """Return a new template with ``kwargs`` merged into its config.

        Args:
            kwargs: Kwargs matching ``ThreadConfig`` fields update the
                config directly; others merge into ``agent_kwargs``.

        Returns:
            A new ``AIFunction`` that differs from ``self`` only in the
            merged fields.
        """
        ...

    # ── ToolProvider ──

    @override
    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Expose this template as a Strands tool for other agents.

        Args:
            kwargs: Ignored; present for protocol compatibility.

        Returns:
            One or more ``AgentTool`` instances whose invocation calls
            ``__call__``.
        """
        ...

    @override
    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...

    @override
    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        ...


# ── Decorator ──


@overload
def ai_function[**P, T](
    output: type[T],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]: ...


@overload
def ai_function[**P, T](
    output: type[T],
    *,
    config: ThreadConfig | None = None,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]: ...


def ai_function[**P, T](  # type: ignore[misc]
    output: type[T],
    *,
    config: ThreadConfig | None = None,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]:
    """Wrap a prompt function as an ``AIFunction`` with the given output type.

    Args:
        output: The declared output type for the wrapped function.
        config: Optional base ``ThreadConfig``; a fresh one is used if
            ``None``.
        kwargs: Overrides merged into ``config``; ``ThreadKwargs`` keys
            update config fields directly, others merge into
            ``agent_kwargs``.

    Returns:
        A decorator that turns the prompt function into an ``AIFunction``.
    """
    ...
