"""Immutable ``@ai_function`` decorator and ``AIFunction`` template."""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence, Unpack, final, overload, override

from strands.tools import ToolProvider
from strands.types.tools import AgentTool
from tstr import Template

from ..protocols import Spawnable
from ..handle import ThreadHandle
from ..types.graph import Result
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
        prompt_fn: Callable[P, str | Template | None],
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
    def prompt_fn(self) -> Callable[P, str | Template | None]:
        """The user-provided prompt builder."""
        ...

    async def render_prompt(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Render the prompt string this template produces for the given arguments.

        The same rendering ``AIThread.execute`` performs before a cycle, exposed
        on the template so callers that need the prompt *without* running the
        function (e.g. cost forecasters ranking models per-task) can obtain it.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The rendered prompt string.

        Raises:
            AIFunctionError: ``prompt_fn`` returned ``None`` and has no
                docstring to use as a template, or returned something other
                than ``str``, ``Template``, or ``None``.

        Ensures:
            - A ``str`` result is returned as-is.
            - A ``Template`` result, or the docstring template when
              ``prompt_fn`` returns ``None``, is dedented, rendered with
              multi-line values indented to their placeholder's column, and
              stripped of leading and trailing newlines.
        """
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

    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]:
        """Run one cycle like ``__call__``, returning a :class:`Result` node.

        The traced counterpart of ``__call__`` for optimization workflows: the
        returned ``Result`` wraps the value plus the provenance needed to
        reconstruct the computation graph afterwards (coordinator, thread id,
        and the ``ParameterView`` / ``Result`` handles discovered in the
        arguments, nested containers included)::

            cat = await joke_writer.trace(topic="cats", joke_guidelines=await memory.recall("joke_guidelines"))
            email = await email_writer.trace(jokes=cat, formatting_guidelines=await memory.recall("fmt"))
            await optimizer.step(email, "titles please", backends=[memory])

        Accepts ``*args: Any`` rather than the wrapped signature ``P`` on
        purpose: each argument may be the declared value *or* a
        ``ParameterView`` / ``Result`` handle wrapping it, and a ``ParamSpec``
        cannot express "``P`` with every parameter widened to accept its
        handle". Handles are unwrapped before ``prompt_fn`` runs, so the body
        still receives the plain declared types.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``; handles are
                recorded as graph edges and unwrapped before the prompt is built.
            kwargs: Keyword arguments, same handling as ``args``.

        Returns:
            A ``Result[T]`` wrapping the typed cycle result.

        Ensures:
            - ``ParameterView`` inputs whose recall event was not emitted at
              recall time are emitted against the traced thread before the
              cycle runs; views already emitted elsewhere are not re-emitted
              (one logical recall, one event).
            - The traced thread is torn down before returning; its event log
              survives on the coordinator for ``build_graph_from_result``.

        Note:
            Pass handles directly (``jokes=cat``) to preserve graph edges;
            interpolating into an f-string computes the same value but drops
            the optimization edge.
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


@final
class _TypedDecorator[T]:
    """Decorator bound to an explicit output type via ``ai_function[T]``.

    Produced by subscripting :data:`ai_function`. Applying it to a prompt
    function yields an ``AIFunction`` whose output type is ``T``. It may be
    applied directly (``@ai_function[T]``) or called first with
    configuration (``@ai_function[T](model=...)``) to obtain a configured
    decorator.

    Args:
        output_type: The declared output type ``T`` bound by the subscript.
        config: Default per-decorator configuration.
    """

    def __init__(self, output_type: type[T], config: ThreadConfig) -> None: ...
    @overload
    def __call__[**P](self, prompt_fn: Callable[P, str | Template | None], /) -> AIFunction[P, T]: ...
    @overload
    def __call__[**P](
        self,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> Callable[[Callable[P, str | Template | None]], AIFunction[P, T]]: ...
    def __call__[**P](  # type: ignore[misc]
        self,
        prompt_fn: Callable[P, str | Template | None] | None = None,
        /,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> AIFunction[P, T] | Callable[[Callable[P, str | Template | None]], AIFunction[P, T]]: ...


@final
class _AIFunctionFactory:
    """The ``ai_function`` decorator object.

    Supports two forms:

    - ``@ai_function[OutputType]`` — the output type is given explicitly in
      brackets and is always used, regardless of any return annotation on
      the prompt function. This is the only form that type-checks cleanly.
    - ``@ai_function`` — the output type is inferred from the prompt
      function's return annotation (``def f(...) -> OutputType``).

    Both forms may be called with configuration before being applied
    (``@ai_function[T](model=...)`` or ``@ai_function(model=...)``).
    """

    def __getitem__[T](self, output_type: type[T]) -> _TypedDecorator[T]:
        """Bind an explicit output type for ``@ai_function[T]``.

        Args:
            output_type: The declared output type for the wrapped function.

        Returns:
            A decorator that turns a prompt function into an ``AIFunction``
            with output type ``output_type``.
        """
        ...

    @overload
    def __call__[**P, T](self, prompt_fn: Callable[P, T], /) -> AIFunction[P, T]: ...
    @overload
    def __call__[**P, T](
        self,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> Callable[[Callable[P, T]], AIFunction[P, T]]: ...
    def __call__[**P, T](  # type: ignore[misc]
        self,
        prompt_fn: Callable[P, T] | None = None,
        /,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> AIFunction[P, T] | Callable[[Callable[P, T]], AIFunction[P, T]]: ...


ai_function: _AIFunctionFactory
"""Decorator that wraps a prompt function as an ``AIFunction``.

Use ``@ai_function[OutputType]`` to declare the output type explicitly (the
type in brackets is always used), or a bare ``@ai_function`` to infer the
output type from the prompt function's return annotation. Either form
accepts configuration when called: ``@ai_function[T](model=...)`` or
``@ai_function(model=...)``. ``ThreadKwargs`` keys update the config fields
directly; others merge into ``agent_kwargs``.

Raises:
    TypeError: If a bare ``@ai_function`` is applied to a prompt function
        that has no usable return annotation. Either annotate the return
        type or use the ``@ai_function[OutputType]`` form.
"""
