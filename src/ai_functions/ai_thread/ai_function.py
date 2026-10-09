"""Immutable ``@ai_function`` decorator and ``AIFunction`` template."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Any, Unpack, cast, final, overload, override  # pyright: ignore[reportAny]

import tstr
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.tools import AgentTool
from tstr import Template

from ..handle import ThreadHandle
from ..protocols import Spawnable
from ..types import InputShape
from ..utils import run_blocking

if TYPE_CHECKING:
    from ..types.graph import Result
from .ai_thread import AIThread
from .config import (
    AgentKwargs,
    ThreadConfig,
    ThreadMergedKwargs,
    split_config_and_agent_kwargs,
)
from .errors import AIFunctionError


def _merge_config(
    base: ThreadConfig,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> ThreadConfig:
    """Return ``base`` with ``kwargs`` merged into it.

    ``ThreadKwargs`` fields replace the base field directly; unknown keys are
    treated as ``strands.Agent`` kwargs and merged into ``base.agent_kwargs``.
    """
    thread_kwargs, agent_kwargs = split_config_and_agent_kwargs(**kwargs)
    combined: dict[str, Any] = {**base.agent_kwargs, **agent_kwargs}  # pyright: ignore[reportExplicitAny]
    merged_agent = AgentKwargs(**combined)  # type: ignore[typeddict-item]  # dynamic dict merge
    return dataclasses.replace(
        base,
        **dict(thread_kwargs),  # type: ignore[arg-type]  # TypedDict -> dataclass fields
        agent_kwargs=merged_agent,
    )


def _infer_input_shape(prompt_fn: Callable[..., Any]) -> InputShape:  # pyright: ignore[reportExplicitAny]
    """Derive an :class:`InputShape` from ``prompt_fn``'s signature.

    Rules:

    - Zero positional parameters → ``NO_ARGS``.
    - Exactly one positional parameter resolved to ``str`` → ``STR_PROMPT``.
    - Anything else → ``STRUCTURED``.

    Uses :func:`typing.get_type_hints` to resolve string-form annotations
    introduced by ``from __future__ import annotations``, ``TypeAlias``
    indirection, etc.
    """
    import typing

    try:
        sig = inspect.signature(prompt_fn)
    except (TypeError, ValueError):
        return InputShape.STRUCTURED
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        return InputShape.NO_ARGS
    if len(positional) == 1:
        try:
            hints = typing.get_type_hints(prompt_fn)
        except Exception:  # noqa: BLE001 — annotations may reference missing names
            return InputShape.STRUCTURED
        if hints.get(positional[0].name) is str:
            return InputShape.STR_PROMPT
    return InputShape.STRUCTURED


def _render_template(template: Template) -> str:
    """Render ``template`` as a prompt: dedented, with leading and trailing newlines stripped.

    Interpolations honor their conversion and format spec. A multi-line value
    whose placeholder is preceded only by whitespace on its line has its
    continuation lines indented to the placeholder's column. Nested
    ``Template`` values are rendered the same way.
    """
    template = tstr.dedent(template)
    out = [template.strings[0]]
    for interp, string in zip(template.interpolations, template.strings[1:], strict=True):
        value = interp.value
        if isinstance(value, Template):
            value = _render_template(value)
        rendered = format(tstr.convert(value, interp.conversion), interp.format_spec)
        line_prefix = "".join(out).rpartition("\n")[2]
        indent = line_prefix if line_prefix.isspace() else ""
        first, *rest = rendered.split("\n")
        out.append("\n".join([first, *(indent + line if line.strip() else line for line in rest)]))
        out.append(string)
    return "".join(out).strip("\n")


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

    __slots__ = (
        "_prompt_fn",
        "_output_type",
        "_config",
        "_executor",
        "_name",
        "_doc",
        "_input_shape",
    )

    def __init__(
        self,
        prompt_fn: Callable[P, str | Template | None],
        output_type: type[T],
        config: ThreadConfig,
    ) -> None:
        self._prompt_fn: Callable[P, str | Template | None] = prompt_fn
        self._output_type: type[T] = output_type
        self._config: ThreadConfig = config
        self._executor: object = None
        self._name: str = getattr(prompt_fn, "__name__", "ai_function")
        self._doc: str | None = inspect.getdoc(prompt_fn)
        self._input_shape: InputShape = _infer_input_shape(prompt_fn)

    @property
    def name(self) -> str:
        """Name of the wrapped function."""
        return self._name

    @property
    def config(self) -> ThreadConfig:
        """The default ``ThreadConfig`` attached to this template."""
        return self._config

    @property
    def output_type(self) -> type[T]:
        """The declared output type for this function."""
        return self._output_type

    @property
    def prompt_fn(self) -> Callable[P, str | Template | None]:
        """The user-provided prompt builder."""
        return self._prompt_fn

    @property
    def input_shape(self) -> InputShape:
        """Coarse classification of this template's ``prompt_fn`` signature."""
        return self._input_shape

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

        Strategy:
            1. Call ``self.prompt_fn(*args, **kwargs)``; if it is a coroutine,
               await it (``async def`` prompt bodies are supported).
            2. A ``str`` result is the prompt as-is. A ``Template`` result
               (a t-string on Python 3.14+) is dedented, rendered with
               multi-line values indented to their placeholder's column, and
               stripped of leading and trailing newlines.
            3. If ``prompt_fn`` returns ``None``, interpret its docstring as a
               ``tstr`` template, interpolate it with the bound arguments and
               the function's globals as context, and render it as in step 2.
        """
        result = self._prompt_fn(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = await result
        if isinstance(result, str):
            return result
        if isinstance(result, Template):
            return _render_template(result)
        if result is not None:
            raise AIFunctionError(
                f"prompt_fn must return str, Template, or None, got {type(result).__name__}",
                function_name=self._name,
            )

        doc = self._doc
        if not doc:
            raise AIFunctionError(
                "prompt_fn returned None and has no docstring to use as a template",
                function_name=self._name,
            )

        # Build context from bound arguments
        sig = inspect.signature(self._prompt_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        context: dict[str, object] = dict(bound.arguments)

        if hasattr(self._prompt_fn, "__globals__"):
            fn_globals = self._prompt_fn.__globals__
        else:
            fn_globals = dict[str, Any]()  # pyright: ignore[reportExplicitAny]
        # use_eval=True (matching both ancestor implementations) so docstring
        # templates may use attribute access and expressions — e.g.
        # ``{response.summary}`` or ``{bullet_points(items)}`` — not just bare
        # variable names. The template is the function's own docstring (trusted
        # author-supplied text), interpolated with the call's bound arguments.
        template = tstr.generate_template(doc, context, globals=fn_globals, use_eval=True)  # pyright: ignore[reportUnknownMemberType]
        return _render_template(template)

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
        merged = _merge_config(self._config, **config_overrides)
        return AIThread(self, merged)

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
        from .. import InMemoryCoordinator
        from ..runtime.worker import LocalWorker

        target = self.replace(**config_overrides) if config_overrides else self
        coord = InMemoryCoordinator()
        worker = LocalWorker(coord)
        return await worker.spawn_locally(target)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one throwaway cycle standalone and return the result.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The typed cycle result.

        Strategy:
            When invoked inside a running thread (an ambient
            :class:`ThreadScope` is set — e.g. this function used as a tool by
            another agent), spawn on that thread's coordinator with the caller
            as ``parent_id``, so the sub-call lands in the same event log and
            build_graph wires it as a child. Otherwise fall back to a private
            throwaway coordinator (``self.spawn``). Either way the child thread
            is torn down before returning.
        """
        handle = await self._spawn_in_context()
        try:
            return await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()

    async def _spawn_in_context(self) -> ThreadHandle[P, T]:
        """Spawn for a ``__call__`` cycle, reusing the ambient thread scope if set.

        With an active :class:`ThreadScope` naming a thread, spawn on its
        coordinator and the caller's worker (looked up from the caller's
        ``ThreadInfo`` — no cloudpickle) with ``parent_id`` set to the caller.
        With a coordinator-only scope (:func:`ai_functions.scope`), spawn on
        that coordinator with no parent. With no scope, defer to
        :meth:`spawn`'s private-worker path.
        """
        from ..types import current_thread_scope

        scope = current_thread_scope()
        if scope is None:
            return await self.spawn()
        if scope.thread_id is None:
            return await scope.coordinator.spawn(self)
        caller = await scope.coordinator.get_thread_info(scope.thread_id)
        return await scope.coordinator.spawn(
            self,
            worker_id=caller.worker_id,
            parent_id=scope.thread_id,
        )

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
        return run_blocking(lambda: self(*args, **kwargs))

    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]:  # pyright: ignore[reportAny]
        """Run one cycle like ``__call__``, returning a :class:`Result` node.

        The traced counterpart of ``__call__`` for optimization workflows:
        the returned ``Result`` wraps the value plus the provenance needed to
        reconstruct the computation graph afterwards (coordinator, thread id,
        and the dataflow inputs discovered in the arguments)::

            cat = await joke_writer.trace(topic="cats", joke_guidelines=await memory.recall("joke_guidelines"))
            email = await email_writer.trace(jokes=cat, formatting_guidelines=await memory.recall("fmt"))
            await optimizer.step(email, "titles please", backends=[memory])

        Accepts ``*args: Any`` rather than the wrapped signature ``P`` on
        purpose: each argument may be the declared value *or* a
        ``ParameterView`` / ``Result`` handle wrapping it (also nested in
        dicts, lists, tuples), and a ``ParamSpec`` cannot express "``P`` with
        every parameter widened to accept its handle". Handles are recorded as
        graph edges and unwrapped to their ``.value`` before ``prompt_fn`` runs,
        so the function body still receives the plain declared types.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``. May contain
                ``ParameterView`` / ``Result`` handles (also nested in dicts,
                lists, tuples); they are recorded as graph edges and unwrapped
                to their values before the prompt is built.
            kwargs: Keyword arguments, same handling as ``args``.

        Returns:
            A ``Result[T]`` wrapping the typed cycle result.

        Ensures:
            - ``ParameterView`` inputs whose recall event was not emitted at
              recall time (no thread existed yet) are emitted against the
              traced thread before the cycle runs; views already emitted
              elsewhere are not re-emitted (one logical recall, one event).
            - The traced thread is torn down before returning; its event log
              survives on the coordinator for ``build_graph_from_result``.

        Note:
            Pass handles directly (``jokes=cat``) to preserve graph edges.
            Interpolating into an f-string (``jokes=f"Joke: {cat}"``) computes
            the same value but drops the optimization edge.
        """
        from ..types.graph import ParameterView, Result, collect_nodes

        inputs = collect_nodes((args, kwargs))
        handle = await self._spawn_in_context()
        try:
            for node in inputs:
                if isinstance(node, ParameterView):
                    await node.backend.emit_recall(node, handle.coordinator, handle.id)
            value = await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()
        return Result(
            value=value,
            coordinator=handle.coordinator,
            thread_id=handle.id,
            inputs=inputs,
        )

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
        merged = _merge_config(self._config, **kwargs)
        return AIFunction(self._prompt_fn, self._output_type, merged)

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
        del kwargs
        template = self
        tool_name = self._config.name or self._name
        tool_description = self._config.description or self._doc or self._name

        async def _invoke(**call_kwargs: object) -> T:
            return await template(**call_kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue]

        _invoke.__name__ = tool_name
        _invoke.__doc__ = tool_description
        _invoke.__signature__ = inspect.signature(self._prompt_fn)  # type: ignore[attr-defined]
        annotations = dict(getattr(self._prompt_fn, "__annotations__", {}))
        annotations["return"] = self._output_type
        _invoke.__annotations__ = annotations

        agent_tool = _strands_tool(
            name=tool_name,
            description=tool_description,
        )(_invoke)
        return [agent_tool]

    @override
    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        del consumer_id, kwargs
        return None

    @override
    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        del consumer_id, kwargs
        return None


# ── Decorator ──


def _infer_output_type(prompt_fn: Callable[..., Any]) -> type[Any]:  # pyright: ignore[reportExplicitAny]
    """Extract the declared output type from ``prompt_fn``'s return annotation.

    Uses :func:`typing.get_type_hints` so that string-form annotations
    introduced by ``from __future__ import annotations`` are resolved.

    Args:
        prompt_fn: The prompt function decorated by a bare ``@ai_function``.

    Returns:
        The type named by the function's ``-> T`` return annotation.

    Raises:
        TypeError: If the function has no return annotation (or it is
            ``None``). The message directs the caller to annotate the
            return type or use the ``@ai_function[OutputType]`` form.
    """
    import typing

    try:
        hints = typing.get_type_hints(prompt_fn)
    except Exception as exc:  # noqa: BLE001 — annotations may reference missing names
        name = getattr(prompt_fn, "__name__", "ai_function")
        raise TypeError(
            f"@ai_function could not resolve the return annotation of {name!r}: {exc}. "
            f"Annotate the return type or use the @ai_function[OutputType] form."
        ) from exc
    output = hints.get("return")
    if output is None:
        name = getattr(prompt_fn, "__name__", "ai_function")
        raise TypeError(
            f"@ai_function requires a return annotation to infer the output type of {name!r}. "
            f"Annotate the return type (def {name}(...) -> OutputType) "
            f"or use the @ai_function[OutputType] form."
        )
    return output


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

    __slots__ = ("_output_type", "_config")

    def __init__(self, output_type: type[T], config: ThreadConfig) -> None:
        self._output_type: type[T] = output_type
        self._config: ThreadConfig = config

    @overload
    def __call__[**P](self, prompt_fn: Callable[P, str | Template | None], /) -> AIFunction[P, T]: ...
    @overload
    def __call__[**P](
        self,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> Callable[[Callable[P, str | Template | None]], AIFunction[P, T]]: ...
    def __call__[**P](  # type: ignore[misc]  # overload implementation not visible to checker
        self,
        prompt_fn: Callable[P, str | Template | None] | None = None,
        /,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> AIFunction[P, T] | Callable[[Callable[P, str | Template | None]], AIFunction[P, T]]:
        base = config if config is not None else self._config
        merged = _merge_config(base, **kwargs)

        def _decorator(fn: Callable[P, str | Template | None]) -> AIFunction[P, T]:
            return AIFunction(fn, self._output_type, merged)

        if prompt_fn is not None:
            return _decorator(prompt_fn)
        return _decorator


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

    __slots__ = ()

    def __getitem__[T](self, output_type: type[T]) -> _TypedDecorator[T]:
        """Bind an explicit output type for ``@ai_function[T]``.

        Args:
            output_type: The declared output type for the wrapped function.

        Returns:
            A decorator that turns a prompt function into an ``AIFunction``
            with output type ``output_type``.
        """
        return _TypedDecorator(output_type, ThreadConfig())

    @overload
    def __call__[**P, T](self, prompt_fn: Callable[P, T], /) -> AIFunction[P, T]: ...
    @overload
    def __call__[**P, T](
        self,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> Callable[[Callable[P, T]], AIFunction[P, T]]: ...
    def __call__[**P, T](  # type: ignore[misc]  # overload implementation not visible to checker
        self,
        prompt_fn: Callable[P, T] | None = None,
        /,
        *,
        config: ThreadConfig | None = None,
        **kwargs: Unpack[ThreadMergedKwargs],
    ) -> AIFunction[P, T] | Callable[[Callable[P, T]], AIFunction[P, T]]:
        base = config if config is not None else ThreadConfig()
        merged = _merge_config(base, **kwargs)

        def _decorator(fn: Callable[P, T]) -> AIFunction[P, T]:
            output = _infer_output_type(fn)
            # ``fn`` builds the prompt (returns ``str | None``); its return
            # annotation names the output type, so the cast is intentional.
            return AIFunction(cast("Callable[P, str | Template | None]", fn), output, merged)

        if prompt_fn is not None:
            return _decorator(prompt_fn)
        return _decorator


ai_function: _AIFunctionFactory = _AIFunctionFactory()
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
