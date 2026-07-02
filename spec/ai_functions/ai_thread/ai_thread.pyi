"""AIThread — live, per-spawn instance backing an ``AIFunction``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from pydantic import BaseModel
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.hooks import HookProvider
from strands.types.content import Messages

from ..protocols import Thread
from ..types import ThreadContext
from .config import ThreadConfig
from .postcondition import PostCondition

if TYPE_CHECKING:
    from .ai_function import AIFunction


@dataclass(frozen=True)
class OutputSpec[T]:
    """Describes the AI output type and how to present it to the model."""

    output_type: type[T]
    """The user-specified output type (e.g. ``str``, ``float``, a pydantic
    model)."""

    is_pydantic: bool
    """True iff ``output_type`` is a ``pydantic.BaseModel`` subclass."""

    is_structured: bool
    """True iff the output can be expressed as a JSON schema for the model."""

    structured_output_model: type[BaseModel] | None
    """Pydantic model passed to Strands as the structured output model."""

    is_wrapped: bool
    """True iff ``structured_output_model`` wraps ``output_type`` in a
    ``FinalAnswer``."""

    @classmethod
    def from_type(cls, output_type: type[T], is_structured: bool = True) -> OutputSpec[T]:
        """Derive an ``OutputSpec`` from a type and a structured-output flag.

        Args:
            output_type: The user-declared output type.
            is_structured: Whether structured output should be used if
                possible.

        Returns:
            An ``OutputSpec`` describing how to request and parse
            ``output_type``.

        Ensures:
            - ``result.is_pydantic`` is true iff ``output_type`` is a
              ``BaseModel`` subclass.
            - ``result.structured_output_model is not None`` iff
              ``result.is_structured``.
            - ``result.is_wrapped`` is true iff
              ``structured_output_model`` wraps ``output_type``.
        """
        ...


@final
class AIThread[**P, T](Thread[P, T]):
    """Live stateful thread instance bound to an ``AIFunction`` template.

    Args:
        template: The ``AIFunction`` whose ``prompt_fn`` and
            ``output_type`` this thread invokes.
        config: The resolved ``ThreadConfig`` (template config merged
            with overrides).

    Raises:
        AIFunctionError: The resolved config sets
            ``agent_kwargs["conversation_manager"]`` to a non-``None``
            value (the runtime owns conversation history; a user-supplied
            manager would desynchronize ``agent.messages`` from the
            event log).

    Implements:
        Thread.
    """

    def __init__(
            self,
            template: "AIFunction[P, T]",
            config: ThreadConfig,
    ) -> None: ...

    # ── Thread ──

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``AIFunction``."""
        ...

    async def notify(self, text: str) -> None:
        """Buffer ``text`` for observation at the next model-call boundary.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - ``text`` is appended to the thread-local inject buffer.
            - The next :meth:`execute` cycle observes ``text`` on its first
              model-call boundary via the event-bridge hook.
        """
        ...

    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        """Render a prompt and drive the executor for one cycle.

        Builds the prompt via ``self._generate_prompt`` and appends it to
        the thread's inject buffer (after any messages already pending from
        :meth:`notify`); the event-bridge hook atomically emits one
        ``MESSAGE_USER`` event per buffer entry and injects the matching
        user turn into the live Strands agent at the first
        ``BeforeModelCallEvent`` boundary.

        Args:
            ctx: Freshly built per-cycle context.
            args: Positional arguments forwarded to ``template.prompt_fn``.
            kwargs: Keyword arguments forwarded to ``template.prompt_fn``.

        Returns:
            The typed cycle result.

        Requires:
            ``ctx`` is a fresh context built by the runtime for this cycle.

        Emits:
            MESSAGE_USER — one per drained buffer entry, via the
            event-bridge hook.

        Strategy:
            1. Call ``self._generate_prompt`` to produce the prompt string.
            2. Append the prompt to ``self._inject_buffer`` so the
               event-bridge hook emits it (and injects it into the live
               agent) atomically at the first ``BeforeModelCallEvent``,
               after any pending inject messages.
            3. Call ``self._run_cycle(ctx)`` and return its result.
        """
        ...

    # ── Internal pipeline steps ──

    async def _run_cycle(self, ctx: ThreadContext) -> T:
        """Run the shared agent execution loop for one cycle.

        Args:
            ctx: Freshly built per-cycle context.

        Returns:
            The typed result produced by the agent.

        Strategy:
            1. Call ``self._config.config_hook(ctx)`` if set and merge the
               returned ``ThreadKwargs`` into a cycle-local config
               (``config_hook`` key itself ignored).
            2. Call ``reconstruct_messages(await
               ctx.coordinator.get_events(ctx.thread_id))`` to obtain the
               up-to-date message history.
            3. Call ``self._build_agent`` to create a Strands ``Agent``.
            4. ``await agent.invoke_async(messages=messages)``.
            5. On interrupts, ``await ctx.on_interrupt(batch)`` and
               resume with decisions.
            6. Call ``self._extract_result`` to extract the typed result.
            7. Emit ``TOKEN_USAGE`` via ``ctx.on_event``.
            8. Run post-conditions via ``self._validate_result``; on
               failure, put the error text on ``ctx.message_queue`` (the
               hook emits ``MESSAGE_USER`` on the next drain) and
               retry from step 2 up to ``cycle_config.max_attempts``
               times.
            9. Return the typed and validated result.
        """
        ...

    def _build_agent(
            self,
            messages: Messages,
            cycle_config: ThreadConfig,
            extra_hooks: list[HookProvider] | None,
            callback_handler: Callable[..., None] | None = None,
    ) -> Agent:
        """Assemble a configured Strands ``Agent`` for one cycle.

        Args:
            messages: Current conversation history passed to the agent.
            cycle_config: The merged per-cycle config.
            extra_hooks: Extra Strands hooks wired into the agent.
            callback_handler: Strands ``callback_handler`` for per-chunk
                streaming callbacks.

        Returns:
            A freshly constructed Strands ``Agent``.

        Strategy:
            1. If ``cycle_config.structured_output`` is ``False``, check
               that ``output_type`` is ``str`` or raise.
            2. If ``structured_output`` is ``True`` and ``output_type`` is
               not a pydantic model, create a pydantic wrapper
               ``class FinalAnswer: answer: output_type``.
            3. Build a Strands agent with the extra hooks wired in (these
               include the event-bridge hook that drains
               ``message_queue``, checks ``cancel_signal``, awaits
               ``pause_signal``, and emits assistant/tool telemetry
               events); if ``structured_output`` is ``True``, set
               ``structured_output_model`` to ``output_type`` or the
               wrapper; pass any field from ``cycle_config`` that applies
               to ``strands.Agent.__init__``.
        """
        ...

    async def _generate_prompt(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Render the prompt string from template arguments.

        Args:
            args: Positional arguments forwarded from ``execute``.
            kwargs: Keyword arguments forwarded from ``execute``.

        Returns:
            The rendered prompt string.

        Raises:
            AIFunctionError: ``prompt_fn`` returned ``None`` and the
                function has no docstring.

        Strategy:
            1. Call ``self._template.prompt_fn(*args, **kwargs)``; await it if
               it is a coroutine (``async def`` prompt bodies are supported).
            2. If ``prompt_fn`` returns ``None``, interpret
               ``prompt_fn.__doc__`` as a ``tstr`` template and
               interpolate it using the function arguments and their
               enclosing globals as context.
            3. If there is no docstring either, raise ``AIFunctionError``.
        """
        ...

    def _extract_result(self, response: AgentResult, state: dict[str, object]) -> T:
        """Extract the typed result from a Strands ``AgentResult``.

        Args:
            response: The Strands agent result.
            state: Per-cycle execution state.

        Returns:
            The typed result as declared by ``output_type``.
        """
        ...

    async def _validate_result(
            self,
            result: T,
            bound_args: dict[str, object],
            post_conditions: tuple[PostCondition, ...],
            function_name: str,
    ) -> list[str]:
        """Evaluate every post-condition against ``result`` in parallel.

        If a condition raises an exception, it is considered failed and
        the exception text is used as the error message.

        Args:
            result: The candidate typed result.
            bound_args: Bound arguments for condition callables that
                accept them.
            post_conditions: Ordered tuple of validators from
                ``ThreadConfig``.
            function_name: Name of the owning function (for error
                attribution).

        Returns:
            A list of error messages from failed post-conditions.
        """
        ...

    async def fork(self) -> "AIFunction[P, T]":
        """Return the owning template as a resumption ``Spawnable``.

        ``AIThread`` carries no per-instance state beyond the event log
        (which the runtime seeds separately via
        :meth:`Coordinator.copy_events`), so the original template is
        itself a valid resumption spawnable.

        Note:
            The fork intentionally does NOT inherit the current thread's
            inject buffer. Pending ``notify`` entries that have
            not yet been observed by a cycle are dropped from the forked
            thread's perspective — the fork starts with an empty buffer.

        Returns:
            ``self.template`` — the ``AIFunction`` this thread was
            spawned from.
        """
        ...

    async def teardown(self) -> None:
        """Release per-instance state on termination.

        Drops the inject buffer. ``AIThread`` owns no other external
        resources.
        """
        ...

    # ── Introspection ──

    @property
    def template(self) -> "AIFunction[P, T]":
        """The template this thread was created from."""
        ...

    @property
    def config(self) -> ThreadConfig:
        """Resolved config.

        ``template.config`` merged with ``to_thread`` overrides.
        """
        ...

