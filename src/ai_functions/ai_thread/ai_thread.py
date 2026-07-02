"""AIThread — live, per-spawn instance backing an ``AIFunction``."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import typing
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final

import tstr
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.types.content import Messages
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)

from .._type import is_json_serializable_type
from ..protocols import Thread
from ..types import (
    ContextSummarizedEvent,
    Event,
    MessageAssistantCompleteEvent,
    MessageAssistantStartEvent,
    MessageAssistantThinkingEvent,
    MessageAssistantTokenEvent,
    MessageId,
    MessageUserEvent,
    ThreadContext,
    TokenUsage,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .config import CodeExecutionMode, ThreadConfig, ThreadKwargs
from .errors import AIFunctionError
from .postcondition import PostCondition
from .reconstruction import reconstruct_messages
from .summarization import (
    DefaultSummarizationStrategy,
    SummarizationFailedError,
    SummarizationStrategy,
    _estimate_message_tokens,
)

if TYPE_CHECKING:
    from .ai_function import AIFunction


# Hard cap on reactive summarization attempts per cycle. Bounds the worst
# case when every strategy configuration still overflows.
_MAX_SUMMARIZATION_ATTEMPTS: int = 3


@dataclass(frozen=True)
class OutputSpec[T]:
    """Describes the AI output type and how to present it to the model."""

    output_type: type[T]
    """The user-specified output type (e.g. ``str``, ``float``, a pydantic model)."""

    is_pydantic: bool
    """True iff ``output_type`` is a ``pydantic.BaseModel`` subclass."""

    is_structured: bool
    """True iff the output can be expressed as a JSON schema for the model."""

    structured_output_model: type[BaseModel] | None
    """The answer model (a ``FinalAnswer`` wrapper, or the pydantic type itself).

    Always populated when the output is wrapped or already pydantic — including
    non-JSON-serializable returns. ``None`` only for plain-str output. Whether
    it is handed to Strands as a structured-output model is gated by
    ``is_structured`` at the call site (``_build_agent``); the code executor
    uses it directly for the ``final_answer`` signature regardless."""

    is_wrapped: bool
    """True iff ``structured_output_model`` wraps ``output_type`` in a ``FinalAnswer``."""

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
            - ``result.is_wrapped`` is true iff ``structured_output_model``
              wraps ``output_type`` in a ``FinalAnswer`` model.
            - ``result.is_structured`` is false for a non-JSON-serializable
              return type (e.g. ``sympy.Expr``); such a type can only produce
              its answer through the code executor's ``final_answer`` path, so
              ``structured_output_model`` is ``None`` (never handed to Strands
              for JSON-schema structured output), while ``is_wrapped`` still
              holds so the executor has a typed ``final_answer`` signature.
        """
        if not is_structured:
            assert output_type is str, "structured_output=False is only supported for str output"
            return cls(
                output_type=output_type,
                is_pydantic=False,
                is_structured=False,
                structured_output_model=None,
                is_wrapped=False,
            )

        if isinstance(output_type, type) and issubclass(output_type, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            return cls(
                output_type=output_type,
                is_pydantic=True,
                is_structured=True,
                structured_output_model=output_type,
                is_wrapped=False,
            )

        # Non-pydantic, non-str type: wrap in a FinalAnswer model. Allow
        # arbitrary (non-pydantic) types so return annotations like
        # ``sympy.Expr`` can be the answer field; the wrapper is used both for
        # structured output AND to render the executor's final_answer signature.
        wrapped = create_model(
            "FinalAnswer",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            answer=(output_type, Field(...)),
        )
        # Structured output is only usable when the type is JSON-serializable
        # (Strands must build a JSON schema for it). For a non-serializable type
        # the answer comes solely from the code executor's final_answer path, so
        # ``is_structured`` is False — but the wrapper model is kept regardless
        # (the executor needs it for the final_answer signature); ``_build_agent``
        # gates whether it is handed to Strands on ``is_structured``.
        json_serializable = is_json_serializable_type(output_type)
        return cls(
            output_type=output_type,
            is_pydantic=False,
            is_structured=json_serializable,
            structured_output_model=wrapped,
            is_wrapped=True,
        )


def _new_message_id() -> MessageId:
    return MessageId(f"msg-{uuid.uuid4().hex}")


class _EventBridgeHook(HookProvider):
    """Bridge Strands model/tool hooks and streaming callbacks to AI Functions events.

    Responsibilities:
      1. Work-boundary management (before every model call): drain the
         owning thread's inject buffer into ``MESSAGE_USER`` events,
         check ``cancel_signal``, await ``pause_signal``.
      2. Assistant-turn span emission: ``MESSAGE_ASSISTANT_START`` before a
         model call, ``MESSAGE_ASSISTANT_COMPLETE`` after it — both carry a
         fresh ``message_id`` that ties in any tool events emitted during
         that model turn.
      3. Tool-call telemetry: ``TOOL_CALL`` before each tool invocation,
         ``TOOL_RESULT`` / ``TOOL_ERROR`` after.
      4. Streaming deltas: the ``callback_handler`` passed to the ``Agent``
         fans ``data`` chunks into ``MESSAGE_ASSISTANT_TOKEN`` and
         ``reasoningText`` chunks into ``MESSAGE_ASSISTANT_THINKING``.
    """

    def __init__(self, ctx: ThreadContext, inject_buffer: list[str]) -> None:
        self._ctx = ctx
        self._inject_buffer = inject_buffer
        self._current_message_id: MessageId | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        """Register every Strands hook this bridge cares about."""
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    # ── Streaming callback ──

    def stream_callback(self, **kwargs: object) -> None:
        """Fan one Strands streaming chunk into assistant token/thinking events.

        Passed to ``Agent(callback_handler=self.stream_callback)``. Strands
        invokes this once per streaming chunk with a kwargs payload whose keys
        vary by provider. We care about ``data`` (text delta) and
        ``reasoningText`` (extended-thinking delta); everything else is ignored.
        """
        ctx = self._ctx
        message_id = self._current_message_id
        if message_id is None:
            return
        complete = bool(kwargs.get("complete", False))
        data = kwargs.get("data")
        if isinstance(data, str) and data:
            ctx.on_event(
                MessageAssistantTokenEvent(
                    message_id=message_id,
                    text=data,
                    complete=complete,
                )
            )
        reasoning = kwargs.get("reasoningText")
        if isinstance(reasoning, str) and reasoning:
            ctx.on_event(
                MessageAssistantThinkingEvent(
                    message_id=message_id,
                    text=reasoning,
                    complete=complete,
                )
            )

    # ── Model hooks ──

    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        ctx = self._ctx
        # Drain the owning thread's inject buffer. Each drained entry is
        # simultaneously (1) emitted as a MESSAGE_USER event for the durable
        # log and (2) appended to the live agent's message list so the next
        # model iteration observes it. The two actions must be paired: any
        # divergence between the event log order and agent-observation order
        # would invalidate the agent's cached prefix on the next rehydration
        # (see I7).
        #
        # We append one Message per drained text. Strands does NOT coalesce
        # consecutive same-role messages (see ``_normalize_messages`` in
        # ``strands.event_loop.streaming`` — it only touches assistant-side
        # blank-text handling). ``reconstruct_messages`` must therefore also
        # emit one user Message per ``MESSAGE_USER`` event to preserve I7's
        # cache-prefix stability property.
        from strands.types.content import ContentBlock, Message

        while self._inject_buffer:
            text = self._inject_buffer.pop(0)
            ctx.on_event(MessageUserEvent(text=text))
            event.agent.messages.append(Message(role="user", content=[ContentBlock(text=text)]))
        # Cooperative cancel check
        if ctx.cancel_signal.is_set():
            raise asyncio.CancelledError
        # Await rate-limit / manual pause
        await ctx.coordinator.wait_until_unpaused(ctx.thread_id)
        # Open a fresh assistant-turn span
        self._current_message_id = _new_message_id()
        ctx.on_event(MessageAssistantStartEvent(message_id=self._current_message_id))

    async def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        ctx = self._ctx
        message_id = self._current_message_id
        if message_id is None:
            return
        stop_response = event.stop_response
        content: list[object] = []
        if stop_response is not None:
            # stop_response.message is a Strands Message (TypedDict).
            message = cast("dict[str, object]", stop_response.message)
            raw_content = message.get("content", [])
            if isinstance(raw_content, list):
                content = list(cast("list[object]", raw_content))
        ctx.on_event(
            MessageAssistantCompleteEvent(
                message_id=message_id,
                content=cast("list[Any]", content),  # pyright: ignore[reportExplicitAny]
            )
        )
        self._current_message_id = None

    # ── Tool hooks ──

    async def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        ctx = self._ctx
        tool_use = cast("dict[str, object]", event.tool_use)
        tool_use_id = str(tool_use.get("toolUseId", ""))
        tool_name = str(tool_use.get("name", ""))
        arguments_raw = tool_use.get("input", {})
        arguments: dict[str, object] = (
            dict(cast("dict[str, object]", arguments_raw))
            if isinstance(arguments_raw, dict)
            else {"input": arguments_raw}
        )
        ctx.on_event(
            ToolCallEvent(
                message_id=self._current_message_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        ctx = self._ctx
        # ``event.result`` is always a Strands ``ToolResult`` dict
        # (``{"toolUseId", "status", "content"}``), even on exception — the
        # tool executor wraps exceptions into
        # ``{"status": "error", "content": [{"text": "Error: ..."}]}`` before
        # the hook fires. We split the dict across top-level event fields
        # (typed attribute access for consumers) and repack it at
        # reconstruction time so the on-wire ``toolResult`` block remains
        # byte-identical to what Strands appends to ``agent.messages``.
        tool_result = cast("dict[str, object]", event.result)
        ctx.on_event(
            ToolResultEvent(
                message_id=self._current_message_id,
                tool_use_id=str(tool_result.get("toolUseId", "")),
                status=cast("Any", tool_result.get("status", "success")),  # pyright: ignore[reportExplicitAny]
                content=cast("Any", tool_result.get("content", [])),  # pyright: ignore[reportExplicitAny]
            )
        )


@final
class AIThread[**P, T](Thread):  # type: ignore[type-arg]
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

    __slots__ = (
        "_template",
        "_config",
        "_output_spec",
        "_summarization_strategy",
        "_inject_buffer",
        "_bound_args",
    )

    def __init__(
        self,
        template: AIFunction[P, T],
        config: ThreadConfig,
    ) -> None:
        # Reject user-supplied conversation_manager (I7/I9): the runtime
        # installs NullConversationManager on every Agent it builds. Any
        # non-None value is a misconfiguration.
        user_manager = config.agent_kwargs.get("conversation_manager")
        if user_manager is not None:
            raise AIFunctionError(
                "agent_kwargs['conversation_manager'] must be unset: the "
                "runtime manages conversation history through the event log "
                "and installs NullConversationManager on every Agent. A "
                "user-supplied conversation manager would mutate "
                "agent.messages behind the runtime's back and desynchronize it "
                "from the event log.",
                function_name=getattr(template, "name", ""),
            )

        self._template = template
        self._config = config
        self._output_spec = OutputSpec[T].from_type(
            output_type=template.output_type,
            is_structured=config.structured_output,
        )

        # A non-JSON-serializable return type (structured off, but wrapped) can
        # only produce its answer via the code executor's final_answer. Without
        # code execution there is no path to a result — reject early.
        if (
            not self._output_spec.is_structured
            and self._output_spec.is_wrapped
            and config.code_execution_mode != CodeExecutionMode.LOCAL
        ):
            raise AIFunctionError(
                f"Return type {template.output_type!r} is not JSON-serializable, so it can only be "
                "produced via code execution; set code_execution_mode='local'.",
                function_name=getattr(template, "name", ""),
            )

        self._summarization_strategy: SummarizationStrategy = (
            config.summarization_strategy
            if config.summarization_strategy is not None
            else DefaultSummarizationStrategy()
        )

        # Thread-owned buffer of pending user turns. execute() appends its
        # generated prompt here; notify() appends arbitrary text
        # (e.g. watcher nudges). The event-bridge hook drains the buffer
        # at the next BeforeModelCallEvent and emits one MESSAGE_USER per
        # entry — atomic pairing preserves I7.
        self._inject_buffer: list[str] = []

        # Bound arguments of the current cycle, captured in execute(). Seeded
        # into the python_executor's namespace when code_execution_mode=LOCAL,
        # so recalled Procedural code and other inputs are available to run.
        self._bound_args: dict[str, object] = {}

    # ── Thread ──

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``AIFunction``."""
        return self._template.name

    async def notify(self, text: str) -> None:
        """Buffer ``text`` for observation at the next model-call boundary.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - ``text`` is appended to the thread-local inject buffer.
            - The next :meth:`execute` cycle observes ``text`` on its first
              model-call boundary via the event-bridge hook.
        """
        self._inject_buffer.append(text)

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
        self._bound_args = self._bind_args(*args, **kwargs)
        prompt = await self._generate_prompt(*args, **kwargs)
        self._inject_buffer.append(prompt)
        return await self._run_cycle(ctx)

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, object]:
        """Bind call args to ``prompt_fn``'s signature, returning a name→value dict.

        Used to seed the optional ``python_executor`` namespace. If strict
        binding fails (e.g. an unexpected positional count), fall back to a
        best-effort mapping that still names positional args by parameter
        position — so positionally-passed inputs are not silently dropped.
        """
        try:
            sig = inspect.signature(self._template.prompt_fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except TypeError:
            try:
                sig = inspect.signature(self._template.prompt_fn)
                names = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                result: dict[str, object] = dict(kwargs)
                for name, value in zip(names, args, strict=False):
                    result.setdefault(name, value)
                return result
            except (TypeError, ValueError):
                return dict(kwargs)

    def _procedural_param_names(self) -> set[str]:
        """Return the names of ``prompt_fn`` params annotated as ``Procedural``.

        Detected via ``ProceduralMarker`` in the parameter's ``Annotated``
        metadata. These hold Python source that the python_executor should
        *define* (run at setup) rather than inject as a plain string variable.
        """
        from ..memory.procedural import ProceduralMarker

        try:
            hints = typing.get_type_hints(self._template.prompt_fn, include_extras=True)
        except Exception:  # noqa: BLE001 — annotations may reference missing names
            return set()
        names: set[str] = set()
        for name, hint in hints.items():
            metadata = getattr(hint, "__metadata__", ())
            if any(isinstance(m, ProceduralMarker) for m in metadata):
                names.add(name)
        return names

    # ── Internal pipeline steps ──

    async def _run_cycle(self, ctx: ThreadContext) -> T:
        """Run the shared agent execution loop for one cycle.

        ``ResultEvent`` is emitted by the runtime dispatcher around the
        cycle, not by the thread itself (see ``ResultEvent`` invariant).

        Args:
            ctx: Freshly built per-cycle context.

        Returns:
            The typed result produced by the agent.

        Strategy:
            1. Call ``self._config.config_hook(ctx)`` if set and merge
               the returned ``ThreadKwargs`` into a cycle-local config
               (all fields replaced, ``config_hook`` key itself
               ignored).
            2. Call ``reconstruct_messages(await
               ctx.coordinator.get_events(ctx.thread_id))`` to obtain
               the up-to-date message history.
            3. Call ``self._build_agent(messages, cycle_config, ...)``
               to create a Strands ``Agent``.
            4. ``await agent.invoke_async(messages=messages)``.
            5. On interrupts, ``await ctx.on_interrupt(batch)`` and
               resume with decisions.
            6. Call ``self._extract_result`` to extract the typed
               result.
            7. Emit ``TOKEN_USAGE`` via ``ctx.on_event``.
            8. Run post-conditions via ``self._validate_result``; on
               failure, emit the errors as a ``MESSAGE_USER`` turn and
               retry from step 2 up to ``cycle_config.max_attempts``
               times.
            9. Return the typed and validated result.
        """
        # Build cycle-local config by applying config_hook if set
        cycle_config = self._config
        if self._config.config_hook is not None:
            patch: ThreadKwargs = self._config.config_hook(ctx)
            patch_dict = dict(patch)
            patch_dict.pop("config_hook", None)
            cycle_config = dataclasses.replace(self._config, **patch_dict)  # type: ignore[arg-type]

        # Inject the default runtime-facing tools (list_threads, send_message).
        # Bound to this cycle's ctx so the LLM can discover peer threads and
        # nudge them via notify. Appended after user tools so a
        # user-supplied tool with the same name wins on resolution. Skipped
        # entirely when ``coordinator_tools_enabled`` is False (single-purpose
        # agents that should not talk to peers).
        if cycle_config.coordinator_tools_enabled:
            from .tools import coordinator_tools

            cycle_config = dataclasses.replace(
                cycle_config,
                tools=(*cycle_config.tools, *coordinator_tools(ctx)),
            )

        post_conditions = cycle_config.post_conditions
        function_name = self._template.name
        # The cycle's bound arguments (captured in execute) are offered to
        # post-conditions: a validator may declare any of the function's
        # parameters (e.g. ``max_length``) and receive its value. Without this,
        # such a post-condition would be called missing that argument and fail
        # every attempt.
        bound_args: dict[str, object] = dict(self._bound_args)

        for attempt in range(cycle_config.max_attempts):
            # Build a fresh python_executor per attempt when code execution is
            # enabled. The smolagents sandbox persists its namespace across
            # calls, so a single executor reused across retries would leak the
            # failed attempt's ad-hoc variables into the next attempt
            # (non-deterministic retries / stale-state masking). A fresh
            # executor re-defines the Procedural helpers from initial_code but
            # starts with no leaked state.
            attempt_config = self._with_python_executor(cycle_config)
            response = await self._invoke_with_summarization(ctx, attempt_config)

            # Handle interrupts if the agent surface them
            while response.interrupts:
                decisions = await ctx.on_interrupt(list(response.interrupts))
                response = await self._invoke_with_summarization(ctx, attempt_config, resume_messages=decisions)

            state: dict[str, object] = response.state or {}
            result = self._extract_result(response, state)

            # Emit token usage
            usage = response.metrics.accumulated_usage
            ctx.on_event(
                TokenUsageEvent(
                    token_usage=TokenUsage(
                        input_tokens=usage.get("inputTokens", 0),
                        output_tokens=usage.get("outputTokens", 0),
                        cache_read_tokens=usage.get("cacheReadInputTokens", 0),
                        cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
                    ),
                )
            )

            if not post_conditions:
                return result

            errors = await self._validate_result(result, bound_args, post_conditions, function_name)
            if not errors:
                return result

            # Enqueue the validation errors as a user turn so the next cycle's
            # hook emits the MESSAGE_USER event. Sole-emitter discipline (I7)
            # keeps the event log and agent-observation order aligned even
            # across retries.
            failures = "\n".join(f"- {e}" for e in errors)
            error_text = (
                f"[{function_name}] Post-condition failures"
                f" (attempt {attempt + 1}/{cycle_config.max_attempts}):\n{failures}"
            )
            self._inject_buffer.append(error_text)

        # Exhausted all attempts — raise with the last set of errors
        raise AIFunctionError(
            f"Post-conditions not satisfied after {cycle_config.max_attempts} attempt(s)",
            function_name=function_name,
        )

    def _with_python_executor(self, cycle_config: ThreadConfig) -> ThreadConfig:
        """Return ``cycle_config`` with a fresh ``python_executor`` tool appended.

        No-op unless ``code_execution_mode == LOCAL``. Called once per attempt so
        each retry gets a clean sandbox: ``Procedural`` parameter code is
        re-defined in the namespace (from ``initial_code``), but no ad-hoc state
        from a prior attempt leaks in.

        Raises:
            AIFunctionError: ``code_execution_mode=LOCAL`` but the output is not
                structured (the ``final_answer`` callback needs a typed model).
        """
        if cycle_config.code_execution_mode != CodeExecutionMode.LOCAL:
            return cycle_config

        from ..tools.local_python_executor import LocalPythonExecutorTool

        # The executor needs a typed model for the final_answer signature. It is
        # present for any pydantic/wrapped return (including non-serializable
        # ones); only plain-str output has none, and code execution cannot
        # produce a bare str answer via final_answer.
        output_model = self._output_spec.structured_output_model
        if output_model is None:
            raise AIFunctionError(
                "code_execution_mode=LOCAL is not supported for a plain str return type "
                "(the python_executor's final_answer needs a typed model).",
                function_name=self._template.name,
            )
        # Split args: Procedural-typed code is DEFINED in the namespace (its
        # functions become callable); everything else is injected as a plain
        # variable. The sandbox forbids exec(), so procedural source must be run
        # at setup rather than handed over as a string.
        procedural_names = self._procedural_param_names()
        initial_code = [str(v) for k, v in self._bound_args.items() if k in procedural_names and isinstance(v, str)]
        initial_state = {k: v for k, v in self._bound_args.items() if k not in procedural_names}
        executor = LocalPythonExecutorTool(
            output_type=output_model,
            initial_state=initial_state,
            initial_code=initial_code,
            additional_authorized_imports=list(cycle_config.code_executor_additional_imports),
            executor_kwargs=dict(cycle_config.code_executor_kwargs),
        )
        return dataclasses.replace(
            cycle_config,
            tools=(*cycle_config.tools, executor.python_executor),
        )

    async def _invoke_with_summarization(
        self,
        ctx: ThreadContext,
        cycle_config: ThreadConfig,
        resume_messages: object = None,
    ) -> AgentResult:
        """Build a Strands Agent from the current event log and invoke it.

        Wraps ``agent.invoke_async`` with a bounded summarization loop. When
        ``cycle_config.summarization_threshold`` is set, the reconstructed history is
        compacted *proactively* at cycle entry if its estimated token count
        exceeds the threshold — before the model call, avoiding the overflow
        error entirely. Reactively, on ``ContextWindowOverflowException`` we run
        the same summarization pipeline, emit a ``ContextSummarizedEvent``,
        rebuild a fresh agent from the updated log, and retry.
        ``MaxTokensReachedException`` is a hard failure — summarization does not
        help if the model cannot produce its current output within the
        max-tokens budget.

        Args:
            ctx: Per-cycle runtime context.
            cycle_config: The already-resolved cycle-local config.
            resume_messages: If not ``None``, passed as ``messages=`` to
                ``invoke_async`` (interrupt-resume path).

        Returns:
            The completed ``AgentResult``.

        Raises:
            SummarizationFailedError: Every summarization attempt in
                this cycle failed to bring the history under the
                context window.
            MaxTokensReachedException: Propagated unchanged;
                unrecoverable.
        """
        summarizations_so_far = 0
        while True:
            events = await ctx.coordinator.get_events(ctx.thread_id)
            messages: Messages = reconstruct_messages(events)

            # Proactive summarization: compact before the model call when the
            # estimated history exceeds the configured threshold. Bounded by the
            # same attempt cap as the reactive path so a strategy that fails to
            # shrink the history cannot loop forever.
            threshold = cycle_config.summarization_threshold
            if (
                threshold is not None
                and summarizations_so_far < _MAX_SUMMARIZATION_ATTEMPTS
                and sum(_estimate_message_tokens(m) for m in messages) > threshold
            ):
                summarizations_so_far += 1
                await self._summarize_and_emit(ctx, cycle_config, events)
                continue

            bridge = _EventBridgeHook(ctx=ctx, inject_buffer=self._inject_buffer)
            agent = self._build_agent(messages, cycle_config, [bridge], bridge.stream_callback)

            try:
                if resume_messages is not None:
                    return await agent.invoke_async(messages=resume_messages)  # type: ignore[arg-type]
                return await agent.invoke_async()
            except ContextWindowOverflowException as exc:
                if summarizations_so_far >= _MAX_SUMMARIZATION_ATTEMPTS:
                    raise SummarizationFailedError(
                        function_name=self._template.name,
                        reason=(
                            f"Context window still overflowed after "
                            f"{_MAX_SUMMARIZATION_ATTEMPTS} summarization attempts: {exc}"
                        ),
                    ) from exc
                summarizations_so_far += 1
                # Re-fetch events: the failed model call ran the event-bridge
                # hook's pre-call phase, which drained the message queue and
                # emitted MESSAGE_USER events. Those are the events we want
                # to summarize — the pre-drain snapshot would miss them.
                fresh_events = await ctx.coordinator.get_events(ctx.thread_id)
                await self._summarize_and_emit(ctx, cycle_config, fresh_events)
                # The next iteration re-reads events (now with the new
                # ContextSummarizedEvent) and rebuilds messages — this is
                # the cache-invalidation point for I9.
                continue
            except MaxTokensReachedException:
                # Hard failure: the model exhausted its output budget
                # mid-cycle, meaning its own reply could not fit. Propagate
                # unchanged; summarization would not help.
                raise

    async def _summarize_and_emit(
        self,
        ctx: ThreadContext,
        cycle_config: ThreadConfig,
        events: list[Event],
    ) -> None:
        """Run the summarization strategy over ``events`` and emit the boundary event.

        Shared by the proactive (pre-call, threshold-driven) and reactive
        (post-overflow) paths. Wraps strategy failures in
        ``SummarizationFailedError`` and appends a ``ContextSummarizedEvent``
        whose ``new_history`` replaces the summarized prefix on the next
        reconstruction (the I9 cache-invalidation point).
        """
        try:
            new_history = await self._summarization_strategy.summarize(events, ctx, cycle_config)
        except SummarizationFailedError:
            raise
        except Exception as strategy_exc:
            raise SummarizationFailedError(
                function_name=self._template.name,
                reason=f"summarization strategy raised: {strategy_exc!r}",
            ) from strategy_exc
        ctx.on_event(ContextSummarizedEvent(new_history=new_history))

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
            cycle_config: The merged per-cycle config (base config
                patched by ``config_hook``).
            extra_hooks: Extra Strands hooks wired into the agent.
            callback_handler: Strands ``callback_handler`` for per-chunk
                streaming callbacks; passed through unchanged when not
                ``None``.

        Returns:
            A freshly constructed Strands ``Agent``.

        Strategy:
            1. If ``cycle_config.structured_output`` is ``False``,
               check that ``output_type`` is ``str`` or raise an
               exception.
            2. If ``cycle_config.structured_output`` is ``True`` and
               ``output_type`` is not a pydantic model, create a
               pydantic wrapper: ``class FinalAnswer: answer:
               output_type``.
            3. Build a Strands agent with:

               - The extra hooks wired in (these include the
                 event-bridge hook that drains ``message_queue``,
                 checks ``cancel_signal``, awaits ``pause_signal``,
                 and emits assistant/tool telemetry events).
               - If ``structured_output`` is ``True``, set
                 ``structured_output_model`` to ``output_type`` or the
                 pydantic wrapper.
               - Pass any field from ``cycle_config`` that applies to
                 ``strands.Agent.__init__``.
        """
        spec = self._output_spec
        # Structured output may legitimately be off in two cases: a plain str
        # return, or a non-JSON-serializable return (wrapped) that produces its
        # answer via the code executor's final_answer. Anything else is a
        # misconfiguration.
        if not spec.is_structured and spec.output_type is not str and not spec.is_wrapped:
            raise AIFunctionError(
                f"structured_output=False is only supported when output_type is str, got {spec.output_type!r}",
                function_name=self._template.name,
            )

        tools = list(cycle_config.tools)

        hooks: list[HookProvider] = list(extra_hooks) if extra_hooks else []
        if "hooks" in cycle_config.agent_kwargs and cycle_config.agent_kwargs["hooks"]:
            hooks.extend(cycle_config.agent_kwargs["hooks"])

        # Drop ``hooks``, ``conversation_manager``, and ``callback_handler`` from
        # the forwarded kwargs: we merged hooks above, install
        # ``NullConversationManager`` unconditionally (I7/I9; a user manager was
        # rejected in ``__init__``), and pass ``callback_handler`` explicitly
        # below — leaving it in ``agent_kwargs`` would double-pass it and raise
        # ``TypeError: got multiple values for keyword argument``.
        agent_kwargs = {
            k: v
            for k, v in cycle_config.agent_kwargs.items()
            if k not in ("hooks", "conversation_manager", "callback_handler")
        }

        # The runtime's streaming callback wins; otherwise honor a
        # user-supplied ``callback_handler`` from ``agent_kwargs``.
        effective_callback = (
            callback_handler if callback_handler is not None else cycle_config.agent_kwargs.get("callback_handler")
        )

        # Only hand the model to Strands for JSON-schema structured output when
        # the type supports it. A wrapped-but-non-serializable type keeps its
        # model on ``spec`` (for the executor's final_answer) but must NOT be
        # passed here, or Strands would fail generating a JSON schema for it.
        strands_output_model = spec.structured_output_model if spec.is_structured else None

        if effective_callback is not None:
            return Agent(
                model=cycle_config.model,
                messages=list(messages),
                system_prompt=cycle_config.system_prompt,
                tools=tools or None,
                structured_output_model=strands_output_model,
                hooks=hooks,
                callback_handler=effective_callback,
                conversation_manager=NullConversationManager(),
                **agent_kwargs,  # pyright: ignore[reportArgumentType]
            )
        return Agent(
            model=cycle_config.model,
            messages=list(messages),
            system_prompt=cycle_config.system_prompt,
            tools=tools or None,
            structured_output_model=strands_output_model,
            hooks=hooks,
            conversation_manager=NullConversationManager(),
            **agent_kwargs,  # pyright: ignore[reportArgumentType]
        )

    def serialize_result(self, result: T) -> str:
        """Encode ``result`` as a string for storage in a ``ResultEvent``.

        Args:
            result: The typed cycle result to serialize.

        Returns:
            A string representation of ``result``.

        Ensures:
            - For a JSON-serializable ``output_type``, the payload round-trips:
              ``deserialize_result(serialize_result(result))`` equals ``result``.
            - For a non-JSON-serializable ``output_type`` (e.g. an arbitrary
              object like ``sympy.Expr``, only producible via code execution),
              the payload is a best-effort ``str(result)`` for observability and
              does NOT round-trip — such a result cannot be reconstructed from
              the event log. The in-process return value is unaffected; only the
              ``ResultEvent`` record degrades. Serialization never raises, so a
              cycle producing an exotic object is not killed by the event log.

        Concurrency:
            Must be synchronous and side-effect-free.
        """
        try:
            adapter: TypeAdapter[T] = TypeAdapter(self._output_spec.output_type)
            return adapter.dump_json(result).decode("utf-8")
        except Exception:  # noqa: BLE001 — non-JSON-serializable result: best-effort record
            return str(result)

    def deserialize_result(self, payload: str) -> T:
        """Recover a result from the string stored in a ``ResultEvent``.

        Args:
            payload: Value previously produced by ``serialize_result``.

        Returns:
            The deserialized result of type ``T``.

        Raises:
            AIFunctionError: The payload is malformed or cannot be decoded as
                ``T``. This includes non-JSON-serializable results, whose
                ``ResultEvent`` payload is a best-effort ``str`` (see
                :meth:`serialize_result`) and cannot be reconstructed — such
                results are only available via the in-process return value.
        """
        try:
            adapter: TypeAdapter[T] = TypeAdapter(self._output_spec.output_type)
            return adapter.validate_json(payload)
        except Exception as exc:
            raise AIFunctionError(
                f"Failed to deserialize result payload: {exc}",
                function_name=self._template.name,
            ) from exc

    async def fork(self) -> AIFunction[P, T]:
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
        return self._template

    async def teardown(self) -> None:
        """Release per-instance state on termination.

        Drops the inject buffer. ``AIThread`` owns no other external
        resources.
        """
        self._inject_buffer.clear()
        return None

    async def _generate_prompt(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Render the prompt string from template arguments.

        Args:
            args: Positional arguments forwarded from ``execute``.
            kwargs: Keyword arguments forwarded from ``execute``.

        Returns:
            The rendered prompt string.

        Strategy:
            1. Call ``self._template.prompt_fn(*args, **kwargs)``; if it is a
               coroutine function, await it (``async def`` prompt bodies are
               supported).
            2. If ``prompt_fn`` returns ``None``, interpret
               ``prompt_fn.__doc__`` as a ``tstr`` template and
               interpolate it using the function arguments and their
               enclosing globals as context.
            3. If there is no docstring either, raise
               ``AIFunctionError``.
        """
        result = self._template.prompt_fn(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = await result
        if result is not None:
            return result

        doc = self._template.prompt_fn.__doc__
        if not doc:
            raise AIFunctionError(
                "prompt_fn returned None and has no docstring to use as a template",
                function_name=self._template.name,
            )

        # Build context from bound arguments
        sig = inspect.signature(self._template.prompt_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        context: dict[str, object] = dict(bound.arguments)

        if hasattr(self._template.prompt_fn, "__globals__"):
            fn_globals = self._template.prompt_fn.__globals__
        else:
            fn_globals = dict[str, Any]()  # pyright: ignore[reportExplicitAny]
        # use_eval=True (matching both ancestor implementations) so docstring
        # templates may use attribute access and expressions — e.g.
        # ``{response.summary}`` or ``{bullet_points(items)}`` — not just bare
        # variable names. The template is the function's own docstring (trusted
        # author-supplied text), interpolated with the call's bound arguments.
        template = tstr.generate_template(doc, context, globals=fn_globals, use_eval=True)  # pyright: ignore[reportUnknownMemberType]
        return tstr.render(template)

    def _extract_result(self, response: AgentResult, state: dict[str, object]) -> T:
        """Extract the typed result from a Strands ``AgentResult``.

        Args:
            response: The Strands agent result.
            state: Per-cycle execution state.

        Returns:
            The typed result as declared by ``output_type``.
        """
        spec = self._output_spec

        # Plain str output (structured_output=False, not wrapped): the answer is
        # the assistant's text. A non-structured *wrapped* type (e.g. an
        # arbitrary sympy.Expr return) is NOT this case — its answer comes from
        # the code executor's final_answer, handled below.
        if not spec.is_structured and not spec.is_wrapped:
            return cast(T, str(response))

        # The answer may come from structured output, or — when code execution
        # is enabled — from a final_answer(...) call inside the python_executor,
        # surfaced on AgentResult.state["python_executor_result"]. When code
        # execution is enabled, prefer the executor result: it explicitly halted
        # the loop with the agent's intended answer, so a stray structured tool
        # call must not shadow it. Otherwise use structured output.
        executor_result = cast("BaseModel | None", state.get("python_executor_result"))
        if self._config.code_execution_mode == CodeExecutionMode.LOCAL and executor_result is not None:
            structured: BaseModel | None = executor_result
        else:
            structured = response.structured_output
            if structured is None:
                structured = executor_result
        if structured is None:
            raise AIFunctionError(
                "Agent produced neither a structured output nor a python_executor final_answer result.",
                function_name=self._template.name,
            )
        if spec.is_wrapped:
            return structured.answer  # pyright: ignore[reportAttributeAccessIssue]
        return cast(T, structured)

    async def _validate_result(
        self,
        result: T,
        bound_args: dict[str, object],
        post_conditions: tuple[PostCondition, ...],
        function_name: str,
    ) -> list[str]:
        """Evaluate every post-condition against ``result`` in parallel.

        Returns the list of all error messages from failed conditions.
        If a condition raises an exception, it is considered failed and
        the exception text is used as error message.

        Args:
            result: The candidate typed result.
            bound_args: Bound arguments for condition callables that
                accept them.
            post_conditions: Ordered tuple of validators from
                ``ThreadConfig``.
            function_name: Name of the owning function (for error
                attribution).

        Returns:
            A list of error messages from failed post conditions.
        """

        async def _run_one(cond: PostCondition) -> str | None:
            # Pass only the keyword args whose names match the condition's signature
            sig = inspect.signature(cond)
            extra = {k: v for k, v in bound_args.items() if k in sig.parameters}
            try:
                cond_result = cond(result, **extra)
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
            except Exception as exc:
                return str(exc)
            if cond_result is None or cond_result.passed:
                return None
            return cond_result.message

        outcomes = await asyncio.gather(*(_run_one(c) for c in post_conditions))
        return [msg for msg in outcomes if msg is not None]

    # ── Introspection ──

    @property
    def template(self) -> AIFunction[P, T]:
        """The template this thread was created from."""
        return self._template

    @property
    def config(self) -> ThreadConfig:
        """Resolved config.

        ``template.config`` merged with ``to_thread`` overrides.
        """
        return self._config
