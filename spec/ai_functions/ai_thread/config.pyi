"""Per-thread execution configuration and its matching kwargs ``TypedDict`` s.

Invariants:
    ``ThreadConfig`` fields exactly match ``ThreadKwargs`` annotations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypedDict, Unpack

from strands.types.tools import AgentTool
JSONSchema = dict[str, Any]  # pyright: ignore[reportExplicitAny]

from .postcondition import PostCondition

if TYPE_CHECKING:
    from ..types import ThreadContext
    from .summarization import SummarizationStrategy
    from strands.agent import ConversationManager
    from strands.agent.state import AgentState
    from strands.hooks import HookProvider
    from strands.models import Model
    from strands.session import SessionManager
    from strands.tools import ToolProvider
    from strands.tools.executors._executor import ToolExecutor
    from strands.types.content import Messages
    from strands.types.traces import AttributeValue


class CodeExecutionMode(enum.StrEnum):
    """Code-execution mode for Python tool calls."""

    LOCAL = "local"
    """Execute code in the current process (default)."""

    DISABLED = "disabled"
    """No code execution — rely solely on structured output."""


class AgentKwargs(TypedDict, total=False):
    """Kwargs pass-through for the ``strands.Agent`` constructor.

    ``conversation_manager`` is accepted in this mapping but must always
    be left unset (or explicitly ``None``) by users. The runtime manages
    conversation history through the event log and installs
    ``NullConversationManager`` on every ``Agent`` it builds; a
    user-supplied manager would mutate ``agent.messages`` behind the
    runtime's back and desynchronize it from the event log, breaking I7
    and I9.
    Setting this key to any non-``None`` value is rejected at config
    resolution with ``AIFunctionError``.
    """

    messages: "Messages | None"
    callback_handler: Callable[..., Any] | None  # pyright: ignore[reportExplicitAny]
    conversation_manager: "ConversationManager | None"
    record_direct_tool_call: bool
    load_tools_from_directory: bool
    trace_attributes: "Mapping[str, AttributeValue] | None"
    agent_id: str | None
    state: "AgentState | dict[str, Any] | None"  # pyright: ignore[reportExplicitAny]
    hooks: "list[HookProvider] | None"
    session_manager: "SessionManager | None"
    tool_executor: "ToolExecutor | None"


class ThreadKwargs(TypedDict, total=False):
    """Kwargs mirror of ``ThreadConfig`` fields; enforced equal at module load."""

    model: "Model | str | None"
    system_prompt: str | None
    tools: "list[AgentTool | ToolProvider | str] | None"
    post_conditions: list[PostCondition]
    max_attempts: int
    structured_output: bool
    agent_kwargs: AgentKwargs
    name: str | None
    description: str | None
    input_schema: JSONSchema | None
    thread_name: str | None
    config_hook: "Callable[[ThreadContext], ThreadKwargs] | None"
    summarization_strategy: "SummarizationStrategy | None"


class ThreadMergedKwargs(AgentKwargs, ThreadKwargs):
    """Union of ``AgentKwargs`` and ``ThreadKwargs`` for decorator kwargs typing."""


@dataclass(frozen=True)
class ThreadConfig:
    """Immutable configuration for one thread's execution.

    Invariants:
        Field names equal ``ThreadKwargs`` annotation keys (enforced at
        module load).
    """

    model: "Model | str | None" = None
    """LLM model to use for the agent."""
    system_prompt: str | None = None
    """System prompt for the agent."""
    tools: "tuple[AgentTool | ToolProvider | str, ...]" = ()
    """Tools provided to the agent."""
    post_conditions: tuple[PostCondition, ...] = ()
    """List of functions to call to validate the output."""
    max_attempts: int = 10
    """Maximum number of times to retry producing an output that satisfies
    the post-conditions."""
    structured_output: bool = True
    """Whether to use structured output mode (agent has to call a tool to
    provide an answer). Can be ``False`` only if the output type is ``str``."""

    agent_kwargs: AgentKwargs = field(default_factory=AgentKwargs)
    """Extra kwargs forwarded to ``strands.Agent``."""

    name: str | None = None
    """Name used when the thread is made available as a tool."""
    description: str | None = None
    """Description used when the thread is made available as a tool."""
    input_schema: JSONSchema | None = None
    """Override the ``input_schema`` when the thread is made available as a tool."""

    thread_name: str | None = None
    """Name of the thread for visualization purposes."""

    config_hook: "Callable[[ThreadContext], ThreadKwargs] | None" = None
    """Optional per-cycle config patch.

    Called at the start of every cycle with the current ``ThreadContext``.
    The returned ``ThreadKwargs`` is merged into this config for that
    cycle only — ``self`` is never mutated. Useful for injecting
    cycle-specific tools or overriding model/system_prompt per-cycle.

    Merge semantics: all fields replace the base config value if present
    in the returned dict (including ``tools``). ``config_hook`` itself is
    excluded from the merge. ``config_hook`` is NOT re-invoked during
    summarization; the already-resolved cycle config is passed to the
    strategy verbatim so a fork-based strategy can keep the cached prompt
    prefix valid.

    Concurrency:
        Called synchronously inside ``_run_cycle``; must not block.
    """

    summarization_strategy: "SummarizationStrategy | None" = None
    """Strategy that decides how to compact the history on context overflow.

    ``None`` (default) uses ``DefaultSummarizationStrategy`` with its own
    defaults. Set to a custom implementation to override the full
    summarization policy. See :class:`SummarizationStrategy` for the
    contract.
    """


def split_config_and_agent_kwargs(
    **kwargs: Unpack[ThreadMergedKwargs],
) -> tuple[ThreadKwargs, AgentKwargs]:
    """Partition merged kwargs into ``ThreadConfig`` fields and ``Agent`` kwargs.

    Args:
        kwargs: Keys drawn from ``ThreadMergedKwargs``.

    Returns:
        ``(thread_kwargs, agent_kwargs)`` disjoint on keys, union equal
        to ``kwargs``.

    Ensures:
        - ``thread_kwargs`` keys are a subset of ``ThreadConfig`` field names.
        - ``agent_kwargs`` keys are disjoint from ``ThreadConfig`` field names.
    """
    ...
