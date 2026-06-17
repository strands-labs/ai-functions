"""AIThread — live, per-spawn Strands-agent thread implementation.

Exports the AIThread class, its ``AIFunction`` factory / decorator, and
all AIThread-specific types (config, errors, post-conditions, message
reconstruction, summarization, runtime-facing tools).
"""

from __future__ import annotations

from .ai_function import AIFunction, ai_function
from .ai_thread import AIThread, OutputSpec
from .config import (
    AgentKwargs,
    CodeExecutionMode,
    ThreadConfig,
    ThreadKwargs,
    ThreadMergedKwargs,
    split_config_and_agent_kwargs,
)
from .errors import AIFunctionError, ValidationError
from .postcondition import PostCondition, PostConditionResult
from .reconstruction import reconstruct_messages, render_renderable_events
from .summarization import (
    DefaultSummarizationStrategy,
    SummarizationFailedError,
    SummarizationStrategy,
)
from .tools import coordinator_tools

__all__ = [
    "ai_function",
    "AgentKwargs",
    "AIFunction",
    "AIFunctionError",
    "AIThread",
    "CodeExecutionMode",
    "coordinator_tools",
    "DefaultSummarizationStrategy",
    "OutputSpec",
    "PostCondition",
    "PostConditionResult",
    "reconstruct_messages",
    "render_renderable_events",
    "split_config_and_agent_kwargs",
    "SummarizationFailedError",
    "SummarizationStrategy",
    "ThreadConfig",
    "ThreadKwargs",
    "ThreadMergedKwargs",
    "ValidationError",
]
