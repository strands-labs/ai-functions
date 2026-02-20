"""Summarizing conversation history management with configurable options."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from strands.agent import Agent
from strands.agent.conversation_manager import ConversationManager
from strands.types.content import Message
from strands.types.exceptions import ContextWindowOverflowException

from ai_functions import ai_function
from ai_functions.core import AIFunction

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)

# Agent state key constants for defensive coding
_STATE_KEY_LAST_PROMPT = "_summarizing_manager_last_prompt"
_STATE_KEY_LAST_TOKENS = "_summarizing_manager_last_current_tokens"

# Default summarization model
_default_summarization_model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

_SUMMARY_TEMPLATE = """\
You are currently working on the following task:
<task>
{prompt}
</task>
The conversation became too long and was restarted. Here is a summary of the current implementation status:
<summary>
{summary}
</summary>
What follows are the last steps that have been executed, please continue your work from there.
"""


@ai_function(model=_default_summarization_model)
def _default_summarizer() -> str:
    """Default conversation summarizer.

    SYSTEM MESSAGE: This conversation is becoming too long. We will summarize it and resume the task.
    Please stop working on the project and use the FinalAnswer tool to provide a summary of all
    information you want to preserve.
    This should include:
      1. What has already been done for the current task
      2. What errors you made in the past that should be avoided, and how to avoid them
      3. A list of next steps to conclude the task.

    IMPORTANT: You have to call `FinalAnswer` now. Do not use any other tool.
    Just answer immediately with the summary described above.
    """
    return None  # type: ignore[return-value]  # Decorator handles actual return


class SummarizingWindowConversationManager(ConversationManager):
    """Implements a summarizing window manager.

    This manager provides a configurable option to summarize older context instead of
    simply trimming it, helping preserve important information while staying within
    context limits.
    """

    def __init__(
        self,
        max_tokens: int,
        preserve_recent_messages: int,
        summarization_function: AIFunction | None = None,
    ):
        """Initialize the summarizing conversation manager.

        Args:
            max_tokens: Maximum number of tokens before triggering summarization.
            preserve_recent_messages: Number of recent messages to always preserve.
            summarization_function: AI Function to use to summarize the conversation history.
        """
        super().__init__()
        self.max_tokens = max_tokens
        self.preserve_recent_messages = preserve_recent_messages
        self.summarization_function = summarization_function or _default_summarizer
        self._max_words_per_message: int = 8000

    def _extract_text_from_message(self, message: Message) -> str:
        """Extract all text content from a message."""
        texts = []
        for content_block in message.get("content", []):
            if "text" in content_block:
                texts.append(content_block["text"])
        return " ".join(texts)

    def max_word_overflow_index(self, messages: List[Message]) -> int:
        """Return the index of the message with the largest overflow, or -1."""
        overflows = [
            (i, len(self._extract_text_from_message(msg).split()) - self._max_words_per_message)
            for i, msg in enumerate(messages)
        ]
        idx, overflow = max(overflows, key=lambda x: x[1], default=(-1, -1))
        return idx if overflow > 0 else -1

    def apply_management(
        self,
        agent: Agent,
        current_tokens: int | None = None,
        uncached_tokens: int | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Apply management strategy to conversation history.

        For the summarizing conversation manager, no proactive management is performed.
        Summarization only occurs when there's a context overflow that triggers reduce_context.

        Args:
            agent: The agent whose conversation history will be managed.
                The agent's messages list is modified in-place.
            current_tokens: Current total token count.
            uncached_tokens: Number of uncached tokens.
            invocation_state: State dictionary containing prompt and other data.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        invocation_state = invocation_state or {}

        # Store the prompt and current_tokens in agent.state for later use in reduce_context
        # This is stateless at the instance level - each agent has its own state
        if "prompt" in invocation_state:
            agent.state.set(_STATE_KEY_LAST_PROMPT, invocation_state["prompt"])
        if current_tokens is not None:
            agent.state.set(_STATE_KEY_LAST_TOKENS, current_tokens)

        if current_tokens is None:
            return
        if current_tokens > self.max_tokens:
            logger.info("TaskConversationManager triggered: tokens=%d exceeds max=%d", current_tokens, self.max_tokens)
            if "prompt" not in invocation_state:
                raise ValueError("Invocation state does not contain the prompt")
            self.summarize_conversation(agent, current_tokens, invocation_state["prompt"])

    def summarize_conversation(
        self, agent: Agent, current_tokens: int | None = None, prompt: str | None = None
    ) -> None:
        """Summarize conversation history.

        Args:
            agent: The agent whose conversation will be summarized.
            current_tokens: Current token count (unused).
            prompt: The task prompt to include in summary message.
        """
        # Validate prompt parameter
        if prompt is None or not prompt.strip():
            logger.warning("Cannot summarize: prompt is None or empty")
            return

        logger.info(
            "Starting conversation summarization (total messages: %d, preserve: %d)",
            len(agent.messages),
            self.preserve_recent_messages,
        )
        max_words_overflow_index = self.max_word_overflow_index(agent.messages)

        messages_to_summarize_count = max(1, len(agent.messages))
        # Ensure we don't summarize recent messages
        messages_to_summarize_count = min(
            messages_to_summarize_count, len(agent.messages) - self.preserve_recent_messages
        )
        messages_to_summarize_count = max(messages_to_summarize_count, max_words_overflow_index + 1)

        if messages_to_summarize_count <= 0:
            return

        # Adjust split point to avoid breaking ToolUse/ToolResult pairs
        messages_to_summarize_count = self._adjust_split_point_for_tool_pairs(
            agent.messages, messages_to_summarize_count
        )

        if messages_to_summarize_count <= 0:
            return

        # Split messages
        messages_to_summarize = agent.messages[:messages_to_summarize_count]
        remaining_messages = agent.messages[messages_to_summarize_count:]

        try:
            # copy the summarization AI function and add the messages to summarize in its agent history
            agent_kwargs = self.summarization_function.config.agent_kwargs | {"messages": messages_to_summarize}
            summarization_function = self.summarization_function.replace(agent_kwargs=agent_kwargs)

            # Generate summary using the agent (no need to pass any argument, messages are already in the history)
            summary: str = summarization_function()

            if not summary:
                logger.error("Agent failed to generate a summary (empty response)")
                return

        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return

        summary_text = _SUMMARY_TEMPLATE.format(prompt=prompt, summary=summary)

        summary_message: Message = {"role": "user", "content": [{"text": summary_text}]}
        # Replace the summarized messages with the summary
        agent.messages[:] = [summary_message] + remaining_messages
        logger.info(
            "Summarization complete: %d messages summarized, %d messages preserved",
            messages_to_summarize_count,
            len(remaining_messages),
        )

    def reduce_context(self, agent: Agent, e: Optional[Exception] = None, **kwargs: Any) -> None:
        """Reduce context using summarization.

        Args:
            agent: The agent whose conversation history will be reduced.
                The agent's messages list is modified in-place.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ContextWindowOverflowException: If the context cannot be summarized.
        """
        # Get current_tokens and prompt from kwargs, or use stored values from agent.state
        # These values were stored by apply_management for stateless async-safe operation
        current_tokens = kwargs.get("current_tokens") or agent.state.get(_STATE_KEY_LAST_TOKENS)
        prompt = kwargs.get("prompt") or agent.state.get(_STATE_KEY_LAST_PROMPT)

        if prompt is None:
            logger.warning("No prompt available for summarization, using empty prompt")
            prompt = ""

        self.summarize_conversation(agent, current_tokens=current_tokens, prompt=prompt)

    def _adjust_split_point_for_tool_pairs(self, messages: List[Message], split_point: int) -> int:
        """Adjust the split point to avoid breaking ToolUse/ToolResult pairs.

        Uses the same logic as SlidingWindowConversationManager for consistency.

        Args:
            messages: The full list of messages.
            split_point: The initially calculated split point.

        Returns:
            The adjusted split point that doesn't break ToolUse/ToolResult pairs.

        Raises:
            ContextWindowOverflowException: If no valid split point can be found.
        """
        if split_point > len(messages):
            raise ContextWindowOverflowException("Split point exceeds message array length")
        if split_point == len(messages):
            return split_point

        def has_key(index: int, key: str) -> bool:
            if index >= len(messages):
                return False
            return any(key in content for content in messages[index]["content"])

        # Find the next valid split_point
        while split_point < len(messages):
            # Oldest message cannot be a toolResult because it needs a toolUse preceding it
            is_tool_result = has_key(split_point, "toolResult")
            # Oldest message can be a toolUse only if a toolResult immediately follows it.
            # Exception: if this is the last message, a toolUse is acceptable (toolResult comes later)
            is_incomplete_tool_use = (
                has_key(split_point, "toolUse")
                and split_point + 1 < len(messages)  # There is a next message
                and not has_key(split_point + 1, "toolResult")  # And it's not a toolResult
            )
            if is_tool_result or is_incomplete_tool_use:
                split_point += 1
            else:
                break
        else:
            # If we didn't find a valid split_point, raise an exception
            raise ContextWindowOverflowException("Unable to trim conversation context!")

        return split_point
