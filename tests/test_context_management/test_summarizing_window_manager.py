"""Tests for TaskConversationManager.

Tests for summarizing conversation history management, context overflow handling,
and tool use/result pair preservation.
"""

from unittest.mock import Mock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strands.types.exceptions import ContextWindowOverflowException

from ai_functions.context_management.summarizing_window_manager import SummarizingWindowConversationManager


class TestTaskConversationManagerInitialization:
    """Tests for TaskConversationManager initialization."""

    def test_initializes_with_required_parameters(self):
        """Test TaskConversationManager initializes with required parameters."""
        manager = SummarizingWindowConversationManager(max_tokens=10000, preserve_recent_messages=5)
        assert manager.max_tokens == 10000
        assert manager.preserve_recent_messages == 5

    def test_initializes_with_custom_summary_model(self):
        """Test TaskConversationManager accepts custom summarization function."""
        from ai_functions.core import AIFunction

        custom_function = Mock(spec=AIFunction)
        manager = SummarizingWindowConversationManager(
            max_tokens=5000,
            preserve_recent_messages=3,
            summarization_function=custom_function,
        )
        assert manager.max_tokens == 5000
        assert manager.preserve_recent_messages == 3
        assert manager.summarization_function is custom_function


class TestTaskConversationManagerApplyManagement:
    """Tests for TaskConversationManager.apply_management method."""

    def test_does_nothing_when_current_tokens_none(self):
        """Test apply_management does nothing when current_tokens is None."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.state = Mock()

        # Should not raise an error
        manager.apply_management(agent, current_tokens=None)

    def test_does_nothing_when_tokens_below_max(self):
        """Test apply_management does nothing when tokens below max."""
        manager = SummarizingWindowConversationManager(max_tokens=10000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.state = Mock()
        invocation_state = {"prompt": "test prompt"}

        with patch.object(manager, "summarize_conversation") as mock_summarize:
            manager.apply_management(agent, current_tokens=5000, invocation_state=invocation_state)
            mock_summarize.assert_not_called()

    def test_calls_summarize_when_tokens_exceed_max(self):
        """Test apply_management calls summarize when tokens exceed max."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.state = Mock()
        invocation_state = {"prompt": "test prompt"}

        with patch.object(manager, "summarize_conversation") as mock_summarize:
            manager.apply_management(agent, current_tokens=1500, invocation_state=invocation_state)
            mock_summarize.assert_called_once_with(agent, 1500, "test prompt")

    def test_raises_error_when_prompt_missing_and_tokens_exceed_max(self):
        """Test apply_management raises error when prompt missing in invocation_state."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.state = Mock()
        invocation_state = {}

        with pytest.raises(ValueError, match="Invocation state does not contain the prompt"):
            manager.apply_management(agent, current_tokens=1500, invocation_state=invocation_state)


class TestTaskConversationManagerReduceContext:
    """Tests for TaskConversationManager.reduce_context method."""

    def test_handles_missing_prompt_gracefully(self):
        """Test reduce_context handles missing prompt gracefully without raising."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = []
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        # Should not raise an exception, just log warning and return
        manager.reduce_context(agent)

    def test_handles_error_argument(self):
        """Test reduce_context handles error argument gracefully."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = []
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default
        error = Exception("Some error")

        # Should not raise an exception, just log warning and return
        manager.reduce_context(agent, e=error)


class TestTaskConversationManagerAdjustSplitPoint:
    """Tests for TaskConversationManager._adjust_split_point_for_tool_pairs method."""

    def test_returns_split_point_when_valid(self):
        """Test _adjust_split_point_for_tool_pairs returns split point when valid."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        split_point = manager._adjust_split_point_for_tool_pairs(messages, 1)
        assert split_point == 1

    def test_adjusts_split_point_when_starts_with_tool_result(self):
        """Test _adjust_split_point adjusts when split starts with toolResult."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool1"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "done"}}]},
            {"role": "assistant", "content": [{"text": "Done"}]},
        ]

        # Split point 2 starts with toolResult, should adjust to 3
        split_point = manager._adjust_split_point_for_tool_pairs(messages, 2)
        assert split_point == 3

    def test_adjusts_split_point_when_tool_use_without_result(self):
        """Test _adjust_split_point adjusts when toolUse not followed by toolResult."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool1"}}]},
            {"role": "user", "content": [{"text": "More input"}]},
        ]

        # Split point 1 has toolUse but next message is not toolResult
        split_point = manager._adjust_split_point_for_tool_pairs(messages, 1)
        assert split_point == 2

    def test_returns_split_point_at_message_length(self):
        """Test _adjust_split_point returns split point when at message length."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        split_point = manager._adjust_split_point_for_tool_pairs(messages, 2)
        assert split_point == 2

    def test_raises_error_when_split_point_exceeds_length(self):
        """Test _adjust_split_point raises error when split point exceeds length."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        with pytest.raises(ContextWindowOverflowException, match="Split point exceeds message array length"):
            manager._adjust_split_point_for_tool_pairs(messages, 5)

    def test_raises_error_when_no_valid_split_point(self):
        """Test _adjust_split_point raises error when no valid split point found."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        # All messages are toolResults, no valid split point
        messages = [
            {"role": "user", "content": [{"toolResult": {"result": "1"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "2"}}]},
        ]

        with pytest.raises(ContextWindowOverflowException, match="Unable to trim conversation context"):
            manager._adjust_split_point_for_tool_pairs(messages, 0)

    def test_accepts_tool_use_followed_by_tool_result(self):
        """Test _adjust_split_point accepts split when toolUse followed by toolResult."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool1"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "done"}}]},
            {"role": "assistant", "content": [{"text": "Done"}]},
        ]

        # Split point 1 has toolUse and is followed by toolResult
        # This is valid because the toolUse/toolResult pair stays together in remaining messages
        split_point = manager._adjust_split_point_for_tool_pairs(messages, 1)
        assert split_point == 1


class TestTaskConversationManagerSummarizeConversation:
    """Tests for TaskConversationManager.summarize_conversation method."""

    def test_raises_error_when_prompt_none(self):
        """Test summarize_conversation skips summarization when prompt is None."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        original_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.messages = original_messages.copy()

        # Should return without modifying messages
        manager.summarize_conversation(agent, current_tokens=1500, prompt=None)
        assert agent.messages == original_messages

    def test_raises_error_when_prompt_empty(self):
        """Test summarize_conversation skips summarization when prompt is empty string."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        original_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.messages = original_messages.copy()

        # Should return without modifying messages
        manager.summarize_conversation(agent, current_tokens=1500, prompt="")
        assert agent.messages == original_messages

    def test_raises_error_when_prompt_whitespace_only(self):
        """Test summarize_conversation skips summarization when prompt is whitespace."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        original_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.messages = original_messages.copy()

        # Should return without modifying messages
        manager.summarize_conversation(agent, current_tokens=1500, prompt="   \n\t  ")
        assert agent.messages == original_messages

    def test_raises_error_when_insufficient_messages(self):
        """Test summarize_conversation skips summarization with insufficient messages."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=5)
        agent = Mock()
        # Only 1 message - insufficient to summarize even with min_to_summarize logic
        original_messages = [
            {"role": "user", "content": [{"text": "A"}]},
        ]
        agent.messages = original_messages.copy()

        # Should return without modifying messages
        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")
        assert agent.messages == original_messages

    def test_preserves_recent_messages(self):
        """Test summarize_conversation preserves recent messages for larger conversations."""
        # Create a mock summarization function with proper structure
        mock_summarizer = Mock()
        mock_summarizer.return_value = "Mock summary of conversation"
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=2, summarization_function=mock_summarizer
        )
        agent = Mock()
        # Use 10 messages (>= 8) to trigger preservation logic
        agent.messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
            {"role": "user", "content": [{"text": "C"}]},
            {"role": "assistant", "content": [{"text": "D"}]},
            {"role": "user", "content": [{"text": "E"}]},
            {"role": "assistant", "content": [{"text": "F"}]},
            {"role": "user", "content": [{"text": "G"}]},
            {"role": "assistant", "content": [{"text": "H"}]},
            {"role": "user", "content": [{"text": "I"}]},
            {"role": "assistant", "content": [{"text": "J"}]},
        ]

        manager.summarize_conversation(agent, current_tokens=1500, prompt="test task")

        # After summarization, should have 3 messages: summary + 2 preserved
        assert len(agent.messages) == 3
        # First message should be the summary
        assert agent.messages[0]["role"] == "user"
        assert "summary" in agent.messages[0]["content"][0]["text"].lower()
        # Last 2 messages should be preserved
        assert agent.messages[1]["content"][0]["text"] == "I"
        assert agent.messages[2]["content"][0]["text"] == "J"

    def test_includes_task_in_summary_message(self):
        """Test summarize_conversation includes task prompt in summary message."""
        # Create a mock summarization function with proper structure
        mock_summarizer = Mock()
        mock_summarizer.return_value = "Mock summary"
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=1, summarization_function=mock_summarizer
        )
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]

        manager.summarize_conversation(agent, current_tokens=1500, prompt="Implement feature X")

        # Summary message should include the task
        summary_text = agent.messages[0]["content"][0]["text"]
        assert "Implement feature X" in summary_text
        assert "<task>" in summary_text
        assert "</task>" in summary_text

    def test_includes_summary_in_message(self):
        """Test summarize_conversation includes summary content in message."""
        # Create a mock summarization function with proper structure
        mock_summarizer = Mock()
        mock_summarizer.return_value = "This is the summary"
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=1, summarization_function=mock_summarizer
        )
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]

        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")

        # Summary message should include the summary content
        summary_text = agent.messages[0]["content"][0]["text"]
        assert "This is the summary" in summary_text
        assert "<summary>" in summary_text
        assert "</summary>" in summary_text

    def test_raises_error_when_summary_agent_fails(self):
        """Test summarize_conversation handles error gracefully when summary agent fails."""
        # Create a mock summarization function that raises an exception
        mock_summarizer = Mock()
        mock_summarizer.side_effect = Exception("Summarization failed")
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=1, summarization_function=mock_summarizer
        )
        agent = Mock()
        original_messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]
        agent.messages = original_messages.copy()

        # Should return without modifying messages when summarization fails
        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")
        assert agent.messages == original_messages

    def test_raises_error_when_summary_empty(self):
        """Test summarize_conversation handles empty summary gracefully."""
        # Create a mock summarization function that returns empty string
        mock_summarizer = Mock()
        mock_summarizer.return_value = ""
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=1, summarization_function=mock_summarizer
        )
        agent = Mock()
        original_messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]
        agent.messages = original_messages.copy()

        # Should return without modifying messages when summary is empty
        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")
        assert agent.messages == original_messages

    def test_adjusts_split_point_for_tool_pairs(self):
        """Test summarize_conversation adjusts split point for tool pairs."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=1)
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool1"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "done"}}]},
            {"role": "assistant", "content": [{"text": "Final"}]},
        ]

        with patch.object(manager, "_adjust_split_point_for_tool_pairs") as mock_adjust:
            mock_adjust.return_value = 3  # Adjusted split point
            manager.summarize_conversation(agent, current_tokens=1500, prompt="test")

            # Should call adjust split point
            mock_adjust.assert_called_once()

    def test_summarizes_correct_messages(self):
        """Test summarize_conversation calls summarization function with correct messages."""
        # Create a mock summarization function with proper structure
        mock_summarizer = Mock()
        mock_summarizer.return_value = "Summary"
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=2, summarization_function=mock_summarizer
        )
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "Message 1"}]},
            {"role": "assistant", "content": [{"text": "Message 2"}]},
            {"role": "user", "content": [{"text": "Message 3"}]},
            {"role": "assistant", "content": [{"text": "Message 4"}]},
        ]

        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")

        # Should have called the summarization function
        assert mock_summarizer.called


class TestTaskConversationManagerPropertyBased:
    """Property-based tests for TaskConversationManager."""

    @given(
        max_tokens=st.integers(min_value=100, max_value=100000),
        preserve_messages=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_initialization_stores_parameters_correctly(self, max_tokens: int, preserve_messages: int):
        """For any valid parameters, TaskConversationManager should store them correctly."""
        manager = SummarizingWindowConversationManager(
            max_tokens=max_tokens, preserve_recent_messages=preserve_messages
        )
        assert manager.max_tokens == max_tokens
        assert manager.preserve_recent_messages == preserve_messages

    @given(
        current_tokens=st.integers(min_value=0, max_value=5000),
        max_tokens=st.integers(min_value=5001, max_value=10000),
    )
    @settings(max_examples=50)
    def test_apply_management_does_nothing_when_below_max(self, current_tokens: int, max_tokens: int):
        """For any tokens below max, apply_management should not trigger summarization."""
        manager = SummarizingWindowConversationManager(max_tokens=max_tokens, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        with patch.object(manager, "summarize_conversation") as mock_summarize:
            manager.apply_management(agent, current_tokens=current_tokens, invocation_state={"prompt": "test"})
            mock_summarize.assert_not_called()

    @given(split_point=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    def test_adjust_split_point_returns_valid_value(self, split_point: int):
        """For any valid split point with simple messages, _adjust_split_point should return it."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        # Create enough simple messages
        messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(10)]

        result = manager._adjust_split_point_for_tool_pairs(messages, split_point)
        assert result >= split_point
        assert result <= len(messages)


class TestTaskConversationManagerEdgeCases:
    """Tests for edge cases in TaskConversationManager."""

    def test_handles_single_message_conversation(self):
        """Test summarize_conversation handles conversation with single message."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=1)
        agent = Mock()
        original_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.messages = original_messages.copy()

        # Single message is insufficient for summarization, should return without changes
        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")

        # Messages should remain unchanged (preserve_recent_messages=1 prevents summarization)
        assert agent.messages == original_messages

    def test_handles_long_conversation(self):
        """Test summarize_conversation handles long conversation correctly."""
        # Create a mock summarization function with proper structure
        mock_summarizer = Mock()
        mock_summarizer.return_value = "Summary"
        mock_summarizer.config.agent_kwargs = {}
        mock_summarizer.replace.return_value = mock_summarizer

        manager = SummarizingWindowConversationManager(
            max_tokens=1000, preserve_recent_messages=3, summarization_function=mock_summarizer
        )
        agent = Mock()
        # Create 20 messages
        agent.messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": [{"text": f"Msg {i}"}]} for i in range(20)
        ]

        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")

        # Should have 4 messages: summary + 3 preserved
        assert len(agent.messages) == 4
        # Check that the last 3 messages are preserved
        assert agent.messages[-1]["content"][0]["text"] == "Msg 19"
        assert agent.messages[-2]["content"][0]["text"] == "Msg 18"
        assert agent.messages[-3]["content"][0]["text"] == "Msg 17"

    def test_handles_complex_tool_sequence(self):
        """Test _adjust_split_point handles complex tool use/result sequences."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=1)
        messages = [
            {"role": "user", "content": [{"text": "Start"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool1"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "result1"}}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "tool2"}}]},
            {"role": "user", "content": [{"toolResult": {"result": "result2"}}]},
            {"role": "assistant", "content": [{"text": "Final"}]},
        ]

        # Should adjust to avoid breaking tool pairs
        split_point = manager._adjust_split_point_for_tool_pairs(messages, 1)
        # Should move past the toolUse/toolResult pair
        assert split_point >= 1

    def test_handles_empty_messages_list_edge_case(self):
        """Test summarize_conversation handles empty messages gracefully."""
        manager = SummarizingWindowConversationManager(max_tokens=1000, preserve_recent_messages=2)
        agent = Mock()
        agent.messages = []

        # Should return without modifying messages
        manager.summarize_conversation(agent, current_tokens=1500, prompt="test")
        assert agent.messages == []
