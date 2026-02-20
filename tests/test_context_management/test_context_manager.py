"""Tests for ContextManager.

Tests for prompt caching infrastructure, checkpoint management, and cache reset logic.
"""

from unittest.mock import Mock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from ai_functions.context_management.context_manager import (
    ContextManager,
    NoCacheModel,
    _remove_checkpoint,
    hash_structure,
)


class TestHashStructure:
    """Tests for hash_structure function."""

    def test_hashes_simple_dict(self):
        """Test hash_structure generates consistent hash for simple dict."""
        data = {"key": "value", "number": 42}
        hash1 = hash_structure(data)
        hash2 = hash_structure(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_hashes_dict_with_different_order(self):
        """Test hash_structure generates same hash regardless of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        assert hash_structure(data1) == hash_structure(data2)

    def test_different_data_generates_different_hash(self):
        """Test hash_structure generates different hashes for different data."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        assert hash_structure(data1) != hash_structure(data2)

    def test_hashes_nested_structure(self):
        """Test hash_structure handles nested structures."""
        data = {"outer": {"inner": {"deep": "value"}}, "list": [1, 2, 3]}
        hash1 = hash_structure(data)
        hash2 = hash_structure(data)
        assert hash1 == hash2

    def test_returns_random_hash_on_serialization_failure(self):
        """Test hash_structure returns random hash when serialization fails."""
        # Create an object that can't be serialized
        data = {"func": lambda x: x}
        hash1 = hash_structure(data)
        hash2 = hash_structure(data)
        # Should return different random hashes
        assert hash1 != hash2
        assert len(hash1) == 36  # UUID length


class TestRemoveCheckpoint:
    """Tests for _remove_checkpoint function."""

    def test_removes_checkpoint_from_message_content_list(self):
        """Test _remove_checkpoint removes cachePoint from message content list."""
        message = {
            "role": "user",
            "content": [
                {"text": "Hello"},
                {"cachePoint": {"type": "default"}},
                {"text": "World"},
            ],
        }
        _remove_checkpoint(message)
        assert len(message["content"]) == 2
        assert message["content"][0] == {"text": "Hello"}
        assert message["content"][1] == {"text": "World"}

    def test_handles_message_without_checkpoint(self):
        """Test _remove_checkpoint handles message without cachePoint."""
        message = {"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]}
        _remove_checkpoint(message)
        assert len(message["content"]) == 2

    def test_handles_string_content(self):
        """Test _remove_checkpoint handles message with string content."""
        message = {"role": "user", "content": "Hello World"}
        _remove_checkpoint(message)
        # Should not raise an error, string content is unchanged
        assert message["content"] == "Hello World"

    def test_removes_multiple_checkpoints(self):
        """Test _remove_checkpoint removes all checkpoints."""
        message = {
            "role": "user",
            "content": [
                {"text": "A"},
                {"cachePoint": {"type": "default"}},
                {"text": "B"},
                {"cachePoint": {"type": "default"}},
            ],
        }
        _remove_checkpoint(message)
        assert len(message["content"]) == 2
        assert message["content"][0] == {"text": "A"}
        assert message["content"][1] == {"text": "B"}


class TestContextManagerInitialization:
    """Tests for ContextManager initialization."""

    def test_initializes_with_defaults(self):
        """Test ContextManager initializes with default values."""
        manager = ContextManager()
        assert manager.manage_conversation_every_cycle is True
        assert manager.max_non_cache_tokens == 8192
        assert NoCacheModel.NOVA in manager.no_cache_list
        assert NoCacheModel.NEMOTRON in manager.no_cache_list

    def test_initializes_with_custom_values(self):
        """Test ContextManager initializes with custom values."""
        manager = ContextManager(
            manage_conversation_every_cycle=False,
            max_non_cache_tokens=4096,
            no_cache_list=[NoCacheModel.NOVA],
        )
        assert manager.manage_conversation_every_cycle is False
        assert manager.max_non_cache_tokens == 4096
        assert manager.no_cache_list == [NoCacheModel.NOVA]

    def test_initializes_with_empty_no_cache_list(self):
        """Test ContextManager uses default when no_cache_list is empty."""
        manager = ContextManager(no_cache_list=[])
        # Empty list gets replaced with default
        assert NoCacheModel.NOVA in manager.no_cache_list
        assert NoCacheModel.NEMOTRON in manager.no_cache_list


class TestContextManagerResetCheckpoint:
    """Tests for ContextManager._reset_checkpoint method."""

    def test_does_nothing_with_empty_messages(self):
        """Test _reset_checkpoint does nothing when messages is empty."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = []
        agent.model.get_config.return_value = {"model_id": "claude"}
        agent.state = Mock()

        manager._reset_checkpoint(agent, cycle_count=1)
        # Should not raise an error
        assert agent.messages == []

    def test_removes_old_checkpoints(self):
        """Test _reset_checkpoint removes all old checkpoints."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "A"}, {"cachePoint": {"type": "default"}}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.state = Mock()

        manager._reset_checkpoint(agent, cycle_count=1)

        # Old checkpoint should be removed
        assert len(agent.messages[0]["content"]) == 1
        assert agent.messages[0]["content"][0] == {"text": "A"}

    def test_adds_checkpoint_to_last_message(self):
        """Test _reset_checkpoint adds cachePoint to last message."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.state = Mock()

        manager._reset_checkpoint(agent, cycle_count=1)

        # New checkpoint should be added to last message
        assert {"cachePoint": {"type": "default"}} in agent.messages[-1]["content"]
        # Checkpoint cycle should be tracked in agent.state
        agent.state.set.assert_called_with("_last_checkpoint_cycle", 1)

    def test_skips_checkpoint_for_nova_model(self):
        """Test _reset_checkpoint skips adding checkpoint for nova model."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.model.get_config.return_value = {"model_id": "nova-pro"}
        agent.state = Mock()

        manager._reset_checkpoint(agent, cycle_count=1)

        # No checkpoint should be added for nova model
        assert len(agent.messages[-1]["content"]) == 1

    def test_skips_checkpoint_for_nemotron_model(self):
        """Test _reset_checkpoint skips adding checkpoint for nemotron model."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.model.get_config.return_value = {"model_id": "nemotron-large"}
        agent.state = Mock()

        manager._reset_checkpoint(agent, cycle_count=1)

        # No checkpoint should be added for nemotron model
        assert len(agent.messages[-1]["content"]) == 1

    def test_handles_message_without_content_list(self):
        """Test _reset_checkpoint logs warning for invalid last message."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [{"role": "user", "content": "string content"}]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.state = Mock()

        with patch("ai_functions.context_management.context_manager.logger") as mock_logger:
            manager._reset_checkpoint(agent, cycle_count=1)
            # Should log a warning
            mock_logger.warning.assert_called_once()

    def test_handles_message_without_content_key(self):
        """Test _reset_checkpoint handles message without content key."""
        manager = ContextManager()
        agent = Mock()
        agent.messages = [{"role": "user"}]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.state = Mock()

        with patch("ai_functions.context_management.context_manager.logger") as mock_logger:
            manager._reset_checkpoint(agent, cycle_count=1)
            # Should log a warning
            mock_logger.warning.assert_called_once()


class TestContextManagerApplyConversationManagement:
    """Tests for ContextManager conversation management via before_model_call."""

    def test_does_nothing_when_no_conversation_manager(self):
        """Test before_model_call does nothing without conversation manager."""
        manager = ContextManager()
        agent = Mock()
        agent.conversation_manager = None
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [Mock()]
        agent.event_loop_metrics.agent_invocations[0].cycles = []

        event = Mock()
        event.agent = agent
        event.invocation_state = {"request_state": {}}

        # Should not raise an error
        manager.before_model_call(event)

    def test_calls_conversation_manager_with_correct_parameters(self):
        """Test before_model_call calls manager with correct params."""
        manager = ContextManager(manage_conversation_every_cycle=True)
        agent = Mock()
        agent.conversation_manager = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "test"}]}]
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default
        invocation_state = {"prompt": "test prompt", "request_state": {}}

        # Mock event_loop_metrics with proper structure
        usage = {"cacheReadInputTokens": 100, "inputTokens": 50, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]  # Need at least 2 cycles
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]

        event = Mock()
        event.agent = agent
        event.invocation_state = invocation_state

        manager.before_model_call(event)

        agent.conversation_manager.apply_management.assert_called_once()
        call_args = agent.conversation_manager.apply_management.call_args
        assert call_args.kwargs["current_tokens"] == 150  # cache_read + input_tokens

    def test_calculates_total_tokens_correctly(self):
        """Test before_model_call calculates total tokens."""
        manager = ContextManager(manage_conversation_every_cycle=True)
        agent = Mock()
        agent.conversation_manager = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "test"}]}]
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        # Mock with specific token values
        usage = {"cacheReadInputTokens": 1200, "inputTokens": 3000, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]

        event = Mock()
        event.agent = agent
        event.invocation_state = {"request_state": {}}

        manager.before_model_call(event)

        # Should calculate cache_tokens + uncached_tokens = 4200
        agent.conversation_manager.apply_management.assert_called_once()
        call_args = agent.conversation_manager.apply_management.call_args
        assert call_args.kwargs["current_tokens"] == 4200


class TestContextManagerBeforeModelCall:
    """Tests for ContextManager.before_model_call hook."""

    def test_resets_checkpoint_when_messages_change(self):
        """Test before_model_call resets checkpoint when conversation changes."""
        manager = ContextManager(manage_conversation_every_cycle=True)

        # Create mock agent with necessary attributes
        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.conversation_manager = Mock()
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        # Mock conversation manager to modify messages
        def modify_messages(*args, **kwargs):
            agent.messages.append({"role": "assistant", "content": [{"text": "Hi"}]})

        agent.conversation_manager.apply_management.side_effect = modify_messages

        # Create mock event
        event = Mock()
        event.agent = agent

        # Mock metrics as dict
        usage = {"cacheReadInputTokens": 100, "inputTokens": 50, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]

        # Mock invocation_state
        event.invocation_state = {"request_state": {}}

        with patch.object(manager, "_reset_checkpoint") as mock_reset:
            manager.before_model_call(event)
            # Should call reset_checkpoint because messages changed
            mock_reset.assert_called_once_with(agent, 1)

    def test_resets_checkpoint_when_input_tokens_exceed_max(self):
        """Test before_model_call resets checkpoint when input tokens exceed max."""
        manager = ContextManager(max_non_cache_tokens=1000)

        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.conversation_manager = None
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        event = Mock()
        event.agent = agent

        # Mock metrics with input_tokens exceeding max
        usage = {"cacheReadInputTokens": 100, "inputTokens": 1500, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]
        event.invocation_state = {"request_state": {}}

        with patch.object(manager, "_reset_checkpoint") as mock_reset:
            manager.before_model_call(event)
            # Should call reset_checkpoint because input_tokens >= max
            mock_reset.assert_called_once_with(agent, 1)

    def test_does_not_reset_when_already_reset_this_cycle(self):
        """Test before_model_call does not reset if checkpoint already reset this cycle."""
        manager = ContextManager(max_non_cache_tokens=1000)

        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.conversation_manager = None
        agent.state = Mock()
        # Mock agent.state.get to return cycle 1 for "_last_checkpoint_cycle"
        # This simulates that we already reset the checkpoint in the current cycle
        agent.state.get.side_effect = lambda key, default=None: (
            1 if key == "_last_checkpoint_cycle" else (0 if key == "_cycle_count" else default)
        )

        event = Mock()
        event.agent = agent

        # Mock metrics with input_tokens exceeding max
        usage = {"cacheReadInputTokens": 100, "inputTokens": 1500, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]
        event.invocation_state = {"request_state": {}}

        with patch.object(manager, "_reset_checkpoint") as mock_reset:
            manager.before_model_call(event)
            # Should not call reset_checkpoint because last_checkpoint_cycle (1) == cycle_count (1)
            mock_reset.assert_not_called()

    def test_syncs_agent_with_session_manager(self):
        """Test before_model_call syncs agent with session manager after changes."""
        manager = ContextManager(manage_conversation_every_cycle=True)

        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.model.get_config.return_value = {"model_id": "claude-sonnet"}
        agent.conversation_manager = Mock()
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default
        session_manager = Mock()
        agent._session_manager = session_manager

        # Mock conversation manager to modify messages
        def modify_messages(*args, **kwargs):
            agent.messages.append({"role": "assistant", "content": [{"text": "Hi"}]})

        agent.conversation_manager.apply_management.side_effect = modify_messages

        event = Mock()
        event.agent = agent

        usage = {"cacheReadInputTokens": 100, "inputTokens": 50, "outputTokens": 0, "cacheWriteInputTokens": 0}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]
        event.invocation_state = {"request_state": {}}

        manager.before_model_call(event)

        # Should sync agent with session manager
        session_manager.sync_agent.assert_called_once_with(agent)


class TestContextManagerMissingMetrics:
    """Tests for ContextManager with missing metrics attributes."""

    def test_handles_missing_cache_read_tokens(self):
        """Test before_model_call handles missing cache_read_tokens attribute."""
        manager = ContextManager(max_non_cache_tokens=1000)

        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.conversation_manager = None
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        event = Mock()
        event.agent = agent

        # Mock usage dict without cacheReadInputTokens key
        usage = {"inputTokens": 500}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]
        event.invocation_state = {"request_state": {}}

        # Should not raise an error
        manager.before_model_call(event)

    def test_handles_missing_input_tokens(self):
        """Test before_model_call handles missing input_tokens attribute."""
        manager = ContextManager(max_non_cache_tokens=1000)

        agent = Mock()
        agent.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        agent.conversation_manager = None
        agent.state = Mock()
        # Mock agent.state.get to properly return default values like dict.get()
        agent.state.get.side_effect = lambda key, default=None: default

        event = Mock()
        event.agent = agent

        # Mock usage dict without inputTokens key
        usage = {"cacheReadInputTokens": 100}
        cycle = Mock()
        cycle.usage = usage
        invocation = Mock()
        invocation.cycles = [Mock(), cycle]
        invocation.cycles[0].usage = usage
        agent.event_loop_metrics = Mock()
        agent.event_loop_metrics.agent_invocations = [invocation]
        event.invocation_state = {"request_state": {}}

        # Should not raise an error
        manager.before_model_call(event)


class TestContextManagerPropertyBased:
    """Property-based tests for ContextManager."""

    @given(max_tokens=st.integers(min_value=1, max_value=100000))
    @settings(max_examples=50)
    def test_max_non_cache_tokens_stored_correctly(self, max_tokens: int):
        """For any max_non_cache_tokens value, ContextManager should store it correctly."""
        manager = ContextManager(max_non_cache_tokens=max_tokens)
        assert manager.max_non_cache_tokens == max_tokens

    @given(manage_every_cycle=st.booleans())
    @settings(max_examples=10)
    def test_manage_conversation_every_cycle_stored_correctly(self, manage_every_cycle: bool):
        """For any manage_conversation_every_cycle value, ContextManager should store it."""
        manager = ContextManager(manage_conversation_every_cycle=manage_every_cycle)
        assert manager.manage_conversation_every_cycle == manage_every_cycle


class TestNoCacheModel:
    """Tests for NoCacheModel enum."""

    def test_has_nova_value(self):
        """Test NoCacheModel has NOVA with correct string value."""
        assert NoCacheModel.NOVA.value == "nova"

    def test_has_nemotron_value(self):
        """Test NoCacheModel has NEMOTRON with correct string value."""
        assert NoCacheModel.NEMOTRON.value == "nemotron"
