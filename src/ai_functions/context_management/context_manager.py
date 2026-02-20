"""Context manager hook for prompt caching and conversation management."""

import hashlib
import json
import logging
import uuid
from enum import Enum
from typing import Any

from strands.agent import Agent
from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry
from strands.types.content import Message
from strands.types.event_loop import Usage

logger = logging.getLogger(__name__)

# Agent state key constants for defensive coding
_STATE_KEY_CYCLE_COUNT = "_cycle_count"
_STATE_KEY_CYCLE_HISTORY = "_cycle_history"
_STATE_KEY_LAST_CHECKPOINT = "_last_checkpoint_cycle"


class NoCacheModel(str, Enum):
    """Models that don't support prompt caching."""

    NOVA = "nova"
    NEMOTRON = "nemotron"


# Token tracking constants
# _TOKEN_KEYS = ["inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens"]
# _ZERO_TOKENS = {k: 0 for k in _TOKEN_KEYS}
_ZERO_TOKENS = Usage(inputTokens=0, outputTokens=0, totalTokens=0, cacheReadInputTokens=0, cacheWriteInputTokens=0)


def hash_structure(data: Any) -> str:
    """Generate SHA256 hash of data structure."""
    try:
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (TypeError, ValueError):
        return str(uuid.uuid4())


def _remove_checkpoint(message: Message) -> None:
    """Remove cache checkpoint from message content."""
    if isinstance(message.get("content"), list):
        message["content"] = [block for block in message["content"] if not block.get("cachePoint")]


class ContextManager(HookProvider):
    """Hook for prompt caching, conversation management, and cycle tracking."""

    def __init__(
        self,
        manage_conversation_every_cycle: bool = True,
        max_non_cache_tokens: int = 8192,
        no_cache_list: list[NoCacheModel] | None = None,
        max_cycles_before_summary: int = 100,
        max_cycles: int = 150,
    ):
        """Initialize context manager with caching and conversation management settings.

        Args:
            manage_conversation_every_cycle: Apply conversation management on every cycle
            max_non_cache_tokens: Maximum uncached tokens before resetting checkpoint
            no_cache_list: List of models that don't support caching
            max_cycles_before_summary: Cycles before forcing summarization
            max_cycles: Maximum cycles before stopping
        """
        self.manage_conversation_every_cycle = manage_conversation_every_cycle
        self.max_non_cache_tokens = max_non_cache_tokens
        self.no_cache_list = no_cache_list or [NoCacheModel.NOVA, NoCacheModel.NEMOTRON]
        self.max_cycles_before_summary = max_cycles_before_summary
        self.max_cycles = max_cycles

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hook callbacks."""
        registry.add_callback(BeforeModelCallEvent, self.before_model_call)

    def before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Handle before model call - use previous cycle tokens for decisions."""
        agent = event.agent
        invocation_state = event.invocation_state

        # Get tokens from previous completed cycle (cycles[-2])
        # Current cycle (cycles[-1]) has zeros because it just started
        try:
            cycles = agent.event_loop_metrics.agent_invocations[-1].cycles
            cycle_usage = cycles[-2].usage if len(cycles) >= 2 else _ZERO_TOKENS
            cycle_tokens = _ZERO_TOKENS | cycle_usage
        except (AttributeError, IndexError, KeyError):
            cycle_tokens = _ZERO_TOKENS.copy()

        input_tokens, cache_read = cycle_tokens["inputTokens"], cycle_tokens["cacheReadInputTokens"]

        # Save cycle history
        cycle_count = agent.state.get(_STATE_KEY_CYCLE_COUNT) or 0
        cycle_history = agent.state.get(_STATE_KEY_CYCLE_HISTORY) or []
        # Convert Usage object to dict for JSON serialization in agent.state
        cycle_history.append({"cycle": cycle_count, "tokens": dict(cycle_tokens)})
        agent.state.set(_STATE_KEY_CYCLE_HISTORY, cycle_history)

        logger.info(
            f"ContextManager: input={input_tokens}, cache_read={cache_read}, "
            f"output={cycle_tokens['outputTokens']}, cache_write={cycle_tokens['cacheWriteInputTokens']}\n"
        )

        # Cycle tracking and limits
        cycle_count = cycle_count + 1
        agent.state.set(_STATE_KEY_CYCLE_COUNT, cycle_count)

        if self.max_cycles is not None and cycle_count > self.max_cycles:
            logger.warning(f"Max cycles {self.max_cycles} exceeded, stopping")
            event.invocation_state["request_state"]["stop_event_loop"] = True
            return

        if (
            self.max_cycles_before_summary > 0
            and cycle_count % self.max_cycles_before_summary == 0
            and agent.conversation_manager
        ):
            logger.info(f"Forcing summarization at cycle {cycle_count}")
            agent.conversation_manager.apply_management(
                agent=agent, current_tokens=float("inf"), invocation_state=invocation_state
            )
            self._reset_checkpoint(agent, cycle_count)
            return

        # Apply conversation management if needed
        if self.manage_conversation_every_cycle and agent.conversation_manager:
            msg_hash = hash_structure(agent.messages)
            agent.conversation_manager.apply_management(
                agent=agent, current_tokens=cache_read + input_tokens, invocation_state=invocation_state
            )

            if hash_structure(agent.messages) != msg_hash:
                logger.info("Resetting checkpoint after conversation change")
                self._reset_checkpoint(agent, cycle_count)
                session_mgr = getattr(agent, "_session_manager", None)
                if session_mgr and hasattr(session_mgr, "sync_agent"):
                    try:
                        session_mgr.sync_agent(agent)
                    except Exception as e:
                        logger.warning(f"Failed to sync agent: {e}")

        # Reset checkpoint if needed (stateless check using agent.state)
        last_checkpoint_cycle = agent.state.get(_STATE_KEY_LAST_CHECKPOINT) or -1
        if input_tokens >= self.max_non_cache_tokens and last_checkpoint_cycle < cycle_count:
            logger.info(f"Resetting checkpoint (input_tokens={input_tokens} >= {self.max_non_cache_tokens})")
            self._reset_checkpoint(agent, cycle_count)

    def _reset_checkpoint(self, agent: Agent, cycle_count: int) -> None:
        """Reset cache checkpoint.

        Args:
            agent: The agent instance
            cycle_count: Current cycle count (stored in agent.state to track when checkpoint was last reset)
        """
        if not agent.messages:
            return

        model_id = agent.model.get_config().get("model_id", "")
        no_cache = any(mdl.value in model_id.lower() for mdl in self.no_cache_list)

        # Clean up old checkpoints
        for message in agent.messages:
            _remove_checkpoint(message)

        # Add new checkpoint if model supports it
        if not no_cache and isinstance(agent.messages[-1].get("content"), list):
            agent.messages[-1]["content"].append({"cachePoint": {"type": "default"}})
            # Track checkpoint cycle in agent.state (stateless at instance level)
            agent.state.set(_STATE_KEY_LAST_CHECKPOINT, cycle_count)
            logger.debug(f"Checkpoint reset for {model_id} at cycle {cycle_count}")
        elif not no_cache:
            logger.warning("Last message content invalid, skipping cachePoint")
        else:
            logger.debug(f"Skipping cachePoint for no-cache model {model_id}")
