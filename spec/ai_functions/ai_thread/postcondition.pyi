"""Post-condition validator types and the shared retry loop over them.

:class:`PostConditionResult` / :data:`PostCondition` define the validator
contract. :func:`run_post_condition_loop` is the turn-level retry loop shared
by the foreign-runtime threads (``ClaudeAgentThread``, ``KiroAgentThread``):
each drives a session that owns its own conversation history, so the loop
never replays the original prompt — retries ride the failure feedback as the
next user turn. ``AIThread`` keeps its own loop because it owns its history
and offers bound keyword arguments to condition callables; these threads take
a single string prompt, so there are none to offer.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Callable

from pydantic import BaseModel, Field

from ..types import ThreadContext


class PostConditionResult(BaseModel):
    """Outcome of running a single post-condition validator.

    Invariants:
        ``passed is False`` implies ``message is not None``.
    """

    passed: bool = Field(description="Whether the condition passed")
    message: str | None = Field(default=None, description="Validation message")

    def model_post_init(self, __context: object) -> None:
        """Validate the ``passed``/``message`` invariant after construction.

        Args:
            __context: Pydantic-internal post-init context (unused).

        Raises:
            ValueError: ``passed`` is false and ``message`` is ``None``.
        """
        ...


PostCondition = Callable[..., "PostConditionResult | None"]
"""Callable validating an AI function result.

The callable receives the result as the first positional argument. If any
argument names in the signature of the callable match keys in
``bound_args``, the callable also receives those values as keyword
arguments.

Return values:

- ``PostConditionResult(passed=True)`` / ``None`` — condition passed.
- ``PostConditionResult(passed=False, message=...)`` — condition failed.
- Raising an exception — treated as a failed condition whose message is
  the exception text.
"""

async def evaluate_post_conditions(
    result: str,
    post_conditions: tuple[PostCondition, ...],
) -> list[str]:
    """Evaluate every post-condition against ``result`` in parallel.

    A condition returning ``None``/``passed`` passes; ``passed=False``
    contributes its message; a raised exception is treated as failure with
    the exception text. Callers here take a single string prompt, so there
    are no bound keyword arguments to offer condition callables.

    Args:
        result: The candidate result string for the turn.
        post_conditions: Validators to run.

    Returns:
        Failure messages; empty when all conditions pass.
    """
    ...

async def run_post_condition_loop(
    ctx: ThreadContext,
    prompt: str,
    *,
    thread_name: str,
    post_conditions: tuple[PostCondition, ...],
    max_attempts: int,
    inject_buffer: list[str],
    send_turn: Callable[[str], Awaitable[str]],
) -> str:
    """Run turns through ``send_turn`` until ``post_conditions`` pass.

    The shared cycle body for threads whose backing session owns its own
    conversation history. Each attempt drains ``inject_buffer``, emitting one
    ``MESSAGE_USER`` event per entry, and prepends the drained entries to the
    outgoing turn. On the first attempt the buffer holds any caller-supplied
    side-channel messages and ``prompt`` is appended (with its own
    ``MESSAGE_USER`` event); on retries the buffer holds the post-condition
    failure feedback, and the original ``prompt`` is NOT re-sent — the session
    owns the history, so retries ride the feedback turn.

    With empty ``post_conditions`` the loop short-circuits after one turn, so
    the default behaviour is a single query regardless of ``max_attempts``.

    Args:
        ctx: The current cycle's context; used only to emit events.
        prompt: User prompt for the first attempt's turn.
        thread_name: Name used in feedback messages and error attribution.
        post_conditions: Validators run against each turn's result.
        max_attempts: Maximum turns to satisfy ``post_conditions``; treated
            as at least 1. Ignored when ``post_conditions`` is empty.
        inject_buffer: The thread's live side-channel buffer. Drained here;
            failure feedback is appended to it between attempts.
        send_turn: Sends one combined user turn to the backing session and
            returns its result text.

    Returns:
        The first turn result that satisfies ``post_conditions`` (or the
        sole turn's result when there are none).

    Emits:
        MESSAGE_USER — one per drained inject-buffer entry, plus one for
        ``prompt`` on the first attempt.

    Raises:
        AIFunctionError: ``post_conditions`` were not satisfied within
            ``max_attempts`` attempts.
    """
    ...
