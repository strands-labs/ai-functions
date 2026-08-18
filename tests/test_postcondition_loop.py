"""The shared post-condition retry loop for session-owned-history threads.

``run_post_condition_loop`` is the cycle body of ``ClaudeAgentThread`` and
``KiroAgentThread``. These tests drive it with a fake ``send_turn`` and a
recording ``ThreadContext``, pinning the contract both backends rely on:
the prompt is sent exactly once, retries carry only the failure feedback,
every outgoing part gets its own ``MESSAGE_USER`` event, and exhaustion
raises ``AIFunctionError``.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.ai_thread.postcondition import (
    PostConditionResult,
    evaluate_post_conditions,
    run_post_condition_loop,
)
from ai_functions.types import Event, MessageUserEvent, ThreadContext, ThreadId
from ai_functions.types.events import EventKind


class _Recorder:
    """Collects the loop's interactions: events emitted and turns sent."""

    def __init__(self, replies: list[str]) -> None:
        self.events: list[Event] = []
        self.sent: list[str] = []
        self._replies = replies

    def ctx(self) -> ThreadContext:
        return ThreadContext(
            thread_id=ThreadId("t-loop"),
            coordinator=None,  # type: ignore[arg-type]  # the loop never touches it
            on_event=self.events.append,
            on_interrupt=None,  # type: ignore[arg-type]  # the loop never touches it
            pause_signal=asyncio.Event(),
            cancel_signal=asyncio.Event(),
        )

    async def send_turn(self, combined: str) -> str:
        self.sent.append(combined)
        return self._replies[len(self.sent) - 1]

    def user_messages(self) -> list[str]:
        return [e.text for e in self.events if e.kind == EventKind.MESSAGE_USER and isinstance(e, MessageUserEvent)]


def _fail(message: str):  # noqa: ANN202 - test helper factory
    def cond(result: str) -> PostConditionResult:
        return PostConditionResult(passed=False, message=message)

    return cond


def _pass_when(expected: str):  # noqa: ANN202 - test helper factory
    def cond(result: str) -> PostConditionResult:
        if result == expected:
            return PostConditionResult(passed=True)
        return PostConditionResult(passed=False, message=f"expected {expected!r}, got {result!r}")

    return cond


async def test_no_post_conditions_is_a_single_turn() -> None:
    """Empty post_conditions short-circuits after one turn, whatever max_attempts says."""
    rec = _Recorder(replies=["first answer"])
    result = await run_post_condition_loop(
        rec.ctx(),
        "do the task",
        thread_name="t",
        post_conditions=(),
        max_attempts=10,
        inject_buffer=[],
        send_turn=rec.send_turn,
    )
    assert result == "first answer"
    assert rec.sent == ["do the task"]
    assert rec.user_messages() == ["do the task"]


async def test_inject_buffer_prepends_and_emits_per_entry() -> None:
    """Each pending side-channel message gets its own MESSAGE_USER and precedes the prompt."""
    rec = _Recorder(replies=["ok"])
    buffer = ["note one", "note two"]
    result = await run_post_condition_loop(
        rec.ctx(),
        "the prompt",
        thread_name="t",
        post_conditions=(),
        max_attempts=1,
        inject_buffer=buffer,
        send_turn=rec.send_turn,
    )
    assert result == "ok"
    assert rec.sent == ["note one\n\nnote two\n\nthe prompt"]
    assert rec.user_messages() == ["note one", "note two", "the prompt"]
    assert buffer == []


async def test_retry_rides_feedback_and_never_resends_prompt() -> None:
    """On failure the next turn carries only the feedback; the prompt is sent once."""
    rec = _Recorder(replies=["wrong", "RIGHT"])
    result = await run_post_condition_loop(
        rec.ctx(),
        "answer with RIGHT",
        thread_name="checker",
        post_conditions=(_pass_when("RIGHT"),),
        max_attempts=5,
        inject_buffer=[],
        send_turn=rec.send_turn,
    )
    assert result == "RIGHT"
    assert len(rec.sent) == 2
    assert rec.sent[0] == "answer with RIGHT"
    # The retry turn is the feedback alone — the session already holds the prompt.
    assert "answer with RIGHT" not in rec.sent[1]
    assert "[checker] Post-condition failures (attempt 1/5)" in rec.sent[1]
    assert "expected 'RIGHT', got 'wrong'" in rec.sent[1]
    # Feedback is a user turn, so it is also shadowed as MESSAGE_USER.
    assert len(rec.user_messages()) == 2


async def test_exhaustion_raises_with_thread_attribution() -> None:
    """max_attempts failures raise AIFunctionError naming the thread."""
    rec = _Recorder(replies=["a", "b", "c"])
    with pytest.raises(AIFunctionError) as exc_info:
        await run_post_condition_loop(
            rec.ctx(),
            "task",
            thread_name="stubborn",
            post_conditions=(_fail("nope"),),
            max_attempts=3,
            inject_buffer=[],
            send_turn=rec.send_turn,
        )
    assert "3 attempt(s)" in str(exc_info.value)
    assert exc_info.value.function_name == "stubborn"
    assert len(rec.sent) == 3


async def test_multiple_failures_are_all_fed_back() -> None:
    """Every failing condition contributes one bullet to the feedback turn."""
    rec = _Recorder(replies=["draft", "still draft"])
    with pytest.raises(AIFunctionError):
        await run_post_condition_loop(
            rec.ctx(),
            "write",
            thread_name="t",
            post_conditions=(_fail("too terse"), _fail("missing citation")),
            max_attempts=2,
            inject_buffer=[],
            send_turn=rec.send_turn,
        )
    feedback = rec.sent[1]
    assert "- too terse" in feedback
    assert "- missing citation" in feedback


async def test_all_conditions_must_pass_to_return() -> None:
    """One passing condition does not mask another's failure."""
    rec = _Recorder(replies=["draft ok", "draft ok"])
    with pytest.raises(AIFunctionError):
        await run_post_condition_loop(
            rec.ctx(),
            "write",
            thread_name="t",
            post_conditions=(_pass_when("draft ok"), _fail("style: too terse")),
            max_attempts=2,
            inject_buffer=[],
            send_turn=rec.send_turn,
        )
    # Only the failing condition appears in the feedback.
    assert "style: too terse" in rec.sent[1]
    assert "expected" not in rec.sent[1]


async def test_condition_exception_is_a_failure_message() -> None:
    """A raising condition fails with the exception text instead of crashing the loop."""

    def explode(result: str) -> PostConditionResult:
        raise ValueError("validator blew up")

    errors = await evaluate_post_conditions("anything", (explode,))
    assert errors == ["validator blew up"]


async def test_async_conditions_are_awaited() -> None:
    """Coroutine-returning conditions are awaited, matching the sync path."""

    async def slow_pass(result: str) -> PostConditionResult:
        await asyncio.sleep(0)
        return PostConditionResult(passed=True)

    async def slow_fail(result: str) -> PostConditionResult:
        await asyncio.sleep(0)
        return PostConditionResult(passed=False, message="slow no")

    errors = await evaluate_post_conditions("x", (slow_pass, slow_fail))
    assert errors == ["slow no"]


async def test_none_result_counts_as_pass() -> None:
    """A condition returning None passes."""

    def silent(result: str) -> None:
        return None

    errors = await evaluate_post_conditions("x", (silent,))
    assert errors == []
