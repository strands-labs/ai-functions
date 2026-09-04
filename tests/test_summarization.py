"""Tests for conversation summarization: reconstruction splice, strategy, flow.

Covers:
- ``reconstruct_messages`` consumes ``ContextSummarizedEvent`` correctly
  (0/1/2 summaries, superseded events, preserved events, I10 post-splice).
- ``DefaultSummarizationStrategy`` exposes a pluggable async ``summarize``
  that returns ``list[RenderableEvent]``; internal split helpers respect
  min/max bounds and tool-pair adjustment.
- ``SummarizationFailedError`` semantics.
- End-to-end: a thread that hits ``ContextWindowOverflowException``
  summarizes via the strategy and retries successfully.
- ``AIThread`` rejects a user-supplied non-``None``
  ``conversation_manager``.
- ``DefaultSummarizationStrategy`` rejects
  ``summarize_by_forking=True`` against structured output at summarize
  time.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, cast

import pytest
from strands.models import Model
from strands.types.content import Message, Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from ai_functions import (
    DefaultSummarizationStrategy,
    SummarizationFailedError,
    ai_function,
)
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.ai_thread.summarization import (
    _adjust_split_point_for_tool_pairs,
    _estimate_message_tokens,
)
from ai_functions.testing import RuntimeHarness, Turn
from ai_functions.types import ThreadId
from ai_functions.types.events import (
    ContextSummarizedEvent,
    Event,
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    RenderableEvent,
    TokenUsageEvent,
)
from ai_functions.types.ids import MessageId


def _user(text: str) -> Message:
    return Message(role="user", content=[{"text": text}])


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[{"text": text}])


def _mk_user_event(text: str, thread_id: str = "t1") -> MessageUserEvent:
    return MessageUserEvent(thread_id=ThreadId(thread_id), text=text)


def _mk_assistant_event(text: str, thread_id: str = "t1") -> MessageAssistantCompleteEvent:
    return MessageAssistantCompleteEvent(
        thread_id=ThreadId(thread_id),
        message_id=MessageId(f"msg-{text[:6]}"),
        content=[{"text": text}],
    )


# ── reconstruct_messages: ContextSummarizedEvent splicing ──────────────────────


def test_reconstruct_no_summary_events_unchanged() -> None:
    """Without any summary event, reconstruction is the usual event-by-event replay."""
    events: list[Event] = [
        _mk_user_event("hi"),
        _mk_assistant_event("hello"),
        _mk_user_event("again"),
    ]
    msgs = reconstruct_messages(events)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"][0].get("text") == "hi"


def test_reconstruct_single_summary_collapses_prefix() -> None:
    """A single ContextSummarizedEvent replaces the prefix with the strategy's new_history."""
    e_user1 = _mk_user_event("old-turn")
    e_assistant1 = _mk_assistant_event("old-answer")
    # The strategy returns a user-message summary followed by a preserved tail.
    new_history: list[RenderableEvent] = [
        _mk_user_event("summary: prior conversation about foo"),
        _mk_user_event("recent user turn"),
    ]
    summary_event = ContextSummarizedEvent(
        thread_id=ThreadId("t1"),
        new_history=new_history,
    )
    e_user2 = _mk_user_event("new-turn")

    msgs = reconstruct_messages(cast("list[Event]", [e_user1, e_assistant1, summary_event, e_user2]))

    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0].get("text") == "summary: prior conversation about foo"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0].get("text") == "recent user turn"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0].get("text") == "new-turn"


def test_reconstruct_multiple_summaries_last_wins() -> None:
    """When multiple ContextSummarizedEvents exist, only the last one is applied."""
    e_user1 = _mk_user_event("very-old")
    e_assistant1 = _mk_assistant_event("very-old-answer")
    summary1 = ContextSummarizedEvent(
        thread_id=ThreadId("t1"),
        new_history=[_mk_user_event("first summary (should be ignored)")],
    )
    e_user2 = _mk_user_event("middle")
    e_assistant2 = _mk_assistant_event("middle-answer")
    summary2 = ContextSummarizedEvent(
        thread_id=ThreadId("t1"),
        new_history=[
            _mk_user_event("second summary (should win)"),
            _mk_user_event("retained"),
        ],
    )
    e_user3 = _mk_user_event("newest")

    events: list[Event] = [e_user1, e_assistant1, summary1, e_user2, e_assistant2, summary2, e_user3]
    msgs = reconstruct_messages(events)

    assert len(msgs) == 3
    assert msgs[0]["content"][0].get("text") == "second summary (should win)"
    assert msgs[1]["content"][0].get("text") == "retained"
    assert msgs[2]["content"][0].get("text") == "newest"


def test_reconstruct_summary_with_empty_new_history() -> None:
    """An empty new_history yields nothing for the summary boundary itself."""
    e_user1 = _mk_user_event("old")
    summary = ContextSummarizedEvent(
        thread_id=ThreadId("t1"),
        new_history=[],
    )
    msgs = reconstruct_messages(cast("list[Event]", [e_user1, summary]))
    assert msgs == []


def test_reconstruct_i10_heals_after_splice() -> None:
    """Orphaned toolUse inside new_history is healed post-splice (I10)."""
    # Construct a new_history whose last assistant event contains a
    # dangling toolUse block. The healer should insert a synthetic
    # toolResult user message after it.
    new_history: list[RenderableEvent] = [
        _mk_user_event("summary"),
        _mk_user_event("ok"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("msg-orphan"),
            content=[
                {"text": "calling tool"},
                {"toolUse": {"toolUseId": "tu-orphan", "name": "foo", "input": {}}},
            ],
        ),
    ]
    e_user1 = _mk_user_event("whatever")
    summary = ContextSummarizedEvent(
        thread_id=ThreadId("t1"),
        new_history=new_history,
    )
    msgs = reconstruct_messages(cast("list[Event]", [e_user1, summary]))

    # Expected: [user(summary), user("ok"), assistant(...tu-orphan...),
    #            user(synthesized toolResult for tu-orphan)]
    assert len(msgs) == 4
    heal_block = msgs[3]["content"][0]
    assert "toolResult" in heal_block
    tr = heal_block["toolResult"]
    assert tr["toolUseId"] == "tu-orphan"
    assert tr["status"] == "error"


# ── DefaultSummarizationStrategy: construction ────────────────────────────────


def test_strategy_init_validates_bounds() -> None:
    with pytest.raises(ValueError):
        DefaultSummarizationStrategy(preserve_min_messages=0)
    with pytest.raises(ValueError):
        DefaultSummarizationStrategy(preserve_min_tokens=100, preserve_max_tokens=50)


# ── DefaultSummarizationStrategy: internal split helper ───────────────────────
#
# The split logic is no longer part of the public strategy protocol, but the
# built-in implementation exposes ``_split`` for internal use. Testing it
# directly gives us coverage of the size-bound behavior without standing up
# a full model harness.


def test_split_respects_min_messages() -> None:
    """With min_messages=2 the tail has at least 2 messages if available."""
    events: list[Event] = [_mk_user_event(f"turn-{i}" * 40) for i in range(5)]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=2,
        preserve_min_tokens=0,
        preserve_max_tokens=100_000,
    )
    to_summarize, preserved, _boundary = strategy._split(events)  # noqa: SLF001
    assert len(preserved) >= 2
    assert len(to_summarize) >= 1
    assert len(to_summarize) + len(preserved) == len(events)


def test_split_raises_when_everything_preserved() -> None:
    """If bounds allow keeping the whole history, the strategy raises."""
    events: list[Event] = [_mk_user_event("x")]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=10,
        preserve_min_tokens=0,
        preserve_max_tokens=100_000,
    )
    with pytest.raises(SummarizationFailedError):
        strategy._split(events)  # noqa: SLF001


def test_split_enforces_max_tokens() -> None:
    """A huge OLDER message is folded into the summarized prefix; recent tail preserved."""
    huge_text = "x" * 200_000  # vastly over preserve_max_tokens=40_000 default
    events: list[Event] = [
        _mk_user_event("first"),
        _mk_user_event("second"),
        _mk_user_event(huge_text),
        _mk_user_event("tail"),
    ]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=40_000,
    )
    to_summarize, preserved, _boundary = strategy._split(events)  # noqa: SLF001
    # The huge message is NOT in preserved (it exceeds the ceiling once the
    # most-recent "tail" message is already kept).
    for ev in preserved:
        if isinstance(ev, MessageUserEvent):
            assert len(ev.text) < 100_000, "huge message should not be preserved"
    # ... but it IS somewhere in to_summarize (as a MessageUserEvent).
    summarized_texts = [e.text for e in to_summarize if isinstance(e, MessageUserEvent)]
    assert any(len(t) >= 100_000 for t in summarized_texts)


def test_split_preserves_newest_and_shrinks_when_newest_fits() -> None:
    """When the newest message fits the ceiling, it is preserved and the prefix shrinks.

    The monotonic-shrink guarantee: a successful split always yields a non-empty
    preserved tail (containing at least the newest message) and a summarized
    prefix strictly smaller than the full history. An empty preserved tail is
    not a valid split; that case raises instead (see the degenerate tests).
    """
    huge_older = "x" * 200_000
    events: list[Event] = [
        _mk_user_event(huge_older),  # must be summarized
        _mk_user_event("recent small turn"),  # newest: fits, must be preserved
    ]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=2_000,  # above the small newest turn, below the huge one
    )
    to_summarize, preserved, _boundary = strategy._split(events)  # noqa: SLF001
    # The newest message is preserved; the huge older one is summarized.
    assert len(preserved) == 1
    assert isinstance(preserved[0], MessageUserEvent)
    assert preserved[0].text == "recent small turn"
    # Prefix is non-empty and strictly smaller than the input (the shrink
    # guarantee).
    assert len(to_summarize) == 1
    assert len(to_summarize) < len(events)


def test_split_raises_on_single_oversized_message() -> None:
    """A single message bigger than the tail budget fails honestly (degenerate case).

    When the most-recent message alone exceeds ``preserve_max_tokens`` there is
    no split that helps: the oversized message must live in either the preserved
    tail or the summarizer's input, and neither fits. ``_split`` raises
    immediately rather than summarizing an ever-shrinking prefix while the
    oversized tail survives every pass.
    """
    huge_text = "x" * 200_000
    events: list[Event] = [
        _mk_user_event("small older turn"),
        _mk_user_event(huge_text),  # newest message: alone exceeds the ceiling
    ]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=1_000,
    )
    with pytest.raises(SummarizationFailedError, match="single oversized message"):
        strategy._split(events)  # noqa: SLF001


def test_split_two_message_degenerate_case_raises() -> None:
    """The exact two-message shape: small prompt + oversized tool result → honest error.

    A user prompt followed by one tool result that by itself fills the context.
    Summarizing the prompt does not help; the giant result cannot be split.
    ``_split`` raises rather than looping.
    """
    huge_result = "y" * 300_000
    events: list[Event] = [
        _mk_user_event("please read the giant file"),
        _mk_assistant_event(huge_result),  # stands in for an oversized tool result
    ]
    strategy = DefaultSummarizationStrategy(
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=2_000,
    )
    with pytest.raises(SummarizationFailedError, match="single oversized message"):
        strategy._split(events)  # noqa: SLF001


# ── Tool-pair adjustment helper ───────────────────────────────────────────────


def test_adjust_skips_orphan_toolresult() -> None:
    """A split landing on an orphaned toolResult advances forward."""
    messages: list[Message] = [
        _user("q"),
        Message(
            role="assistant",
            content=[{"toolUse": {"toolUseId": "tu-1", "name": "f", "input": {}}}],
        ),
        Message(
            role="user",
            content=[{"toolResult": {"toolUseId": "tu-1", "status": "success", "content": [{"text": "r"}]}}],
        ),
        _assistant("done"),
    ]
    # Splitting at idx=2 would leave toolResult without toolUse — must advance.
    adjusted = _adjust_split_point_for_tool_pairs(messages, 2)
    assert adjusted >= 3


def test_adjust_allows_tool_use_pair_if_result_next() -> None:
    """A split at a toolUse whose result immediately follows is legal."""
    messages: list[Message] = [
        _user("q"),
        Message(
            role="assistant",
            content=[{"toolUse": {"toolUseId": "tu-1", "name": "f", "input": {}}}],
        ),
        Message(
            role="user",
            content=[{"toolResult": {"toolUseId": "tu-1", "status": "success", "content": [{"text": "r"}]}}],
        ),
    ]
    # split at idx=1 (the toolUse message) is legal because idx+1 carries
    # the matching toolResult.
    adjusted = _adjust_split_point_for_tool_pairs(messages, 1)
    assert adjusted == 1


# ── Token estimator ──────────────────────────────────────────────────────────


def test_estimate_proportional_to_chars() -> None:
    short = _estimate_message_tokens(_user("hi"))
    long = _estimate_message_tokens(_user("x" * 1000))
    assert long > short * 10  # roughly linear


# ── AIThread config validation ───────────────────────────────────────────────


def test_aithread_rejects_user_supplied_conversation_manager() -> None:
    """A non-None conversation_manager in agent_kwargs is rejected at init."""
    import dataclasses as _dc

    from strands.agent.conversation_manager import SlidingWindowConversationManager

    from ai_functions.ai_thread.ai_thread import AIThread
    from ai_functions.ai_thread.config import AgentKwargs

    @ai_function[str](structured_output=False)
    def _greet() -> str:
        return "hi"

    bad_config = _dc.replace(
        _greet.config,
        agent_kwargs=AgentKwargs(conversation_manager=SlidingWindowConversationManager()),
    )
    with pytest.raises(AIFunctionError, match="conversation_manager"):
        AIThread(_greet, bad_config)


def test_build_agent_user_callback_handler_does_not_collide() -> None:
    """A user ``callback_handler`` in agent_kwargs must not double-pass to Agent.

    The runtime passes its streaming ``callback_handler`` explicitly; leaving a
    user-supplied one in the forwarded ``agent_kwargs`` used to raise
    ``TypeError: got multiple values for keyword argument 'callback_handler'``.
    """
    import dataclasses as _dc

    from ai_functions.ai_thread.ai_thread import AIThread
    from ai_functions.ai_thread.config import AgentKwargs

    calls: list[object] = []

    def user_cb(**kwargs: object) -> None:
        calls.append(kwargs)

    @ai_function[str](structured_output=False)
    def _greet() -> str:
        return "hi"

    config = _dc.replace(_greet.config, agent_kwargs=AgentKwargs(callback_handler=user_cb))
    thread = AIThread(_greet, config)

    # Runtime callback wins when provided (no collision with the user's).
    runtime_cb = lambda **_: None  # noqa: E731
    agent = thread._build_agent([], config, None, runtime_cb)  # noqa: SLF001
    assert agent.callback_handler is runtime_cb

    # With no runtime callback, the user-supplied one is honored.
    agent2 = thread._build_agent([], config, None, None)  # noqa: SLF001
    assert agent2.callback_handler is user_cb


async def test_strategy_rejects_forking_with_structured_output() -> None:
    """summarize_by_forking=True + structured_output=True is rejected at summarize()."""
    from ai_functions.ai_thread.config import ThreadConfig

    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=True,
        preserve_min_messages=1,
        preserve_min_tokens=0,
    )
    events: list[Event] = [_mk_user_event("something"), _mk_assistant_event("reply"), _mk_user_event("more")]

    # Build a minimal cycle_config with structured_output=True. The strategy
    # does not need a real ThreadContext for this early-validation path.
    cycle_config = ThreadConfig(structured_output=True)
    with pytest.raises(SummarizationFailedError, match="summarize_by_forking"):
        await strategy.summarize(events, cast(Any, object()), cycle_config)  # pyright: ignore[reportExplicitAny]


# ── summarization_enabled: recursion guard on summarizer templates ────────────


def test_summarizer_templates_disable_summarization() -> None:
    """Both summarizer builders produce a template with summarization disabled.

    This is the recursion guard: a summarizer thread must never itself summarize
    (proactively or reactively), or a summarization cycle could spawn another
    summarizer without bound.
    """
    from ai_functions.ai_thread.config import ThreadConfig
    from ai_functions.ai_thread.summarization import (
        _build_dedicated_summarizer_template,
        _build_fork_summarizer_template,
    )

    # Fork path: inherits parent config but must force the guard on. Parent has
    # summarization on with a threshold set — neither may leak to the child.
    parent = ThreadConfig(
        structured_output=False,
        summarization_enabled=True,
        summarization_threshold=1_000,
    )
    fork = _build_fork_summarizer_template(parent)  # noqa: SLF001
    assert fork.config.summarization_enabled is False
    assert fork.config.summarization_threshold is None

    # Dedicated path: minimal config, also guarded.
    dedicated = _build_dedicated_summarizer_template(parent)  # noqa: SLF001
    assert dedicated.config.summarization_enabled is False
    assert dedicated.config.summarization_threshold is None


async def test_summarization_disabled_reraises_overflow() -> None:
    """With summarization_enabled=False, a ContextWindowOverflowException propagates.

    The thread fails loudly instead of compacting. This is both a standalone
    option (surface overflow rather than rewrite history) and the mechanism that
    makes summarizer helpers non-recursive.
    """
    # A model that always overflows. With summarization disabled there is no
    # reactive compaction, so the overflow must surface unchanged.
    always_overflow = _OverflowAlwaysModel()

    @ai_function[str](
        structured_output=False,
        model=cast(Any, always_overflow),  # pyright: ignore[reportExplicitAny]
        summarization_enabled=False,
    )
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        with pytest.raises(ContextWindowOverflowException):
            await handle.run("hello")


async def test_forked_summarizer_that_overflows_does_not_recurse() -> None:
    """A summarizer child that itself overflows re-raises — it never re-summarizes.

    Both the parent and the summarizer model overflow. Because the summarizer
    template disables summarization, the child raises instead of spawning a
    grandchild, and the parent surfaces a bounded SummarizationFailedError. A
    call counter proves only one summarizer thread was ever created.
    """
    summarizer_calls = 0

    class _ParentOverflowSummarizerOverflowModel(Model):
        """Parent call overflows once; every summarizer call also overflows."""

        def __init__(self) -> None:
            super().__init__()
            self._parent_raised = False

        def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
            pass

        def get_config(self) -> dict[str, object]:
            return {}

        def stream(
            self,
            messages: Messages,
            tool_specs: list[ToolSpec] | None = None,
            system_prompt: str | None = None,
            *,
            tool_choice: ToolChoice | None = None,
            system_prompt_content: list[SystemContentBlock] | None = None,
            invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
            **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
        ) -> AsyncIterable[StreamEvent]:
            del tool_specs, tool_choice, system_prompt_content, invocation_state, kwargs
            nonlocal summarizer_calls
            if system_prompt and "summarization" in system_prompt.lower():
                summarizer_calls += 1
                raise ContextWindowOverflowException("summarizer input also too long")
            if not self._parent_raised:
                self._parent_raised = True
                raise ContextWindowOverflowException("parent overflow triggers summarization")
            from ai_functions.testing.scripted_model import _stream_turn  # noqa: PLC0415

            return _stream_turn(Turn(text="unreachable"))

        def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
            del args, kwargs
            raise NotImplementedError

    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=False,  # dedicated path routes by summarizer system prompt
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=100_000,
    )

    @ai_function[str](
        structured_output=False,
        model=cast(Any, _ParentOverflowSummarizerOverflowModel()),  # pyright: ignore[reportExplicitAny]
        summarization_strategy=strategy,
    )
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        await handle.notify("seed 1")
        await handle.notify("seed 2")
        # Parent overflows → summarize → summarizer overflows → child re-raises
        # (no recursion) → parent wraps as SummarizationFailedError. Bounded.
        with pytest.raises(SummarizationFailedError):
            await handle.run("hello")

    # The summarizer thread ran and overflowed, but did NOT spawn another
    # summarizer: exactly one summarizer model call, not an unbounded chain.
    assert summarizer_calls == 1


# ── End-to-end: reactive summarization on overflow ────────────────────────────


class _OverflowAlwaysModel(Model):
    """Model that raises ``ContextWindowOverflowException`` on every call."""

    def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        pass

    def get_config(self) -> dict[str, object]:
        return {}

    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> AsyncIterable[StreamEvent]:
        del messages, tool_specs, system_prompt, tool_choice
        del system_prompt_content, invocation_state, kwargs
        raise ContextWindowOverflowException("simulated overflow: input is too long")

    def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
        del args, kwargs
        raise NotImplementedError


class _OverflowOnceModel(Model):
    """Scripted-like model that raises ``ContextWindowOverflowException`` exactly once.

    Subsequent calls play back a canned turn.
    """

    def __init__(self, after_overflow_turn: Turn) -> None:
        super().__init__()
        self._raised: bool = False
        self._after = after_overflow_turn

    def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        pass

    def get_config(self) -> dict[str, object]:
        return {}

    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> AsyncIterable[StreamEvent]:
        del tool_specs, system_prompt, tool_choice, system_prompt_content, invocation_state, kwargs
        if not self._raised:
            self._raised = True
            raise ContextWindowOverflowException("simulated overflow: input is too long")
        from ai_functions.testing.scripted_model import _stream_turn  # noqa: PLC0415

        return _stream_turn(self._after)

    def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
        del args, kwargs
        raise NotImplementedError


async def test_summarization_flow_emits_context_summarized_event() -> None:
    """A thread that overflows summarizes (dedicated path) and emits ContextSummarizedEvent.

    We wire a composite model that routes by ``system_prompt``:
    - when the prompt is the dedicated summarizer's system prompt → emit
      "<summary>SUM</summary>";
    - otherwise → overflow on the first call, emit the final answer on the
      second call.

    This isolates the dedicated-summarization path so we can verify the end-to-end
    flow (overflow → strategy.summarize → ContextSummarizedEvent → retry → success).
    """
    parent_model = _OverflowOnceModel(after_overflow_turn=Turn(text="final-answer-after-summary"))
    summary_turn = Turn(text="<summary>SUM</summary>")

    from ai_functions.testing.scripted_model import _stream_turn

    def summary_turn_stream() -> AsyncIterable[StreamEvent]:
        return _stream_turn(summary_turn)

    class _CompositeModel(Model):
        def __init__(self) -> None:
            super().__init__()

        def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
            pass

        def get_config(self) -> dict[str, object]:
            return {}

        def stream(
            self,
            messages: Messages,
            tool_specs: list[ToolSpec] | None = None,
            system_prompt: str | None = None,
            *,
            tool_choice: ToolChoice | None = None,
            system_prompt_content: list[SystemContentBlock] | None = None,
            invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
            **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
        ) -> AsyncIterable[StreamEvent]:
            if system_prompt and "summarization" in system_prompt.lower():
                return summary_turn_stream()
            return parent_model.stream(
                messages,
                tool_specs,
                system_prompt,
                tool_choice=tool_choice,
                system_prompt_content=system_prompt_content,
                invocation_state=invocation_state,
                **kwargs,
            )

        def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
            del args, kwargs
            raise NotImplementedError

    model = _CompositeModel()

    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=False,  # dedicated path so the summarizer system_prompt route fires
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=100_000,
    )

    @ai_function[str](
        structured_output=False,
        model=cast(Any, model),  # pyright: ignore[reportExplicitAny]
        summarization_strategy=strategy,
    )
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        # Seed pre-existing user turns so the strategy has events to summarize.
        await handle.notify("seed 1")
        await handle.notify("seed 2")
        result = await handle.run("hello")
        assert result.strip() == "final-answer-after-summary"

        events = await h.worker.coordinator.get_events(handle.id)
        summaries = [e for e in events if isinstance(e, ContextSummarizedEvent)]
        assert len(summaries) == 1
        # The first event in new_history should be the MessageUserEvent
        # carrying the extracted summary text.
        new_history = summaries[0].new_history
        assert len(new_history) >= 1
        first = new_history[0]
        assert isinstance(first, MessageUserEvent)
        assert first.text == "SUM"


async def test_proactive_summarization_fires_on_token_threshold() -> None:
    """summarization_threshold compacts accumulated history BEFORE a model call.

    Proactive summarization runs at cycle entry over the logged history. A big
    turn is seeded via ``notify`` and logged by the first ``run``; the second
    ``run`` enters above the tiny threshold and compacts proactively, summarizing
    the big older turn while preserving the small recent tail. No
    ``ContextWindowOverflowException`` is involved.
    """
    big_older_turn = "word " * 400  # ~570 tokens, becomes an older logged turn
    first_answer = Turn(text="first-answer")
    short_answer = Turn(text="second-answer")
    summary_turn = Turn(text="<summary>COMPACTED</summary>")

    from ai_functions.testing.scripted_model import _stream_turn

    class _CompositeModel(Model):
        def __init__(self) -> None:
            super().__init__()
            self._answers = 0

        def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
            pass

        def get_config(self) -> dict[str, object]:
            return {}

        def stream(
            self,
            messages: Messages,
            tool_specs: list[ToolSpec] | None = None,
            system_prompt: str | None = None,
            *,
            tool_choice: ToolChoice | None = None,
            system_prompt_content: list[SystemContentBlock] | None = None,
            invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
            **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
        ) -> AsyncIterable[StreamEvent]:
            del tool_specs, tool_choice, system_prompt_content, invocation_state, kwargs
            if system_prompt and "summarization" in system_prompt.lower():
                return _stream_turn(summary_turn)
            self._answers += 1
            return _stream_turn(first_answer if self._answers == 1 else short_answer)

        def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
            del args, kwargs
            raise NotImplementedError

    # min_messages=1 preserves only the most-recent (small) turn; the big older
    # turn falls into the summarized prefix. preserve_max_tokens sits above the
    # small tail and below the big turn so ONE compaction drops the history
    # under the threshold.
    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=False,
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=200,
    )

    @ai_function[str](
        structured_output=False,
        model=cast(Any, _CompositeModel()),  # pyright: ignore[reportExplicitAny]
        summarization_strategy=strategy,
        summarization_threshold=100,  # the big older turn (~570 tokens) will exceed this
    )
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        # Seed the big turn as OLDER content: it is drained + logged by run 1,
        # then summarized when run 2 enters above the threshold.
        await handle.notify(big_older_turn)
        first = await handle.run("start")  # logs [user(big), user("start"), assistant(first)]
        assert first.strip() == "first-answer"

        events_after_first = await h.worker.coordinator.get_events(handle.id)
        assert not [e for e in events_after_first if isinstance(e, ContextSummarizedEvent)]

        second = await handle.run("continue")  # enters with a large history → proactive compaction
        assert second.strip() == "second-answer"

        events = await h.worker.coordinator.get_events(handle.id)
        summaries = [e for e in events if isinstance(e, ContextSummarizedEvent)]
        assert len(summaries) == 1  # compacted proactively, before the second answer
        assert isinstance(summaries[0].new_history[0], MessageUserEvent)
        assert summaries[0].new_history[0].text == "COMPACTED"


async def test_no_proactive_summarization_when_threshold_unset() -> None:
    """With summarization_threshold=None (default), a large history is NOT compacted."""
    answer_turn = Turn(text="plain-answer")

    from ai_functions.testing.scripted_model import _stream_turn

    class _AnswerModel(Model):
        def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
            pass

        def get_config(self) -> dict[str, object]:
            return {}

        def stream(
            self,
            messages: Messages,
            tool_specs: list[ToolSpec] | None = None,
            system_prompt: str | None = None,
            *,
            tool_choice: ToolChoice | None = None,
            system_prompt_content: list[SystemContentBlock] | None = None,
            invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
            **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
        ) -> AsyncIterable[StreamEvent]:
            del tool_specs, system_prompt, tool_choice, system_prompt_content, invocation_state, kwargs
            return _stream_turn(answer_turn)

        def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
            del args, kwargs
            raise NotImplementedError

    @ai_function[str](structured_output=False, model=cast(Any, _AnswerModel()))  # pyright: ignore[reportExplicitAny]
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        await handle.notify("x " * 500)
        result = await handle.run("continue")
        assert result.strip() == "plain-answer"

        events = await h.worker.coordinator.get_events(handle.id)
        summaries = [e for e in events if isinstance(e, ContextSummarizedEvent)]
        assert len(summaries) == 0  # default is reactive-only: no proactive compaction


# ── Token usage is emitted even when a cycle raises before returning ──

_USAGE_METADATA: StreamEvent = cast(
    StreamEvent,
    {
        "metadata": {
            "usage": {
                "inputTokens": 200,
                "outputTokens": 777,
                "totalTokens": 977,
                "cacheReadInputTokens": 150,
                "cacheWriteInputTokens": 50,
            },
            "metrics": {"latencyMs": 0},
        }
    },
)


class _MaxTokensModel(Model):
    """Streams a real turn with usage metadata, then stops with ``max_tokens``."""

    def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        pass

    def get_config(self) -> dict[str, object]:
        return {}

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[StreamEvent]:  # pyright: ignore[reportExplicitAny]
        del args, kwargs

        async def _stream() -> AsyncIterable[StreamEvent]:
            yield cast(StreamEvent, {"messageStart": {"role": "assistant"}})
            yield cast(StreamEvent, {"contentBlockStart": {"start": {}}})
            yield cast(StreamEvent, {"contentBlockDelta": {"delta": {"text": "thinking..."}}})
            yield cast(StreamEvent, {"contentBlockStop": {}})
            yield cast(StreamEvent, {"messageStop": {"stopReason": "max_tokens"}})
            yield _USAGE_METADATA

        return _stream()

    def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
        del args, kwargs
        raise NotImplementedError


def _only_usage_event(events: list[Event]) -> TokenUsageEvent:
    usage_events = [e for e in events if isinstance(e, TokenUsageEvent)]
    assert len(usage_events) == 1, f"expected exactly one TokenUsageEvent, got {len(usage_events)}"
    return usage_events[0]


async def test_max_tokens_attempt_emits_token_usage_including_cache() -> None:
    """A max-tokens-terminated cycle still emits its token usage."""

    @ai_function[str](structured_output=False, model=cast(Any, _MaxTokensModel()))
    def _ask(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(_ask)
        with pytest.raises(MaxTokensReachedException):
            await handle.run("solve this")
        events = await h.worker.coordinator.get_events(handle.id)

    usage = _only_usage_event(events).token_usage
    assert usage.input_tokens == 200
    assert usage.output_tokens == 777
    assert usage.cache_read_tokens == 150
    assert usage.cache_write_tokens == 50
