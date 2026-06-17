"""Scripted ``strands.Model`` implementation for deterministic agent tests.

``ScriptedModel`` plays back a fixed sequence of model turns. Each ``Turn`` describes
one model call: streamed text chunks followed by optional tool calls. Tests drive
agent behaviour by writing a small script; streaming is real (chunks are yielded
between ``await`` points) so concurrency tests can observe partial state.

Barriers (``await_before``/``await_after`` on a ``Turn``, and ``AwaitBarrier``
sentinels inside ``text_chunks``) suspend the stream until ``RuntimeHarness.release``
is called. The connection between model and harness is carried by a ``ContextVar``
set in ``RuntimeHarness.__aenter__`` — tests don't attach models explicitly."""

from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any, final

from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


@final
@dataclass(frozen=True)
class AwaitBarrier:
    """Sentinel placed inside ``Turn.text_chunks`` to suspend streaming mid-turn.

    When the scripted model reaches this sentinel it yields every chunk emitted so
    far, then awaits the harness barrier named ``name``. The next chunk after the
    sentinel is only streamed once ``RuntimeHarness.release(name)`` is called.
    """

    name: str
    """Barrier name; matched against ``RuntimeHarness.release`` calls."""


@final
@dataclass(frozen=True)
class Turn:
    """One scripted model call.

    A ``Turn`` describes what the model will emit on one invocation of
    ``Model.stream``: zero-or-more text chunks, then zero-or-more tool calls. Stop
    reason is ``"tool_use"`` if ``tool_calls`` is non-empty, ``"end_turn"`` otherwise.

    Exactly one of ``text`` or ``text_chunks`` may be provided (or neither,
    if the turn is a pure tool call). When ``text`` is given, ``ScriptedModel``
    splits it into word-sized chunks automatically; when ``text_chunks`` is given,
    each entry becomes one streamed chunk verbatim. A mid-stream ``AwaitBarrier``
    must appear inside ``text_chunks``.
    """

    text: str | None = None
    """Plain text to stream; automatically split on whitespace into word
    chunks."""

    text_chunks: tuple[str | AwaitBarrier, ...] | None = None
    """Explicit chunk list; each entry is one ``contentBlockDelta`` (or a
    barrier)."""

    tool_calls: tuple[tuple[str, dict[str, object]], ...] = ()
    """``(tool_name, tool_input)`` pairs emitted as tool-use blocks after
    text."""

    await_before: str | None = None
    """Barrier name blocked on at the start of this turn, before any chunk
    is yielded."""

    await_after: str | None = None
    """Barrier name blocked on at the end of this turn, after
    ``messageStop``."""

    input_tokens: int = 0
    """Reported in the terminal ``metadata.usage.inputTokens`` for this turn."""

    output_tokens: int = 0
    """Reported in the terminal ``metadata.usage.outputTokens`` for this turn."""

    def __post_init__(self) -> None:
        """Reject ambiguous or empty configurations.

        Raises:
            ValueError: Both ``text`` and ``text_chunks`` are set, or the turn has no
                text and no tool calls.
        """
        ...


class ScriptExhausted(RuntimeError):
    """Raised when the agent requests more model calls than the script provides."""


@final
class ScriptedModel(Model):
    """Deterministic ``strands.Model`` that plays back a ``list[Turn]``.

    The script is fixed at construction; per-call state (cursor) is mutable.

    Args:
        turns: One ``Turn`` per expected model call, in order.

    Ensures:
        ``self.remaining_turns == len(turns)``.
    """

    def __init__(self, turns: list[Turn]) -> None: ...

    @property
    def remaining_turns(self) -> int:
        """Turns not yet consumed by a ``stream`` call."""
        ...

    def update_config(self, **model_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        """No-op for compatibility with ``strands.Model``.

        Args:
            model_config: Ignored; ``ScriptedModel`` has no mutable config.
        """
        ...

    def get_config(self) -> dict[str, object]:
        """Return a snapshot of the model's state.

        Returns:
            A dict with ``{"remaining_turns": int}``.
        """
        ...

    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> AsyncIterable[StreamEvent]:
        """Yield the Bedrock-shaped stream for the next scripted turn.

        Args:
            messages: Conversation history; not consulted by ``ScriptedModel``.
            tool_specs: Available tool specs; ignored (scripted turns name their target
                tool directly).
            system_prompt: Ignored.
            tool_choice: Ignored.
            system_prompt_content: Ignored.
            invocation_state: Ignored.
            kwargs: Ignored.

        Returns:
            An async iterable that yields one ``messageStart``, the turn's text and
            tool-use content blocks (with barriers honored), then ``messageStop`` and
            ``metadata``.

        Raises:
            ScriptExhausted: The script has no more turns.
        """
        ...

    def structured_output(
        self,
        output_model: type[BaseModel],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> AsyncGenerator[dict[str, BaseModel | Any]]:  # pyright: ignore[reportExplicitAny]
        """Unsupported path — ``ScriptedModel`` drives agents via ``stream`` only.

        Args:
            output_model: Ignored.
            prompt: Ignored.
            system_prompt: Ignored.
            kwargs: Ignored.

        Raises:
            NotImplementedError: Always; tests should rely on tool-call turns targeting
                the structured-output wrapper instead.
        """
        ...
