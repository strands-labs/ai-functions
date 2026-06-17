"""Pluggable conversation-summarization strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Event, RenderableEvent, ThreadContext
from .config import ThreadConfig
from .errors import AIFunctionError


@runtime_checkable
class SummarizationStrategy(Protocol):
    """Produce a compacted history for a thread whose context is too long.

    Strategies are the pluggable decision layer for context management: they choose
    which model to call, what to preserve verbatim, and what to fold into a synthetic
    summary. Built-in implementations live in :mod:`ai_functions.summarization`; user code
    may supply its own. A thread selects its strategy via
    :attr:`ThreadConfig.summarization_strategy`.
    """

    async def summarize(
        self,
        events: list[Event],
        ctx: ThreadContext,
        cycle_config: ThreadConfig,
    ) -> list[RenderableEvent]:
        """Compact ``events`` into a shorter synthetic history.

        Invoked by the thread when its accumulated event log exceeds the configured
        context bounds. The returned list is appended as the ``new_history`` payload
        of a :class:`ContextSummarizedEvent`; on the next reconstruction every event
        before that marker is dropped and the returned list is rendered in its place.

        Args:
            events: Full event log at the moment summarization was triggered.
            ctx: Per-cycle context of the thread requesting summarization.
            cycle_config: Resolved cycle config the parent thread was using when the
                overflow occurred.

        Returns:
            A synthetic sequence of renderable events representing the
            compacted history.

        Requires:
            ``events`` is sorted by append order (oldest first).

        Ensures: - The first event in the returned list (when present) has user-turn
        semantics on render. - Every ``toolUse`` block reachable through the returned
        events has a matching ``toolResult`` event later in the list, or will be
        healed by :func:`reconstruct_messages` (I10).

        Raises:
            SummarizationFailedError: No useful compaction could be produced.

        Concurrency:
            May perform I/O (model calls, runtime spawns). Must not emit events
            on the parent thread directly: the runtime is the sole emitter of the outer
            `ContextSummarizedEvent` tied to this summarization.
        """
        ...


class DefaultSummarizationStrategy:
    """Summarize a prefix via a model call, preserve a bounded tail.

    The built-in strategy. It chooses a tail of recent messages to keep verbatim (
    bounded below by ``preserve_min_messages`` / ``preserve_min_tokens`` and above by
    ``preserve_max_tokens``), advances the split to a legal tool-pair boundary (I10),
    and replaces everything before the split with a single synthetic user turn
    carrying a narrative summary produced by a helper thread.

    The helper-thread path is governed by ``summarize_by_forking``:

    - ``True`` — the helper inherits the parent's resolved cycle config (same model,
      system prompt, tools; tool execution denied via an internal hook).
      Requires ``cycle_config.structured_output is False``.
    - ``False`` — the helper uses a minimal dedicated config: parent's
      model only, no tools, a hard-coded summarization system prompt.
    - ``None`` — resolves to ``True`` iff ``cycle_config.structured_output is False``,
      else ``False``. Resolved per call, so one instance is reusable across threads with
      different output shapes.

    Args:
        summarize_by_forking: Fork policy (see above). ``None`` resolves lazily per call
            based on ``cycle_config``.
        preserve_min_messages: Floor on the number of tail messages kept verbatim.
        preserve_min_tokens: Floor on the number of tokens (estimated) kept verbatim.
        preserve_max_tokens: Ceiling on the number of tokens (estimated) kept verbatim.
            Messages that push the tail past this bound are summarized instead.

    Requires:
        - ``preserve_min_messages >= 1``.
        - ``preserve_min_tokens <= preserve_max_tokens``.

    Raises:
        ValueError: A ``Requires`` clause is violated.
    """

    def __init__(
        self,
        *,
        summarize_by_forking: bool | None = None,
        preserve_min_messages: int = 6,
        preserve_min_tokens: int = 4_000,
        preserve_max_tokens: int = 40_000,
    ) -> None: ...

    async def summarize(
        self,
        events: list[Event],
        ctx: ThreadContext,
        cycle_config: ThreadConfig,
    ) -> list[RenderableEvent]:
        """Produce the compacted history per the class docstring algorithm.

        See :meth:`SummarizationStrategy.summarize` for the full protocol contract (
        ordering requirement on ``events``, rendering guarantees on the returned
        list, concurrency rules). This implementation additionally guarantees the
        shape below.

        Args:
            events: Parent thread's full event log at summarization time.
            ctx: Parent thread's per-cycle context.
            cycle_config: Parent thread's resolved cycle config.

        Returns:
            ``[MessageUserEvent(summary_text), *preserved_events]``.

        Raises:
            SummarizationFailedError: The history cannot be compacted at the  configured
                bounds, the helper thread failed, or ``summarize_by_forking=True`` was
                requested against a structured-output parent with a non-``str`` output.
        """
        ...


class SummarizationFailedError(AIFunctionError):
    """Summarization could not produce a usable compaction.

    Raised when every available strategy attempt fails to fit within the model's
    context window, or when no legal split point exists (for instance, a single
    preserved message is itself larger than the context limit). Unrecoverable by
    retry; callers must intervene (reset session, change config, shrink tool outputs).

    Args:
        function_name: Name of the ``AIFunction`` whose thread attempted summarization.
        reason: Short explanation of why summarization could not succeed.
    """

    def __init__(self, function_name: str, reason: str) -> None: ...
