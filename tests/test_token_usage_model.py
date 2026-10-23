"""A Strands thread names the model that spent the tokens on its ``TOKEN_USAGE`` event, so the usage can be priced.

No model runs: the thread gets a stub ``Model`` that reports usage metadata and, or not, a Bedrock ``model_id``.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, cast

from strands.models import Model
from strands.types.streaming import StreamEvent

from ai_functions import ai_function
from ai_functions.testing import RuntimeHarness
from ai_functions.types import TokenUsageEvent


class _NamedModel(Model):
    """Streams one short answer with usage metadata and reports a Bedrock ``model_id``."""

    def __init__(self, model_id: str | None) -> None:
        self._model_id = model_id

    def update_config(self, **_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        pass

    def get_config(self) -> dict[str, object]:
        return {} if self._model_id is None else {"model_id": self._model_id}

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[StreamEvent]:  # pyright: ignore[reportExplicitAny]
        del args, kwargs

        async def _stream() -> AsyncIterable[StreamEvent]:
            yield cast(StreamEvent, {"messageStart": {"role": "assistant"}})
            yield cast(StreamEvent, {"contentBlockStart": {"start": {}}})
            yield cast(StreamEvent, {"contentBlockDelta": {"delta": {"text": "done"}}})
            yield cast(StreamEvent, {"contentBlockStop": {}})
            yield cast(StreamEvent, {"messageStop": {"stopReason": "end_turn"}})
            yield cast(
                StreamEvent,
                {
                    "metadata": {
                        "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
                        "metrics": {"latencyMs": 0},
                    }
                },
            )

        return _stream()

    def structured_output(self, *args: object, **kwargs: object) -> Any:  # pyright: ignore[reportExplicitAny]
        del args, kwargs
        raise NotImplementedError


async def _usage_events_of(model: Model) -> list[TokenUsageEvent]:
    @ai_function[str](structured_output=False, model=cast(Any, model))
    def answer(q: str) -> str:
        return q

    async with RuntimeHarness() as h:
        handle = await h.spawn(answer)
        await handle.run("go")
        events = await h.coordinator.get_events(handle.id)
    return [e for e in events if isinstance(e, TokenUsageEvent)]


async def test_a_strands_thread_names_its_models_id() -> None:
    usage = await _usage_events_of(_NamedModel("global.anthropic.claude-opus-5-5"))
    assert [e.model for e in usage] == ["global.anthropic.claude-opus-5-5"]
    assert usage[0].token_usage.input_tokens == 12


async def test_a_strands_model_with_no_model_id_names_none() -> None:
    usage = await _usage_events_of(_NamedModel(None))
    assert [e.model for e in usage] == [None]
