"""Test doubles for the demo's ``Judge`` protocol (``examples/skillify_demo/full_pipeline/quality.py``).

``StubJudge`` lets judged-metric tests inject canned verdicts without a live
model, so they can assert scoring/aggregation logic against known outputs.
Imported by the ``tests/test_example_skillify_demo_*.py`` modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from pydantic import BaseModel

VerdictT = TypeVar("VerdictT", bound=BaseModel)


class StubJudge:
    """Fake ``Judge`` returning canned verdicts keyed by schema type.

    Args:
        verdicts: Maps a verdict schema type to either a pre-built instance
            of that schema (returned as-is on every matching call) or a
            zero-arg factory producing one (called fresh each time — useful
            when a test wants a distinct instance, or randomized/incrementing
            values, per call).

    Ensures:
        ``self.calls`` records every ``(prompt, schema, system_prompt)``
        triple passed to ``score``, in call order, so tests can assert what
        a metric sent the judge (e.g. verbatim prompt-composition parity,
        or the exact per-metric system prompt) without a real model.
    """

    def __init__(self, verdicts: dict[type[BaseModel], BaseModel | Callable[[], BaseModel]]) -> None:
        self._verdicts = verdicts
        self.calls: list[tuple[str, type[BaseModel], str | None]] = []

    async def score(self, prompt: str, schema: type[VerdictT], *, system_prompt: str | None = None) -> VerdictT:
        """Return the canned verdict registered for ``schema``.

        Args:
            prompt: Ignored for scoring, but recorded in ``self.calls``.
            schema: Verdict type to look up.
            system_prompt: Ignored for scoring, but recorded in ``self.calls``.

        Returns:
            The registered instance, or a fresh one from the registered
            factory.

        Raises:
            KeyError: No verdict (or factory) is registered for ``schema``.
        """
        self.calls.append((prompt, schema, system_prompt))
        if schema not in self._verdicts:
            raise KeyError(f"StubJudge: no canned verdict registered for schema {schema!r}")
        entry = self._verdicts[schema]
        verdict = entry if isinstance(entry, schema) else entry()  # type: ignore[operator]
        # ``entry``'s static type is ``BaseModel | Callable[[], BaseModel]``; the
        # dict's construction contract (values match their key's schema) is a
        # runtime invariant pyright can't see through ``isinstance(entry, schema)``.
        return cast("VerdictT", verdict)
