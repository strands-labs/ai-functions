"""Post-condition validator types."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field


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
