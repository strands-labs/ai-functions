"""Marker for memory fields that should not receive gradients by default."""

from typing import Annotated, TypeVar

T = TypeVar("T")


class FrozenMarker:
    """Tag that identifies Frozen-typed fields (requires_grad=False by default)."""


Frozen = Annotated[T, FrozenMarker()]
"""Annotate a memory field as frozen: recalled into prompts, never optimized.

A field typed ``Frozen[T]`` is read like any other parameter but defaults to
``requires_grad=False``, so the optimizer's ``consolidate`` never mutates it.
"""
