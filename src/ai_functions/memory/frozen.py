"""Marker for memory fields that should not receive gradients by default."""

from typing import Annotated, TypeVar

T = TypeVar("T")


class FrozenMarker:
    """Tag that identifies Frozen-typed fields (requires_grad=False by default)."""


Frozen = Annotated[T, FrozenMarker()]
