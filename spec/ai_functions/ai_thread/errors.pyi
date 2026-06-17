"""Error types raised by AI function execution."""

from __future__ import annotations


class AIFunctionError(Exception):
    """Base error for AI function execution failures.

    Args:
        message: Human-readable explanation.
        function_name: Name of the ``AIFunction`` that raised.
    """

    function_name: str

    def __init__(self, message: str, function_name: str = "") -> None: ...


class ValidationError(AIFunctionError):
    """Post-condition validation failed for a cycle's result.

    Args:
        function_name: Name of the ``AIFunction`` whose result failed.
        errors: Post-condition name → failure message.

    Ensures:
        The formatted message joins ``errors`` as ``"k: v; ..."``.
    """

    validation_errors: dict[str, str]

    def __init__(self, function_name: str, errors: dict[str, str]) -> None: ...
