"""Marker for memory fields that store reusable Python code."""

import textwrap
from typing import Annotated

from pydantic import AfterValidator, Field


class ProceduralMarker:
    """Tag that identifies Procedural-typed fields."""


def validate_procedural(value: str) -> str:
    """Validate that a procedural memory value contains parseable Python.

    Args:
        value: The candidate code string.

    Returns:
        ``value`` unchanged.

    Raises:
        SyntaxError: ``value`` is non-empty and not parseable by
            :func:`ast.parse`.

    Ensures:
        - An empty / whitespace-only ``value`` is returned without parsing.
        - A non-empty ``value`` round-trips through ``ast.parse`` before
          being returned.
    """
    ...


Procedural = Annotated[
    str,
    ProceduralMarker(),
    AfterValidator(validate_procedural),
    Field(
        default="# No code yet.",
        description=textwrap.dedent("""\
            Python functions available in the agent's execution environment.
            Feedback to this parameter MUST include code snippets to add/modify.
        """),
    ),
]
"""Annotate a memory field as a reusable Python-code parameter.

A field typed ``Procedural`` stores Python source the agent can run. It is
optimized **as code**: ``consolidate`` feeds the current code plus feedback
to a code-merge AI function, and the result is validated with
:func:`validate_procedural`. Feedback for such a field must itself contain
code snippets.
"""
