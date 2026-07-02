"""Marker for memory fields that store reusable Python code."""

import ast
import textwrap
from typing import Annotated

from pydantic import AfterValidator, Field


class ProceduralMarker:
    """Tag that identifies Procedural-typed fields."""


def validate_procedural(value: str) -> str:
    """Validate that a procedural memory value contains parseable Python."""
    if not value.strip():
        return value
    ast.parse(value)
    return value


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
