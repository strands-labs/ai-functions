"""Prompt rendering: docstring templates and returned ``Template`` values render identically."""

from __future__ import annotations

import pytest
from tstr import Interpolation, Template

from ai_functions import ai_function
from ai_functions.ai_thread import AIFunctionError

EXPECTED = """\
Review 'report.md' (score 0.83).
Findings:
    - first
    - second
Inline: - first
- second
Signed,
  the reviewer"""


async def test_docstring_and_returned_template_render_identically() -> None:
    """Dedent, newline stripping, conversions, format specs, block indentation, and nested templates."""
    findings = "- first\n- second"
    signature = Template("\n    Signed,\n      ", Interpolation("the reviewer", "who"), "\n    ")

    @ai_function[str]
    def from_docstring(path: str, score: float, findings: str, signature: Template) -> None:
        """
        Review {path!r} (score {score:.2f}).
        Findings:
            {findings}
        Inline: {findings}
        {signature}
        """

    # A t-string on Python 3.14+; spelled with the constructor so the module parses on 3.12.
    @ai_function[str]
    def from_template(path: str, score: float, findings: str, signature: Template) -> Template:
        return Template(
            "\n        Review ",
            Interpolation(path, "path", "r"),
            " (score ",
            Interpolation(score, "score", None, ".2f"),
            ").\n        Findings:\n            ",
            Interpolation(findings, "findings"),
            "\n        Inline: ",
            Interpolation(findings, "findings"),
            "\n        ",
            Interpolation(signature, "signature"),
            "\n        ",
        )

    args = ("report.md", 0.8312, findings, signature)
    assert await from_docstring.render_prompt(*args) == EXPECTED
    assert await from_template.render_prompt(*args) == EXPECTED


async def test_unsupported_prompt_return_type_raises() -> None:
    """A prompt function returning anything but ``str``, ``Template``, or ``None`` is rejected."""

    @ai_function[str]
    def bad() -> int:
        return 42

    with pytest.raises(AIFunctionError, match="must return str, Template, or None, got int"):
        await bad.render_prompt()
