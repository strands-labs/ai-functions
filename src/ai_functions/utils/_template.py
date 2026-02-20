"""Template utilities for AI Functions.

This module provides template string handling with support for Python 3.10-3.14+.
On Python 3.14+, uses native t-string types from string.templatelib.
On earlier versions, provides a compatible backport implementation.
"""

import string
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

if sys.version_info >= (3, 14):
    # Use native t-string types from Python 3.14+
    from string.templatelib import Interpolation, Template
else:
    # Backport implementation for Python 3.10-3.13

    @dataclass
    class Interpolation:
        """Represents an interpolated value in a template."""

        value: Any
        expr: str
        conv: str | None = None

    class Template(tuple):
        """A template containing strings and interpolations."""

        def __new__(cls, *args: str | Interpolation) -> "Template":
            return super().__new__(cls, args)


# Reuse a single Formatter instance for parsing
_formatter = string.Formatter()


def generate_template(template_str: str, context: dict[str, Any], use_eval: bool = False) -> str:
    """Substitute {placeholders} in template string with values from context.

    Reference from tstr lib: https://github.com/ilotoki0804/tstr

    Args:
        template_str: String with {placeholder} syntax
        context: Dictionary of variable names to values
        use_eval: If True, evaluate expressions; if False, simple lookup only

    Returns:
        Rendered string with placeholders substituted.
    """
    parts: list[str] = []

    for literal, field_name, format_spec, conversion in _formatter.parse(template_str):
        # Add the literal text
        parts.append(literal)

        # If there's a field to substitute
        if field_name is not None:
            # Try simple lookup first
            if field_name in context:
                value = context[field_name]
            elif use_eval:
                # Evaluate as expression
                try:
                    value = eval(field_name, {"__builtins__": {}}, context)
                except Exception:
                    # Keep original placeholder on eval failure
                    parts.append("{" + field_name + "}")
                    continue
            else:
                # Keep original placeholder if not found and not using eval
                parts.append("{" + field_name + "}")
                continue

            # Apply conversion if specified
            if conversion == "s":
                value = str(value)
            elif conversion == "r":
                value = repr(value)
            elif conversion == "a":
                value = ascii(value)

            # Apply format spec if specified
            if format_spec:
                value = format(value, format_spec)
            else:
                value = str(value)

            parts.append(value)

    return "".join(parts)


def render_template_with_indent(template: Template) -> str:
    """Render a template while preserving indentation for interpolated values.

    Dedents the template, then renders values while maintaining the indentation
    level of their placeholder.

    Args:
        template: A Template object containing strings and Interpolations

    Returns:
        The rendered string with proper indentation
    """
    # Build temp string with placeholders, collect values
    parts: list[str] = []
    values: list[Any] = []
    for item in template:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Interpolation):
            parts.append(f"__INTERP_{len(values)}__")
            values.append(item.value)

    temp_str = "".join(parts)
    result = textwrap.dedent(temp_str).strip("\n")
    lines = result.split("\n")

    # Replace each interpolation placeholder while preserving indentation
    for idx, value in enumerate(values):
        placeholder = f"__INTERP_{idx}__"
        str_value = str(value)

        for i, line in enumerate(lines):
            indent_len = _count_leading_spaces_to_match(line, placeholder)
            if indent_len is None:
                continue
            elif indent_len == 0:
                lines[i] = line.replace(placeholder, str_value)
            else:
                indented_value = textwrap.indent(str_value, " " * indent_len)
                lines[i] = line[indent_len:].replace(placeholder, indented_value)

    return "\n".join(lines)


def _count_leading_spaces_to_match(string: str, substring: str) -> int | None:
    """Count leading spaces before a substring match.

    If there are only spaces from the start of the string to the match,
    returns the count of those spaces. Otherwise returns 0.

    Args:
        string: The string to search in
        substring: The substring to find

    Returns:
        Number of leading spaces, 0 if non-space chars precede match, None if not found
    """
    index = string.find(substring)
    if index == -1:
        return None

    before_match = string[:index]
    if before_match == " " * len(before_match):
        return len(before_match)
    else:
        return 0
