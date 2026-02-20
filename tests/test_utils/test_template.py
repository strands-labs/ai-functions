"""Tests for template utilities.

Tests for Template, Interpolation, generate_template, render_template_with_indent,
and _count_leading_spaces_to_match functions.
"""

from ai_functions.utils._template import (
    Interpolation,
    Template,
    _count_leading_spaces_to_match,
    generate_template,
    render_template_with_indent,
)


class TestGenerateTemplate:
    """Tests for generate_template function."""

    def test_substitutes_simple_placeholder(self):
        """Simple {name} placeholder is substituted."""
        result = generate_template("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_substitutes_multiple_placeholders(self):
        """Multiple placeholders are all substituted."""
        result = generate_template("{a} + {b} = {c}", {"a": "1", "b": "2", "c": "3"})
        assert result == "1 + 2 = 3"

    def test_leaves_unknown_placeholder_unchanged(self):
        """Unknown placeholders are left as-is."""
        result = generate_template("Hello {name}!", {})
        assert result == "Hello {name}!"

    def test_eval_mode_evaluates_expressions(self):
        """With use_eval=True, expressions are evaluated."""
        result = generate_template("{x + y}", {"x": 1, "y": 2}, use_eval=True)
        assert result == "3"

    def test_eval_mode_handles_string_methods(self):
        """With use_eval=True, string methods work."""
        result = generate_template("{name.upper()}", {"name": "hello"}, use_eval=True)
        assert result == "HELLO"

    def test_eval_mode_leaves_invalid_expr_unchanged(self):
        """Invalid expressions are left unchanged in eval mode."""
        result = generate_template("{undefined_var}", {}, use_eval=True)
        assert result == "{undefined_var}"


class TestRenderTemplateWithIndent:
    """Tests for indentation preservation in template rendering.

    Tests that render_template_with_indent correctly preserves indentation
    when interpolating multiline values into templates.
    """

    def test_preserves_indentation_for_multiline_code(self):
        """Multiline code value gets indentation applied to subsequent lines."""
        code = "def foo():\n    return 42"
        template = Template("Code:\n    ", Interpolation(code, "code", None))
        result = render_template_with_indent(template)

        expected = "Code:\n    def foo():\n        return 42"
        assert result == expected

    def test_complex_nested_code_indentation(self):
        """Complex nested code maintains relative indentation."""
        code = """def outer():
    def inner():
        return 1
    return inner()"""

        template = Template("Execute:\n  ", Interpolation(code, "code", None))
        result = render_template_with_indent(template)

        lines = result.split("\n")
        assert lines[0] == "Execute:"
        assert lines[1] == "  def outer():"
        assert lines[2] == "      def inner():"
        assert lines[3] == "          return 1"
        assert lines[4] == "      return inner()"

    def test_dedents_template(self):
        """Template with common indentation is dedented."""
        template = Template(
            "\n        Process this:\n        ",
            Interpolation("value", "x", None),
            "\n        ",
        )
        result = render_template_with_indent(template)

        assert "        Process" not in result
        assert "Process this:" in result

    def test_handles_multiple_interpolations(self):
        """Multiple interpolations are all rendered correctly."""
        template = Template(
            "A: ",
            Interpolation("first", "a", None),
            "\nB: ",
            Interpolation("second", "b", None),
        )
        result = render_template_with_indent(template)

        assert "A: first" in result
        assert "B: second" in result

    def test_multiline_value_empty_lines_not_indented(self):
        """Empty lines in multiline values are not indented (textwrap.indent behavior)."""
        code = "line1\n\nline3"
        template = Template("Code:\n    ", Interpolation(code, "code", None))
        result = render_template_with_indent(template)

        lines = result.split("\n")
        assert lines[0] == "Code:"
        assert lines[1] == "    line1"
        assert lines[2] == ""  # Empty line not indented
        assert lines[3] == "    line3"


class TestCountLeadingSpacesToMatch:
    """Tests for _count_leading_spaces_to_match helper function."""

    def test_returns_none_when_substring_not_found(self):
        """Returns None when substring is not in string."""
        result = _count_leading_spaces_to_match("hello world", "xyz")
        assert result is None

    def test_returns_zero_when_no_leading_spaces(self):
        """Returns 0 when substring is at start of string."""
        result = _count_leading_spaces_to_match("hello world", "hello")
        assert result == 0

    def test_returns_space_count_when_only_spaces_before(self):
        """Returns count of spaces when only spaces precede the match."""
        result = _count_leading_spaces_to_match("    hello", "hello")
        assert result == 4

    def test_returns_zero_when_non_space_chars_before(self):
        """Returns 0 when non-space characters precede the match."""
        result = _count_leading_spaces_to_match("prefix hello", "hello")
        assert result == 0

    def test_returns_zero_when_mixed_chars_before(self):
        """Returns 0 when mix of spaces and other chars precede match."""
        result = _count_leading_spaces_to_match("  prefix hello", "hello")
        assert result == 0


class TestGenerateTemplateEdgeCases:
    """Tests for edge cases in generate_template using string.Formatter.parse()."""

    def test_escaped_braces_become_literal(self):
        """Escaped braces {{}} become literal braces."""
        result = generate_template("Use {{braces}} for literals", {})
        assert result == "Use {braces} for literals"

    def test_format_spec_applied(self):
        """Format specifications are applied to values."""
        result = generate_template("Pi is {pi:.2f}", {"pi": 3.14159})
        assert result == "Pi is 3.14"

    def test_format_spec_with_width(self):
        """Width format spec works correctly."""
        result = generate_template("Value: {x:>5}", {"x": 42})
        assert result == "Value:    42"

    def test_conversion_str(self):
        """!s conversion calls str()."""
        result = generate_template("Value: {x!s}", {"x": 123})
        assert result == "Value: 123"

    def test_conversion_repr(self):
        """!r conversion calls repr()."""
        result = generate_template("Value: {x!r}", {"x": "hello"})
        assert result == "Value: 'hello'"

    def test_conversion_ascii(self):
        """!a conversion calls ascii()."""
        result = generate_template("Value: {x!a}", {"x": "héllo"})
        assert result == "Value: 'h\\xe9llo'"

    def test_mixed_escaped_and_placeholders(self):
        """Mix of escaped braces and placeholders works."""
        result = generate_template("{{literal}} and {var}", {"var": "value"})
        assert result == "{literal} and value"

    def test_format_spec_with_conversion(self):
        """Format spec combined with conversion works."""
        result = generate_template("{x!s:>10}", {"x": 42})
        assert result == "        42"

    def test_empty_placeholder_left_unchanged(self):
        """Empty placeholder {} is left unchanged."""
        result = generate_template("Empty: {}", {})
        assert result == "Empty: {}"

    def test_nested_attribute_with_eval(self):
        """Nested attribute access works with eval."""

        class Obj:
            value = 42

        result = generate_template("Value: {obj.value}", {"obj": Obj()}, use_eval=True)
        assert result == "Value: 42"

    def test_list_index_with_eval(self):
        """List indexing works with eval."""
        result = generate_template("First: {items[0]}", {"items": ["a", "b", "c"]}, use_eval=True)
        assert result == "First: a"
