"""Offline tests for ``params/reducers.py``: each reducer's arithmetic, the escape hatch, and the defaults.

No reducer here makes a model call: every test runs with no model in scope at all, so a reducer that
reached for one would fail on import, not on a mock.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ai_functions.experimental.params import (
    NUMERIC_ADD,
    ORDERED,
    CallableReducer,
    FeedbackReducer,
    Local,
    ParamWiringError,
    ReducerError,
)
from ai_functions.types.graph import GradFeedback


def fb(*texts: str) -> list[GradFeedback]:
    return [GradFeedback(text=t, score=0.5) for t in texts]


# ── ORDERED ──


def test_ordered_appends_in_order_one_line_each() -> None:
    assert ORDERED.reduce("base", fb("a", "b")) == "base\na\nb"


def test_ordered_dedupes_verbatim_repeats_against_the_value_and_within_the_batch() -> None:
    assert ORDERED.reduce("base\na", fb("a", "b", "b", " b ")) == "base\na\nb"


def test_ordered_on_an_empty_value_starts_clean() -> None:
    assert ORDERED.reduce("", fb("a")) == "a"


def test_ordered_appends_items_to_a_list_value() -> None:
    assert ORDERED.reduce(["x"], fb("y", "x", "z")) == ["x", "y", "z"]


def test_ordered_with_no_feedback_returns_the_value_unchanged() -> None:
    assert ORDERED.reduce("base", []) == "base"


def test_ordered_dedupes_a_multi_line_text_as_one_unit() -> None:
    """TextGrad gradients are prose; a two-line gradient applied twice is appended once, not per line."""
    gradient = "Cite sources.\nPrefer primary ones."
    once = ORDERED.reduce("Find facts.", fb(gradient))
    assert once == "Find facts.\nCite sources.\nPrefer primary ones."
    assert ORDERED.reduce(once, fb(gradient)) == once
    assert ORDERED.reduce("Find facts.", fb(gradient, gradient)) == once


def test_ordered_keeps_the_current_value_verbatim_blank_lines_included() -> None:
    value = "Rules:\n\n- be brief\n\n- cite"
    assert ORDERED.reduce(value, fb("- be kind")) == "Rules:\n\n- be brief\n\n- cite\n- be kind"


def test_ordered_returns_the_value_itself_when_every_feedback_is_a_repeat() -> None:
    value = "Rules:\n\n- be brief"
    result = ORDERED.reduce(value, fb("- be brief", "Rules:"))
    assert result is value
    items = ["x", "y"]
    assert ORDERED.reduce(items, fb("x", " y ")) == items


def test_ordered_a_run_of_lines_already_present_in_order_is_a_repeat_but_a_reordered_one_is_not() -> None:
    assert ORDERED.reduce("a\nb\nc", fb("b\nc")) == "a\nb\nc"
    assert ORDERED.reduce("a\nb\nc", fb("c\nb")) == "a\nb\nc\nc\nb"


def test_ordered_refuses_a_value_it_cannot_append_to() -> None:
    with pytest.raises(ReducerError, match=r"ordered.*float"):
        ORDERED.reduce(1.5, fb("a"))


# ── NUMERIC_ADD ──


def test_numeric_add_sums_scalars() -> None:
    assert NUMERIC_ADD.reduce(1.0, fb("0.5", "-0.25")) == pytest.approx(1.25)
    assert NUMERIC_ADD.reduce(2, fb("3")) == 5


def test_numeric_add_is_pointwise_over_a_vector() -> None:
    assert NUMERIC_ADD.reduce([1.0, 0.5], fb("[0.1, 0.2]", "[0.0, -0.5]")) == pytest.approx([1.1, 0.2])


def test_numeric_add_refuses_a_length_mismatch() -> None:
    with pytest.raises(ReducerError, match=r"2.*3"):
        NUMERIC_ADD.reduce([1.0, 0.5], fb("[1, 2, 3]"))


def test_numeric_add_refuses_feedback_that_is_not_a_number() -> None:
    with pytest.raises(ReducerError, match=r"numeric_add.*'be nicer'"):
        NUMERIC_ADD.reduce(1.0, fb("be nicer"))


def test_numeric_add_refuses_a_non_numeric_value() -> None:
    with pytest.raises(ReducerError, match=r"numeric_add.*str"):
        NUMERIC_ADD.reduce("text", fb("1"))


def test_numeric_add_with_no_feedback_returns_the_value_unchanged() -> None:
    assert NUMERIC_ADD.reduce([1.0, 2.0], []) == [1.0, 2.0]


# ── TextGrad's bullet markers (FR5) ──


def test_ordered_removes_the_add_marker_textgrad_writes() -> None:
    """The backward prompt asks for ``- add: <text>``; the marker is format, so only the text is appended."""
    assert ORDERED.reduce("base", fb("- add: Cite sources.")) == "base\nCite sources."
    assert ORDERED.reduce("base", fb("- add: a\n- add: b")) == "base\na\nb"
    assert ORDERED.reduce("base", fb("Add: a", "* add:b")) == "base\na\nb"


def test_ordered_splits_a_bullet_joined_onto_a_line_after_a_dash() -> None:
    """A live backward model wrote ``- add: A — add: B``, copying the dash from the prompt's format line."""
    assert ORDERED.reduce("base", fb("- add: A \u2014 add: B")) == "base\nA\nB"
    assert ORDERED.reduce("base", fb("- add: A\u2013add: B", "- add: C - add: D")) == "base\nA\nB\nC\nD"


def test_ordered_refuses_a_delete_joined_onto_an_add() -> None:
    with pytest.raises(ReducerError, match="asks to delete"):
        ORDERED.reduce("base", fb("- add: A \u2014 delete: B"))


def test_ordered_keeps_a_dash_that_is_not_followed_by_a_marker() -> None:
    assert ORDERED.reduce("base", fb("- add: Be brief \u2014 three sentences at most.")) == (
        "base\nBe brief \u2014 three sentences at most."
    )


def test_ordered_compares_repeats_after_removing_the_marker() -> None:
    assert ORDERED.reduce("base\nCite sources.", fb("- add: Cite sources.")) == "base\nCite sources."


def test_ordered_removes_the_marker_from_a_list_item() -> None:
    assert ORDERED.reduce(["x"], fb("- add: y", "- add: x")) == ["x", "y"]


def test_ordered_keeps_a_line_that_only_mentions_a_verb() -> None:
    assert ORDERED.reduce("base", fb("- Always add a greeting.")) == "base\n- Always add a greeting."


@pytest.mark.parametrize("line", ["- update: Use a formal tone.", "- delete: Thank the customer.", "Delete:x"])
def test_ordered_refuses_an_update_or_delete_line(line: str) -> None:
    with pytest.raises(ReducerError, match="ordered reducer can only add"):
        ORDERED.reduce("base", fb("- add: fine", line))


def test_numeric_add_reads_the_delta_after_the_add_marker() -> None:
    assert NUMERIC_ADD.reduce(10, fb("- add: -3", "add: 0.5")) == 7.5
    assert NUMERIC_ADD.reduce([1, 2], fb("- add: [1, 1]")) == [2, 3]


def test_numeric_add_refuses_an_update_line() -> None:
    with pytest.raises(ReducerError, match="numeric_add reducer can only add"):
        NUMERIC_ADD.reduce(10, fb("- update: 20"))


def test_a_callable_reducer_receives_the_marked_text_unchanged() -> None:
    seen: list[str] = []

    def record(current: object, feedback: Sequence[GradFeedback]) -> object:
        seen.extend(f.text for f in feedback)
        return current

    CallableReducer(record).reduce("v", fb("- update: x"))
    assert seen == ["- update: x"]


# ── The escape hatch ──


def test_a_callable_is_a_reducer() -> None:
    def keep_last(current: str, feedback: Sequence[GradFeedback]) -> str:
        """The last feedback replaces the value."""
        return feedback[-1].text if feedback else current

    reducer = CallableReducer(keep_last)
    assert isinstance(reducer, FeedbackReducer)
    assert reducer.name == "keep_last"
    assert reducer.guidance == "The last feedback replaces the value."
    assert reducer.reduce("a", fb("b", "c")) == "c"


def test_the_built_ins_satisfy_the_protocol() -> None:
    assert isinstance(ORDERED, FeedbackReducer)
    assert isinstance(NUMERIC_ADD, FeedbackReducer)
    assert ORDERED.name == "ordered"
    assert NUMERIC_ADD.name == "numeric_add"
    assert ORDERED.guidance
    assert NUMERIC_ADD.guidance


# ── Defaults: the reducer follows the value's shape ──


def test_a_declaration_infers_the_reducer_from_the_initial_value() -> None:
    local = Local(text="t", lines=["a"], weight=1.0, weights=[1.0, 2.0], count=3)
    assert local.param("text").reducer == "ordered"
    assert local.param("lines").reducer == "ordered"
    assert local.param("weight").reducer == "numeric_add"
    assert local.param("weights").reducer == "numeric_add"
    assert local.param("count").reducer == "numeric_add"


def test_a_declaration_with_no_inferable_reducer_is_refused_naming_the_knob() -> None:
    with pytest.raises(ParamWiringError, match=r"'flag'.*reducer="):
        Local(flag=True)
    with pytest.raises(ParamWiringError, match=r"'table'.*reducer="):
        Local(table={"a": 1})


def test_a_declaration_accepts_a_bare_callable_as_its_reducer() -> None:
    local = Local()
    local.declare("flag", True, reducer=lambda current, feedback: not current)
    assert local.param("flag").reducer == "<lambda>"
    local.consolidate("flag", fb("flip"))
    assert local.load("flag") == (False, 1)
