"""``FeedbackReducer``: the rule for combining several pieces of feedback aimed at one parameter.

TextGrad's backward pass routes a list of ``GradFeedback`` onto a parameter node, and
``consolidate`` has to turn that list plus the current value into the next value. For a
memory backend that rule is a property of the host class (one model rewrite over the
texts). A local parameter declares the rule itself, so a string can ask for "append in
order" and a vector for "add pointwise" without a new host class and without a model call.

Three reducers ship. :data:`ORDERED` appends each feedback's text as one block of lines (or one
list item), in order, skipping verbatim repeats and keeping the current value as it is.
:data:`NUMERIC_ADD` parses each feedback's text as a number or a JSON array of numbers and adds
it to the value, pointwise for vectors. A plain callable ``(current, feedback) -> new`` is a reducer too, through
:class:`CallableReducer`. Each reducer carries a ``guidance`` sentence that the local host
puts in the parameter's description, so the backward model is told what shape of feedback
the target accepts.

Invariants:
    FR1 — a reducer is a pure function of ``(current, feedback)``: no model call, no
    I/O, no state. Equal inputs give equal outputs.

    FR2 — a reducer refuses a value or a feedback text it cannot combine with a
    :class:`ReducerError` naming the reducer and the offending input; it never coerces,
    drops or guesses.

    FR3 — no feedback is a no-op: ``reduce(current, [])`` returns ``current`` unchanged.

    FR4 — :data:`ORDERED` appends each text whole, in arrival order, and skips a text whose
    lines already appear in the value as a contiguous run (each line stripped); it never rewrites
    the current value, so blank lines survive and an all-repeat batch returns it unchanged.
    :data:`NUMERIC_ADD` refuses a length mismatch.

    FR5 — TextGrad's backward model writes feedback to a parameter as ``- add:``, ``- update:``
    and ``- delete:`` bullets. :data:`ORDERED` and :data:`NUMERIC_ADD` remove the ``- add:``
    marker from each bullet before combining the text, including a bullet joined onto a line
    after a dash, and refuse a ``- update:`` or ``- delete:`` bullet with a :class:`ReducerError`,
    because appending and adding cannot apply it. A
    :class:`CallableReducer` receives the text unchanged.
"""

from __future__ import annotations

import inspect
import json
import numbers
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.types.graph import GradFeedback


class ParamsError(AIFunctionError):
    """Base of the params package's failure modes.

    Args:
        message: Human-readable explanation.
        function_name: The compiled tool name that raised, when known.
    """


class ReducerError(ParamsError):
    """A reducer was handed a value or a feedback text it cannot combine (FR2)."""


@runtime_checkable
class FeedbackReducer(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    What a parameter declaration names as its combining rule (FR1).

    Structural, so a user's own object works without inheriting anything.
    """

    @property
    def name(self) -> str:
        """Short stable name, recorded on the parameter and on every recall event's meta."""
        ...

    @property
    def guidance(self) -> str:
        """One sentence for the backward model about the feedback shape this reducer accepts."""
        ...

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:  # pyright: ignore[reportExplicitAny]
        """The next value from the current one and the routed feedback."""
        ...


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, each stripped: the unit ORDERED compares on."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Whether ``needle`` appears in ``haystack`` as a contiguous run, in order."""
    if not needle:
        return True
    width = len(needle)
    return any(list(haystack[i : i + width]) == list(needle) for i in range(len(haystack) - width + 1))


# The bullet markers TextGrad's backward prompt asks for on a parameter's feedback.
_MARKER = re.compile(r"^\s*(?:[-*]\s*)?(add|update|delete)\s*:\s*", re.IGNORECASE)
# A second bullet the model joined onto the line with a dash, as in the prompt's own
# "- add: <text> — information to add" format line.
_JOINED = re.compile(r"(?:\s*[\u2014\u2013]\s*|\s+-\s+)(?=(?:add|update|delete)\s*:)", re.IGNORECASE)


def _additions(reducer: str, text: str) -> str:
    """``text`` with each line's ``- add:`` marker removed, for a reducer that only adds (FR5).

    A marker joined onto a line after a dash starts a bullet of its own. A line with no marker is
    kept as it is.

    Raises:
        ReducerError: A line is marked ``update`` or ``delete``.
    """
    kept: list[str] = []
    for line in _JOINED.sub("\n", text).split("\n"):
        marker = _MARKER.match(line)
        if marker is None:
            kept.append(line)
        elif marker.group(1).lower() == "add":
            kept.append(line[marker.end() :])
        else:
            raise ReducerError(
                f"the {reducer} reducer can only add to a value; feedback line {line.strip()!r} asks to "
                f"{marker.group(1).lower()} it. Declare the parameter with a reducer that rewrites, such as a "
                f"CallableReducer"
            )
    return "\n".join(kept)


class Ordered:
    """Append each feedback text in order, one block or one item each, skipping verbatim repeats (FR4).

    A text is one unit however many lines it has: a two-line gradient is appended whole and is a
    repeat when its lines already appear as a contiguous run, in order, in the value (or in a text
    appended earlier in the batch). Comparison strips each line; the current value itself is kept
    verbatim, blank lines included, and comes back as the same object when nothing was added.
    """

    name = "ordered"
    guidance = (
        "Feedback is appended to this value in order; write each change as a '- add:' line, because "
        "this value is never updated or deleted in place."
    )

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:  # pyright: ignore[reportExplicitAny]
        """Append.

        Args:
            current: A ``str`` (blocks of lines) or a ``list[str]`` (items).
            feedback: The routed gradients; each ``text`` is one addition.

        Raises:
            ReducerError: ``current`` is neither a string nor a list, or a feedback line asks to
                update or delete (FR5).
        """
        if isinstance(current, str):
            seen = _lines(current)
            added: list[str] = []
            for entry in feedback:
                unit = _lines(_additions(self.name, entry.text))
                if unit and not _contains_run(seen, unit):
                    added.append("\n".join(unit))
                    seen.extend(unit)
            if not added:
                return current
            if not current.strip():
                return "\n".join(added)
            separator = "" if current.endswith("\n") else "\n"
            return current + separator + "\n".join(added)
        if isinstance(current, list):
            items: list[Any] = list(current)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            seen_items = {str(item).strip() for item in items}  # pyright: ignore[reportUnknownVariableType]
            for entry in feedback:
                text = _additions(self.name, entry.text).strip()
                if text and text not in seen_items:
                    items.append(text)
                    seen_items.add(text)
            return items
        raise ReducerError(
            f"the ordered reducer appends to a str or a list, not to {type(current).__name__}; "
            f"declare the parameter with a reducer that fits its value"
        )


class NumericAdd:
    """Add each feedback's number (or vector, pointwise) to the value (FR4)."""

    name = "numeric_add"
    guidance = (
        "Feedback to this value is added to it numerically; write the delta as '- add: <number>', or as "
        "'- add: <JSON array of numbers>' of the same length."
    )

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:  # pyright: ignore[reportExplicitAny]
        """Add.

        Args:
            current: A number, or a sequence of numbers.
            feedback: The routed gradients; each ``text`` parses as a number or a JSON array of numbers.

        Raises:
            ReducerError: ``current`` is not numeric, a text does not parse as a number or array,
                a vector delta has the wrong length, or a feedback line asks to update or delete (FR5).
        """
        if _is_number(current):
            total = current
            for entry in feedback:
                delta = self._parse(_additions(self.name, entry.text))
                if not _is_number(delta):
                    raise ReducerError(
                        f"the numeric_add reducer adds a scalar to a scalar; feedback {entry.text!r} is not a number"
                    )
                total = total + delta  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
            return total
        if isinstance(current, (list, tuple)) and all(_is_number(v) for v in current):  # pyright: ignore[reportUnknownVariableType]
            vector: list[Any] = list(current)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            for entry in feedback:
                delta = self._parse(_additions(self.name, entry.text))
                if not isinstance(delta, list) or not all(_is_number(v) for v in delta):  # pyright: ignore[reportUnknownVariableType]
                    raise ReducerError(
                        f"the numeric_add reducer adds a vector to a vector; feedback {entry.text!r} is not a JSON "
                        f"array of numbers"
                    )
                if len(delta) != len(vector):  # pyright: ignore[reportUnknownArgumentType]
                    raise ReducerError(
                        f"the numeric_add reducer needs a delta of length {len(vector)}, got {len(delta)} "  # pyright: ignore[reportUnknownArgumentType]
                        f"from {entry.text!r}"
                    )
                vector = [a + b for a, b in zip(vector, delta, strict=True)]  # pyright: ignore[reportUnknownVariableType]
            return vector if feedback else current
        raise ReducerError(
            f"the numeric_add reducer adds numbers, not {type(current).__name__}; "
            f"declare the parameter with a reducer that fits its value"
        )

    @staticmethod
    def _parse(text: str) -> object:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReducerError(
                f"the numeric_add reducer could not read feedback {text!r} as a number or a JSON array of numbers"
            ) from exc


class CallableReducer:
    """A plain ``(current, feedback) -> new`` callable as a reducer: the escape hatch.

    Args:
        fn: The combining function.
        name: Stable name; the function's ``__name__`` when omitted.
        guidance: One sentence for the backward model; the first line of ``fn``'s docstring when omitted.
    """

    def __init__(
        self,
        fn: Callable[[Any, Sequence[GradFeedback]], Any],  # pyright: ignore[reportExplicitAny]
        *,
        name: str | None = None,
        guidance: str | None = None,
    ) -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", type(fn).__name__)
        doc = inspect.getdoc(fn) or ""
        self._guidance = guidance if guidance is not None else doc.split("\n", 1)[0]

    @property
    def name(self) -> str:
        """The function's name unless one was given."""
        return self._name

    @property
    def guidance(self) -> str:
        """The docstring's first line unless one was given."""
        return self._guidance

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:  # pyright: ignore[reportExplicitAny]
        """Call the function."""
        return self._fn(current, feedback)

    def __repr__(self) -> str:
        """The reducer's name."""
        return f"<CallableReducer {self._name}>"


ORDERED: FeedbackReducer = Ordered()
"""Append in order, skipping verbatim repeats; the default for text values."""

NUMERIC_ADD: FeedbackReducer = NumericAdd()
"""Add pointwise; the default for numeric values."""


def as_reducer(value: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any]) -> FeedbackReducer:  # pyright: ignore[reportExplicitAny]
    """A reducer from a reducer or a bare callable.

    Args:
        value: A :class:`FeedbackReducer` (returned as is) or a callable (wrapped).
    """
    if isinstance(value, FeedbackReducer):
        return value
    return CallableReducer(value)


def default_reducer(initial: object) -> FeedbackReducer | None:
    """The reducer a value's shape implies, or ``None`` when no built-in fits.

    Text and lists of text take :data:`ORDERED`; numbers and sequences of numbers take
    :data:`NUMERIC_ADD`. Anything else has no obvious combining rule and needs ``reducer=``.

    Args:
        initial: The declared initial value.
    """
    if isinstance(initial, str):
        return ORDERED
    if _is_number(initial):
        return NUMERIC_ADD
    if isinstance(initial, (list, tuple)):
        items = list(initial)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        if items and all(_is_number(v) for v in items):  # pyright: ignore[reportUnknownVariableType]
            return NUMERIC_ADD
        if all(isinstance(v, str) for v in items):  # pyright: ignore[reportUnknownVariableType]
            return ORDERED
    return None
