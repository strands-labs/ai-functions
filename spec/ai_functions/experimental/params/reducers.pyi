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

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ...ai_thread.errors import AIFunctionError
from ...types.graph import GradFeedback

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

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:
        """The next value from the current one and the routed feedback."""
        ...

class Ordered:
    """Append each feedback text in order, one block or one item each, skipping verbatim repeats (FR4).

    A text is one unit however many lines it has: a two-line gradient is appended whole and is a
    repeat when its lines already appear as a contiguous run, in order, in the value (or in a text
    appended earlier in the batch). Comparison strips each line; the current value itself is kept
    verbatim, blank lines included, and comes back as the same object when nothing was added.
    """

    name: str
    guidance: str
    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:
        """Append.

        Args:
            current: A ``str`` (blocks of lines) or a ``list[str]`` (items).
            feedback: The routed gradients; each ``text`` is one addition.

        Raises:
            ReducerError: ``current`` is neither a string nor a list, or a feedback line asks to
                update or delete (FR5).
        """
        ...

class NumericAdd:
    """Add each feedback's number (or vector, pointwise) to the value (FR4)."""

    name: str
    guidance: str
    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:
        """Add.

        Args:
            current: A number, or a sequence of numbers.
            feedback: The routed gradients; each ``text`` parses as a number or a JSON array of numbers.

        Raises:
            ReducerError: ``current`` is not numeric, a text does not parse as a number or array,
                a vector delta has the wrong length, or a feedback line asks to update or delete (FR5).
        """
        ...

class CallableReducer:
    """A plain ``(current, feedback) -> new`` callable as a reducer: the escape hatch.

    Args:
        fn: The combining function.
        name: Stable name; the function's ``__name__`` when omitted.
        guidance: One sentence for the backward model; the first line of ``fn``'s docstring when omitted.
    """

    def __init__(
        self,
        fn: Callable[[Any, Sequence[GradFeedback]], Any],
        *,
        name: str | None = None,
        guidance: str | None = None,
    ) -> None: ...
    @property
    def name(self) -> str:
        """The function's name unless one was given."""
        ...

    @property
    def guidance(self) -> str:
        """The docstring's first line unless one was given."""
        ...

    def reduce(self, current: Any, feedback: Sequence[GradFeedback]) -> Any:
        """Call the function."""
        ...

    def __repr__(self) -> str:
        """The reducer's name."""
        ...

ORDERED: FeedbackReducer
"""Append in order, skipping verbatim repeats; the default for text values."""

NUMERIC_ADD: FeedbackReducer
"""Add pointwise; the default for numeric values."""

def as_reducer(value: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any]) -> FeedbackReducer:
    """A reducer from a reducer or a bare callable.

    Args:
        value: A :class:`FeedbackReducer` (returned as is) or a callable (wrapped).
    """
    ...

def default_reducer(initial: object) -> FeedbackReducer | None:
    """The reducer a value's shape implies, or ``None`` when no built-in fits.

    Text and lists of text take :data:`ORDERED`; numbers and sequences of numbers take
    :data:`NUMERIC_ADD`. Anything else has no obvious combining rule and needs ``reducer=``.

    Args:
        initial: The declared initial value.
    """
    ...
