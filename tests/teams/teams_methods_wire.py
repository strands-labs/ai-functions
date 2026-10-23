"""Fixtures for the MethodAgent-backed member tests: a MethodAgent cast mirroring ``teams_wire``'s functions.

``teams_wire.read`` is the ``AIFunction`` a ``FunctionMember`` wraps; ``Analyst.read`` here is the same
capability as an ``@ai_method`` on a ``MethodAgent``, so the conformance suite can
run one set of assertions against both member kinds. Agents are module level because
``compile_ai_method`` resolves annotations with ``typing.get_type_hints`` against module globals.
"""

from __future__ import annotations

from strands.tools.decorator import tool as strands_tool

from ai_functions import ai_function
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.testing import Turn

from .teams_wire import Reading

# ── Function twins of the shapes below, for the conformance suite's function kind ──


@ai_function[Reading]
def brief_fn(context: str = "none", request: str = "") -> str | None:
    """Context: {context}. Request: {request}."""


@ai_function[Reading]
def nothing_fn() -> str | None:
    """Report."""


# ── Tools a member may carry itself ──


@strands_tool(name="members_own", description="the member carried this itself")
def members_own(x: str) -> str:
    return x


@strands_tool(name="equipped_gave", description="the equipped hook contributed this")
def equipped_gave(x: str) -> str:
    return x


@strands_tool(name="first_gave", description="the first equipped hook contributed this")
def first_gave(x: str) -> str:
    return x


# ── The cast: MethodAgents, the shape a MethodMember wraps ──


class Analyst(MethodAgent):
    """An analyst whose capability compiles to ``{source}-analyst.read``."""

    def __init__(self, source: str) -> None:
        self.name = f"{source}-analyst"
        self.source = source

    @ai_method(Reading, description="Read one source and report what it alone shows")
    def read(self, focus: str, depth: int = 2) -> Reading:
        """Read the {self.source} source with {focus} in mind, to depth {depth}."""
        ...


class ToolCarrier(MethodAgent):
    """An analyst whose ``@ai_method`` carries a tool at the decorator level (the compiled-tools case)."""

    name = "carrier"

    @ai_method(Reading, description="Read with a tool of one's own", tools=[members_own])
    def read(self, focus: str, depth: int = 2) -> Reading:
        """Read with {focus} in mind, to depth {depth}."""
        ...


class Briefed(MethodAgent):
    """The ``teams_members`` ``brief`` function as a method: the first positional is not the request."""

    name = "briefed"

    @ai_method(Reading, description="Brief")
    def brief(self, context: str = "none", request: str = "") -> Reading:
        """Context: {context}. Request: {request}."""
        ...


class Silent(MethodAgent):
    """A capability with no parameters: nowhere for a request to go."""

    name = "silent"

    @ai_method(Reading, description="Report")
    def report(self) -> Reading:
        """Report."""
        ...


# ── Turns ──


def answered(text: str) -> Turn:
    """One turn for an ``@ai_method(str)`` capability, which wraps ``str`` in a generated ``FinalAnswer`` model."""
    return Turn(tool_calls=(("FinalAnswer", {"answer": text}),))
