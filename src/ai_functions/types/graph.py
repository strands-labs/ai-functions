"""Graph node types for AI Function execution traces."""

import typing
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from strands import Agent

if typing.TYPE_CHECKING:
    from ..core import AIFunction
    from ..memory import MemoryBackend

T = TypeVar("T")


# Base class for both Result and Parameter
@dataclass(kw_only=True)
class Node(Generic[T]):
    """Base graph node holding a value and optional gradients."""

    value: T
    name: str
    gradients: list[str] = field(default_factory=list)
    requires_grad: bool = True

    def __str__(self) -> str:  # noqa: D105
        return str(self.value)


# Node that represents the result of an AIFunction call, tracks additional information
@dataclass(kw_only=True)
class Result(Node[T]):
    """Node representing the output of an AIFunction invocation."""

    func: "AIFunction"
    agent: Agent
    # corresponding tool call id if this function is being used as a tool
    tool_id: str | None = None

    inputs: list[Node[Any]] = field(default_factory=list)
    # result nodes of other AI Functions that where called as tools by this one
    # (these are part of the graph but are not explicit inputs)
    tool_results: list["Result[Any]"] = field(default_factory=list)


@dataclass
class Derivation:
    """Describes how a ParameterView was derived (full recall, query, or search)."""

    kind: Literal["full", "query", "search"]
    meta: "ParameterMeta" = field(default_factory=dict)


ParameterMeta = dict[str, Any]


@dataclass
class ParameterGradient:
    """A gradient entry that preserves the derivation context of the ParameterView it came from."""

    feedback: str
    derivation: Derivation

    def __str__(self) -> str:  # noqa: D105
        return self.feedback


@dataclass(eq=False)
class ParameterRef:
    """Unique reference to a parameter on a memory backend. Does not hold a value."""

    name: str
    memory: "MemoryBackend"
    description: str
    procedural: bool = False
    gradients: list[ParameterGradient] = field(default_factory=list)

    def consolidate(self) -> None:
        """Apply accumulated gradients to the backing memory store."""
        self.memory.consolidate(self)


@dataclass(kw_only=True)
class ParameterView(Node[T]):
    """A view of a parameter produced by recall/query/search on the backend."""

    source: ParameterRef
    derivation: Derivation

    @property
    def description(self) -> str:
        """Return the parameter description from the source ref."""
        return self.source.description

    @property
    def procedural(self) -> bool:
        """Return whether this parameter holds procedural (code) content."""
        return self.source.procedural
