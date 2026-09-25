"""Tool results, observations, and certificates."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from rich.console import Console

from ..lean.server import Source

type Guarantees = str | Sequence[str] | Callable[..., str | Sequence[str]]


@dataclass(frozen=True)
class Certified:
    """A tool's return value with trusted claims about it."""

    value: object
    guarantees: Guarantees | None = None


@dataclass(frozen=True)
class Observation:
    """One committed fact about an opaque symbol: a tool result or a model judgment."""

    name: str
    kind: Literal["tool", "judgment"]
    symbol: str
    arguments: tuple[str, ...]
    value: str
    origin: str
    axioms: Mapping[str, str]


LOGICAL_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})


@dataclass(frozen=True)
class Certificate:
    """A cold-checked answer and proof, with the inventory of its assumptions."""

    answer: object
    goal: str
    proof: str
    artifact: str
    project_fingerprint: str
    axioms: tuple[str, ...]
    observations: Mapping[str, Observation]
    elaborations: int

    @property
    def used(self) -> Mapping[str, Observation]:
        """The observations with an axiom in ``axioms``: what the proof depends on."""
        return MappingProxyType(
            {name: o for name, o in self.observations.items() if any(a in self.axioms for a in o.axioms)}
        )

    def summary(self, console: Console | None = None) -> None:
        """Print the answer, the goal, the logical axioms used, and each used observation."""
        console = console or Console()
        console.print(f"Certified answer: {self.answer!r}", markup=False)
        console.print(f"Goal: {self.goal}", markup=False)
        logical = [a for a in self.axioms if a in LOGICAL_AXIOMS]
        console.print(f"Logical axioms: {', '.join(logical) or 'none'}", markup=False)
        for name, o in self.used.items():
            reading = " ".join([o.symbol, *o.arguments, "=", o.value])
            console.print(f"  {name} ({o.kind}): {reading} [{o.origin}]", markup=False)

    def write(self, path: str | Path) -> Path:
        """Write ``artifact`` to ``path``, creating parents; return ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.artifact)
        return path


@dataclass(frozen=True)
class LeanTerm:
    """A certified Lean expression, the result of a term-mode certified function."""

    expression: str
    lean_type: str
    declaration: str
    proof_declaration: str
    certificate: Certificate
    source: Source = field(repr=False, compare=False)

    def __str__(self) -> str:
        """The expression."""
        return self.expression
