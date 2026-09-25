"""Tool results, observations, and certificates."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.console import Console

from ..lean.server import Source

type Guarantees = str | Sequence[str] | Callable[..., str | Sequence[str]]
"""Lean propositions a tool asserts about one returned value.

A builder receives the value as a typed Lean term, then each argument of the call
as a Lean term. Each proposition becomes an axiom ``<observation>_contract<i>``,
stated at the argument literals, and is restated at the arguments as the model
wrote them.
"""

@dataclass(frozen=True)
class Certified:
    """A tool's return value with trusted claims about it.
    """

    value: object
    guarantees: Guarantees | None = None

@dataclass(frozen=True)
class Observation:
    """One committed fact about an opaque symbol: a tool result or a model judgment.

    Attributes:
        name: The transparent value, e.g. ``H.price1``, or the judgment axiom, e.g.
            ``J.reqDays1``.
        kind: Who supplied the value.
        symbol: The observed symbol's name.
        arguments: Canonical Lean literals of the arguments.
        value: Canonical Lean literal of the value; for a judgment at a non-boundary
            type, such as a proposition, its closed text.
        origin: The tool call as made, e.g. ``lookup_price(sku='widget')``, or the
            judgment's justification.
        axioms: The axioms added, mapped to their propositions; the equation first,
            then any guarantees.
    """

    name: str
    kind: Literal["tool", "judgment"]
    symbol: str
    arguments: tuple[str, ...]
    value: str
    origin: str
    axioms: Mapping[str, str]

LOGICAL_AXIOMS: frozenset[str]
"""``propext``, ``Classical.choice`` and ``Quot.sound``: Lean's standard axioms,
permitted in every certificate."""

@dataclass(frozen=True)
class Certificate:
    """A cold-checked answer and proof, with the inventory of its assumptions.

    Attributes:
        goal: The proved proposition.
        artifact: The complete, annotated module the checker accepted.
        project_fingerprint: The project needed to replay ``artifact``.
        axioms: The cold axiom inventory of the answer and proof, sorted: a subset
            of ``LOGICAL_AXIOMS`` plus axioms of ``observations``.
        observations: Every observation of the call, used or not, by name.
        elaborations: Lean elaborations the call performed.

    Invariants:
        V1, V5.
    """

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
        """The observations with an axiom in ``axioms``: what the proof depends on.

        Each observation's ``kind`` says whether a tool supplied it or the model
        judged it.
        """
        ...

    def summary(self, console: Console | None = None) -> None:
        """Print the answer, the goal, the logical axioms used, and each observation
        in ``used``."""
        ...

    def write(self, path: str | Path) -> Path:
        """Write ``artifact`` to ``path``, creating parents; return ``path``."""
        ...

@dataclass(frozen=True)
class LeanTerm:
    """A certified Lean expression, the result of a term-mode certified function.

    Passed as an argument of a certified function, it is bound as that input's
    ``H.<parameter>`` (see ``ProofSession``).

    Attributes:
        expression: The answer as closed text over the project: ledger definitions
            (``H``, ``A``) unfolded and names fully qualified, so it elaborates in the
            project alone (``LeanServer.closed_term``). An answer without such a form,
            e.g. one calling a recursive ``A`` helper, keeps its text as written and
            cannot be passed as an argument.
        lean_type: The rendered result type, e.g. ``Prop``.
        declaration: The harness binding of the expression in ``certificate.artifact``.
        proof_declaration: The harness theorem proving the contract of that binding.
        source: ``certificate.artifact`` with the model's byte ranges, to compile it confined again.
    """

    expression: str
    lean_type: str
    declaration: str
    proof_declaration: str
    certificate: Certificate
    source: Source = field(repr=False, compare=False)

    def __str__(self) -> str:
        """The expression."""
        ...
