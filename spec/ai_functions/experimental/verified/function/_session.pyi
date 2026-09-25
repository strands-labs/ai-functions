"""The proof session behind ``verified.ai_function``; ``verified.ai_compile`` reaches it through that.

A session is one ledger over a prepared project: harness bindings in namespace ``H``,
model declarations in ``A``, and judgment axioms in ``J``. Two modes differ in how
the answer is materialized: in value mode it reduces to a literal at the contract's
result type and is decoded; in term mode it is bound as ``H.answer<n>``, possibly a
function, and nothing is decoded. In term mode an input without an argument is
quantified instead of bound: the answer is a function of those inputs, and the goal
is ``∀ a b, C (answer a b) a b`` (``verified.ai_compile`` binds no input).

The session renders every harness application itself (the goal, observed and
judged applications), encoding each Python value at its declared Lean type with
``lean.types.encode``. The session's records are the only copy of the ledger; the
server holds none.

Invariants:
    V1, V2, V3, V5.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from ..dsl.types import LeanType
from ..lean import LeanProject, LeanSymbol
from ..lean.server import Source
from .types import Certificate, Guarantees, LeanTerm, Observation

Kind = Literal["definition", "provenance", "guarantee", "judgment", "value", "lemma"]

@dataclass(frozen=True)
class Decl:
    """One ledger record.

    Attributes:
        kind: The record's role; the rendered ledger prints ``-- [kind] origin``.
        origin: Who or what produced it.
        source: Its Lean text.
        expected: The twin the checker elaborates without agent code; ``None`` for
            model text.
    """

    kind: Kind
    origin: str
    source: str | Source
    expected: str | None = None

    @property
    def comment(self) -> str:
        """The annotation, with the origin folded onto one line."""
        ...

@dataclass(frozen=True)
class Outcome:
    """Feedback for one model operation.
    """

    ok: bool
    message: str
    code: str | None = None
    detail: str | None = None
    excerpt: str | None = None
    """What the model reads instead of ``code``, when ``code`` restates facts it already has."""

class ProofSession:
    """One ledger over a prepared project, owned and closed by its consumer.

    Lifecycle:
        OPEN → CERTIFIED on an accepted submission; OPEN → ABORTED on a harness
        failure or an exhausted budget; any state → CLOSED on ``close``.

    Invariants:
        - One observation exists per canonical application of a symbol, across
          tools and judgments.
        - Every committed harness block has its twin recorded.
        - In ABORTED, every operation re-raises the abort.
        - The session's records are exactly what the server has committed.
        - When a model operation times out, the operation fails with model
          feedback and the session starts a fresh server, committing its records
          again as one block; the ledger is unchanged. Any other server failure,
          or a failed recommit, aborts the session.
    """

    def __init__(
        self,
        contract: LeanSymbol,
        arguments: dict[str, object],
        *,
        judgments: Sequence[LeanSymbol] = (),
        term_result: bool = False,
        max_attempts: int = 10,
        timeout: float = 120.0,
    ) -> None:
        """Open a ledger and bind the contract's inputs.

        Args:
            contract: Result-first ``Prop``: first explicit parameter the result, the
                rest the inputs.
            arguments: A Python value for each input, by parameter name; in term mode,
                inputs without one are quantified.
            judgments: Opaque symbols the model may judge.
            term_result: Term mode instead of value mode.
            max_attempts: Retries allowed after the first attempt; an attempt is a rejected
                submission, or stopping without a certified answer.
            timeout: Seconds per Lean request and per cold check.

        Ensures:
            - Each input with an argument is bound as ``def H.<name>``, encoded at its
              Lean type.
            - A ``LeanTerm`` argument is bound to its ``expression`` instead, as
              model text (confined) that is its own twin, before the other inputs.

        Raises:
            LeanError: The contract is not a monomorphic, result-first ``Prop`` with
                explicit parameters; an input has no argument in value mode; a judged symbol is
                not opaque with boundary argument types; or a ``LeanTerm`` argument comes from
                another project (by ``certificate.project_fingerprint``), has another
                ``lean_type`` than the input, or does not elaborate over the project alone.
            EncodingError: An argument does not fit its Lean type.
        """
        ...

    @property
    def project(self) -> LeanProject:
        """The contract's project."""
        ...

    @property
    def result_type(self) -> LeanType:
        """The answer's type: the contract's result type, as a function of the quantified inputs."""
        ...

    @property
    def certificate(self) -> Certificate | None:
        """The accepted certificate; ``None`` until CERTIFIED."""
        ...

    @property
    def result(self) -> object | LeanTerm | None:
        """The certified value, or the ``LeanTerm`` in term mode; ``None`` before."""
        ...

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Every committed observation, in commit order."""
        ...

    def fresh(self, stem: str) -> str:
        """Allocate an unused name ``H.<stem><n>``.

        Raises:
            LeanError: ``stem`` is not a plain Lean identifier.
        """
        ...

    def goal(self, answer: str) -> str:
        """Return the contract applied to Lean text ``answer`` and the inputs, quantifying the unbound ones."""
        ...

    def ledger(self) -> str:
        """Return the annotated ledger as Lean source, rendered from the records.

        Ensures:
            - The imports, the prelude, then each committed record in
              order, with ``-- [kind] origin`` above it.
            - The final artifact is this text plus the submission, with the model's
              byte ranges kept for the checker's confined compile.
        """
        ...

    def evaluate(self, slots: Sequence[tuple[str, str, LeanType]]) -> list[object]:
        """Evaluate ``(label, expression, type)`` slots to decoded Python values.

        Ensures:
            - All slots elaborate as one probe.
            - Each value equals its decoded literal by kernel ``rfl``.
            - An error is reported under the slot Lean positions it in, with that
              slot's label and expression.

        Raises:
            ValueError: An expression is empty, not one term, or does not reduce to
                a closed value; the message is model feedback.
        """
        ...

    def observe(
        self,
        symbol: LeanSymbol,
        arguments: Sequence[object],
        value: object,
        *,
        origin: str,
        stem: str,
        guarantees: Guarantees | None = None,
        written: Sequence[str] | None = None,
    ) -> Outcome:
        """Commit a tool's observation ``symbol arguments = value``.

        Args:
            written: The model's Lean expressions the arguments were evaluated from.

        Ensures:
            - A new observation commits ``def H.<stem><n>`` and
              ``axiom H.<stem><n>_spec``, then one axiom per guarantee, as one block.
            - When ``written`` differs from the argument literals, the block also
              commits ``theorem H.<stem><n>_spec_as_written``, the equation at the
              written expressions (model text), proved by the axiom; the outcome
              leads with it. It adds no assumption.
            - Observing the same application with the same value commits nothing
              and names the existing observation.

        Raises:
            LeanError: The value conflicts with an earlier tool or judgment
                observation of the application; the session aborts.
            EncodingError: A value does not fit its Lean type.
        """
        ...

    def judge(self, symbol: str, arguments: Sequence[str], value: str, justification: str) -> Outcome:
        """Evaluate a model judgment of ``symbol arguments = value`` and commit it.

        Args:
            symbol: The name of an allowlisted judgment symbol.
            arguments: Lean expressions at the symbol's argument types.
            value: A Lean expression at the symbol's result type, which may be ``Prop``
                or a theory type such as an inductive of the project.
            justification: The reading's reasoning, at least four words.

        Ensures:
            - A new judgment commits ``axiom J.<short><n>``, ``short`` being the
              symbol's last name component, at the argument literals.
            - At a boundary result type the value is its literal; at any other it is
              the closed text of ``LeanServer.closed_term``, so it cannot depend on
              the ledger (``A`` helpers are unfolded, ``J`` axioms rejected).
            - When ``arguments`` differ from those literals, the block also commits
              ``theorem J.<short><n>_as_written``, the reading at the written
              expressions (model text), proved by the axiom; the outcome leads with
              it. It adds no assumption.
            - Judging the same application with the same value commits nothing.
            - A conflicting value is rejected in the outcome; the earlier value stands.
        """
        ...

    def append(self, code: str) -> Outcome:
        """Commit model declarations in namespace ``A``.

        Harness text is outside ``A`` and never opens it, so no model declaration
        changes how harness names resolve; the model refers to its own as ``A.name``.

        Ensures:
            - On success, the outcome names every declaration the block declared,
              with a hint when one lies under ``A.A``.
            - On failure, the ledger is unchanged.
        """
        ...

    def inspect(self, expression: str) -> Outcome:
        """Show a term's value, or run ``#check``/``#print`` commands; keep nothing.

        Ensures:
            - Input that parses as one term is reduced with ``whnf`` and shown as a
              literal; no compiled code runs.
            - Input starting with ``#check``/``#print`` is split at each line that
              starts with one, each argument parsed as one term, and all of them
              elaborated as one probe.
        """
        ...

    def submit(self, answer: str, proof: str) -> Outcome:
        """Certify ``answer`` with ``proof`` of the goal.

        Ensures:
            - A failed warm check, or axioms outside the permitted base, is a
              failed outcome that counts as an attempt.
            - On success the session is CERTIFIED: the cold check accepted the
              artifact, and the cold inventory is logical axioms plus observation
              axioms and equals the warm inventory.

        Raises:
            ContractNotProved: The submission exhausted ``max_attempts``; the
                session aborts.
            LeanError: The cold check rejected an artifact the warm check accepted,
                or the inventories disagree; the session aborts.
        """
        ...

    def fail_attempt(self, diagnostic: str) -> None:
        """Count an attempt that ended without a certified answer.

        Raises:
            ContractNotProved: More than ``max_attempts`` attempts failed; the session aborts.
        """
        ...

    def check_result(self, returned: object) -> None:
        """Require ``returned`` to be the certified result.

        Raises:
            ContractNotProved: The session is not CERTIFIED, or ``returned`` differs
                from ``result``.
        """
        ...

    def on_abort(self, callback: Callable[[], object]) -> None:
        """Call ``callback`` once when the session aborts; at once if it has."""
        ...

    def abort(self, exc: BaseException) -> None:
        """Enter ABORTED with ``exc``, unless already aborted."""
        ...

    def close(self) -> None:
        """Stop the server; idempotent.

        Concurrency:
            Waits for an operation in progress to finish.
        """
        ...
