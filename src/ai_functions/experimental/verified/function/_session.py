"""The proof session behind ``verified.ai_function``; ``verified.ai_compile`` reaches it through that."""

from __future__ import annotations

import functools
import re
import textwrap
import threading
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ..dsl.types import PROP, Arrow, LeanType, Unsupported, render_type
from ..lean import LeanError, LeanProject, LeanSymbol
from ..lean.checker import CheckSpec, check_source, name_parts
from ..lean.errors import LeanTimeoutError
from ..lean.server import ElabResult, LeanServer, Source
from ..lean.types import EncodingError, RawLean, decode, encode, is_boundary, type_from_json
from .errors import ContractNotProved
from .types import LOGICAL_AXIOMS, Certificate, Guarantees, LeanTerm, Observation

Kind = Literal["definition", "provenance", "guarantee", "judgment", "value", "lemma"]


@dataclass(frozen=True)
class Decl:
    """One ledger record."""

    kind: Kind
    origin: str
    source: str | Source
    expected: str | None = None

    @property
    def comment(self) -> str:
        """The annotation, with the origin folded onto one line."""
        return " ".join(f"[{self.kind}] {self.origin}".split())


@dataclass(frozen=True)
class Outcome:
    """Feedback for one model operation."""

    ok: bool
    message: str
    code: str | None = None
    detail: str | None = None
    excerpt: str | None = None
    """What the model reads instead of ``code``, when ``code`` restates facts it already has."""


class _Feedback(Exception):
    """A model operation failed; the message is model feedback and the ledger is unchanged."""


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")

_ARTIFACT_HEADER = """\
Machine-checked certificate, produced by a verified AI Function.

  answer:   {answer}
  contract: {goal}

Before this certificate was issued, this file was compiled in a fresh Lean
process, and a trusted checker replayed every declaration through Lean's kernel,
compared every harness declaration with its independently elaborated statement,
and collected the axioms the final theorem depends on. Recompiling it needs the
project it was produced in. `#print axioms` at the end lists the trust base:
Lean's logical axioms and the observation axioms the proof uses.

  [definition]  an argument of the call, or declarations the agent wrote.
  [provenance]  a tool observation: the value, and an axiom equating the opaque
                symbol at the logged arguments with it.
  [guarantee]   a claim a tool made about the value it returned, at that call.
  [judgment]    the agent's reading of an allowlisted symbol, as an axiom. Only a
                human can check a reading: re-read the input and the quoted
                justification, and reject the certificate if it is wrong.
  [value]       the returned Lean term; the harness fixes its type.
  [lemma]       the proof of the goal.
"""

_TACTIC_HINT = "Tactic proofs start with `by`, for example `by simp`; anything else must be a proof term."
_INSPECT_HINT = "Use #check to inspect a declaration's type or #print to inspect its definition."
_COLUMNS = (
    "If the rest belongs to a tactic block, check its indentation. Tactic blocks are column-sensitive: "
    "every tactic in a block starts at the column of its first tactic, a line indented deeper continues "
    "the tactic above it, and a line indented less ends the block."
)


def _block_indentation_hint(result: ElabResult) -> str:
    """A hint when Lean ended a declaration mid-line: usually a mis-indented tactic."""
    for m in result.messages:
        if m.severity == "error" and m.text.endswith("expected command") and (m.col or 0) > 0:
            return (
                f"\n\nHint: Lean ended a declaration before line {m.line}, column {m.col}, "
                f"and could not read the rest as a command. {_COLUMNS}"
            )
    return ""


def _proof_indentation_hint(parsed: ElabResult) -> str:
    """A hint when a `by` proof parsed as a complete term that ended on a later line."""
    ended = parsed.raw.get("ended")
    if not parsed.raw.get("leading_by") or not isinstance(ended, dict) or ended.get("line", 1) <= 1:
        return ""
    return (
        f"\n\nHint: Lean's proof ended at line {ended['line']}, column {ended['col']}, "
        f"before the end of what you submitted. {_COLUMNS}"
    )


def _slot_errors(
    result: ElabResult, slots: Sequence[tuple[str, str, LeanType]], spans: Sequence[tuple[int, int]]
) -> str:
    """Lean's errors, each under the slot whose byte span Lean positions it in."""
    grouped: list[list[str]] = [[] for _ in slots]
    unplaced = []
    for m in result.messages:
        if m.severity != "error":
            continue
        index = next((i for i, (a, b) in enumerate(spans) if m.offset is not None and a <= m.offset < b), None)
        if index is None:
            unplaced.append(f"{m.line}:{m.col}: {m.text}" if m.line else m.text)
        else:
            grouped[index].append(m.text)
    parts = [
        f"{label}: Lean rejected `{source}`:\n" + "\n".join(texts)
        for (label, source, _), texts in zip(slots, grouped, strict=True)
        if texts
    ]
    if unplaced or not parts:
        parts.append("\n".join(unplaced) or result.errors)
    return "\n\n".join(parts)


def _unshowable(lean_type: LeanType) -> str | None:
    """Why `lean_eval` cannot show a value of this type, or ``None`` if it can."""
    if lean_type == PROP:
        return "The term is a proposition (Prop), which has no value to show. Write `decide (…)` to get a Bool."
    if is_boundary(lean_type):
        return None
    return (
        "lean_eval shows values of Nat, Int, String and Bool, and lists and tuples of them; "
        f"this term has type {render_type(lean_type)}."
        + (" Apply the function to inputs." if isinstance(lean_type, Arrow) else "")
    )


def _raw(value: object) -> bool:
    return isinstance(value, RawLean) or isinstance(value, (list, tuple)) and any(map(_raw, value))


def _apply(symbol: LeanSymbol, arguments: Sequence[str | Source]) -> Source:
    return Source.join(" ", [symbol.name, *("(" + a + ")" for a in arguments)])


def _written(written: Sequence[str], literals: Sequence[str]) -> tuple[list[str], list[Source]] | None:
    """The arguments as the model wrote them, as text and as confined model source; ``None`` if the literals."""
    texts = [textwrap.dedent(text).strip() for text in written]
    if texts == list(literals):
        return None
    # A line comment in model text would swallow the closing parenthesis.
    return texts, [Source.model(t) if "--" not in t else "\n" + Source.model(t) + "\n" for t in texts]


def _as_written(
    kind: Kind, axiom: str, symbol: LeanSymbol, written: Sequence[str], literals: Sequence[str], value: str
) -> tuple[Decl, str] | None:
    """A theorem restating ``axiom`` at the arguments as the model wrote them, and its statement on one line.

    ``None`` when the arguments are the literals. The proof is ``axiom``: the arguments reduce to the literals.
    """
    if (restated := _written(written, literals)) is None:
        return None
    texts, arguments = restated
    source = f"theorem {axiom}_as_written : " + _apply(symbol, arguments) + f" = {value} := {axiom}"
    shown = f"{axiom}_as_written : {_apply(symbol, texts).text} = {value}"
    return Decl(kind, f"{axiom}, restated at the arguments as written", source), shown


_MARKER = "__ai_functions_argument{}__"


def _splice(claim: str, arguments: Sequence[str | Source]) -> Source:
    """``claim`` with each argument marker replaced by that argument, parenthesized."""
    parts = re.split(r"__ai_functions_argument(\d+)__", claim)
    pieces: list[str | Source] = [parts[0]]
    for index, text in zip(parts[1::2], parts[2::2], strict=True):
        pieces += ["(", arguments[int(index)], ")", text]
    return Source.join("", pieces)


def _check_opaque(symbol: LeanSymbol, *, judged: bool = False) -> None:
    """Require an opaque symbol with explicit parameters and boundary argument and result types.

    A judged symbol's result may have any type the structural model describes, such as ``Prop``.
    """
    info = symbol.info
    types = [p.type for p in info.parameters]
    result = info.result or PROP
    if (
        info.kind != "opaque"
        or not info.simple
        or not all(p.explicit for p in info.parameters)
        or not all(map(is_boundary, types))
        or not (is_boundary(result) or judged and not isinstance(result, Unsupported))
    ):
        raise LeanError(
            f"{symbol.name} must be an opaque with explicit parameters whose argument and result types are "
            f"Int, Nat, Bool, String, or lists and products of them; it is a {info.kind} : {info.type_str}"
        )


def _guarded(method):
    """Serialize a session method; in ABORTED re-raise the abort, and a harness failure aborts."""

    @functools.wraps(method)
    def run(self: ProofSession, *args, **kwargs):
        with self._lock:
            if self._fatal is not None:
                raise self._fatal
            try:
                return method(self, *args, **kwargs)
            except (LeanError, EncodingError, ContractNotProved) as exc:
                self.abort(exc)
                raise

    return run


def _operation(method):
    """A guarded model operation: model feedback is a failed ``Outcome``."""

    @_guarded
    @functools.wraps(method)
    def run(self: ProofSession, *args, **kwargs) -> Outcome:
        try:
            return method(self, *args, **kwargs)
        except _Feedback as exc:
            return Outcome(False, str(exc))

    return run


class ProofSession:
    """One ledger over a prepared project, owned and closed by its consumer."""

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
        self._contract = contract
        self._project = contract.project
        self._term = term_result
        self._max_attempts = max_attempts
        self._timeout = timeout
        self._lock = threading.RLock()
        self._fatal: BaseException | None = None
        self._callback: Callable[[], object] | None = None
        self._records: list[Decl] = []
        # The conflict index across tools and judgments, in commit order.
        self._index: dict[tuple[str, tuple[str, ...]], Observation] = {}
        self._diagnostics: list[str] = []
        self._certificate: Certificate | None = None
        self._result: object = None
        self._elaborations = 0
        info = contract.info
        if not info.prop or not info.simple or not info.parameters or not all(p.explicit for p in info.parameters):
            raise LeanError(
                f"The contract {contract.name} must be a monomorphic Prop with explicit parameters, "
                f"the result first: {info.type_str}"
            )
        result, *inputs = info.parameters
        if not is_boundary(result.type) and not (term_result and not isinstance(result.type, Unsupported)):
            raise LeanError(f"The contract's result type {result.type_str} cannot cross the certified data boundary")
        for p in inputs:
            if not _IDENTIFIER.fullmatch(p.name) or p.name not in arguments and not term_result:
                raise LeanError(f"The contract's input {p.name!r} is not a parameter of the function")
        # In term mode an input without an argument is quantified: the answer is a function of those inputs.
        self._free = tuple(p for p in inputs if p.name not in arguments)
        self._result_type = functools.reduce(lambda r, p: Arrow(p.type, r), reversed(self._free), result.type)
        self._inputs = tuple(p.name for p in inputs)
        self._names = {f"H.{name}" for name in self._inputs}
        self._judgments = {symbol.name: symbol for symbol in judgments}
        # Symbols checked to be observable or judgeable, each once.
        self._opaque: set[tuple[str, bool]] = set()
        for symbol in judgments:
            self._check_observable(symbol, judged=True)
        self._answer = self.fresh("answer")
        self._proof = self.fresh("final")
        bindings = [self._bind(p.name, p.type, arguments[p.name]) for p in inputs if p.name in arguments]
        # A LeanTerm is closed: it must elaborate over the project alone, and is bound before the other inputs.
        bindings.sort(key=lambda d: not isinstance(d.source, Source))
        self._server = LeanServer(self._project, timeout=timeout)
        try:
            for d in bindings:
                if isinstance(d.source, Source) and not (probe := self._server.probe(d.source)).ok:
                    raise LeanError(f"{d.origin} does not elaborate over the project alone:\n{probe.errors}")
            if bindings and not self._server.commit(Source.join("\n\n", (d.source for d in bindings))).ok:
                raise LeanError(f"Could not bind the inputs of {contract.name}")
        except BaseException:
            self._server.close()
            raise
        self._records.extend(bindings)

    def _bind(self, name: str, lean_type: LeanType, value: object) -> Decl:
        """The binding of one input: a Python value encoded, or a ``LeanTerm``'s closed expression."""
        head = f"def H.{name} : {render_type(lean_type)} := "
        if not isinstance(value, LeanTerm):
            source = head + encode(value, lean_type, slot=f"input {name}")
            return Decl("definition", f"the AI function's `{name}` argument", source, source)
        if value.certificate.project_fingerprint != self._project.fingerprint:
            raise LeanError(f"The LeanTerm for input {name} was certified in a different project")
        if value.lean_type != render_type(lean_type):
            raise LeanError(f"The LeanTerm for input {name} has type {value.lean_type}, not {render_type(lean_type)}")
        # The expression is model-originated, so confined; being closed, it is its own twin.
        source = head + "(\n" + Source.model(value.expression) + "\n)"
        return Decl("definition", f"the AI function's `{name}` argument, a LeanTerm", source, source.text)

    @property
    def project(self) -> LeanProject:
        return self._project

    @property
    def result_type(self) -> LeanType:
        return self._result_type

    @property
    def certificate(self) -> Certificate | None:
        return self._certificate

    @property
    def result(self) -> object | LeanTerm | None:
        return self._result

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._index.values())

    def _allocate(self, namespace: str, stem: str) -> str:
        with self._lock:
            if not _IDENTIFIER.fullmatch(stem):
                raise LeanError(f"Expected a plain Lean identifier, got {stem!r}")
            n = 1
            while f"{namespace}.{stem}{n}" in self._names:
                n += 1
            self._names.add(name := f"{namespace}.{stem}{n}")
            return name

    def fresh(self, stem: str) -> str:
        return self._allocate("H", stem)

    def _goal(self, answer: str | Source) -> Source:
        free = [p.name for p in self._free]
        if free:
            answer = Source.join(" ", ["(" + answer + ")", *free])
        inputs = [name if name in free else f"H.{name}" for name in self._inputs]
        binders = "".join(f"({p.name} : {render_type(p.type)}) " for p in self._free)
        return (f"∀ {binders[:-1]}, " if free else "") + _apply(self._contract, [answer, *inputs])

    def goal(self, answer: str) -> str:
        return self._goal(answer).text

    def _render(self, records: Sequence[Decl]) -> Source:
        imports = "\n".join(f"import {name}" for name in self._server.imports)
        blocks = [f"-- {d.comment}\n" + Source.of(d.source) for d in records]
        return Source.join("\n\n", [imports, self._project.prepare().prelude, *blocks]) + "\n"

    def ledger(self) -> str:
        with self._lock:
            return self._render(self._records).text

    def _lean(self, op: str, *args, **kwargs):
        """Send one request; after a timeout, commit the records again to a fresh server."""
        self._elaborations += 1
        try:
            return getattr(self._server, op)(*args, **kwargs)
        except LeanTimeoutError:
            self._server.close()
            self._server = LeanServer(self._project, timeout=self._timeout)
            if not self._server.commit(Source.join("\n\n", (d.source for d in self._records))).ok:
                raise LeanError("The ledger did not commit again to a fresh Lean server after a timeout") from None
            raise _Feedback(
                f"Lean exceeded its {self._timeout:g} s limit and nothing was kept. "
                "Try a smaller computation or a more direct proof."
            ) from None

    def _commit(self, decls: Sequence[Decl], observation: Observation | None = None) -> ElabResult:
        result = self._lean("commit", Source.join("\n\n", (d.source for d in decls)))
        if result.ok:
            self._records.extend(decls)
            if observation is not None:
                self._index[observation.symbol, observation.arguments] = observation
        elif any(d.expected is not None for d in decls):
            raise LeanError(f"Lean rejected harness declarations:\n{result.errors}")
        return result

    @staticmethod
    def _text(text: object, label: str = "input") -> str:
        if not isinstance(text, str) or not text.strip() or len(text) > 131_072:
            raise _Feedback(f"The {label} must be nonempty Lean source of at most 131072 characters.")
        return textwrap.dedent(text).strip()

    def _one_term(self, text: object, label: str = "input") -> str:
        text = self._text(text, label)
        parsed = self._lean("parse_term", text)
        if not parsed.ok:
            raise _Feedback(f"The {label} must be one complete Lean term:\n{parsed.errors}")
        return text

    def _check_observable(self, symbol: LeanSymbol, *, judged: bool = False) -> None:
        if (symbol.name, judged) not in self._opaque:
            if symbol.project is not self._project:
                raise LeanError(f"{symbol.name} belongs to a different LeanProject instance")
            _check_opaque(symbol, judged=judged)
            self._opaque.add((symbol.name, judged))

    @_guarded
    def evaluate(self, slots: Sequence[tuple[str, str, LeanType]]) -> list[object]:
        try:
            return self._evaluate(slots)
        except _Feedback as exc:
            raise ValueError(str(exc)) from None

    def _evaluate(self, slots: Sequence[tuple[str, str, LeanType]]) -> list[object]:
        if not slots:
            return []
        names, spans, block = [], [], Source()
        for label, expression, lean_type in slots:
            expression = self._one_term(expression, label)
            names.append(name := self.fresh("value"))
            block = block + ("\n\n" if block.text else "")
            start = len(block.text.encode())
            block = block + f"def {name} : {render_type(lean_type)} := (\n" + Source.model(expression) + "\n)"
            spans.append((start, len(block.text.encode())))
        result, values = self._lean("evaluate", block, names)
        if not result.ok:
            raise _Feedback(_slot_errors(result, slots, spans))
        if len(values) != len(slots):
            raise LeanError("Lean returned an incomplete evaluation")
        decoded = []
        for (label, expression, lean_type), value in zip(slots, values, strict=True):
            if isinstance(value, dict):
                raise _Feedback(f"{label}: reducing `{expression}` failed: {value['error']}")
            if value is None:
                raise _Feedback(
                    f"{label}: `{expression}` does not reduce to a closed value. Use a literal or transparent "
                    "ledger definitions; the value of an opaque application comes only from an observation."
                )
            decoded.append(decode(value, lean_type, slot=label))
        pins = [
            f"example : {name} = ({encode(value, t)} : {render_type(t)}) := by rfl"
            for name, value, (_, _, t) in zip(names, decoded, slots, strict=True)
        ]
        if not self._lean("probe", block + "\n\n" + "\n\n".join(pins)).ok:
            raise LeanError("Decoded values failed their kernel equality check")
        return decoded

    @_operation
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
        self._check_observable(symbol)
        info = symbol.info
        assert info.result is not None
        if _raw(value) or _raw(arguments) or len(arguments) != len(info.parameters):
            raise LeanError(f"An observation of {symbol.name} takes {len(info.parameters)} Python arguments")
        args = tuple(
            encode(a, p.type, slot=f"parameter {p.name} of {symbol.name}")
            for a, p in zip(arguments, info.parameters, strict=True)
        )
        literal = encode(value, info.result, slot=f"the result of {symbol.name}")
        if prior := self._index.get((symbol.name, args)):
            if prior.value != literal:
                raise LeanError(f"Conflicting observations: {prior.name} records {prior.value}, the tool {literal}")
            return Outcome(True, f"Already observed as {prior.name}; reuse {next(iter(prior.axioms))}.")
        name = self.fresh(stem)
        typed = f"({literal} : {render_type(info.result)})"
        # A builder sees each argument as a marker, so its claims can be stated at the
        # literals, which the axioms use, and again at the arguments as written.
        markers = [RawLean(_MARKER.format(i)) for i in range(len(args))]
        claims = guarantees(RawLean(typed), *markers) if callable(guarantees) else guarantees
        claims = () if claims is None else (claims,) if isinstance(claims, str) else tuple(claims)
        if not all(isinstance(claim, str) for claim in claims):
            raise LeanError("Tool guarantees must be Lean proposition strings")
        axioms = {f"{name}_spec": f"{_apply(symbol, args).text} = {typed}"}
        axioms.update((f"{name}_contract{i}", _splice(claim, args).text) for i, claim in enumerate(claims, 1))
        call = f"{origin} -> {literal}"
        decls = [Decl("provenance", call, source := f"def {name} : {render_type(info.result)} := {literal}", source)]
        for i, (axiom, proposition) in enumerate(axioms.items()):
            kind: Kind = "guarantee" if i else "provenance"
            decls.append(Decl(kind, call, source := f"axiom {axiom} : {proposition}", source))
        message = f"Recorded {name}, with {', '.join(axioms)}."
        excerpt = None
        if written is not None and (forms := _written(written, args)):
            spec = _as_written("provenance", f"{name}_spec", symbol, written, args, typed)
            assert spec is not None
            # Restate each fact at the arguments as written, after the literal one it is proved
            # from; the model then reads those, and not the literals, which may be large.
            texts, sources = forms
            lines = [f"Recorded {spec[1]}"]
            decls.append(spec[0])
            for i, claim in enumerate(claims, 1):
                axiom = f"{name}_contract{i}"
                if "__ai_functions_argument" not in claim:
                    lines.append(f"Recorded {axiom} : {claim}")
                    continue
                source = f"theorem {axiom}_as_written : " + _splice(claim, sources) + f" := {axiom}"
                decls.append(Decl("guarantee", f"{axiom}, restated at the arguments as written", source))
                lines.append(f"Recorded {axiom}_as_written : {_splice(claim, texts).text}")
            message = "\n".join([*lines, message])
            excerpt = Source.of(decls[0].source).text
        code = "\n\n".join(Source.of(d.source).text for d in decls)
        self._commit(decls, Observation(name, "tool", symbol.name, args, literal, origin, MappingProxyType(axioms)))
        return Outcome(True, message, code=code, excerpt=excerpt)

    @_operation
    def judge(self, symbol: str, arguments: Sequence[str], value: str, justification: str) -> Outcome:
        target = self._judgments.get(symbol)
        if target is None:
            raise _Feedback(f"`{symbol}` cannot be judged. Judgeable symbols: {', '.join(self._judgments)}.")
        if not isinstance(justification, str) or len(justification.split()) < 4:
            raise _Feedback("Justify the reading: one sentence of reasoning from the input to this value.")
        info = target.info
        if not isinstance(arguments, (list, tuple)) or len(arguments) != len(info.parameters):
            raise _Feedback(f"{symbol} takes {len(info.parameters)} arguments: {info.type_str}")
        result_type = info.result or PROP
        slots = [(f"argument {p.name}", a, p.type) for a, p in zip(arguments, info.parameters, strict=True)]
        if is_boundary(result_type):
            *args, result = self._evaluate([*slots, ("value", value, result_type)])
            literal = encode(result, result_type)
        else:
            # A proposition or theory value: its closed text is the canonical form.
            args, literal = self._evaluate(slots), self._closed(value, result_type, "value")
        literals = tuple(encode(a, p.type) for a, p in zip(args, info.parameters, strict=True))
        if prior := self._index.get((target.name, literals)):
            if prior.value != literal:
                return Outcome(
                    False,
                    f"{prior.name} already records the value {prior.value}; the earlier value stands. "
                    "Build your proof on it.",
                )
            return Outcome(True, f"Already recorded as {prior.name}; cite that name.")
        short = name_parts(target.name)[-1]
        name = self._allocate("J", short if _IDENTIFIER.fullmatch(short) else "judgment")
        typed = f"({literal} : {render_type(result_type)})"
        proposition = f"{_apply(target, literals).text} = {typed}"
        decls = [Decl("judgment", f'{name}: "{justification}"', source := f"axiom {name} : {proposition}", source)]
        message = f"Recorded {name} : {proposition}"
        if restated := _as_written("judgment", name, target, arguments, literals, typed):
            decls.append(restated[0])
            message = f"Recorded {restated[1]}\n{message}"
        observation = Observation(
            name, "judgment", target.name, literals, literal, justification, MappingProxyType({name: proposition})
        )
        self._commit(decls, observation)
        return Outcome(True, message, code="\n\n".join(Source.of(d.source).text for d in decls))

    def _closed(self, expression: str, lean_type: LeanType, label: str) -> str:
        """``expression`` as closed text over the project: ledger definitions unfolded, full names."""
        name = self.fresh("value")
        block = (
            f"def {name} : {render_type(lean_type)} := (\n" + Source.model(self._one_term(expression, label)) + "\n)"
        )
        result, text = self._lean("closed_term", block, name)
        if text is None:
            raise _Feedback(f"{label}: Lean rejected `{expression}`:\n{result.errors}")
        return text

    @_operation
    def append(self, code: str) -> Outcome:
        block = "namespace A\n" + Source.model(self._text(code, "code")) + "\nend A"
        result = self._commit([Decl("definition", "agent: lean", block)])
        if not result.ok:
            return Outcome(False, result.errors + _block_indentation_hint(result), code=block.text)
        declared = sorted({name for name in result.declared if name.startswith("A.")})
        message = "Declarations committed." + (f"\nNow available: {', '.join(declared)}." if declared else "")
        if any(name.startswith("A.A.") for name in declared):
            message += "\nHint: the harness already wraps code in namespace A; write `def foo`, not `def A.foo`."
        return Outcome(True, message, code=block.text, detail=result.stdout.strip() or None)

    @_operation
    def inspect(self, expression: str) -> Outcome:
        expression = self._text(expression, "expression")
        token = expression.split(maxsplit=1)[0]
        if token in ("#check", "#print"):
            # One probe for every #check/#print command that starts a line.
            parts = re.split(r"(?m)^(#check|#print)(?=\s|$)", expression)[1:]
            pairs = zip(parts[::2], parts[1::2], strict=True)
            code = Source.join("\n\n", (f"{c} " + Source.model(self._one_term(t, "term")) for c, t in pairs))
            result = self._lean("probe", code)
            return Outcome(result.ok, (result.stdout.strip() if result.ok else result.errors) or "Checked.", code.text)
        if token in ("#eval", "#reduce"):
            raise _Feedback(f"Write the term alone, without `{token}`; only #check and #print may precede it.")
        if token == "do":
            raise _Feedback("A `do` block needs a monad. Write `Id.run do …` for a pure computation.")
        name = self.fresh("value")
        block = f"def {name} := (\n" + Source.model(self._one_term(expression, "expression")) + "\n)"
        # `whnf` reduces the value in the elaborator: no compiled code runs.
        result, values = self._lean("evaluate", block, [name])
        if not result.ok:
            return Outcome(False, f"{result.errors}\n{_INSPECT_HINT}", code=block.text)
        [shape] = result.raw["types"]
        lean_type = type_from_json(shape["type"], shape["type_str"])
        problem = _unshowable(lean_type)
        if problem is None and isinstance(values[0], dict):
            problem = f"Reducing the term failed: {values[0]['error']}"
        if problem is None and values[0] is None:
            problem = (
                "The term does not reduce to a closed value: it depends on an opaque symbol, "
                "or its reduction got stuck. Only literals and transparent definitions reduce."
            )
        if problem is not None:
            return Outcome(False, problem, code=block.text)
        return Outcome(True, encode(decode(values[0], lean_type), lean_type), code=block.text)

    @_guarded
    def fail_attempt(self, diagnostic: str) -> None:
        """Count an attempt that ended without a certified answer; past ``max_attempts`` retries, abort."""
        self._diagnostics.append(diagnostic)
        if len(self._diagnostics) > self._max_attempts:
            attempts = len(self._diagnostics)
            raise ContractNotProved(f"No proof was accepted in {attempts} attempt(s).", tuple(self._diagnostics))

    @_operation
    def submit(self, answer: str, proof: str) -> Outcome:
        try:
            outcome = self._certify(answer, proof)
        except _Feedback as exc:
            outcome = Outcome(False, str(exc))
        if not outcome.ok:
            self.fail_attempt(outcome.message)
        return outcome

    def _certify(self, answer: str, proof: str) -> Outcome:
        proof = self._text(proof, "proof")
        lean_type = render_type(self._result_type)
        if self._term:
            value: object = self._one_term(answer, "answer")
            binding = f"def {self._answer} : {lean_type} := (\n" + Source.model(str(value)) + "\n)"
            submission = [Decl("value", "the returned Lean term", binding, f"axiom {self._answer} : {lean_type}")]
            goal, shown = self.goal(self._answer), self._goal(Source.model(str(value)))
        else:
            [value] = self._evaluate([("answer", answer, self._result_type)])
            goal = self.goal(encode(value, self._result_type))
            shown, submission = Source(goal), []
        parsed = self._lean("parse_term", proof)
        hint = "" if parsed.raw.get("leading_by") else f"\n{_TACTIC_HINT}"
        if not parsed.ok:
            return Outcome(False, parsed.errors + hint + _proof_indentation_hint(parsed), code=proof)
        theorem = f"theorem {self._proof} : {goal} := (show " + shown + " from (\n" + Source.model(proof) + "\n))"
        submission.append(Decl("lemma", "the proof of the goal", theorem, f"axiom {self._proof} : {goal}"))
        block = Source.join("\n\n", (d.source for d in submission))
        result, inventory = self._lean("collect_axioms", self._proof, extra=block)
        if not result.ok or inventory is None:
            return Outcome(False, result.errors + hint, code=block.text)
        allowed = LOGICAL_AXIOMS | {axiom for o in self._index.values() for axiom in o.axioms}
        if outside := sorted(set(inventory) - allowed):
            return Outcome(False, f"The proof depends on axioms outside its permitted base: {', '.join(outside)}")
        records = [*self._records, *submission]
        shown_answer = " ".join((str(value) if self._term else encode(value, self._result_type)).split())
        header = _ARTIFACT_HEADER.format(answer=shown_answer, goal=" ".join(goal.split()))
        artifact = (
            "".join(f"-- {line}\n" if line else "--\n" for line in header.splitlines())
            + "\n"
            + self._render(records)
            + f"\n#print axioms {self._proof}\n"
        )
        audited = (self._proof, self._answer) if self._term else (self._proof,)
        twins = [self._project.prepare().prelude, *(d.expected for d in records if d.expected is not None)]
        spec = CheckSpec(expected="\n\n".join(twins) + "\n", axioms=audited, imports=self._server.imports)
        self._elaborations += 1
        try:
            cold = check_source(self._project, artifact, spec, timeout=self._timeout)
        except LeanTimeoutError:
            raise _Feedback(f"The final check exceeded its {self._timeout:g} s limit.") from None
        if not cold.ok:
            raise LeanError(f"The trusted checker rejected an artifact the warm check accepted:\n{cold.errors}")
        if any(name not in cold.axioms for name in audited):
            raise LeanError("The trusted checker omitted an axiom inventory")
        axioms = tuple(sorted({axiom for name in audited for axiom in cold.axioms[name]}))
        if set(axioms) != set(inventory) or not set(axioms) <= allowed:
            raise LeanError(f"The cold axiom inventory {axioms} disagrees with the warm inventory {inventory}")
        expression = str(value)
        if self._term:
            # Portable when closed; an answer that calls recursive helpers, say, keeps its text as written.
            expression = self._lean("closed_term", submission[0].source, self._answer)[1] or expression
        self._certificate = Certificate(
            answer=deepcopy(value),
            goal=goal,
            proof=proof,
            artifact=artifact.text,
            project_fingerprint=self._project.fingerprint,
            axioms=axioms,
            observations=MappingProxyType({o.name: o for o in self._index.values()}),
            elaborations=self._elaborations,
        )
        self._result = (
            LeanTerm(expression, lean_type, self._answer, self._proof, self._certificate, artifact)
            if self._term
            else value
        )
        return Outcome(True, f"Certified {shown_answer}. Axioms: {', '.join(axioms) or 'none'}.", code=block.text)

    def check_result(self, returned: object) -> None:
        with self._lock:
            if self._fatal is not None:
                raise self._fatal
            if self._certificate is None:
                message = "Use lean_submit(answer, proof) to obtain a checked certificate first."
                self.fail_attempt(message)
                raise ContractNotProved(message)
            if self._term:
                if returned != self._result:
                    raise ContractNotProved("The returned Lean term is not the certified one.")
                return
            try:
                same = encode(returned, self._result_type) == encode(self._result, self._result_type)
            except EncodingError as exc:
                raise ContractNotProved(f"The returned value has the wrong Lean type: {exc}") from exc
            if not same:
                raise ContractNotProved(f"Returned {returned!r}, but the certificate proves {self._result!r}.")

    def on_abort(self, callback: Callable[[], object]) -> None:
        with self._lock:
            self._callback = callback
            if self._fatal is not None:
                callback()

    def abort(self, exc: BaseException) -> None:
        with self._lock:
            if self._fatal is None:
                self._fatal = exc
                if self._callback is not None:
                    self._callback()

    def close(self) -> None:
        with self._lock:
            self._server.close()
