"""Deterministic, typed translation of a deliberately small Python contract subset.

No source, prompt builder, validator, or generated Python is executed here. A
validator is translated to its *successful completion* predicate, including
branches, early returns, assertions and explicit failures. Unsupported syntax
is an error, including syntax on an unreachable path.

Lean's operations are total where Python's raise, so every expression also has
a definedness condition: the inputs for which Python evaluates it without
raising. A contract holds only where everything it evaluates is defined, in
Python's evaluation order and with Python's short-circuiting.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import math
import operator
import re
import struct
import textwrap
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from ...ai_thread.postcondition import PostConditionResult
from .errors import ContractError

Scalar = int | bool | float | list[int]


class _Pairs:
    """Internal kind of a zip() or enumerate() domain; it never crosses the boundary."""


class _Tuple:
    """Internal kind of a pair element or a loop state; it never crosses the boundary."""


Kind = type[int] | type[bool] | type[float] | type[list] | type[_Pairs] | type[_Tuple]
_MAX_NODES = 4096
# Boolean nodes that denote a Bool value rather than logical structure: a Bool
# literal, input, or result, a component of a loop state, and the Bool-valued
# core Float classifiers.
_BOOL_VALUES = ("literal", "variable", "proj", "isfinite", "isnan", "isinf")
# Element types of iteration domains, as Lean binder types.
_ELEMENT_TYPES = {"int": "Int", "zip": "Int × Int", "enumerate": "Int × Nat"}
# Names a helper or parameter may not take, because the generated Lean uses them.
_RESERVED = frozenset(
    {
        "at", "by", "do", "else", "end", "fun", "have", "if", "in", "let", "match", "show", "then", "with",
        "where", "from", "def", "theorem", "abbrev", "instance", "structure", "class", "inductive",
        "namespace", "section", "open", "import", "export", "example", "axiom", "mutual", "private",
        "protected", "public", "noncomputable", "partial", "unsafe", "calc", "suffices", "obtain", "forall",
        "exists", "Type", "Prop", "Sort", "true", "false", "True", "False", "not", "and", "or", "return",
        "for", "unless", "break", "continue", "mut", "sorry", "admit", "decide", "min", "max", "abs", "id",
        "pre", "post", "implementation", "nativeEntry", "r",
        "pythonIndex", "pythonSlice", "pythonAt", "pythonRange",
    }
)  # fmt: skip
_GENERATED = re.compile(r"[a-z]\d+|(?:pre|post|loop)\d*|.*_defined")

_SLICE_HELPERS = """public def pythonIndex (length : Nat) (index : Int) : Nat :=
  (max 0 (min (Int.ofNat length) (if index < 0 then Int.ofNat length + index else index))).toNat
public def pythonSlice (xs : List Int) (start stop : Option Int) : List Int :=
  let first := (start.map (pythonIndex xs.length)).getD 0
  let last := (stop.map (pythonIndex xs.length)).getD xs.length
  (xs.drop first).take (last - first)
public theorem pythonIndex_ofNat (length index : Nat) (h : index <= length) :
    pythonIndex length (Int.ofNat index) = index := by
  simp [pythonIndex, Int.min_def, Int.max_def]
  omega
public theorem pythonSlice_prefix (xs : List Int) (n : Nat) (h : n <= xs.length) :
    pythonSlice xs none (some (Int.ofNat n)) = xs.take n := by
  change (xs.drop 0).take (pythonIndex xs.length (Int.ofNat n) - 0) = xs.take n
  rw [pythonIndex_ofNat xs.length n h]
  simp
public theorem pythonSlice_suffix (xs : List Int) (n : Nat) (h : n <= xs.length) :
    pythonSlice xs (some (Int.ofNat n)) none = xs.drop n := by
  change (xs.drop (pythonIndex xs.length (Int.ofNat n))).take
    (xs.length - pythonIndex xs.length (Int.ofNat n)) = xs.drop n
  rw [pythonIndex_ofNat xs.length n h]
  rw [← List.length_drop, List.take_length]
"""
# Python indexing, including negative indices. Out-of-range indices raise in
# Python; the specification's definedness condition excludes them.
_AT_HELPERS = """public def pythonAt (xs : List Int) (index : Int) : Int :=
  xs.getD (if index < 0 then Int.ofNat xs.length + index else index).toNat 0
public theorem pythonAt_ofNat (xs : List Int) (i : Nat) (h : i < xs.length) :
    pythonAt xs (Int.ofNat i) = xs[i] := by
  unfold pythonAt
  simp only [Int.ofNat_eq_natCast]
  rw [if_neg (by omega)]
  simp [h]
"""
# range() with a step other than 1. A zero step raises in Python; the
# specification's definedness condition excludes it.
_RANGE_HELPERS = """public def pythonRange (start stop step : Int) : List Int :=
  if 0 < step then
    List.map (fun k => start + step * Int.ofNat k) (List.range (Int.toNat ((stop - start + step - 1) / step)))
  else if step < 0 then
    List.map (fun k => start + step * Int.ofNat k) (List.range (Int.toNat ((start - stop - step - 1) / (-step))))
  else []
"""
# Every helper a specification may use; each module emits only what it needs.
SPEC_HELPERS = _SLICE_HELPERS + _AT_HELPERS + _RANGE_HELPERS


def kind_name(kind: Kind) -> str:
    """Describe a supported Python value type."""
    return {list: "list[int]", _Pairs: "pairs", _Tuple: "tuple"}.get(kind, kind.__name__)


def lean_type(kind: Kind) -> str:
    """Map Python value kinds to the corresponding logical and native types."""
    return {int: "Int", bool: "Bool", float: "Float", list: "List Int"}[kind]


def _kind(annotation: object) -> Kind | None:
    if any(annotation is value for value in (int, bool, float)):
        return annotation
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (int,):
        return list
    return None


def _lean_name(name: str) -> str | None:
    """Return a Python name unchanged if the generated Lean can use it, else None."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name not in _RESERVED and not _GENERATED.fullmatch(name):
        return name
    return None


@dataclass(frozen=True)
class Definition:
    """A helper function or a loop, emitted as a named Lean definition.

    A helper returning ``bool`` is a proposition; every other definition is a
    term. ``defined`` is the condition under which Python evaluates the body
    without raising; when it is not trivial it is emitted as ``<name>_defined``.
    """

    name: str
    parameters: tuple[tuple[str, Kind], ...]
    kind: Kind
    result_type: str
    body: Expr
    defined: Expr
    proposition: bool

    @property
    def binders(self) -> str:
        """Typed Lean binders for the parameters."""
        return " ".join(f"({name} : {lean_type(kind)})" for name, kind in self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        """Every name this definition introduces into the generated module."""
        return (self.name, f"{self.name}_defined") if self.defined != TRUE else (self.name,)

    def declarations(self) -> list[str]:
        """Emit the definition and, if needed, its definedness condition."""
        head = f"{self.name} {self.binders}".strip()
        if self.proposition:
            lines = [f"abbrev {head} : Prop := {self.body.prop()}"]
        else:
            lines = [f"def {head} : {self.result_type} := {self.body.term()}"]
        if self.defined != TRUE:
            lines.append(f"abbrev {f'{self.name}_defined {self.binders}'.strip()} : Prop := {self.defined.prop()}")
        return lines

    def bind(self, arguments: Sequence[object]) -> dict[str, object]:
        """Map evaluated arguments onto parameter names."""
        return {name: value for (name, _), value in zip(self.parameters, arguments, strict=True)}


@dataclass(frozen=True)
class Expr:
    """A typed expression with identical Python evaluation and emitted semantics."""

    op: str
    kind: Kind
    args: tuple[Expr, ...] = ()
    value: int | bool | float | str | tuple[typing.Any, ...] | Definition = False

    def evaluate(self, values: Mapping[str, typing.Any]) -> typing.Any:
        """Evaluate the supported expression without Python eval or assert."""
        op, args = self.op, self.args
        if op == "literal":
            return self.value
        if op == "variable":
            return values[str(self.value)]
        if op == "local":
            return args[0].evaluate(values)
        if op in ("list", "tuple"):
            items = [a.evaluate(values) for a in args]
            return items if op == "list" else tuple(items)
        if op in ("all", "any"):
            name = str(self.value)
            outcomes = (bool(args[1].evaluate({**values, name: item})) for item in args[0].evaluate(values))
            return all(outcomes) if op == "all" else any(outcomes)
        if op == "comprehension":
            name = typing.cast(tuple[str, str], self.value)[0]
            result = []
            for item in args[0].evaluate(values):
                scope = {**values, name: item}
                if args[2].evaluate(scope):
                    result.append(args[1].evaluate(scope))
            return result
        if op == "fold":
            state_name, element_name = typing.cast(tuple[str, str, str, str], self.value)[:2]
            state = args[0].evaluate(values)
            for item in args[2].evaluate(values):
                state = args[1].evaluate({**values, state_name: state, element_name: item})
            return state
        if op in ("call", "call_defined"):
            definition = typing.cast(Definition, self.value)
            scope = definition.bind([a.evaluate(values) for a in args])
            return (definition.body if op == "call" else definition.defined).evaluate(scope)
        if op == "if":
            branch = 1 if args[0].evaluate(values) else 2
            return args[branch].evaluate(values)
        if op == "and":
            return bool(args[0].evaluate(values)) and bool(args[1].evaluate(values))
        if op == "or":
            return bool(args[0].evaluate(values)) or bool(args[1].evaluate(values))
        if op == "implies":
            return not args[0].evaluate(values) or bool(args[1].evaluate(values))
        if op == "range":
            return list(range(*(a.evaluate(values) for a in args)))
        left = args[0].evaluate(values)
        if op in _UNARY:
            return _UNARY[op](left)
        if op == "ordered":
            return all(a <= b for a, b in zip(left, left[1:], strict=False))
        if op == "slice":
            begin = args[1].evaluate(values) if "start" in str(self.value) else None
            end = args[2].evaluate(values) if "stop" in str(self.value) else None
            return left[begin:end]
        if op == "pow":
            return left**self.value
        if op == "proj":
            return left[typing.cast(tuple[int, int], self.value)[0]]
        right = args[1].evaluate(values)
        if op in _BINARY:
            return _BINARY[op](left, right)
        raise AssertionError(f"Unknown expression operation: {op}")

    def is_value(self) -> bool:
        """Whether a Boolean node denotes Bool data rather than logical structure."""
        node = _bare(self)
        if node.op == "call":
            return not typing.cast(Definition, node.value).proposition
        return node.op in _BOOL_VALUES or node.op == "fold"

    def term(self) -> str:
        """Render as a Lean term; a Boolean node becomes a ``Bool`` value."""
        return self.bool_term() if self.kind is bool else self.lean()

    def bool_term(self) -> str:
        """Render a Boolean node as Bool data, as executable library functions require."""
        node = _bare(self)
        if node.is_value():
            return node.lean()
        if node.op == "if" and node.args[1].is_value() and node.args[2].is_value():
            return f"(if {node.args[0].prop()} then {node.args[1].lean()} else {node.args[2].lean()})"
        return f"(decide {self.prop()})"

    def lean(self) -> str:
        """Render a term: an Int, Float, or List Int value, or a Bool value.

        Logical structure never renders here; see :meth:`prop`. The only Bool
        terms are values (see :meth:`is_value`): Bool inputs and results, which
        cross the FFI as Bool, loop state, and the Bool-valued core Float
        classifiers.
        """
        op, args = self.op, self.args
        if op == "local":
            return args[0].lean()
        if op == "literal":
            if self.kind is bool:
                return "true" if self.value else "false"
            if self.kind is float:
                bits = struct.unpack(">Q", struct.pack(">d", typing.cast(float, self.value)))[0]
                return f"(Float.ofBits (0x{bits:016x} : UInt64))"
            integer = typing.cast(int, self.value)
            sign = "-" if integer < 0 else ""
            return f"({sign}0x{abs(integer):x} : Int)"
        if op == "variable":
            return str(self.value)
        if self.kind is bool and not self.is_value():
            raise AssertionError(f"Boolean {op!r} renders only as a proposition")
        if op == "if":
            # Lean's `if` takes a decidable proposition, so the condition is a Prop.
            return f"(if {args[0].prop()} then {args[1].term()} else {args[2].term()})"
        if op == "call":
            definition = typing.cast(Definition, self.value)
            return f"({definition.name} {' '.join(a.term() for a in args)})" if args else definition.name
        if op == "tuple":
            return f"({', '.join(a.term() for a in args)})"
        if op == "range":
            return _range_term(*args)
        if op == "comprehension":
            name, element_type = typing.cast(tuple[str, str], self.value)
            domain, element, guard = args
            text = domain.lean()
            if guard != TRUE:
                # List.filter executes its predicate, so the predicate is Bool-valued.
                text = f"(List.filter (fun ({name} : {element_type}) => {guard.bool_term()}) {text})"
            if not (element.op == "variable" and element.value == name):
                text = f"(List.map (fun ({name} : {element_type}) => {element.term()}) {text})"
            return text
        if op == "fold":
            state_name, element_name, state_type, element_type = typing.cast(tuple[str, str, str, str], self.value)
            init, step, domain = args
            return (
                f"(List.foldl (fun ({state_name} : {state_type}) ({element_name} : {element_type}) => "
                f"{step.term()}) {init.term()} {domain.lean()})"
            )
        rendered = [a.lean() for a in args]
        if op in _TEMPLATES:
            return _TEMPLATES[op].format(*rendered)
        if op == "list":
            return f"([{', '.join(rendered)}] : List Int)"
        if op == "slice":
            begin = f"(some {rendered[1]})" if "start" in str(self.value) else "none"
            end = f"(some {rendered[2]})" if "stop" in str(self.value) else "none"
            return f"(pythonSlice {rendered[0]} {begin} {end})"
        if op == "pow":
            return f"({rendered[0]} ^ ({self.value} : Nat))"
        if op == "proj":
            index, count = typing.cast(tuple[int, int], self.value)
            path = ".2" * index + (".1" if index < count - 1 else "")
            return f"({rendered[0]}{path})"
        if op == "abs":
            value = rendered[0]
            return f"(if {value} < 0 then (-{value}) else {value})" if self.kind is int else f"(Float.abs {value})"
        if op in ("min", "max"):
            first, second = rendered
            if self.kind is int:
                return f"({op} {first} {second})"
            # Python keeps the first argument unless a later one is strictly smaller
            # (larger); that differs from IEEE minimum/maximum when NaN is involved.
            smaller, larger = (second, first) if op == "min" else (first, second)
            return f"(if (Float.lt {smaller} {larger} = true) then {second} else {first})"
        if self.kind is list and op == "add":
            return f"({rendered[0]} ++ {rendered[1]})"
        operators = {"add": "+", "sub": "-", "mul": "*"}
        return f"({rendered[0]} {operators[op]} {rendered[1]})"

    def prop(self) -> str:
        """Render a Boolean-valued node as a decidable ``Prop``.

        All logical structure renders here. Bool appears only at the leaves: a
        Bool value ``b`` becomes ``b = true``, and Lean's Bool-valued Float
        comparisons and classifiers become ``f x = true``. Every atom stays
        decidable, so ``decide`` works and the specification remains executable.
        """
        op, args = self.op, self.args
        if op == "local":
            return args[0].prop()
        if self.kind is not bool:
            return self.lean()
        if op == "literal":
            return "True" if self.value else "False"
        if op in ("call", "call_defined") and (op == "call_defined" or not self.is_value()):
            definition = typing.cast(Definition, self.value)
            name = definition.name if op == "call" else f"{definition.name}_defined"
            return f"({name} {' '.join(a.term() for a in args)})" if args else name
        if self.is_value():
            return f"({self.lean()} = true)"
        if op == "and":
            return f"({args[0].prop()} ∧ {args[1].prop()})"
        if op == "or":
            return f"({args[0].prop()} ∨ {args[1].prop()})"
        if op == "implies":
            return f"({args[0].prop()} → {args[1].prop()})"
        if op == "not":
            return f"(¬{args[0].prop()})"
        if op == "if":
            return f"(if {args[0].prop()} then {args[1].prop()} else {args[2].prop()})"
        if op in ("all", "any"):
            quantifier = "∀" if op == "all" else "∃"
            return f"({quantifier} {self.value} ∈ {args[0].lean()}, {args[1].prop()})"
        if op == "contains":
            return f"({args[0].lean()} ∈ {args[1].lean()})"
        if op == "ordered":
            return f"(List.Pairwise (fun (a b : Int) => a <= b) {args[0].lean()})"
        left, right = args
        if left.kind is bool:
            # Two Bool values compare as values. Anything with logical structure
            # compares as propositions, so no Boolean connective reaches the spec.
            if left.is_value() and right.is_value():
                return f"({_bare(left).lean()} {'=' if op == 'eq' else '≠'} {_bare(right).lean()})"
            equivalence = f"({left.prop()} ↔ {right.prop()})"
            return equivalence if op == "eq" else f"(¬{equivalence})"
        if left.kind is float:
            # Lean's Float comparisons are Bool-valued core functions.
            name = {"eq": "beq", "ne": "beq", "lt": "lt", "le": "le", "gt": "lt", "ge": "le"}[op]
            first, second = (right, left) if op in ("gt", "ge") else (left, right)
            comparison = f"(Float.{name} {first.lean()} {second.lean()} = true)"
            return f"(¬{comparison})" if op == "ne" else comparison
        operators = {"eq": "=", "ne": "≠", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
        return f"({left.lean()} {operators[op]} {right.lean()})"

    def defined(self, *, strict: bool = False) -> Expr:
        """Return the condition under which Python evaluates this without raising.

        Short-circuiting follows Python: the right operand of ``and`` is only
        evaluated when the left holds. Quantifiers require every element to be
        defined, which is stricter than Python's early exit; stricter is sound,
        because a precondition then rejects more inputs and a postcondition asks
        the proof for more. ``strict`` drops short-circuiting altogether and
        requires every operation to be defined.
        """
        op, args = self.op, self.args
        if op in ("literal", "variable", "local", "fold"):
            # A local's condition was established at its assignment, and a loop's
            # at the loop statement.
            return TRUE
        if strict and op in ("and", "implies", "or", "if"):
            result = TRUE
            for argument in args:
                result = _and(result, argument.defined(strict=True))
            return result
        if op in ("and", "implies"):
            return _and(args[0].defined(), _implies(args[0], args[1].defined()))
        if op == "or":
            return _and(args[0].defined(), _implies(Expr("not", bool, (args[0],)), args[1].defined()))
        if op == "if":
            condition, yes, no = args
            branches = _and(_implies(condition, yes.defined()), _implies(Expr("not", bool, (condition,)), no.defined()))
            return _and(condition.defined(), branches)
        if op in ("all", "any"):
            body = args[1].defined(strict=strict)
            return _and(args[0].defined(strict=strict), _forall(args[0], body, str(self.value)))
        if op == "comprehension":
            domain, element, guard = args
            if strict:
                inner = _and(guard.defined(strict=True), element.defined(strict=True))
            else:
                inner = _and(guard.defined(), _implies(guard, element.defined()))
            name = typing.cast(tuple[str, str], self.value)[0]
            return _and(domain.defined(strict=strict), _forall(domain, inner, name))
        result = TRUE
        for argument in args:
            result = _and(result, argument.defined(strict=strict))
        if op == "call":
            definition = typing.cast(Definition, self.value)
            if definition.defined != TRUE:
                result = _and(result, Expr("call_defined", bool, args, value=definition))
        elif op in ("fdiv", "fmod", "div"):
            result = _and(result, _nonzero(args[1]))
        elif op == "sqrt":
            value = _literal_value(args[0])
            if value is None or value < 0:
                result = _and(
                    result, Expr("not", bool, (Expr("lt", bool, (args[0], Expr("literal", float, value=0.0))),))
                )
        elif op == "index":
            length = Expr("len", int, (args[0],))
            within = _and(Expr("le", bool, (Expr("neg", int, (length,)), args[1])), Expr("lt", bool, (args[1], length)))
            result = _and(result, within)
        elif op in ("lmin", "lmax"):
            result = _and(result, Expr("ne", bool, (Expr("len", int, (args[0],)), ZERO)))
        elif op == "index_of":
            result = _and(result, Expr("contains", bool, (args[1], args[0])))
        elif op == "range":
            result = _and(result, _nonzero(args[2]))
        return result

    def walk(self) -> Iterator[Expr]:
        """Yield this node and every node below it, excluding definition bodies."""
        yield self
        for argument in self.args:
            yield from argument.walk()

    def free(self) -> dict[str, Kind]:
        """Return the variables this expression uses without binding them."""
        op, args = self.op, self.args
        if op == "variable":
            return {str(self.value): self.kind}
        if op in ("all", "any"):
            inner = {k: v for k, v in args[1].free().items() if k != self.value}
            return {**args[0].free(), **inner}
        if op == "comprehension":
            name = typing.cast(tuple[str, str], self.value)[0]
            inner = {k: v for k, v in {**args[1].free(), **args[2].free()}.items() if k != name}
            return {**args[0].free(), **inner}
        if op == "fold":
            bound = set(typing.cast(tuple[str, str, str, str], self.value)[:2])
            inner = {k: v for k, v in args[1].free().items() if k not in bound}
            return {**args[0].free(), **args[2].free(), **inner}
        result: dict[str, Kind] = {}
        for argument in args:
            result.update(argument.free())
        return result


def _range_general(step: Expr) -> bool:
    return _literal_value(step) != 1


def _range_term(start: Expr, stop: Expr, step: Expr) -> str:
    if _range_general(step):
        return f"(pythonRange {start.lean()} {stop.lean()} {step.lean()})"
    if _literal_value(start) == 0:
        return f"(List.map Int.ofNat (List.range (Int.toNat {stop.lean()})))"
    return (
        f"(List.map (fun k0 => {start.lean()} + Int.ofNat k0) "
        f"(List.range (Int.toNat ({stop.lean()} - {start.lean()}))))"
    )


_UNARY: dict[str, Callable[[typing.Any], typing.Any]] = {
    "len": len,
    "sorted": sorted,
    "sum": sum,
    "lmin": min,
    "lmax": max,
    "abs": abs,
    "sqrt": math.sqrt,
    "isfinite": math.isfinite,
    "isnan": math.isnan,
    "isinf": math.isinf,
    "reverse": lambda items: list(reversed(items)),
    # Lean's List.zipIdx pairs each element with its index, in that order.
    "enumerate": lambda items: [(item, index) for index, item in enumerate(items)],
    "fst": operator.itemgetter(0),
    "snd": operator.itemgetter(1),
    "snd_nat": operator.itemgetter(1),
    "not": operator.not_,
    "neg": operator.neg,
}
_BINARY: dict[str, Callable[[typing.Any, typing.Any], typing.Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "fdiv": operator.floordiv,
    "fmod": operator.mod,
    "div": operator.truediv,
    "min": min,
    "max": max,
    "contains": lambda item, items: item in items,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "zip": lambda first, second: list(zip(first, second, strict=False)),
    "index": lambda items, index: items[index],
    "count": lambda items, item: items.count(item),
    "index_of": lambda items, item: items.index(item),
}
_TEMPLATES = {
    "len": "(Int.ofNat (List.length {0}))",
    # List.mergeSort executes its comparator, so the comparator is Bool-valued.
    "sorted": "(List.mergeSort {0} (fun a b => decide (a <= b)))",
    "sum": "(List.sum {0})",
    "lmin": "(Option.getD (List.min? {0}) (0 : Int))",
    "lmax": "(Option.getD (List.max? {0}) (0 : Int))",
    "reverse": "(List.reverse {0})",
    "zip": "(List.zip {0} {1})",
    "enumerate": "(List.zipIdx {0})",
    "fst": "({0}.1)",
    "snd": "({0}.2)",
    "snd_nat": "(Int.ofNat {0}.2)",
    "index": "(pythonAt {0} {1})",
    "count": "(Int.ofNat (List.count {1} {0}))",
    "index_of": "(Int.ofNat (List.idxOf {1} {0}))",
    # Python's // and % round toward negative infinity, like Int.fdiv/Int.fmod;
    # Lean's `/` on Int is Euclidean and differs for negative divisors.
    "fdiv": "(Int.fdiv {0} {1})",
    "fmod": "(Int.fmod {0} {1})",
    "div": "({0} / {1})",
    "sqrt": "(Float.sqrt {0})",
    "neg": "(-{0})",
    "isfinite": "(Float.isFinite {0})",
    "isnan": "(Float.isNaN {0})",
    "isinf": "(Float.isInf {0})",
}

TRUE = Expr("literal", bool, value=True)
FALSE = Expr("literal", bool, value=False)
ZERO = Expr("literal", int, value=0)
ONE = Expr("literal", int, value=1)


def _bare(expr: Expr) -> Expr:
    while expr.op == "local":
        expr = expr.args[0]
    return expr


def _literal_value(expr: Expr) -> int | float | None:
    """Return the value of a numeric literal, including a negated literal."""
    expr = _bare(expr)
    if expr.op == "literal" and expr.kind in (int, float):
        return typing.cast(int | float, expr.value)
    if expr.op == "neg" and expr.args[0].op == "literal" and expr.args[0].kind in (int, float):
        return -typing.cast(int | float, expr.args[0].value)
    return None


def _and(left: Expr, right: Expr) -> Expr:
    if left == TRUE:
        return right
    if right == TRUE:
        return left
    return Expr("and", bool, (left, right))


def _implies(condition: Expr, consequence: Expr) -> Expr:
    # A guard that is the condition itself, as in `if x != 0: ... // x`, needs nothing more.
    negated = condition.op == "not" and condition.args[0].op == "eq" and consequence.op == "ne"
    if consequence == TRUE or consequence == condition or negated and consequence.args == condition.args[0].args:
        return TRUE
    return Expr("implies", bool, (condition, consequence))


def _if(condition: Expr, yes: Expr, no: Expr) -> Expr:
    """Combine the definedness conditions of two branches."""
    if yes == no:
        return yes
    if no == TRUE:
        return _implies(condition, yes)
    if yes == TRUE:
        return _implies(Expr("not", bool, (condition,)), no)
    return Expr("if", bool, (condition, yes, no))


def _forall(domain: Expr, body: Expr, name: str) -> Expr:
    return TRUE if body == TRUE else Expr("all", bool, (domain, body), value=name)


def _nonzero(divisor: Expr) -> Expr:
    value = _literal_value(divisor)
    if value is not None and value != 0:
        return TRUE
    zero = Expr("literal", float, value=0.0) if divisor.kind is float else ZERO
    return Expr("ne", bool, (divisor, zero))


def _local(value: Expr, name: str) -> Expr:
    """Mark an assigned value whose definedness was checked at the assignment."""
    return value if value.defined() == TRUE else Expr("local", value.kind, (value,), value=name)


def _truth(expr: Expr) -> Expr:
    if expr.kind is bool:
        return expr
    if expr.kind is list:
        return Expr("ne", bool, (Expr("len", int, (expr,)), Expr("literal", int, value=0)))
    zero = Expr("literal", float, value=0.0) if expr.kind is float else Expr("literal", int, value=0)
    return Expr("ne", bool, (expr, zero))


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        return [name for element in target.elts for name in _target_names(element)]
    return []


def _assigned_names(body: Sequence[ast.stmt]) -> list[str]:
    """Names assigned anywhere in the statements, in order of first assignment."""
    names: list[str] = []
    for node in body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            names.extend(name for name in _target_names(target) if name not in names)
        if isinstance(node, ast.If):
            names.extend(name for name in _assigned_names(node.body + node.orelse) if name not in names)
    return names


def _parameters(free: Mapping[str, Kind]) -> tuple[tuple[str, Kind], ...]:
    """Order free variables as inputs, then the result, then helper parameters."""

    def key(item: tuple[str, Kind]) -> tuple[int, int, str]:
        name = item[0]
        if re.fullmatch(r"v\d+", name):
            return (0, int(name[1:]), "")
        return (1, 0, "") if name == "r" else (2, 0, name)

    return tuple(sorted(free.items(), key=key))


@dataclass(frozen=True)
class Contract:
    """An original validator's source location and successful-completion predicate."""

    name: str
    location: str
    predicate: Expr


@dataclass(frozen=True)
class Specification:
    """A frozen typed specification, independent of individual function calls."""

    name: str
    signature: inspect.Signature
    parameters: tuple[tuple[str, Kind], ...]
    output_type: Kind
    pre: tuple[Contract, ...]
    post: tuple[Contract, ...]
    guidance: str
    definitions: tuple[Definition, ...] = ()

    def bind(self, *args: object, **kwargs: object) -> dict[str, Scalar]:
        """Bind defaults, reject values outside the proved input types, and check the preconditions."""
        values = self.bind_types(*args, **kwargs)
        self.check_pre_conditions(values)
        return values

    def bind_types(self, *args: object, **kwargs: object) -> dict[str, Scalar]:
        """Bind defaults and reject values outside the proved input types."""
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values: dict[str, Scalar] = {}
        for index, (name, kind) in enumerate(self.parameters):
            value = bound.arguments[name]
            if type(value) is not kind:
                raise TypeError(f"{self.name}(): {name!r} must be {kind_name(kind)}, got {type(value).__name__}")
            if kind is list:
                # Snapshot before validation and compilation so caller mutation
                # cannot change the values covered by the preconditions.
                value = list(value)
                if any(type(item) is not int for item in value):
                    raise TypeError(f"{self.name}(): {name!r} must contain only int values")
            values[f"v{index}"] = value
        return values

    def check_pre_conditions(self, values: Mapping[str, Scalar]) -> None:
        """Reject inputs outside the preconditions, which the proof does not cover."""
        for contract in self.pre:
            if not contract.predicate.evaluate(values):
                raise ContractError(
                    f"Precondition {contract.name!r} failed for {self.name!r} ({contract.location}).",
                    function_name=self.name,
                )

    def identity(self) -> str:
        """Return canonical compilation inputs for a content-addressed cache."""
        return json.dumps(
            {
                "name": self.name,
                "parameters": [(name, kind_name(kind)) for name, kind in self.parameters],
                "output": kind_name(self.output_type),
                "pre": [(c.name, c.predicate.prop()) for c in self.pre],
                "post": [(c.name, c.predicate.prop()) for c in self.post],
                "definitions": [d.declarations() for d in self.definitions],
                "guidance": self.guidance,
            },
            sort_keys=True,
        )

    @property
    def binders(self) -> str:
        """Compiler binders use generated names to prevent name injection."""
        return " ".join(f"(v{i} : {lean_type(t)})" for i, (_, t) in enumerate(self.parameters))

    @property
    def arguments(self) -> str:
        """Return the generated positional argument list."""
        return " ".join(f"v{i}" for i in range(len(self.parameters)))

    @property
    def result_type(self) -> str:
        """Return the private compiler's scalar result type."""
        return lean_type(self.output_type)

    @property
    def definition_names(self) -> frozenset[str]:
        """Names of helpers and loops, which candidates may use."""
        return frozenset(name for d in self.definitions for name in d.names)

    def helpers(self) -> str:
        """Emit the fixed helper definitions this specification uses, and only those."""
        nodes = [node for c in self.pre + self.post for node in c.predicate.walk()]
        nodes += [node for d in self.definitions for e in (d.body, d.defined) for node in e.walk()]
        ops = {node.op for node in nodes}
        blocks = []
        if "slice" in ops:
            blocks.append(_SLICE_HELPERS)
        if "index" in ops:
            blocks.append(_AT_HELPERS)
        if any(node.op == "range" and _range_general(node.args[2]) for node in nodes):
            blocks.append(_RANGE_HELPERS)
        return "".join(blocks)

    def declarations(self) -> str:
        """Emit helpers and loops, then one abbreviation per contract, then their conjunction.

        ``pre`` and ``post`` reference the indexed contracts rather than
        repeating their text, so a contract appears exactly once. They are
        abbreviations so that instance search sees through them and the
        specification stays decidable.
        """
        lines = [line for d in self.definitions for line in d.declarations()]
        for prefix, contracts in (("pre", self.pre), ("post", self.post)):
            result = f"(r : {self.result_type}) " if prefix == "post" else ""
            applied = f"r {self.arguments}".strip() if prefix == "post" else self.arguments
            for i, c in enumerate(contracts):
                lines.append(f"abbrev {prefix}{i} {result}{self.binders} : Prop := {c.predicate.prop()}")
            conjunction = " ∧ ".join(f"{prefix}{i} {applied}".strip() for i in range(len(contracts)))
            lines.append(f"abbrev {prefix} {result}{self.binders} : Prop := {conjunction or 'True'}")
        return "\n".join(lines)


_BUILTINS = (builtins.all, builtins.any, builtins.len, builtins.sorted, builtins.sum, builtins.min, builtins.max)
_BUILTINS += (builtins.abs, builtins.list)


class _Registry:
    """Helpers and loops shared by every contract of one specification, in dependency order."""

    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.definitions: dict[str, Definition] = {}
        self.helpers: dict[Callable[..., object], Definition] = {}
        self.active: set[Callable[..., object]] = set()
        self.loops = 0

    def add(self, definition: Definition) -> Definition:
        self.definitions[definition.name] = definition
        return definition

    def helper(
        self, fn: Callable[..., object], kinds: tuple[Kind, ...], caller: _Translator, node: ast.AST
    ) -> Definition:
        if fn in self.helpers:
            definition = self.helpers[fn]
            if tuple(kind for _, kind in definition.parameters) != kinds:
                caller.fail(node, f"helper {fn.__name__!r} is called with different argument types")
            return definition
        if fn in self.active:
            caller.fail(node, f"helper {fn.__name__!r} is recursive, and recursion is not supported")
        name = _lean_name(fn.__name__)
        if name is None or name in self.definitions:
            caller.fail(node, f"rename helper {fn.__name__!r}; the generated Lean reserves that name")
        self.active.add(fn)
        try:
            definition = _Translator(fn, self.function_name, self).helper(name, kinds)
        finally:
            self.active.discard(fn)
        self.helpers[fn] = definition
        return self.add(definition)


class _Translator:
    def __init__(self, fn: Callable[..., object], function_name: str, registry: _Registry) -> None:
        self.fn = fn
        self.function_name = function_name
        self.registry = registry
        self.count = 0
        self.quantifier_count = 0
        self.name = getattr(fn, "__name__", type(fn).__name__)
        if not inspect.isfunction(fn) or inspect.iscoroutinefunction(fn):
            raise ContractError(
                f"Contract {self.name!r} must be a synchronous Python function with available source.",
                function_name=function_name,
            )
        try:
            source, self.line = inspect.getsourcelines(fn)
            tree = ast.parse(textwrap.dedent("".join(source)))
        except (OSError, TypeError, SyntaxError) as exc:
            raise ContractError(
                f"Source is unavailable for contract {self.name!r}; define it in a Python source file.",
                function_name=function_name,
            ) from exc
        self.path = inspect.getsourcefile(fn) or "<unknown>"
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == self.name]
        if len(nodes) != 1 or sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
            self.fail(tree, "contract must be one small, named function")
        self.node = nodes[0]
        if self.node.decorator_list or hasattr(fn, "__wrapped__"):
            self.fail(self.node, "decorated validators are not supported; use a plain Python function")
        closure = inspect.getclosurevars(fn)
        # Generator bodies live in nested code objects, so getclosurevars alone
        # misses builtins referenced only inside a nested all()/any(). Resolve
        # globals without executing them, while retaining Python's shadowing.
        self.constants = {**vars(builtins), **fn.__globals__, **closure.nonlocals}
        # Python already identifies locals, including cells captured by generators.
        # Only the branch environment may supply their values, never globals.
        local_names = set(fn.__code__.co_varnames) | set(fn.__code__.co_cellvars)
        self.constants = {name: value for name, value in self.constants.items() if name not in local_names}

    def fail(self, node: ast.AST, message: str) -> typing.NoReturn:
        line = self.line + getattr(node, "lineno", 1) - 1
        raise ContractError(
            f"Unsupported contract {self.name!r} at {self.path}:{line}: {message}.",
            function_name=self.function_name,
        )

    def tick(self, node: ast.AST) -> None:
        self.count += 1
        if self.count > _MAX_NODES:
            self.fail(node, "contract is too complex for the supported subset")

    def expr(self, node: ast.expr, env: dict[str, Expr]) -> Expr:
        self.tick(node)
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool, float):
            return Expr("literal", type(node.value), value=node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            value = self.constants.get(node.id)
            if type(value) in (int, bool, float):
                return Expr("literal", type(value), value=value)
            self.fail(node, f"{node.id!r} is not a parameter, local value, or immutable numeric constant")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in env and self.constants.get(node.value.id) is math and node.attr in ("inf", "nan"):
                return Expr("literal", float, value=getattr(math, node.attr))
        if isinstance(node, ast.List):
            items = tuple(self.expr(item, env) for item in node.elts)
            if any(item.kind is not int for item in items):
                self.fail(node, "list literals must contain only integers")
            return Expr("list", list, items)
        if isinstance(node, ast.ListComp):
            return self.comprehension(node, env)
        if isinstance(node, ast.GeneratorExp):
            self.fail(
                node,
                "generator expressions are supported only inside the builtin "
                "all(), any(), sum(), min(), max(), sorted(), and list()",
            )
        if isinstance(node, ast.Subscript):
            sequence = self.expr(node.value, env)
            if sequence.kind is not list:
                self.fail(node, "only list[int] values support indexing and slicing")
            if not isinstance(node.slice, ast.Slice):
                index = self.expr(node.slice, env)
                if index.kind is not int:
                    self.fail(node, "list indices must be integers")
                return Expr("index", int, (sequence, index))
            if node.slice.step is not None:
                self.fail(node, "only list slices without a step are supported")
            parts = [sequence]
            present = []
            for label, bound in (("start", node.slice.lower), ("stop", node.slice.upper)):
                value = self.expr(bound, env) if bound is not None else Expr("literal", int, value=0)
                if value.kind is not int:
                    self.fail(node, "slice bounds must be integers")
                parts.append(value)
                if bound is not None:
                    present.append(label)
            return Expr("slice", list, tuple(parts), value=" ".join(present))
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return Expr("not", bool, (_truth(value),))
            if isinstance(node.op, (ast.USub, ast.UAdd)) and value.kind in (int, float):
                return Expr("neg", value.kind, (value,)) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            return self.binary(node, env)
        if isinstance(node, ast.BoolOp):
            values = [self.expr(v, env) for v in node.values]
            if any(v.kind is not bool for v in values):
                self.fail(node, "and/or operands must be Boolean expressions")
            result = values[0]
            for value in values[1:]:
                result = Expr("and" if isinstance(node.op, ast.And) else "or", bool, (result, value))
            return result
        if isinstance(node, ast.Compare):
            ops = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
            left = self.expr(node.left, env)
            result = TRUE
            for op, item in zip(node.ops, node.comparators, strict=True):
                right = self.expr(item, env)
                if isinstance(op, (ast.In, ast.NotIn)):
                    if left.kind is not int or right.kind is not list:
                        self.fail(node, "membership requires an int and a list[int]")
                    comparison = Expr("contains", bool, (left, right))
                    if isinstance(op, ast.NotIn):
                        comparison = Expr("not", bool, (comparison,))
                    result = _and(result, comparison)
                    left = right
                    continue
                if type(op) not in ops or left.kind is not right.kind:
                    self.fail(node, "comparisons require matching types; use float literals such as 0.0 with floats")
                if left.kind in (bool, list) and not isinstance(op, (ast.Eq, ast.NotEq)):
                    self.fail(node, "ordering comparisons require int or float operands")
                comparison = Expr(ops[type(op)], bool, (left, right))
                # For integer lists, equality with sorted(self) is exactly the
                # ordering property. Keep this semantic property visible to
                # the prover instead of requiring it to reason about sorting.
                if left.kind is list:
                    if right.op == "sorted" and _bare(right.args[0]) == _bare(left):
                        comparison = Expr("ordered", bool, (left,))
                    elif left.op == "sorted" and _bare(left.args[0]) == _bare(right):
                        comparison = Expr("ordered", bool, (right,))
                    if isinstance(op, ast.NotEq) and comparison.op == "ordered":
                        comparison = Expr("not", bool, (comparison,))
                result = _and(result, comparison)
                left = right
            return result
        if isinstance(node, ast.IfExp):
            cond, yes, no = _truth(self.expr(node.test, env)), self.expr(node.body, env), self.expr(node.orelse, env)
            if yes.kind is not no.kind:
                self.fail(node, "conditional branches must have the same type")
            return Expr("if", yes.kind, (cond, yes, no))
        self.fail(node, f"{type(node).__name__} is not supported in contracts")

    def binary(self, node: ast.BinOp, env: dict[str, Expr]) -> Expr:
        left, right = self.expr(node.left, env), self.expr(node.right, env)
        op = type(node.op)
        if op is ast.Add and left.kind is list and right.kind is list:
            return Expr("add", list, (left, right))
        if op is ast.Pow:
            exponent = _literal_value(right)
            if left.kind is not int or right.kind is not int or right.op != "literal" or typing.cast(int, exponent) < 0:
                # Python's 2 ** -1 is a float, so only a known non-negative exponent has an int result.
                self.fail(node, "** requires an int base and a non-negative int literal exponent")
            return Expr("pow", int, (left,), value=exponent)
        if op not in (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Div):
            self.fail(node, f"{type(node.op).__name__} is not supported in contracts")
        if left.kind not in (int, float) or left.kind is not right.kind:
            self.fail(node, "arithmetic requires matching int or float operands; use float literals such as 0.0")
        if op in (ast.FloorDiv, ast.Mod):
            if left.kind is not int:
                self.fail(node, "// and % require int operands")
            return Expr("fdiv" if op is ast.FloorDiv else "fmod", int, (left, right))
        if op is ast.Div:
            if left.kind is not float:
                self.fail(node, "/ on integers produces a float; use // for integer division")
            return Expr("div", float, (left, right))
        return Expr({ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[op], left.kind, (left, right))

    def builtin(self, node: ast.expr, env: dict[str, Expr]) -> object | None:
        """Return the global a call refers to, or None if it is not a plain global call."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in env:
            return self.constants.get(node.func.id)
        return None

    def call(self, node: ast.Call, env: dict[str, Expr]) -> Expr:
        func = node.func
        if isinstance(func, ast.Attribute):
            if (
                isinstance(func.value, ast.Name)
                and func.value.id not in env
                and self.constants.get(func.value.id) is math
            ):
                return self.math_call(func.attr, node, env)
            return self.method_call(func, node, env)
        if not isinstance(func, ast.Name):
            self.fail(node, "this function call is not supported in contracts")
        if func.id in env:
            self.fail(node, "calls through local values are not supported")
        target = self.constants.get(func.id)
        if target is builtins.zip or target is builtins.enumerate:
            self.fail(node, f"{func.id}() is supported only as what a loop, comprehension, or quantifier iterates over")
        if target is builtins.range or target is builtins.reversed:
            self.fail(
                node,
                f"{func.id}() is supported only as what a loop, comprehension, or quantifier iterates over, "
                "or inside list(), sum(), min(), max(), and sorted()",
            )
        if any(target is function for function in _BUILTINS):
            return self.builtin_call(typing.cast(Callable[..., object], target), node, env)
        if inspect.isfunction(target):
            return self.helper_call(target, node, env)
        self.fail(node, "this function call is not supported in contracts")

    def builtin_call(self, target: Callable[..., object], node: ast.Call, env: dict[str, Expr]) -> Expr:
        name = target.__name__
        if node.keywords:
            self.fail(node, f"{name}() keyword arguments are not supported")
        if target is builtins.all or target is builtins.any:
            generator = node.args[0] if len(node.args) == 1 else None
            if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
                self.fail(node, "all/any require a generator over a supported list")
            domain, variable, _, scope = self.generator(generator, env)
            predicate = _truth(self.expr(generator.elt, scope))
            guard = self.guard(generator.generators[0].ifs, scope)
            if guard != TRUE:
                predicate = Expr("if", bool, (guard, predicate, TRUE if target is builtins.all else FALSE))
            return Expr("all" if target is builtins.all else "any", bool, (domain, predicate), value=variable)
        if (target is builtins.min or target is builtins.max) and len(node.args) >= 2:
            values = [self.expr(argument, env) for argument in node.args]
            kind = values[0].kind
            if kind not in (int, float) or any(value.kind is not kind for value in values):
                self.fail(node, f"{name}() of several values requires int or float arguments of one type")
            result = values[0]
            for value in values[1:]:
                result = Expr(name, kind, (result, value))
            return result
        if len(node.args) != 1:
            self.fail(node, "supported contract builtins take one positional argument")
        argument = node.args[0]
        if target is builtins.len:
            if self.builtin(argument, env) is builtins.range:
                items = self.domain(argument, env)[0]
            else:
                items = self.expr(argument, env)
                if items.kind is not list:
                    self.fail(node, "len() requires a list[int] value or a range()")
            return Expr("len", int, (items,))
        if target is builtins.abs:
            value = self.expr(argument, env)
            if value.kind not in (int, float):
                self.fail(node, "abs() requires an int or float")
            return Expr("abs", value.kind, (value,))
        items = self.iterable(argument, env)
        if target is builtins.list:
            return items
        if target is builtins.sorted:
            return Expr("sorted", list, (items,))
        return Expr({builtins.sum: "sum", builtins.min: "lmin", builtins.max: "lmax"}[target], int, (items,))

    def math_call(self, attribute: str, node: ast.Call, env: dict[str, Expr]) -> Expr:
        if attribute not in ("isfinite", "isnan", "isinf", "sqrt"):
            self.fail(node, "this function call is not supported in contracts")
        if node.keywords or len(node.args) != 1:
            self.fail(node, "supported contract builtins take one positional argument")
        value = self.expr(node.args[0], env)
        if value.kind is not float:
            self.fail(
                node,
                "math.sqrt() requires a float"
                if attribute == "sqrt"
                else "floating-point classification requires a float",
            )
        return Expr(attribute, float if attribute == "sqrt" else bool, (value,))

    def method_call(self, func: ast.Attribute, node: ast.Call, env: dict[str, Expr]) -> Expr:
        if func.attr not in ("count", "index"):
            self.fail(node, "this function call is not supported in contracts")
        receiver = self.expr(func.value, env)
        if receiver.kind is not list or node.keywords or len(node.args) != 1:
            self.fail(node, f"list.{func.attr}() takes one int argument on a list[int]")
        item = self.expr(node.args[0], env)
        if item.kind is not int:
            self.fail(node, f"list.{func.attr}() takes one int argument on a list[int]")
        return Expr("count" if func.attr == "count" else "index_of", int, (receiver, item))

    def helper_call(self, fn: Callable[..., object], node: ast.Call, env: dict[str, Expr]) -> Expr:
        parameters = list(inspect.signature(fn).parameters.values())
        if any(p.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters):
            self.fail(node, f"helper {fn.__name__!r} must take plain named parameters")
        names = [p.name for p in parameters]
        if len(node.args) > len(parameters):
            self.fail(node, f"helper {fn.__name__!r} takes {len(parameters)} arguments")
        bound = {name: self.expr(argument, env) for name, argument in zip(names, node.args, strict=False)}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in names or keyword.arg in bound:
                self.fail(node, f"unexpected keyword argument for helper {fn.__name__!r}")
            bound[keyword.arg] = self.expr(keyword.value, env)
        for p in parameters:
            if p.name not in bound:
                if type(p.default) not in (int, bool, float):
                    self.fail(node, f"helper {fn.__name__!r} is missing argument {p.name!r}")
                bound[p.name] = Expr("literal", type(p.default), value=p.default)
        arguments = tuple(bound[name] for name in names)
        if any(argument.kind not in (int, bool, float, list) for argument in arguments):
            self.fail(node, f"helper {fn.__name__!r} arguments must be int, bool, float, or list[int]")
        definition = self.registry.helper(fn, tuple(a.kind for a in arguments), self, node)
        return Expr("call", definition.kind, arguments, value=definition)

    def iterable(self, node: ast.expr, env: dict[str, Expr]) -> Expr:
        """Translate an iterable of integers: a list, a generator, range(), or reversed()."""
        domain, shape, _ = self.domain(node, env)
        if shape != "int":
            self.fail(node, "zip() and enumerate() produce pairs; iterate over them with a two-name tuple target")
        return domain

    def domain(self, node: ast.expr, env: dict[str, Expr]) -> tuple[Expr, str, Expr | None]:
        """Translate what a loop, comprehension, or quantifier iterates over."""
        target = self.builtin(node, env)
        if isinstance(node, ast.Call) and target is builtins.range:
            if node.keywords or not 1 <= len(node.args) <= 3:
                self.fail(node, "range() takes one to three positional arguments")
            parts = [self.expr(argument, env) for argument in node.args]
            if any(part.kind is not int for part in parts):
                self.fail(node, "range() arguments must be integers")
            if len(parts) == 1:
                start, stop, step = ZERO, parts[0], ONE
            else:
                start, stop, step = parts[0], parts[1], parts[2] if len(parts) == 3 else ONE
            return Expr("range", list, (start, stop, step)), "int", None
        if isinstance(node, ast.Call) and target is builtins.zip:
            if node.keywords or len(node.args) != 2:
                self.fail(node, "zip() takes exactly two iterables")
            first, second = (self.iterable(argument, env) for argument in node.args)
            return Expr("zip", _Pairs, (first, second)), "zip", None
        if isinstance(node, ast.Call) and target is builtins.enumerate:
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if set(keywords) - {"start"} or not node.args or len(node.args) + len(keywords) > 2:
                self.fail(node, "enumerate() takes an iterable and an optional start")
            start = ZERO
            if len(node.args) == 2 or keywords:
                start = self.expr(node.args[1] if len(node.args) == 2 else keywords["start"], env)
            if start.kind is not int:
                self.fail(node, "enumerate() start must be an integer")
            return Expr("enumerate", _Pairs, (self.iterable(node.args[0], env),)), "enumerate", start
        if isinstance(node, ast.Call) and target is builtins.reversed:
            if node.keywords or len(node.args) != 1:
                self.fail(node, "reversed() takes one iterable")
            return Expr("reverse", list, (self.iterable(node.args[0], env),)), "int", None
        if isinstance(node, ast.GeneratorExp):
            return self.comprehension(node, env), "int", None
        items = self.expr(node, env)
        if items.kind is not list:
            self.fail(node, "iteration requires a list[int], a slice, range(), zip(), enumerate(), or reversed()")
        return items, "int", None

    def bind_target(
        self, target: ast.expr, shape: str, start: Expr | None, name: str, env: dict[str, Expr]
    ) -> dict[str, Expr]:
        scope = dict(env)
        if shape == "int":
            if not isinstance(target, ast.Name):
                self.fail(target, "iterating over integers binds a single name")
            scope[target.id] = Expr("variable", int, value=name)
            return scope
        elements = target.elts if isinstance(target, ast.Tuple) else []
        if len(elements) != 2 or not all(isinstance(e, ast.Name) for e in elements):
            self.fail(target, f"iterating over {shape}() requires a two-name tuple target")
        first, second = typing.cast(list[ast.Name], elements)
        if first.id == second.id:
            self.fail(target, "tuple targets must use distinct names")
        pair = Expr("variable", _Tuple, value=name)
        if shape == "zip":
            scope[first.id], scope[second.id] = Expr("fst", int, (pair,)), Expr("snd", int, (pair,))
        else:
            index = Expr("snd_nat", int, (pair,))
            if start is not None and _literal_value(start) != 0:
                index = Expr("add", int, (start, index))
            scope[first.id], scope[second.id] = index, Expr("fst", int, (pair,))
        return scope

    def generator(
        self, node: ast.ListComp | ast.GeneratorExp, env: dict[str, Expr]
    ) -> tuple[Expr, str, str, dict[str, Expr]]:
        if len(node.generators) != 1:
            self.fail(node, "comprehensions take one for clause; nest them instead")
        loop = node.generators[0]
        if loop.is_async:
            self.fail(node, "asynchronous comprehensions are not supported")
        domain, shape, start = self.domain(loop.iter, env)
        name = f"q{self.quantifier_count}"
        self.quantifier_count += 1
        return domain, name, _ELEMENT_TYPES[shape], self.bind_target(loop.target, shape, start, name, env)

    def guard(self, conditions: Sequence[ast.expr], scope: dict[str, Expr]) -> Expr:
        guard = TRUE
        for condition in conditions:
            guard = _and(guard, _truth(self.expr(condition, scope)))
        return guard

    def comprehension(self, node: ast.ListComp | ast.GeneratorExp, env: dict[str, Expr]) -> Expr:
        domain, name, element_type, scope = self.generator(node, env)
        element = self.expr(node.elt, scope)
        if element.kind is not int:
            self.fail(node.elt, "comprehensions must produce integers")
        guard = self.guard(node.generators[0].ifs, scope)
        return Expr("comprehension", list, (domain, element, guard), value=(name, element_type))

    def message(self, node: ast.expr | None, env: dict[str, Expr], *, total: bool = False) -> None:
        if node is None or isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
            return
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    continue
                if (
                    isinstance(part, ast.FormattedValue)
                    and part.format_spec is None
                    and part.conversion in (-1, 115, 114)
                ):
                    expression = self.expr(part.value, env)
                    if total and expression.kind in (int, list):
                        self.fail(
                            part, "use a constant result message; integer formatting can exceed Python's digit limit"
                        )
                    if expression.defined() != TRUE:
                        self.fail(part, "message fields must not contain operations that can raise")
                    continue
                self.fail(part, "messages support strings and simple int/bool f-string fields")
            return
        self.fail(node, "messages must be strings or simple f-strings")

    def returned(self, node: ast.Return, env: dict[str, Expr]) -> Expr:
        if node.value is None or isinstance(node.value, ast.Constant) and node.value.value is None:
            return TRUE
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id not in env and self.constants.get(value.func.id) is PostConditionResult and not value.args:
                kwargs = {k.arg: k.value for k in value.keywords}
                if len(kwargs) != len(value.keywords) or set(kwargs) - {"passed", "message"} or "passed" not in kwargs:
                    self.fail(value, "PostConditionResult requires passed= and optional message=")
                passed = self.expr(kwargs["passed"], env)
                if passed.kind is not bool:
                    self.fail(value, "PostConditionResult.passed must be Boolean")
                self.message(kwargs.get("message"), env, total=True)
                return _and(passed.defined(), passed)
        self.fail(node, "return None or PostConditionResult(passed=..., message=...)")

    def raised(self, node: ast.Raise, env: dict[str, Expr]) -> None:
        if node.cause is not None or not isinstance(node.exc, ast.Call) or not isinstance(node.exc.func, ast.Name):
            self.fail(node, "use raise AssertionError/ValueError/RuntimeError with a string message")
        if self.constants.get(node.exc.func.id) not in (AssertionError, ValueError, RuntimeError, TypeError):
            self.fail(node, "custom exception constructors are not supported")
        if node.exc.keywords or len(node.exc.args) > 1:
            self.fail(node, "exception constructors accept at most one message")
        self.message(node.exc.args[0] if node.exc.args else None, env)

    def assign(self, node: ast.stmt, env: dict[str, Expr]) -> tuple[dict[str, Expr], Expr, Expr]:
        """Translate an assignment into new bindings and their exact and strict definedness."""
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                self.fail(node, "augmented assignment requires a single local name")
            current = ast.copy_location(ast.Name(id=node.target.id, ctx=ast.Load()), node)
            if self.expr(current, env).kind is list:
                self.fail(node, "augmented assignment mutates a list; write xs = xs + [...] instead")
            combined = ast.copy_location(ast.BinOp(left=current, op=node.op, right=node.value), node)
            value = self.expr(combined, env)
            return {node.target.id: _local(value, node.target.id)}, value.defined(), value.defined(strict=True)
        assert isinstance(node, (ast.Assign, ast.AnnAssign))
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or node.value is None:
            self.fail(node, "only assignment to a single local name is supported")
        target = targets[0]
        if isinstance(target, ast.Name):
            value = self.expr(node.value, env)
            return {target.id: _local(value, target.id)}, value.defined(), value.defined(strict=True)
        names = _target_names(target)
        sources = node.value.elts if isinstance(node.value, ast.Tuple) else []
        if (
            not isinstance(target, ast.Tuple)
            or not all(isinstance(element, ast.Name) for element in target.elts)
            or len(sources) != len(names)
            or len(set(names)) != len(names)
        ):
            self.fail(
                node,
                "only assignment to a single local name, or of a tuple of values to a tuple of names, is supported",
            )
        # Python evaluates every right-hand value before binding any name.
        values = [self.expr(source, env) for source in sources]
        defined, strict = TRUE, TRUE
        for value in values:
            defined, strict = _and(defined, value.defined()), _and(strict, value.defined(strict=True))
        return {name: _local(value, name) for name, value in zip(names, values, strict=True)}, defined, strict

    def loop(self, node: ast.For, env: dict[str, Expr]) -> tuple[dict[str, Expr], Expr]:
        """Translate a for loop into a List.foldl definition over the variables it updates."""
        if node.orelse:
            self.fail(node, "for-else is not supported")
        domain, shape, start = self.domain(node.iter, env)
        element = f"q{self.quantifier_count}"
        self.quantifier_count += 1
        targets = _target_names(node.target)
        assigned = _assigned_names(node.body)
        if set(targets) & set(assigned):
            self.fail(node, "a loop may not assign to its own loop variable")
        carried = [name for name in assigned if name in env]
        if not carried:
            self.fail(node, "a loop must update a variable initialized before it")
        kinds = [env[name].kind for name in carried]
        if any(kind not in (int, bool, float, list) for kind in kinds):
            self.fail(node, "loop variables must be int, bool, float, or list[int]")
        index = self.registry.loops
        self.registry.loops += 1
        state_name = f"s{index}"
        if len(carried) == 1:
            state_kind: Kind = kinds[0]
            state = Expr("variable", state_kind, value=state_name)
            reads = {carried[0]: state}
            state_type = lean_type(kinds[0])
        else:
            state_kind = _Tuple
            state = Expr("variable", _Tuple, value=state_name)
            reads = {
                name: Expr("proj", kind, (state,), value=(i, len(carried)))
                for i, (name, kind) in enumerate(zip(carried, kinds, strict=True))
            }
            state_type = " × ".join(lean_type(kind) for kind in kinds)
        scope = self.bind_target(node.target, shape, start, element, {**env, **reads})
        updated, defined, strict = self.loop_body(node.body, scope)
        if state_name in defined.free():
            # The exact condition depends on the loop's own state, for instance through
            # a branch on it. Require every operation to be defined on every iteration
            # instead: stricter, so still sound.
            defined = strict
        if state_name in defined.free():
            self.fail(
                node,
                "an operation that can raise inside a loop depends on a variable the loop updates; "
                "move it out of the loop, or guard it with a condition on the loop element",
            )
        if any(updated[name].kind is not kind for name, kind in zip(carried, kinds, strict=True)):
            self.fail(node, "a loop variable must keep its type")
        if len(carried) == 1:
            init, step = env[carried[0]], updated[carried[0]]
        else:
            init = Expr("tuple", _Tuple, tuple(env[name] for name in carried))
            step = Expr("tuple", _Tuple, tuple(updated[name] for name in carried))
        fold = Expr(
            "fold", state_kind, (init, step, domain), value=(state_name, element, state_type, _ELEMENT_TYPES[shape])
        )
        parameters = _parameters(fold.free())
        definition = self.registry.add(
            Definition(f"loop{index}", parameters, state_kind, state_type, fold, TRUE, proposition=False)
        )
        call = Expr("call", state_kind, tuple(Expr("variable", k, value=n) for n, k in parameters), value=definition)
        # Python leaves loop targets bound to the last element; using them after the
        # loop depends on whether it ran, so they are not visible here.
        after = {name: value for name, value in env.items() if name not in targets}
        if len(carried) == 1:
            after[carried[0]] = call
        else:
            for i, (name, kind) in enumerate(zip(carried, kinds, strict=True)):
                after[name] = Expr("proj", kind, (call,), value=(i, len(carried)))
        return after, _and(domain.defined(), _forall(domain, defined, element))

    def loop_body(self, body: Sequence[ast.stmt], env: dict[str, Expr]) -> tuple[dict[str, Expr], Expr, Expr]:
        """Translate one iteration into the updated variables and its exact and strict definedness."""
        defined, strict = TRUE, TRUE
        for node in body:
            self.tick(node)
            if (
                isinstance(node, ast.Pass)
                or isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                bindings, condition, strict_condition = self.assign(node, env)
                defined, strict = _and(defined, condition), _and(strict, strict_condition)
                env = {**env, **bindings}
                continue
            if isinstance(node, ast.If):
                test = _truth(self.expr(node.test, env))
                yes, yes_defined, yes_strict = self.loop_body(node.body, env)
                no, no_defined, no_strict = self.loop_body(node.orelse, env)
                merged: dict[str, Expr] = {}
                for name in yes.keys() & no.keys():
                    first, second = yes[name], no[name]
                    if first == second:
                        merged[name] = first
                    elif first.kind is not second.kind:
                        self.fail(node, f"{name!r} must have the same type on both branches")
                    else:
                        merged[name] = Expr("if", first.kind, (test, first, second))
                defined = _and(defined, _and(test.defined(), _if(test, yes_defined, no_defined)))
                strict = _and(strict, _and(test.defined(strict=True), _and(yes_strict, no_strict)))
                env = merged
                continue
            if isinstance(node, ast.Assert):
                self.fail(node, "assert inside a loop is not supported; assert all(...) over the loop's domain instead")
            self.fail(node, f"{type(node).__name__} inside a loop is not supported; loops support assignments and if")
        return env, defined, strict

    def block(self, body: list[ast.stmt], env: dict[str, Expr]) -> Expr:
        if not body:
            return TRUE
        node, rest = body[0], body[1:]
        self.tick(node)
        env = dict(env)
        if (
            isinstance(node, ast.Pass)
            or isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return self.block(rest, env)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            bindings, defined, _ = self.assign(node, env)
            return _and(defined, self.block(rest, {**env, **bindings}))
        if isinstance(node, ast.Assert):
            condition = _truth(self.expr(node.test, env))
            self.message(node.msg, env)
            return _and(condition.defined(), _and(condition, self.block(rest, env)))
        if isinstance(node, ast.If):
            test = _truth(self.expr(node.test, env))
            branches = Expr("if", bool, (test, self.block(node.body + rest, env), self.block(node.orelse + rest, env)))
            return _and(test.defined(), branches)
        if isinstance(node, ast.For):
            bindings, defined = self.loop(node, env)
            return _and(defined, self.block(rest, bindings))
        if isinstance(node, ast.Return):
            result = self.returned(node, env)
            self.block(rest, env)  # Validate even unreachable source; never drop unsupported syntax.
            return result
        if isinstance(node, ast.Raise):
            self.raised(node, env)
            self.block(rest, env)
            return FALSE
        self.fail(node, f"{type(node).__name__} is not supported")

    def helper_block(self, body: list[ast.stmt], env: dict[str, Expr]) -> tuple[Expr | None, Expr]:
        """Translate a helper body into its returned value and definedness condition."""
        if not body:
            self.fail(self.node, "a helper must return a value on every path")
        node, rest = body[0], body[1:]
        self.tick(node)
        if (
            isinstance(node, ast.Pass)
            or isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return self.helper_block(rest, env)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            bindings, defined, _ = self.assign(node, env)
            value, rest_defined = self.helper_block(rest, {**env, **bindings})
            return value, _and(defined, rest_defined)
        if isinstance(node, ast.Assert):
            condition = _truth(self.expr(node.test, env))
            self.message(node.msg, env)
            value, rest_defined = self.helper_block(rest, env)
            return value, _and(condition.defined(), _and(condition, rest_defined))
        if isinstance(node, ast.If):
            test = _truth(self.expr(node.test, env))
            yes, yes_defined = self.helper_block(node.body + rest, env)
            no, no_defined = self.helper_block(node.orelse + rest, env)
            if yes is None or no is None:
                value = no if yes is None else yes
            else:
                if yes.kind is not no.kind:
                    self.fail(node, "every return of a helper must have the same type")
                value = yes if yes == no else Expr("if", yes.kind, (test, yes, no))
            branches = _if(test, yes_defined, no_defined)
            return value, _and(test.defined(), branches)
        if isinstance(node, ast.For):
            bindings, defined = self.loop(node, env)
            value, rest_defined = self.helper_block(rest, bindings)
            return value, _and(defined, rest_defined)
        if isinstance(node, ast.Return):
            if node.value is None:
                self.fail(node, "a helper must return a value")
            value = self.expr(node.value, env)
            if rest:
                self.helper_block(rest, env)  # Validate even unreachable source.
            return value, value.defined()
        if isinstance(node, ast.Raise):
            self.raised(node, env)
            return None, FALSE
        self.fail(node, f"{type(node).__name__} is not supported in helpers")

    def helper(self, name: str, kinds: tuple[Kind, ...]) -> Definition:
        parameters = list(inspect.signature(self.fn).parameters.values())
        names: list[str] = []
        env: dict[str, Expr] = {}
        for index, (p, kind) in enumerate(zip(parameters, kinds, strict=True)):
            lean = _lean_name(p.name)
            lean = lean if lean is not None and lean not in names and lean != name else f"a{index}"
            names.append(lean)
            env[p.name] = Expr("variable", kind, value=lean)
        value, defined = self.helper_block(self.node.body, env)
        if value is None:
            self.fail(self.node, "a helper must return a value")
        if value.kind not in (int, bool, float, list):
            self.fail(self.node, "a helper must return an int, bool, float, or list[int]")
        result_type = "Prop" if value.kind is bool else lean_type(value.kind)
        return Definition(
            name, tuple(zip(names, kinds, strict=True)), value.kind, result_type, value, defined, value.kind is bool
        )

    def translate(self, arguments: dict[str, Expr], result_type: Kind | None) -> Contract:
        signature = inspect.signature(self.fn)
        env: dict[str, Expr] = {}
        parameters = list(signature.parameters.values())
        if result_type is not None:
            if not parameters or parameters[0].kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                self.fail(self.node, "a postcondition must accept its result as the first positional argument")
            first = parameters.pop(0)
            if first.name in arguments:
                self.fail(self.node, "the result parameter must not collide with an original input parameter")
            env[first.name] = Expr("variable", result_type, value="r")
        for p in parameters:
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                self.fail(self.node, "contract inputs must be named parameters")
            if p.name in arguments:
                env[p.name] = arguments[p.name]
            elif type(p.default) in (int, bool, float):
                env[p.name] = Expr("literal", type(p.default), value=p.default)
            else:
                self.fail(self.node, f"parameter {p.name!r} has no matching function input or scalar default")
        predicate = self.block(self.node.body, env)
        return Contract(self.name, f"{self.path}:{self.line}", predicate)


def specification(
    fn: Callable[..., object],
    pre_conditions: Sequence[Callable[..., object]],
    post_conditions: Sequence[Callable[..., object]],
    output_type: type | None = None,
) -> Specification:
    """Freeze Python contract syntax into an explicit typed specification."""
    name = getattr(fn, "__name__", "verified_function")
    if not post_conditions:
        raise ContractError("verified_ai_compile requires at least one postcondition.", function_name=name)
    try:
        hints = typing.get_type_hints(fn)
        signature = inspect.signature(fn)
    except Exception as exc:
        raise ContractError(f"Could not resolve the Python signature of {name!r}: {exc}", function_name=name) from exc
    result_type = _kind(output_type or hints.get("return"))
    if result_type is None:
        raise ContractError(f"{name!r} must return int, bool, float, or list[int].", function_name=name)
    parameters: list[tuple[str, Kind]] = []
    for p in signature.parameters.values():
        kind = _kind(hints.get(p.name))
        if kind is None or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ContractError(
                f"{name!r}: parameter {p.name!r} must be int, bool, float, or list[int].", function_name=name
            )
        if p.default is not inspect.Parameter.empty and type(p.default) is not kind:
            raise ContractError(f"{name!r}: default for {p.name!r} must be {kind_name(kind)}.", function_name=name)
        parameters.append((p.name, kind))
    arguments = {n: Expr("variable", t, value=f"v{i}") for i, (n, t) in enumerate(parameters)}
    registry = _Registry(name)
    pre = tuple(_Translator(c, name, registry).translate(arguments, None) for c in pre_conditions)
    post = tuple(_Translator(c, name, registry).translate(arguments, result_type) for c in post_conditions)
    return Specification(
        name,
        signature,
        tuple(parameters),
        result_type,
        pre,
        post,
        inspect.getdoc(fn) or "",
        tuple(registry.definitions.values()),
    )
