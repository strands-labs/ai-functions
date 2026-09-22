"""Deterministic, typed translation of a deliberately small Python contract subset.

No source, prompt builder, validator, or generated Python is executed here. A
validator is translated to its *successful completion* predicate, including
branches, early returns, assertions and explicit failures. Unsupported syntax
is an error, including syntax on an unreachable path.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import math
import struct
import textwrap
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ...ai_thread.postcondition import PostConditionResult
from .errors import ContractError

Scalar = int | bool | float | list[int]
Kind = type[int] | type[bool] | type[float] | type[list]
_MAX_NODES = 4096


def kind_name(kind: Kind) -> str:
    """Describe a supported Python value type."""
    return "list[int]" if kind is list else kind.__name__


def lean_type(kind: Kind) -> str:
    """Map Python value kinds to the corresponding logical and native types."""
    return {int: "Int", bool: "Bool", float: "Float", list: "List Int"}[kind]


def _kind(annotation: object) -> Kind | None:
    if any(annotation is value for value in (int, bool, float)):
        return annotation
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (int,):
        return list
    return None


@dataclass(frozen=True)
class Expr:
    """A typed expression with identical Python evaluation and emitted semantics."""

    op: str
    kind: Kind
    args: tuple[Expr, ...] = ()
    value: int | bool | float | str = False

    def evaluate(self, values: Mapping[str, Scalar]) -> Scalar:
        """Evaluate the supported expression without Python eval or assert."""
        if self.op == "literal":
            return typing.cast(Scalar, self.value)
        if self.op == "variable":
            return values[str(self.value)]
        if self.op == "list":
            return [typing.cast(int, a.evaluate(values)) for a in self.args]
        if self.op in ("all", "any"):
            items = typing.cast(list[int], self.args[0].evaluate(values))
            outcomes = (bool(self.args[1].evaluate({**values, str(self.value): item})) for item in items)
            return all(outcomes) if self.op == "all" else any(outcomes)
        if self.op == "if":
            branch = 1 if self.args[0].evaluate(values) else 2
            return self.args[branch].evaluate(values)
        left = self.args[0].evaluate(values)
        if self.op == "len":
            return len(typing.cast(list[int], left))
        if self.op == "sorted":
            return sorted(typing.cast(list[int], left))
        if self.op == "ordered":
            items = typing.cast(list[int], left)
            return all(a <= b for a, b in zip(items, items[1:], strict=False))
        if self.op == "slice":
            begin = typing.cast(int, self.args[1].evaluate(values)) if "start" in str(self.value) else None
            end = typing.cast(int, self.args[2].evaluate(values)) if "stop" in str(self.value) else None
            return typing.cast(list[int], left)[begin:end]
        if self.op in ("isfinite", "isnan", "isinf"):
            return getattr(math, self.op)(typing.cast(float, left))
        if self.op == "not":
            return not left
        if self.op == "neg":
            return -left
        if self.op == "and":
            return bool(left) and bool(self.args[1].evaluate(values))
        if self.op == "or":
            return bool(left) or bool(self.args[1].evaluate(values))
        right = self.args[1].evaluate(values)
        match self.op:
            case "add":
                return left + right
            case "sub":
                return left - right
            case "mul":
                return left * right
            case "contains":
                return left in right
            case "eq":
                return left == right
            case "ne":
                return left != right
            case "lt":
                return left < right
            case "le":
                return left <= right
            case "gt":
                return left > right
            case "ge":
                return left >= right
        raise AssertionError(f"Unknown expression operation: {self.op}")

    def lean(self) -> str:
        """Render a closed, typed expression in the private compiler language."""
        if self.op == "literal":
            if self.kind is bool:
                return "true" if self.value else "false"
            if self.kind is float:
                bits = struct.unpack(">Q", struct.pack(">d", typing.cast(float, self.value)))[0]
                return f"(Float.ofBits (0x{bits:016x} : UInt64))"
            integer = typing.cast(int, self.value)
            sign = "-" if integer < 0 else ""
            return f"({sign}0x{abs(integer):x} : Int)"
        if self.op == "variable":
            return str(self.value)
        args = [a.lean() for a in self.args]
        if self.op == "list":
            return f"([{', '.join(args)}] : List Int)"
        if self.op in ("all", "any"):
            return f"(List.{self.op} {args[0]} (fun ({self.value} : Int) => {args[1]}))"
        if self.op == "len":
            return f"(Int.ofNat (List.length {args[0]}))"
        if self.op == "sorted":
            return f"(List.mergeSort {args[0]} (fun a b => decide (a <= b)))"
        if self.op == "ordered":
            return f"(decide (List.Pairwise (fun (a b : Int) => a <= b) {args[0]}))"
        if self.op == "slice":
            begin = f"(some {args[1]})" if "start" in str(self.value) else "none"
            end = f"(some {args[2]})" if "stop" in str(self.value) else "none"
            return f"(pythonSlice {args[0]} {begin} {end})"
        if self.op == "contains":
            return f"(List.elem {args[0]} {args[1]})"
        if self.op in ("isfinite", "isnan", "isinf"):
            operation = {"isfinite": "isFinite", "isnan": "isNaN", "isinf": "isInf"}[self.op]
            return f"(Float.{operation} {args[0]})"
        if self.op == "if":
            return f"(if {args[0]} then {args[1]} else {args[2]})"
        if self.op in ("not", "neg"):
            return f"({'!' if self.op == 'not' else '-'}{args[0]})"
        if self.args[0].kind is float and self.op in ("eq", "ne", "lt", "le", "gt", "ge"):
            operations = {"eq": "beq", "ne": "beq", "lt": "lt", "le": "le", "gt": "lt", "ge": "le"}
            operands = list(reversed(args)) if self.op in ("gt", "ge") else args
            comparison = f"(Float.{operations[self.op]} {operands[0]} {operands[1]})"
            return f"(!{comparison})" if self.op == "ne" else comparison
        if self.kind is list and self.op == "add":
            return f"({args[0]} ++ {args[1]})"
        operators = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "and": "&&",
            "or": "||",
            "eq": "=",
            "ne": "≠",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }
        expression = f"({args[0]} {operators[self.op]} {args[1]})"
        return f"(decide {expression})" if self.op in ("eq", "ne", "lt", "le", "gt", "ge") else expression


TRUE = Expr("literal", bool, value=True)
FALSE = Expr("literal", bool, value=False)


def _and(left: Expr, right: Expr) -> Expr:
    if left == TRUE:
        return right
    if right == TRUE:
        return left
    return Expr("and", bool, (left, right))


def _truth(expr: Expr) -> Expr:
    if expr.kind is bool:
        return expr
    if expr.kind is list:
        return Expr("ne", bool, (Expr("len", int, (expr,)), Expr("literal", int, value=0)))
    zero = Expr("literal", float, value=0.0) if expr.kind is float else Expr("literal", int, value=0)
    return Expr("ne", bool, (expr, zero))


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

    def bind(self, *args: object, **kwargs: object) -> dict[str, Scalar]:
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
        for contract in self.pre:
            if not contract.predicate.evaluate(values):
                raise ContractError(
                    f"Precondition {contract.name!r} failed for {self.name!r} ({contract.location}).",
                    function_name=self.name,
                )
        return values

    def identity(self) -> str:
        """Return canonical compilation inputs for a content-addressed cache."""
        return json.dumps(
            {
                "name": self.name,
                "parameters": [(name, kind_name(kind)) for name, kind in self.parameters],
                "output": kind_name(self.output_type),
                "pre": [(c.name, c.predicate.lean()) for c in self.pre],
                "post": [(c.name, c.predicate.lean()) for c in self.post],
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

    def declarations(self) -> str:
        """Emit all trusted specification definitions, including individual contracts."""
        lines: list[str] = []
        for prefix, contracts in (("pre", self.pre), ("post", self.post)):
            result = f"(r : {self.result_type}) " if prefix == "post" else ""
            for i, c in enumerate(contracts):
                lines.append(f"def {prefix}{i} {result}{self.binders} : Bool := {c.predicate.lean()}")
            combined = TRUE
            for c in contracts:
                combined = _and(combined, c.predicate)
            lines.append(f"def {prefix} {result}{self.binders} : Bool := {combined.lean()}")
        return "\n".join(lines)


class _Translator:
    def __init__(self, fn: Callable[..., object], function_name: str) -> None:
        self.fn = fn
        self.function_name = function_name
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
        if isinstance(node, ast.Subscript):
            sequence = self.expr(node.value, env)
            if sequence.kind is not list or not isinstance(node.slice, ast.Slice) or node.slice.step is not None:
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
            if isinstance(node.func, ast.Name) and node.func.id in env:
                self.fail(node, "calls through local values are not supported")
            target = self.constants.get(node.func.id) if isinstance(node.func, ast.Name) else None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if (
                    node.func.value.id not in env
                    and self.constants.get(node.func.value.id) is math
                    and node.func.attr in ("isfinite", "isnan", "isinf")
                ):
                    target = getattr(math, node.func.attr)
            if node.keywords or len(node.args) != 1:
                self.fail(node, "supported contract builtins take one positional argument")
            if target is builtins.all or target is builtins.any:
                generator = node.args[0]
                if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
                    self.fail(node, "all/any require a generator over a supported list")
                loop = generator.generators[0]
                if loop.is_async or not isinstance(loop.target, ast.Name):
                    self.fail(node, "quantifiers require one synchronous element binding")
                sequence = self.expr(loop.iter, env)
                if sequence.kind is not list:
                    self.fail(loop.iter, "quantifier domains must be list[int] values or slices")
                variable = f"q{self.quantifier_count}"
                self.quantifier_count += 1
                scope = {**env, loop.target.id: Expr("variable", int, value=variable)}
                predicate = _truth(self.expr(generator.elt, scope))
                guard = TRUE
                for condition in loop.ifs:
                    guard = _and(guard, _truth(self.expr(condition, scope)))
                if guard != TRUE:
                    predicate = Expr("if", bool, (guard, predicate, TRUE if target is builtins.all else FALSE))
                return Expr("all" if target is builtins.all else "any", bool, (sequence, predicate), value=variable)
            if target is builtins.len or target is builtins.sorted:
                sequence = self.expr(node.args[0], env)
                if sequence.kind is not list:
                    self.fail(node, "len/sorted require list[int] values")
                return Expr(
                    "len" if target is builtins.len else "sorted", int if target is builtins.len else list, (sequence,)
                )
            if any(target is predicate for predicate in (math.isfinite, math.isnan, math.isinf)):
                value = self.expr(node.args[0], env)
                if value.kind is not float:
                    self.fail(node, "floating-point classification requires a float")
                return Expr(target.__name__, bool, (value,))
            self.fail(node, "this function call is not supported in contracts")
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return Expr("not", bool, (_truth(value),))
            if isinstance(node.op, (ast.USub, ast.UAdd)) and value.kind in (int, float):
                return Expr("neg", value.kind, (value,)) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if left.kind is list and right.kind is list and isinstance(node.op, ast.Add):
                return Expr("add", list, (left, right))
            if left.kind not in (int, float) or left.kind is not right.kind:
                self.fail(node, "arithmetic requires matching int or float operands; use float literals such as 0.0")
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[type(node.op)]
            return Expr(op, left.kind, (left, right))
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
                    if right.op == "sorted" and right.args[0] == left:
                        comparison = Expr("ordered", bool, (left,))
                    elif left.op == "sorted" and left.args[0] == right:
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
                return passed
        self.fail(node, "return None or PostConditionResult(passed=..., message=...)")

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
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name) or node.value is None:
                self.fail(node, "only assignment to a single local name is supported")
            env[targets[0].id] = self.expr(node.value, env)
            return self.block(rest, env)
        if isinstance(node, ast.Assert):
            condition = _truth(self.expr(node.test, env))
            self.message(node.msg, env)
            return _and(condition, self.block(rest, env))
        if isinstance(node, ast.If):
            return Expr(
                "if",
                bool,
                (
                    _truth(self.expr(node.test, env)),
                    self.block(node.body + rest, env),
                    self.block(node.orelse + rest, env),
                ),
            )
        if isinstance(node, ast.Return):
            result = self.returned(node, env)
            self.block(rest, env)  # Validate even unreachable source; never drop unsupported syntax.
            return result
        if isinstance(node, ast.Raise):
            if node.cause is not None or not isinstance(node.exc, ast.Call) or not isinstance(node.exc.func, ast.Name):
                self.fail(node, "use raise AssertionError/ValueError/RuntimeError with a string message")
            if self.constants.get(node.exc.func.id) not in (AssertionError, ValueError, RuntimeError, TypeError):
                self.fail(node, "custom exception constructors are not supported")
            if node.exc.keywords or len(node.exc.args) > 1:
                self.fail(node, "exception constructors accept at most one message")
            self.message(node.exc.args[0] if node.exc.args else None, env)
            self.block(rest, env)
            return FALSE
        self.fail(node, f"{type(node).__name__} is not supported")

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
    pre = tuple(_Translator(c, name).translate(arguments, None) for c in pre_conditions)
    post = tuple(_Translator(c, name).translate(arguments, result_type) for c in post_conditions)
    return Specification(name, signature, tuple(parameters), result_type, pre, post, inspect.getdoc(fn) or "")
