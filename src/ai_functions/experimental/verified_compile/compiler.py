"""Private proof checking, native compilation, and typed native calls.

Candidate text is filtered to a small lexical vocabulary and placed in expression
and proof positions in a trusted template. The filter rejects known unsupported
constructs; it is not a Lean parser or a process sandbox.
The saved declarations are replayed by the kernel in a separate process before
native compilation or loading. Shared Lean tooling provisions the pinned compiler
and builds the direct Python/Lean bridge locally when needed.
During synthesis, a filtered implementation can also run without a proof in a
separate Lean process on sampled inputs; that test only guides the model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ..lean.errors import LeanError, LeanTimeoutError
from ..lean.execution import run_command_async
from ._runtime import _NATIVE_LOCK, Runtime
from .contracts import Scalar, Specification, lean_type
from .errors import CandidateError, CompilerError, ContractError

# Part of every cache key. Bump it whenever the emitted Lean for a contract changes.
TRANSLATOR_VERSION = 4
_NATIVE_KINDS = {int: 1, bool: 2, float: 3, list: 4}
_MAX_SOURCE = 32_768


class Candidate(BaseModel):
    """The model provides two terms, never module configuration or declarations."""

    implementation: str = Field(max_length=_MAX_SOURCE, description="An executable Lean expression using v0, v1, ...")
    proof: str = Field(max_length=_MAX_SOURCE, description="A Lean proof beginning with by, for the supplied theorem")


_IMPL_WORDS = {
    "if",
    "then",
    "else",
    "let",
    "in",
    "true",
    "false",
    "Int",
    "Nat",
    "Bool",
    "min",
    "max",
    "abs",
    "Int.natAbs",
    "Int.ofNat",
    "Int.negOfNat",
    "Int.negSucc",
    "Int.toNat",
    # `(hi - lo).toNat`, the usual termination measure for a search over an Int range.
    "toNat",
    "Int.ediv",
    "Int.fdiv",
    "Int.fmod",
    "Option.getD",
    "Float.sqrt",
    "pythonAt",
    "pythonRange",
    "List",
    "Array",
    "Option",
    "Float",
    "UInt64",
    "some",
    "none",
    "fun",
    "match",
    "with",
    "rec",
    "termination_by",
    "decreasing_by",
    "by",
    "decide",
    "Nat.sqrt",
    "Float.ofBits",
    "Float.add",
    "Float.sub",
    "Float.mul",
    "Float.div",
    "Float.neg",
    "Float.abs",
    "Float.beq",
    "Float.le",
    "Float.lt",
    "Float.isFinite",
    "Float.isInf",
    "Float.isNaN",
    "nil",
    "cons",
    "id",
    "Nat.succ",
    "Nat.pred",
    "Nat.zero",
}
_PROOF_WORDS = _IMPL_WORDS | {
    "by",
    "intro",
    "intros",
    "exact",
    "apply",
    "refine",
    "have",
    "show",
    "from",
    "fun",
    "cases",
    "case",
    "induction",
    "match",
    "with",
    "constructor",
    "left",
    "right",
    "split",
    "simp",
    "simp_all",
    "only",
    "at",
    "all_goals",
    "any_goals",
    "first",
    "next",
    "repeat",
    "try",
    "omega",
    "grind",
    "decide",
    "rfl",
    "assumption",
    "contradiction",
    "trivial",
    "done",
    "generalize",
    "classical",
    "by_cases",
    "by_contra",
    "rename_i",
    "subst",
    "rw",
    "rwa",
    "simpa",
    "using",
    "suffices",
    "pre",
    "post",
    "implementation",
    "isTrue",
    "isFalse",
    "pos",
    "neg",
    "zero",
    "succ",
    "True",
    "False",
    "And",
    "Or",
    "Not",
    "Eq",
    "congrArg",
    "congrFun",
    "of_decide_eq_true",
    "decide_eq_true_eq",
    "and_true",
    "true_and",
    "and_false",
    "false_and",
    "or_true",
    "true_or",
    "or_false",
    "false_or",
    "and_self",
    "or_self",
    "Bool.true_eq_false",
    "True.intro",
    "False.elim",
    "And.intro",
    "And.left",
    "And.right",
    "Or.inl",
    "Or.inr",
    "Or.elim",
    "Eq.refl",
    "Eq.symm",
    "Eq.trans",
    "unfold",
    "dsimp",
    "change",
    "revert",
    "specialize",
    "rcases",
    "rintro",
    "obtain",
    "calc",
    "ext",
    "congr",
    "trans",
    "exact_mod_cast",
    "assumption_mod_cast",
    "min_def",
    "max_def",
    "le_refl",
    "le_trans",
    "lt_of_lt_of_le",
    "lt_of_le_of_lt",
    "le_of_lt",
    "not_lt",
    "not_le",
    "le_total",
    "pythonSlice",
    "pythonIndex",
    "pythonIndex_ofNat",
    "pythonSlice_prefix",
    "pythonSlice_suffix",
    "pythonAt_ofNat",
    "if_neg",
    "if_pos",
    "if_false",
    "if_true",
    "dif_neg",
    "dif_pos",
    "take",
    "drop",
    "length",
    "getD",
}

_FORBIDDEN_WORDS = {
    "unsafe",
    "unsafeCast",
    "sorry",
    "sorryAx",
    "admit",
    "axiom",
    "axioms",
    "partial",
    "partial_fixpoint",
    "initialize",
    "builtin_initialize",
    "set_option",
    "attribute",
    "extern",
    "implemented_by",
    "import",
    "namespace",
    "section",
    "end",
    "open",
    "export",
    "def",
    "theorem",
    "opaque",
    "constant",
    "macro",
    "syntax",
    "elab",
    "run_tac",
    "run_elab",
    "run_cmd",
    "native_decide",
    "native",
    "IO",
    "BaseIO",
    "EIO",
    "Lean",
    "System",
    "eval",
    "eval_expr",
    "panic",
    "unreachable",
}


def _local_names(source: str) -> set[str]:
    """Collect apparent binder names for the lexical vocabulary checks."""
    names: set[str] = set()
    patterns = [
        r"\b(?:intro|intros|rintro|rename_i)\s+([^;\n<|]+)",
        r"\b(?:let|have|by_cases|by_contra|obtain)\s+(?:rec\s+)?([A-Za-z_][A-Za-z0-9_']*)",
        # `obtain ⟨n, hn, hle⟩ := h` binds every name inside the brackets. Without
        # this the tactic is permitted while the names it introduces are rejected.
        # Non-greedy, so a later group on the same line cannot pull in the names of
        # the term being destructured.
        r"\b(?:obtain|rintro|rcases|refine)\s*[⟨<]([^\n]*?)[⟩>]",
        r"\bfun\s+([^=\n]+?)\s*=>",
        r"\(([A-Za-z_][A-Za-z0-9_' ]*)\s*:",
        r"\|\s*([^=>\n]+?)\s*=>",
        r"\bcase\s+([^=>\n]+?)\s*=>",
        r"\brcases[^\n]*?\bwith\s+([^;\n]+)",
    ]
    for pattern in patterns:
        for group in re.findall(pattern, source):
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", group))
    return names - _FORBIDDEN_WORDS


def validate_candidate(candidate: Candidate, arity: int, names: frozenset[str] = frozenset()) -> None:
    """Apply lexical restrictions before Lean parsing and proof checking."""
    for name, source, words in (
        ("implementation", candidate.implementation, _IMPL_WORDS),
        ("proof", candidate.proof, _PROOF_WORDS),
    ):
        if not source.strip() or len(source) > _MAX_SOURCE or any(c in source for c in ("--", "/-", "-/")):
            raise CandidateError(
                f"The {name} must be a nonempty term without comments, at most {_MAX_SOURCE} characters."
            )
        if re.search(r"[^a-zA-Z0-9_\s()\[\]{}:;,=<>+*/%^!&|.?'\-≤≥≠¬∧∨→←↔↦∀∃∈∉⟨⟩↑·⊢×]", source):
            raise CandidateError(f"The {name} contains unsupported syntax. Do not use strings, comments, or commands.")
        if re.search(r"[\]A-Za-z0-9_']!(?!=)", source):
            raise CandidateError("Panicking operations and native proof shortcuts are not permitted.")
        names_source = re.sub(r"\b(?:0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\b", "0", source)
        locals_ = _local_names(source)
        for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_'.]*", names_source):
            if word in _FORBIDDEN_WORDS or word.split(".")[0] in ("Lean", "IO", "System"):
                raise CandidateError(f"The {name} cannot use {word!r}.")
            if word in words or word == "_" or word in names:
                continue
            if word in locals_:
                continue
            if re.fullmatch(r"v\d+", word) and int(word[1:]) < arity:
                continue
            if re.fullmatch(r"t\d+", word):
                continue
            root, _, projection = word.partition(".")
            if name == "proof" and root in locals_ and projection:
                continue
            # A `let rec` inside the implementation is `implementation.<name>`, with generated
            # lemmas such as `implementation.go.eq_1` and `implementation.go.induct`. Without
            # them, local recursion is permitted but cannot be reasoned about.
            if name == "proof" and root == "implementation" and projection:
                continue
            if (root in locals_ or re.fullmatch(r"[vt]\d+", root)) and projection in (
                "length",
                "size",
                "toArray",
                "toList",
                "toNat",
                "natAbs",
                "isEmpty",
                "1",
                "2",
                "mp",
                "mpr",
            ):
                continue
            if re.fullmatch(r"(?:List|Array)\.[a-zA-Z0-9_'.]+", word):
                continue
            if name == "implementation" and word in _PROOF_WORDS and "by" in source:
                continue
            if name == "proof":
                if re.fullmatch(r"[a-z][0-9']*", word) or re.fullmatch(r"(?:pre|post)\d+", word):
                    continue
                if re.fullmatch(r"(?:Int|Nat|Bool|Float|Option|UInt64)\.[a-zA-Z0-9_'.]+", word):
                    continue
            raise CandidateError(
                f"Unsupported {name} identifier {word!r}; use the vocabulary in the synthesis instructions."
            )
    if not candidate.proof.lstrip().startswith("by"):
        raise CandidateError("The proof must begin with 'by'.")


def source(spec: Specification, candidate: Candidate, module: str) -> str:
    """Place model terms in a fixed module with a fixed theorem and FFI entry."""
    validate_candidate(candidate, len(spec.parameters), spec.definition_names)
    quantifier = f"forall {spec.binders}, " if spec.parameters else ""
    args = spec.arguments
    actual = f"(implementation {args})" if args else "implementation"
    theorem = f"{quantifier}{f'pre {args}'.strip()} -> {f'post {actual} {args}'.strip()}"
    declarations = re.sub(r"^(abbrev |def )", r"public \1", spec.declarations(), flags=re.MULTILINE)
    implementation = textwrap.indent(candidate.implementation.strip(), "    ")
    proof = textwrap.indent(candidate.proof.strip(), "  ")
    return f"""module
public import Init
meta import all Lean
set_option maxHeartbeats 400000
set_option maxRecDepth 1024
set_option linter.unusedVariables false
namespace {module}
{spec.helpers()}
{declarations}
public def implementation {spec.binders} : {spec.result_type} :=
  (
{implementation}
  )
public theorem implementation_correct : {theorem} :=
{proof}
open Lean Elab Command in
run_cmd do
  let axioms <- collectAxioms `{module}.implementation_correct
  for ax in axioms do
    unless #[`propext, `Quot.sound, `Classical.choice].contains ax do
      throwError "Unexpected proof assumption: {{ax}}"
@[export ai_verified_entry_{module}]
public def nativeEntry (_token : UInt8) {spec.binders} : {spec.result_type} :=
  {actual}
end {module}
"""


def native_shim(spec: Specification, module: str) -> str:
    """Generate the small, typed C adapter; it never contains model-written C."""
    c_types = {int: "lean_object *", list: "lean_object *", bool: "uint8_t", float: "double"}
    members = {int: "object", list: "object", bool: "boolean", float: "floating"}
    kinds = [kind for _, kind in spec.parameters]
    parameters = ", ".join(["uint8_t", *(c_types[kind] for kind in kinds)])
    arguments = ", ".join(["0", *(f"arguments[{i}].{members[kind]}" for i, kind in enumerate(kinds))])
    signature = ", ".join(str(_NATIVE_KINDS[kind]) for kind in [*kinds, spec.output_type])
    return f"""#include "ffi.h"
extern {c_types[spec.output_type]} ai_verified_entry_{module}({parameters});
extern lean_object *runtime_initialize_{module}(uint8_t);
static void invoke(const av_value *arguments, av_value *result) {{
    (void)arguments;
    result->{members[spec.output_type]} = ai_verified_entry_{module}({arguments});
}}
static const uint8_t signature[] = {{{signature}}};
static const av_descriptor descriptor = {{
    AV_ABI_MAGIC, AV_ABI_VERSION, {len(kinds)}, signature,
    runtime_initialize_{module}, invoke
}};
LEAN_EXPORT const av_descriptor *ai_verified_descriptor_v1(void) {{ return &descriptor; }}
"""


async def build_candidate(
    runtime: Runtime,
    spec: Specification,
    candidate: Candidate,
    directory: Path,
    timeout: float,
) -> str:
    """Check the proof, replay its declarations, then build the exact checked code."""
    identity = f"{TRANSLATOR_VERSION}\n{runtime.identity}\n{spec.identity()}\n{candidate.model_dump_json()}"
    module = "Verified" + hashlib.sha256(identity.encode()).hexdigest()[:40]
    code = source(spec, candidate, module)
    (directory / f"{module}.lean").write_text(code)
    tools = runtime.toolchain
    environment = tools.environment(directory)
    try:
        result = await run_command_async(
            [str(tools.lean), "-o", f"{module}.olean", "-c", f"{module}.c", f"{module}.lean"],
            cwd=directory,
            environment=environment,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise CandidateError(result.stdout + result.stderr or "The candidate did not satisfy the specification.")
        result = await run_command_async(
            [str(tools.leanchecker), module],
            cwd=directory,
            environment=environment,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise CandidateError(result.stdout + result.stderr or "The proof failed independent kernel replay.")
    except LeanTimeoutError as exc:
        raise CandidateError("Proof checking exceeded its time budget; simplify the implementation or proof.") from exc
    except LeanError as exc:
        raise CompilerError(str(exc)) from exc
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    shim = f"{module}.ffi.c"
    (directory / shim).write_text(native_shim(spec, module))
    try:
        result = await run_command_async(
            [
                str(tools.leanc),
                "-shared",
                "-DLEAN_EXPORTING",
                "-O2",
                "-ffp-contract=off",
                "-o",
                module + suffix,
                f"{module}.c",
                shim,
                "-I",
                str(runtime.directory),
                *tools.native_flags,
                *tools.link_args(),
            ],
            cwd=directory,
            environment=environment,
            timeout=timeout,
            check=False,
        )
    except LeanTimeoutError as exc:
        raise CompilerError("Native compilation exceeded its time budget.") from exc
    except LeanError as exc:
        raise CompilerError(str(exc)) from exc
    if result.returncode:
        error = CompilerError("The verified implementation could not be compiled to native code.")
        error.diagnostics = result.stdout + result.stderr
        raise error
    return module


# Values to try per input type when testing a candidate. The contracts add their own
# integer literals and neighbors, and the preconditions decide which draws are kept.
# The largest integer is 2**20, so a loop bounded by an input still finishes quickly.
_TEST_VALUES: dict[type, list[Scalar]] = {
    int: [0, 1, -1, 2, 3, 10, -10, 100, 1000, -1000, 10000, 2**20],
    bool: [True, False],
    float: [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e10, -1e10],
    list: [[], [0], [1], [1, 2, 3], [3, 1, 2], [-3, -1, 0, 2, 5], [5, -2, 7, 0, 7], [2, 2, 2], [-1000, 1000]],
}
_TEST_INPUTS = 100
_TEST_DRAWS = 20_000


def sample_inputs(spec: Specification, count: int = _TEST_INPUTS, draws: int = _TEST_DRAWS) -> list[dict[str, Scalar]]:
    """Draw distinct inputs that satisfy the preconditions, for testing candidates.

    Values come from fixed pools per type, plus the contracts' integer literals and
    their neighbors, which are the usual boundaries. A draw that fails a
    precondition is discarded, so the result is short, or empty, when the pools
    rarely meet the preconditions. The proof, not this sample, is the guarantee.
    """
    expressions = [contract.predicate for contract in (*spec.pre, *spec.post)]
    expressions += [part for definition in spec.definitions for part in (definition.body, definition.defined)]
    literals = {
        node.value + offset
        for expression in expressions
        for node in expression.walk()
        if node.op == "literal" and type(node.value) is int and abs(node.value) <= 2**64
        for offset in (-1, 0, 1)
    }
    pools = []
    for _, kind in spec.parameters:
        pool = list(_TEST_VALUES[kind])
        if kind is int:
            pool += sorted(literals.difference(pool), key=abs)[:24]
        pools.append(pool)
    generator = random.Random(0)
    seen: set[str] = set()
    inputs: list[dict[str, Scalar]] = []
    for _ in range(draws):
        values = {f"v{index}": generator.choice(pool) for index, pool in enumerate(pools)}
        key = repr(values)
        if key in seen:
            continue
        seen.add(key)
        try:
            spec.check_pre_conditions(values)
        except (ContractError, ArithmeticError):
            continue
        inputs.append(values)
        if len(inputs) == count:
            break
    return inputs


def _lean_value(value: Scalar) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is list:
        return "[" + ", ".join(str(item) for item in value) + "]"
    return repr(value)


def sampled_test_source(spec: Specification, implementation: str, inputs: Sequence[Mapping[str, Scalar]]) -> str:
    """A standalone Lean program that runs one implementation on sampled inputs.

    It reuses the translated `pre` and `post` verbatim and checks `pre` again, so an
    input the Python evaluation admits but Lean does not is skipped.
    """
    args = spec.arguments
    types = [lean_type(kind) for _, kind in spec.parameters]
    if not types:
        case_type, binder, literals = "Unit", "_", ["()"] * len(inputs)
    elif len(types) == 1:
        case_type, binder, literals = types[0], "v0", [_lean_value(values["v0"]) for values in inputs]
    else:
        case_type = " × ".join(types)
        binder = "(" + ", ".join(f"v{index}" for index in range(len(types))) + ")"
        literals = [
            "(" + ", ".join(_lean_value(values[f"v{index}"]) for index in range(len(types))) + ")" for values in inputs
        ]
    reported = " ".join(f"v{index}={{v{index}}}" for index in range(len(types)))
    declarations = f"{spec.helpers()}\n{spec.declarations()}".replace("public ", "")
    body = textwrap.indent(implementation.strip(), "    ")
    return f"""set_option maxHeartbeats 1000000
set_option linter.unusedVariables false
{declarations}
def implementation {spec.binders} : {spec.result_type} :=
  (
{body}
  )
def cases : List ({case_type}) := [{", ".join(literals)}]
def main : IO Unit := do
  let mut admitted := 0
  let mut failures := 0
  let mut first := ""
  for {binder} in cases do
    if pre {args} then
      admitted := admitted + 1
      unless post (implementation {args}) {args} do
        failures := failures + 1
        if first.isEmpty then
          first := s!"{reported} result={{implementation {args}}}"
  IO.println s!"admitted={{admitted}} failures={{failures}}"
  unless first.isEmpty do IO.println s!"counterexample {{first}}"
"""


async def run_sampled_test(
    runtime: Runtime,
    spec: Specification,
    implementation: str,
    inputs: Sequence[Mapping[str, Scalar]],
    directory: Path,
    timeout: float,
) -> str:
    """Run a filtered implementation, without a proof, and report the result to the model."""
    try:
        validate_candidate(
            Candidate(implementation=implementation, proof="by trivial"), len(spec.parameters), spec.definition_names
        )
    except (CandidateError, ValueError) as error:
        return f"REJECTED before running: {error}"
    if not inputs:
        return (
            "INCONCLUSIVE: no sampled input satisfied the preconditions, so nothing ran. "
            "Write the proof and call check_lean."
        )
    (directory / "Test.lean").write_text(sampled_test_source(spec, implementation, inputs))
    tools = runtime.toolchain
    try:
        result = await run_command_async(
            [str(tools.lean), "--run", "Test.lean"],
            cwd=directory,
            environment=tools.environment(directory),
            timeout=timeout,
            check=False,
        )
    except LeanTimeoutError:
        return f"The implementation did not finish on the sampled inputs within {timeout:g} seconds."
    except LeanError as exc:
        raise CompilerError(str(exc)) from exc
    if result.returncode:
        return f"The implementation did not compile:\n{(result.stdout + result.stderr)[:1500]}"
    report = re.search(r"admitted=(\d+) failures=(\d+)", result.stdout)
    if report is None:
        return f"The test printed no report:\n{result.stdout[:1500]}"
    admitted, failures = int(report[1]), int(report[2])
    if failures:
        counterexample = re.search(r"^counterexample .*$", result.stdout, re.MULTILINE)
        shown = counterexample[0] if counterexample else ""
        return f"WRONG on {failures} of {admitted} sampled inputs. {shown}\nFix the implementation and test again."
    return f"PASSED on {admitted} sampled inputs that satisfy the preconditions. Now write a proof and call check_lean."


@dataclass(frozen=True)
class Artifact:
    """A published, immutable native artifact and its compilation identity."""

    directory: Path
    module: str
    runtime: Runtime
    _native: Callable[[dict[str, Scalar]], Scalar] | None = field(default=None, compare=False, repr=False)

    def invoke(self, spec: Specification, values: dict[str, Scalar]) -> Scalar:
        """Call a cached typed C entry without serializing arguments or results."""
        native = self._native
        if native is None:
            with _NATIVE_LOCK:
                native = self._native
                if native is None:
                    bridge = self.runtime.bridge()
                    suffix = ".dylib" if sys.platform == "darwin" else ".so"
                    signature = bytes(
                        [_NATIVE_KINDS[kind] for _, kind in spec.parameters] + [_NATIVE_KINDS[spec.output_type]]
                    )
                    native = bridge.load(str(self.directory / (self.module + suffix)), signature)
                    object.__setattr__(self, "_native", native)
        return native(values)


def cache_key(spec: Specification, runtime: Runtime) -> str:
    """Separate artifacts by semantics, compiler version, platform and ABI."""
    value = f"{TRANSLATOR_VERSION}\n{runtime.identity}\n{spec.identity()}"
    return hashlib.sha256(value.encode()).hexdigest()


def write_manifest(directory: Path, module: str, key: str) -> None:
    """Record every artifact byte before atomically publishing the directory."""
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    (directory / "manifest.json").write_text(json.dumps({"key": key, "module": module, "files": files}, sort_keys=True))


def read_artifact(directory: Path, runtime: Runtime, key: str) -> Artifact | None:
    """Reject missing, incomplete or corrupted cache entries."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            return None
        module = manifest["module"]
        if not isinstance(module, str) or manifest["key"] != key or not re.fullmatch(r"Verified[0-9a-f]{40}", module):
            return None
        files = manifest["files"]
        if not isinstance(files, dict):
            return None
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        if not {module + ".lean", module + ".olean", module + suffix}.issubset(files):
            return None
        for name, expected in files.items():
            if not isinstance(name, str) or Path(name).name != name or (directory / name).is_symlink():
                return None
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                return None
        return Artifact(directory, module, runtime)
    except (OSError, ValueError, KeyError, TypeError):
        return None
