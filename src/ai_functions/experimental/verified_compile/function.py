"""One Python decorator for synthesis, proof checking, caching, and native calls."""

from __future__ import annotations

import asyncio
import functools
import logging
import shutil
import tempfile
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import platformdirs
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.exceptions import ReadTimeoutError as BotoReadTimeoutError
from strands import tool
from strands.models import BedrockModel
from strands.types.exceptions import MaxTokensReachedException
from urllib3.exceptions import ReadTimeoutError as HTTPReadTimeoutError

from ...ai_thread.ai_function import ai_function
from ...ai_thread.errors import AIFunctionError
from ...utils import run_blocking
from ..lean import LeanConfig
from ..lean.execution import run_in_thread
from ..lean.locking import async_exclusive_file_lock
from ..lean.toolchain import DEFAULT_LEAN_TOOLCHAIN
from ._runtime import Runtime, require_supported_python, resolve_runtime
from .compiler import (
    Artifact,
    Candidate,
    build_candidate,
    cache_key,
    read_artifact,
    run_sampled_test,
    sample_inputs,
    write_manifest,
)
from .contracts import Scalar, Specification, kind_name, specification
from .errors import CandidateError, CompilerError, ModelSetupError, SynthesisError

if TYPE_CHECKING:
    from strands.models import Model

_DEFAULT_MODEL_ID = "global.anthropic.claude-opus-5-5"
_DEFAULT_MAX_TOKENS = 32768
_DEFAULT_READ_TIMEOUT = 900
# Calls per synthesis attempt, shared by test_implementation and check_lean.
_TOOL_CALLS = 24
_logger = logging.getLogger(__name__)

_TOOLS_PROMPT = """
You have two tools, and {calls} calls in total across both.

test_implementation(implementation) runs your implementation against the contracts
on sampled inputs that satisfy the preconditions, and returns a counterexample if it
is wrong. It needs no proof and takes a few seconds.
check_lean(implementation, proof) runs the real checker and returns VERIFIED or the
exact rejection text.

Work in this order. First call test_implementation with your best guess at the
implementation, even a rough one. If it reports a counterexample, fix the
implementation and call it again. Only once it passes should you write a proof and
call check_lean. After a check_lean rejection, fix only the first reported error and
call again. Do not derive at length before calling a tool, and do not simulate Lean
in your head: a counterexample costs seconds and tells you more than reasoning does.
When check_lean reports VERIFIED, return the Candidate containing exactly the two
strings that verified.
"""


def _prompt(spec: Specification) -> str:
    arguments = ", ".join(f"{name} is v{i}: {kind_name(kind)}" for i, (name, kind) in enumerate(spec.parameters))
    quantifier = f"forall {spec.binders}, " if spec.parameters else ""
    args = spec.arguments
    helpers = spec.helpers()
    notes = []
    if spec.definitions:
        names = ", ".join(d.name for d in spec.definitions)
        notes.append(
            f"The definitions {names} come from Python helpers and loops. Unfold them with\n"
            "`simp only [name]` or `unfold name`; the implementation may call them. A loop is\n"
            "a List.foldl; prove facts about it by induction on the list, generalizing the\n"
            "accumulator. A name ending in _defined states when Python evaluates it without raising.\n"
        )
    if "pythonAt" in helpers:
        notes.append("pythonAt xs i is Python's xs[i]; pythonAt_ofNat rewrites it for an in-range natural index.\n")
    return f"""Synthesize a pure, total Lean {spec.result_type} function and its proof.
Return the structured Candidate with implementation and proof fields only.
Inputs: {arguments or "(no arguments)"}.
Author guidance (the formal contracts below are authoritative):
{spec.guidance}

The following declarations are fixed and cannot be changed:
{helpers}
{spec.declarations()}
{"".join(notes)}
Your implementation field is ONLY the expression body of:
def implementation {spec.binders} : {spec.result_type} := ...
Your proof field is ONLY the term beginning with `by` proving:
{quantifier}pre {args} -> post (implementation {args}) {args}

Implementation vocabulary: inputs v0, v1, ...; locals t0, t1, ...; decimal or
hexadecimal integer literals; true/false; if/then/else; let; +, -, *, /, %, ^; comparisons and
Boolean operators; min, max, Int.natAbs, Int.ofNat, Int.ediv, Int.fdiv, Int.fmod, Nat.sqrt,
Option.getD; pure List/Array functions, and Float arithmetic/classification including Float.sqrt.
Local recursion uses `let rec go ... termination_by ...` inside the implementation; proofs
refer to it as implementation.go, with implementation.go.eq_1 and implementation.go.induct.
Python's // and % are Int.fdiv and Int.fmod; Lean's / on Int is Euclidean and differs for
negative divisors.
List inputs/outputs are List Int. Float inputs/outputs use binary64, not real arithmetic.
Use Float.beq for floating-point equality; NaN is unequal to itself, while signed zeroes compare equal.
Do not use floating-point bit inspection, IO, partial definitions, or panicking operations.
Proof vocabulary: intro, exact, apply, refine, have, show, cases, constructor,
split, simp, simp_all, only, at, all_goals, first, try, repeat, omega, grind,
decide, rfl, assumption, contradiction, trivial, done, by_cases, subst, rw, simpa,
unfold, dsimp, change, revert, rcases, obtain, induction, calc.
Destructuring binders such as `obtain ⟨n, hn⟩ := h` are supported.
Core Int/Nat/Bool/List/Array/Float lemmas are allowed.
The specification is stated in Prop, built from decidable atoms with ∧, ∨, ¬, ↔,
`if c then P else Q`, and bounded `∀ x ∈ xs,` / `∃ x ∈ xs,`. Bool values appear
only as `b = true`, including Lean's Bool-valued Float comparisons and classifiers,
and a Bool result compared with a condition appears as `r = true ↔ P`.
Your implementation is executable, so it computes with Bool: List.all, List.any and
List.findIdx take Bool predicates. Relate it to the specification with
List.all_eq_true, List.any_eq_true, Bool.and_eq_true, and decide_eq_true_eq, which
plain simp also applies. `pre` and `post` are abbreviations of the indexed contracts,
so `decide` and instance search see through them. Unfold with `simp only [pre, post]`,
adding the indexed names such as `pre0` and `post0` that appear in the declarations
above. Take a hypothesis apart
with `obtain ⟨h0, h1⟩ := h` or `h.left` and `h.right`; build a conjunction goal with
`refine ⟨?_, ?_⟩` or `constructor`; use `split` for an `if` in the goal, not for a
conjunction.
The trusted environment is {DEFAULT_LEAN_TOOLCHAIN}, with `public import Init`
and `meta import all Lean`, including omega and grind, but no Mathlib.
Useful list lemmas include List.pairwise_cons,
List.findIdx_nil, List.findIdx_cons, List.findIdx_le_length, List.not_of_lt_findIdx,
List.Pairwise.rel_of_mem_take_of_mem_drop, List.take_succ_cons, List.drop_succ_cons,
List.length_take, List.length_drop, and List.length_cons. Sortedness is List.Pairwise.
For min/max arithmetic, unfold Int.min_def and Int.max_def before using omega.
For integer division, the exact core lemma signatures are:
  Int.mul_ediv_self_le {{x k : Int}} (h : k ≠ 0) : k * (x / k) <= x
  Int.lt_mul_ediv_self_add {{x k : Int}} (h : 0 < k) : x < k * (x / k) + k
  Int.mul_nonneg {{a b : Int}} (ha : 0 <= a) (hb : 0 <= b) : 0 <= a * b
  Int.mul_le_mul_of_nonneg_left {{a b c : Int}} : a <= b -> 0 <= c -> c * a <= c * b
Implicit arguments can be supplied by name, e.g. (x := t0) (k := t1).
Normalize distributive products with Int.add_mul, Int.mul_add, and Int.sub_mul.
Use explicit product sign or monotonicity facts to reduce the remaining
obligations to linear arithmetic for omega.
Before omega on Int.ofNat expressions, normalize casts with
`simp only [Int.ofNat_eq_natCast] at *`. For nonnegative, in-range slice indices,
pythonIndex_ofNat, pythonSlice_prefix, and pythonSlice_suffix are available.
An often useful proof is:
  by intro v0 v1 v2 h; simp only [pre, post, implementation] at *; refine ⟨?_, ?_⟩ <;> split <;> omega
For nested conditionals, split all remaining branches, not just the outermost one.
Where omega stalls, use `first | omega | grind` to finish that branch instead of
treating the omega failure as a counterexample.
The `first` tactic accepts the first alternative that does not fail, even if goals
remain. Do not put bare simp or simp_all among its closing alternatives; follow
simplification with a tactic that closes every remaining goal.
Use the appropriate number of inputs. No comments, strings, imports, commands,
custom attributes, sorry/admit, unsafe code, native_decide, or run_tac.
{_TOOLS_PROMPT.format(calls=_TOOL_CALLS)}"""


def _synthesis_tools(
    runtime: Runtime,
    spec: Specification,
    inputs: list[dict[str, Scalar]],
    cache: Path,
    timeout: float,
    state: dict[str, int],
) -> list[Any]:
    """Build the tools for one compilation; they share a call budget per attempt."""

    def exhausted() -> bool:
        state["calls"] += 1
        return state["calls"] > _TOOL_CALLS

    @tool(name="test_implementation")
    async def test_implementation(implementation: str) -> str:
        """Run an implementation against the contracts on sampled inputs, without a proof.

        Args:
            implementation: The expression body of `def implementation ... := ...`.

        Returns:
            PASSED, a counterexample, or the reason nothing ran.
        """
        if exhausted():
            return "Call budget exhausted. Return your best Candidate now."
        with tempfile.TemporaryDirectory(prefix="test-", dir=cache) as temporary:
            return await run_sampled_test(runtime, spec, implementation, inputs, Path(temporary), timeout)

    @tool(name="check_lean")
    async def check_lean(implementation: str, proof: str) -> str:
        """Check an implementation and proof with the real checker.

        Args:
            implementation: The expression body of `def implementation ... := ...`.
            proof: A term beginning with `by` that proves the fixed theorem.

        Returns:
            VERIFIED, or the exact rejection text.
        """
        if exhausted():
            return "Call budget exhausted. Return your best Candidate now."
        try:
            candidate = Candidate(implementation=implementation, proof=proof)
        except ValueError as error:
            return f"REJECTED.\n{error}"
        with tempfile.TemporaryDirectory(prefix="check-", dir=cache) as temporary:
            try:
                await build_candidate(runtime, spec, candidate, Path(temporary), timeout)
            except (CandidateError, CompilerError) as error:
                return f"REJECTED.\n{error}"
        return "VERIFIED. Return exactly these two strings as your Candidate now."

    return [test_implementation, check_lean]


class _VerifiedFunction[**P, T]:
    """A Python callable whose cached implementation has passed formal verification."""

    def __init__(
        self,
        fn: Callable[P, T],
        *,
        pre_conditions: Sequence[Callable[..., object]] = (),
        post_conditions: Sequence[Callable[..., object]] = (),
        model: Model | str | None = None,
        max_attempts: int = 10,
        compile_timeout: float = 120,
        cache_dir: str | Path | None = None,
        lean_config: LeanConfig | None = None,
        offline: bool = False,
        check_pre_conditions: bool = False,
        check_post_conditions: bool = False,
        output_type: type[T] | None = None,
    ) -> None:
        require_supported_python()
        if type(max_attempts) is not int or max_attempts < 0:
            raise ValueError("max_attempts must be a nonnegative integer (the number of retries).")
        if type(compile_timeout) not in (int, float) or not 0 < compile_timeout < float("inf"):
            raise ValueError("compile_timeout must be a positive, finite number of seconds.")
        self._spec = specification(fn, tuple(pre_conditions), tuple(post_conditions), output_type)
        self._model = _DEFAULT_MODEL_ID if model is None else model
        self._max_attempts = max_attempts
        self._timeout = compile_timeout
        self._cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(platformdirs.user_cache_dir("ai_functions")) / "verified_compile"
        )
        self._lean_config = lean_config or LeanConfig()
        self._offline = offline
        self._check_pre = check_pre_conditions
        self._check_post = check_post_conditions
        self._artifact: Artifact | None = None
        functools.update_wrapper(self, fn, updated=())

    @property
    def name(self) -> str:
        """Return the original Python function's name."""
        return self._spec.name

    @property
    def is_compiled(self) -> bool:
        """Whether this object has resolved a verified native artifact."""
        return self._artifact is not None

    @property
    def artifact_dir(self) -> Path | None:
        """Directory of the verified sources and binary, or None before compilation.

        This is optional diagnostic access. Reading the property never starts
        synthesis or compilation. Treat the cached files as read-only.
        """
        return self._artifact.directory if self._artifact is not None else None

    async def compile(self) -> _VerifiedFunction[P, T]:
        """Prepare one reusable implementation, or reuse a verified cached artifact.

        Synthesis runs once for all inputs satisfying the Python preconditions.
        Resolve the pinned Lean toolchain and build the Python bridge locally.
        Setup reuses installed/cached tools and provisions missing tools unless
        offline=True or the toolchain uses system mode.
        """
        if self._artifact is not None:
            return self
        runtime = await run_in_thread(resolve_runtime, self._lean_config, offline=self._offline)
        key = cache_key(self._spec, runtime)
        self._cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = self._cache / key
        async with async_exclusive_file_lock(self._cache / (key + ".lock")):
            artifact = await asyncio.to_thread(read_artifact, target, runtime, key)
            await run_in_thread(runtime.preflight, self._timeout)
            if artifact is not None:
                self._artifact = artifact
                _logger.info("Reusing verified %s from %s", self.name, target)
                return self

            synthesis_model = self._model
            if isinstance(synthesis_model, str) and synthesis_model == _DEFAULT_MODEL_ID:
                synthesis_model = BedrockModel(
                    model_id=_DEFAULT_MODEL_ID,
                    max_tokens=_DEFAULT_MAX_TOKENS,
                    boto_client_config=BotocoreConfig(read_timeout=_DEFAULT_READ_TIMEOUT, connect_timeout=30),
                )

            # Test-then-prove: the model can run an implementation on sampled inputs and
            # check a proof before it submits. The submitted candidate is checked again below.
            inputs = await asyncio.to_thread(sample_inputs, self._spec)
            budget = {"calls": 0}
            tools = _synthesis_tools(runtime, self._spec, inputs, self._cache, self._timeout, budget)

            @ai_function[Candidate](
                model=synthesis_model,
                max_attempts=0,
                coordinator_tools_enabled=False,
                callback_handler=None,
                system_prompt=(
                    "Produce only the requested implementation and a complete proof of the fixed specification."
                ),
                tools=tools,
            )
            def synthesize(prompt: str) -> str:
                return prompt

            # Reuse the caller's scope so explicit event subscribers can inspect
            # synthesis and its verification retries.
            handle = await synthesize._spawn_in_context()
            diagnostics: list[str] = []
            prompt = _prompt(self._spec)
            try:
                for _attempt in range(self._max_attempts + 1):
                    _logger.info("Synthesizing %s: attempt %d/%d", self.name, _attempt + 1, self._max_attempts + 1)
                    budget["calls"] = 0
                    try:
                        candidate = await handle.run(prompt)
                    except MaxTokensReachedException:
                        diagnostics.append(
                            "The model exhausted its output-token limit before returning a complete candidate."
                        )
                        _logger.warning("%s: %s", self.name, diagnostics[-1])
                        prompt = (
                            "Your previous output reached the token limit. Return a concise, complete Candidate "
                            "containing implementation and proof. Reuse standard-library lemmas where possible. "
                            "Do not repeat the specification or explain your approach."
                        )
                        continue
                    except (NoCredentialsError, PartialCredentialsError):
                        raise ModelSetupError(
                            f"Cannot synthesize {self.name!r}: Amazon Bedrock credentials are missing or incomplete. "
                            "Set AWS_PROFILE to an authenticated AWS profile, or pass a configured model=.",
                            function_name=self.name,
                        ) from None
                    except (BotoReadTimeoutError, HTTPReadTimeoutError):
                        raise ModelSetupError(
                            f"The model request for {self.name!r} timed out before returning a complete candidate. "
                            "Retry the call, or pass a model configured with a longer read timeout.",
                            function_name=self.name,
                        ) from None
                    with tempfile.TemporaryDirectory(prefix="candidate-", dir=self._cache) as temporary:
                        _logger.debug("Candidate for %s:\n%s", self.name, candidate.model_dump_json(indent=2))
                        directory = Path(temporary)
                        try:
                            module = await build_candidate(runtime, self._spec, candidate, directory, self._timeout)
                        except CandidateError as exc:
                            diagnostics.append(str(exc))
                            _logger.warning("Verification failed for %s:\n%s", self.name, exc)
                            prompt = (
                                "Your candidate failed verification. Keep the specification unchanged and return "
                                "a revised "
                                "implementation and proof. Fix the first proof or elaboration errors; "
                                "a later sorryAx audit error can be caused by Lean's recovery from those errors. "
                                "Check the revision with check_lean before you return it. "
                                f"Diagnostics:\n{exc}"
                            )
                            continue
                        write_manifest(directory, module, key)
                        if target.exists():
                            # This is an invalid entry in our own generated cache;
                            # it must never mask a newly checked artifact.
                            shutil.rmtree(target)
                        directory.rename(target)
                        self._artifact = Artifact(target, module, runtime)
                        _logger.info("Verified and compiled %s in %s", self.name, target)
                        return self
            finally:
                await handle.terminate_now()
            raise SynthesisError(self.name, diagnostics)

    def compile_sync(self) -> _VerifiedFunction[P, T]:
        """Prepare the verified implementation from synchronous Python code."""
        return run_blocking(self.compile)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Check argument types, compile if needed, and call the verified native function."""
        values = self._values(*args, **kwargs)
        await self.compile()
        return await asyncio.to_thread(self._invoke, values)

    def _values(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, Scalar]:
        # Types are always checked: the native bridge converts only the proved types.
        # The proof covers every input satisfying the preconditions, so checking them
        # at runtime only matters for callers that might pass inputs outside them.
        values = self._spec.bind_types(*args, **kwargs)
        if self._check_pre:
            self._spec.check_pre_conditions(values)
        return values

    def _invoke(self, values: dict[str, Scalar]) -> T:
        artifact = self._artifact
        if artifact is None:
            raise CompilerError("The verified implementation is not ready.", function_name=self.name)
        try:
            result = artifact.invoke(self._spec, values)
        except AIFunctionError:
            raise
        except Exception as exc:
            error = CompilerError(f"Native execution failed for {self.name!r}.", function_name=self.name)
            error.diagnostics = str(exc)
            raise error from None
        if self._check_post:
            # An additional check of the trusted native compiler, runtime, and
            # conversion boundary, not a substitute for the proof. It evaluates the
            # translated contracts and does not execute Python callbacks.
            for condition in self._spec.post:
                if not condition.predicate.evaluate({**values, "r": result}):
                    raise CompilerError(
                        f"Compiled result failed contract {condition.name!r} ({condition.location}).",
                        function_name=self.name,
                    )
        return typing.cast(T, result)

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call from synchronous Python, with no model calls after compilation."""
        values = self._values(*args, **kwargs)
        if self._artifact is None:
            self.compile_sync()
        return self._invoke(values)


class _TypedDecorator[T]:
    def __init__(self, output_type: type[T]) -> None:
        self.output_type = output_type

    @overload
    def __call__[**P](self, fn: Callable[P, T], /) -> _VerifiedFunction[P, T]: ...

    @overload
    def __call__(self, **kwargs: Any) -> Callable[[Callable[..., T]], _VerifiedFunction[..., T]]: ...

    def __call__(self, fn: Callable[..., T] | None = None, /, **kwargs: Any) -> Any:
        def decorate(function: Callable[..., T]) -> _VerifiedFunction[..., T]:
            return _VerifiedFunction(function, output_type=self.output_type, **kwargs)

        return decorate(fn) if fn is not None else decorate


class _VerifiedFactory:
    """The single public decorator; implementation and runtime types stay private."""

    def __getitem__[T](self, output_type: type[T]) -> _TypedDecorator[T]:
        return _TypedDecorator(output_type)

    @overload
    def __call__[**P, T](self, fn: Callable[P, T], /) -> _VerifiedFunction[P, T]: ...

    @overload
    def __call__[T](
        self,
        *,
        pre_conditions: Sequence[Callable[..., object]] = (),
        post_conditions: Sequence[Callable[..., object]] = (),
        model: Model | str | None = None,
        max_attempts: int = 10,
        compile_timeout: float = 120,
        cache_dir: str | Path | None = None,
        lean_config: LeanConfig | None = None,
        offline: bool = False,
        check_pre_conditions: bool = False,
        check_post_conditions: bool = False,
    ) -> Callable[[Callable[..., T]], _VerifiedFunction[..., T]]: ...

    def __call__(self, fn: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
        def decorate(function: Callable[..., Any]) -> _VerifiedFunction[..., Any]:
            return _VerifiedFunction(function, **kwargs)

        return decorate(fn) if fn is not None else decorate


verified_ai_compile = _VerifiedFactory()
"""Generate and cache a verified native implementation from Python contracts.

Experimental. Requires standard CPython 3.12+; Lean and the native bridge are
prepared on explicit or first-use compilation. Preconditions and
postconditions use ordinary synchronous Python validator functions. The initial
supported domain is pure ``int``/``bool``/``float``/``list[int]`` functions with explicit type hints.
``max_attempts`` is the number of retries after the initial synthesis attempt.
Calls check argument types but, by default, do not evaluate the contracts.
``check_pre_conditions=True`` rejects inputs outside the preconditions, which the
proof does not cover. ``check_post_conditions=True`` re-checks each native result
against the postconditions, guarding the trusted compiler, runtime, and conversion.
"""
