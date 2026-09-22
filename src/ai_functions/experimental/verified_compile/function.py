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
from ._runtime import require_supported_python, resolve_runtime
from .compiler import (
    SPEC_HELPERS,
    Artifact,
    Candidate,
    build_candidate,
    cache_key,
    read_artifact,
    write_manifest,
)
from .contracts import Scalar, Specification, kind_name, specification
from .errors import CandidateError, CompilerError, ModelSetupError, SynthesisError

if TYPE_CHECKING:
    from strands.models import Model

_DEFAULT_MODEL_ID = "global.anthropic.claude-opus-5"
_DEFAULT_MAX_TOKENS = 65536
_DEFAULT_READ_TIMEOUT = 900
_logger = logging.getLogger(__name__)


def _prompt(spec: Specification) -> str:
    arguments = ", ".join(f"{name} is v{i}: {kind_name(kind)}" for i, (name, kind) in enumerate(spec.parameters))
    quantifier = f"forall {spec.binders}, " if spec.parameters else ""
    args = spec.arguments
    return f"""Synthesize a pure, total Lean {spec.result_type} function and its proof.
Return the structured Candidate with implementation and proof fields only.
Inputs: {arguments or "(no arguments)"}.
Author guidance (the formal contracts below are authoritative):
{spec.guidance}

The following declarations are fixed and cannot be changed:
{SPEC_HELPERS}
{spec.declarations()}

Your implementation field is ONLY the expression body of:
def implementation {spec.binders} : {spec.result_type} := ...
Your proof field is ONLY the term beginning with `by` proving:
{quantifier}pre {args} = true -> post (implementation {args}) {args} = true

Implementation vocabulary: inputs v0, v1, ...; locals t0, t1, ...; decimal or
hexadecimal integer literals; true/false; if/then/else; let; +, -, *; comparisons and Boolean
operators; min, max, abs, Int.natAbs, Int.ofNat, Int.ediv, Nat.sqrt; pure List/Array functions,
and Float arithmetic/classification. Explicitly terminating local recursion is allowed.
List inputs/outputs are List Int. Float inputs/outputs use binary64, not real arithmetic.
Use Float.beq for floating-point equality; NaN is unequal to itself, while signed zeroes compare equal.
Do not use floating-point bit inspection, IO, partial definitions, or panicking operations.
Proof vocabulary: intro, exact, apply, refine, have, show, cases, constructor,
split, simp, simp_all, only, at, all_goals, first, try, repeat, omega, grind,
decide, rfl, assumption, contradiction, trivial, by_cases, subst, rw, simpa, unfold,
dsimp, change, revert, rcases, induction, calc. Use explicit binders for local names.
Core Int/Nat/Bool/List/Array/Float lemmas are allowed. Propositional simplification
lemmas and_true, true_and, and_false, false_and, or_true, true_or, or_false, false_or,
and_self, and or_self are allowed too; these differ from the Bool-prefixed lemmas.
The trusted environment is {DEFAULT_LEAN_TOOLCHAIN}, with `public import Init`
and `meta import all Lean`, including omega and grind, but no Mathlib.
You have only the Candidate output tool; Lean checking runs after you submit it.
Useful list lemmas include List.all_eq_true, List.any_eq_true, List.pairwise_cons,
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
An often useful proof is: by intro v0 v1 v2 h; simp_all [pre, post, implementation]; split <;> simp_all <;> omega
For nested conditionals, split all remaining branches, not just the outermost one.
After simplifying Boolean contracts and case-splitting comparisons, omega may still
fail on goals containing conjunctions and disjunctions. Use `first | omega | grind`
to finish those branches instead of treating omega failure as a counterexample.
The `first` tactic accepts the first alternative that does not fail, even if goals
remain. Do not put bare simp or simp_all among its closing alternatives; follow
simplification with a tactic that closes every remaining goal.
Use the appropriate number of inputs. No comments, strings, imports, commands,
custom attributes, sorry/admit, unsafe code, native_decide, or run_tac.
"""


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

            @ai_function[Candidate](
                model=synthesis_model,
                max_attempts=0,
                coordinator_tools_enabled=False,
                callback_handler=None,
                system_prompt=(
                    "Produce only the requested implementation and a complete proof of the fixed specification."
                ),
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
        """Validate inputs, compile if needed, and call the verified native function."""
        values = self._spec.bind(*args, **kwargs)
        await self.compile()
        return await asyncio.to_thread(self._invoke, values)

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
        # This is an additional check of the trusted conversion boundary, not
        # a substitute for the proof, and does not execute Python callbacks.
        for condition in self._spec.post:
            if not condition.predicate.evaluate({**values, "r": result}):
                raise CompilerError(
                    f"Compiled result failed contract {condition.name!r} ({condition.location}).",
                    function_name=self.name,
                )
        return typing.cast(T, result)

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call from synchronous Python, with no model calls after compilation."""
        values = self._spec.bind(*args, **kwargs)
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
"""
