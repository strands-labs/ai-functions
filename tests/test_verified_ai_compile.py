"""Real proof/compiler/FFI tests with a deterministic model and no model service."""

from __future__ import annotations

import asyncio
import json
import math
import runpy
import struct
import subprocess
import sys
from bisect import bisect_left
from itertools import product
from pathlib import Path

import pytest

from ai_functions import scope
from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.verified_compile import verified_ai_compile
from ai_functions.experimental.verified_compile.compiler import Candidate, validate_candidate
from ai_functions.experimental.verified_compile.errors import (
    CandidateError,
    CompilerError,
    ContractError,
    SynthesisError,
)
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import EventKind


def clamp(x: int, lo: int = 0, hi: int = 10) -> int:
    """Clamp x to the inclusive interval [lo, hi]."""


def valid_bounds(lo, hi):
    assert lo <= hi


def check_clamp(result, x, lo, hi):
    assert lo <= result <= hi
    if x < lo:
        assert result == lo
    elif x > hi:
        assert result == hi
    else:
        assert result == x


GOOD = Candidate(
    implementation="if v0 < v1 then v1 else if v0 > v2 then v2 else v0",
    proof=(
        "by\n  intro v0 v1 v2 h\n"
        "  simp only [pre, pre0, post, post0, implementation] at *\n"
        "  repeat first | omega | split | constructor"
    ),
)
WRONG = Candidate(
    implementation="v0",
    proof="by\n  intro v0 v1 v2 h\n  simp_all [pre, post, implementation]\n  omega",
)


def model(*candidates):
    return ScriptedModel([Turn(tool_calls=(("Candidate", c.model_dump()),)) for c in candidates])


def decorate(cache, llm, **kwargs):
    return verified_ai_compile(
        pre_conditions=[valid_bounds],
        post_conditions=[check_clamp],
        cache_dir=cache,
        model=llm,
        **kwargs,
    )(clamp)


async def test_real_proof_retry_native_calls_and_unbounded_integers(tmp_path, native_runtime, caplog):
    llm = model(WRONG, GOOD)
    fn = decorate(tmp_path, llm, max_attempts=1)
    assert fn.artifact_dir is None
    events = []
    with caplog.at_level("INFO", logger="ai_functions.experimental.verified_compile.function"):
        async with scope(on_event=events.append):
            assert await fn(12) == 10
    candidates = [event for event in events if event.kind == EventKind.TOOL_CALL]
    assert [event.arguments for event in candidates] == [WRONG.model_dump(), GOOD.model_dump()]
    assert "Verification failed for clamp:" in caplog.text
    assert "Synthesizing clamp: attempt 2/2" in caplog.text
    assert "Verified and compiled clamp" in caplog.text
    assert fn.is_compiled
    assert fn.artifact_dir is not None
    assert next(fn.artifact_dir.glob("*.lean")).is_file()
    assert fn.run_sync(-5, lo=-2, hi=8) == -2
    big = 2**20000
    assert await fn(big, -big * 2, big * 2) == big
    assert fn.run_sync(-big, -big * 2, big * 2) == -big
    assert llm.remaining_turns == 0
    assert len(list(tmp_path.glob("*/manifest.json"))) == 1


async def test_concurrent_calls_compile_once_and_reuse_across_instances(tmp_path, native_runtime):
    llm = model(GOOD)
    fn = decorate(tmp_path, llm, max_attempts=0)
    assert await asyncio.gather(*(fn(x) for x in [-1, 0, 3, 10, 12])) == [0, 0, 3, 10, 10]
    assert llm.remaining_turns == 0
    cached = decorate(tmp_path, model(), max_attempts=0)
    assert await cached.compile() is cached
    assert cached.run_sync(7) == 7
    assert cached.artifact_dir == fn.artifact_dir


async def test_property_specified_median_matches_independent_sorting_oracle(tmp_path, native_runtime):
    example = Path(__file__).resolve().parents[1] / "examples" / "verified_median.py"
    definitions = runpy.run_path(str(example), run_name="median_test_definitions")
    candidate = Candidate(
        implementation="max (min v0 v1) (min (max v0 v1) v2)",
        proof=(
            "by\n  intro v0 v1 v2 h\n"
            "  simp_all [pre, post, post0, implementation, Int.min_def, Int.max_def]\n"
            "  repeat first | omega | (split <;> simp_all)"
        ),
    )
    fn = verified_ai_compile(
        post_conditions=[definitions["is_median"]],
        model=model(candidate),
        cache_dir=tmp_path,
        max_attempts=0,
    )(definitions["median_of_three"].__wrapped__)
    await fn.compile()
    for values in product((-7, 0, 9), repeat=3):
        assert fn.run_sync(*values) == sorted(values)[1]
    assert fn.run_sync(2**10000, -(2**10000), 42) == 42


async def test_quantified_lower_bound_matches_bisect(tmp_path, native_runtime):
    # Captured from a real Opus 5 synthesis, then checked by the kernel. The
    # independent Python oracle tests execution and the conversion boundary.
    root = Path(__file__).resolve().parents[1]
    definitions = runpy.run_path(str(root / "examples" / "verified_lower_bound.py"), run_name="lower_bound_tests")
    candidate = Candidate.model_validate_json((root / "tests" / "fixtures" / "verified_lower_bound.json").read_text())
    fn = verified_ai_compile(
        pre_conditions=[definitions["sorted_values"]],
        post_conditions=[definitions["insertion_position"]],
        model=model(candidate),
        cache_dir=tmp_path,
        max_attempts=0,
        check_pre_conditions=True,
    )(definitions["lower_bound"].__wrapped__)
    await fn.compile()
    for length in range(5):
        for values in sorted({tuple(sorted(values)) for values in product((-2, 0, 2), repeat=length)}):
            for key in range(-3, 4):
                assert fn.run_sync(list(values), key) == bisect_left(values, key)
    large = [-(2**10000), 0, 2**10000]
    assert fn.run_sync(large, 2**10000) == 2
    with pytest.raises(ContractError, match="sorted_values"):
        fn.run_sync([2, 1], 1)


async def test_maximum_payout_matches_exhaustive_fee_accounting(tmp_path, native_runtime):
    root = Path(__file__).resolve().parents[1]
    definitions = runpy.run_path(str(root / "examples" / "verified_payout.py"), run_name="payout_tests")
    candidate = Candidate.model_validate_json((root / "tests" / "fixtures" / "verified_payout.json").read_text())
    fn = verified_ai_compile(
        pre_conditions=[definitions["payout_inputs"]],
        post_conditions=[definitions["maximum_safe_payout"]],
        model=model(candidate),
        cache_dir=tmp_path,
        max_attempts=0,
        check_pre_conditions=True,
    )(definitions["max_payout"].__wrapped__)
    await fn.compile()
    for balance, fixed, rate, cap in product(range(41), (0, 1, 3, 10, 50), (0, 1, 290, 3333, 10000), (0, 1, 5, 20, 50)):
        # Enumerate proposed payments and compute the actual rounded fees.
        # This oracle does not use the generated closed-form payout formula.
        feasible = []
        for proposed in range(min(balance, cap) + 1):
            fees = fixed + (proposed * rate + 9999) // 10000 if proposed else 0
            if proposed + fees <= balance:
                feasible.append(proposed)
        expected = max(feasible)
        assert fn.run_sync(balance, fixed, rate, cap) == expected
        for wrong in (expected - 1, expected + 1):
            with pytest.raises(AssertionError):
                definitions["maximum_safe_payout"](wrong, balance, fixed, rate, cap)
    assert fn.run_sync(10000, 30, 290, 20000) == 9689
    huge = 2**20000
    assert fn.run_sync(huge, 0, 0, huge) == huge
    assert fn.run_sync(huge, 0, 10000, huge) == huge // 2
    for invalid in ((-1, 0, 0, 1), (1, -1, 0, 1), (1, 0, 10001, 1), (1, 0, 0, -1)):
        with pytest.raises(ContractError, match="payout_inputs"):
            fn.run_sync(*invalid)


async def test_level_payment_matches_a_cent_by_cent_search(tmp_path, native_runtime):
    # Captured from a real Opus 5.5 synthesis. The implementation never calls
    # balance_after: it bisects over a Nat simulation that stops once the loan is
    # paid off, and the proof relates that simulation to the contract's loop.
    root = Path(__file__).resolve().parents[1]
    definitions = runpy.run_path(str(root / "examples" / "verified_loan_payment.py"), run_name="loan_tests")
    candidate = Candidate.model_validate_json((root / "tests" / "fixtures" / "verified_loan_payment.json").read_text())
    assert "balance_after" not in candidate.implementation
    fn = verified_ai_compile(
        pre_conditions=[definitions["loan_terms"]],
        post_conditions=[definitions["smallest_payment"]],
        model=model(candidate),
        cache_dir=tmp_path,
        max_attempts=0,
    )(definitions["level_payment"].__wrapped__)
    await fn.compile()
    balance_after = definitions["balance_after"]
    for principal, rate, periods in product(range(0, 60, 7), (0, 1, 50, 5000, 10000), (1, 2, 3, 12)):
        # The oracle scans one cent at a time, so it does not share the bisection.
        expected = next(p for p in range(2 * principal + 1) if balance_after(principal, rate, p, periods) <= 0)
        assert fn.run_sync(principal, rate, periods) == expected
    assert fn.run_sync(25_000_000, 50, 360) == 149_888
    huge = 2**200
    payment = fn.run_sync(huge, 50, 12)
    assert balance_after(huge, 50, payment, 12) <= 0 < balance_after(huge, 50, payment - 1, 12)
    # Outside loan_terms the proof says nothing, and by default nothing checks. At 200%
    # interest per period no payment up to twice the principal clears the loan, so the
    # search returns its upper bound, which leaves 1,000 cents owed.
    assert fn.run_sync(1_000, 20_000, 12) == 2_000
    assert balance_after(1_000, 20_000, 2_000, 12) == 1_000
    checked = verified_ai_compile(
        pre_conditions=[definitions["loan_terms"]],
        post_conditions=[definitions["smallest_payment"]],
        model=model(),
        cache_dir=tmp_path,
        check_pre_conditions=True,
    )(definitions["level_payment"].__wrapped__)
    for invalid in ((-1, 0, 1), (1, -1, 1), (1, 10001, 1), (1, 0, 0), (1, 0, 1201)):
        with pytest.raises(ContractError, match="loan_terms"):
            checked.run_sync(*invalid)
    # The options are not part of the cache key, so the checked function reuses the artifact.
    assert checked.run_sync(25_000_000, 50, 360) == 149_888


async def test_cached_artifact_works_in_a_fresh_python_process(tmp_path, native_runtime):
    fn = decorate(tmp_path, model(GOOD), max_attempts=0)
    await fn.compile()
    script = """import importlib.util, sys
s = importlib.util.spec_from_file_location('verified_cases', sys.argv[1])
m = importlib.util.module_from_spec(s)
sys.modules[s.name] = m
s.loader.exec_module(m)
f = m.decorate(sys.argv[2], m.model(), max_attempts=0)
print(f.run_sync(8))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, __file__, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8"


async def test_exhaustion_never_publishes_or_calls_unverified_code(tmp_path, native_runtime, monkeypatch):
    from ai_functions.experimental.verified_compile.compiler import Artifact

    def must_not_run(*args, **kwargs):
        pytest.fail("Unverified native code was invoked")

    monkeypatch.setattr(Artifact, "invoke", must_not_run)
    bad = Candidate(implementation="0", proof="by sorry")
    llm = model(bad, bad, bad)
    fn = decorate(tmp_path, llm, max_attempts=2)
    with pytest.raises(SynthesisError, match="after 3 attempt") as caught:
        await fn(5)
    assert caught.value.attempts == 3
    assert len(caught.value.diagnostics) == 3
    assert "Lean" not in str(caught.value)
    assert not fn.is_compiled
    assert llm.remaining_turns == 0
    assert not list(tmp_path.glob("*/manifest.json"))
    assert not list(tmp_path.glob("candidate-*"))


async def test_precondition_and_type_errors_do_not_start_synthesis(tmp_path, monkeypatch):
    def must_not_resolve_runtime(*args, **kwargs):
        pytest.fail("Invalid input reached compiler setup")

    monkeypatch.setattr("ai_functions.experimental.verified_compile.function.resolve_runtime", must_not_resolve_runtime)
    fn = decorate(tmp_path, model(), max_attempts=0, check_pre_conditions=True)
    with pytest.raises(ContractError, match="valid_bounds"):
        await fn(1, 10, -10)
    # Argument types are checked whether or not the contracts are.
    for typed in (fn, decorate(tmp_path, model(), max_attempts=0)):
        with pytest.raises(TypeError, match="must be int"):
            typed.run_sync(True)
    assert not fn.is_compiled


async def test_runtime_contract_checks_are_opt_in(tmp_path, monkeypatch):
    # A stand-in for a wrong native result, as a bug in the trusted compiler,
    # runtime, or value conversion could produce. The proof excludes it only when
    # those components are correct; the optional check catches it on the inputs used.
    class WrongArtifact:
        def invoke(self, spec, values):
            return 99

    unchecked = decorate(tmp_path, model(), max_attempts=0)
    checked = decorate(tmp_path, model(), max_attempts=0, check_pre_conditions=True, check_post_conditions=True)
    for fn in (unchecked, checked):
        monkeypatch.setattr(fn, "_artifact", WrongArtifact())
    assert unchecked.run_sync(1, 10, -10) == 99
    assert await unchecked(5) == 99
    with pytest.raises(ContractError, match="valid_bounds"):
        checked.run_sync(1, 10, -10)
    with pytest.raises(CompilerError, match="failed contract 'check_clamp'"):
        await checked(5)


async def test_sampled_test_reports_passes_failures_and_untested_candidates(tmp_path, native_runtime):
    from ai_functions.experimental.verified_compile.compiler import run_sampled_test, sample_inputs

    spec = decorate(tmp_path, model())._spec
    inputs = sample_inputs(spec)

    async def report(implementation, cases=inputs):
        return await run_sampled_test(native_runtime, spec, implementation, cases, tmp_path, 120)

    assert (await report(GOOD.implementation)).startswith(f"PASSED on {len(inputs)} sampled inputs")
    wrong = await report(WRONG.implementation)
    assert wrong.startswith("WRONG on") and "counterexample v0=" in wrong
    assert (await report("v0 +")).startswith("The implementation did not compile")
    assert (await report("IO.println 1")).startswith("REJECTED before running")
    assert (await report(GOOD.implementation, [])).startswith("INCONCLUSIVE")


async def test_synthesis_tests_and_checks_before_it_submits(tmp_path, native_runtime):
    llm = ScriptedModel(
        [
            Turn(tool_calls=(("test_implementation", {"implementation": WRONG.implementation}),)),
            Turn(tool_calls=(("test_implementation", {"implementation": GOOD.implementation}),)),
            Turn(tool_calls=(("check_lean", GOOD.model_dump()),)),
            Turn(tool_calls=(("Candidate", GOOD.model_dump()),)),
        ]
    )
    fn = decorate(tmp_path, llm, max_attempts=0)
    events = []
    async with scope(on_event=events.append):
        assert await fn(12) == 10
    reports = [json.dumps(event.content) for event in events if event.kind == EventKind.TOOL_RESULT]
    assert "WRONG on" in reports[0]
    assert "PASSED on" in reports[1]
    assert "VERIFIED" in reports[2]
    assert llm.remaining_turns == 0


async def test_missing_toolchain_fails_before_model_calls(tmp_path, monkeypatch):
    from ai_functions.experimental.lean import LeanConfig, LeanSetupError

    def unavailable(*args, **kwargs):
        raise LeanSetupError("Lean is missing in offline mode")

    monkeypatch.setattr(LeanConfig, "setup", unavailable)
    llm = model(GOOD)
    fn = decorate(tmp_path / "cache", llm)
    with pytest.raises(CompilerError, match="offline"):
        await fn.compile()
    assert llm.remaining_turns == 1


async def test_boolean_inputs_and_result_via_native_ffi(tmp_path, native_runtime):
    def xor(a: bool, b: bool) -> bool:
        """Exclusive or."""

    def contract(result, a, b):
        return PostConditionResult(passed=result == (a != b), message="expected xor")

    candidate = Candidate(
        implementation="v0 != v1",
        proof="by\n  intro v0 v1 h\n  cases v0 <;> cases v1 <;> decide",
    )
    llm = model(candidate)
    fn = verified_ai_compile[bool](post_conditions=[contract], model=llm, cache_dir=tmp_path, max_attempts=0)(xor)
    await fn.compile()
    for a in (False, True):
        for b in (False, True):
            assert fn.run_sync(a, b) is (a != b)
    assert llm.remaining_turns == 0


async def test_zero_argument_function_and_explicit_sync_compile(tmp_path, native_runtime):
    def seven() -> int:
        """Return seven."""

    def contract(result):
        assert result == 7

    llm = model(Candidate(implementation="7", proof="by simp [pre, post, post0, implementation]"))
    fn = verified_ai_compile(post_conditions=[contract], cache_dir=tmp_path, model=llm)(seven)
    # The sync bridge must also work when the caller already has an event loop.
    assert fn.compile_sync() is fn
    assert await fn() == 7


async def test_core_propositional_simp_lemmas_are_accepted(tmp_path, native_runtime):
    def identity(value: int) -> int:
        """Return value."""

    def contract(result, value):
        assert result == value and True

    # `and True` in a contract leaves an `∧ True` conjunct, closed by the core
    # propositional lemma and_true. The lexical filter must admit it.
    candidate = Candidate(
        implementation="v0",
        proof=("by\n  intro v0 h\n  simp only [post, post0, implementation, and_true]"),
    )
    fn = verified_ai_compile(post_conditions=[contract], model=model(candidate), cache_dir=tmp_path, max_attempts=0)(
        identity
    )
    await fn.compile()
    assert fn.run_sync(-17) == -17


async def test_list_quantifiers_and_list_results_via_native_ffi(tmp_path, native_runtime):
    def all_below(values: list[int], limit: int) -> bool:
        """Check whether every value is below the limit."""

    def bounded(result, values, limit):
        assert result == all(value < limit for value in values)

    candidate = Candidate(
        implementation="List.all v0 (fun t0 => decide (t0 < v1))",
        proof="by intro v0 v1 h; simp [pre, post, post0, implementation]",
    )
    fn = verified_ai_compile(post_conditions=[bounded], model=model(candidate), cache_dir=tmp_path, max_attempts=0)(
        all_below
    )
    assert await fn([], 0) is True
    assert fn.run_sync([-4, 3], 4) is True
    assert fn.run_sync([-4, 4], 4) is False

    def echo(values: list[int]) -> list[int]:
        """Copy an integer list."""

    def same(result, values):
        assert result == values

    echo_model = model(Candidate(implementation="v0", proof="by intro v0 h; simp [pre, post, post0, implementation]"))
    copied = verified_ai_compile(post_conditions=[same], model=echo_model, cache_dir=tmp_path, max_attempts=0)(echo)
    values = [-(2**20000), 0, 2**20000]
    assert await copied(values) == values
    assert copied.run_sync([]) == []


async def test_float_classification_signed_zero_and_nan_normalization(tmp_path, native_runtime):
    def finite_nonnegative(x: float) -> bool:
        """Whether x is finite and nonnegative."""

    def classified(result, x):
        assert result == (math.isfinite(x) and x >= 0.0)

    candidate = Candidate(
        implementation="Float.isFinite v0 && Float.le (Float.ofBits (0x0000000000000000 : UInt64)) v0",
        proof="by intro v0 h; simp [pre, post, post0, implementation]",
    )
    fn = verified_ai_compile(post_conditions=[classified], model=model(candidate), cache_dir=tmp_path, max_attempts=0)(
        finite_nonnegative
    )
    for value in (0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan, 5e-324):
        assert await fn(value) is (math.isfinite(value) and value >= 0.0)

    def identity(x: float) -> float:
        """Preserve a float."""

    def preserves_classification(result, x):
        assert math.isnan(result) == math.isnan(x)
        assert math.isfinite(result) == math.isfinite(x)

    copied = verified_ai_compile(
        post_conditions=[preserves_classification],
        model=model(Candidate(implementation="v0", proof="by intro v0 h; simp [pre, post, post0, implementation]")),
        cache_dir=tmp_path,
        max_attempts=0,
    )(identity)
    for bits in (0, 0x8000000000000000, 1, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000012):
        value = struct.unpack(">d", struct.pack(">Q", bits))[0]
        result = await copied(value)
        if math.isnan(value):
            assert math.isnan(result)
            assert struct.pack(">d", result).hex() == "7ff8000000000000"
        else:
            assert struct.pack(">d", result) == struct.pack(">d", value)


async def test_float_arithmetic_preserves_rounding_and_does_not_fuse_operations(tmp_path, native_runtime):
    def multiply_add_equals(a: float, b: float, c: float, expected: float) -> bool:
        """Compare a separately rounded multiply/add with the expected value."""

    def contract(result, a, b, c, expected):
        assert result == ((a * b + c) == expected)

    candidate = Candidate(
        implementation="Float.beq (v0 * v1 + v2) v3",
        proof="by intro v0 v1 v2 v3 h; simp [pre, post, post0, implementation]",
    )
    fn = verified_ai_compile(post_conditions=[contract], model=model(candidate), cache_dir=tmp_path, max_attempts=0)(
        multiply_add_equals
    )
    assert await fn(1.0 + 2**-27, 1.0 - 2**-27, -1.0, 0.0) is True
    assert fn.run_sync(math.inf, 0.0, 1.0, math.nan) is False
    assert fn.run_sync(1e308, 2.0, 0.0, math.inf) is True


async def test_output_token_exhaustion_retries_within_the_budget(tmp_path, native_runtime):
    from strands.types.exceptions import MaxTokensReachedException

    class TruncatedOnce(ScriptedModel):
        truncated = False

        def stream(self, *args, **kwargs):
            if not self.truncated:
                self.truncated = True
                raise MaxTokensReachedException("truncated")
            return super().stream(*args, **kwargs)

    llm = TruncatedOnce([Turn(tool_calls=(("Candidate", GOOD.model_dump()),))])
    fn = decorate(tmp_path, llm, max_attempts=1)
    assert await fn(12) == 10
    assert llm.remaining_turns == 0


async def test_default_synthesis_uses_opus_5_5_with_a_proof_sized_budget(tmp_path, native_runtime, monkeypatch):
    settings = {}

    def configured_model(**kwargs):
        settings.update(kwargs)
        return model(GOOD)

    monkeypatch.setattr("ai_functions.experimental.verified_compile.function.BedrockModel", configured_model)
    fn = decorate(tmp_path, None, max_attempts=0)
    assert await fn(12) == 10
    assert settings["model_id"] == "global.anthropic.claude-opus-5-5"
    assert settings["max_tokens"] == 32768
    assert settings["boto_client_config"].read_timeout == 900


async def test_corrupted_cache_is_rebuilt(tmp_path, native_runtime):
    fn = decorate(tmp_path, model(GOOD), max_attempts=0)
    await fn.compile()
    manifest = next(tmp_path.glob("*/manifest.json"))
    manifest.write_text("invalid JSON")
    llm = model(GOOD)
    rebuilt = decorate(tmp_path, llm, max_attempts=0)
    assert await rebuilt(15) == 10
    assert llm.remaining_turns == 0
    assert json.loads(manifest.read_text())["module"].startswith("Verified")


@pytest.mark.parametrize(
    "proof",
    [
        "by sorry",
        "by admit",
        "by native_decide",
        "by run_tac do pure ()",
        "by\n  rfl\nset_option debug.skipKernelTC true",
        "by exact unsafeCast True.intro",
        "by\n  rfl\naxiom bad : False",
        'by exact "injected"',
        "by rfl /- comment -/",
    ],
)
def test_untrusted_proof_commands_and_axioms_are_rejected(proof):
    with pytest.raises(CandidateError):
        validate_candidate(Candidate(implementation="v0", proof=proof), 1)


@pytest.mark.parametrize(
    "implementation",
    [
        "by native_decide",
        "IO.println 1",
        "v0\ninitialize bad : IO Unit := pure ()",
        "v0\nend\nimport Fake",
        "v0 -- comment",
        "System.Platform.isWindows",
        "v99",
    ],
)
def test_implementation_cannot_escape_its_expression(implementation):
    with pytest.raises(CandidateError):
        validate_candidate(Candidate(implementation=implementation, proof="by rfl"), 1)


def test_only_one_new_decorator_is_exported():
    import ai_functions

    assert "verified_ai_compile" not in ai_functions.__all__
    assert not hasattr(ai_functions, "ai_verified_function")
    assert callable(verified_ai_compile)
    assert not hasattr(ai_functions, "verified_ai_compile")


def test_original_function_attributes_cannot_override_compilation_state(tmp_path, monkeypatch):
    monkeypatch.setattr(clamp, "_artifact", object(), raising=False)
    fn = decorate(tmp_path, model(), max_attempts=0)
    assert not fn.is_compiled
    assert fn.__name__ == "clamp"


@pytest.mark.parametrize("partial", [False, True])
async def test_missing_provider_credentials_fail_once_with_setup_guidance(tmp_path, monkeypatch, partial):
    from types import SimpleNamespace

    from botocore.exceptions import NoCredentialsError, PartialCredentialsError

    from ai_functions.experimental.verified_compile.errors import ModelSetupError

    class MissingCredentialsModel(ScriptedModel):
        calls = 0

        def stream(self, *args, **kwargs):
            self.calls += 1
            if partial:
                raise PartialCredentialsError(provider="test", cred_var="secret_key")
            raise NoCredentialsError()

    monkeypatch.setattr(
        "ai_functions.experimental.verified_compile.function.resolve_runtime",
        lambda *args, **kwargs: SimpleNamespace(identity="test", preflight=lambda timeout: None),
    )
    llm = MissingCredentialsModel([])
    fn = decorate(tmp_path / "cache", llm, max_attempts=3)
    with pytest.raises(ModelSetupError, match="AWS_PROFILE") as caught:
        await fn.compile()
    assert caught.value.function_name == "clamp"
    assert llm.calls == 1
    assert not fn.is_compiled
    assert not list(tmp_path.glob("**/manifest.json"))
