"""Native ABI, ownership, threading, and integer-representation regressions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import os
import random
import struct
import sys
import threading

import pytest

from ai_functions.experimental.verified_compile import verified_ai_compile
from ai_functions.experimental.verified_compile.compiler import Candidate
from ai_functions.testing import ScriptedModel, Turn


def make_function(fn, contract, candidate, cache):
    model = ScriptedModel([Turn(tool_calls=(("Candidate", candidate.model_dump()),))])
    return verified_ai_compile(post_conditions=[contract], model=model, cache_dir=cache, max_attempts=0)(fn)


def integer(value: int) -> int:
    """Return the integer unchanged."""


def same(result, value):
    assert result == value


IDENTITY = Candidate(implementation="v0", proof="by intro v0 h; simp [pre, post, post0, implementation]")


async def test_integer_digits_cross_python_and_gmp_boundaries(tmp_path, native_runtime):
    fn = make_function(integer, same, IDENTITY, tmp_path)
    await fn.compile()
    randomizer = random.Random(7403)
    values = [0, 1, -1]
    for bits in (15, 29, 30, 31, 32, 59, 60, 63, 64, 65, 89, 90, 127, 128, 129, 1024, 20000, 100000):
        for number in (2**bits - 1, 2**bits, 2**bits + 1, randomizer.getrandbits(bits) | 1):
            values.extend((number, -number))
    for value in values:
        assert fn.run_sync(value) == value
    # The same primitive must preserve ownership on Python worker threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(fn.run_sync, values)) == values


async def test_native_list_ownership_and_input_references(tmp_path, native_runtime):
    def reverse(values: list[int]) -> list[int]:
        """Reverse the sequence."""

    def length_preserved(result, values):
        assert len(result) == len(values)

    fn = make_function(
        reverse,
        length_preserved,
        Candidate(implementation="List.reverse v0", proof="by intro v0 h; simp [pre, post, post0, implementation]"),
        tmp_path,
    )
    await fn.compile()
    values = [-(2**20000), 2**64, 0, -(2**31), 2**20000 + 19]
    expected = values[::-1]
    # Account for the independent oracle's references before checking for leaks.
    references = [sys.getrefcount(item) for item in values]
    for _ in range(300):
        result = fn.run_sync(values)
        assert result == expected
        assert result is not values
        del result
    assert [sys.getrefcount(item) for item in values] == references
    assert fn.run_sync([]) == []
    large = list(range(10000))
    assert fn.run_sync(large) == large[::-1]
    assert large == list(range(10000))


async def test_async_arguments_are_frozen_before_yielding(tmp_path, native_runtime, monkeypatch):
    def sequence(value: list[int]) -> list[int]:
        """Return the sequence unchanged."""

    fn = make_function(sequence, same, IDENTITY, tmp_path)
    await fn.compile()
    original = fn._invoke
    entered, release = threading.Event(), threading.Event()

    def delayed(values):
        entered.set()
        assert release.wait(10)
        return original(values)

    monkeypatch.setattr(fn, "_invoke", delayed)
    value = [2**20000, -7]
    expected = value[:]
    task = asyncio.create_task(fn(value))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        value[:] = [False, 3.5]
    finally:
        release.set()
    assert await task == expected


async def test_mixed_native_argument_registers(tmp_path, native_runtime):
    def mixed(count: int, enabled: bool, threshold: float, values: list[int]) -> bool:
        """Check a mixed scalar/list condition."""

    def contract(result, count, enabled, threshold, values):
        assert result == (enabled and count == len(values) and threshold >= 0.0)

    fn = make_function(
        mixed,
        contract,
        Candidate(
            implementation="v1 && decide (v0 = Int.ofNat v3.length) && Float.le (Float.ofBits 0) v2",
            proof="by intro v0 v1 v2 v3 h; simp [pre, post, post0, implementation]",
        ),
        tmp_path,
    )
    await fn.compile()
    for count in (-1, 0, 2, 2**20000):
        for enabled in (False, True):
            for threshold in (-1.0, -0.0, 1.0, math.inf, math.nan):
                assert fn.run_sync(count, enabled, threshold, [3, 7]) is (enabled and count == 2 and threshold >= 0.0)


async def test_large_arity_and_consumed_unused_arguments(tmp_path, native_runtime):
    def ninth(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> int:
        """Return the ninth argument."""

    def contract(result, i):
        assert result == i

    fn = make_function(
        ninth,
        contract,
        Candidate(
            implementation="v8",
            proof="by intro v0 v1 v2 v3 v4 v5 v6 v7 v8 h; simp [pre, post, post0, implementation]",
        ),
        tmp_path,
    )
    await fn.compile()
    values = [2 ** (2000 + index) for index in range(9)]
    for _ in range(100):
        assert fn.run_sync(*values) == values[-1]


async def test_float_bit_patterns_through_native_calls(tmp_path, native_runtime):
    def echo(value: float) -> float:
        """Preserve the float's value and classification."""

    def contract(result, value):
        assert math.isnan(result) == math.isnan(value)
        assert math.isfinite(result) == math.isfinite(value)

    fn = make_function(echo, contract, IDENTITY, tmp_path)
    await fn.compile()
    randomizer = random.Random(629)
    bits = [0, 1, 2**63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000001, 0xFFF8123456789ABC]
    bits.extend(randomizer.getrandbits(64) for _ in range(500))
    for pattern in bits:
        value = struct.unpack("=d", struct.pack("=Q", pattern))[0]
        output = fn.run_sync(value)
        expected = 0x7FF8000000000000 if math.isnan(value) else pattern
        assert struct.unpack("=Q", struct.pack("=d", output))[0] == expected


async def test_bridge_rejects_signature_mismatch_and_bad_values(tmp_path, native_runtime):
    fn = make_function(integer, same, IDENTITY, tmp_path)
    await fn.compile()
    bridge = native_runtime.bridge()
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = str(fn.artifact_dir / (fn._artifact.module + extension))
    with pytest.raises(ImportError, match="types do not match"):
        bridge.load(library, bytes([2, 2]))
    direct = bridge.load(library, bytes([1, 1]))
    for bad in (True, 1.5, "1", [1]):
        with pytest.raises(TypeError, match="exact Python ints"):
            direct({"v0": bad})
    with pytest.raises(TypeError, match="Missing native argument"):
        direct({"wrong": 1})
    with pytest.raises(TypeError, match="Incorrect number"):
        direct({})
    assert direct({"v0": 2**20000}) == 2**20000


async def test_native_execution_releases_the_gil(tmp_path, native_runtime):
    def total(n: int) -> int:
        """Sum the natural numbers below the magnitude of n."""

    def nonnegative(result):
        assert result >= 0

    fn = make_function(
        total,
        nonnegative,
        Candidate(
            implementation="Int.ofNat (List.foldl (fun s x => s + x) 0 (List.range v0.natAbs))",
            proof="by intro v0 h; simp [pre, post, post0, implementation]",
        ),
        tmp_path,
    )
    await fn.compile()
    fn.run_sync(0)  # Resolve the native entry before starting the scheduling check.
    ready, go, ran, finish = (threading.Event() for _ in range(4))

    def observer():
        ready.set()
        go.wait()
        ran.set()
        finish.wait()

    thread = threading.Thread(target=observer)
    thread.start()
    previous = sys.getswitchinterval()
    try:
        assert ready.wait(10)
        sys.setswitchinterval(10)
        go.set()
        n = 1_000_000
        assert fn._artifact.invoke(fn._spec, {"v0": n}) == n * (n - 1) // 2
        assert ran.is_set(), "The observer could not acquire the GIL during native execution"
    finally:
        sys.setswitchinterval(previous)
        finish.set()
        go.set()
        thread.join(10)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork regression")
async def test_cached_native_callable_rejects_fork_reuse(tmp_path, native_runtime):
    fn = make_function(integer, same, IDENTITY, tmp_path)
    await fn.compile()
    assert fn.run_sync(7) == 7
    pid = os.fork()
    if pid == 0:
        try:
            fn._artifact.invoke(fn._spec, {"v0": 8})
        except RuntimeError as error:
            os._exit(0 if "after fork" in str(error) else 2)
        os._exit(3)
    _, status = await asyncio.to_thread(os.waitpid, pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
