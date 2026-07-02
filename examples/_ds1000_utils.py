"""Shared utilities for the DS-1000 backpropagation demo (examples 30 and 31).

Each run recalls memory parameters onto a live coordinator, generates and
executes a candidate solution, then reconstructs the run's graph node with
``build_graph(events, [memory])`` for the optimizer to backpropagate through.
Provides the pure-Python execution/test harness (``execute_and_test`` and
friends) plus ``run_problem`` / ``run_batch_parallel`` / ``build_feedback``.
"""

from __future__ import annotations

import asyncio
import signal
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_functions import build_graph

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_functions import JSONMemoryBackend
    from ai_functions.ai_thread import AIFunction
    from ai_functions.runtime import InMemoryCoordinator, LocalWorker
    from ai_functions.types.graph import ThreadNode


# ── ExecutionResult + DS-1000 executor (pure Python, no model calls) ──────────


@dataclass
class ExecutionResult:
    """Result of executing and testing a candidate solution."""

    passed: bool
    error: str | None = None
    solution_code: str = ""
    test_input: str | None = None
    expected_output: str | None = None
    actual_output: str | None = None


@contextmanager
def _timeout(seconds: int = 30) -> Iterator[None]:
    """Raise ``TimeoutError`` if the wrapped block runs longer than ``seconds``."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Execution timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _truncate_repr(obj: object, max_len: int = 200) -> str:
    s = repr(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


def _get_assertion_detail(
    solution_code: str, code_context: str, timeout_sec: int = 10
) -> tuple[str | None, str | None, str | None]:
    """Re-run the solution to capture (test_input, expected, actual) for a failed assertion."""
    try:
        with _timeout(timeout_sec):
            test_env: dict[str, Any] = {}
            exec(code_context, test_env)  # noqa: S102 — DS-1000 test harness runs by design
            exec_context_str = test_env.get("exec_context", "")
            generate_test_case = test_env.get("generate_test_case")
            if not exec_context_str or not generate_test_case:
                return None, None, None
            test_input, expected = generate_test_case(1)
            code = exec_context_str.replace("[insert]", solution_code)
            run_env: dict[str, Any] = {"test_input": test_input}
            exec(code, run_env)  # noqa: S102
            actual = run_env.get("result")
            return _truncate_repr(test_input), _truncate_repr(expected), _truncate_repr(actual)
    except Exception:  # noqa: BLE001 — best-effort detail capture
        return None, None, None


def extract_solution_code(raw_output: str) -> str:
    """Strip markdown fences and solution markers from LLM output."""
    text = raw_output
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        newline = text.find("\n", start)
        if newline != -1:
            start = newline + 1
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end]
    for marker in ("BEGIN SOLUTION", "END SOLUTION", "<code>", "</code>"):
        text = text.replace(marker, "")
    return text


def execute_and_test(solution_code: str, code_context: str, timeout_sec: int = 30) -> ExecutionResult:
    """Run the DS-1000 test harness against a candidate solution."""
    try:
        with _timeout(timeout_sec):
            test_env: dict[str, Any] = {}
            exec(code_context, test_env)  # noqa: S102
            test_fn = test_env.get("test_execution")
            if test_fn is None:
                return ExecutionResult(
                    passed=False, error="No test_execution found in code_context", solution_code=solution_code
                )
            test_fn(solution_code)
            return ExecutionResult(passed=True, solution_code=solution_code)
    except TimeoutError as e:
        return ExecutionResult(passed=False, error=str(e), solution_code=solution_code)
    except AssertionError as e:
        test_input, expected_output, actual_output = _get_assertion_detail(solution_code, code_context, timeout_sec)
        msg = f"Test assertion failed: {e}" if str(e) else "Test assertion failed"
        return ExecutionResult(
            passed=False,
            error=msg,
            solution_code=solution_code,
            test_input=test_input,
            expected_output=expected_output,
            actual_output=actual_output,
        )
    except Exception as e:  # noqa: BLE001 — any solution error is a test failure, captured as feedback
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        short_tb = "".join(tb[-3:]) if len(tb) > 3 else "".join(tb)
        return ExecutionResult(passed=False, error=f"{type(e).__name__}: {e}\n{short_tb}", solution_code=solution_code)


# ── Running generate_code over problems (our async + build_graph API) ─────────

# Memory parameters recalled per run and traced back for backward/consolidate.
PARAM_NAMES = ("coding_patterns", "common_pitfalls")


async def run_problem(
    problem: dict[str, Any],
    memory: JSONMemoryBackend,
    generate_fn: AIFunction[..., str],
    coordinator: InMemoryCoordinator,
    worker: LocalWorker,
) -> tuple[str, ExecutionResult, ThreadNode]:
    """Generate code for one problem, execute it, and return the reconstructed graph node.

    Spawns a thread on the shared coordinator, recalls each memory parameter
    into that thread's log (so ``build_graph`` can wire the parameters back to
    the live backend), runs ``generate_fn``, executes the solution, and rebuilds
    the thread node for the optimizer.
    """
    handle = await worker.spawn_locally(generate_fn, thread_name=f"generate_code:{problem['id']}")
    recalled = {name: await memory.recall(name, coordinator=coordinator, thread_id=handle.id) for name in PARAM_NAMES}
    raw = await handle.run(
        coding_patterns=recalled["coding_patterns"],
        common_pitfalls=recalled["common_pitfalls"],
        problem_prompt=problem["prompt"],
        library=problem["library"],
    )
    solution = extract_solution_code(str(raw))
    exec_result = execute_and_test(solution, problem["code_context"])
    node = build_graph(await coordinator.get_events(handle.id), [memory])
    return solution, exec_result, node


async def run_batch_parallel(
    batch: list[dict[str, Any]],
    memory: JSONMemoryBackend,
    generate_fn: AIFunction[..., str],
    coordinator: InMemoryCoordinator,
    worker: LocalWorker,
) -> list[tuple[str, ExecutionResult, ThreadNode]]:
    """Run a batch of problems concurrently, returning results in original order."""
    return await asyncio.gather(*(run_problem(problem, memory, generate_fn, coordinator, worker) for problem in batch))


def build_feedback(problem: dict[str, Any], solution_code: str, exec_result: ExecutionResult) -> str:
    """Build the optimizer feedback string, including expected/actual output when available."""
    if exec_result.passed:
        return (
            f"[{problem['library']}] {problem['id']} SOLVED.\n"
            f"Working solution:\n{solution_code}\n"
            f"Remember this pattern for similar future problems."
        )

    error_parts = [f"Error: {exec_result.error}"]
    if exec_result.test_input:
        error_parts.append(f"Test input: {exec_result.test_input}")
    if exec_result.expected_output:
        error_parts.append(f"Expected output: {exec_result.expected_output}")
    if exec_result.actual_output:
        error_parts.append(f"Actual output: {exec_result.actual_output}")

    return f"[{problem['library']}] {problem['id']} FAILED.\nYour code:\n{solution_code}\n" + "\n".join(error_parts)
