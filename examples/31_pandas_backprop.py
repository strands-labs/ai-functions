"""Pandas backpropagation demo — memory-driven code improvement (DS-1000).

Sibling of ``30_scipy_backprop`` on the same ``_ds1000_utils.py`` harness, over
the DS-1000 Pandas problem set. See example 30 for the full explanation of the
approach (recall → generate → execute → backward → consolidate, with each run's
graph reconstructed via ``build_graph``). Only the problem data and the
``library`` label differ.
"""

import asyncio
import tempfile
from dataclasses import dataclass

from _ds1000_utils import build_feedback, run_batch_parallel, run_problem
from _pandas_problems import TEST_PROBLEMS, TRAIN_PROBLEMS
from pydantic import BaseModel, Field
from strands.models import BedrockModel

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types.graph import ThreadNode

# temperature=0 for deterministic generation — the demo reproduces run to run.
_model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0", temperature=0)


@dataclass
class ExecResultView:
    """Minimal record of a run for the before/after comparison."""

    solution: str
    passed: bool


class LearningMemory(BaseModel):
    coding_patterns: str = Field(
        default="No learned patterns yet.",
        description=(
            "Concise bullet-point list (MAX 15 items) of general, reusable coding patterns "
            "and idioms for pandas/data science. Each bullet should be one sentence. "
            "Merge similar patterns into a single bullet. Do not include problem-specific details."
        ),
    )
    common_pitfalls: str = Field(
        default="No known pitfalls yet.",
        description=(
            "Concise bullet-point list (MAX 15 items) of common, reusable pitfalls "
            "and mistakes to avoid. Each bullet should be one sentence. "
            "Merge similar pitfalls into a single bullet. Do not include problem-specific details."
        ),
    )


@ai_function(str, model=_model)
def generate_code(coding_patterns: str, common_pitfalls: str, problem_prompt: str, library: str):
    """Solve the data science problem below by generating Python code.

    Output ONLY the Python code — no explanations, no markdown fences.
    The code will be inserted directly into an execution environment where
    pandas, numpy, and the input variables are already defined.

    <learned_patterns>{coding_patterns}</learned_patterns>
    <pitfalls_to_avoid>{common_pitfalls}</pitfalls_to_avoid>

    <library>{library}</library>
    <problem>
    {problem_prompt}
    </problem>
    """


def _print_result(tag: str, problem: dict, solution: str, passed: bool, error: str | None) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{tag}] {problem['id']}: {status}")
    print("--- generated code ---")
    print(solution.strip())
    if not passed and error:
        print(f"--- error ---\n{error.strip()}")


async def main(path: str) -> None:
    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    # ── Step 1: direct test with empty memory ──
    print("=" * 70, "\nStep 1 — Direct test (empty memory)\n", "=" * 70, sep="")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    direct: dict[str, ExecResultView] = {}
    for problem in TEST_PROBLEMS:
        solution, exec_result, _ = await run_problem(problem, memory, generate_code, coord, worker)
        _print_result("direct", problem, solution, exec_result.passed, exec_result.error)
        direct[problem["id"]] = ExecResultView(solution, exec_result.passed)
    memory.close()

    # ── Step 2: train on 8 problems, backprop each, consolidate once ──
    print("\n", "=" * 70, "\nStep 2 — Training (8 problems)\n", "=" * 70, sep="")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    optimizer = TextGradOptimizer(model=_model)

    batch = await run_batch_parallel(TRAIN_PROBLEMS, memory, generate_code, coord, worker)
    train_nodes: list[ThreadNode] = []
    for problem, (solution, exec_result, node) in zip(TRAIN_PROBLEMS, batch, strict=True):
        status = "PASS" if exec_result.passed else "FAIL"
        print(f"  {problem['id']}: {status}")
        optimizer.backward(node, build_feedback(problem, solution, exec_result))
        train_nodes.append(node)

    # Gather every training node under a synthetic root so one consolidate call
    # merges all gradients per parameter (grouped by backend + name).
    root = ThreadNode(
        node_id="pandas_backprop:consolidate", thread_id="pandas_backprop:consolidate", child_threads=train_nodes
    )
    for node in train_nodes:
        node.parent = root
    optimizer.consolidate(root)

    memory.close()

    # ── Step 3: re-test with the trained memory ──
    print("\n", "=" * 70, "\nStep 3 — Trained test\n", "=" * 70, sep="")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    print("\n--- learned coding_patterns ---")
    print(await memory.recall("coding_patterns"))
    print("\n--- learned common_pitfalls ---")
    print(await memory.recall("common_pitfalls"))

    for problem in TEST_PROBLEMS:
        solution, exec_result, _ = await run_problem(problem, memory, generate_code, coord, worker)
        _print_result("trained", problem, solution, exec_result.passed, exec_result.error)
        before = direct[problem["id"]]
        if not before.passed and exec_result.passed:
            print(f"\n>>> {problem['id']}: FAIL → PASS (memory-driven improvement)")
    memory.close()

    await worker.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
