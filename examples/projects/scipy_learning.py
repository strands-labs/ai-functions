"""Scipy backpropagation demo — memory-driven code improvement (DS-1000).

Teaches an agent to write correct scipy code by backpropagating execution
errors into memory:

  1. Direct test  — solve the test problem with empty memory (it fails).
  2. Training     — solve 8 training problems, backprop each result's
                    pass/fail feedback into ``coding_patterns`` /
                    ``common_pitfalls``, then consolidate once.
  3. Trained test — re-solve the test problem with the learned memory.

The mental model is PyTorch autograd (see ``../memory/optimization.py``): ``recall`` reads a
learnable weight, ``backward`` distributes textual gradients, ``consolidate``
writes improvements back. The 8 training nodes are gathered under a synthetic
root so a single ``consolidate`` merges every parameter's gradients (grouped by
backend + name) into one update.
"""

import asyncio
import tempfile
from dataclasses import dataclass

from example_helpers.ds1000 import build_feedback, run_batch_parallel, run_problem
from example_helpers.ds1000_scipy import TEST_PROBLEMS, TRAIN_PROBLEMS
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field
from strands.models import BedrockModel

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function
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
            "and idioms for scipy/data science. Each bullet should be one sentence. "
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


@ai_function(model=_model)
def generate_code(coding_patterns: str, common_pitfalls: str, problem_prompt: str, library: str) -> str:
    """Solve the data science problem below by generating Python code.

    Output ONLY the Python code — no explanations, no markdown fences.
    The code will be inserted directly into an execution environment where
    scipy, numpy, and the input variables are already defined.

    <learned_patterns>{coding_patterns}</learned_patterns>
    <pitfalls_to_avoid>{common_pitfalls}</pitfalls_to_avoid>

    <library>{library}</library>
    <problem>
    {problem_prompt}
    </problem>
    """


def _show_result(tag: str, problem: dict, solution: str, passed: bool, error: str | None) -> None:
    status = "PASS" if passed else "FAIL"
    content = solution.strip()
    if not passed and error:
        content += "\n\n# error:\n# " + error.strip().replace("\n", "\n# ")
    display(f"[{tag}] {problem['id']}: {status}", content, lang="python")


async def main(path: str) -> None:
    # Step 1: direct test with empty memory (expected to fail).
    rule("Step 1 — Direct test (empty memory)")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    direct: dict[str, ExecResultView] = {}
    for problem in TEST_PROBLEMS:
        solution, exec_result, _ = await run_problem(problem, memory, generate_code)
        _show_result("direct", problem, solution, exec_result.passed, exec_result.error)
        direct[problem["id"]] = ExecResultView(solution, exec_result.passed)
    memory.close()

    # Step 2: train on 8 problems, backprop each, consolidate once.
    rule("Step 2 — Training (8 problems)")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    optimizer = TextGradOptimizer(model=_model)

    batch = await run_batch_parallel(TRAIN_PROBLEMS, memory, generate_code)
    train_nodes: list[ThreadNode] = []
    train_status: list[str] = []
    for problem, (solution, exec_result, node) in zip(TRAIN_PROBLEMS, batch, strict=True):
        status = "PASS" if exec_result.passed else "FAIL"
        train_status.append(f"{problem['id']}: {status}")
        optimizer.backward(node, build_feedback(problem, solution, exec_result))
        train_nodes.append(node)
    display("Training Results", "\n".join(train_status), lang="text")

    # Gather every training node under a synthetic root so one consolidate call
    # merges all gradients per parameter (grouped by backend + name).
    root = ThreadNode(
        node_id="scipy_backprop:consolidate", thread_id="scipy_backprop:consolidate", child_threads=train_nodes
    )
    for node in train_nodes:
        node.parent = root
    optimizer.consolidate(root)

    memory.close()

    # Step 3: re-test with the trained memory.
    rule("Step 3 — Trained test")
    memory = JSONMemoryBackend(LearningMemory, "demo", path=path, model=_model)
    display("Learned coding_patterns", str(await memory.recall("coding_patterns")))
    display("Learned common_pitfalls", str(await memory.recall("common_pitfalls")))

    for problem in TEST_PROBLEMS:
        solution, exec_result, _ = await run_problem(problem, memory, generate_code)
        _show_result("trained", problem, solution, exec_result.passed, exec_result.error)
        before = direct[problem["id"]]
        if not before.passed and exec_result.passed:
            rule(f"{problem['id']}: FAIL → PASS (memory-driven improvement)")
    memory.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
