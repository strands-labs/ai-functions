"""Procedural memory: optimize a Python-code parameter the agent executes.

A ``Procedural`` memory parameter holds reusable Python helper functions. With
``code_execution_mode="local"``, the recalled code is loaded into a sandboxed
Python environment (smolagents' AST-based ``LocalPythonExecutor``); the agent
runs a task using those helpers and returns its answer by calling
``final_answer(...)`` inside executed code. Feedback is then backpropagated and
consolidated to improve the stored code.

Requires the optional dependency:
``pip install strands-ai-functions[procedural]``.
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, Procedural, TextGradOptimizer, ai_function, build_graph
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@ai_function(str, model=model, code_execution_mode="local")
def run_task(helper_functions: Procedural):
    """
    Call the `secret_greeting(name)` helper function (already defined in the
    python execution environment) with name "Alice", and return exactly what it
    returns via final_answer. Do not invent the greeting yourself — you must
    call the helper to obtain it.
    """


class Schema(BaseModel):
    # A non-obvious helper: the model cannot guess its output, so the task only
    # succeeds if the recalled code is actually executed (not improvised).
    helper_functions: Procedural = Field(
        default="def secret_greeting(name):\n    return f'Zphqr, {name}! (code 7731)'\n",
        description="Reusable Python helper functions available to the agent.",
    )


async def main(path: str | Path) -> None:
    memory = JSONMemoryBackend(Schema, actor_id="coder-1", path=path, model=model)
    optimizer = TextGradOptimizer(model=model)

    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    print("=== Initial Procedural Memory ===")
    print(memory)

    # ── Forward pass: recall the code, run a task that uses it ──
    task_h = await worker.spawn_locally(run_task, thread_name="run_task")
    helpers = await memory.recall("helper_functions", coordinator=coord, thread_id=task_h.id)
    result = await task_h.run(helper_functions=helpers)
    print(f"\n=== Result ===\n{result}")

    # ── Optimize: improve the stored helper code ──
    feedback = (
        "Analyze the execution trace and create reusable, well-named helper "
        "functions for greeting in multiple languages."
    )
    print(f"\n=== Feedback ===\n{feedback}")

    node = build_graph(await coord.get_events(task_h.id), [memory])

    print("\nRunning backward pass...")
    optimizer.backward(node, feedback)
    for p in node.parameters:
        if p.gradients:
            print(f"  {p.name}: {p.gradients}")

    print("\nConsolidating code...")
    optimizer.consolidate(node)

    print("\n=== Final Procedural Memory ===")
    print(memory)

    memory.close()
    await worker.close()
    print("\nDone.")


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
