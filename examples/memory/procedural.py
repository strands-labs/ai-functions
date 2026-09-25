"""Procedural memory: optimize a Python-code parameter the agent executes.

A ``Procedural`` memory parameter holds reusable Python helper functions. With
``code_execution_mode="local"``, the recalled code is loaded into a sandboxed
Python environment (smolagents' AST-based ``LocalPythonExecutor``); the agent
runs a task using those helpers and returns its answer by calling
``final_answer(...)`` inside executed code. Feedback is then backpropagated and
consolidated to improve the stored code.

The prompt parameter is annotated ``Procedural``: the body receives the raw
code string (``trace`` unwraps the ``ParameterView`` that ``recall`` returns
before the body runs), and the ``Procedural`` marker tells the runtime to
define the code in the execution environment.

This example is also a live check of the code-execution *advertisement*: the
runtime tells the agent which helpers are already defined — by signature and
docstring — so the task prompts never name ``special_greeting`` or spell out how
to call it. The second turn proves both that the seed helper survives the
optimizer's rewrite and that the advertisement is enough for the model to find
and call it unaided (the greeting's code is non-guessable, so a correct answer
can only come from actually executing the recalled helper).
"""

import asyncio
import tempfile
from pathlib import Path

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, Procedural, TextGradOptimizer, ai_function


@ai_function(model=models.medium, code_execution_mode="local")
def run_task(task: str, helper_functions: Procedural) -> str:
    """
    {task}

    Use the Python execution environment. Prefer helpers that already exist over
    writing new logic, and return your answer via final_answer.
    """


class Schema(BaseModel):
    # A non-obvious helper: the model cannot guess its output, so any task that
    # depends on it only succeeds if the recalled code is actually executed. The
    # docstring is what the runtime advertises, so the agent can pick the right
    # helper without the prompt naming it.
    helper_functions: Procedural = Field(
        default=(
            "def special_greeting(name):\n"
            '    """Return a special greeting, use only if requested."""\n'
            "    return f'Zphqr, {name}! (code 7731)'\n"
        ),
        description="Reusable Python helper functions available to the agent.",
    )


async def main(path: str | Path) -> None:
    memory = JSONMemoryBackend(Schema, actor_id="coder-1", path=path, model=models.medium)
    optimizer = TextGradOptimizer(model=models.medium)

    display("Initial Procedural Memory", str(memory))

    # Turn 1: an open-ended task that the seed code does not directly solve.
    # Only `special_greeting` is defined, so the run exercises the environment;
    # feedback then grows the stored code into reusable multi-language helpers.
    result = await run_task.trace(
        task="Greet Alice in Spanish.",
        helper_functions=await memory.recall("helper_functions"),
    )
    display("Turn 1 Result", str(result))

    feedback = "Analyze the execution trace and create and save reusable helper functions."
    display("Feedback", feedback)

    rule("Running optimizer step")
    graph = await optimizer.step(result, feedback, backends=[memory])
    gradients = [f"{p.name}: {p.gradients}" for p in graph.parameters if p.gradients]
    display("Parameter Gradients", "\n".join(gradients), lang="text")

    display("Updated Procedural Memory", str(memory))

    # Turn 2: does the seed helper survive, and can the agent find it unaided?
    # We recall the *updated* code and ask for the special greeting WITHOUT naming
    # the function or its signature. The agent has to read the advertised helper
    # signatures + docstrings to pick special_greeting; a correct, non-guessable
    # answer proves both that it survived consolidation and that the
    # advertisement works.
    special_greeting = await run_task.trace(
        task="Return the caller's special greeting for the name 'Bob', exactly as the helper produces it.",
        helper_functions=await memory.recall("helper_functions"),
    )
    display("Turn 2 Result (expected to contain 'Zphqr, Bob! (code 7731)')", str(special_greeting))
    ok = "Zphqr, Bob! (code 7731)" in str(special_greeting)
    display("special_greeting survived and was found via the advertisement", str(ok), lang="text")

    memory.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
