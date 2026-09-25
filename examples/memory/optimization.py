"""Memory + optimizer — self-improving workflow via textual gradients.

Two ``joke_writer`` runs feed an ``email_writer``. Memory parameters start with
generic defaults; ``recall`` pulls them into each run, and ``trace`` records
which recalled parameters and prior results each run consumed. After the runs,
``optimizer.step`` reconstructs the execution graph from the event logs,
backpropagates natural-language feedback through it, and consolidates the
result back into memory.

The mental model is PyTorch autograd:
- ``recall()``   ≈ reading a learnable weight into the forward pass
- ``trace()``    ≈ a forward pass that remembers its inputs
- ``step(fb)``   ≈ ``loss.backward()`` + ``optimizer.step()`` in one call

Passing a ``Result`` (``cat_joke``) or a recalled ``ParameterView`` directly
as an argument is what creates the graph edge. Interpolating them into an
f-string still computes the right value, but drops the edge.
"""

import asyncio
import tempfile
from pathlib import Path

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function


# The prompt function declares its parameters as the plain types it receives:
# trace() unwraps any recalled parameter / traced result to its value before
# the body runs, so the body only ever sees a ``str`` here.
@ai_function(model=models.small)
def joke_writer(topic: str, joke_guidelines: str) -> str:
    """
    Write a joke about the following topic: "{topic}".
    Use the following guidelines:
    <joke_guidelines>
    {joke_guidelines}
    </joke_guidelines>
    """


@ai_function(model=models.small)
def email_writer(joke_1: str, joke_2: str, formatting_guidelines: str) -> str:
    """
    Write an email to Jane Doe containing the following jokes:
    Joke 1: {joke_1}
    Joke 2: {joke_2}
    Use the following email formatting guidelines:
    <formatting_guidelines>
    {formatting_guidelines}
    </formatting_guidelines>
    """


class WritingMemory(BaseModel):
    joke_guidelines: str = Field(
        "No specific guidelines yet.",
        description="Guidelines to write a good joke",
    )
    formatting_guidelines: str = Field(
        "No specific guidelines yet.",
        description="Guidelines for the layout and typography of the email.",
    )


async def main(path: str | Path) -> None:
    memory = JSONMemoryBackend(WritingMemory, actor_id="writer-1", path=path, model=models.small)
    optimizer = TextGradOptimizer(model=models.small)

    display("Initial Memory", str(memory))

    # Forward pass: trace() runs the function like a call, but returns a Result
    # that remembers the recalled parameters and Results passed to it.
    cat_joke = await joke_writer.trace(
        topic="cats",
        joke_guidelines=await memory.recall("joke_guidelines"),
    )
    display("Cat Joke", str(cat_joke))

    prog_joke = await joke_writer.trace(
        topic="programmers",
        joke_guidelines=await memory.recall("joke_guidelines"),
    )
    display("Programmer Joke", str(prog_joke))

    # Passing the Results directly (not f-strings of them) wires the edges.
    email = await email_writer.trace(
        joke_1=cat_joke,
        joke_2=prog_joke,
        formatting_guidelines=await memory.recall("formatting_guidelines"),
    )
    display("Email", str(email))

    # Optimize: build graph + backward + consolidate, in a single call.
    feedback = (
        "Jokes about cats should always be about Siamese cats. "
        "Jokes about programmers should be about coffee. "
        "The email should include a title for each joke."
    )
    display("Feedback", feedback)

    rule("Running optimizer step")
    graph = await optimizer.step(email, feedback, backends=[memory])

    # The returned graph exposes the gradients backpropagated to each parameter —
    # the root's own parameters (formatting_guidelines) and each child's.
    lines = [f"Root {graph.thread_id}: params={[p.name for p in graph.parameters]}"]
    lines.extend(f"  {p.name}: {p.gradients}" for p in graph.parameters if p.gradients)
    for child in graph.child_threads:
        lines.append(f"  {child.thread_id}: params={[p.name for p in child.parameters]}")
        lines.extend(f"    {p.name}: {p.gradients}" for p in child.parameters if p.gradients)
    display("Optimizer Graph", "\n".join(lines), lang="text")

    display("Final Memory", str(memory))

    memory.close()
    # recall() now returns the improved guidelines on every future run.


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
