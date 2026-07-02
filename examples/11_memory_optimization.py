"""Memory + optimizer — self-improving workflow via textual gradients.

Two ``joke_writer`` runs feed an ``email_writer``. Memory parameters start with
generic defaults; ``recall`` pulls them into each run and — given the
coordinator + thread id — records a ``ParameterRecalledEvent`` immediately. After
the run we reconstruct the execution graph, backpropagate natural-language
feedback through it, and consolidate the result back into memory.

The mental model is PyTorch autograd:
- ``recall()``      ≈ reading a learnable weight into the forward pass
- ``backward(fb)``  ≈ ``loss.backward()`` (textual gradients, not numeric)
- ``consolidate()`` ≈ ``optimizer.step()`` (writes improvements into memory)

Because the optimizer rebuilds the graph from the coordinator's event log, the
workflow runs on an explicit coordinator the caller keeps alive (not the
throwaway one a direct ``joke_writer(...)`` call would create and tear down).
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function, build_graph
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@ai_function(str, model=model)
def joke_writer(topic: str, joke_guidelines: str):
    """
    Write a joke about the following topic: "{topic}".
    Use the following guidelines:
    <joke_guidelines>
    {joke_guidelines}
    </joke_guidelines>
    """


@ai_function(str, model=model)
def email_writer(jokes: str, formatting_guidelines: str):
    """
    Write an email to Jane Doe containing the following jokes:
    {jokes}
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
    memory = JSONMemoryBackend(WritingMemory, actor_id="writer-1", path=path, model=model)
    optimizer = TextGradOptimizer(model=model)

    # Explicit coordinator + worker the caller keeps alive, so the event log
    # survives the runs for the optimizer to read back.
    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    print("=== Initial Memory ===")
    print(memory)

    # ── Forward pass ──
    # recall() returns the stored value AND, given coord + thread_id, records a
    # ParameterRecalledEvent into that thread's log right away — even though the
    # thread is spawned on the next line.
    cat_h = await worker.spawn_locally(joke_writer, thread_name="joke_writer")
    cat_guidelines = await memory.recall("joke_guidelines", coordinator=coord, thread_id=cat_h.id)
    cat_joke = await cat_h.run(topic="cats", joke_guidelines=cat_guidelines)
    print(f"\n=== Cat Joke ===\n{cat_joke}")

    prog_h = await worker.spawn_locally(joke_writer, thread_name="joke_writer")
    prog_guidelines = await memory.recall("joke_guidelines", coordinator=coord, thread_id=prog_h.id)
    prog_joke = await prog_h.run(topic="programmers", joke_guidelines=prog_guidelines)
    print(f"\n=== Programmer Joke ===\n{prog_joke}")

    email_h = await worker.spawn_locally(email_writer, thread_name="email_writer")
    fmt_guidelines = await memory.recall("formatting_guidelines", coordinator=coord, thread_id=email_h.id)
    email = await email_h.run(jokes=f"Joke 1: {cat_joke}\n\nJoke 2: {prog_joke}", formatting_guidelines=fmt_guidelines)
    print(f"\n=== Email ===\n{email}")

    # ── Reconstruct the graph ──
    # build_graph rebuilds one node per thread; the cross-thread dataflow
    # (jokes feeding the email) lives only in this script, so we wire it here.
    email_node = build_graph(await coord.get_events(email_h.id), [memory])
    joke1_node = build_graph(await coord.get_events(cat_h.id), [memory])
    joke2_node = build_graph(await coord.get_events(prog_h.id), [memory])
    email_node.child_threads = [joke1_node, joke2_node]
    joke1_node.parent = email_node
    joke2_node.parent = email_node

    print("\n=== Graph ===")
    print(f"Root {email_node.thread_id}: params={[p.name for p in email_node.parameters]}")
    for child in email_node.child_threads:
        print(f"  {child.thread_id}: params={[p.name for p in child.parameters]}")

    # ── Optimize: backward (distribute feedback) then consolidate (write back) ──
    feedback = (
        "Jokes about cats should always be about Siamese cats. "
        "Jokes about programmers should be about coffee. "
        "The email should include a title for each joke."
    )
    print(f"\n=== Feedback ===\n{feedback}")

    print("\nRunning backward pass...")
    optimizer.backward(email_node, feedback)
    for node in (email_node, joke1_node, joke2_node):
        for p in node.parameters:
            if p.gradients:
                print(f"  {node.thread_id} / {p.name}: {p.gradients}")

    print("\nConsolidating memory...")
    optimizer.consolidate(email_node)

    print("\n=== Final Memory ===")
    print(memory)

    memory.close()
    await worker.close()
    print("\nDone — recall() now returns the improved guidelines.")


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
