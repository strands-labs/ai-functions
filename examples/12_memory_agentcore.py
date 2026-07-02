"""Memory + optimizer with the AWS Bedrock AgentCore backend.

Same self-improving joke -> email flow as ``11_memory_optimization.py``, but
parameters live in an AgentCore Memory resource instead of a JSON file. The
optimizer interface is identical — only the backend changes.

AgentCore consolidation differs from the JSON backend: rather than running an
explicit merge AI function, ``consolidate`` appends feedback as conversation
turns, which AgentCore's semantic-memory strategy extracts into long-term
memory **asynchronously**. Short-term memory reflects the turns immediately;
long-term records appear after a delay.

Requires AWS credentials with Bedrock AgentCore permissions and the optional
dependency:  ``pip install strands-ai-functions[agentcore]``.
Set AWS_REGION if you are not using us-east-1.
"""

import asyncio
import os
import uuid

from pydantic import BaseModel, Field

from ai_functions import AgentCoreMemoryBackend, TextGradOptimizer, ai_function, build_graph
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

model = None  # use the default model
region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


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


async def main() -> None:
    # Get-or-create an AgentCore memory resource by name; namespaced per actor.
    memory = AgentCoreMemoryBackend(
        WritingMemory,
        actor_id=f"writer-{uuid.uuid4()}",
        memory_name="ai_function_backprop_test",
        region_name=region,
        model=model,
    )
    optimizer = TextGradOptimizer(model=model)

    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    print("=== Initial Memory ===")
    print(memory)

    # ── Forward pass ──
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

    # ── Reconstruct the graph (cross-thread edges wired by the caller) ──
    email_node = build_graph(await coord.get_events(email_h.id), [memory])
    joke1_node = build_graph(await coord.get_events(cat_h.id), [memory])
    joke2_node = build_graph(await coord.get_events(prog_h.id), [memory])
    email_node.child_threads = [joke1_node, joke2_node]
    joke1_node.parent = email_node
    joke2_node.parent = email_node

    # ── Optimize ──
    feedback = (
        "Jokes about cats should always be about Siamese cats. "
        "Jokes about programmers should be about coffee. "
        "The email should include a title for each joke."
    )
    print(f"\n=== Feedback ===\n{feedback}")

    print("\nRunning backward pass...")
    optimizer.backward(email_node, feedback)

    print("Consolidating memory (appends turns; LTM extraction is async)...")
    optimizer.consolidate(email_node)

    # Short-term memory reflects the new turns immediately.
    print("\n=== Current Memory (STM) ===")
    print(memory)

    # Tidy up the test memory resource. Drop these two lines to inspect LTM
    # in the AgentCore console once semantic extraction has run.
    memory.delete_all(wait=False)
    memory.close()
    await worker.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
