"""Memory list search + consolidation.

Stores cooking tips as a ``list[str]`` memory parameter, retrieves the most
relevant ones with BM25 search, writes a recipe using them, then optimizes:
feedback is backpropagated to the ``tips`` parameter and consolidated (the
whole list is rewritten by the consolidation AI function).

Search requires the optional dependency:
``pip install strands-ai-functions[search]``.
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function, build_graph
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


@ai_function(str, model=model)
def recipe_assistant(dish: str, tips: str):
    """
    Write a short recipe for "{dish}".
    Incorporate the following cooking tips where relevant:
    <tips>
    {tips}
    </tips>
    """


class CookingMemory(BaseModel):
    tips: list[str] = Field(
        default=[
            "Always season pasta water generously with salt",
            "Let meat rest after cooking before slicing",
            "Toast spices in a dry pan to release their aroma",
            "Deglaze the pan with wine or stock for a quick sauce",
            "Add a pinch of sugar to tomato sauces to balance acidity",
            "Pat proteins dry before searing for a better crust",
            "Finish dishes with fresh herbs for brightness",
            "Rest pizza dough overnight in the fridge for better flavor",
        ],
        description="A collection of cooking tips the assistant can search and use.",
    )


async def main(path: str | Path) -> None:
    memory = JSONMemoryBackend(CookingMemory, actor_id="chef-1", path=path, model=model)
    optimizer = TextGradOptimizer(model=model)

    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    print("=== Initial Tips ===")
    print(memory)

    # ── Forward pass: BM25-search the relevant tips, then write a recipe ──
    cook_h = await worker.spawn_locally(recipe_assistant, thread_name="recipe_assistant")
    tips = await memory.search("tips", "pasta sauce tomato", k=5, coordinator=coord, thread_id=cook_h.id)
    print(f"\n=== Retrieved Tips (search 'pasta sauce tomato') ===\n{_bullets(tips)}")

    recipe = await cook_h.run(dish="spaghetti pomodoro", tips=_bullets(tips))
    print(f"\n=== Recipe ===\n{recipe}")

    # ── Optimize: feedback → backward → consolidate ──
    feedback = (
        "The tip about sugar in tomato sauce is wrong — use a splash of balsamic vinegar instead. "
        "Add a tip about using San Marzano tomatoes for the best pomodoro. "
        "The pasta water tip should mention saving a cup of pasta water for the sauce."
    )
    print(f"\n=== Feedback ===\n{feedback}")

    node = build_graph(await coord.get_events(cook_h.id), [memory])

    print("\nRunning backward pass...")
    optimizer.backward(node, feedback)
    for p in node.parameters:
        if p.gradients:
            print(f"  {p.name}: {p.gradients}")

    print("\nConsolidating tips...")
    optimizer.consolidate(node)

    print("\n=== Updated Tips ===")
    print(memory)

    memory.close()
    await worker.close()
    print("\nDone.")


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
