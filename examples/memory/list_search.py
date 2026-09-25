"""Memory list search + consolidation.

Stores cooking tips as a ``list[str]`` memory parameter, retrieves the most
relevant ones with BM25 search, writes a recipe using them, then optimizes:
feedback is backpropagated to the ``tips`` parameter and consolidated by an
agentic memory manager that edits entries in place (add/update/delete by
stable entry id) — and, because the search's ``{entry_id: value}`` results
ride along in the recall event, consolidation targets exactly the entries
this run retrieved instead of the whole list.

The prompt function declares ``tips`` as the plain ``list[str]`` it receives:
``trace`` unwraps the ``ParameterView`` that ``search`` returns before the body
runs, so the body works with plain data directly.
"""

import asyncio
import tempfile
from pathlib import Path

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


@ai_function(model=models.small)
def recipe_assistant(dish: str, tips: list[str]) -> str:
    """Build the recipe prompt from the retrieved tips."""
    return (
        f'Write a short recipe for "{dish}".\n'
        "Incorporate the following cooking tips where relevant:\n"
        f"<tips>\n{_bullets(tips)}\n</tips>"
    )


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
    memory = JSONMemoryBackend(CookingMemory, actor_id="chef-1", path=path, model=models.small)
    optimizer = TextGradOptimizer(model=models.small)

    display("Initial Tips", str(memory))

    # Forward pass: BM25-search the relevant tips, then write a recipe. search()
    # returns a ParameterView; passing it to trace() (rather than an f-string of
    # it) is what records the tips -> recipe graph edge.
    tips = await memory.search("tips", "pasta sauce tomato", k=5)
    display("Retrieved Tips (search 'pasta sauce tomato')", _bullets(tips.value))

    recipe = await recipe_assistant.trace(dish="spaghetti pomodoro", tips=tips)
    display("Recipe", str(recipe))

    # Optimize: build graph + backward + consolidate, in one call.
    feedback = (
        "The tip about sugar in tomato sauce is wrong — use a splash of balsamic vinegar instead. "
        "Add a tip about using San Marzano tomatoes for the best pomodoro. "
        "The pasta water tip should mention saving a cup of pasta water for the sauce."
    )
    display("Feedback", feedback)

    rule("Running optimizer step")
    graph = await optimizer.step(recipe, feedback, backends=[memory])
    gradients = [f"{p.name}: {p.gradients}" for p in graph.parameters if p.gradients]
    display("Parameter Gradients", "\n".join(gradients), lang="text")

    display("Updated Tips", str(memory))

    memory.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        asyncio.run(main(f.name))
