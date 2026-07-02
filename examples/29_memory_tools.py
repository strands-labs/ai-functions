"""Memory as tools — give an agent direct read/write access to its memory.

``memory.tool_provider(*names)`` generates uniquely-named Strands tools from the
schema (``recall_<name>``, ``query_<name>``, ``search_<name>`` for lists,
``save_<name>`` / ``delete_<name>`` for scalars). Attaching them via
``.replace(tools=[...])`` lets a travel assistant look up the user's preferences
and past trips — and update them — during a cycle. These tools are pure
fetches/writes: they do not feed the optimization graph (that path uses
``recall(coordinator, thread_id)`` + ``build_graph``, see examples 11-13).
"""

import asyncio
import tempfile

from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.memory.json_backend import JSONMemoryBackend


def display(title: str, body: object) -> None:
    """Print a titled section."""
    print(f"\n{'=' * 8} {title} {'=' * 8}\n{body}")


class TravelMemory(BaseModel):
    preferences: str = Field(
        default="Prefers warm destinations. Likes hiking and local food. Budget-conscious.",
        description="Travel preferences and style of the user",
    )
    visited: list[str] = Field(
        default=[
            "Tokyo, Japan - loved the street food and temples",
            "Barcelona, Spain - enjoyed the architecture and beaches",
            "Banff, Canada - amazing hiking trails",
            "Marrakech, Morocco - great markets and riads",
            "Reykjavik, Iceland - stunning landscapes but too cold",
        ],
        description="Places the user has visited with brief notes",
    )


@ai_function(str)
def travel_assistant(request: str):
    """You are a travel planning assistant with access to the user's travel memory.
    Use the available tools to look up their preferences and past trips before
    making recommendations. You can also update their memory when they share
    new information.

    User request: {request}
    """


async def main(path: str):
    memory = JSONMemoryBackend(TravelMemory, "traveler-1", path=path)
    display("Initial Memory", memory.dump())

    # Give the agent tools for both parameters.
    tools = memory.tool_provider("preferences", "visited")
    assistant = travel_assistant.replace(tools=[tools])

    # Ask for a recommendation: the agent should search/recall memory.
    response = await assistant("I want to go somewhere new next month. Suggest a destination and explain why.")
    display("Recommendation", response)

    # Ask the agent to update memory.
    response = await assistant(
        "I just got back from Lisbon, Portugal — loved the pastéis de nata and the tram rides. "
        "Also, I've decided I want to try more European cities. Please update my memory."
    )
    display("Update Response", response)

    display("Updated Memory", memory.dump())

    # One more query to verify the agent uses the updated memory.
    response = await assistant("Based on what you know about me, what European city should I visit next?")
    display("Follow-up Recommendation", response)

    memory.close()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        asyncio.run(main(f.name))
