"""Multi-turn conversation — same thread, accumulated history.

Use handle.run() to drive each turn. The thread keeps its conversation
history across calls. Each run() is a complete cycle that returns a
typed result.
"""

import asyncio

from example_helpers import models
from example_helpers.utils import display

from ai_functions import ai_function


@ai_function(model=models.medium)
def assistant(message: str) -> str:
    """{message}"""


async def main():
    # spawn() creates a handle with its own session.
    handle = await assistant.spawn()

    r1 = await handle.run(message="What is the capital of France?")
    display("Turn 1", str(r1))

    # The follow-up sees the full conversation history from turn 1.
    r2 = await handle.run(message="What about Germany?")
    display("Turn 2", str(r2))

    # notify() injects context without starting a cycle.
    await handle.notify("The user prefers single-word answers.")

    # The injected message is visible on the next run.
    r3 = await handle.run(message="And Spain?")
    display("Turn 3", str(r3))

    await handle.terminate_now()


if __name__ == "__main__":
    asyncio.run(main())
