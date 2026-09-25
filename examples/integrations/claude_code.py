"""Claude Agent thread — run a Claude Agent session, visualize events, print the result.

``ClaudeAgent`` is a ``Spawnable`` that drives a ``claude`` subprocess via
``claude_agent_sdk``. The SDK owns the conversation transcript; ai_functions
observes the SDK's message stream and re-emits each element as a ai_functions
event for observability.

This example:

1. Spawns a ``ClaudeAgent`` thread on a ``LocalWorker``.
2. Subscribes ``print_event`` on the coordinator to pretty-print events as
   they arrive.
3. Sends a task as a prompt and awaits the result.
4. Tears the thread down so the ``claude`` subprocess exits cleanly.
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import ClaudeAgentOptions
from example_helpers.utils import display

from ai_functions.claude_code import ClaudeAgent
from ai_functions.cli import print_event
from ai_functions.runtime import InMemoryCoordinator, LocalWorker


async def main() -> None:
    # Bypass the Claude Agent permission prompt so the example runs non-interactively.
    # Don't do this outside trusted environments.
    options = ClaudeAgentOptions(permission_mode="bypassPermissions")
    template = ClaudeAgent(options=options, name="claude_agent_example")

    coord = InMemoryCoordinator()
    coord.on(print_event)

    worker = await LocalWorker(coord).register()

    handle = await worker.spawn_locally(template, thread_name="claude_agent_example")
    try:
        result = await handle.run(
            "List the Markdown files in the current directory, "
            "and in one short sentence say what the project is about based on README.md.",
        )
        display("Result", str(result))
    finally:
        await handle.terminate_now()
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
