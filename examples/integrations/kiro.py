"""Kiro Agent thread — run a Kiro session, visualize events, print the result.

``KiroAgent`` is a ``Spawnable`` that drives a ``kiro-cli acp`` subprocess over
the Agent Client Protocol (ACP). The ACP agent owns the conversation transcript;
ai_functions observes the ``session/update`` stream and re-emits each element as
a ai_functions event for observability.

This example:

1. Spawns a ``KiroAgent`` thread on a ``LocalWorker``.
2. Subscribes ``print_event`` on the coordinator to pretty-print events as
   they arrive.
3. Sends a task as a prompt and awaits the result.
4. Tears the thread down so the ``kiro-cli acp`` subprocess exits cleanly.

Prerequisites:
    - The ``kiro`` extra: ``pip install 'strands-ai-functions[kiro]'``.
    - The ``kiro-cli`` binary on your PATH (or pass ``KiroAgent(executable=...)``
      with its full path). Install it from
      https://kiro.dev/docs/cli/installation/.

Run ``uv run integrations/kiro.py`` from ``examples/``. The agent's file tools
see that directory (the ACP session uses the process working directory by default).
"""

from __future__ import annotations

import asyncio

from example_helpers.utils import display

from ai_functions.cli import print_event
from ai_functions.kiro import KiroAgent
from ai_functions.runtime import InMemoryCoordinator, LocalWorker


async def main() -> None:
    # ``auto_approve`` lets the ACP agent run its tools non-interactively.
    # Don't bypass approvals outside trusted environments.
    template = KiroAgent(auto_approve=True, name="kiro_agent_example")

    coord = InMemoryCoordinator()
    coord.on(print_event)

    worker = LocalWorker(coord)
    await worker.register()

    handle = await worker.spawn_locally(template, thread_name="kiro_agent_example")
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
