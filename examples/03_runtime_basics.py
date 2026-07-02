"""Coordinator basics — in-process orchestration with a single worker.

``InMemoryCoordinator`` is the registry + event log + router. A
``LocalWorker`` registers with the coordinator and hosts the dispatcher
task for every thread it runs. Spawning happens via the worker
(``spawn_locally`` — no serialization, the spawnable is passed by
reference); lifecycle events flow out through the coordinator.
"""

import asyncio

from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import CompletedEvent, Event, FailedEvent, StartedEvent


class ResearchPlan(BaseModel):
    subtasks: list[str]


# These single-purpose agents don't message peers, so we disable the default
# coordinator tools (``list_threads`` / ``send_message``; see examples 05/06).
@ai_function(ResearchPlan, coordinator_tools_enabled=False)
def planner(topic: str):
    """Break down the research topic into 2-3 subtasks: {topic}"""


@ai_function(str, coordinator_tools_enabled=False)
def researcher(subtask: str):
    """Research this subtask thoroughly: {subtask}"""


@ai_function(str, coordinator_tools_enabled=False)
def synthesizer(findings: str):
    """Synthesize these findings into a summary:\n\n{findings}"""


def log_event(event: Event) -> None:
    match event:
        case StartedEvent(thread_id=thread_id, thread_name=thread_name):
            print(f"  ▶ {thread_name or thread_id} started")
        case CompletedEvent(thread_id=thread_id, thread_name=thread_name):
            print(f"  ✓ {thread_name or thread_id} completed")
        case FailedEvent(thread_id=thread_id, thread_name=thread_name, error=error):
            print(f"  ✗ {thread_name or thread_id}: {error}")
        case _:
            pass


async def main() -> None:
    coord = InMemoryCoordinator()
    coord.on(log_event)

    worker = LocalWorker(coord)
    await worker.register()

    # Spawn and run the planner.
    planner_h = await worker.spawn_locally(planner, thread_name="planner")
    plan = await planner_h.run(topic="quantum computing applications")

    # Spawn researchers as children of the planner — token usage rolls up.
    researcher_handles = [
        await worker.spawn_locally(
            researcher,
            thread_name=f"researcher-{i}",
            parent_id=planner_h.id,
        )
        for i in range(len(plan.subtasks))
    ]

    # Run all researchers in parallel.
    findings = await asyncio.gather(
        *[h.run(subtask=task) for h, task in zip(researcher_handles, plan.subtasks, strict=True)]
    )

    # Synthesize.
    synth_h = await worker.spawn_locally(synthesizer, thread_name="synthesizer")
    summary = await synth_h.run(findings="\n\n".join(findings))
    print(f"\nSummary: {summary}")

    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
