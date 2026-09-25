"""Two workers on one coordinator — in-process rehearsal for the network layer.

A single ``InMemoryCoordinator`` hosts two ``LocalWorker`` s. Each worker
hosts one thread:

- ``alice`` — a chat responder on worker A.
- ``bob`` — a chat responder on worker B.

We give ``alice`` two tasks:

1. List the threads registered with the coordinator and report what she
   finds. This exercises the coordinator's registry seen across workers.
2. Use ``send_message`` to ask ``bob`` for a fact, then report his
   answer. This exercises cross-worker routing: alice's ``send_message``
   tool call lands on worker-A's dispatcher, reaches out to the
   coordinator, which routes the ``run`` to worker-B's dispatcher,
   which returns the result through the coordinator back to alice.

Both ``list_threads`` and ``send_message`` are injected automatically by
``AIThread``'s default ``coordinator_tools``.
"""

from __future__ import annotations

import asyncio

from example_helpers import models
from example_helpers.utils import display, rule

from ai_functions import ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import WorkerId


@ai_function(model=models.medium, structured_output=False)
def chat(message: str) -> str:
    """{message}"""


async def main() -> None:
    coord = InMemoryCoordinator()
    worker_a = LocalWorker(coord, worker_id=WorkerId("worker-A"))
    worker_b = LocalWorker(coord, worker_id=WorkerId("worker-B"))
    await worker_a.register()
    await worker_b.register()

    # bob lives on worker B. alice lives on worker A. Each thread has
    # no idea which worker hosts it; the coordinator handles routing.
    alice = await worker_a.spawn_locally(chat, thread_name="alice")
    bob = await worker_b.spawn_locally(chat, thread_name="bob")

    display(
        "Threads",
        "\n".join(
            [
                f"alice → {alice.id} (hosted on worker-A)",
                f"bob   → {bob.id} (hosted on worker-B)",
            ]
        ),
        lang="text",
    )

    # alice has list_threads injected as a tool by default.
    rule("Task 1: ask alice to list threads")
    listing = await alice.run(
        "Call the list_threads tool with no arguments. Then report back "
        "in one sentence: which threads are registered, and which of them is you?",
    )
    display("Alice", listing.strip())

    rule("Task 2: ask alice to query bob (mode=wait)")
    relayed = await alice.run(
        "Use send_message to ask bob "
        "'What is the capital of Japan?' with mode='wait'. "
        "Then report bob's reply verbatim.",
    )
    display("Alice", relayed.strip())

    # continue_then_receive — fire-and-continue, then resume. alice dispatches
    # a question to bob and her cycle ends immediately. When bob replies, the
    # coordinator tool schedules a fresh cycle on alice with bob's reply as the
    # user turn. Subscribe to alice's RESULT events so we get the follow-up
    # cycle's output directly off the queue — no event-log rescan.
    rule("Task 3: ask alice to query bob (mode=continue_then_receive)")
    from ai_functions.types import EventKind, ResultEvent

    results: asyncio.Queue[ResultEvent] = asyncio.Queue()

    def _on_result(event: object) -> None:
        assert isinstance(event, ResultEvent)
        results.put_nowait(event)

    with coord.on(_on_result, thread_id=alice.id, kinds=[EventKind.RESULT]):
        cycle1 = await alice.run(
            "Use send_message to ask bob "
            "'In one sentence, what is the tallest mountain on Earth?', "
            "using mode='continue_then_receive'. "
            "Then reply 'dispatched, awaiting bob' and stop.",
        )
        display("Alice (cycle 1)", cycle1.strip())

        # First RESULT is cycle 1 (already fired synchronously above);
        # drop it. The second one is the follow-up driven by bob's reply.
        _ = await results.get()
        followup = await results.get()

    display("Alice (cycle 2, triggered by bob's reply)", followup.payload.strip())

    await worker_a.close()
    await worker_b.close()


if __name__ == "__main__":
    asyncio.run(main())
