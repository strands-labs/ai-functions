"""Two workers talking through a remote CoordinatorEndpoint (same process).

Port of ``two_workers_local.py`` to the network layer. Instead of an
:class:`InMemoryCoordinator` directly driving two local workers, this
example stands up a :class:`CoordinatorEndpoint` (a WebSocket server)
and connects two clients to it. Each client hosts one worker; each
worker hosts one ``AIFunction`` thread.

Everything runs in a single process here — the wire still carries every
RPC through the loopback WebSocket, exercising the full symmetric
channel machinery (client → endpoint for coordinator.* calls;
endpoint → client for worker.* calls; endpoint → client for event
broadcasts).

Tasks mirror ``two_workers_local.py``:

1. ``alice`` calls ``list_threads`` — sees both threads registered on
   the endpoint, even though she and bob are on different clients
   connecting from different sockets.
2. ``alice`` calls ``send_message(mode='wait')`` — blocks on bob's
   cycle, relays his reply.
3. ``alice`` calls ``send_message(mode='continue_then_receive')`` —
   her first cycle ends immediately; the endpoint schedules a
   follow-up on alice when bob replies. A subscribed ``RESULT`` queue
   on the endpoint's coordinator surfaces the follow-up result.
"""

from __future__ import annotations

import asyncio

from example_helpers import models
from example_helpers.utils import display, rule

from ai_functions import ai_function
from ai_functions.network import CoordinatorClient, CoordinatorEndpoint
from ai_functions.runtime import LocalWorker
from ai_functions.types import EventKind, ResultEvent, WorkerId


@ai_function(model=models.medium, structured_output=False)
def chat(message: str) -> str:
    """{message}"""


async def main() -> None:
    # The endpoint is the authoritative coordinator, served over WebSocket.
    endpoint = CoordinatorEndpoint()
    await endpoint.start(host="127.0.0.1", port=9901)
    display("Endpoint", f"listening at {endpoint.url}", lang="text")

    # Two clients, each hosting one worker.
    client_a = await CoordinatorClient.connect(endpoint.url)
    worker_a = LocalWorker(client_a, worker_id=WorkerId("worker-A"))
    await worker_a.register()

    client_b = await CoordinatorClient.connect(endpoint.url)
    worker_b = LocalWorker(client_b, worker_id=WorkerId("worker-B"))
    await worker_b.register()

    # Spawn one thread on each worker.
    alice = await worker_a.spawn_locally(chat, thread_name="alice")
    bob = await worker_b.spawn_locally(chat, thread_name="bob")

    display(
        "Threads",
        "\n".join(
            [
                f"alice → {alice.id} (hosted on worker-A, client A)",
                f"bob   → {bob.id} (hosted on worker-B, client B)",
            ]
        ),
        lang="text",
    )

    try:
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

        # The second cycle is kicked off on the endpoint side (the
        # coordinator_tools handler schedules a peer.run(notification) on
        # alice). We subscribe to alice's RESULT events on the endpoint's
        # inner coordinator to wait for the follow-up cycle.
        rule("Task 3: ask alice to query bob (mode=continue_then_receive)")
        results: asyncio.Queue[ResultEvent] = asyncio.Queue()

        def _on_result(event: object) -> None:
            assert isinstance(event, ResultEvent)
            results.put_nowait(event)

        # We subscribe on the endpoint's inner coordinator (still in
        # this process) rather than on client_a — the inner coordinator
        # sees every RESULT event from every connected worker, so it's
        # the simplest place to watch.
        with endpoint.coordinator.on(_on_result, thread_id=alice.id, kinds=[EventKind.RESULT]):
            cycle1 = await alice.run(
                "Use send_message to ask bob "
                "'In one sentence, what is the tallest mountain on Earth?', "
                "using mode='continue_then_receive'. "
                "Then reply 'dispatched, awaiting bob' and stop.",
            )
            display("Alice (cycle 1)", cycle1.strip())

            # First RESULT is cycle 1 (already fired above); drop it.
            # The second one is the follow-up driven by bob's reply.
            _ = await asyncio.wait_for(results.get(), timeout=30.0)
            followup = await asyncio.wait_for(results.get(), timeout=30.0)

        display("Alice (cycle 2, triggered by bob's reply)", followup.payload.strip())
    finally:
        # Clean up: close clients (their channels close, workers deregister
        # from the endpoint automatically via the connection teardown),
        # then stop the endpoint.
        await worker_a.close()
        await worker_b.close()
        await client_a.close()
        await client_b.close()
        await endpoint.stop()


if __name__ == "__main__":
    asyncio.run(main())
