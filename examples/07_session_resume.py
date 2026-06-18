"""Two-thread system with save / resume via ``FileSessionStore``.

A session here holds *two* independent agents that build up context over
several turns:

- ``researcher`` — accumulates facts about a topic, one per turn.
- ``writer`` — keeps a running summary of the facts so far.

Each agent is a separate thread with its own event log, run in isolation
on its own coordinator. After the run, both logs are persisted to a
session directory. Re-running with ``--resume`` reloads both logs and
re-spawns both threads *under their original ids*, pre-seeded with their
history — so a follow-up turn sees everything from the first run, for
both agents, with no replay on the user's part.

This is the pattern from ``ai_functions.session``: the event log is the sole
source of truth, so ``save`` = dump ``coordinator.get_events`` and
``resume`` = ``spawn(seed_events=..., thread_id=...)``. The persisted log
is the whole session — nothing the agents did is lost across the restart.

A single run does both halves of the cycle in one process — a fresh turn
on each thread, then a resumed follow-up turn that reloads both logs from
disk — so the save/resume round-trip is visible end to end without
re-invoking the script::

    python examples/09_session_resume.py --session-dir /tmp/ai_functions-demo

The work is factored into :func:`run_session` so the same flow runs both
from the CLI (against a real model) and from the integration test
(``tests/test_example_session_resume.py``) against a scripted model — the
test passes ``models=`` to drive the agents deterministically.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ai_functions import FileSessionStore, SessionStore, ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import Event

# Thread names — also the keys under which each log is persisted.
RESEARCHER = "researcher"
WRITER = "writer"


@ai_function(str, structured_output=False)
def researcher(prompt: str) -> str:
    """{prompt}"""


@ai_function(str, structured_output=False)
def writer(prompt: str) -> str:
    """{prompt}"""


# A model per thread, injected for tests. ``object`` because a real run
# leaves these ``None`` (the default model is used) and the test passes a
# ``strands.Model`` (``ScriptedModel``) — the only common supertype here.
ThreadModels = dict[str, object]


async def _run_one_thread(
    name: str,
    template: object,
    prompt: str,
    *,
    seed: list[Event] | None,
    thread_id: object,
    model: object,
) -> tuple[str, object, list[Event]]:
    """Spawn one thread on its own coordinator, run a cycle, return its log.

    Each thread gets a private :class:`InMemoryCoordinator` and worker so
    the two agents are fully isolated — they cannot discover or message
    one another. (Every ``AIThread`` is given ``list_threads`` /
    ``send_message`` tools by default; on a shared coordinator an agent can
    message a peer — or itself — and a synchronous self-message deadlocks.
    A coordinator per thread sidesteps that and matches the "two
    independent contexts" intent of this example.)

    Args:
        name: Thread name (also its key in the persisted session).
        template: The ``ai_function`` spawnable for this thread.
        prompt: The prompt to run this turn.
        seed: Prior event log to pre-seed on resume, or ``None`` for fresh.
        thread_id: Original id to resume under, or ``None`` to mint one.
        model: Optional ``strands.Model`` to inject; ``None`` uses default.

    Returns:
        ``(reply, thread_id, event_log)`` for this thread after the cycle.
    """
    coordinator = InMemoryCoordinator()
    worker = LocalWorker(coordinator)
    await worker.register()

    spawnable = template.replace(model=model) if model is not None else template  # type: ignore[attr-defined]
    handle = await coordinator.spawn(
        spawnable,
        thread_name=name,
        thread_id=thread_id,  # type: ignore[arg-type]  # ThreadId | None
        seed_events=seed or None,
    )
    try:
        reply = await handle.run(prompt)
    finally:
        events = await coordinator.get_events(handle.id)
        thread_id = handle.id
        await worker.close()
    return (reply or "").strip(), thread_id, events


async def run_session(
    store: SessionStore,
    session_id: str,
    prompts: dict[str, str],
    *,
    resume: bool,
    models: ThreadModels | None = None,
) -> dict[str, str]:
    """Run one turn of the two-thread system, persisting both logs.

    Runs ``researcher`` and ``writer`` — fresh, or (when ``resume`` is
    set) pre-seeded from the persisted session so both keep their full
    history — then saves both event logs back to ``store`` under the same
    ``session_id``. Each thread runs in isolation (see
    :func:`_run_one_thread`).

    Args:
        store: Where the session's event logs are persisted.
        session_id: Session identifier; the same id round-trips a resume.
        prompts: Mapping of thread name (``"researcher"`` / ``"writer"``)
            to the prompt for this turn.
        resume: When ``True`` and the session exists, both threads are
            re-spawned under their saved ids, seeded with their logs.
        models: Optional mapping of thread name to a ``strands.Model`` to
            inject (used by the integration test for determinism). When
            ``None``, each agent uses its default model.

    Returns:
        Mapping of thread name to that thread's reply this turn.

    Ensures:
        Both threads' event logs are saved before returning.
    """
    models = models or {}

    # On resume, pull both logs + original ids from disk; else start clean.
    seed: dict[str, list[Event]] = {}
    ids: dict[str, object] = {}
    if resume and store.exists(session_id):
        data = store.load(session_id)
        seed = data.threads
        ids = dict(data.thread_ids)

    templates = {RESEARCHER: researcher, WRITER: writer}
    replies: dict[str, str] = {}
    threads: dict[str, list[Event]] = {}
    thread_ids: dict[str, object] = {}
    for name, template in templates.items():
        reply, tid, events = await _run_one_thread(
            name,
            template,
            prompts[name],
            seed=seed.get(name),
            thread_id=ids.get(name),
            model=models.get(name),
        )
        replies[name] = reply
        threads[name] = events
        thread_ids[name] = tid

    # Persist both logs — the event log is the whole session.
    store.save(session_id, threads, thread_ids)  # type: ignore[arg-type]  # dict values are ThreadId
    return replies


FRESH_PROMPTS = {
    RESEARCHER: "Share one interesting fact about octopuses, in one sentence.",
    WRITER: "Begin a one-line running summary titled 'Octopus facts'.",
}
RESUME_PROMPTS = {
    RESEARCHER: "Add one more interesting fact about octopuses. Keep your prior facts in mind.",
    WRITER: "Add one short line to your running summary.",
}


def _print_replies(replies: dict[str, str]) -> None:
    """Print each thread's reply for one turn."""
    for name, reply in replies.items():
        print(f"{name}: {reply}")


async def main() -> None:
    """CLI entry point: run a fresh turn then a resumed turn, in one process."""
    parser = argparse.ArgumentParser(description="Two-thread save/resume demo.")
    _ = parser.add_argument("--session-dir", required=True, type=Path)
    _ = parser.add_argument("--session-id", default="demo")
    args = parser.parse_args()

    store = FileSessionStore(args.session_dir)

    # ── Phase 1: fresh session. Both threads start clean and are saved. ──
    print("── fresh session ──")
    fresh = await run_session(store, args.session_id, FRESH_PROMPTS, resume=False)
    _print_replies(fresh)

    # ── Phase 2: resume from disk. Both logs are reloaded and each thread
    # re-spawns under its original id, so the follow-up turn sees phase 1. ──
    print("\n── resuming session (both threads keep their history) ──")
    resumed = await run_session(store, args.session_id, RESUME_PROMPTS, resume=True)
    _print_replies(resumed)

    print(f"\nsession saved to {args.session_dir / args.session_id}")


if __name__ == "__main__":
    asyncio.run(main())
