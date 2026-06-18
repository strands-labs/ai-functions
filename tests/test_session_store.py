"""``FileSessionStore`` persists event logs and resumes threads in place.

A thread's event log is the sole source of truth: dumping it to disk and
re-seeding a fresh spawn with ``seed_events`` + the original ``thread_id``
must restore the thread as if it had never stopped. These tests pin the
store's round-trip, its atomic-overwrite behavior, and a full
save → load → resume cycle through the in-memory coordinator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_functions import FileSessionStore, SessionData, SessionStore, ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import StartedEvent, ThreadId


def test_filesessionstore_satisfies_protocol(tmp_path: Path) -> None:
    """The concrete store is usable where the protocol is expected."""
    # Structural conformance: a static SessionStore binding accepts it, and
    # every protocol method is present and callable.
    store: SessionStore = FileSessionStore(tmp_path)
    assert callable(store.save)
    assert callable(store.load)
    assert callable(store.exists)


def test_exists_false_before_save(tmp_path: Path) -> None:
    """A never-saved session does not exist."""
    store = FileSessionStore(tmp_path)
    assert store.exists("nope") is False


def test_load_missing_raises(tmp_path: Path) -> None:
    """Loading an absent session raises FileNotFoundError."""
    store = FileSessionStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = store.load("nope")


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    """Saved event logs, ids, and metadata come back identically."""
    store = FileSessionStore(tmp_path)
    tid = ThreadId("t-abc")
    events = [StartedEvent(thread_name="agent")]
    store.save(
        "sess1",
        {"agent": events},
        {"agent": tid},
        metadata={"cc_session": "xyz"},
    )

    assert store.exists("sess1") is True
    data = store.load("sess1")
    assert isinstance(data, SessionData)
    assert list(data.threads.keys()) == ["agent"]
    assert len(data.threads["agent"]) == 1
    assert isinstance(data.threads["agent"][0], StartedEvent)
    assert data.thread_ids == {"agent": tid}
    assert data.metadata == {"cc_session": "xyz"}


def test_save_without_metadata(tmp_path: Path) -> None:
    """Omitting metadata loads back as None."""
    store = FileSessionStore(tmp_path)
    store.save("sess2", {"agent": []}, {"agent": ThreadId("t-1")})
    data = store.load("sess2")
    assert data.metadata is None


def test_save_overwrites_atomically(tmp_path: Path) -> None:
    """A second save for the same id replaces the prior state."""
    store = FileSessionStore(tmp_path)
    store.save("sess3", {"agent": []}, {"agent": ThreadId("t-1")})
    store.save(
        "sess3",
        {"agent": [StartedEvent(thread_name="agent")]},
        {"agent": ThreadId("t-2")},
    )
    data = store.load("sess3")
    assert len(data.threads["agent"]) == 1
    assert data.thread_ids == {"agent": ThreadId("t-2")}


def test_multiple_named_threads(tmp_path: Path) -> None:
    """A session can carry more than one named thread log."""
    store = FileSessionStore(tmp_path)
    store.save(
        "sess4",
        {"planner": [StartedEvent(thread_name="planner")], "impl": []},
        {"planner": ThreadId("p-1"), "impl": ThreadId("i-1")},
    )
    data = store.load("sess4")
    assert set(data.threads.keys()) == {"planner", "impl"}
    assert data.thread_ids["planner"] == ThreadId("p-1")
    assert data.thread_ids["impl"] == ThreadId("i-1")


async def test_save_load_resume_through_coordinator(tmp_path: Path) -> None:
    """A persisted log re-seeds a spawn under its original id, restoring history."""

    @ai_function(str)
    def echo(task: str) -> str:
        """{task}"""

    store = FileSessionStore(tmp_path)

    # First run: spawn, capture the event log, persist it.
    coord = InMemoryCoordinator()
    worker = LocalWorker(coord)
    await worker.register()
    handle = await coord.spawn(echo, thread_name="agent")
    original_id = handle.id
    events = await coord.get_events(handle.id)
    store.save("run", {"agent": events}, {"agent": handle.id})
    await worker.close()

    # Second run: resume from disk under the SAME id, pre-seeded with the log.
    data = store.load("run")
    coord2 = InMemoryCoordinator()
    worker2 = LocalWorker(coord2)
    await worker2.register()
    resumed = await coord2.spawn(
        echo,
        thread_name="agent",
        thread_id=data.thread_ids["agent"],
        seed_events=data.threads["agent"] or None,
    )

    # Resumed in place: same id, and the seeded history is present.
    assert resumed.id == original_id
    resumed_events = await coord2.get_events(resumed.id)
    assert len(resumed_events) >= len(data.threads["agent"])
    await worker2.close()
