"""Integration test for ``examples/08_session_resume.py``.

Exercises the two-thread save/resume flow end to end against deterministic
``ScriptedModel`` s (no real model calls): a fresh "process" runs one turn
on each of two threads and persists both logs; a second "process" — fresh
coordinator, worker, and store instance, as if relaunched — resumes from
disk and runs a follow-up turn. The test asserts that both threads come
back under their original ids carrying their full prior history, which is
the property the example exists to demonstrate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_functions import FileSessionStore
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import EventKind


def _load_example() -> object:
    """Import ``examples/08_session_resume.py`` as a module."""
    path = Path(__file__).resolve().parent.parent / "examples" / "08_session_resume.py"
    spec = importlib.util.spec_from_file_location("_example_08_session_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example() -> object:
    return _load_example()


def _models(researcher_text: str, writer_text: str) -> dict[str, object]:
    """One single-turn ScriptedModel per thread for one ``run_session`` call."""
    return {
        "researcher": ScriptedModel([Turn(text=researcher_text)]),
        "writer": ScriptedModel([Turn(text=writer_text)]),
    }


async def test_two_thread_fresh_then_resume(example: object, tmp_path: Path) -> None:
    """Fresh run persists both logs; resume restores both with full history."""
    run_session = example.run_session  # type: ignore[attr-defined]
    store = FileSessionStore(tmp_path)
    session_id = "demo"

    # ── Run 1: fresh session, one turn per thread. ──
    replies1 = await run_session(
        store,
        session_id,
        {"researcher": "fact one", "writer": "summary one"},
        resume=False,
        models=_models("octopuses have three hearts", "summary: three hearts"),
    )
    assert replies1["researcher"] == "octopuses have three hearts"
    assert replies1["writer"] == "summary: three hearts"

    # Capture the original thread ids so we can prove resume-in-place.
    data_after_first = store.load(session_id)
    researcher_id = data_after_first.thread_ids["researcher"]
    writer_id = data_after_first.thread_ids["writer"]

    # ── Run 2: resume=True. run_session spins up fresh coordinators
    # internally, modelling a relaunched process. ──
    replies2 = await run_session(
        store,
        session_id,
        {"researcher": "fact two", "writer": "summary two"},
        resume=True,
        models=_models("octopuses have blue blood", "summary: three hearts, blue blood"),
    )
    assert replies2["researcher"] == "octopuses have blue blood"
    assert replies2["writer"] == "summary: three hearts, blue blood"

    # ── The point: both threads resumed in place, with prior history. ──
    final = store.load(session_id)
    # Same ids across the relaunch — resume-in-place, not a new thread.
    assert final.thread_ids["researcher"] == researcher_id
    assert final.thread_ids["writer"] == writer_id

    # Each thread's second-run log contains TWO user turns (turn 1 was
    # seeded from disk, turn 2 was the resume prompt) — proof the context
    # carried across the "process boundary" for BOTH threads.
    for name in ("researcher", "writer"):
        kinds = [e.kind for e in final.threads[name]]
        assert kinds.count(EventKind.MESSAGE_USER) >= 2, (
            f"{name}: expected the seeded turn + the resume turn, got {kinds}"
        )


async def test_resume_without_prior_session_starts_fresh(example: object, tmp_path: Path) -> None:
    """``resume=True`` with no saved session behaves like a fresh start."""
    run_session = example.run_session  # type: ignore[attr-defined]
    store = FileSessionStore(tmp_path)

    replies = await run_session(
        store,
        "missing",
        {"researcher": "r", "writer": "w"},
        resume=True,  # nothing on disk yet → no seed, no error
        models=_models("r-reply", "w-reply"),
    )
    assert replies == {"researcher": "r-reply", "writer": "w-reply"}
    # Exactly one user turn each — no seeded history.
    data = store.load("missing")
    for name in ("researcher", "writer"):
        kinds = [e.kind for e in data.threads[name]]
        assert kinds.count(EventKind.MESSAGE_USER) == 1
