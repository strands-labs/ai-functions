"""Persist and restore thread sessions between processes.

A thread's event log is the sole source of truth for its state: the
conversation history, tool calls, results, and summarization boundaries
are all recorded as :class:`~ai_functions.types.Event` s, and a thread rebuilds
its live state by replaying them. Persisting that log to disk and
restoring it via :meth:`Coordinator.spawn` (with ``seed_events`` and the
original ``thread_id``) is therefore sufficient to resume a thread as if
no interruption had occurred — no separate state snapshot is required.

This module provides that persistence layer:

- :class:`SessionStore` — the protocol for "where the bytes live".
- :class:`FileSessionStore` — a file-backed implementation that writes
  one directory per session, with atomic writes so a crash mid-save
  never corrupts previously persisted state.
- :class:`SessionData` — the value returned by :meth:`SessionStore.load`.

A session may hold more than one thread (e.g. cooperating agents), so
both the store and :class:`SessionData` key event logs by a caller-chosen
thread *name*, alongside the original :class:`~ai_functions.types.ThreadId` so
the thread can be resumed in place.

Example::

    from ai_functions.session import FileSessionStore

    store = FileSessionStore(Path("/sessions"))

    # Resume if we have prior state, else start fresh.
    if store.exists(session_id):
        data = store.load(session_id)
        seed = data.threads.get("agent")
        tid = data.thread_ids.get("agent")
    else:
        seed, tid = None, None

    handle = await coordinator.spawn(
        agent,
        thread_name="agent",
        thread_id=tid,
        seed_events=seed,
    )
    try:
        result = await handle.run(prompt)
    finally:
        events = await coordinator.get_events(handle.id)
        store.save(session_id, {"agent": events}, {"agent": handle.id})

Backend state outside the event log:
    A pure event-log replay fully reconstructs threads whose state lives
    in the log (e.g. ``ai_function`` agents). A backend that keeps state
    *outside* the log — for instance a thread driving an external
    subprocess session — must persist its resume token through
    :attr:`SessionData.metadata`; the event log alone is not sufficient
    to resume it.

Relationship to ``strands.session``:
    The underlying ``strands`` library ships its own persistence layer
    (``SessionManager`` with ``FileSessionManager`` / ``S3SessionManager``
    implementations). That layer persists a single ``strands`` *agent*'s
    conversation and internal state, and is driven by agent lifecycle
    hooks. This module operates one level down, on the **event log** that
    is this library's source of truth, and it persists a *whole session*
    of one or more named, cooperating threads — keyed by thread name and
    resumable in place via ``thread_id``. The two are therefore not
    interchangeable: a ``strands`` ``SessionManager`` cannot reconstruct
    an :class:`~ai_functions.runtime.coordinator.InMemoryCoordinator`'s
    event log, and this store does not capture agent-internal state that
    never reaches the log.

    A ``strands`` ``SessionManager`` may still be supplied per agent (via
    ``agent_kwargs``); it is passed straight through to the underlying
    ``strands`` ``Agent`` and is independent of this store. If both are in
    use, they persist different things to different places — treat any
    such agent as a "backend state outside the event log" case (see
    above) and record what is needed to reconcile them in
    :attr:`SessionData.metadata`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from pydantic import TypeAdapter

from .types import Event, ThreadId

_EventListAdapter: TypeAdapter[list[Event]] = TypeAdapter(list[Event])


@final
@dataclass
class SessionData:
    """Data restored from a persisted session.

    Attributes:
        threads: Mapping of thread name to its full event log, oldest
            event first. Pass an entry to ``Coordinator.spawn`` as
            ``seed_events`` to resume that thread.
        thread_ids: Mapping of thread name to the
            :class:`~ai_functions.types.ThreadId` the thread had when it was
            saved. Pass an entry to ``Coordinator.spawn`` as
            ``thread_id`` to resume the thread under its original id.
        metadata: Optional free-form metadata persisted alongside the
            session, for backend state that does not live in the event
            log (e.g. an external subprocess session id). ``None`` when
            no metadata was saved.
    """

    threads: dict[str, list[Event]]
    thread_ids: dict[str, ThreadId]
    metadata: dict[str, object] | None = None


class SessionStore(Protocol):
    """Persist and restore thread event logs between invocations.

    Implementations must write atomically: a crash mid-save must not
    corrupt previously persisted state. The recommended strategy is
    write-to-temp then atomic rename.
    """

    def save(
        self,
        session_id: str,
        threads: dict[str, list[Event]],
        thread_ids: dict[str, ThreadId],
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Persist event logs for all threads in a session.

        Args:
            session_id: Opaque session identifier.
            threads: Mapping of thread name to its full event log.
            thread_ids: Mapping of thread name to its thread id.
            metadata: Optional metadata to persist alongside the session.

        Ensures:
            A subsequent ``load(session_id)`` returns the same data.
            Writes are atomic: partial failures leave prior state intact.
        """
        ...

    def load(self, session_id: str) -> SessionData:
        """Restore a previously persisted session.

        Args:
            session_id: Session to restore.

        Returns:
            The restored session data.

        Raises:
            FileNotFoundError: No session with this id exists.
        """
        ...

    def exists(self, session_id: str) -> bool:
        """Check whether a session has been persisted.

        Args:
            session_id: Session to check.

        Returns:
            ``True`` iff ``load(session_id)`` would succeed.
        """
        ...


@final
class FileSessionStore:
    """File-backed implementation of :class:`SessionStore`.

    Layout::

        base_dir/
          <session_id>/
            session.json          # manifest: thread names, ids, metadata
            <name>.events.json    # one event log per thread name

    Writes use temp-file + atomic rename to prevent corruption on kill.

    Args:
        base_dir: Directory under which session subdirectories are
            created. Created on first ``save`` if it does not exist.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir: Path = base_dir

    def save(
        self,
        session_id: str,
        threads: dict[str, list[Event]],
        thread_ids: dict[str, ThreadId],
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Persist event logs for all threads in a session.

        Args:
            session_id: Opaque session identifier.
            threads: Mapping of thread name to its full event log.
            thread_ids: Mapping of thread name to its thread id.
            metadata: Optional metadata to persist alongside the session.

        Ensures:
            A subsequent ``load(session_id)`` returns the same data.
            Writes are atomic: partial failures leave prior state intact.
        """
        session_dir = self._base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write each thread's events atomically.
        for name, events in threads.items():
            target = session_dir / f"{name}.events.json"
            data = _EventListAdapter.dump_json(events)
            _atomic_write(target, data)

        # Write the manifest.
        manifest: dict[str, object] = {
            "thread_names": list(threads.keys()),
            "thread_ids": {name: str(tid) for name, tid in thread_ids.items()},
        }
        if metadata:
            manifest["metadata"] = metadata
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        _atomic_write(session_dir / "session.json", manifest_bytes)

    def load(self, session_id: str) -> SessionData:
        """Restore a previously persisted session.

        Args:
            session_id: Session to restore.

        Returns:
            The restored session data.

        Raises:
            FileNotFoundError: No session with this id exists.
        """
        session_dir = self._base_dir / session_id
        manifest_path = session_dir / "session.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No session with id {session_id!r} exists")

        manifest = json.loads(manifest_path.read_bytes())
        thread_names: list[str] = manifest["thread_names"]
        raw_ids: dict[str, str] = manifest["thread_ids"]

        threads: dict[str, list[Event]] = {}
        for name in thread_names:
            events_path = session_dir / f"{name}.events.json"
            raw = events_path.read_bytes()
            threads[name] = _EventListAdapter.validate_json(raw)

        thread_ids: dict[str, ThreadId] = {name: ThreadId(tid) for name, tid in raw_ids.items()}
        metadata: dict[str, object] | None = manifest.get("metadata")
        return SessionData(threads=threads, thread_ids=thread_ids, metadata=metadata)

    def exists(self, session_id: str) -> bool:
        """Check whether a session has been persisted.

        Args:
            session_id: Session to check.

        Returns:
            ``True`` iff ``load(session_id)`` would succeed.
        """
        return (self._base_dir / session_id / "session.json").exists()


def _atomic_write(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` via temp-file + atomic rename."""
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    closed = False
    try:
        _ = os.write(fd, data)
        os.close(fd)
        closed = True
        os.replace(tmp_path, target)
    except BaseException:
        if not closed:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
