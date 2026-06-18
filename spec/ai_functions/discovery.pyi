"""Discover a coordinator running on this host via a runtime file.

ai_functions follows the "runtime file" pattern (similar to Jupyter's kernel
connection files, or systemd runtime state) to let an unrelated client
process — a user script, a CLI command, a TUI session — find a
coordinator that was started earlier by ``ai-functions server``, without
hard-coding ports or ranges.

Single-coordinator model
------------------------
Only one coordinator may be advertised per user at a time. The runtime
file lives at a single, well-known path on disk (see
:func:`runtime_file_path`); a second ``ai-functions server`` invocation refuses
to start while a live file is present. This keeps discovery a no-brainer
for the common case: there is exactly one coordinator to find, or none.

Users who need multiple coordinators on the same host (e.g. isolated
test fixtures) construct :class:`CoordinatorClient` by URL directly and
skip discovery.

File format
-----------
The file is a small JSON document matching :class:`RuntimeInfo`. It is
written atomically — to a sibling ``*.tmp`` path followed by an
``os.replace`` — so concurrent readers never observe a partial file.
Permissions are ``0o600`` on POSIX so only the owning user can read the
URL. On Windows, file ACLs inherit from the parent directory (Windows
has no POSIX-mode analogue); the directory itself is a per-user
platformdirs path, which gives equivalent isolation in practice.

Liveness
--------
The file records the PID of the process that wrote it. Readers MUST
treat a file whose PID no longer exists as stale: they remove it and
behave as if no coordinator were running. This is the one recovery
path from a crash-without-cleanup; the coordinator itself removes the
file on graceful shutdown.

Example::

    from ai_functions import discover_coordinator
    from ai_functions.network import CoordinatorClient

    info = discover_coordinator()           # raises NoCoordinatorError if none
    async with CoordinatorClient.connect(info.url) as client:
        threads = await client.list_threads()
"""

from __future__ import annotations

from pathlib import Path
from typing import final

from pydantic import BaseModel


# ── Runtime file model ──────────────────────────────────────────────────────


@final
class RuntimeInfo(BaseModel):
    """Contents of the ai-functions coordinator runtime file.

    Written by ``ai-functions server`` when it starts the endpoint and removed
    on graceful shutdown. The schema is intentionally small; new optional
    fields may be added in later revisions, but existing fields MUST NOT
    change semantics (tools older than the server may still read the
    file).

    Attributes:
        url: Full WebSocket URL of the running endpoint
            (e.g. ``"ws://127.0.0.1:54123/rpc"``). The port is assigned
            by the OS at server start; clients MUST read it from this
            field rather than assume a default.
        pid: Operating-system PID of the process that wrote the file.
            Readers use this to detect stale files via ``os.kill(pid,
            0)`` (POSIX) or ``OpenProcess`` on Windows; a file pointing
            at a dead PID is treated as absent and removed.
        started_at: ISO-8601 timestamp (UTC, with ``"Z"`` suffix) of
            when the server became ready to accept connections. Purely
            informational — used by ``ai-functions server status`` and surfaced
            to humans; discovery does not depend on it.
        version: The ``ai_functions`` package version that wrote the file. A
            reader MAY warn on mismatch but MUST NOT refuse to connect;
            protocol compatibility is the wire layer's responsibility.
    """

    url: str
    pid: int
    started_at: str
    version: str


# ── Errors ──────────────────────────────────────────────────────────────────


class NoCoordinatorError(RuntimeError):
    """No live coordinator is advertised on this host.

    Raised by :func:`discover_coordinator` when the runtime file is
    missing or points at a dead PID. The CLI surfaces this as a friendly
    "run ``ai-functions server`` first" message.
    """


class CoordinatorAlreadyRunningError(RuntimeError):
    """Attempted to advertise a coordinator while another one is live.

    Raised by :func:`write_runtime_info` when the runtime file already
    exists and its PID is alive. The CLI surfaces this as "a ai_functions
    server is already running (pid=..., url=...)".
    """


# ── File path ───────────────────────────────────────────────────────────────


def runtime_dir() -> Path:
    """Return the directory the ai-functions runtime file lives in.

    The path is resolved via ``platformdirs.user_runtime_dir("ai_functions")``
    on platforms that have a runtime dir concept (XDG), and falls back
    to ``platformdirs.user_state_dir("ai_functions") / "run"`` elsewhere. The
    directory is NOT created by this function — callers that write do
    so; callers that read tolerate a missing directory as "no
    coordinator".

    Returns:
        The platform-appropriate directory path. The path may not
        exist on disk.
    """
    ...


def runtime_file_path() -> Path:
    """Return the absolute path to the single coordinator runtime file.

    Equivalent to ``runtime_dir() / "coordinator.json"``.

    Returns:
        The full path. The file may not exist on disk.
    """
    ...


# ── File I/O ────────────────────────────────────────────────────────────────


def write_runtime_info(info: RuntimeInfo) -> Path:
    """Atomically publish ``info`` as the active coordinator runtime file.

    Creates the parent directory if needed, then writes ``info`` to a
    sibling ``*.tmp`` file and renames it into place. On POSIX, the
    destination file's mode is set to ``0o600`` after rename.

    If the target path already exists and its recorded PID is alive,
    this raises :class:`CoordinatorAlreadyRunningError` — the
    single-coordinator invariant is enforced here, not at a higher
    level. If the existing file is stale (PID dead or file unparseable)
    it is silently removed and replaced.

    Args:
        info: The runtime metadata to publish.

    Returns:
        The path the file was written to.

    Raises:
        CoordinatorAlreadyRunningError: An existing live coordinator is
            advertised at :func:`runtime_file_path`.
        OSError: The runtime directory could not be created or the
            file could not be written.
    """
    ...


def read_runtime_info() -> RuntimeInfo | None:
    """Read and validate the current coordinator runtime file.

    Stale-file handling: if the file exists but its recorded PID is not
    alive, the file is removed and this function returns ``None``. A
    file that fails to parse as :class:`RuntimeInfo` is treated the
    same way — discovery must not wedge on a corrupted file left by an
    older version.

    Returns:
        The parsed :class:`RuntimeInfo` if a live coordinator is
        advertised; ``None`` if the file is absent or was stale.
    """
    ...


def delete_runtime_info() -> None:
    """Remove the coordinator runtime file if present; idempotent.

    Called by the server on graceful shutdown and by readers when they
    detect a stale file. No error is raised for a missing file.
    """
    ...


# ── Primary entry point ─────────────────────────────────────────────────────


def discover_coordinator() -> RuntimeInfo:
    """Return the live coordinator advertised on this host.

    Shorthand for :func:`read_runtime_info` that raises rather than
    returning ``None``. Suitable as the first line of any client-side
    helper (the CLI, :func:`ai_functions.run`, user scripts).

    Environment override: if ``AI_FUNCTIONS_COORDINATOR_URL`` is set, it is
    returned as a synthetic :class:`RuntimeInfo` (pid=0, empty
    ``version``, ``started_at`` = the current UTC timestamp) without
    touching the filesystem. This is the escape hatch for tests and
    for connecting to a coordinator run outside the runtime-file flow
    (e.g. one started directly with :class:`CoordinatorEndpoint`).

    Returns:
        The active coordinator's :class:`RuntimeInfo`.

    Raises:
        NoCoordinatorError: No runtime file exists, the file was stale
            (PID dead), and ``AI_FUNCTIONS_COORDINATOR_URL`` was not set.
    """
    ...
