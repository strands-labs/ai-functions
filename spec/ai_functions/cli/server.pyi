"""``ai-functions server`` subcommand group — start / stop / status.

``ai-functions server`` (no further args) runs the coordinator in the
foreground: it binds an :class:`~ai_functions.network.CoordinatorEndpoint` to
an OS-assigned port, publishes the runtime file, and blocks until
interrupted. On SIGINT / SIGTERM it stops the endpoint cleanly and
removes the runtime file.

Daemonisation is deliberately not implemented in v1 — running the
server under ``tmux``, ``systemd --user``, or a simple ``&`` is the
workflow for now.

Port selection
--------------
The endpoint binds port ``0`` and asks the OS to assign a free port.
The actual port is read back from the listening socket and recorded in
the runtime file before any client is told the server is ready. The
coordinator MUST NOT reuse :data:`ai_functions.network.DEFAULT_PORT` here —
that value exists for direct :class:`CoordinatorEndpoint` tests and
colliding with it on every ``ai-functions server`` invocation would make the
tool unusable alongside ad-hoc endpoints.

Single-instance invariant
-------------------------
:func:`ai_functions.discovery.write_runtime_info` refuses to publish while a
live file is present, so a second ``ai-functions server`` fails fast with a
clear error. Users who really want multiple coordinators run
:class:`CoordinatorEndpoint` directly (see
``examples/08_remote_coordination.py``) and skip the CLI.
"""

from __future__ import annotations

from typing import Any


server_app: Any  # pyright: ignore[reportExplicitAny]  # typer.Typer sub-app
"""Typer sub-app mounted at ``ai-functions server``. Exported so tests can invoke
subcommands via ``CliRunner`` without spawning a subprocess."""


def start(
    *,
    host: str = ...,
    url: str | None = None,
) -> int:
    """Implementation of ``ai-functions server`` (no subcommand given).

    Binds a new :class:`CoordinatorEndpoint` to ``host:0``, writes the
    runtime file, and blocks until SIGINT / SIGTERM. On shutdown the
    runtime file is removed even if the endpoint's own stop path
    raises.

    Args:
        host: Interface to bind; defaults to ``"127.0.0.1"``. Callers
            who want LAN exposure pass ``"0.0.0.0"`` explicitly —
            discovery still only advertises this single URL via the
            runtime file, so LAN clients must set
            ``AI_FUNCTIONS_COORDINATOR_URL`` or pass ``--url``.
        url: Reserved for tests that want to advertise a pre-bound
            endpoint without rebinding. When set, skips binding and
            just publishes the runtime file; the CLI flag is hidden.

    Returns:
        Exit code; ``0`` on clean shutdown, non-zero on bind failure
        or if another coordinator is already advertised.
    """
    ...


def stop() -> int:
    """Implementation of ``ai-functions server stop``.

    Reads the runtime file, opens a short-lived
    :class:`CoordinatorClient`, asks the server to shut down, and
    returns once the runtime file is gone. If the runtime file is
    absent, this is a no-op that returns ``0``.

    Returns:
        Exit code; ``0`` if the server stopped (or was already down),
        ``1`` if the stop request could not be delivered.
    """
    ...


def status() -> int:
    """Implementation of ``ai-functions server status``.

    Prints the active :class:`RuntimeInfo` (URL, PID, started_at,
    version) in a human-readable format, or ``"no coordinator"`` if
    none is advertised.

    Returns:
        Exit code; ``0`` if a coordinator is advertised (live or
        stale-and-cleaned), ``2`` if none.
    """
    ...
