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
``examples/07_two_workers_remote.py``) and skip the CLI.
"""
# pyright: reportUnusedFunction=false
# Typer registers each @server_app.command-decorated function by side effect;
# pyright cannot see the indirect reference and flags every body as unused.

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib.metadata
import os
import signal

import typer

from ..connect import connect
from ..discovery import (
    CoordinatorAlreadyRunningError,
    RuntimeInfo,
    delete_runtime_info,
    read_runtime_info,
    write_runtime_info,
)
from ..network import CoordinatorEndpoint

server_app: typer.Typer = typer.Typer(
    help="Coordinator server (start / stop / status).",
    no_args_is_help=False,
    invoke_without_command=True,
)


_DEFAULT_HOST = "127.0.0.1"


def _package_version() -> str:
    try:
        return importlib.metadata.version("ai_functions")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@server_app.callback(invoke_without_command=True)
def _server_root(
    ctx: typer.Context,
    host: str = typer.Option(_DEFAULT_HOST, "--host", help="Interface to bind."),
) -> None:
    """Start the coordinator in the foreground (``ai-functions server``)."""
    if ctx.invoked_subcommand is not None:
        return
    code = start(host=host)
    raise typer.Exit(code)


@server_app.command("start")
def _start_cmd(
    host: str = typer.Option(_DEFAULT_HOST, "--host", help="Interface to bind."),
) -> None:
    """Explicit form of ``ai-functions server`` (start in the foreground)."""
    code = start(host=host)
    raise typer.Exit(code)


@server_app.command("stop")
def _stop_cmd() -> None:
    """Stop the advertised coordinator."""
    code = stop()
    raise typer.Exit(code)


@server_app.command("status")
def _status_cmd() -> None:
    """Show the advertised coordinator's URL, PID, and uptime."""
    code = status()
    raise typer.Exit(code)


def start(
    *,
    host: str = _DEFAULT_HOST,
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
    del url  # reserved for future test hook; not wired in v1.

    async def _run() -> int:
        endpoint = CoordinatorEndpoint()
        # Bind to port 0; the endpoint reads back the OS-assigned port.
        await endpoint.start(host=host, port=0)
        live_url = endpoint.url

        info = RuntimeInfo(
            url=live_url,
            pid=os.getpid(),
            started_at=_utc_now_iso(),
            version=_package_version(),
        )
        try:
            path = write_runtime_info(info)
        except CoordinatorAlreadyRunningError as exc:
            await endpoint.stop()
            typer.echo(f"error: {exc}", err=True)
            return 1

        typer.echo(f"ai-functions coordinator listening at {live_url}")
        typer.echo(f"runtime file: {path}")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        try:
            _ = await stop_event.wait()
        finally:
            delete_runtime_info()
            await endpoint.stop()

        typer.echo("ai-functions coordinator stopped")
        return 0

    return asyncio.run(_run())


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
    info = read_runtime_info()
    if info is None:
        typer.echo("no ai-functions coordinator is running")
        return 0

    async def _ask() -> int:
        try:
            async with connect(info.url) as client:
                # Best-effort: terminate everything we can see so teardown
                # runs cleanly before the server exits.
                for thread in await client.list_threads():
                    try:
                        await client.terminate(thread.thread_id)
                    except Exception:  # noqa: BLE001
                        pass
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 1

        try:
            os.kill(info.pid, signal.SIGTERM)
        except ProcessLookupError:
            delete_runtime_info()
            return 0
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 1

        # Poll for the runtime file to disappear.
        for _ in range(50):
            if read_runtime_info() is None:
                return 0
            await asyncio.sleep(0.1)
        typer.echo("warning: server did not remove its runtime file within 5s", err=True)
        return 1

    return asyncio.run(_ask())


def status() -> int:
    """Implementation of ``ai-functions server status``.

    Prints the active :class:`RuntimeInfo` (URL, PID, started_at,
    version) in a human-readable format, or ``"no coordinator"`` if
    none is advertised.

    Returns:
        Exit code; ``0`` if a coordinator is advertised (live or
        stale-and-cleaned), ``2`` if none.
    """
    info = read_runtime_info()
    if info is None:
        typer.echo("no ai-functions coordinator is running")
        return 2

    typer.echo(f"url:        {info.url}")
    typer.echo(f"pid:        {info.pid}")
    typer.echo(f"started_at: {info.started_at}")
    typer.echo(f"version:    {info.version}")
    return 0
