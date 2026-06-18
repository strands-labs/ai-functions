"""Top-level Typer app for the ``ai_functions`` CLI.

This module is an internal implementation detail; end users interact
with the CLI via the ``ai_functions`` console script, which is wired to
:func:`ai_functions.cli.main` in :mod:`ai_functions.cli.__init__`.
"""
# pyright: reportUnusedFunction=false
# Typer registers each @app.command-decorated function by side effect; pyright
# cannot see the indirect reference and flags every command body as unused.
# Ruff B008 flags typer.Argument / typer.Option in default values; this is
# the standard Typer pattern so suppress it for the whole file.
# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

from ..types import ThreadId
from . import commands as _cmd
from .server import server_app

app: typer.Typer = typer.Typer(
    help="ai_functions — distributed AI thread coordination.",
    no_args_is_help=True,
)

app.add_typer(server_app, name="server")


@app.command("ps")
def _ps() -> None:
    """List registered threads."""
    raise typer.Exit(_cmd.ps())


@app.command("logs")
def _logs(
    thread_id: str = typer.Argument(..., help="Thread id."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Tail new events."),
    since: str | None = typer.Option(None, "--since", help="Replay only events after this id."),
) -> None:
    """Print the event log for a thread."""
    raise typer.Exit(_cmd.logs(ThreadId(thread_id), follow=follow, since=since))


@app.command("notify")
def _notify(
    thread_id: str = typer.Argument(..., help="Thread id."),
    text: str = typer.Argument(..., help="Message body."),
) -> None:
    """Inject a side-channel message into a thread (no cycle)."""
    raise typer.Exit(_cmd.notify(ThreadId(thread_id), text))


@app.command("submit")
def _submit(
    thread_id: str = typer.Argument(..., help="Thread id."),
    text: str = typer.Argument(..., help="Prompt body."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON: result plus token usage and timing."),
) -> None:
    """Run one cycle with a prompt and print the result (blocks)."""
    raise typer.Exit(_cmd.submit(ThreadId(thread_id), text, as_json=json_out))


@app.command("kill")
def _kill(
    thread_id: str = typer.Argument(..., help="Thread id."),
    now: bool = typer.Option(False, "--now", help="Use terminate_now (hard stop)."),
) -> None:
    """Terminate a thread."""
    raise typer.Exit(_cmd.kill(ThreadId(thread_id), now=now))


@app.command("attach")
def _attach(
    thread_id: str = typer.Argument(..., help="Thread id."),
) -> None:
    """Open the TUI for a thread."""
    # Import lazily so `ai-functions ps` doesn't pay the Textual import cost.
    from .attach import attach as _attach_impl

    raise typer.Exit(_attach_impl(ThreadId(thread_id)))


@app.command("run")
def _run(
    script: Path = typer.Argument(..., help="Path to the user script."),
    attr: str = typer.Option("main", "--attr", help="Module attribute holding the spawnable."),
) -> None:
    """Host a spawnable defined in a user script."""
    raise typer.Exit(_cmd.run_cmd(script, attr=attr))
