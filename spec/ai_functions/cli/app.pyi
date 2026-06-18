"""Top-level Typer app for the ``ai_functions`` CLI.

This module is an internal implementation detail; end users interact
with the CLI via the ``ai_functions`` console script, which is wired to
:func:`ai_functions.cli.main` in :mod:`ai_functions.cli.__init__`.
"""

from __future__ import annotations

from typing import Any


app: Any  # pyright: ignore[reportExplicitAny]  # typer.Typer root app
"""The root Typer app. Exposed so tests can drive subcommands through
``typer.testing.CliRunner`` without spawning a subprocess."""
