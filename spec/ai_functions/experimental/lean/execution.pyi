"""Shared process execution and cancellation for Lean consumers."""

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

__all__ = ["run_command", "run_command_async", "run_in_thread"]

def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]: ...

async def run_command_async(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]: ...

async def run_in_thread[**P, T](function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T: ...
