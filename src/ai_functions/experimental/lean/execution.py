"""Shared process execution and cancellation for Lean consumers and provisioning."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .errors import LeanSetupError, LeanTimeoutError

__all__ = ["run_command", "run_command_async", "run_in_thread"]

_MAX_DIAGNOSTICS = 32_768


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _completed(
    command: Sequence[str], returncode: int, stdout: bytes, stderr: bytes, check: bool
) -> subprocess.CompletedProcess[str]:
    result = subprocess.CompletedProcess(
        command,
        returncode,
        stdout.decode("utf-8", errors="replace")[-_MAX_DIAGNOSTICS:],
        stderr.decode("utf-8", errors="replace")[-_MAX_DIAGNOSTICS:],
    )
    if check and returncode:
        raise LeanSetupError(f"Command failed: {' '.join(command)}\n{result.stdout}{result.stderr}")
    return result


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a process, killing and reaping its whole group on timeout/interruption.

    ``environment`` replaces the inherited environment when supplied. Nonzero
    exit codes raise ``LeanSetupError`` unless ``check=False``. Returned output
    retains the last 32,768 characters of each stream. Timeouts raise ``LeanTimeoutError``.
    """
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise LeanSetupError(f"Could not run {command[0]}: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException as exc:
        _kill_process_group(proc.pid)
        proc.wait()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise LeanTimeoutError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc
        raise
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    return _completed(command, proc.returncode, stdout, stderr, check)


async def run_command_async(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run with the same policy as ``run_command`` and reap children on cancellation."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise LeanSetupError(f"Could not run {command[0]}: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except BaseException as exc:
        _kill_process_group(proc.pid)
        await proc.wait()
        if isinstance(exc, TimeoutError):
            raise LeanTimeoutError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc
        raise
    return _completed(command, proc.returncode or 0, stdout, stderr, check)


async def run_in_thread[**P, T](function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Drain blocking work on cancellation before the caller releases its locks."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise
