"""Shared process lifetimes and sync/async execution semantics."""

import asyncio
import os
import sys
import threading

import pytest

from ai_functions.experimental.lean import LeanSetupError, LeanTimeoutError
from ai_functions.experimental.lean.execution import run_command, run_command_async, run_in_thread


@pytest.mark.parametrize("how", ["cancel", "timeout"])
async def test_compiler_process_is_reaped_on_cancel_or_timeout(tmp_path, how, monkeypatch):
    binary = tmp_path / "toolchain" / "bin" / "slow"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
    binary.chmod(0o755)
    original_spawn = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    processes = []

    async def record_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_spawn)
    task = asyncio.create_task(run_command_async([str(binary)], cwd=tmp_path, timeout=0.2 if how == "timeout" else 30))
    await asyncio.wait_for(spawned.wait(), 5)
    pid = processes[0].pid
    if how == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    else:
        with pytest.raises(LeanTimeoutError):
            await asyncio.wait_for(task, 5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_command_runners_share_results_and_error_policy(tmp_path, asynchronous):
    command = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"]
    if asynchronous:
        result = await run_command_async(command, cwd=tmp_path, check=False)
        with pytest.raises(LeanSetupError, match="out"):
            await run_command_async(command)
    else:
        result = run_command(command, cwd=tmp_path, check=False)
        with pytest.raises(LeanSetupError, match="out"):
            run_command(command)
    assert result.returncode == 7
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_supplied_environment_is_not_merged_with_ambient_state(asynchronous, monkeypatch):
    monkeypatch.setenv("LEAN_TEST_AMBIENT", "must not escape")
    command = [sys.executable, "-c", "import os; print(os.environ.get('LEAN_TEST_AMBIENT', 'absent'))"]
    if asynchronous:
        result = await run_command_async(command, environment={})
    else:
        result = run_command(command, environment={})
    assert result.stdout.strip() == "absent"


async def test_cancelled_thread_work_finishes_before_caller_releases_state():
    entered, finish = threading.Event(), threading.Event()
    state = []

    def blocking_work():
        entered.set()
        assert finish.wait(5)
        state.append("work finished")

    async def caller():
        try:
            await run_in_thread(blocking_work)
        finally:
            state.append("caller released state")

    task = asyncio.create_task(caller())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert state == []
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert state == ["work finished", "caller released state"]
