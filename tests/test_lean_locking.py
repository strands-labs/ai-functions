"""Cache exclusion shared across synchronous and asynchronous consumers."""

import asyncio
import threading

import pytest

from ai_functions.experimental.lean import LeanTimeoutError
from ai_functions.experimental.lean.execution import run_in_thread
from ai_functions.experimental.lean.locking import async_exclusive_file_lock, exclusive_file_lock


async def test_waiting_for_a_cache_lock_is_cancellable(tmp_path):
    path = tmp_path / "lock"
    entered = asyncio.Event()

    async def waiter():
        async with async_exclusive_file_lock(path):
            entered.set()

    async with async_exclusive_file_lock(path):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert not entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with async_exclusive_file_lock(path):
        pass


async def test_async_waiters_respect_sync_lock_and_timeout(tmp_path):
    path = tmp_path / "cache.lock"
    with exclusive_file_lock(path):
        with pytest.raises(LeanTimeoutError, match="cache lock"):
            async with async_exclusive_file_lock(path, timeout=0.05):
                pytest.fail("acquired a lock held by a synchronous owner")
    async with async_exclusive_file_lock(path, timeout=1):
        pass


async def test_sync_waiters_respect_async_owner(tmp_path):
    path = tmp_path / "cache.lock"
    entered = threading.Event()

    def acquire():
        with exclusive_file_lock(path, timeout=2):
            entered.set()

    async with async_exclusive_file_lock(path):
        task = asyncio.create_task(run_in_thread(acquire))
        await asyncio.sleep(0.05)
        assert not entered.is_set()
    await asyncio.wait_for(task, 5)
    assert entered.is_set()
