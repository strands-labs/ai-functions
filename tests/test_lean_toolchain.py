"""Shared provisioning behavior ported independently of project support."""

import io
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ai_functions.experimental.lean import LeanConfig, LeanSetupError, LeanTimeoutError


def _fake_installation(directory: Path, version: str = "4.33.1") -> Path:
    binary_dir = directory / "bin"
    binary_dir.mkdir(parents=True)
    for name in ("lean", "lake", "leanc"):
        binary = binary_dir / name
        binary.write_text(f"#!/bin/sh\nprintf 'Lean (version {version}, test installation)\\n'\n")
        binary.chmod(0o755)
    return binary_dir


def test_toolchain_defaults_and_construction_do_not_perform_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_functions.experimental.lean import _provision

    monkeypatch.delenv("AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE", raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("constructor ran a process"))
    monkeypatch.setattr(_provision, "_download", lambda *a, **kw: pytest.fail("constructor downloaded"))
    assert LeanConfig().mode == "auto"
    monkeypatch.setenv("AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE", "system")
    tools = LeanConfig(cache_dir=tmp_path / "cache")
    assert tools.mode == "system"
    assert LeanConfig(mode="managed").mode == "managed"
    assert not (tmp_path / "cache").exists()
    monkeypatch.setenv("AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE", "invalid")
    with pytest.raises(ValueError, match="mode"):
        LeanConfig()


@pytest.mark.parametrize("registered", [False, True])
def test_auto_setup_reuses_matching_system_release_without_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    from ai_functions.experimental.lean import _provision

    home = tmp_path / "elan"
    root = home / "toolchains" / "leanprover--lean4---v4.33.1" if registered else tmp_path / "system"
    binary_dir = _fake_installation(root)
    monkeypatch.setenv("ELAN_HOME", str(home))
    monkeypatch.delenv("AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE", raising=False)
    monkeypatch.setattr(_provision.shutil, "which", lambda name: None if registered else str(binary_dir / name))
    monkeypatch.setattr(_provision, "_download", lambda *a, **kw: pytest.fail("unnecessary download"))
    cache = tmp_path / "cache"
    tools = LeanConfig(cache_dir=cache).setup(timeout=5)
    assert tools.lean == binary_dir / "lean"
    assert not cache.exists()


def test_auto_setup_falls_back_to_cached_release_when_system_version_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_functions.experimental.lean import _provision

    system = _fake_installation(tmp_path / "system", "4.0.0")
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
    monkeypatch.setattr(_provision.shutil, "which", lambda name: str(system / name))
    monkeypatch.setattr(_provision, "_download", lambda *a, **kw: pytest.fail("cached release downloaded"))
    cache = tmp_path / "cache"
    asset, _ = _provision._elan_asset()
    bootstrap = cache / "lean/toolchains" / f"elan-{_provision._ELAN_VERSION}-{asset.removesuffix('.tar.gz')}"
    (bootstrap / "bin").mkdir(parents=True)
    (bootstrap / "bin/elan").touch()
    (bootstrap / ".bootstrap-complete").touch()
    installed = _fake_installation(bootstrap / "toolchains/leanprover--lean4---v4.33.1")
    with pytest.raises(LeanSetupError, match="reports"):
        LeanConfig(mode="system", cache_dir=cache).setup(offline=True, timeout=5)
    result = LeanConfig(mode="auto", cache_dir=cache).setup(offline=True, timeout=5)
    assert result.lean == installed / "lean"


def test_auto_setup_downloads_only_when_no_matching_installation_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_functions.experimental.lean import _provision

    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
    monkeypatch.setattr(_provision.shutil, "which", lambda name: None)
    attempts = []

    def download(url, path, *, timeout):
        attempts.append((url, timeout))
        raise LeanSetupError("test stopped the download")

    monkeypatch.setattr(_provision, "_download", download)
    tools = LeanConfig(mode="auto", cache_dir=tmp_path / "cache")
    assert attempts == []
    with pytest.raises(LeanSetupError, match="test stopped"):
        tools.setup(timeout=5)
    assert len(attempts) == 1
    assert 0 < attempts[0][1] <= 5


def test_explicit_setup_timeout_terminates_slow_version_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_functions.experimental.lean import _provision

    binary_dir = _fake_installation(tmp_path / "system")
    (binary_dir / "lean").write_text("#!/bin/sh\nsleep 60 &\nwait\n")
    monkeypatch.setenv("ELAN_HOME", str(tmp_path / "elan"))
    monkeypatch.setattr(_provision.shutil, "which", lambda name: str(binary_dir / name))
    tools = LeanConfig(mode="system")
    start = time.monotonic()
    with pytest.raises(LeanTimeoutError):
        tools.setup(timeout=0.1)
    assert time.monotonic() - start < 5
    with pytest.raises(ValueError, match="positive"):
        tools.setup(timeout=0)


def test_setup_deadline_also_applies_to_streaming_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_functions.experimental.lean import _provision

    class SlowResponse(io.BytesIO):
        def read1(self, size=-1):
            time.sleep(0.02)
            return super().read1(size)

    monkeypatch.setattr(_provision.urllib.request, "urlopen", lambda *a, **kw: SlowResponse(b"installer"))
    with pytest.raises(LeanTimeoutError, match="deadline"):
        _provision._download("https://example.invalid/installer", tmp_path / "download", timeout=0.01)


def test_setup_cache_lock_respects_timeout(tmp_path: Path) -> None:
    from ai_functions.experimental.lean.locking import exclusive_file_lock

    path = tmp_path / "lock"
    with exclusive_file_lock(path), ThreadPoolExecutor(max_workers=1) as pool:

        def wait_for_lock():
            with exclusive_file_lock(path, timeout=0.1):
                pytest.fail("lock acquired concurrently")

        future = pool.submit(wait_for_lock)
        with pytest.raises(LeanTimeoutError, match="cache lock"):
            future.result(timeout=5)
