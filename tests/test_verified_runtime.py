"""Local bridge builds, publication, and setup failures."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import pytest

from ai_functions.experimental.lean import LeanConfig, LeanSetupError, ResolvedToolchain
from ai_functions.experimental.verified_compile import _runtime, verified_ai_compile
from ai_functions.experimental.verified_compile._runtime import Runtime
from ai_functions.experimental.verified_compile.errors import CompilerError
from ai_functions.testing import ScriptedModel


@pytest.fixture
def bridge_build(tmp_path, monkeypatch):
    root = tmp_path / "toolchain"
    tools = ResolvedToolchain(root, "test Lean")
    tools.__dict__["native_flags"] = ()
    runtime = Runtime(
        tools,
        tmp_path / "bridges/test",
        "test identity",
        (),
        {"bridge.c": b"test bridge", "ffi.h": b"test header"},
    )
    builds = []

    def compile_bridge(command, **kwargs):
        if command[0].endswith("leanc"):
            builds.append(command)
            Path(command[command.index("-o") + 1]).write_bytes(b"compiled bridge")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_runtime, "run_command", compile_bridge)
    monkeypatch.setattr(Runtime, "bridge", lambda self: None)
    return runtime, builds


def test_concurrent_bridge_builds_publish_once_and_reuse_cache(bridge_build):
    runtime, builds = bridge_build
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: runtime.preflight(), range(8)))
    assert len(builds) == 1
    assert runtime._valid()
    assert (runtime.directory / "ffi.h").read_bytes() == runtime.sources["ffi.h"]
    assert not list(runtime.directory.parent.glob("bridge-*"))
    manifest = json.loads((runtime.directory / "manifest.json").read_text())
    assert set(manifest["files"]) == {"bridge.c", "ffi.h", runtime.extension.name}


@pytest.mark.parametrize("damage", ["binary", "header", "manifest", "missing"])
def test_corrupt_bridge_cache_is_rebuilt(bridge_build, damage):
    runtime, builds = bridge_build
    runtime.preflight()
    if damage == "binary":
        runtime.extension.write_bytes(b"corrupt")
    elif damage == "header":
        (runtime.directory / "ffi.h").write_bytes(b"corrupt")
    elif damage == "manifest":
        (runtime.directory / "manifest.json").write_text('{"files": []}')
    else:
        runtime.extension.unlink()
    runtime.preflight()
    assert len(builds) == 2
    assert runtime._valid()


def test_failed_bridge_build_never_publishes(bridge_build, monkeypatch):
    runtime, _ = bridge_build

    def fail(*args, **kwargs):
        raise LeanSetupError("compiler rejected the bridge")

    monkeypatch.setattr(_runtime, "run_command", fail)
    with pytest.raises(CompilerError, match="Python/Lean bridge") as caught:
        runtime.preflight()
    assert "compiler rejected" in caught.value.diagnostics
    assert not runtime.directory.exists()
    assert not list(runtime.directory.parent.glob("bridge-*"))


async def test_native_build_failure_precedes_any_model_request(bridge_build, monkeypatch, tmp_path):
    runtime, _ = bridge_build

    def fail(*args, **kwargs):
        raise LeanSetupError("missing C headers")

    monkeypatch.setattr(_runtime, "run_command", fail)
    monkeypatch.setattr(
        "ai_functions.experimental.verified_compile.function.resolve_runtime", lambda *args, **kwargs: runtime
    )

    def constant() -> int:
        """Return seven."""

    def contract(result):
        assert result == 7

    model = ScriptedModel([])
    fn = verified_ai_compile(post_conditions=[contract], model=model, cache_dir=tmp_path / "functions")(constant)
    with pytest.raises(CompilerError, match="Python/Lean bridge"):
        await fn.compile()
    assert not fn.is_compiled


def test_existing_bridge_does_not_recompile(native_runtime, monkeypatch):
    monkeypatch.setattr(_runtime, "run_command", lambda *a, **kw: pytest.fail("cached bridge was recompiled"))
    native_runtime.preflight()


def test_missing_leanchecker_is_a_setup_error(tmp_path):
    root = tmp_path / "toolchain"
    (root / "bin").mkdir(parents=True)
    for name in ("lean", "leanc"):
        (root / "bin" / name).touch()
    tools = ResolvedToolchain(root, "test")
    with pytest.raises(LeanSetupError, match="leanchecker"):
        _ = tools.native_flags


def test_missing_python_headers_are_actionable(native_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(LeanConfig, "setup", lambda *a, **kw: native_runtime.toolchain)
    monkeypatch.setattr(_runtime.sysconfig, "get_path", lambda name: str(tmp_path / "missing"))
    with pytest.raises(CompilerError, match="Python.h"):
        _runtime.resolve_runtime(offline=True)


def test_bridge_identity_separates_abi_and_build_settings(native_runtime, monkeypatch):
    original = _runtime.resolve_runtime(offline=True)
    original_get = _runtime.sysconfig.get_config_var
    monkeypatch.setattr(
        _runtime.sysconfig, "get_config_var", lambda key: "different-abi" if key == "SOABI" else original_get(key)
    )
    assert _runtime.resolve_runtime(offline=True).directory != original.directory
    monkeypatch.setattr(_runtime.sysconfig, "get_config_var", original_get)
    flags = native_runtime.toolchain.native_flags
    monkeypatch.setattr(ResolvedToolchain, "native_flags", property(lambda self: (*flags, "-g")))
    assert _runtime.resolve_runtime(offline=True).directory != original.directory


def test_cached_managed_toolchain_builds_bridge_offline_in_a_fresh_process(native_runtime, tmp_path):
    from ai_functions.experimental.lean import _provision

    asset, _ = _provision._elan_asset()
    bootstrap = tmp_path / "lean/toolchains" / f"elan-{_provision._ELAN_VERSION}-{asset.removesuffix('.tar.gz')}"
    (bootstrap / "bin").mkdir(parents=True)
    # Deliberately non-executable: offline reuse must invoke the concrete tools.
    (bootstrap / "bin/elan").touch()
    (bootstrap / ".bootstrap-complete").touch()
    installed = bootstrap / "toolchains/leanprover--lean4---v4.33.1"
    installed.parent.mkdir()
    installed.symlink_to(native_runtime.toolchain.root, target_is_directory=True)
    script = """
from ai_functions.experimental.lean import LeanConfig, _provision
from ai_functions.experimental.verified_compile._runtime import resolve_runtime
def forbidden(*args, **kwargs):
    raise AssertionError("offline setup downloaded")
_provision._download = forbidden
r = resolve_runtime(LeanConfig(mode="managed"), offline=True)
r.preflight()
assert r.extension.is_file()
assert r.bridge().ABI_VERSION == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "AI_FUNCTIONS_LEAN_CACHE_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_source_resources_and_imports_need_no_runtime_setup(tmp_path):
    resource = files("ai_functions.experimental.verified_compile").joinpath("_native")
    assert resource.joinpath("bridge.c").is_file()
    assert resource.joinpath("ffi.h").is_file()
    script = """
import subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError("import ran toolchain setup")
subprocess.Popen = forbidden
import ai_functions
assert "ai_functions.experimental.verified_compile" not in sys.modules
assert not hasattr(ai_functions, "verified_ai_compile")
assert not hasattr(ai_functions, "ai_verified_function")
from ai_functions.experimental.verified_compile import verified_ai_compile
from ai_functions.experimental.lean import LeanConfig
LeanConfig()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "AI_FUNCTIONS_LEAN_CACHE_DIR": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "cache").exists()
