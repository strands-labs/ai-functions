"""Shared fixtures for comprehensive unit tests.

This module provides common fixtures used across all test modules.

It also installs an autouse **no-external-network guard**: every test in this
suite is expected to run offline (LLM-backed code paths are always exercised
through injected fakes/stubs), so an outbound socket connection to anything
but loopback is a bug -- typically an accidentally-live model call. The guard
patches ``socket.socket.connect``/``connect_ex`` to raise ``RuntimeError`` for
non-loopback addresses; loopback (``127.0.0.0/8``, ``::1``, ``localhost``,
IPv4-mapped forms such as ``::ffff:127.0.0.1``) and
non-IP address families (e.g. ``AF_UNIX``) stay allowed so tests that spin up
a local server keep working. Deliberately a plain socket-level guard rather
than a new dependency (no ``pytest-socket``).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import platformdirs
import pytest

_LOOPBACK_HOSTS = frozenset(
    {
        "",
        "0.0.0.0",  # noqa: S104 - a bind-all *bind* address, never an outbound destination
        "::",
        "::1",
        "localhost",
        "localhost.localdomain",
    }
)


def _is_local_address(address: Any) -> bool:
    """Whether ``address`` is loopback (or not a host/port address at all)."""
    if not isinstance(address, tuple) or not address:
        # AF_UNIX paths and anything else non-host-based: not network traffic.
        return True
    host = address[0]
    if isinstance(host, bytes | bytearray):
        host = host.decode("utf-8", "replace")
    host = str(host)
    if host.startswith("::ffff:"):
        # IPv4-mapped IPv6: judge it by the embedded IPv4 address.
        host = host[len("::ffff:") :]
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block outbound non-loopback socket connections for the duration of a test."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if not _is_local_address(address):
            raise RuntimeError(
                f"blocked outbound network connection to {address!r} during tests -- "
                "this suite must run offline (inject a fake/stub instead of calling a real service)"
            )
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if not _is_local_address(address):
            raise RuntimeError(
                f"blocked outbound network connection to {address!r} during tests -- "
                "this suite must run offline (inject a fake/stub instead of calling a real service)"
            )
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture(scope="session", autouse=True)
def lean_cache():
    """Keep compiled bridges and support executables in a persistent test-only cache.

    Entries are keyed by toolchain and source identity, so reusing them across
    runs is safe and saves rebuilding them on every run.
    """
    previous = os.environ.get("AI_FUNCTIONS_LEAN_CACHE_DIR")
    if previous is None:
        os.environ["AI_FUNCTIONS_LEAN_CACHE_DIR"] = platformdirs.user_cache_dir("ai_functions-tests")
    yield
    if previous is None:
        os.environ.pop("AI_FUNCTIONS_LEAN_CACHE_DIR", None)


@pytest.fixture(scope="session")
def lean_tools(lean_cache):
    """Resolve Lean without requiring the C bridge or a native SDK."""
    from ai_functions.experimental.verified.lean.toolchain import LeanConfig

    return LeanConfig().setup(offline=True)


@pytest.fixture(scope="session")
def trusted_lean_modules(lean_tools):
    """Core modules the native audit permits as executable dependencies."""
    core = lean_tools.root / "lib" / "lean"
    return frozenset(".".join(path.relative_to(core).with_suffix("").parts) for path in core.rglob("*.olean"))


@pytest.fixture(scope="session")
def native_runtime(lean_cache):
    """Run real Lean tests offline, optionally requiring native prerequisites."""
    from ai_functions.experimental.verified.compile._runtime import resolve_runtime
    from ai_functions.experimental.verified.compile.errors import CompilerError
    from ai_functions.experimental.verified.lean import LeanError
    from ai_functions.experimental.verified.lean.toolchain import LeanConfig, cache_path

    try:
        config = LeanConfig()
        runtime = resolve_runtime(config.setup(offline=True), cache_path(config))
        runtime.preflight()
        return runtime
    except (CompilerError, LeanError):
        if os.environ.get("AI_FUNCTIONS_REQUIRE_VERIFIED_NATIVE"):
            raise
        pytest.skip("Prepare Lean with LeanConfig().setup() to run native integration tests")


# Shared prepared projects, one per fixture project. Each is prepared once per
# session; a test that uses one is marked with the matching ``xdist_group`` so
# that, under ``--dist loadgroup``, one worker prepares it once. Tests about
# preparation itself create their own project instead.


@pytest.fixture(scope="session")
def init_project(lean_cache):
    """A prepared project with only Lean core in scope and nothing registered. Group: ``init``."""
    from ai_functions.experimental.verified.lean import LeanProject

    project = LeanProject(offline=True)
    project.prepare()
    return project


@pytest.fixture(scope="session")
def pricing_project(lean_cache):
    """The prepared `fixtures/verified/pricing` project with the contracts the function tests use.

    Group: ``verified_pricing``.
    """
    from ai_functions.experimental.verified.lean import LeanProject

    project = LeanProject(Path(__file__).parent / "fixtures" / "verified" / "pricing", imports=("Pricing",))
    project.add("def Identity (result : Nat) (x : Nat) : Prop := result = x")
    project.add("def Truth (result : Nat) : Prop := True")
    project.add("opaque days : String → Nat")
    project.add("def Days (result : Nat) (message : String) : Prop := result = days message")
    project.identity, project.truth = project.symbols.Identity, project.symbols.Truth
    project.days, project.days_contract = project.symbols.days, project.symbols.Days
    project.prepare()
    return project


@pytest.fixture(scope="session")
def verified_contracts(lean_cache):
    """The prepared `fixtures/verified/contracts` project. Group: ``verified_contracts``."""
    from verified_model import fixture_project

    project = fixture_project()
    project.prepare()
    return project


@pytest.fixture(scope="session")
def compile_project(lean_cache):
    """The prepared `fixtures/verified/compile` project. Group: ``compile_fixture``."""
    from ai_functions.experimental.verified.lean import LeanProject

    project = LeanProject(
        Path(__file__).parent / "fixtures" / "verified" / "compile", imports=("Fixture",), offline=True
    )
    project.prepare()
    return project


_LEAN_EXECUTABLES = frozenset({"elan", "lake", "lean", "leanc", "leanchecker"})


def _starts_real_lean(args: object, executable: object) -> bool:
    """Whether a process launch runs a real Lean tool rather than a test's shell-script stand-in."""
    if executable is None:
        executable = args if isinstance(args, (str, bytes, os.PathLike)) else next(iter(args))  # type: ignore[call-overload]
    program = os.fsdecode(executable)  # type: ignore[arg-type]
    program = program.split()[0] if program.strip() else program
    if os.path.basename(program) not in _LEAN_EXECUTABLES:
        return False
    path = program if os.sep in program else shutil.which(program)
    if path is None or not os.path.isfile(path):
        return False
    with open(path, "rb") as file:
        return file.read(2) != b"#!"


class _NoLeanPopen(subprocess.Popen):
    def __init__(self, args, *rest, **kwargs):  # type: ignore[no-untyped-def]
        if _starts_real_lean(args, kwargs.get("executable")):
            pytest.fail(f"a test not marked `lean` started a Lean process: {args!r}", pytrace=False)
        super().__init__(args, *rest, **kwargs)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # type: ignore[no-untyped-def]
    """Fail any test not marked `lean` that starts a real Lean process.

    Every launch (``execution.spawn_lake``, ``run_lake``, ``run_command`` and
    its async form, and direct ``subprocess`` calls) goes through
    ``subprocess.Popen``, so guarding it covers them all.
    """
    if item.get_closest_marker("lean") is not None:
        yield
        return
    original = subprocess.Popen
    subprocess.Popen = _NoLeanPopen  # type: ignore[misc]
    try:
        yield
    finally:
        subprocess.Popen = original  # type: ignore[misc]
