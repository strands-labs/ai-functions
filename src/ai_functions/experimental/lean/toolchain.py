"""Lean configuration and the resolved installation shared by Lean consumers."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

from .errors import LeanSetupError
from .execution import run_command

DEFAULT_LEAN_TOOLCHAIN = "leanprover/lean4:v4.33.1"

__all__ = ["DEFAULT_LEAN_TOOLCHAIN", "LeanConfig", "ResolvedToolchain"]


def _default_mode() -> Literal["auto", "managed", "system"]:
    return cast(Literal["auto", "managed", "system"], os.environ.get("AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE", "auto"))


def _default_cache_dir() -> str | None:
    return os.environ.get("AI_FUNCTIONS_LEAN_CACHE_DIR")


@dataclass(frozen=True)
class LeanConfig:
    """Configure how to obtain Lean, without performing setup at construction.

    Args:
        mode: ``auto`` prefers an exact matching installed release, then the
            managed cache, provisioning only when necessary. ``system`` requires
            an installed release and never downloads. ``managed`` uses only the
            library cache. Defaults to ``AI_FUNCTIONS_LEAN_TOOLCHAIN_MODE`` or
            ``auto``. An explicit argument overrides the environment.
        cache_dir: Shared toolchain and bridge cache. Defaults to
            ``AI_FUNCTIONS_LEAN_CACHE_DIR`` or the platform's AI Functions cache.
    """

    mode: Literal["auto", "managed", "system"] = field(default_factory=_default_mode)
    cache_dir: str | Path | None = field(default_factory=_default_cache_dir)

    def __post_init__(self) -> None:
        """Reject unsupported provisioning options."""
        if self.mode not in ("auto", "managed", "system"):
            raise ValueError("mode must be 'auto', 'managed', or 'system'")

    def setup(
        self,
        lean_toolchain: str = DEFAULT_LEAN_TOOLCHAIN,
        *,
        offline: bool = False,
        timeout: float = 900.0,
    ) -> ResolvedToolchain:
        """Find or install an exact release and return its resolved installation.

        ``offline`` forbids downloads. ``timeout`` bounds provisioning, including
        cache lock waits. Setup does not build a bridge or inspect native SDKs.
        Repeated calls reuse the installation without changing global elan
        settings or the user's shell.

        Raises:
            LeanSetupError: No matching installation can be obtained.
            LeanTimeoutError: The setup deadline expires.
        """
        from ._provision import resolve_toolchain

        return resolve_toolchain(self, lean_toolchain=lean_toolchain, offline=offline, timeout=timeout)


@dataclass(frozen=True)
class ResolvedToolchain:
    """One concrete Lean installation returned by :meth:`LeanConfig.setup`.

    All executable paths and native build settings belong to this installation.
    Native SDK/header checks are deferred until ``native_flags`` or
    ``native_identity`` is accessed, so proof-only consumers need no host SDK.
    """

    root: Path
    identity: str
    environment_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def lean(self) -> Path:
        """Return the Lean executable without invoking an elan shim."""
        return self.root / "bin" / "lean"

    @property
    def lake(self) -> Path:
        """Return the Lake executable from this installation."""
        return self.root / "bin" / "lake"

    @property
    def leanc(self) -> Path:
        """Return this installation's native compiler driver."""
        return self.root / "bin" / "leanc"

    @property
    def leanchecker(self) -> Path:
        """Return the independent kernel checker from the same installation."""
        return self.root / "bin" / "leanchecker"

    def environment(self, directory: Path | None = None) -> dict[str, str]:
        """Build an isolated environment, optionally exposing a generated module."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("LEAN", "LAKE", "DYLD_"))
            and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "LIBRARY_PATH")
        }
        env.update(self.environment_overrides)
        env.update(
            PATH=str(self.root / "bin") + os.pathsep + os.environ.get("PATH", ""),
            LEAN_SYSROOT=str(self.root),
            LC_ALL="C",
        )
        if directory is not None:
            env["LEAN_PATH"] = str(directory)
        if sys.platform == "darwin":
            env["MACOSX_DEPLOYMENT_TARGET"] = "15.0"
        return env

    @cached_property
    def native_flags(self) -> tuple[str, ...]:
        """Validate native prerequisites and cache this host's compilation flags."""
        for path in (self.lean, self.leanc, self.leanchecker, self.root / "include" / "lean" / "lean.h"):
            if not path.is_file():
                raise LeanSetupError(f"Lean native compilation requires {path}. Reinstall the pinned toolchain.")
        flags = ["-fPIC", "-ffp-contract=off"]
        if sys.platform == "darwin":
            sdk = run_command(["/usr/bin/xcrun", "--show-sdk-path"], timeout=15).stdout.strip()
            # leanc disables normal system-header discovery, even with -isysroot.
            flags += ["-isysroot", sdk, "-isystem", str(Path(sdk) / "usr/include"), "-mmacosx-version-min=15.0"]
        return tuple(flags)

    def link_args(self, *, python_extension: bool = False) -> list[str]:
        """Link against this runtime, allowing Python symbols only for extensions."""
        library_dir = self.root / "lib" / "lean"
        flags = [
            "-L",
            str(library_dir),
            "-lleanshared",
            "-lInit_shared",
            f"-Wl,-rpath,{library_dir}",
            f"-Wl,-rpath,{self.root / 'lib'}",
        ]
        if sys.platform == "darwin" and python_extension:
            flags += ["-undefined", "dynamic_lookup"]
        elif sys.platform == "linux" and not python_extension:
            flags += ["-Wl,-z,defs"]
        return flags

    @property
    def native_identity(self) -> str:
        """Identify the installation, host, and settings used by native artifacts."""
        return json.dumps(
            [
                self.identity,
                str(self.root.resolve()),
                sys.platform,
                platform.machine(),
                self.native_flags,
                self.link_args(),
            ]
        )
