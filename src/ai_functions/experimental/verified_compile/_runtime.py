"""Build and load the direct CPython/Lean bridge in a local cache."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import ModuleType

import platformdirs

from ..lean import LeanConfig, ResolvedToolchain
from ..lean.errors import LeanError
from ..lean.execution import run_command
from ..lean.locking import exclusive_file_lock
from .errors import CompilerError

FFI_ABI_VERSION = 1
_NATIVE_LOCK = threading.RLock()
_BRIDGES: dict[Path, ModuleType] = {}
_BRIDGE_PROCESS: int | None = None


def require_supported_python() -> None:
    """Check the interpreter and platforms supported by the direct bridge."""
    if sys.implementation.name != "cpython" or sys.version_info < (3, 12):
        raise CompilerError("verified_ai_compile requires CPython 3.12 or newer.")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise CompilerError("verified_ai_compile requires a CPython build with the GIL enabled.")
    if sysconfig.get_config_var("Py_DEBUG"):
        raise CompilerError("verified_ai_compile requires a non-debug CPython build.")
    if sys.platform not in ("darwin", "linux"):
        raise CompilerError("verified_ai_compile currently supports macOS and Linux.")


@dataclass(frozen=True)
class Runtime:
    """A resolved Lean toolchain and its locally compiled Python bridge."""

    toolchain: ResolvedToolchain
    directory: Path
    identity: str
    include_dirs: tuple[str, ...]
    sources: dict[str, bytes]

    @property
    def extension(self) -> Path:
        """Return the bridge compiled for this interpreter's ABI."""
        return self.directory / f"_bridge{sysconfig.get_config_var('EXT_SUFFIX')}"

    def _valid(self) -> bool:
        try:
            manifest = json.loads((self.directory / "manifest.json").read_text())
            expected_names = {self.extension.name, *self.sources}
            if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
                return False
            if manifest["identity"] != self.identity or set(manifest["files"]) != expected_names:
                return False
            for name, expected in manifest["files"].items():
                path = self.directory / name
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def preflight(self, timeout: float = 120) -> None:
        """Build and load the bridge before any model call, or validate its cache."""
        self.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with exclusive_file_lock(self.directory.with_suffix(".lock"), timeout=timeout):
                if not self._valid():
                    with tempfile.TemporaryDirectory(prefix="bridge-", dir=self.directory.parent) as temporary:
                        work = Path(temporary)
                        for name, source in self.sources.items():
                            (work / name).write_bytes(source)
                        output = work / self.extension.name
                        command = [
                            str(self.toolchain.leanc),
                            "-shared",
                            "-O3",
                            "-std=c11",
                            "-Wall",
                            "-Wextra",
                            "-Werror=implicit-function-declaration",
                            *self.toolchain.native_flags,
                            *(flag for include in self.include_dirs for flag in ("-I", include)),
                            str(work / "bridge.c"),
                            *self.toolchain.link_args(python_extension=True),
                            "-o",
                            str(output),
                        ]
                        run_command(command, environment=self.toolchain.environment(work), timeout=timeout)
                        if sys.platform == "darwin":
                            run_command(["/usr/bin/codesign", "--force", "--sign", "-", str(output)], timeout=timeout)
                        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
                        (work / "manifest.json").write_text(json.dumps({"identity": self.identity, "files": hashes}))
                        if self.directory.exists():
                            shutil.rmtree(self.directory)
                        work.rename(self.directory)
                self.bridge()
        except LeanError as exc:
            error = CompilerError(
                "Could not build the Python/Lean bridge. Check the selected Lean installation, "
                "the running Python's development headers, and the host C SDK/linker."
            )
            error.diagnostics = str(exc)
            raise error from exc

    def bridge(self) -> ModuleType:
        """Load one bridge per process, retaining the original lifecycle guards."""
        global _BRIDGE_PROCESS
        with _NATIVE_LOCK:
            if _BRIDGES and _BRIDGE_PROCESS != os.getpid():
                raise CompilerError("Compiled functions require a fresh Python process after fork; use spawn.")
            if _BRIDGES and self.directory not in _BRIDGES:
                raise CompilerError("The compiler runtime cannot be switched in a running process. Restart Python.")
            if self.directory not in _BRIDGES:
                try:
                    module_name = "ai_functions.experimental.verified_compile._bridge"
                    spec = importlib.util.spec_from_file_location(module_name, self.extension)
                    if spec is None or spec.loader is None:
                        raise ImportError("Missing native extension loader")
                    bridge = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(bridge)
                    if bridge.ABI_VERSION != FFI_ABI_VERSION:
                        raise ImportError("Native bridge ABI does not match the compiler adapter")
                except (ImportError, OSError, RuntimeError, SystemError) as exc:
                    raise CompilerError(f"The locally compiled Python/Lean bridge could not be loaded: {exc}") from exc
                _BRIDGES[self.directory] = bridge
                _BRIDGE_PROCESS = os.getpid()
            return _BRIDGES[self.directory]


def resolve_runtime(lean_config: LeanConfig | None = None, *, offline: bool = False) -> Runtime:
    """Resolve ordinary Lean tools and the identity of a local bridge build."""
    require_supported_python()
    config = lean_config or LeanConfig()
    try:
        toolchain = config.setup(offline=offline)
        native_identity = toolchain.native_identity
    except LeanError as exc:
        raise CompilerError(f"Could not prepare Lean for verified_ai_compile: {exc}") from exc
    includes = tuple(dict.fromkeys(sysconfig.get_path(name) for name in ("include", "platinclude")))
    for name in ("Python.h", "pyconfig.h", "cpython/longintrepr.h"):
        if not any((Path(include) / name).is_file() for include in includes):
            raise CompilerError(
                f"The running CPython installation is missing {name}. "
                "Install its matching Python development headers before compiling."
            )
    sources = {name: files(__package__).joinpath("_native", name).read_bytes() for name in ("bridge.c", "ffi.h")}
    identity = json.dumps(
        {
            "build_version": 1,
            "native": native_identity,
            "python": sys.version,
            "soabi": sysconfig.get_config_var("SOABI"),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "includes": includes,
            "abi": FFI_ABI_VERSION,
            "bridge_link_args": toolchain.link_args(python_extension=True),
            "sources": {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()},
        },
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode()).hexdigest()
    cache = Path(config.cache_dir or platformdirs.user_cache_dir("ai_functions")).expanduser().resolve()
    return Runtime(toolchain, cache / "lean" / "bridges" / key, identity, includes, sources)
