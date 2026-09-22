"""Automatic, isolated Lean toolchain provisioning."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import platformdirs

from .errors import LeanSetupError, LeanTimeoutError
from .execution import run_command
from .locking import exclusive_file_lock
from .toolchain import LeanConfig, ResolvedToolchain

_logger = logging.getLogger(__name__)

_ELAN_VERSION = "4.2.3"
_ELAN_RELEASE_BASE = f"https://github.com/leanprover/elan/releases/download/v{_ELAN_VERSION}"
_ELAN_ASSETS = {
    ("darwin", "aarch64"): (
        "elan-aarch64-apple-darwin.tar.gz",
        "7cae4c03b2f0de4053fb04a91359d5804551e6e37a6ddd1b2e0097dc561ae4a9",
    ),
    ("darwin", "x86_64"): (
        "elan-x86_64-apple-darwin.tar.gz",
        "10d037a69731c0593723e018130c5f54afde175796b4af8ba1317e561e55598c",
    ),
    ("linux", "aarch64"): (
        "elan-aarch64-unknown-linux-gnu.tar.gz",
        "cb69af0803b04157bc30201c29c12fca882bb3ad8b43476b8d2d3064810bc3ac",
    ),
    ("linux", "x86_64"): (
        "elan-x86_64-unknown-linux-gnu.tar.gz",
        "df0b2b3a439961ffcbb3985214365ffe40f49bc871df04dff268c7d8e21ca8b2",
    ),
}


def _normalized_machine() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"amd64", "x86_64"}:
        return "x86_64"
    return machine


def _elan_asset() -> tuple[str, str]:
    if sys.platform == "darwin":
        platform_name = "darwin"
    elif sys.platform.startswith("linux"):
        platform_name = "linux"
    else:
        platform_name = sys.platform
    key = (platform_name, _normalized_machine())
    try:
        return _ELAN_ASSETS[key]
    except KeyError as exc:
        raise LeanSetupError(
            f"Automatic Lean installation does not support {sys.platform!r} on {platform.machine()!r}",
        ) from exc


def _download(url: str, destination: Path, *, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "strands-ai-functions"})
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
            while True:
                chunk = response.read1(65536)
                if time.monotonic() >= deadline:
                    raise LeanTimeoutError("Lean installer download exceeded its deadline")
                if not chunk:
                    break
                output.write(chunk)
    except TimeoutError as exc:
        raise LeanTimeoutError("Lean installer download timed out") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise LeanSetupError(
            f"Could not download the managed Lean installer from {url}. "
            "Check network access, or use LeanConfig(mode='system').",
        ) from exc


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise LeanSetupError(f"Managed Lean installer checksum mismatch: expected {expected}, got {actual}")


def _install_elan(archive_path: Path, destination: Path, *, timeout: float) -> None:
    with tempfile.TemporaryDirectory(prefix="elan-install-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            try:
                installer_member = archive.getmember("elan-init")
            except KeyError as exc:
                raise LeanSetupError("Managed Lean installer archive does not contain `elan-init`") from exc
            archive.extract(installer_member, temporary_path, filter="data")

        installer = temporary_path / "elan-init"
        installer.chmod(installer.stat().st_mode | stat.S_IXUSR)
        staging_home = temporary_path / "elan-home"
        run_command(
            [str(installer), "-y", "--no-modify-path", "--default-toolchain", "none"],
            environment={**os.environ, "ELAN_HOME": str(staging_home)},
            timeout=timeout,
        )

        lake = staging_home / "bin" / "lake"
        if not lake.exists():
            raise LeanSetupError("Managed Lean installer completed without creating the `lake` shim")
        (staging_home / ".bootstrap-complete").write_text(f"elan {_ELAN_VERSION}\n")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging_home, destination)


def _resolved(bin_dir: Path, pin: str, environment: dict[str, str], timeout: float) -> ResolvedToolchain:
    lean, lake, leanc = (bin_dir / name for name in ("lean", "lake", "leanc"))
    if not all(path.is_file() for path in (lean, lake, leanc)):
        raise LeanSetupError(f"Incomplete Lean installation in {bin_dir}")
    environment = {**environment, "ELAN_TOOLCHAIN": pin}
    tools = ResolvedToolchain(bin_dir.parent, pin, environment)
    version = run_command([str(lean), "--version"], environment=tools.environment(), timeout=timeout).stdout.strip()
    requested_version = pin.rsplit(":", 1)[-1].removeprefix("v")
    match = re.search(r"version\s+([^,\s]+)", version)
    if re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.]+)?", requested_version):
        if match is None or match[1] != requested_version:
            raise LeanSetupError(f"Requested {pin}, but {lean} reports {version}")
    identity = pin + "\n" + version
    return ResolvedToolchain(bin_dir.parent, identity, environment)


def _system_toolchain(pin: str, *, timeout: float) -> ResolvedToolchain:
    """Find a matching release without ever invoking an elan shim."""
    directory_name = pin.replace("/", "--").replace(":", "---")
    home = Path(os.environ.get("ELAN_HOME", str(Path.home() / ".elan"))).expanduser().resolve()
    installed = home / "toolchains" / directory_name / "bin"
    if installed.is_dir():
        return _resolved(installed, pin, {}, timeout)
    candidate = shutil.which("lean")
    if candidate:
        binary = Path(candidate).resolve()
        # Elan shims are the elan executable itself, often hard-linked.
        elan = binary.parent / "elan"
        shim = elan.is_file() and os.path.samefile(binary, elan)
        if not shim and binary.parent != home / "bin" and binary.name != "elan":
            if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:-[\w.]+)?", pin.rsplit(":", 1)[-1]):
                raise LeanSetupError(f"Cannot verify non-release pin {pin} against an unregistered system installation")
            return _resolved(binary.parent, pin, {}, timeout)
    raise LeanSetupError(f"Lean {pin} is not installed; install it with elan or use LeanConfig() to provision it")


def resolve_toolchain(
    config: LeanConfig,
    *,
    lean_toolchain: str,
    offline: bool,
    timeout: float = 900.0,
) -> ResolvedToolchain:
    """Resolve concrete executables without allowing elan to download implicitly."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise LeanTimeoutError("Toolchain setup exceeded its deadline")
        return seconds

    pin = lean_toolchain.strip()
    if not pin or pin.startswith("-") or not re.fullmatch(r"[\w./:+-]+", pin):
        raise LeanSetupError(f"Invalid Lean toolchain pin: {lean_toolchain!r}")
    if config.mode in ("auto", "system"):
        try:
            return _system_toolchain(pin, timeout=remaining())
        except LeanSetupError:
            if config.mode == "system":
                raise
            _logger.debug("No matching system Lean for %s; checking the managed cache", pin)

    directory_name = pin.replace("/", "--").replace(":", "---")
    cache = Path(config.cache_dir or platformdirs.user_cache_dir("ai_functions")).expanduser().resolve()
    asset, checksum = _elan_asset()
    destination = cache / "lean" / "toolchains" / f"elan-{_ELAN_VERSION}-{asset.removesuffix('.tar.gz')}"
    installed = destination / "toolchains" / directory_name / "bin"
    environment = {"ELAN_HOME": str(destination)}
    with exclusive_file_lock(destination.parent / ".locks" / f"{destination.name}.lock", timeout=remaining()):
        elan = destination / "bin" / "elan"
        if not (destination / ".bootstrap-complete").is_file() or not elan.is_file():
            if offline:
                raise LeanSetupError(f"Managed Lean bootstrap is missing in offline mode: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="elan-download-", dir=destination.parent) as temporary:
                archive = Path(temporary) / asset
                _logger.info("Downloading Lean installer to %s", destination)
                _download(f"{_ELAN_RELEASE_BASE}/{asset}", archive, timeout=remaining())
                _verify_sha256(archive, checksum)
                _install_elan(archive, destination, timeout=remaining())
        if not (installed / "lean").is_file():
            if offline:
                raise LeanSetupError(f"Lean {pin} is not cached and offline=True")
            _logger.info("Installing Lean %s in %s", pin, destination)
            run_command(
                [str(elan), "toolchain", "install", pin],
                environment={**os.environ, **environment},
                timeout=remaining(),
            )
        return _resolved(installed, pin, environment, remaining())
