"""Lean configuration and the resolved installation shared by Lean consumers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_LEAN_TOOLCHAIN: str

__all__ = ["DEFAULT_LEAN_TOOLCHAIN", "LeanConfig", "ResolvedToolchain"]

@dataclass(frozen=True)
class LeanConfig:
    """Choose installed or managed Lean tools without performing setup on construction."""

    mode: Literal["auto", "managed", "system"] = ...
    cache_dir: str | Path | None = ...
    def __post_init__(self) -> None: ...
    def setup(
        self,
        lean_toolchain: str = DEFAULT_LEAN_TOOLCHAIN,
        *,
        offline: bool = False,
        timeout: float = 900.0,
    ) -> ResolvedToolchain: ...

@dataclass(frozen=True)
class ResolvedToolchain:
    """A concrete installation whose native build settings are prepared lazily."""

    root: Path
    identity: str
    environment_overrides: dict[str, str] = ...
    @property
    def lean(self) -> Path: ...
    @property
    def lake(self) -> Path: ...
    @property
    def leanc(self) -> Path: ...
    @property
    def leanchecker(self) -> Path: ...
    def environment(self, directory: Path | None = None) -> dict[str, str]: ...
    @property
    def native_flags(self) -> tuple[str, ...]: ...
    def link_args(self, *, python_extension: bool = False) -> list[str]: ...
    @property
    def native_identity(self) -> str: ...
