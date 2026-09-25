"""Lean projects: build directories, ordered registration, symbol lookup and introspection."""

from __future__ import annotations

import bisect
import hashlib
import importlib.resources
import re
import sys
import textwrap
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ..dsl.definitions import Definition
from ..dsl.errors import TranslationError
from ..dsl.translate import adapt, site, translate
from ..dsl.types import LeanType
from . import _workspace
from ._support import SupportSession, build_support_executable
from .errors import LeanError, LeanProjectStateError, LeanSetupError
from .execution import Deadline, LakeEnv, file_lock, run_lake
from .toolchain import DEFAULT_LEAN_TOOLCHAIN, LeanConfig, cache_path
from .types import type_from_json

_IDENTIFIER = r"[^\W\d][\w']*"
_NAME_PART = rf"(?:{_IDENTIFIER}|«[^«»\n]+»)"
_NAME = re.compile(rf"{_NAME_PART}(?:\.{_NAME_PART})*\Z")
# Root namespaces owned by the harness: agent blocks, harness bindings,
# judgment axioms, and internals such as the ledger module.
_RESERVED = ("H", "A", "J", "AIFunctions")
_LEDGER_MODULE = "AIFunctions.Ledger"


def _check_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"Invalid Lean name: {name!r}")
    return name


def _set_options(options: Mapping[str, bool | int]) -> str:
    """The ``set_option`` commands that open the prelude."""
    lines = []
    for name, value in options.items():
        if type(value) is not bool and (type(value) is not int or value < 0):
            raise ValueError(f"Option {name!r} must be a bool or a nonnegative int, not {value!r}")
        lines.append(f"set_option {_check_name(name)} {str(value).lower()}")
    return "\n".join(lines)


def _name_parts(name: str) -> list[str]:
    """Split a Lean name into its components, unquoting quoted parts."""
    return [part[1:-1] if part.startswith("«") else part for part in re.findall(_NAME_PART, _check_name(name))]


def _render_name(parts: Sequence[str]) -> str:
    """Spell name components reported by Lean, quoting those that are not plain identifiers."""
    return ".".join(part if re.fullmatch(_IDENTIFIER, part) else f"«{part}»" for part in parts)


@dataclass(frozen=True)
class _Block:
    source: str
    site: str


@dataclass(frozen=True)
class PreparedProject:
    """The fixed result of one successful ``LeanProject.prepare``."""

    lake: LakeEnv
    imports: tuple[str, ...]
    prelude: str
    fixed_declarations: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class Parameter:
    """One binder of a declaration's type, as Lean reports it."""

    name: str
    type: LeanType
    type_str: str
    explicit: bool


@dataclass(frozen=True)
class SymbolInfo:
    """Lean's metadata for one declaration."""

    name: str
    kind: str
    module: str
    source_modules: tuple[str, ...]
    type: LeanType
    type_str: str
    parameters: tuple[Parameter, ...]
    result: LeanType | None
    result_str: str
    simple: bool

    @property
    def explicit(self) -> tuple[Parameter, ...]:
        """The explicit parameters, in order."""
        return tuple(p for p in self.parameters if p.explicit)

    @property
    def prop(self) -> bool:
        """Whether the full application is a ``Prop``."""
        return self.result is None

    @classmethod
    def _from_json(cls, data: Mapping[str, Any]) -> SymbolInfo:
        parameters = tuple(
            Parameter(p["name"], type_from_json(p["type"], p["type_str"]), p["type_str"], p["explicit"])
            for p in data["params"]
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            module=data["module"],
            source_modules=tuple(data["source_modules"]),
            type=type_from_json(data["type"], data["type_str"]),
            type_str=data["type_str"],
            parameters=parameters,
            result=None if data["prop"] else type_from_json(data["result"], data["result_str"]),
            result_str=data["result_str"],
            simple=data["simple"],
        )


@dataclass(frozen=True)
class LeanSymbol:
    """A fully qualified Lean name in one project; constructing it runs no Lean."""

    project: LeanProject
    name: str

    def __post_init__(self) -> None:
        """Validate ``name``."""
        _check_name(self.name)

    @property
    def info(self) -> SymbolInfo:
        """Lean's metadata for this declaration; prepares the project."""
        return self.project.describe(self)[0]

    def __getattr__(self, name: str) -> LeanSymbol:
        """Return the symbol ``<self.name>.<name>``."""
        if name.startswith("__"):
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, name: str) -> LeanSymbol:
        """Return ``<self.name>.<name>`` for a non-identifier component."""
        return LeanSymbol(self.project, f"{self.name}.{name}")


class _Symbols:
    """Name lookup rooted at the project's top-level namespace."""

    def __init__(self, project: LeanProject) -> None:
        self._project = project

    def __getattr__(self, name: str) -> LeanSymbol:
        if name.startswith("__"):
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, qualified_name: str) -> LeanSymbol:
        return LeanSymbol(self._project, qualified_name)


class LeanProject:
    """A Lean environment: a Lake project's imports plus a registered prelude."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        imports: Sequence[str] | None = None,
        toolchain: LeanConfig | None = None,
        lean_toolchain: str | None = None,
        offline: bool = False,
        options: Mapping[str, bool | int] | None = None,
    ) -> None:
        if path is not None and imports is None:
            raise ValueError("A project path needs imports=, the modules to import")
        self._imports = tuple(dict.fromkeys(_check_name(n) for n in imports or ())) or ("Init",)
        self._source = Path(path).expanduser().resolve() if path is not None else None
        self._toolchain = toolchain or LeanConfig()
        self._offline = offline
        self._options = _set_options(options or {})
        if self._source is None:
            self._pin = lean_toolchain or DEFAULT_LEAN_TOOLCHAIN
            self._config = {
                "lakefile.toml": b'name = "ai_functions_lean"\n',
                "lean-toolchain": f"{self._pin}\n".encode(),
            }
            key = f"core\0{self._pin}"
        else:
            self._config = {
                n: (self._source / n).read_bytes() for n in _workspace.CONFIG if (self._source / n).is_file()
            }
            if "lean-toolchain" not in self._config or not {"lakefile.toml", "lakefile.lean"} & set(self._config):
                raise LeanSetupError(f"{self._source} needs a lakefile.toml or lakefile.lean and a lean-toolchain")
            self._pin = self._config["lean-toolchain"].decode().strip()
            if lean_toolchain is not None and lean_toolchain != self._pin:
                raise LeanSetupError(f"Requested toolchain {lean_toolchain} differs from project pin {self._pin}")
            key = str(self._source)
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        self._root = cache_path(self._toolchain) / "lean" / "projects" / digest
        self._lock = threading.RLock()
        self._sealed = False
        self._prepared: PreparedProject | None = None
        self._modules: dict[str, Path] = {}
        self._blocks: list[_Block] = []
        self._adapters: list[Definition] = []
        self._symbols = _Symbols(self)
        self._info: dict[str, SymbolInfo] = {}

    @classmethod
    def from_package(
        cls,
        package: str,
        resource: str,
        *,
        imports: Sequence[str],
        toolchain: LeanConfig | None = None,
        offline: bool = False,
        options: Mapping[str, bool | int] | None = None,
    ) -> LeanProject:
        """Open a project shipped as data inside an importable Python package."""
        if Path(resource).is_absolute() or ".." in Path(resource).parts:
            raise ValueError("resource must be a package-relative directory")
        try:
            path = Path(str(importlib.resources.files(package).joinpath(resource)))
        except (OSError, ImportError) as exc:
            raise LeanSetupError(f"Could not load Lean project from {package!r}/{resource}: {exc}") from exc
        return cls(path, imports=imports, toolchain=toolchain, offline=offline, options=options)

    # ── Registration ──

    def _append(self, source: str, site: str) -> None:
        with self._lock:
            if self._sealed:
                raise LeanProjectStateError(f"{site}: register Lean before the project is prepared")
            self._blocks.append(_Block(source, site))

    def add(self, source: str) -> None:
        """Append a raw Lean block to the prelude."""
        source = textwrap.dedent(source).strip()
        if not source:
            raise ValueError("source must contain Lean")
        caller = sys._getframe(1)
        self._append(source, f"{caller.f_code.co_filename}:{caller.f_lineno}")

    @overload
    def function(self, fn: Callable[..., Any], /) -> Definition: ...
    @overload
    def function(
        self, fn: None = None, /, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Definition]: ...
    def function(self, fn: Callable[..., Any] | None = None, /, *, name: str | None = None) -> Any:
        """Translate an expression-bodied function to a Lean ``def`` and append it."""
        return self._decorator(fn, name, prop=False)

    @overload
    def proposition(self, fn: Callable[..., Any], /) -> Definition: ...
    @overload
    def proposition(
        self, fn: None = None, /, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Definition]: ...
    def proposition(self, fn: Callable[..., Any] | None = None, /, *, name: str | None = None) -> Any:
        """Translate a proposition block to a ``Prop``-valued ``def`` and append it."""
        return self._decorator(fn, name, prop=True)

    def _decorator(self, fn: Callable[..., Any] | None, name: str | None, *, prop: bool) -> Any:
        def register(fn: Callable[..., Any]) -> Definition:
            lean_name = name
            if lean_name is None:
                if "<locals>" in fn.__qualname__:
                    raise TranslationError(f"{site(fn)}: {fn.__qualname__} is nested in a function; pass name=")
                lean_name = fn.__qualname__
            try:
                root = _name_parts(lean_name)[0]
            except ValueError as exc:
                raise TranslationError(f"{site(fn)}: {exc}") from exc
            if root in _RESERVED:
                raise TranslationError(f"{site(fn)}: {lean_name} lies under {', '.join(_RESERVED)}")
            with self._lock:
                definition = translate(fn, self.symbols[lean_name], prop=prop)
                self._append(definition.source, site(fn))
                return definition

        return register if fn is None else register(fn)

    def external(
        self, symbol: LeanSymbol, *, prop: bool = False, runtime: bool = True
    ) -> Callable[[Callable[..., Any]], Definition]:
        """Adapt an existing Lean declaration for calls from DSL bodies; appends nothing."""
        if not isinstance(symbol, LeanSymbol) or symbol.project is not self:
            raise TypeError("external requires a LeanSymbol of this project")

        def register(fn: Callable[..., Any]) -> Definition:
            definition = adapt(symbol, fn, prop=prop, runtime=runtime)
            with self._lock:
                self._adapters.append(definition)
            return definition

        return register

    # ── Lookup and preparation ──

    @property
    def symbols(self) -> _Symbols:
        """Lookup as ``symbols.Shop.Contract`` or ``symbols["Shop.Contract"]``."""
        return self._symbols

    @property
    def root(self) -> Path:
        """The build directory; performs no I/O."""
        return self._root

    @property
    def imports(self) -> tuple[str, ...]:
        """The configured imports."""
        return self._imports

    @property
    def declarations(self) -> str:
        """The prelude; prepares the project."""
        return self.prepare().prelude

    @property
    def toolchain(self) -> LeanConfig:
        """The provisioning configuration."""
        return self._toolchain

    @property
    def fingerprint(self) -> str:
        """The content identity computed by ``prepare``."""
        prepared = self._prepared
        if prepared is None:
            raise LeanProjectStateError("Call prepare() before reading the project fingerprint")
        return prepared.fingerprint

    def prepare(self, *, timeout: float | None = None) -> PreparedProject:
        """Seal registration, build the imports, and elaborate the prelude once, as one block."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        prepared = self._prepared
        if prepared is not None:
            return prepared
        with self._lock:
            if self._prepared is not None:
                return self._prepared
            self._sealed = True
            remaining = Deadline(timeout or 900.0, "Project preparation exceeded its deadline").remaining
            tools = self.toolchain.setup(self._pin, offline=self._offline, timeout=remaining())
            lake = LakeEnv(tools, self.root, self._offline)
            core = lake.tools.root / "lib" / "lean"
            targets = [
                "+" + n for n in self._imports if not core.joinpath(*n.split(".")).with_suffix(".olean").exists()
            ]
            prelude = "\n\n".join([self._options] * bool(self._options) + [block.source for block in self._blocks])
            prelude += "\n" if prelude else ""
            with file_lock(self.root.with_suffix(".lock"), timeout=remaining()):
                _workspace.sync(self._source, self.root, self._config)
                if targets:
                    built = run_lake(lake, ["build", *targets], timeout=remaining(), check=False)
                    if built.returncode:
                        raise LeanSetupError(
                            f"Could not build {', '.join(self._imports)}:\n{built.stdout}{built.stderr}"
                        )
                if self._source is not None:
                    self._modules = _workspace.modules(self._source, self.root)
                identity = _workspace.fingerprint(
                    self._source, self.root, identity=lake.tools.identity, imports=self._imports, prelude=prelude
                )
            with self._server(lake, timeout=remaining()) as session:
                reply = session.request({"op": "elab", "source": prelude, "commit": True})
                if not reply.get("ok"):
                    raise self._elaboration_error(reply)
                declared = tuple(_render_name(parts) for parts in reply.get("declared", []))
                adapters = [d.symbol.name for d in self._adapters]
                info = self._introspect(session, [*declared, *adapters])
            for definition in self._adapters:
                if info[definition.symbol.name].type != definition.type:
                    raise LeanError(
                        f"external {definition.__qualname__} does not match {definition.symbol.name}'s type"
                    )
            prepared = PreparedProject(lake, self._imports, prelude, declared, identity)
            self._info.update(info)
            self._prepared = prepared
            return prepared

    def _elaboration_error(self, reply: Mapping[str, Any]) -> LeanError:
        """Attribute each Lean error to the Python site that registered its block."""
        starts, line = [], self._options.count("\n") + 3 if self._options else 1
        for block in self._blocks:
            starts.append(line)
            line += block.source.count("\n") + 2
        details = []
        for message in reply.get("messages", []):
            if message.get("severity") == "error":
                index = max(bisect.bisect_right(starts, message.get("line", 1)) - 1, 0)
                block_line = message.get("line", 1) - starts[index] + 1
                details.append(f"{self._blocks[index].site} (block line {block_line}): {message.get('text', '')}")
        return LeanError("The registered Lean does not elaborate:\n" + ("\n".join(details) or str(reply)))

    @contextmanager
    def _server(self, lake: LakeEnv, *, source: str = "", timeout: float) -> Iterator[SupportSession]:
        """Start one short-lived server over the imports, elaborating ``source`` first."""
        remaining = Deadline(timeout, "Preparing the project exceeded its deadline").remaining
        executable = build_support_executable(tools=lake.tools, cache=cache_path(self.toolchain), timeout=remaining())
        with SupportSession(lake, executable, cwd=self.root / _workspace.SCRATCH, timeout=remaining()) as session:
            init = {"op": "init", "imports": list(self._imports), "source": source, "module": _LEDGER_MODULE}
            reply = session.request(init)
            if not reply.get("ok"):
                errors = "\n".join(m.get("text", "") for m in reply.get("messages", []))
                raise LeanError(f"Preparing the project failed:\n{errors or reply}")
            yield session

    def _introspect(self, session: SupportSession, names: Sequence[str]) -> dict[str, SymbolInfo]:
        owned = [*self._modules, _LEDGER_MODULE]
        request = {"op": "describe", "names": [[n, _name_parts(n)] for n in dict.fromkeys(names)], "owned": owned}
        reply = session.request(request)
        if not reply.get("ok"):
            raise LeanError(str(reply.get("error", "Introspection failed")))
        return {data["name"]: SymbolInfo._from_json(data) for data in reply["symbols"]}

    def describe(self, *symbols: LeanSymbol) -> tuple[SymbolInfo, ...]:
        """Return each symbol's ``SymbolInfo``, introspecting uncached ones in one Lean run."""
        if any(symbol.project is not self for symbol in symbols):
            raise ValueError("Described symbols must belong to this project")
        prepared = self.prepare()
        with self._lock:
            missing = [s.name for s in symbols if s.name not in self._info]
            if missing:
                with self._server(prepared.lake, source=prepared.prelude, timeout=120.0) as session:
                    self._info.update(self._introspect(session, missing))
            return tuple(self._info[s.name] for s in symbols)

    def source(self, *symbols: LeanSymbol) -> str:
        """Return the source of the project modules the symbols reach, all imported ones when none."""
        if any(symbol.project is not self for symbol in symbols):
            raise ValueError("Requested source symbols must belong to this project")
        prepared = self.prepare()
        if symbols:
            selected = {m for info in self.describe(*symbols) for m in (info.module, *info.source_modules)}
        else:
            selected = set(prepared.imports)
        modules = sorted(selected & set(self._modules))
        return "\n\n".join(f"-- Module: {name}\n{self._modules[name].read_text()}" for name in modules)
