"""Lean projects: build directories, ordered registration, and symbol lookup.

A project is a Lake project's imports plus a prelude of registered declarations. The
source folder is only read; Lake runs in a persistent per-project build directory in
the user cache, which mirrors the source tree.

Invariants:
    V4, V6.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ..dsl.definitions import Definition
from ..dsl.types import LeanType
from .execution import LakeEnv
from .toolchain import LeanConfig

@dataclass(frozen=True)
class PreparedProject:
    """The fixed result of one successful ``LeanProject.prepare``.

    Attributes:
        lake: Lake environment of the build directory.
        imports: The configured imports; ``("Init",)`` for none.
        prelude: The registered blocks, concatenated in registration order.
        fixed_declarations: The declarations the prelude introduced that have a
            source range, as Lean reports them.
        fingerprint: Content identity of the prepared project.
    """

    lake: LakeEnv
    imports: tuple[str, ...]
    prelude: str
    fixed_declarations: tuple[str, ...]
    fingerprint: str

@dataclass(frozen=True)
class Parameter:
    """One binder of a declaration's type, as Lean reports it.

    Attributes:
        name: The binder name.
        type: The binder type as data; ``Unsupported`` outside the model.
        type_str: The binder type as Lean prints it.
        explicit: Whether the binder is explicit.
    """

    name: str
    type: LeanType
    type_str: str
    explicit: bool

@dataclass(frozen=True)
class SymbolInfo:
    """Lean's metadata for one declaration: the record Lean introspection returns.

    Attributes:
        name: The fully qualified name, as the symbol spells it.
        kind: ``def``, ``theorem``, ``axiom``, ``opaque``, ``inductive``, ``ctor``,
            ``recursor`` or ``quot``.
        module: The defining module; ``AIFunctions.Ledger`` for the prelude.
        source_modules: Project modules the declaration's type and value reach,
            the defining one included; used to select prompt source.
        type: The complete type as data; ``Unsupported`` outside the model.
        type_str: The complete type as Lean prints it.
        parameters: Every leading binder of the type, in order.
        result: The type after the binders; ``None`` when it is a ``Prop``.
        result_str: That type as Lean prints it.
        simple: No universe parameters, and no binder or result depends on an
            earlier binder.

    Immutable:
        Yes.
    """

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
        ...

    @property
    def prop(self) -> bool:
        """Whether the full application is a ``Prop``."""
        ...

@dataclass(frozen=True)
class LeanSymbol:
    """A fully qualified Lean name in one project, with metadata read from Lean.

    Constructing or traversing a symbol runs no Lean and registers nothing; whether
    the name exists is checked when metadata is first read. Two symbols are equal
    iff they name the same declaration of the same project.

    Attributes:
        project: The owning project.
        name: The fully qualified Lean name.
    """

    project: LeanProject
    name: str

    def __post_init__(self) -> None:
        """Validate ``name``.

        Raises:
            ValueError: ``name`` is not a Lean name.
        """
        ...

    @property
    def info(self) -> SymbolInfo:
        """Lean's metadata for this declaration.

        ``LeanProject.describe`` introspects several symbols in one Lean run; this
        property reads the same cache. It is the symbol's only attribute besides
        ``project`` and ``name``, so every other attribute is a child name; reach a
        child named ``info`` as ``symbol["info"]``.

        Ensures:
            The project is prepared.

        Raises:
            LeanError: No declaration has this name.
        """
        ...

    def __getattr__(self, name: str) -> LeanSymbol:
        """Return the symbol ``<self.name>.<name>``; runs no Lean."""
        ...

    def __getitem__(self, name: str) -> LeanSymbol:
        """Return ``<self.name>.<name>`` for a non-identifier component."""
        ...

class _Symbols:
    """Name lookup rooted at the project's top-level namespace."""

    def __init__(self, project: LeanProject) -> None: ...
    def __getattr__(self, name: str) -> LeanSymbol: ...
    def __getitem__(self, qualified_name: str) -> LeanSymbol: ...

class LeanProject:
    """A Lean environment: a Lake project's imports plus a registered prelude.

    Registration happens at application setup, in source order; the first
    ``prepare`` seals it. Consumers (``verified.ai_function``, ``verified.ai_compile``) read the
    prepared project and never register.

    Lifecycle:
        OPEN → PREPARED; a failed preparation leaves it SEALED, and ``prepare``
        may be retried.

    Invariants:
        - The prelude is the ``options`` as ``set_option`` commands, then the
          registered blocks in registration order.
        - A DSL definition appears after every definition it references.
    """

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
        """Describe a project; performs no Lean, network or filesystem-writing work.

        Args:
            path: Source Lake project directory; ``None`` for core Lean only.
            imports: Modules to import; their own imports are available too.
            toolchain: Provisioning configuration; defaults to ``LeanConfig()``.
            lean_toolchain: Release pin when ``path`` is ``None``.
            offline: Forbid downloading tools and Lake dependencies.
            options: Lean options, such as ``{"maxRecDepth": 100000}``, set with
                ``set_option`` at the top of the prelude, and so in every certificate.

        Raises:
            ValueError: ``path`` is given without ``imports``, a module name is
                not a Lean name, or an option name is not a Lean name or its value
                is neither a ``bool`` nor a nonnegative ``int``.
            LeanSetupError: ``path`` lacks a lakefile or ``lean-toolchain``, or
                ``lean_toolchain`` differs from the project's pin.
        """
        ...

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
        """Open a project shipped as data inside an importable Python package.

        Args:
            package: Python import name, e.g. ``"my_app"``.
            resource: Directory relative to the package, e.g. ``"lean"``.

        Ensures:
            Installed package files are never modified.

        Raises:
            ValueError: ``resource`` is absolute or contains ``..``.
            LeanSetupError: The resource is not a Lake project.
        """
        ...

    # ── Registration ──

    def add(self, source: str) -> None:
        """Append a raw Lean block to the prelude.

        The block may declare any number of names; refer to them through
        ``symbols``. It may use the imports and earlier registrations.

        Ensures:
            - The dedented block is the prelude's last block.
            - Lean diagnostics for the block, at preparation, name the Python line
              that called ``add``.

        Raises:
            ValueError: ``source`` is empty after dedenting.
            LeanProjectStateError: Preparation has begun.

        Concurrency:
            Thread-safe; serialized with ``prepare``.
        """
        ...

    @overload
    def function(self, fn: Callable[..., Any], /) -> Definition: ...
    @overload
    def function(
        self, fn: None = None, /, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Definition]:
        """Translate an expression-bodied function to a Lean ``def`` and append it.

        Names in the body are resolved now, from the function's globals, closure
        and builtins.

        Args:
            fn: The function; usable bare or as ``@project.function(name=...)``.
            name: The Lean name; defaults to ``__qualname__``.

        Requires:
            Every DSL definition the body calls is a ``Definition`` of this project.

        Ensures:
            The returned definition's ``source`` is the prelude's last block.

        Raises:
            TranslationError: The body leaves the DSL; a name is unbound (e.g. a
                helper defined further down); a referenced definition belongs to
                another project; the function is nested and ``name`` is missing; or
                the name is registered already or lies under ``H``, ``A``, ``J`` or
                ``AIFunctions``.
            LeanProjectStateError: Preparation has begun.
        """
        ...

    @overload
    def proposition(self, fn: Callable[..., Any], /) -> Definition: ...
    @overload
    def proposition(
        self, fn: None = None, /, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Definition]:
        """Translate a proposition block to a ``Prop``-valued ``def`` and append it.

        Same requirements and errors as ``function``; the body may use DSL statements.
        A contract is a proposition whose first parameter is the result, e.g.
        ``check(result: int, x: int) -> bool``.
        """
        ...

    def external(
        self, symbol: LeanSymbol, *, prop: bool = False, runtime: bool = True
    ) -> Callable[[Callable[..., Any]], Definition]:
        """Adapt an existing Lean declaration for calls from DSL bodies.

        The decorated function gives the Python signature and, with ``runtime``,
        the Python evaluation. Nothing is appended.

        Args:
            symbol: A declaration of this project.
            prop: The declaration is a ``Prop``; the Python function returns ``bool``.
            runtime: Keep the Python function as the evaluation.

        Raises:
            TypeError: ``symbol`` belongs to another project.
            LeanError: At preparation, the signature does not match the symbol's type.
        """
        ...

    # ── Lookup and preparation ──

    @property
    def symbols(self) -> _Symbols:
        """Lookup as ``symbols.Shop.Contract`` or ``symbols["Shop.Contract"]``."""
        ...

    def prepare(self, *, timeout: float | None = None) -> PreparedProject:
        """Seal registration, build the imports, and elaborate the prelude.

        Consumers call it at first use; calling it explicitly fails early once all
        decorators have run. Downloads happen here unless ``offline``, never earlier.

        Args:
            timeout: Bound on setup and builds together; defaults to 900 seconds.

        Ensures:
            - Registration is sealed, whether or not preparation succeeds.
            - The build directory mirrors the current source tree.
            - The prelude elaborated once, as one block, over the imports.
            - Repeated calls return the same record without running Lean.

        Raises:
            LeanError: A block does not elaborate (naming its Python site); an
                adapter's signature does not match; or the imports or prelude
                declare a name under ``H``, ``A``, ``J`` or ``AIFunctions``.
            LeanSetupError: Tools, dependencies or imports cannot be prepared.
            LeanTimeoutError: ``timeout`` expired.

        Concurrency:
            - Serialized per instance; concurrent callers wait for one preparation.
            - Serialized per build directory across processes.
        """
        ...

    def describe(self, *symbols: LeanSymbol) -> tuple[SymbolInfo, ...]:
        """Return each symbol's ``SymbolInfo``, introspecting uncached ones in one Lean run.

        Ensures:
            The project is prepared.

        Raises:
            ValueError: A symbol belongs to another project.
            LeanError: A name has no declaration.
        """
        ...

    def source(self, *symbols: LeanSymbol) -> str:
        """Return application source for prompts and review.

        Args:
            symbols: Restrict to their defining modules and the project modules
                their types and bodies reference; all imported project modules
                when empty.

        Ensures:
            - Each module appears once, labelled; the prelude is not included
              (see ``declarations``).
            - Modules outside the source folder (Lean core, Lake dependencies)
              never appear.
            - The project is prepared.

        Raises:
            ValueError: A symbol belongs to another project.
        """
        ...

    @property
    def root(self) -> Path:
        """The build directory; performs no I/O."""
        ...

    @property
    def imports(self) -> tuple[str, ...]:
        """The configured imports."""
        ...

    @property
    def declarations(self) -> str:
        """The prelude; prepares the project."""
        ...

    @property
    def toolchain(self) -> LeanConfig:
        """The provisioning configuration."""
        ...

    @property
    def fingerprint(self) -> str:
        """The content identity computed by ``prepare``.

        Covers the toolchain identity, the imports, the prelude, the source folder's
        files other than ``.lake``, and ``lake-manifest.json``; never build outputs
        or fetched dependencies.

        Raises:
            LeanProjectStateError: ``prepare`` has not completed.
        """
        ...
