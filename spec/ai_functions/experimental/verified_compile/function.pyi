"""The Python-only interface of a verified native function."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

from strands.models import Model

from ..lean import LeanConfig

class _VerifiedFunction[**P, T]:
    def __init__(
        self,
        fn: Callable[P, T],
        *,
        pre_conditions: Sequence[Callable[..., object]] = (),
        post_conditions: Sequence[Callable[..., object]] = (),
        model: Model | str | None = None,
        max_attempts: int = 10,
        compile_timeout: float = 120,
        cache_dir: str | Path | None = None,
        lean_config: LeanConfig | None = None,
        offline: bool = False,
        check_pre_conditions: bool = False,
        check_post_conditions: bool = False,
        output_type: type[T] | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def is_compiled(self) -> bool: ...
    @property
    def artifact_dir(self) -> Path | None: ...
    async def compile(self) -> _VerifiedFunction[P, T]: ...
    def compile_sync(self) -> _VerifiedFunction[P, T]: ...
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

class _TypedDecorator[T]:
    def __init__(self, output_type: type[T]) -> None: ...
    @overload
    def __call__[**P](self, fn: Callable[P, T], /) -> _VerifiedFunction[P, T]: ...
    @overload
    def __call__(self, **kwargs: Any) -> Callable[[Callable[..., T]], _VerifiedFunction[..., T]]: ...

class _VerifiedFactory:
    def __getitem__[T](self, output_type: type[T]) -> _TypedDecorator[T]: ...
    @overload
    def __call__[**P, T](self, fn: Callable[P, T], /) -> _VerifiedFunction[P, T]: ...
    @overload
    def __call__[T](
        self,
        *,
        pre_conditions: Sequence[Callable[..., object]] = (),
        post_conditions: Sequence[Callable[..., object]] = (),
        model: Model | str | None = None,
        max_attempts: int = 10,
        compile_timeout: float = 120,
        cache_dir: str | Path | None = None,
        lean_config: LeanConfig | None = None,
        offline: bool = False,
        check_pre_conditions: bool = False,
        check_post_conditions: bool = False,
    ) -> Callable[[Callable[..., T]], _VerifiedFunction[..., T]]: ...

verified_ai_compile: _VerifiedFactory
