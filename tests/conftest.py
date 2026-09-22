"""Shared fixtures for comprehensive unit tests.

This module provides common fixtures used across all test modules.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def lean_cache(tmp_path_factory):
    """Keep locally compiled bridges outside the user's persistent cache."""
    previous = os.environ.get("AI_FUNCTIONS_LEAN_CACHE_DIR")
    if previous is None:
        os.environ["AI_FUNCTIONS_LEAN_CACHE_DIR"] = str(tmp_path_factory.mktemp("lean-cache"))
    yield
    if previous is None:
        os.environ.pop("AI_FUNCTIONS_LEAN_CACHE_DIR", None)


@pytest.fixture(scope="session")
def native_runtime(lean_cache):
    """Run real Lean tests offline, optionally requiring native prerequisites."""
    from ai_functions.experimental.verified_compile._runtime import resolve_runtime
    from ai_functions.experimental.verified_compile.errors import CompilerError

    try:
        runtime = resolve_runtime(offline=True)
        runtime.preflight()
        return runtime
    except CompilerError:
        if os.environ.get("AI_FUNCTIONS_REQUIRE_VERIFIED_NATIVE"):
            raise
        pytest.skip("Prepare Lean with LeanConfig().setup() to run native integration tests")
