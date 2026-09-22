"""Failure-boundary checks that do not need an installed native toolchain."""

from __future__ import annotations

import json

import pytest

from ai_functions.experimental.verified_compile._runtime import require_supported_python
from ai_functions.experimental.verified_compile.compiler import read_artifact
from ai_functions.experimental.verified_compile.errors import CompilerError


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        None,
        {"module": [], "key": "k", "files": {}},
        {"module": "Verified" + "a" * 40, "key": "k", "files": []},
        {"module": "../escape", "key": "k", "files": {}},
    ],
)
def test_malformed_cache_metadata_is_a_cache_miss(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert read_artifact(tmp_path, None, "k") is None


def test_unsupported_python_is_rejected_only_by_the_new_feature(monkeypatch):
    import ai_functions
    from ai_functions.experimental.verified_compile import _runtime as compiler

    assert callable(ai_functions.ai_function)
    monkeypatch.setattr(compiler.sys, "version_info", (3, 11, 0))
    with pytest.raises(CompilerError, match="requires CPython 3.12"):
        require_supported_python()


@pytest.mark.parametrize("version", [(3, 12, 0), (3, 13, 0), (3, 14, 0)])
def test_supported_cpython_versions(version, monkeypatch):
    from ai_functions.experimental.verified_compile import _runtime as compiler

    monkeypatch.setattr(compiler.sys, "version_info", version)
    require_supported_python()


def test_free_threaded_python_is_rejected(monkeypatch):
    from ai_functions.experimental.verified_compile import _runtime as compiler

    monkeypatch.setattr(compiler.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(compiler.sysconfig, "get_config_var", lambda name: 1 if name == "Py_GIL_DISABLED" else None)
    with pytest.raises(CompilerError, match="GIL"):
        require_supported_python()
