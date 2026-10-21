"""A recorded workflow run, as the ``aif-workflow`` runtime actually writes it.

Field names mirror the library's own events exactly — ``tool_name`` /
``tool_use_id`` on a call, no tool name at all on a result, ``token_usage`` rather
than ``usage``, ``payload`` on a result event. Guessing them is how a parser ends
up reading zeros while claiming success, so these fixtures are the contract.

Shared by the reader tests and the ``uv run python graphs.py view`` TUI tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST: dict[str, Any] = {
    "runtime_version": 4,
    "slug": "demo-slug",
    "task": "add a --json flag",
    "objective": "serve_cli.py accepts --json and emits parseable JSON",
    "workspace": "/tmp/workspace",
    "backend": "kiro",
    "steps": [
        {
            "name": "implement",
            "prompt": "add the flag",
            "thread_id": "th-implement",
            "backend": "kiro",
            "input_type": "str",
            "output_type": "str",
            "pre_conditions": [
                {"name": "dir_exists", "kind": "code", "description": "target dir exists"},
            ],
            "post_conditions": [
                {
                    "name": "flag_present",
                    "kind": "code",
                    "description": "flag is present",
                    "signature": "flag_present(result: str) -> PostConditionResult | None",
                    "source": "def flag_present(result: str) -> PostConditionResult | None:\n    ...",
                },
            ],
            "depends_on": [],
            "max_attempts": 4,
        },
        {
            "name": "verify",
            "prompt": "run tests",
            "thread_id": "th-verify",
            "backend": "claude",
            "pre_conditions": [],
            "post_conditions": [],
            "depends_on": ["implement"],
            "max_attempts": 4,
        },
    ],
}

RUN_META: dict[str, Any] = {
    "runtime_version": 4,
    "run_id": "20260915-120000-abc123",
    "slug": "demo-slug",
    "task": "add a --json flag",
    "session": "terminal ttys001",
    "started_at": 1000.0,
    "finished_at": 1005.0,
    "status": "failed",
    "backend": "kiro",
}


def events() -> list[dict[str, Any]]:
    """Return a run that fails a post-condition once, then passes on the retry."""
    return [
        {
            "kind": "aif_precondition",
            "thread_id": "th-implement",
            "thread_name": "implement",
            "timestamp": 1.0,
            "condition": "dir_exists",
            "passed": True,
            "step": "implement",
        },
        {"kind": "started", "thread_id": "th-implement", "thread_name": "implement", "timestamp": 1.1},
        {"kind": "message_user", "thread_id": "th-implement", "timestamp": 1.2, "text": "add the flag"},
        {
            "kind": "tool_call",
            "thread_id": "th-implement",
            "timestamp": 2.0,
            "tool_name": "fs_write",
            "tool_use_id": "tu-1",
            "arguments": {"path": "cli.py"},
            "message_id": "msg-1",
        },
        {
            "kind": "tool_result",
            "thread_id": "th-implement",
            "timestamp": 2.4,
            "tool_use_id": "tu-1",
            "status": "success",
            "content": [{"text": "ok"}],
            "message_id": "msg-1",
        },
        {"kind": "message_assistant_token", "thread_id": "th-implement", "timestamp": 2.5, "text": "wro"},
        {
            "kind": "message_assistant_complete",
            "thread_id": "th-implement",
            "timestamp": 2.6,
            "content": [{"text": "wrote the flag"}],
            "message_id": "msg-1",
        },
        {
            "kind": "aif_postcondition",
            "thread_id": "th-implement",
            "timestamp": 2.7,
            "condition": "flag_present",
            "condition_kind": "code",
            "attempt": 1,
            "passed": False,
            "message": "no --json flag; add it",
            "step": "implement",
        },
        {
            "kind": "aif_postcondition",
            "thread_id": "th-implement",
            "timestamp": 3.5,
            "condition": "flag_present",
            "condition_kind": "code",
            "attempt": 2,
            "passed": True,
            "step": "implement",
        },
        {
            "kind": "token_usage",
            "thread_id": "th-implement",
            "timestamp": 3.6,
            "token_usage": {"input_tokens": 120, "output_tokens": 45},
        },
        {"kind": "result", "thread_id": "th-implement", "timestamp": 3.65, "payload": "added --json"},
        {"kind": "completed", "thread_id": "th-implement", "timestamp": 3.7},
        {"kind": "started", "thread_id": "th-verify", "thread_name": "verify", "timestamp": 4.0},
        {
            "kind": "message_user",
            "thread_id": "th-verify",
            "timestamp": 4.1,
            "text": "[verify] Post-condition failures (attempt 1/3):\n- pytest failed: 1 error",
        },
        {"kind": "failed", "thread_id": "th-verify", "timestamp": 4.9},
    ]


def write_run(
    root: Path,
    *,
    slug: str = "demo-slug",
    run_id: str = "20260915-120000-abc123",
    session: str = "terminal ttys001",
    status: str = "failed",
    lines: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Write one run to disk in the layout the runtime produces; return its run directory."""
    package = root / slug
    package.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps({**(manifest or MANIFEST), "slug": slug})
    (package / "manifest.json").write_text(snapshot, encoding="utf-8")
    run_dir = package / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(snapshot, encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({**RUN_META, "run_id": run_id, "slug": slug, "session": session, "status": status}),
        encoding="utf-8",
    )
    payload = events() if lines is None else lines
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in payload),
        encoding="utf-8",
    )
    return run_dir
