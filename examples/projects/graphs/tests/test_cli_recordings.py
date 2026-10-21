"""Parsing and aggregation: recorded workflow events on disk to the board's thread cards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recording_fixtures import MANIFEST, RUN_META, events, write_run

from recordings import (
    RUNTIME_VERSION,
    Board,
    ConditionCell,
    ConditionSpec,
    Manifest,
    RunMeta,
    RunView,
    StepSpec,
    ThreadView,
    default_root,
    parse_run,
    read_events,
    thread_cards,
)


def _steps() -> dict[str, StepSpec]:
    """Manifest steps keyed by name, as the board passes them in."""
    return Manifest.model_validate(MANIFEST).steps_by_name


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An `.aif`-shaped tree in a temp directory."""
    return tmp_path / ".aif"


# ── Schema ────────────────────────────────────────────────────────────────────


def test_schema_tolerates_unknown_keys_and_missing_fields() -> None:
    """A newer runtime's extra keys and an older runtime's missing keys both validate."""
    meta = RunMeta.model_validate({"run_id": "r", "backend": "codex", "brand_new": 1})
    assert meta.run_id == "r"
    assert meta.status == "running"
    assert meta.backend == "codex"
    assert RunMeta().backend == ""  # runs written before backends existed
    assert Manifest.model_validate({}).steps == []


def test_stale_runtime_is_flagged_below_the_reader_version() -> None:
    """A run written by an older runtime is marked, not rejected."""
    assert RunMeta(runtime_version=RUNTIME_VERSION - 1).stale_runtime is True
    assert RunMeta(runtime_version=RUNTIME_VERSION).stale_runtime is False


def test_default_root_honours_aif_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AIF_HOME overrides the ./.aif default."""
    monkeypatch.setenv("AIF_HOME", str(tmp_path / "elsewhere"))
    assert default_root() == (tmp_path / "elsewhere").resolve()
    monkeypatch.delenv("AIF_HOME")
    monkeypatch.chdir(tmp_path)
    assert default_root() == tmp_path / ".aif"


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parse_builds_one_lane_per_thread() -> None:
    """Each thread becomes a lane with its own timeline and conditions."""
    view = parse_run("run-x", events())
    assert [t.name for t in view.threads] == ["implement", "verify"]
    implement = view.threads[0]
    assert implement.status == "passed"
    assert implement.tool_calls == 1
    assert implement.attempts == 2


def test_parse_reads_the_librarys_own_field_names() -> None:
    """Usage, result payload, and tool naming follow the library's real schema."""
    view = parse_run("run-x", events())
    implement = view.threads[0]
    assert view.usage_reported is True
    assert (implement.input_tokens, implement.output_tokens) == (120, 45)
    assert implement.result == "added --json"
    labels = {e.kind: e.label for e in implement.timeline}
    assert labels["tool_call"] == "fs_write"
    assert labels["tool_result"] == "fs_write"  # A result names no tool; it is paired by tool_use_id.
    details = {e.kind: e.detail for e in implement.timeline}
    assert details["tool_result"] == "ok"  # Content blocks are flattened, not repr'd.
    assert details["assistant"] == "wrote the flag"  # ``content`` blocks, no ``text`` field.


def test_parse_unwraps_kiros_own_content_shape() -> None:
    """Live Kiro results arrive as {"items": [{"Text": ...}]} — capital T, one level in."""
    kiro_shaped = [
        {"kind": "tool_call", "thread_id": "th-1", "timestamp": 1.0, "tool_name": "fs_read", "tool_use_id": "tu-9"},
        {
            "kind": "tool_result",
            "thread_id": "th-1",
            "timestamp": 1.1,
            "tool_use_id": "tu-9",
            "status": "success",
            "content": [{"text": '{"items": [{"Text": "date,region\\n2026-01-04,north"}]}'}],
        },
    ]
    (thread,) = parse_run("run-kiro", kiro_shaped).threads
    detail = next(e.detail for e in thread.timeline if e.kind == "tool_result")
    assert detail == "date,region\n2026-01-04,north"


def test_parse_leaves_plain_text_results_alone() -> None:
    """A result that is already prose is not mangled by the JSON unwrapping."""
    plain = [
        {"kind": "tool_call", "thread_id": "th-1", "timestamp": 1.0, "tool_name": "bash", "tool_use_id": "tu-1"},
        {
            "kind": "tool_result",
            "thread_id": "th-1",
            "timestamp": 1.1,
            "tool_use_id": "tu-1",
            "status": "success",
            "content": [{"text": "rows: 10"}],
        },
    ]
    (thread,) = parse_run("run-plain", plain).threads
    assert next(e.detail for e in thread.timeline if e.kind == "tool_result") == "rows: 10"


def test_parse_collapses_assistant_fragments() -> None:
    """Streaming fragments are dropped in favour of the completed turn."""
    kinds = [entry.kind for entry in parse_run("run-x", events()).threads[0].timeline]
    assert kinds.count("assistant") == 1
    assert "message_assistant_token" not in kinds


def test_parse_records_every_condition_evaluation() -> None:
    """Post-conditions land as one cell per condition per attempt."""
    implement = parse_run("run-x", events()).threads[0]
    cells = {(c.condition, c.attempt): c for c in implement.post_conditions}
    assert cells[("flag_present", 1)].passed is False
    assert cells[("flag_present", 1)].message == "no --json flag; add it"
    assert cells[("flag_present", 2)].passed is True
    assert [c.condition for c in implement.pre_conditions] == ["dir_exists"]


def test_parse_recovers_conditions_from_native_feedback() -> None:
    """A failure carried only by the library's feedback turn is still shown."""
    verify = parse_run("run-x", events()).threads[1]
    (cell,) = verify.post_conditions
    assert (cell.inferred, cell.attempt) == (True, 1)
    assert "pytest failed" in (cell.message or "")


def test_the_feedback_fallback_yields_to_real_condition_events() -> None:
    """A wrapped condition is not double-counted via its own feedback turn."""
    with_feedback = [
        *events()[:9],
        {
            "kind": "message_user",
            "thread_id": "th-implement",
            "timestamp": 3.0,
            "text": "[implement] Post-condition failures (attempt 1/4):\n- no --json flag; add it",
        },
    ]
    implement = parse_run("run-both", with_feedback).threads[0]
    assert [c.condition for c in implement.post_conditions] == ["flag_present", "flag_present"]
    assert not any(c.inferred for c in implement.post_conditions)


def test_events_without_a_thread_do_not_become_a_phantom_thread() -> None:
    """Claude Code's system messages carry no thread id; they must not appear as a node."""
    chatter = [
        {"kind": "claude_system_init", "subtype": "init", "timestamp": 0.5},
        *events(),
        {"kind": "claude_system_commands_changed", "subtype": "commands_changed", "timestamp": 4.95},
    ]
    view = parse_run("run-claude", chatter)
    assert [t.name for t in view.threads] == ["implement", "verify"]
    assert view.event_count == len(chatter)  # still counted, just not attributed


def test_parse_tolerates_unknown_event_kinds() -> None:
    """An unrecognized kind degrades to a timeline entry instead of raising."""
    view = parse_run("run-z", [{"kind": "codex_plan", "thread_id": "th-1", "timestamp": 1.0}])
    assert [e.kind for e in view.threads[0].timeline] == ["codex_plan"]


def test_missing_usage_is_reported_as_absent() -> None:
    """Kiro may omit usage entirely; the viewer must say so rather than show zeros."""
    without = [e for e in events() if e["kind"] != "token_usage"]
    assert parse_run("run-y", without).usage_reported is False


def test_tool_activity_pairs_calls_with_their_results() -> None:
    """A call, its arguments, and its output are one unit."""
    (thread,) = parse_run("run-tools", events()).threads[:1]
    (activity,) = thread.tools
    assert activity.tool == "fs_write"
    assert activity.status == "success"
    assert activity.output == "ok"
    assert activity.arguments == {"path": "cli.py"}
    assert activity.duration == 0.4


def test_tool_purpose_is_lifted_out_of_the_arguments() -> None:
    """Kiro's __tool_use_purpose leads the card instead of padding the arguments."""
    with_purpose = [
        {
            "kind": "tool_call",
            "thread_id": "th-1",
            "timestamp": 1.0,
            "tool_name": "fs_read",
            "tool_use_id": "tu-1",
            "arguments": {"__tool_use_purpose": "Check the CSV header.", "path": "data.csv"},
        },
    ]
    (thread,) = parse_run("run-purpose", with_purpose).threads
    (activity,) = thread.tools
    assert activity.purpose == "Check the CSV header."
    assert activity.arguments == {"path": "data.csv"}
    assert activity.status == "pending"


def test_a_json_result_block_renders_as_json_not_a_repr() -> None:
    """Kiro wraps command output in {"Json": {...}}; a Python repr is unreadable."""
    command_result = [
        {"kind": "tool_call", "thread_id": "th-1", "timestamp": 1.0, "tool_name": "bash", "tool_use_id": "tu-1"},
        {
            "kind": "tool_result",
            "thread_id": "th-1",
            "timestamp": 1.2,
            "tool_use_id": "tu-1",
            "status": "success",
            "content": [{"text": '{"Json": {"exit_status": "0", "stdout": "rows: 10"}}'}],
        },
    ]
    (thread,) = parse_run("run-json", command_result).threads
    output = thread.tools[0].output
    assert '"exit_status": "0"' in output
    assert "'exit_status'" not in output


# ── Tickets ───────────────────────────────────────────────────────────────────


def test_tickets_preserve_observed_order_metadata_and_pending_declarations() -> None:
    spec = StepSpec(
        name="step",
        pre_conditions=[ConditionSpec(name="input")],
        post_conditions=[
            ConditionSpec(name="pending", kind="ai", source="pending source"),
            ConditionSpec(name="checked", kind="code", source="checked source"),
        ],
    )
    thread = ThreadView(
        thread_id="th-step",
        name="step",
        pre_conditions=[ConditionCell(condition="input", attempt=0, passed=True)],
        post_conditions=[
            ConditionCell(condition="checked", attempt=2, timestamp=1, passed=False, message="old failure"),
            ConditionCell(condition="undeclared", attempt=1, passed=False, message="failure", inferred=True),
            ConditionCell(condition="checked", attempt=2, timestamp=2, passed=True, message="stale message"),
        ],
    )
    (card,) = thread_cards(RunView(run_id="run", threads=[thread]), slug="demo", steps={"step": spec})
    assert [(t.stage, t.name, t.attempts) for t in card.tickets] == [
        ("pre", "input", 1),
        ("post", "checked", 2),
        ("post", "undeclared", 1),
        ("post", "pending", 0),
    ]
    _, checked, undeclared, pending = card.tickets
    assert (checked.passed, checked.message, checked.source) == (True, None, "checked source")
    assert (undeclared.inferred, undeclared.message, undeclared.source) == (True, "failure", "")
    assert (pending.kind, pending.message, pending.source) == ("ai", "not evaluated", "pending source")


def test_tickets_carry_the_latest_verdict_and_attempt_count() -> None:
    """Each condition collapses to one ticket holding its most recent verdict."""
    cards = thread_cards(parse_run("run-x", events()), slug="demo-slug", steps=_steps())
    tickets = {t.name: t for t in next(c for c in cards if c.name == "implement").tickets}
    assert (tickets["flag_present"].passed, tickets["flag_present"].attempts) == (True, 2)
    assert tickets["flag_present"].description == "flag is present"
    assert tickets["flag_present"].source.startswith("def flag_present")
    assert tickets["dir_exists"].stage == "pre"
    by_name = {c.name: c for c in cards}
    assert (by_name["implement"].backend, by_name["verify"].backend) == ("kiro", "claude")


def test_a_thread_with_a_failing_ticket_is_not_verified() -> None:
    """Verification is all-or-nothing across a thread's tickets."""
    cards = thread_cards(parse_run("run-x", events()), slug="demo-slug", steps=_steps())
    verify = next(c for c in cards if c.name == "verify")
    assert verify.verified is False
    assert verify.tickets[0].inferred is True


def test_declared_but_unevaluated_conditions_show_as_pending() -> None:
    """A blocked step still shows what it would have checked."""
    started = [{"kind": "started", "thread_id": "th-implement", "thread_name": "implement", "timestamp": 1.0}]
    cards = thread_cards(parse_run("run-b", started), slug="demo-slug", steps=_steps())
    tickets = {t.name: t for t in cards[0].tickets}
    assert (tickets["flag_present"].attempts, tickets["flag_present"].message) == (0, "not evaluated")
    assert cards[0].verified is False


# ── Discovery ─────────────────────────────────────────────────────────────────


def test_board_discovers_runs_by_walking_the_tree(root: Path) -> None:
    """Runs are found from the filesystem alone — no index to keep in sync."""
    write_run(root, run_id="20260915-120000-aaaaaa")
    write_run(root, slug="other-slug", run_id="20260915-130000-bbbbbb", session="terminal ttys002")
    board = Board(root=root)
    assert {r.meta.run_id for r in board.runs()} == {"20260915-120000-aaaaaa", "20260915-130000-bbbbbb"}
    assert {c.session for c in board.cards()} == {"terminal ttys001", "terminal ttys002"}
    assert len(board.cards()) == 4  # Two threads per run.


def test_board_on_a_missing_folder_is_empty_not_an_error(tmp_path: Path) -> None:
    """Pointing the reader at a folder with no runs yet is a normal state."""
    assert Board(root=tmp_path / "nothing-here").workflows() == []


def test_legacy_runs_fall_back_to_the_package_manifest(root: Path) -> None:
    """Version 3 recordings without snapshots retain their original reader behavior."""
    run_dir = write_run(root)
    (run_dir / "manifest.json").unlink()
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "runtime_version": 3}), encoding="utf-8")
    (run,) = Board(root=root).runs()
    assert run.manifest.objective == MANIFEST["objective"]
    assert len(Board(root=root).cards()) == 2


@pytest.mark.parametrize("snapshot", [None, "{unfinished"])
def test_missing_or_corrupt_new_snapshots_do_not_borrow_a_different_graph(root: Path, snapshot: str | None) -> None:
    """An unavailable snapshot cannot be replaced with a possibly newer package graph."""
    run_dir = write_run(root)
    manifest_path = run_dir / "manifest.json"
    if snapshot is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(snapshot, encoding="utf-8")
    (run,) = Board(root=root).runs()
    assert run.manifest.steps == []
    assert run.manifest.objective == ""
    assert Board(root=root).cards()  # recorded events remain readable


def test_board_reuses_its_cache_until_a_log_grows(root: Path) -> None:
    """An unchanged log is not re-parsed; an appended one is."""
    run_dir = write_run(root)
    board = Board(root=root)
    board.cards()
    cached = board._cache[RUN_META["run_id"]]
    board.cards()
    assert board._cache[RUN_META["run_id"]] is cached

    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "started", "thread_id": "th-third", "timestamp": 9.0}) + "\n")
    board.cards()
    assert board._cache[RUN_META["run_id"]] is not cached


def test_board_marks_live_threads_running(root: Path) -> None:
    """A thread with no terminal event in a running run reads as live."""
    write_run(root, status="running", lines=events()[:4])
    implement, verify = sorted(Board(root=root).cards(), key=lambda c: c.name)
    assert implement.status == "running"
    assert verify.status == "pending"  # declared, not reached: drawn but not lit


@pytest.mark.parametrize("run_status", ["running", "failed"])
def test_a_step_stopped_by_a_pre_condition_is_blocked_not_running(root: Path, run_status: str) -> None:
    """A failed pre-condition means the agent never started, live run or not."""
    blocked = [{**events()[0], "passed": False, "message": "target dir is missing"}]
    write_run(root, status=run_status, lines=blocked)
    implement = next(card for card in Board(root=root).cards() if card.name == "implement")
    assert implement.status == "blocked"


def test_a_finished_run_marks_unreached_steps_skipped(root: Path) -> None:
    """Once the run is over, a declared step with no events was skipped, not pending."""
    blocked = [{**events()[0], "passed": False, "message": "target dir is missing"}]
    write_run(root, status="failed", lines=blocked)
    verify = next(card for card in Board(root=root).cards() if card.name == "verify")
    assert verify.status == "skipped"


def test_board_survives_a_partially_written_run(root: Path) -> None:
    """A torn final line is skipped rather than breaking the whole board."""
    run_dir = write_run(root)
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "tool_call", "thread_i')
    assert len(read_events(run_dir / "events.jsonl")) == len(events())
    assert Board(root=root).cards()


def test_board_ignores_an_unreadable_run(root: Path) -> None:
    """A corrupt run.json is skipped, not fatal."""
    write_run(root)
    broken = root / "demo-slug" / "runs" / "broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{not json", encoding="utf-8")
    assert [r.meta.run_id for r in Board(root=root).runs()] == [RUN_META["run_id"]]


def test_board_view_returns_none_for_an_unknown_run(root: Path) -> None:
    """Asking for a run that is not on disk is a miss, not an error."""
    write_run(root)
    assert Board(root=root).view("run-missing") is None


def test_workflows_group_threads_and_put_running_first(root: Path) -> None:
    """The workflow list's shape: one entry per run, running ones first."""
    write_run(root, run_id="20260915-120000-old", status="passed")
    write_run(root, slug="live-slug", run_id="20260915-130000-live", status="running", lines=events()[:4])
    groups = Board(root=root).workflows()
    assert [g["run"]["status"] for g in groups] == ["running", "passed"]
    assert [(t["name"], t["status"]) for t in groups[0]["threads"]] == [("implement", "running"), ("verify", "pending")]
    assert groups[0]["package"] == ".aif/live-slug"


# ── The whole graph before it has run ─────────────────────────────────────────


def test_declared_steps_without_events_are_pending_cards_in_manifest_order() -> None:
    """A run that has only reached its first step still shows every step."""
    started = [{"kind": "started", "thread_id": "th-implement", "thread_name": "implement", "timestamp": 1.0}]
    cards = thread_cards(parse_run("run-early", started), slug="demo-slug", steps=_steps())
    assert [(c.name, c.status) for c in cards] == [("implement", "running"), ("verify", "pending")]
    verify = cards[1]
    assert verify.thread_id == "th-verify" and verify.depends_on == ["implement"]
    assert verify.tickets == []  # nothing declared for verify in the fixture manifest
    implement_only = thread_cards(parse_run("run-none", []), slug="demo-slug", steps=_steps())
    assert [c.status for c in implement_only] == ["pending", "pending"]


def test_a_written_but_unrun_package_is_listed(root: Path) -> None:
    package = root / "fresh-slug"
    package.mkdir(parents=True)
    (package / "workflow.py").write_text("# generated\n", encoding="utf-8")
    groups = Board(root=root).workflows()
    (group,) = groups
    assert group["run"]["slug"] == "fresh-slug" and group["run"]["status"] == "unrun"
    assert group["run"]["run_id"] == "unrun:fresh-slug" and group["threads"] == []
    assert group["package"] == ".aif/fresh-slug"
    # a manifest left by a dry run fills the steps in as pending
    (package / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (group,) = Board(root=root).workflows()
    assert [(t["name"], t["status"]) for t in group["threads"]] == [("implement", "pending"), ("verify", "pending")]
    assert group["run"]["task"] == "add a --json flag"
    # once it has a run, it is a run, not an unrun package
    write_run(root, slug="fresh-slug")
    assert [g["run"]["status"] for g in Board(root=root).workflows()] == ["failed"]
