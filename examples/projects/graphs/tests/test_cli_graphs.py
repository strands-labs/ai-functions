"""Behavioural tests for ``uv run python graphs.py view`` — the Textual TUI over recorded runs.

The app is driven headlessly through Textual's ``run_test`` pilot against a temp
folder written in the layout the runtime produces, so no agent, server or terminal
is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from recording_fixtures import MANIFEST, events, write_run
from rich.console import Console, RenderableType
from textual.widgets import Tab, TabbedContent
from typer.testing import CliRunner

from graphs import (
    Card,
    Connector,
    DetailPanel,
    GraphComponent,
    GraphRow,
    NodeCard,
    PassThrough,
    PromptRow,
    Splitter,
    TaskRow,
    Terminal,
    ThreadRow,
    TicketRow,
    ToolRow,
    WorkflowsApp,
    _node_label,
    _prompt_note,
    _thread_label,
    _ticket_detail,
    _ticket_row,
    _tool_detail,
    _tool_line,
    _workflow_card,
    app,
)


def test_view_command_forwards_the_folder_and_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path | None, float]] = []

    def open_view(viewer: WorkflowsApp) -> None:
        calls.append((viewer.root, viewer._interval))

    monkeypatch.setattr(WorkflowsApp, "run", open_view)
    result = CliRunner().invoke(app, ["view", str(tmp_path), "--interval", "0.5"])
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, 0.5)]


def test_view_command_rejects_a_nonpositive_scan_interval() -> None:
    result = CliRunner().invoke(app, ["view", "--interval", "0"])
    assert result.exit_code == 2


def _text(renderable: RenderableType | str) -> str:
    """Flatten a Rich renderable to plain text."""
    console = Console(file=None, color_system=None, force_terminal=False, width=10_000)
    with console.capture() as capture:
        console.print(renderable, end="")
    return capture.get()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An ``.aif``-shaped tree in a temp directory."""
    return tmp_path / ".aif"


def _cards(app: WorkflowsApp, pane: str) -> list[Card]:
    return list(app.query_one(pane).query(Card))


def _titles(app: WorkflowsApp, pane: str) -> list[str]:
    return [_text(card.border_title) for card in _cards(app, pane)]


def _rows(app: WorkflowsApp, pane: str) -> list[str]:
    return [_text(row.shown) for row in app.query_one(pane).query("Row")]


def _nodes(app: WorkflowsApp) -> list[NodeCard]:
    return list(app.query(NodeCard))


def _detail(app: WorkflowsApp) -> str:
    return _text(app.query_one("#detail_body", DetailPanel).shown)


def _detail_title(app: WorkflowsApp) -> str:
    return _text(app.query_one("#detail").border_title)


async def _focus_node(app: WorkflowsApp, pilot: Any, name: str = "implement") -> NodeCard:
    """Focus the graph node ``name`` and let the right pane follow."""
    await pilot.pause()
    node = next(n for n in _nodes(app) if n.payload["name"] == name)
    node.focus()
    await pilot.pause()
    await pilot.pause()
    assert app._selected_thread == node.payload["thread_id"]
    return node


# ── Left pane: one card per run ───────────────────────────────────────────────


async def test_each_run_is_a_card_holding_its_threads(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test():
        titles = _titles(app, "#workflows")
        rows = _rows(app, "#workflows")
        (card,) = _cards(app, "#workflows")
        assert card.has_class("-failed")  # border colour follows the run status
        subtitle = _text(card.border_subtitle)
        assert subtitle.endswith(" · kiro · terminal ttys001") and subtitle[:4].isdigit()
        assert not card.query(".note")  # the task is not repeated on the left
    assert len(titles) == 1 and "demo-slug" in titles[0] and "1/2" in titles[0]
    assert "implement" in rows[0] and "2/2" in rows[0] and "×2" in rows[0]
    assert "verify" in rows[1] and "0/1" in rows[1]


async def test_running_workflows_sort_first(root: Path) -> None:
    write_run(root, run_id="20260915-120000-old", status="passed")
    write_run(root, slug="live-slug", run_id="20260915-130000-live", status="running", lines=events()[:4])
    app = WorkflowsApp(root, interval=60)
    async with app.run_test():
        titles = _titles(app, "#workflows")
        assert _cards(app, "#workflows")[0].has_class("-running")
    assert "live-slug" in titles[0]


async def test_repeated_workflow_runs_can_be_selected_independently(root: Path) -> None:
    """Same-slug runs with the same timestamps and thread ids keep distinct graph details."""
    first_id, second_id = "20261009-120000-aaaaaa", "20261009-120000-bbbbbb"
    first_dir = write_run(root, run_id=first_id)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test(size=(160, 48)) as pilot:
        changed = {
            **MANIFEST,
            "objective": "a revised objective",
            "steps": [{**MANIFEST["steps"][0], "prompt": "a revised prompt"}],
        }
        write_run(root, run_id=second_id, manifest=changed, status="running", lines=events()[:4])
        await app.refresh_board()
        await pilot.pause()
        cards = _cards(app, "#workflows")
        assert len(cards) == 2
        assert second_id in _text(cards[0].border_subtitle)
        assert first_id in _text(cards[1].border_subtitle)
        # Even deleting the shared manifest leaves both runs inspectable.
        (first_dir.parent.parent / "manifest.json").unlink()
        for run_id, prompt in ((second_id, "a revised prompt"), (first_id, "add the flag")):
            row = next(
                row for row in app.query(ThreadRow) if row.payload == {"run_id": run_id, "thread_id": "th-implement"}
            )
            row.focus()
            await pilot.pause()
            await pilot.pause()
            assert app._selected_run == run_id
            assert _text(app.query(PromptRow).first().shown) == prompt
            assert len(_nodes(app)) == (1 if run_id == second_id else 2)


async def test_empty_folder_is_a_normal_state(tmp_path: Path) -> None:
    app = WorkflowsApp(tmp_path / "not-yet", interval=60)
    async with app.run_test():
        assert _cards(app, "#workflows") == []
        note = _text(app.query_one("#workflows .empty").render())
    assert "no runs yet" in note


async def test_a_stale_runtime_is_badged_on_its_card(root: Path) -> None:
    run_dir = write_run(root)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    (run_dir / "run.json").write_text(json.dumps({**meta, "runtime_version": 2}), encoding="utf-8")
    app = WorkflowsApp(root, interval=60)
    async with app.run_test():
        (card,) = _cards(app, "#workflows")
        assert _text(card.border_subtitle).endswith(" · stale runtime v2")


# ── Middle pane: the graph ────────────────────────────────────────────────────


async def test_the_selected_run_is_drawn_as_a_graph(root: Path) -> None:
    """implement → verify: two rows of one node, joined by a connector carrying the edge type."""
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()
        nodes = _nodes(app)
        labels = [_text(n.shown) for n in nodes]
        rows = [[type(w).__name__ for w in row.children] for row in app.query(GraphRow)]
        arrows = [_text(c.render()) for c in app.query(Connector)]
        terminals = [_text(t.render()).strip() for t in app.query(Terminal)]
        assert app.sub_title == "demo-slug"
        task = app.query_one("#graph").query(TaskRow).first()
        assert _text(task.shown) == "add a --json flag  ▸ open"  # one line; the full text opens below
        assert _text(task.border_title) == "task"
        assert app.query_one("#graph_detail").has_class("-hidden")
        assert nodes[0].has_class("-passed") and nodes[1].has_class("-failed")
    assert rows == [["Terminal"], ["NodeCard"], ["NodeCard"], ["Terminal"]]
    assert terminals == ["● start", "● end"]
    assert labels[0].splitlines() == ["✓ implement ×2", "str → str · kiro", "PRE  ✓ dir_exists", "POST ✓ flag_present"]
    assert labels[1].splitlines() == ["✗ verify", "str → str · claude", "POST ✗ pytest failed:…"]
    assert len(arrows) == 3 and all("▼ str" in a and "│" in a for a in arrows)


async def test_a_fan_in_graph_puts_parallel_nodes_side_by_side_with_a_pass_through(root: Path) -> None:
    manifest = {
        **MANIFEST,
        "steps": [
            {"name": "north", "thread_id": "th-north", "depends_on": []},
            {"name": "south", "thread_id": "th-south", "depends_on": []},
            {"name": "totals", "thread_id": "th-totals", "depends_on": ["north", "south"]},
            {"name": "publish", "thread_id": "th-publish", "depends_on": ["totals", "north"]},
        ],
    }
    lines = [
        {"kind": "started", "thread_id": f"th-{n}", "thread_name": n, "timestamp": float(i)}
        for i, n in enumerate(["north", "south", "totals", "publish"])
    ]
    write_run(root, lines=lines, manifest=manifest)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        rows = [[type(w).__name__ for w in row.children] for row in app.query(GraphRow)]
        arrows = [_text(c.render()) for c in app.query(Connector)]
        assert len(app.query(PassThrough)) == 1
    assert rows == [
        ["Terminal"],  # start
        ["NodeCard", "NodeCard"],  # north, south — in parallel, side by side
        ["NodeCard", "PassThrough"],  # totals, and north→publish passing through
        ["NodeCard"],  # publish
        ["Terminal"],  # end
    ]
    assert [a.count("▼") for a in arrows] == [2, 1, 1, 1]  # pass-throughs have no arrow or repeated type label
    assert [a.count("str") for a in arrows] == [2, 1, 1, 1]


@pytest.mark.parametrize("terminal_size", [(120, 40), (200, 80)])
async def test_graph_rows_fit_nodes_and_keep_long_edges_continuous(root: Path, terminal_size: tuple[int, int]) -> None:
    """Long-edge slots match the tallest node without stretching rows to the viewport."""
    dependencies = {
        "start": [],
        "work": ["start"],
        "review": ["work"],
        "finish": ["review", "start"],
    }
    manifest = {
        **MANIFEST,
        "steps": [
            {
                "name": name,
                "thread_id": f"th-{name}",
                "depends_on": deps,
                "post_conditions": [{"name": f"check_{i}"} for i in range(5)] if name == "work" else [],
            }
            for name, deps in dependencies.items()
        ],
    }
    write_run(root, manifest=manifest, lines=[])
    app = WorkflowsApp(root, interval=60)
    async with app.run_test(size=terminal_size) as pilot:
        await pilot.pause()
        assert len(app.query(PassThrough)) == 2
        for size in (terminal_size, (terminal_size[0] + 20, terminal_size[1] + 10)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for row in app.query(GraphRow):
                nodes = list(row.query(NodeCard))
                if not nodes:
                    continue
                assert row.size.height == max(node.outer_size.height for node in nodes)
                for edge in row.query(PassThrough):
                    assert edge.size.height == row.size.height
                    assert _text(edge.render()).splitlines() == [" " * 13 + "│"] * row.size.height


@pytest.mark.parametrize("terminal_size", [(120, 40), (200, 80)])
async def test_disconnected_graph_has_separate_start_and_end_markers(
    root: Path, terminal_size: tuple[int, int]
) -> None:
    dependencies = {
        "a_start": [],
        "a_work": ["a_start"],
        "a_finish": ["a_work"],
        "b_start": [],
        "b_work": ["b_start"],
        "b_finish": ["b_work"],
        "independent": [],
    }
    manifest = {
        **MANIFEST,
        "steps": [{"name": name, "thread_id": f"th-{name}", "depends_on": deps} for name, deps in dependencies.items()],
    }
    write_run(root, manifest=manifest, lines=[])
    app = WorkflowsApp(root, interval=60)
    async with app.run_test(size=terminal_size) as pilot:
        await pilot.pause()
        assert len(app.query(Terminal)) == 6
        assert not app.query(PassThrough)
        groups = list(app.query(GraphComponent))
        assert [[node.payload["name"] for node in group.query(NodeCard)] for group in groups] == [
            ["a_start", "a_work", "a_finish"],
            ["b_start", "b_work", "b_finish"],
            ["independent"],
        ]
        for size in (terminal_size, (terminal_size[0] + 20, terminal_size[1] + 10)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for group in groups:
                assert [_text(marker.render()).strip() for marker in group.query(Terminal)] == ["● start", "● end"]
                assert all(_text(connector.render()).count("▼") == 1 for connector in group.query(Connector))
                for row in group.query(GraphRow):
                    nodes = list(row.query(NodeCard))
                    assert row.size.height == (max(node.outer_size.height for node in nodes) if nodes else 1)
            assert groups[2].size.height < groups[0].size.height
            assert all(left.region.right <= right.region.x for left, right in zip(groups, groups[1:], strict=False))


async def test_the_graph_is_redrawn_only_when_the_run_changes(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()
        before = _nodes(app)
        await _focus_node(app, pilot, "verify")  # selecting a node must not rebuild the graph
        assert _nodes(app) == before


# ── Right pane: the node ──────────────────────────────────────────────────────


async def test_focusing_a_node_fills_the_conditions_tab(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        node = await _focus_node(app, pilot)
        assert node.has_class("-selected")
        assert _text(app.query_one("#right").border_title) == "NODE · implement"
        assert [t.label_text for t in app.query(Tab)] == ["AI-Thread", "Tool calls"]
        assert _text(app.query(PromptRow).first().border_title) == "prompt"
        prompts = [_text(r.shown) for r in app.query(PromptRow)]
        tickets = [_text(r.shown) for r in app.query(TicketRow)]
    assert prompts == ["add the flag"]
    assert "dir_exists" in tickets[0] and "PRE" in tickets[0]
    assert "flag_present" in tickets[1] and "2 ATTEMPTS" in tickets[1]


async def test_the_left_pane_row_selects_the_same_node(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()
        next(r for r in app.query(ThreadRow) if r.payload["thread_id"] == "th-verify").focus()
        await pilot.pause()
        await pilot.pause()
        assert app._selected_thread == "th-verify"
        selected = [n.payload["name"] for n in _nodes(app) if n.has_class("-selected")]
        tickets = [_text(r.shown) for r in app.query(TicketRow)]
    assert selected == ["verify"]
    assert "pytest failed" in tickets[0]


async def test_the_tool_calls_tab_groups_by_attempt(root: Path) -> None:
    retry = [
        *events()[:8],
        {
            "kind": "tool_call",
            "thread_id": "th-implement",
            "timestamp": 3.0,
            "tool_name": "fs_read",
            "tool_use_id": "tu-2",
            "arguments": {"path": "README.md"},
        },
        *events()[8:],
    ]
    write_run(root, lines=retry)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await _focus_node(app, pilot)
        app.query_one("#node_tabs", TabbedContent).active = "tab-tools"
        await pilot.pause()
        titles = _titles(app, "#tools")
        per_card = [len(card.query(ToolRow)) for card in _cards(app, "#tools")]
        first, second = app.query(ToolRow)
        first.focus()
        await pilot.pause()
        assert first.expanded and not second.expanded
        assert '"path": "cli.py"' in _text(first.shown) and "ok" in _text(first.shown)
        assert _detail_title(app).startswith("TOOL · fs_write")
    assert titles == ["ATTEMPT 1", "ATTEMPT 2"]
    assert per_card == [1, 1]


async def test_focusing_a_ticket_or_the_prompt_fills_the_detail_panel(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await _focus_node(app, pilot)
        next(r for r in app.query(TicketRow) if r.key == "th-implement::post::flag_present").focus()
        await pilot.pause()
        assert _detail_title(app) == "POST-CONDITION"
        assert _detail(app).startswith("def flag_present(result: str) -> PostConditionResult | None")
        app.query(PromptRow).first().focus()
        await pilot.pause()
        assert _detail_title(app) == "PROMPT · implement"
        assert _detail(app).strip() == "add the flag"
        app.query(TaskRow).first().focus()
        await pilot.pause()
        window = app.query_one("#graph_detail")
        assert not window.has_class("-hidden") and _text(window.border_title) == "TASK · demo-slug"
        full = _text(app.query_one("#graph_detail_body", DetailPanel).shown)
        assert full.index("add a --json flag") < full.index("what success means") < full.index("serve_cli.py accepts")
        assert _detail_title(app) == "PROMPT · implement"  # the right-hand detail panel is untouched
        next(iter(app.query(NodeCard))).focus()  # a node closes the task window again
        await pilot.pause()
        assert window.has_class("-hidden")


async def test_a_ticket_without_source_shows_its_message(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await _focus_node(app, pilot, "verify")
        app.query(TicketRow).first().focus()
        await pilot.pause()
        assert _detail(app) == "pytest failed: 1 error"


async def test_arrow_keys_move_between_nodes(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await _focus_node(app, pilot)
        await pilot.press("down")
        await pilot.pause()
        await pilot.pause()
        assert app._selected_thread == "th-verify"
        await pilot.press("down")  # last node: stays put
        await pilot.pause()
        assert app._selected_thread == "th-verify"
        await pilot.press("up")
        await pilot.pause()
        await pilot.pause()
        assert app._selected_thread == "th-implement"


# ── Polling ───────────────────────────────────────────────────────────────────


async def test_a_new_run_appears_on_the_next_scan(root: Path) -> None:
    write_run(root, run_id="20260915-120000-first")
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        assert len(_cards(app, "#workflows")) == 1
        write_run(root, slug="second-slug", run_id="20260915-130000-second")
        await app.refresh_board()
        await pilot.pause()
        titles = _titles(app, "#workflows")
    assert len(titles) == 2 and any("second-slug" in t for t in titles)


async def test_an_unchanged_folder_does_not_rebuild_the_panes(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        node = await _focus_node(app, pilot)
        await app.refresh_board()
        await pilot.pause()
        assert app.focused is node


async def test_a_growing_log_recolours_the_node_and_keeps_the_selection(root: Path) -> None:
    run_dir = write_run(root, status="running", lines=events()[:4])
    app = WorkflowsApp(root, interval=60)
    async with app.run_test() as pilot:
        await _focus_node(app, pilot)
        assert next(n for n in _nodes(app) if n.payload["name"] == "implement").has_class("-running")
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            for event in events()[4:]:
                handle.write(json.dumps(event) + "\n")
        await app.refresh_board()
        await pilot.pause()
        await pilot.pause()
        assert app._selected_thread == "th-implement"
        node = next(n for n in _nodes(app) if n.payload["name"] == "implement")
        assert node.has_class("-passed") and node.has_class("-selected")


# ── Splitters ─────────────────────────────────────────────────────────────────


async def test_splitters_resize_their_target_and_respect_the_minimum(root: Path) -> None:
    write_run(root)
    app = WorkflowsApp(root, interval=60)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        left, task, right, detail = app.query(Splitter)
        assert [s.target for s in (left, task, right, detail)] == ["#workflows", "#graph_detail", "#right", "#detail"]
        assert task.has_class("-hidden")  # shown only while the task window is open
        before = app.query_one("#workflows").outer_size.width
        assert left.drag(10) == before + 10
        await pilot.pause()
        assert app.query_one("#workflows").outer_size.width == before + 10
        right_before = app.query_one("#right").outer_size.width
        assert right.drag(5) == right_before - 5  # dragging right shrinks the right-hand pane
        detail_before = app.query_one("#detail").outer_size.height
        assert detail.drag(-4) == detail_before + 4  # dragging the bar up grows the panel below it
        assert detail.drag(1000) == detail.minimum


# ── Renderers ─────────────────────────────────────────────────────────────────

_TICKET: dict[str, Any] = {
    "name": "c",
    "kind": "code",
    "description": "",
    "docstring": "",
    "signature": "",
    "source": "",
    "message": None,
    "inferred": False,
    "stage": "post",
    "passed": False,
    "attempts": 0,
}

_CARD: dict[str, Any] = {
    "name": "implement",
    "status": "passed",
    "input_type": "str",
    "output_type": "str",
    "thread_id": "th-implement",
    "prompt": "add it",
    "backend": "codex",
    "attempts": 2,
    "depends_on": [],
    "tickets": [{**_TICKET, "passed": True, "attempts": 2}],
}


def test_ticket_row_states() -> None:
    assert "NOT RUN" in _text(_ticket_row(_TICKET))
    assert "POST" in _text(_ticket_row({**_TICKET, "passed": True, "attempts": 1}))
    failed_twice = _text(_ticket_row({**_TICKET, "attempts": 2}))
    assert "2 ATTEMPTS" in failed_twice and "✗" in failed_twice


def test_ticket_detail_is_the_source_alone() -> None:
    with_source = {
        **_TICKET,
        "signature": "c(result: str) -> None",
        "docstring": "The thing holds.",
        "message": "no",
        "source": "def c(result: str) -> None:\n    ...",
    }
    detail = _text(_ticket_detail(with_source))
    assert detail.startswith("def c(result: str) -> None:")
    assert "The thing holds." not in detail
    assert _text(_ticket_detail({**_TICKET, "message": "it does not"})) == "it does not"
    assert _text(_ticket_detail({**_TICKET, "name": "bare"})) == "bare"


def test_prompt_note_keeps_the_first_paragraph_on_one_line() -> None:
    long = "Write `clean.py` that reads\n`raw_sales.csv`.\n\nRules follow:\n- drop blanks"
    assert _text(_prompt_note(long)) == "Write `clean.py` that reads `raw_sales.csv`."
    assert _text(_prompt_note("x" * 300)).endswith("…") and len(_text(_prompt_note("x" * 300))) == 240


def test_node_label_and_workflow_card() -> None:
    label = _text(_node_label(_CARD))
    assert label.splitlines() == ["✓ implement ×2", "str → str · codex", "POST ✓ c"]
    two = _text(
        _node_label(
            {**_CARD, "tickets": [{**_TICKET, "stage": "pre"}, {**_TICKET, "name": "d"}, {**_TICKET, "name": "e"}]}
        )
    )
    assert two.splitlines()[2:] == ["PRE  ○ c", "POST ○ d", "     ○ e"]  # the stage tag only on the first of its kind
    long = _text(_node_label({**_CARD, "name": "a_very_long_step_name_indeed"}))
    assert "…" in long.splitlines()[0] and long.splitlines()[0].endswith(" ×2")  # names are cut to the box width
    assert "×2" in _text(_thread_label(_CARD))
    group = {
        "run": {
            "status": "running",
            "slug": "s",
            "runtime_version": 1,
            "run_id": "r",
            "session": "terminal A",
            "started_at": 1789613477.0,
            "backend": "kiro",
        },
        "threads": [_CARD],
        "stale_runtime": True,
    }
    workflow = _workflow_card(group)
    assert "1/1" in _text(workflow.border_title) and workflow.has_class("-running")
    assert _text(workflow.border_subtitle).endswith(" · kiro · stale runtime v1")


def test_tool_line_and_detail() -> None:
    with_purpose = {
        "tool": "fs_read",
        "purpose": "Check the header.",
        "arguments": {"path": "x.csv"},
        "status": "success",
        "duration": 0.4,
        "output": "",
    }
    assert "Check the header." in _text(_tool_line(with_purpose))
    without = {**with_purpose, "purpose": ""}
    assert "x.csv" in _text(_tool_line(without))
    assert "[error]" in _text(_tool_line({**without, "status": "error"}))
    multiline = _text(_tool_line({**without, "tool": "Running: python3 -c '\nimport csv\n" + "x" * 80}))
    assert "\n" not in multiline and "…" in multiline
    detail = _text(_tool_detail({**with_purpose, "output": "date,region"}))
    assert detail.index("Check the header.") < detail.index("arguments") < detail.index('"path": "x.csv"')
    assert detail.index("output") < detail.index("date,region")
    assert "no arguments or output" in _text(_tool_detail({**without, "arguments": {}}))
