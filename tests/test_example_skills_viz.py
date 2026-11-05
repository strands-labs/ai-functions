"""Archive/source fidelity, causal replay, imports, and read-only serving."""

import json
import shutil
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from skills_viz.capture import Recorder, replay, snapshot, source_map, timeline
from skills_viz.viewer import make_server

RAW = (
    "---\nname: alpha\ndescription: Heat a sample\n---\n\n"
    "## Instructions\n1. Heat the sample on the stove.\n2. Read the thermometer.\n"
)


def skill():
    return snapshot(
        {"path": "alpha", "raw": RAW, "body": RAW.split("---", 2)[-1].lstrip("\n"), "frontmatter": {"name": "alpha"}},
        "lib",
    )


def call(identifier, tool, args, stamp=1):
    return {
        "kind": "tool_call",
        "id": identifier,
        "tool_use_id": identifier,
        "tool_name": tool,
        "arguments": args,
        "timestamp": stamp,
    }


def result(identifier, text, status="success", stamp=2):
    return {
        "kind": "tool_result",
        "id": identifier + "-result",
        "tool_use_id": identifier,
        "content": [{"text": text}],
        "status": status,
        "timestamp": stamp,
    }


def test_source_line_ranges_include_frontmatter_and_ignore_code_fence_steps():
    raw = RAW.replace("1. Heat", "### Preparation\n```text\n1. Not an instruction\n```\n1. Heat")
    mapped = source_map(raw)
    assert [step["title"] for step in mapped["steps"]] == ["Heat the sample on the stove.", "Read the thermometer."]
    for step in mapped["steps"]:
        assert "\n".join(raw.splitlines()[step["line_start"] - 1 : step["line_end"]]) == step["text"]
    assert mapped["steps"][0]["line_start"] == 11


def test_procedure_blocks_keep_nested_steps_and_conditions_inside_the_parent():
    raw = (
        "# Instructions\n"
        "1. **Prepare**: Set up the sample.\n"
        "   1. Find a container.\n"
        "   2. Add the sample.\n\n"
        "   If the container leaks, replace it.\n\n"
        "2. Read the thermometer.\n\n"
        "Additional context outside the procedure.\n"
    )
    mapped = source_map(raw)
    assert len(mapped["steps"]) == 2
    first, second = mapped["steps"]
    assert (first["number"], first["line_start"], first["line_end"]) == ("1", 2, 6)
    assert "   2. Add the sample." in first["text"] and "If the container leaks" in first["text"]
    assert second["text"] == "2. Read the thermometer."
    assert (second["number"], second["line_start"], second["line_end"]) == ("2", 8, 8)
    assert mapped["lines"][-1] == "Additional context outside the procedure."


def test_draft_blocks_use_draft_lines_without_claiming_a_formatted_skill(tmp_path):
    r = Recorder(tmp_path)
    body = "Before formatting:\n\n1. Prepare.\n2. Read.\n"
    r({"kind": "draft.created", "body": body})
    state = r.state()
    assert not state["skills"]
    mapping = state["activity"][0]["draft_map"]
    assert mapping["lines"] == body.splitlines()
    assert [(s["line_start"], s["line_end"]) for s in mapping["steps"]] == [(3, 3), (4, 4)]
    assert "draft_map" not in r.journal()[0]  # A derived view leaves the captured evidence untouched.


def test_actions_only_match_after_a_successful_load_of_the_exact_revision():
    s = skill()
    events = [
        call("before", "sw_act", {"action": "heat sample stove"}),
        call("load", "Skill", {"skill": "alpha"}),
        call("pending", "sw_act", {"action": "heat sample stove"}),
        result("load", f"# alpha\n\n{s['body']}"),
        call("after", "sw_act", {"action": "heat sample stove"}, stamp=3),
    ]
    rows = timeline(events, [s])
    assert not rows[0]["matches"] and not rows[2]["matches"]
    assert rows[1]["confirmed"]
    assert rows[3]["matches"][0]["method"] == "inferred"
    for reply in [
        result("load", "No skill named alpha"),
        result("load", "# alpha\n\nA different revision"),
        result("load", f"# alpha\n\n{s['body']}", "error"),
    ]:
        rows = timeline([events[1], reply, events[-1]], [s])
        assert not rows[0]["confirmed"] and not rows[-1]["matches"]


def test_fallback_trace_order_and_retry_boundaries_do_not_invent_instruction_use():
    s = skill()
    events = [
        call("load", "Skill", {"skill": "alpha"}, stamp=2),
        result("load", f"# alpha\n\n{s['body']}", stamp=3),
        call("first", "sw_act", {"action": "heat sample stove"}, stamp=4),
        call("retry", "sw_act", {"action": "heat sample stove"}, stamp=6),
        {"kind": "started", "id": "start-first", "timestamp": 1},
        {"kind": "started", "id": "start-retry", "timestamp": 5},
    ]
    rows = timeline(events, [s])
    assert rows[0]["confirmed"] and rows[1]["matches"]
    assert not rows[2]["matches"]


def test_sources_survive_deletion_and_export_replay_never_reads_live_files(tmp_path):
    source = tmp_path / "source/alpha"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(RAW)
    r = Recorder(tmp_path / "capture")
    r({"kind": "library.published", "library": "lib", "version": 1, "skill_dirs": [source]})
    r({"kind": "run.started", "library": "lib", "run_id": "run", "phase": "call", "skill_dirs": [source]})
    (source / "SKILL.md").write_text(RAW.replace("stove", "oven"))
    r({"kind": "library.published", "library": "lib", "version": 2, "skill_dirs": [source]})
    shutil.rmtree(source.parent)
    early = r.state(2)
    assert len(early["skills"]) == 1 and "oven" not in json.dumps(early)
    assert len(r.state()["skills"]) == 2
    export = tmp_path / "recording.json"
    r.export(export)
    other = Recorder(tmp_path / "import")
    other.import_json(export)
    assert other.state() == r.state()
    assert other.state(2) == early
    assert replay(r.journal())["last_seq"] == 3


@pytest.mark.parametrize("input_kind", ["markdown", "skill_file", "skill_directory"])
def test_written_markdown_import_preserves_source_without_inventing_history(tmp_path, input_kind):
    package = tmp_path / "written-skill"
    package.mkdir()
    source = package / ("my-procedure.md" if input_kind == "markdown" else "SKILL.md")
    raw = "# Instructions\n1. Heat the sample.\n2. Read the thermometer.\n" if input_kind == "markdown" else RAW
    source.write_text(raw)
    (package / "helper.py").write_text("raise AssertionError('Imported resources must never execute')\n")
    recorder = Recorder(tmp_path / "archive")
    recorder.import_skill(package if input_kind == "skill_directory" else source)
    assert source.read_text() == raw
    state = recorder.state()
    saved = next(iter(state["skills"].values()))
    assert saved["raw"] == raw and saved["lines"] == raw.splitlines()
    assert saved["filename"] == source.name and len(saved["steps"]) == 2
    assert saved["status"] == "document"
    assert saved["resources"] == ([] if input_kind == "markdown" else ["helper.py"])
    assert state["session"]["mode"] == "Document inspection"
    assert not state["runs"] and not state["rounds"]
    assert all(not lib["versions"] for lib in state["libraries"].values())
    assert not any(e["kind"] == "library.published" for e in recorder.journal())
    with pytest.raises(ValueError, match="empty archive"):
        recorder.import_skill(source)


def test_same_slug_in_two_libraries_has_separate_identity():
    one = snapshot({"slug": "alpha", "raw": RAW, "body": RAW}, "one")
    two = snapshot({"slug": "alpha", "raw": RAW, "body": RAW}, "two")
    assert one["id"] != two["id"] and one["fingerprint"] == two["fingerprint"]


def test_evaluation_baseline_cannot_seed_an_empty_published_library(tmp_path):
    r = Recorder(tmp_path)
    common = {"kind": "run.started", "library": "lib", "version": 1, "phase": "evaluation"}
    r({**common, "run_id": "without", "arm": "skill_off", "skills": []})
    r({**common, "run_id": "with", "arm": "skill_on", "skills": [skill()]})
    state = r.state()
    assert len(state["libraries"]["lib"]["versions"]["1"]) == 1
    assert not state["runs"]["without"]["skills"]


def test_directory_slug_remains_identity_when_frontmatter_name_differs():
    captured = snapshot(
        {"path": "/skills/actual-slug", "raw": RAW, "body": RAW, "frontmatter": {"name": "display-name"}}
    )
    assert captured["slug"] == "actual-slug" and captured["name"] == "display-name"


def test_tool_json_cannot_be_mistaken_for_a_captured_skill(tmp_path):
    r = Recorder(tmp_path)
    event = result("tool", "")
    event["content"] = [{"json": skill()}]
    r({"kind": "run.event", "run_id": "r", "event": event})
    state = r.state()
    assert not state["skills"]
    assert state["runs"]["r"]["events"][0] == event


def test_rejected_revision_does_not_mark_the_published_source_rejected(tmp_path):
    r = Recorder(tmp_path)
    old = skill()
    new = snapshot({"slug": "alpha", "raw": RAW.replace("stove", "oven"), "body": "replacement"}, "lib")
    r({"kind": "library.published", "library": "lib", "version": 1, "skills": [old]})
    r({"kind": "candidate.revised", "library": "lib", "round_id": "round", "skill": new})
    r(
        {
            "kind": "history.appended",
            "library": "lib",
            "round_id": "round",
            "entry": {"type": "round", "round": 2, "rejected": {"alpha": {"stage": "trial", "scores": {}}}},
        }
    )
    state = r.state()
    assert state["skills"][old["id"]]["status"] == "stored"
    assert state["skills"][new["id"]]["status"] == "rejected"


def test_completed_trace_does_not_duplicate_live_events(tmp_path):
    r = Recorder(tmp_path)
    event = call("call", "Skill", {"skill": "alpha"})
    r({"kind": "run.started", "run_id": "r", "skills": [skill()]})
    r({"kind": "run.event", "run_id": "r", "event": event})
    r({"kind": "run.events", "run_id": "r", "events": [event, result("call", f"# alpha\n\n{skill()['body']}")]})
    state = r.state()
    assert len(state["runs"]["r"]["events"]) == 2
    assert state["runs"]["r"]["timeline"][0]["confirmed"]


def test_environment_progress_is_causal_and_isolated_between_concurrent_runs(tmp_path):
    r = Recorder(tmp_path)
    progress = {"kind": "environment_progress", "action": "inspect", "task_id": "same-task", "done": False}
    r({**progress, "environment_id": "one", "observation": "sample\nreading: 10", "steps": 0})
    r({"kind": "run.started", "run_id": "a", "environment_id": "one"})
    r({"kind": "run.started", "run_id": "b", "environment_id": "two"})
    r({**progress, "environment_id": "two", "observation": "sample\nreading: 90", "steps": 9})
    r({**progress, "environment_id": "one", "observation": "sample\nreading: 20", "steps": 1})
    early = r.state()
    assert early["runs"]["a"]["progress"]["steps"] == 1
    assert early["runs"]["a"]["progress"]["changed_lines"] == ["reading: 20"]
    assert early["runs"]["b"]["progress"]["steps"] == 9
    r({**progress, "environment_id": "two", "observation": "done", "done": True, "passed": True})
    r({"kind": "run.finished", "run_id": "b"})
    assert r.state(5) == early
    assert r.state()["runs"]["b"]["progress"]["passed"]
    assert r.state()["runs"]["b"]["outcome"] is None  # Never invent the function's configured score.
    assert not r.state()["runs"]["a"]["progress"]["done"]


def test_interactive_replay_focuses_tool_results_at_their_recorded_prefix(tmp_path):
    r = Recorder(tmp_path)
    s = skill()
    r({"kind": "run.started", "library": "lib", "run_id": "a", "skills": [s]})
    r({"kind": "run.event", "run_id": "a", "event": call("load", "Skill", {"skill": "alpha"})})
    r({"kind": "run.event", "run_id": "a", "event": result("load", f"# alpha\n\n{s['body']}")})
    r({"kind": "run.event", "run_id": "a", "event": call("act", "heat", {"action": "heat sample stove"}, stamp=3)})
    r({"kind": "run.event", "run_id": "a", "event": result("act", "sample heated", stamp=4)})
    r({"kind": "run.finished", "run_id": "a", "outcome": 1.0})
    plan = replay(r.journal())
    points = plan["points"]
    assert [p["seq"] for p in points] == [1, 2, 3, 4, 5, 6]
    assert points[0]["skill_id"] == s["id"]
    load, action = points[2], points[4]
    assert load["entry_id"] == "load" and action["entry_id"] == "act"
    assert not r.state(load["seq"] - 1)["runs"]["a"]["timeline"][0]["confirmed"]
    assert r.state(load["seq"])["runs"]["a"]["timeline"][0]["confirmed"]
    assert r.state(action["seq"])["runs"]["a"]["timeline"][-1]["matches"]
    assert r.state(action["seq"])["runs"]["a"].get("outcome") is None
    assert plan["last_seq"] == 6
    assert sum(p["duration"] for p in points) == plan["duration"]
    assert points[-1]["at"] + points[-1]["duration"] == plan["duration"]
    assert replay([]) == {"duration": 0, "points": [], "last_seq": 0}


def test_legacy_import_preserves_sample_labels_skipped_stages_and_recorded_scores(tmp_path):
    source = tmp_path / "old.sqlite3"
    entries = [
        {
            "kind": "archive_snapshot",
            "snapshot": json.dumps(
                {"documents": [["meta", "session", json.dumps({"sample": True, "history_complete": False})]]}
            ),
        },
        {"kind": "session_started"},
        {"kind": "stage_skipped", "stage": "segmentation"},
        {"kind": "draft_created", "name": "Heat a sample", "body": "Use the stove."},
        {"kind": "run_started", "run_id": "run", "phase": "gate", "arguments": {"task": "heat"}},
        {"kind": "run_scored", "run_id": "run", "outcome": 0.75},
        {"kind": "run_finished", "run_id": "run", "status": "completed"},
    ]
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE journal (seq INTEGER PRIMARY KEY, data TEXT)")
        db.executemany(
            "INSERT INTO journal (data) VALUES (?)",
            [(json.dumps({**e, "timestamp": i}),) for i, e in enumerate(entries)],
        )
    original = source.read_bytes()
    r = Recorder(tmp_path / "new")
    r.import_legacy(source)
    assert source.read_bytes() == original
    state = r.state()
    assert state["session"]["mode"] == "Synthetic sample"
    assert "incomplete" in state["session"]["note"]
    assert state["runs"]["run"]["phase"] == "replay"
    assert state["runs"]["run"]["outcome"] == 0.75
    activity = {e["kind"]: e for e in state["activity"]}
    assert activity["stage.skipped"]["stage"] == "segmentation"
    assert activity["draft.created"]["body"] == "Use the stove."


def test_local_server_has_fixed_routes_and_escapable_untrusted_data(tmp_path):
    r = Recorder(tmp_path)
    r({"kind": "session.started", "title": "<script>alert(1)</script>"})
    server = make_server(r, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(url + "/api/state") as response:
            assert json.load(response)["session"]["title"] == "<script>alert(1)</script>"
        with urlopen(url) as response:
            page = response.read().decode()
            assert "<script>alert(1)</script>" not in page
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/../../.git/config")
        assert error.value.code == 404
        with urlopen(url + "/api/export") as response:
            assert json.load(response)["format"] == "skillify-observation-v1"
        with urlopen(url + "/api/replay") as response:
            assert json.load(response)["points"][0]["seq"] == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_browser_shares_procedure_blocks_and_distinguishes_load_match_and_environment(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api", reason="optional viewer browser checks")
    raw = RAW.replace(
        "1. Heat the sample on the stove.",
        "1. **Heat safely**: Heat the sample on the stove.\n"
        "   - Keep the thermometer in contact. <img src=x onerror=alert(1)>",
    )
    s = snapshot({"slug": "alpha", "raw": raw, "body": raw.split("---", 2)[-1].lstrip("\n")}, "lib")
    r = Recorder(tmp_path)
    r({"kind": "library.published", "library": "lib", "version": 1, "skills": [s]})
    r({"kind": "run.started", "run_id": "r", "library": "lib", "skills": [s], "environment_id": "env"})
    for reading in ("Reading: 10", "Reading: 20"):
        r({"kind": "environment.progress", "environment_id": "env", "action": "inspect", "observation": reading})
    for event in [call("failed", "Skill", {"skill": "alpha"}), result("failed", "Not available", "error")]:
        r({"kind": "run.event", "run_id": "r", "event": event})
    failed_seq = r.journal()[-1]["seq"]
    for event in [
        call("load", "Skill", {"skill": "alpha"}, stamp=3),
        result("load", f"# alpha\n\n{s['body']}", stamp=4),
    ]:
        r({"kind": "run.event", "run_id": "r", "event": event})
    loaded_seq = r.journal()[-1]["seq"]
    for event in [
        call("act", "heat", {"action": "heat sample stove"}, stamp=5),
        result("act", "sample heated", stamp=6),
    ]:
        r({"kind": "run.event", "run_id": "r", "event": event})
    server = make_server(r, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as error:
                if "Executable doesn't exist" in str(error):
                    pytest.skip("Install the optional Playwright Chromium browser to run viewer checks")
                raise
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                url = f"http://127.0.0.1:{server.server_port}"
                page.route(
                    "**/*", lambda route: route.continue_() if route.request.url.startswith(url) else route.abort()
                )
                page.goto(url)
                page.wait_for_function("() => window.skillsViz?.ready")
                assert page.locator("#source .procedure-step").count() == 2
                source = page.locator("#source .procedure-step.inferred")
                action = page.locator("#timeline .procedure-step.inferred")
                for selector in (".step-number", ".step-title", ".step-text", ".step-source"):
                    assert source.locator(selector).inner_text() == action.locator(selector).inner_text()
                assert source.evaluate("e => getComputedStyle(e).backgroundColor") == action.evaluate(
                    "e => getComputedStyle(e).backgroundColor"
                )
                assert source.locator(".step-source").inner_text() == "SKILL.md:L7–L8 ↗"
                assert "inferred" in source.inner_text()
                assert "<img src=x" in source.inner_text() and page.locator("#source img").count() == 0
                assert "environment output" in page.locator(".state-changes").inner_text().lower()
                assert source.evaluate("e => getComputedStyle(e).backgroundColor") != page.locator(
                    ".state-changes"
                ).evaluate("e => getComputedStyle(e).backgroundColor")
                source.locator(".step-source").click()
                assert page.locator("#raw-view").get_attribute("aria-pressed") == "true"
                assert page.locator("#source .source-line.highlight").count() == 2
                assert "**Heat safely**" in page.locator("#source .source-line.highlight").first.inner_text()
                page.get_by_role("button", name="Procedure blocks", exact=True).click()
                await_seek = "async seq => window.skillsViz.seek(seq)"
                page.evaluate(await_seek, failed_seq)
                assert page.locator("#source-delivery").is_hidden()
                assert page.locator(".procedure-step.inferred").count() == 0
                page.evaluate(await_seek, loaded_seq)
                assert "Skill load confirmed" in page.locator("#source-delivery").inner_text()
                assert page.locator(".procedure-step.inferred").count() == 0
                assert page.locator("#source .procedure-step").count() == 2
                page.get_by_role("button", name="Replay recorded events").click()
                page.wait_for_function("() => window.skillsViz.status.playing && window.skillsViz.status.seq === 1")
                page.get_by_role("button", name="Replay recorded events").click()
                assert page.locator("#play").inner_text() == "▶ Replay"
                for width in (850, 600):
                    page.set_viewport_size({"width": width, "height": 900})
                    assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
