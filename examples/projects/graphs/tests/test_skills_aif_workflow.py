"""The aif-workflow skill's runtime and checks, exercised with a fake agent in place of Kiro.

The runtime under test is the file the skill copies into every generated
``.aif/<slug>/`` package, loaded from this example's bundled skill — so a break here
would be inherited by every future workflow. Everything except the ACP subprocess is
real: the coordinator, the worker and its dispatcher, the thread scope, the library's
post-condition retry loop, the ``observed`` wrapper, the sink, and the on-disk layout.

The ``Board`` assertions read the same package back through the CLI's reader with no
shared code path, which is the round trip the wire format exists for.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
from ai_functions.ai_thread.postcondition import PostConditionResult, run_post_condition_loop
from ai_functions.protocols import Spawnable, Thread
from ai_functions.types import (
    InputShape,
    MessageAssistantCompleteEvent,
    ThreadContext,
    ToolCallEvent,
)

from recordings import RUNTIME_VERSION, Board

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "aif-workflow" / "runtime"


def _load(name: str) -> ModuleType:
    """Load one of the skill's runtime files as a module, the way a generated package does."""
    spec = importlib.util.spec_from_file_location(name, RUNTIME_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # ``inspect.getsource`` and dataclasses need it registered.
    spec.loader.exec_module(module)
    return module


rt = _load("runtime")
checks = _load("checks")


# ── A fake agent standing in for Kiro ─────────────────────────────────────────


@dataclass
class FakeThread(Thread[[str], str]):
    """Stands in for ``KiroAgentThread``: answers turns from a scripted list."""

    template: FakeAgent

    @property
    def name(self) -> str:
        return self.template.name

    async def execute(self, ctx: ThreadContext, *args: object, **kwargs: object) -> str:
        prompt = str(args[0]) if args else ""
        self.template.prompts.append(prompt)

        async def send_turn(text: str) -> str:
            del text
            index = min(len(self.template.seen), len(self.template.replies) - 1)
            reply = self.template.replies[index]
            self.template.seen.append(reply)
            ctx.on_event(ToolCallEvent(tool_use_id="tu-1", tool_name="fs_write", arguments={"path": "cli.py"}))
            # ``content`` blocks, as the library emits: a ``text=`` kwarg is silently dropped.
            ctx.on_event(MessageAssistantCompleteEvent(content=[{"text": reply}]))
            return reply

        return await run_post_condition_loop(
            ctx,
            prompt,
            thread_name=self.name,
            post_conditions=self.template.post_conditions,
            max_attempts=self.template.max_attempts,
            inject_buffer=[],
            send_turn=send_turn,
        )

    async def notify(self, text: str) -> None:
        """Side-channel messages are irrelevant to the fake."""

    def serialize_result(self, result: str) -> str:
        return result

    def deserialize_result(self, payload: str) -> str:
        return payload


@dataclass
class FakeAgent(Spawnable[[str], str]):
    """Template producing ``FakeThread``s, mirroring ``KiroAgent``'s surface."""

    name: str = "fake"
    replies: Sequence[str] = ("done",)
    post_conditions: tuple[object, ...] = ()
    max_attempts: int = 4
    prompts: list[str] = field(default_factory=list)
    seen: list[str] = field(default_factory=list)

    @property
    def input_shape(self) -> InputShape:
        return InputShape.STR_PROMPT

    def to_thread(self) -> FakeThread:
        return FakeThread(template=self)


def spawner(agents: dict[str, FakeAgent]):
    """Hand each step its pre-registered fake agent, wired to the wrapped conditions."""

    def spawn(step, workspace: Path, conditions: tuple[object, ...]) -> FakeAgent:
        del workspace
        agent = agents[step.name]
        agent.post_conditions = conditions
        agent.max_attempts = step.max_attempts
        return agent

    return spawn


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `.aif/<slug>/` package directory the runtime writes into."""
    directory = tmp_path / ".aif" / "demo-slug"
    directory.mkdir(parents=True)
    monkeypatch.setattr(rt, "package_dir", lambda: directory)
    return directory


async def run(package: Path, steps, agents: dict[str, FakeAgent]):
    """Run a workflow into ``package`` with fakes standing in for Kiro."""
    return await rt.run_workflow(
        task="add a --json flag",
        objective="the flag exists and is verified",
        steps=steps,
        workspace=package,
        echo=False,
        spawn=spawner(agents),
    )


def only_run_dir(package: Path) -> Path:
    """The single run directory the workflow produced."""
    (run_dir,) = (package / "runs").iterdir()
    return run_dir


# ── Orchestration ─────────────────────────────────────────────────────────────


async def test_a_passing_step_records_a_verified_thread(package: Path) -> None:
    """A step whose conditions pass finishes as a passed, verified thread."""

    def flag_present(result: str) -> PostConditionResult | None:
        """The result mentions the flag."""
        if "--json" not in result:
            return PostConditionResult(passed=False, message="say --json")
        return None

    step = rt.Step(name="implement", prompt="add the flag", post_conditions=((flag_present, "code"),))
    results = await run(package, [step], {"implement": FakeAgent(name="implement", replies=["added --json"])})

    assert results["implement"].status == "passed"
    run_dir = only_run_dir(package)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "passed"
    assert meta["finished_at"] is not None
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["implement"]["status"] == "passed"

    # The CLI's reader consumes the same package with no shared code path.
    (card,) = Board(root=package.parent).cards()
    assert card.verified is True
    assert card.tool_calls == 1
    assert card.tickets[0].attempts == 1
    assert card.tickets[0].description == "The result mentions the flag."


async def test_a_failing_condition_retries_and_feeds_feedback_back(package: Path) -> None:
    """The library's loop re-prompts on failure, and the ticket counts the attempts."""

    def flag_present(result: str) -> PostConditionResult | None:
        """The result mentions the flag."""
        if "--json" not in result:
            return PostConditionResult(passed=False, message="no --json flag; add it")
        return None

    step = rt.Step(name="implement", prompt="add the flag", max_attempts=3, post_conditions=((flag_present, "code"),))
    results = await run(package, [step], {"implement": FakeAgent(name="implement", replies=["nope", "added --json"])})

    assert results["implement"].status == "passed"
    (card,) = Board(root=package.parent).cards()
    assert (card.tickets[0].attempts, card.tickets[0].passed) == (2, True)

    log = (only_run_dir(package) / "events.jsonl").read_text(encoding="utf-8")
    assert "Post-condition failures (attempt 1/3)" in log
    assert "no --json flag" in log


async def test_an_async_judge_can_fail(package: Path) -> None:
    """An async condition is awaited, not treated as a truthy coroutine.

    A sync wrapper would let every ``@ai_function`` judge pass silently, so this
    guards the reason ``observed`` is async.
    """

    async def judge(result: str) -> PostConditionResult:
        """The report reads as a report."""
        del result
        return PostConditionResult(passed=False, message="not a report")

    step = rt.Step(name="assess", prompt="write it", max_attempts=1, post_conditions=((judge, "ai"),))
    results = await run(package, [step], {"assess": FakeAgent(name="assess", replies=["nope"])})

    assert results["assess"].status == "failed"
    (card,) = Board(root=package.parent).cards()
    assert card.verified is False
    assert card.tickets[0].kind == "ai"
    assert card.tickets[0].message == "not a report"


async def test_exhausting_attempts_fails_the_step(package: Path) -> None:
    """A condition that never passes fails the step rather than returning unverified work."""

    def never(result: str) -> PostConditionResult:
        """Always fails, to exhaust the attempt budget."""
        del result
        return PostConditionResult(passed=False, message="still wrong")

    step = rt.Step(name="implement", prompt="add it", max_attempts=2, post_conditions=((never, "code"),))
    results = await run(package, [step], {"implement": FakeAgent(name="implement", replies=["a", "b"])})

    assert results["implement"].status == "failed"
    assert json.loads((only_run_dir(package) / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    cells = [
        json.loads(line)
        for line in (only_run_dir(package) / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == rt.POSTCONDITION
    ]
    assert [c["attempt"] for c in cells] == [1, 2]
    assert all(c["passed"] is False for c in cells)


async def test_a_failing_pre_condition_blocks_without_spawning(package: Path) -> None:
    """A blocked step never starts its agent, and its dependents are skipped."""

    def workspace_ready() -> PostConditionResult:
        """The workspace holds the file the step needs."""
        return PostConditionResult(passed=False, message="cli.py is missing")

    first = rt.Step(name="implement", prompt="add it", pre_conditions=(workspace_ready,))
    second = rt.Step(name="verify", prompt="check it", depends_on=("implement",))
    agents = {"implement": FakeAgent(name="implement"), "verify": FakeAgent(name="verify")}
    results = await run(package, [first, second], agents)

    assert results["implement"].status == "blocked"
    assert results["implement"].blocked_by == ["workspace_ready"]
    assert results["verify"].status == "skipped"
    assert agents["implement"].prompts == [], "a blocked step must not start its agent"

    implement, verify = sorted(Board(root=package.parent).cards(), key=lambda c: c.name)
    # The viewer agrees with result.json: the gate blocked one step and skipped the other.
    assert implement.status == "blocked"
    assert verify.status == "skipped"
    card = implement
    pre = {t.name: t for t in card.tickets if t.stage == "pre"}
    assert pre["workspace_ready"].passed is False
    assert pre["workspace_ready"].message == "cli.py is missing"


async def test_dependent_steps_receive_upstream_results(package: Path) -> None:
    """A dependent step's prompt is templated with the upstream result."""
    first = rt.Step(name="implement", prompt="add it")
    second = rt.Step(name="verify", prompt="check this: {implement}", depends_on=("implement",))
    agents = {
        "implement": FakeAgent(name="implement", replies=["wrote cli.py"]),
        "verify": FakeAgent(name="verify", replies=["looks right"]),
    }

    results = await run(package, [first, second], agents)

    assert results["verify"].status == "passed"
    assert agents["verify"].prompts == ["check this: wrote cli.py"]


async def test_other_braces_in_a_prompt_survive_interpolation(package: Path) -> None:
    """A JSON example or a stray brace neither blocks substitution nor fails the step."""
    first = rt.Step(name="implement", prompt="add it")
    prompt = 'check {implement}, then reply like {"ok": true} } or {unknown}'
    second = rt.Step(name="verify", prompt=prompt, depends_on=("implement",))
    agents = {
        "implement": FakeAgent(name="implement", replies=["wrote cli.py"]),
        "verify": FakeAgent(name="verify", replies=["looks right"]),
    }

    results = await run(package, [first, second], agents)

    assert results["verify"].status == "passed"
    assert agents["verify"].prompts == ['check wrote cli.py, then reply like {"ok": true} } or {unknown}']


async def test_the_manifest_describes_the_workflow_for_the_reader(package: Path) -> None:
    """manifest.json carries step shape, condition docstrings, signatures and source."""
    step = rt.Step(
        name="implement", prompt="add it", pre_conditions=(_dir_exists,), post_conditions=((_flag_written, "code"),)
    )
    await run(package, [step], {"implement": FakeAgent(name="implement")})

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_version"] == rt.RUNTIME_VERSION == RUNTIME_VERSION
    assert json.loads((only_run_dir(package) / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["task"] == "add a --json flag"
    (declared,) = manifest["steps"]
    assert declared["thread_id"] == "th-implement"
    assert declared["pre_conditions"][0]["description"] == "The target directory exists."
    assert declared["post_conditions"][0]["kind"] == "code"
    assert declared["post_conditions"][0]["signature"] == "_flag_written(result: str) -> None"
    assert declared["post_conditions"][0]["source"].startswith("def _flag_written")


async def test_repeated_runs_preserve_history_and_recheck_conditions(package: Path) -> None:
    """Rerunning a graph creates independent events and checks, preserving old graph metadata."""
    ready = True

    def input_ready() -> PostConditionResult | None:
        """The input is ready for this execution."""
        if not ready:
            return PostConditionResult(passed=False, message="input is unavailable")
        return None

    step = rt.Step(
        name="implement", prompt="add it", pre_conditions=(input_ready,), post_conditions=((_flag_written, "code"),)
    )
    agent = FakeAgent(name="implement")
    await run(package, [step], {"implement": agent})
    first_dir = only_run_dir(package)
    original_files = {path.name: path.read_bytes() for path in first_dir.iterdir()}
    board = Board(root=package.parent)
    assert board.cards()[0].status == "passed"

    ready = False
    outcomes = await run(package, [step], {"implement": agent})
    second_dir = next(path for path in (package / "runs").iterdir() if path != first_dir)
    assert outcomes["implement"].status == "blocked"
    assert json.loads((second_dir / "run.json").read_text())["status"] == "failed"
    assert agent.prompts == ["add it"]  # the second run's precondition prevented spawning

    ready = True
    changed = rt.Step(name="implement", prompt="new requirements", post_conditions=((_flag_written, "code"),))
    # No conditions from a later definition may appear on the first two runs.
    await run(package, [changed], {"implement": FakeAgent(name="implement")})
    assert {path.name: path.read_bytes() for path in first_dir.iterdir()} == original_files
    groups = {group["run"]["run_id"]: group for group in board.workflows()}
    assert len(groups) == 3
    for run_dir in (first_dir, second_dir):
        (thread,) = groups[run_dir.name]["threads"]
        assert thread["prompt"] == "add it"
        assert {ticket["name"] for ticket in thread["tickets"]} == {"input_ready", "_flag_written"}
    (latest,) = [group for key, group in groups.items() if key not in {first_dir.name, second_dir.name}]
    assert latest["threads"][0]["prompt"] == "new requirements"


def _dir_exists() -> None:
    """The target directory exists."""


def _flag_written(result: str) -> None:
    """The flag was written."""
    del result


def test_prompt_section_reads_one_step_out_of_prompt_md(tmp_path: Path) -> None:
    """Prompts live in prompt.md, one section per step."""
    source = tmp_path / "prompt.md"
    source.write_text("## implement\nAdd the flag.\n\n## verify\nRun the tests.\n", encoding="utf-8")
    assert rt.prompt_section("implement", path=source) == "Add the flag."
    assert rt.prompt_section("verify", path=source) == "Run the tests."
    with pytest.raises(KeyError):
        rt.prompt_section("missing", path=source)


# ── Checks: the condition plumbing ────────────────────────────────────────────


def test_close_tolerates_a_fraction_of_a_cent() -> None:
    assert checks.close(2078.4499999, 2078.45)
    assert checks.close(2078, 2078.0)
    assert not checks.close(2078.99, 2078.45)
    assert not checks.close("2078.45", 2078.45)  # A formatted string is not a number.


def test_close_rejects_booleans() -> None:
    assert not checks.close(True, 1.0)
    assert not checks.is_number(False)


def test_matches_compares_by_expected_type() -> None:
    assert checks.matches(3, 3)
    assert not checks.matches(3.5, 3)
    assert checks.matches("north", "north")
    assert checks.matches(419.789999, 419.79)


def test_read_json_is_defensive(tmp_path: Path) -> None:
    assert checks.read_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert checks.read_json(bad) is None
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert checks.read_json(array) is None
    assert checks.read_json_list(array) == [1, 2]


def test_readers_are_not_cached(tmp_path: Path) -> None:
    """A re-read must see the new content — conditions run once per attempt."""
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"total": 1}), encoding="utf-8")
    assert checks.read_json(path) == {"total": 1}
    path.write_text(json.dumps({"total": 2}), encoding="utf-8")
    assert checks.read_json(path) == {"total": 2}
    path.write_text("second", encoding="utf-8")
    assert checks.read_text(path) == "second"


def test_missing_strings_reports_only_the_absent(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    report.write_text("# Revenue report\nrows: 10\n$2,078.45\n", encoding="utf-8")
    assert checks.missing_strings(report, ["rows: 10", "$2,078.45"]) == []
    assert checks.missing_strings(report, ["north", "rows: 10"]) == ["north"]


def test_currency_formats_with_separators_and_two_decimals() -> None:
    assert checks.currency(2078.4) == "$2,078.40"
    assert checks.currency(1234567.891) == "$1,234,567.89"


def test_file_unchanged_captures_the_digest_when_built(tmp_path: Path) -> None:
    spec = tmp_path / "test_spec.py"
    spec.write_text("assert True\n", encoding="utf-8")
    guard = checks.file_unchanged(spec, because="It is the specification")
    assert guard("") is None
    assert guard.__doc__ == "test_spec.py is byte-identical to how it started."
    spec.write_text("assert False\n", encoding="utf-8")
    outcome = guard("")
    assert outcome is not None
    assert "was modified" in outcome.message
    assert "It is the specification" in outcome.message


def test_file_unchanged_notices_deletion(tmp_path: Path) -> None:
    spec = tmp_path / "data.csv"
    spec.write_text("a,b\n", encoding="utf-8")
    guard = checks.file_unchanged(spec)
    spec.unlink()
    outcome = guard("")
    assert outcome is not None
    assert "deleted" in outcome.message


def test_must_exit_zero_reports_a_missing_script(tmp_path: Path) -> None:
    outcome = checks.must_exit_zero(tmp_path / "stats.py")
    assert outcome is not None
    assert "was not created" in outcome.message


def test_must_exit_zero_carries_stderr_into_the_message(tmp_path: Path) -> None:
    script = tmp_path / "boom.py"
    script.write_text("raise SystemExit('deliberate failure')\n", encoding="utf-8")
    outcome = checks.must_exit_zero(script)
    assert outcome is not None
    assert "deliberate failure" in outcome.message


def test_must_exit_zero_passes_a_working_script(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('rows: 3')\n", encoding="utf-8")
    assert checks.must_exit_zero(script) is None
    assert "rows: 3" in checks.run_script(script).stdout


def test_ast_helpers_find_what_a_refactor_broke(tmp_path: Path) -> None:
    module = tmp_path / "code.py"
    module.write_text(
        '"""Module."""\n'
        "\n"
        "def documented(value: int) -> int:\n"
        '    """Doubles."""\n'
        "    return value * 2\n"
        "\n"
        "def bare(value):\n"
        "    return value\n"
        "\n"
        "def long_one() -> None:\n"
        '    """Long."""\n' + "    x = 1\n" * 6,
        encoding="utf-8",
    )
    assert checks.function_names(module) == {"documented", "bare", "long_one"}
    assert checks.undocumented_functions(module) == ["bare"]
    assert checks.unannotated_functions(module) == ["bare"]
    assert checks.long_functions(module, limit=3) == {"long_one": 7}
    assert checks.long_functions(module, limit=10) == {}


def test_ast_helpers_include_async_functions(tmp_path: Path) -> None:
    module = tmp_path / "code.py"
    module.write_text("async def fetch(url):\n    return url\n", encoding="utf-8")
    assert checks.function_names(module) == {"fetch"}
    assert checks.undocumented_functions(module) == ["fetch"]
    assert checks.unannotated_functions(module) == ["fetch"]


def test_ast_helpers_survive_a_missing_or_broken_file(tmp_path: Path) -> None:
    assert checks.function_names(tmp_path / "nope.py") == set()
    broken = tmp_path / "broken.py"
    broken.write_text("def oops(:\n", encoding="utf-8")
    assert checks.functions_in(broken) == []


def test_tail_keeps_the_end_of_long_output() -> None:
    assert checks.tail("x" * 100, limit=10) == "x" * 10
    assert checks.tail("short") == "short"


@pytest.mark.parametrize("value", [1, 2.5, -3])
def test_is_number_accepts_real_numbers(value: float) -> None:
    assert checks.is_number(value)


# ── Packaging ─────────────────────────────────────────────────────────────────


def test_the_skill_ships_every_file_the_procedure_copies() -> None:
    """SKILL.md's copy step names four files; all four are in the wheel beside it."""
    skill = RUNTIME_DIR.parent
    assert (skill / "SKILL.md").is_file()
    for name in ("runtime.py", "checks.py", "workflow_template.py", "conditions_template.py"):
        assert (RUNTIME_DIR / name).is_file(), name
    assert (skill / "examples" / "regional-report" / "data.csv").is_file()  # the README's try-it-out input
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert "S=.claude/skills" not in text, "the copy source must not assume one target"


# ── Round trip: runtime → disk → TUI ──────────────────────────────────────────


async def test_a_recorded_run_renders_in_the_workflows_tui(package: Path) -> None:
    """What the runtime writes, the TUI shows: tree, tickets with source, and tool calls.

    This is the one test that ties the producer (the skill's runtime) to the
    consumer (the CLI) through nothing but the files on disk, so it is where a
    wire-format drift would surface first.
    """
    from rich.console import Console
    from textual.widgets import TabbedContent

    from graphs import Card, DetailPanel, NodeCard, TicketRow, ToolRow, WorkflowsApp

    def flag_present(result: str) -> PostConditionResult | None:
        """The result mentions the flag."""
        if "--json" not in result:
            return PostConditionResult(passed=False, message="no --json flag; add it")
        return None

    step = rt.Step(name="implement", prompt="add the flag", max_attempts=3, post_conditions=((flag_present, "code"),))
    await run(package, [step], {"implement": FakeAgent(name="implement", replies=["nope", "added --json"])})

    def text(renderable: object) -> str:
        console = Console(file=None, color_system=None, force_terminal=False, width=10_000)
        with console.capture() as capture:
            console.print(renderable, end="")
        return capture.get()

    app = WorkflowsApp(package.parent, interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()
        (workflow,) = app.query_one("#workflows").query(Card)
        assert "demo-slug" in text(workflow.border_title) and "1/1" in text(workflow.border_title)
        (node,) = app.query(NodeCard)  # one step, one node in the graph
        assert "implement" in text(node.shown) and node.has_class("-passed")

        node.focus()
        await pilot.pause()
        await pilot.pause()
        ticket_rows = [text(row.shown) for row in app.query(TicketRow)]
        assert any("flag_present" in row and "2 ATTEMPTS" in row for row in ticket_rows)

        next(r for r in app.query(TicketRow) if r.key == "th-implement::post::flag_present").focus()
        await pilot.pause()
        detail = text(app.query_one("#detail_body", DetailPanel).shown)
        assert "flag_present(result: str) -> PostConditionResult | None" in detail
        assert 'if "--json" not in result:' in detail  # the source that actually ran

        app.query_one("#node_tabs", TabbedContent).active = "tab-tools"
        await pilot.pause()
        attempts = [text(card.border_title) for card in app.query_one("#tools").query(Card)]
        tool_rows = [text(row.shown) for row in app.query(ToolRow)]
    assert attempts == ["ATTEMPT 1", "ATTEMPT 2"]  # the retry's tool call lands under the second attempt
    assert sum("fs_write" in row for row in tool_rows) == 2


# ── Backends ──────────────────────────────────────────────────────────────────


@dataclass
class _Recorded:
    """A stand-in agent class that records the keyword arguments it was built with."""

    kwargs: dict[str, object]


def _recorder() -> type:
    class Agent:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    return Agent


def _wrapped(*conditions: object) -> tuple[object, ...]:
    return tuple(conditions)


def test_backend_precedence_is_step_then_run_then_env_then_kiro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rt.BACKEND_ENV, raising=False)
    assert rt.resolve_backend(rt.Step(name="s")) is rt.Backend.KIRO
    monkeypatch.setenv(rt.BACKEND_ENV, "codex")
    assert rt.resolve_backend(rt.Step(name="s")) is rt.Backend.CODEX
    assert rt.resolve_backend(rt.Step(name="s"), "claude") is rt.Backend.CLAUDE
    assert rt.resolve_backend(rt.Step(name="s", backend="ai_thread"), "claude") is rt.Backend.AI_THREAD
    with pytest.raises(ValueError):
        rt.resolve_backend(rt.Step(name="s", backend="gemini"))


def test_kiro_factory_maps_workspace_model_and_unattended_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _recorder()
    monkeypatch.setattr(rt, "_import_kiro", lambda: agent)
    monkeypatch.delenv(rt.MODEL_ENV, raising=False)
    monkeypatch.setenv("KIRO_MODEL", "claude-sonnet-4")
    built = rt._template_for(rt.Step(name="implement", max_attempts=3), tmp_path, _wrapped("c1"), rt.Backend.KIRO)
    assert built.kwargs == {
        "name": "implement",
        "cwd": str(tmp_path),
        "model": "claude-sonnet-4",
        "auto_approve": True,
        "post_conditions": ("c1",),
        "max_attempts": 3,
    }


def test_claude_factory_puts_cwd_and_bypass_in_the_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent, options = _recorder(), _recorder()
    monkeypatch.setattr(rt, "_import_claude", lambda: (agent, options))
    monkeypatch.setenv(rt.MODEL_ENV, "claude-opus-5")
    step = rt.Step(name="implement", backend="claude", max_attempts=2)
    built = rt._template_for(step, tmp_path, _wrapped("c1"), rt.Backend.KIRO)  # the step wins over the run default
    assert built.kwargs["name"] == "implement"
    assert built.kwargs["options"].kwargs == {
        "cwd": str(tmp_path),
        "permission_mode": "bypassPermissions",
        "model": "claude-opus-5",
    }
    assert (built.kwargs["post_conditions"], built.kwargs["max_attempts"]) == (("c1",), 2)


def test_codex_factory_uses_the_workspace_write_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _recorder()

    class Sandbox:
        workspace_write = "workspace_write"

    monkeypatch.setattr(rt, "_import_codex", lambda: (agent, Sandbox))
    monkeypatch.delenv(rt.MODEL_ENV, raising=False)
    step = rt.Step(name="implement", model="gpt-5-codex")
    built = rt._template_for(step, tmp_path, _wrapped(), rt.Backend.CODEX)
    assert built.kwargs["cwd"] == str(tmp_path)
    assert built.kwargs["model"] == "gpt-5-codex"  # the step's model beats the environment
    assert built.kwargs["sandbox"] == "workspace_write"


def test_ai_thread_factory_builds_an_ai_function_told_where_to_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ai_functions.ai_thread import AIFunction

    def fake_tool() -> str:
        """A tool."""
        return ""

    monkeypatch.delenv(rt.MODEL_ENV, raising=False)
    step = rt.Step(name="implement", tools=(fake_tool,), max_attempts=5)
    built = rt._template_for(step, tmp_path, _wrapped("c1"), rt.Backend.AI_THREAD)
    assert isinstance(built, AIFunction)
    assert built.name == "implement"
    assert built.input_shape is InputShape.STR_PROMPT
    assert str(tmp_path) in (built.config.system_prompt or "")
    assert built.config.tools == (fake_tool,)
    assert built.config.post_conditions == ("c1",)
    assert built.config.max_attempts == 5


def test_a_missing_extra_names_its_install_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def no_sdk(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("claude_agent_sdk", "openai_codex", "strands_tools")):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    for backend, hint in ((rt.Backend.CLAUDE, "[claude-code]"), (rt.Backend.CODEX, "[codex]")):
        with pytest.raises(rt.BackendUnavailableError, match=re.escape(hint)):
            rt._template_for(rt.Step(name="s"), tmp_path, _wrapped(), backend)
    with pytest.raises(rt.BackendUnavailableError, match="strands-agents-tools"):
        rt._template_for(rt.Step(name="s"), tmp_path, _wrapped(), rt.Backend.AI_THREAD)


async def test_the_backend_is_recorded_per_step_and_per_run(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rt.BACKEND_ENV, raising=False)
    steps = [
        rt.Step(name="implement", prompt="add it"),
        rt.Step(name="review", prompt="check it", backend="claude", depends_on=("implement",)),
    ]
    agents = {"implement": FakeAgent(name="implement"), "review": FakeAgent(name="review")}
    await rt.run_workflow(
        task="t", objective="o", steps=steps, workspace=package, echo=False, spawn=spawner(agents), backend="codex"
    )
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "codex"
    assert [s["backend"] for s in manifest["steps"]] == ["codex", "claude"]
    meta = json.loads((only_run_dir(package) / "run.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "codex"
