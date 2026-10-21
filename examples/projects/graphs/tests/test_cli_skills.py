"""``uv run python graphs.py skills`` — listing and installing the bundled skills into a project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphs import app
from skills import (
    KiroAgentConfigError,
    SkillConflictError,
    SkillNotFoundError,
    Target,
    available,
    install,
    kiro_skill_uri,
    register_kiro_skill,
    skills_root,
)

runner = CliRunner()


def test_script_installs_from_another_working_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "graphs.py"
    project = tmp_path / "project"
    result = subprocess.run(
        [sys.executable, str(script), "skills", "install", "aif-workflow", "--kiro", "--project", str(project)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    copied = project / ".kiro/skills/aif-workflow/runtime/runtime.py"
    assert copied.read_bytes() == (skills_root() / "aif-workflow/runtime/runtime.py").read_bytes()
    config = json.loads((project / ".kiro/agents/ai-functions.json").read_text())
    assert kiro_skill_uri(project / ".kiro/skills/aif-workflow/SKILL.md") in config["resources"]


def test_the_workflow_skill_is_bundled() -> None:
    assert available() == ["aif-workflow"]
    assert (skills_root() / "aif-workflow" / "runtime" / "runtime.py").is_file()


def test_install_copies_the_whole_skill_folder(tmp_path: Path) -> None:
    report = install("aif-workflow", Target.CLAUDE, tmp_path)
    assert report.destination == tmp_path / ".claude" / "skills" / "aif-workflow"
    names = sorted(p.relative_to(report.destination).as_posix() for p in report.written)
    assert names == [
        "SKILL.md",
        "examples/regional-report/data.csv",
        "runtime/checks.py",
        "runtime/conditions_template.py",
        "runtime/runtime.py",
        "runtime/workflow_template.py",
    ]
    source = skills_root() / "aif-workflow" / "runtime" / "runtime.py"
    assert (report.destination / "runtime" / "runtime.py").read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    ("target", "folder"),
    [(Target.CLAUDE, ".claude/skills"), (Target.KIRO, ".kiro/skills"), (Target.CODEX, ".agents/skills")],
)
def test_each_target_has_its_own_folder(tmp_path: Path, target: Target, folder: str) -> None:
    report = install("aif-workflow", target, tmp_path)
    assert report.destination == tmp_path / folder / "aif-workflow"
    assert (report.destination / "SKILL.md").is_file()


def test_reinstall_is_idempotent(tmp_path: Path) -> None:
    first = install("aif-workflow", Target.CLAUDE, tmp_path)
    second = install("aif-workflow", Target.CLAUDE, tmp_path)
    assert second.written == []
    assert sorted(second.unchanged) == sorted(first.written)


def test_a_differing_file_is_a_conflict_and_nothing_is_written(tmp_path: Path) -> None:
    install("aif-workflow", Target.CLAUDE, tmp_path)
    edited = tmp_path / ".claude" / "skills" / "aif-workflow" / "runtime" / "checks.py"
    edited.write_text("# local edit\n", encoding="utf-8")
    (tmp_path / ".claude" / "skills" / "aif-workflow" / "SKILL.md").unlink()

    with pytest.raises(SkillConflictError) as excinfo:
        install("aif-workflow", Target.CLAUDE, tmp_path)
    assert excinfo.value.conflicts == [edited]
    assert edited.read_text(encoding="utf-8") == "# local edit\n"
    assert not (tmp_path / ".claude" / "skills" / "aif-workflow" / "SKILL.md").exists(), "a conflict writes nothing"


def test_force_overwrites_only_what_differs(tmp_path: Path) -> None:
    install("aif-workflow", Target.CLAUDE, tmp_path)
    edited = tmp_path / ".claude" / "skills" / "aif-workflow" / "runtime" / "checks.py"
    edited.write_text("# local edit\n", encoding="utf-8")
    report = install("aif-workflow", Target.CLAUDE, tmp_path, force=True)
    assert report.written == [edited]
    assert edited.read_bytes() == (skills_root() / "aif-workflow" / "runtime" / "checks.py").read_bytes()


def test_unknown_skill_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SkillNotFoundError, match="aif-workflow"):
        install("nope", Target.CLAUDE, tmp_path)


# ── The command surface ───────────────────────────────────────────────────────


def test_skills_list_names_the_skill() -> None:
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0, result.output
    assert "aif-workflow" in result.output


def test_skills_install_requires_a_target(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "aif-workflow", "--project", str(tmp_path)])
    assert result.exit_code == 2
    assert "--claude" in result.output


def test_skills_install_all_into_two_targets(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--all", "--claude", "--kiro", "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for folder in (".claude/skills", ".kiro/skills"):
        assert (tmp_path / folder / "aif-workflow" / "SKILL.md").is_file()
    assert result.output.count("wrote") == 2 * 6


def test_skills_install_reports_a_conflict_with_exit_1(tmp_path: Path) -> None:
    runner.invoke(app, ["skills", "install", "aif-workflow", "--codex", "--project", str(tmp_path)])
    (tmp_path / ".agents" / "skills" / "aif-workflow" / "SKILL.md").write_text("mine", encoding="utf-8")
    result = runner.invoke(app, ["skills", "install", "aif-workflow", "--codex", "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


# ── Kiro agent registration ───────────────────────────────────────────────────


def test_kiro_install_creates_a_minimal_agent_naming_the_skill(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "aif-workflow", "--kiro", "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    config_path = tmp_path / ".kiro" / "agents" / "ai-functions.json"
    assert f"registered {config_path}" in result.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    skill_md = (tmp_path / ".kiro" / "skills" / "aif-workflow" / "SKILL.md").resolve()
    assert config["name"] == "ai-functions"
    assert config["resources"] == ["file://AGENTS.md", f"skill://{skill_md}"]
    assert "fs_read" in config["allowedTools"]


def test_kiro_registration_appends_to_an_existing_agent_and_keeps_its_keys(tmp_path: Path) -> None:
    agents = tmp_path / ".kiro" / "agents"
    agents.mkdir(parents=True)
    existing = {"name": "mine", "model": "claude", "resources": ["file://README.md"], "mcpServers": {"x": {}}}
    (agents / "mine.json").write_text(json.dumps(existing), encoding="utf-8")

    path, changed = register_kiro_skill(tmp_path, tmp_path / ".kiro" / "skills" / "s" / "SKILL.md", "mine")
    assert (path, changed) == (agents / "mine.json", True)
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["model"] == "claude" and config["mcpServers"] == {"x": {}}
    assert config["resources"][0] == "file://README.md"
    assert config["resources"][1].startswith("skill:///") and config["resources"][1].endswith("/s/SKILL.md")


def test_kiro_registration_is_idempotent(tmp_path: Path) -> None:
    skill_md = tmp_path / ".kiro" / "skills" / "s" / "SKILL.md"
    register_kiro_skill(tmp_path, skill_md)
    path, changed = register_kiro_skill(tmp_path, skill_md)
    assert changed is False
    assert json.loads(path.read_text(encoding="utf-8"))["resources"].count(kiro_skill_uri(skill_md)) == 1


def test_kiro_registration_refuses_a_corrupt_agent(tmp_path: Path) -> None:
    agents = tmp_path / ".kiro" / "agents"
    agents.mkdir(parents=True)
    (agents / "ai-functions.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(KiroAgentConfigError):
        register_kiro_skill(tmp_path, tmp_path / "SKILL.md")
    result = runner.invoke(app, ["skills", "install", "aif-workflow", "--kiro", "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output


def test_kiro_agent_option_picks_the_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["skills", "install", "--all", "--kiro", "--kiro-agent", "kiro-workflows", "--project", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / ".kiro" / "agents" / "kiro-workflows.json").read_text(encoding="utf-8"))
    assert len([r for r in config["resources"] if r.startswith("skill://")]) == 1
