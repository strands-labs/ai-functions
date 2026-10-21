"""``uv run python graphs.py skills`` — list and install the example's workflow skill.

The workflow skill (``aif-workflow``) lives beside this script. A coding assistant only sees a
skill once it sits in that assistant's own tree inside the project, so this command
copies a skill there::

    uv run python graphs.py skills list
    uv run python graphs.py skills install aif-workflow --claude            # → ./.claude/skills/aif-workflow/
    uv run python graphs.py skills install aif-workflow --kiro              # → ./.kiro/skills/aif-workflow/
    uv run python graphs.py skills install aif-workflow --codex             # → ./.agents/skills/aif-workflow/
    uv run python graphs.py skills install --all --claude --kiro --project ~/work/repo

Behaviour
---------
- **A copy, nothing cleverer.** Every file under the skill's folder is copied,
  ``SKILL.md`` and its ``runtime/`` alike, so the installed skill is self-contained
  and its instructions can say "the ``runtime/`` folder beside this file".
- **Refuses to clobber.** A destination file whose content differs from the
  example's is a conflict: nothing is written and the command exits 1 naming the
  files, unless ``--force`` is given. Identical files are left alone, so re-running
  after updating the example writes exactly the files that changed.
- **Kiro needs registering.** Kiro CLI loads skills only when an agent config in
  ``.kiro/agents/`` names them as ``skill:///abs/path/SKILL.md`` resources, so the
  ``--kiro`` target also appends that line to ``.kiro/agents/<agent>.json``
  (``--kiro-agent``, default ``ai-functions``), creating a minimal agent when none
  exists. Re-running is a no-op once the line is present.
- **Experimental.** The skills' location inside this example and the set of
  targets may change; the command's surface is meant to stay.
"""

# pyright: reportUnusedFunction=false
# Typer registers each @skills_app.command-decorated function by side effect.
# ruff: noqa: B008

from __future__ import annotations

import enum
import filecmp
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import typer


class Target(enum.StrEnum):
    """Where each assistant looks for skills, relative to the project root."""

    CLAUDE = ".claude/skills"
    KIRO = ".kiro/skills"
    CODEX = ".agents/skills"


class SkillNotFoundError(LookupError):
    """The named skill is not bundled with this example."""


class SkillConflictError(RuntimeError):
    """A destination file differs from the example's copy and ``force`` was not given."""

    def __init__(self, conflicts: list[Path]) -> None:
        """Record the conflicting files and list them in the message.

        Args:
            conflicts: Destination files whose content differs from the bundled copy.
        """
        self.conflicts = conflicts
        listed = "\n".join(f"  {path}" for path in conflicts)
        super().__init__(f"these files differ from the bundled skill; pass --force to overwrite:\n{listed}")


@dataclass
class InstallReport:
    """What one install did, file by file."""

    skill: str
    destination: Path
    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)


def skills_root() -> Path:
    """Find the bundled skill beside this script, independently of the working directory.

    Returns:
        The directory holding this module, where each bundled skill is a subfolder.
    """
    return Path(__file__).resolve().parent


def available() -> list[str]:
    """Return the names of every bundled skill, alphabetically.

    Returns:
        Names of the subfolders beside this module that contain a ``SKILL.md``.
    """
    root = skills_root()
    if not root.is_dir():
        return []
    # A folder is a skill only if it carries the instructions file.
    return sorted(path.name for path in root.iterdir() if (path / "SKILL.md").is_file())


def _skill_files(source: Path) -> list[Path]:
    """Return every file under a skill folder, relative to it, skipping bytecode caches.

    Args:
        source: The skill folder to list.

    Returns:
        Relative paths of all files beneath ``source``, sorted, excluding ``__pycache__``.
    """
    return sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )


def install(name: str, target: Target, project: Path, *, force: bool = False) -> InstallReport:
    """Copy one bundled skill into ``project``'s tree for ``target``.

    Args:
        name: A skill name from :func:`available`.
        target: Which assistant's skills folder to write into.
        project: The project root; the target folder is created under it.
        force: Overwrite files that differ instead of refusing.

    Returns:
        Which files were written and which were already identical.

    Raises:
        SkillNotFoundError: ``name`` is not a bundled skill.
        SkillConflictError: A destination file differs and ``force`` is false.
            Nothing has been written when this is raised.
    """
    source = skills_root() / name
    if not (source / "SKILL.md").is_file():
        raise SkillNotFoundError(f"no bundled skill named {name!r}; available: {', '.join(available()) or 'none'}")
    destination = project.expanduser().resolve() / target.value / name
    report = InstallReport(skill=name, destination=destination)

    # Find every conflict before writing, so a refused install writes nothing at all.
    files = _skill_files(source)
    conflicts = [
        destination / relative
        for relative in files
        if (destination / relative).exists()
        and not filecmp.cmp(source / relative, destination / relative, shallow=False)
    ]
    if conflicts and not force:
        raise SkillConflictError(conflicts)

    for relative in files:
        src, dst = source / relative, destination / relative
        # An existing file that is not a conflict is already identical; leave it alone.
        if dst.exists() and dst not in conflicts:
            report.unchanged.append(dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        report.written.append(dst)
    return report


DEFAULT_KIRO_AGENT = "ai-functions"
"""Agent config the ``--kiro`` target registers skills in when ``--kiro-agent`` is not given."""


class KiroAgentConfigError(RuntimeError):
    """The Kiro agent config exists but cannot be read as a JSON object."""


def kiro_skill_uri(skill_md: Path) -> str:
    """Return the ``skill://`` resource URI Kiro expects for a SKILL.md, always absolute.

    Args:
        skill_md: Path to an installed ``SKILL.md``; ``~`` and relative paths are resolved.

    Returns:
        ``skill://`` followed by the absolute path, e.g. ``skill:///repo/.kiro/skills/x/SKILL.md``.
    """
    return f"skill://{skill_md.expanduser().resolve()}"


def register_kiro_skill(project: Path, skill_md: Path, agent: str = DEFAULT_KIRO_AGENT) -> tuple[Path, bool]:
    """Ensure ``.kiro/agents/<agent>.json`` lists ``skill_md`` among its resources.

    Kiro discovers workspace agents from ``.kiro/agents/*.json`` only, and loads a
    skill only when the agent's ``resources`` names its SKILL.md by absolute path.

    Args:
        project: The project root holding ``.kiro/``.
        skill_md: The installed skill's SKILL.md.
        agent: The agent config's name, without ``.json``.

    Returns:
        The config path and whether it was changed.

    Raises:
        KiroAgentConfigError: The config exists but is not a JSON object.
    """
    config_path = project.expanduser().resolve() / ".kiro" / "agents" / f"{agent}.json"
    uri = kiro_skill_uri(skill_md)
    # Load the existing agent config, or start a minimal one when there is none.
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KiroAgentConfigError(f"{config_path} is not valid JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise KiroAgentConfigError(f"{config_path} does not hold a JSON object")
    else:
        config = _minimal_kiro_agent(agent)
    # Append the skill once; an already-registered skill leaves the file untouched.
    resources = config.setdefault("resources", [])
    if not isinstance(resources, list):
        raise KiroAgentConfigError(f"{config_path}: 'resources' is not a list")
    if uri in resources:
        return config_path, False
    resources.append(uri)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, True


def _minimal_kiro_agent(name: str) -> dict[str, object]:
    """Build a workspace agent that can read the project and use the registered skills.

    Reads (``fs_read``, ``glob``, ``grep``) are pre-approved; writes and commands still
    ask for permission unless Kiro is started with ``--trust-all-tools``.

    Args:
        name: The agent's name, matching its config file's stem.

    Returns:
        The agent config as a JSON-serialisable dict, loading ``AGENTS.md`` as a resource.
    """
    return {
        "name": name,
        "description": "Generate and run verified AI workflows with the aif-workflow skill.",
        "prompt": None,
        "tools": ["*"],
        "allowedTools": ["fs_read", "glob", "grep"],
        "resources": ["file://AGENTS.md"],
        "includeMcpJson": True,
    }


# ── Typer surface ─────────────────────────────────────────────────────────────

skills_app: typer.Typer = typer.Typer(
    help="(experimental) List and install the packaged workflow skills into a project.",
    no_args_is_help=True,
)


@skills_app.command("list")
def _list() -> None:
    """List the skills bundled with this example."""
    names = available()
    if not names:
        typer.echo("no bundled skills found", err=True)
        raise typer.Exit(1)
    for name in names:
        first_line = _description(skills_root() / name / "SKILL.md")
        typer.echo(f"{name:<16} {first_line}")


def _description(skill_md: Path) -> str:
    """Return the ``description:`` line of a SKILL.md front matter, truncated.

    Args:
        skill_md: The ``SKILL.md`` to read.

    Returns:
        The description, cut to 100 characters with an ellipsis, or ``""`` when absent.
    """
    for line in skill_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("description:"):
            text = line.removeprefix("description:").strip()
            return text if len(text) <= 100 else text[:99] + "…"
    return ""


@skills_app.command("install")
def _install(
    names: list[str] = typer.Argument(None, help="Skills to install; see `uv run python graphs.py skills list`."),
    all_skills: bool = typer.Option(False, "--all", help="Install every bundled skill."),
    claude: bool = typer.Option(False, "--claude", help="Install into .claude/skills/ (Claude Code)."),
    kiro: bool = typer.Option(False, "--kiro", help="Install into .kiro/skills/ (Kiro CLI)."),
    codex: bool = typer.Option(False, "--codex", help="Install into .agents/skills/ (Codex)."),
    project: Path = typer.Option(Path.cwd(), "--project", help="Project root to install into."),
    force: bool = typer.Option(False, "--force", help="Overwrite files that differ from the example's copy."),
    kiro_agent: str = typer.Option(
        DEFAULT_KIRO_AGENT, "--kiro-agent", help="Kiro agent config (.kiro/agents/<name>.json) to register skills in."
    ),
) -> None:
    """Install one or more bundled skills into a project's assistant tree."""
    # Validate the flags before touching the filesystem; exit code 2 is a usage error.
    targets = [t for t, chosen in ((Target.CLAUDE, claude), (Target.KIRO, kiro), (Target.CODEX, codex)) if chosen]
    if not targets:
        typer.echo("error: pick at least one target: --claude, --kiro, --codex", err=True)
        raise typer.Exit(2)
    chosen = available() if all_skills else list(names or [])
    if not chosen:
        typer.echo("error: name a skill or pass --all", err=True)
        raise typer.Exit(2)
    raise typer.Exit(install_many(chosen, targets, project, force=force, kiro_agent=kiro_agent))


def install_many(
    names: list[str],
    targets: list[Target],
    project: Path,
    *,
    force: bool = False,
    kiro_agent: str = DEFAULT_KIRO_AGENT,
) -> int:
    """Install each skill into each target, printing a line per file; return an exit code.

    A failure for one skill or target is reported and the rest still install, so one
    conflict does not hide the outcome of the others.

    Args:
        names: Bundled skill names to install.
        targets: Assistant trees to install into.
        project: The project root the targets are created under.
        force: Overwrite files that differ instead of refusing.
        kiro_agent: Kiro agent config to register skills in for the Kiro target.

    Returns:
        0 when every install succeeded, 1 when any of them failed.
    """
    code = 0
    for name in names:
        for target in targets:
            try:
                # Kiro also needs the skill named in an agent config before it loads it.
                report = install(name, target, project, force=force)
                registered = (
                    register_kiro_skill(project, report.destination / "SKILL.md", kiro_agent)
                    if target is Target.KIRO
                    else None
                )
            except (SkillNotFoundError, SkillConflictError, KiroAgentConfigError) as exc:
                typer.echo(f"error: {exc}", err=True)
                code = 1
                continue
            # Report each written file, but only a count of the unchanged ones.
            for path in report.written:
                typer.echo(f"wrote     {path}")
            if report.unchanged:
                typer.echo(f"unchanged {len(report.unchanged)} file(s) under {report.destination}")
            if registered is not None:
                config_path, changed = registered
                typer.echo(f"{'registered' if changed else 'already in'} {config_path}")
    return code


__all__ = [
    "DEFAULT_KIRO_AGENT",
    "InstallReport",
    "KiroAgentConfigError",
    "SkillConflictError",
    "SkillNotFoundError",
    "Target",
    "available",
    "install",
    "install_many",
    "kiro_skill_uri",
    "register_kiro_skill",
    "skills_app",
    "skills_root",
]
