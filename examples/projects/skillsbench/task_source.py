"""SkillsBenchSource — TaskSource adapter for SkillsBench directories.

Loads tasks from a SkillsBench checkout, sandboxes their input/output files,
and grades agent runs with each task's pytest oracle. Used by
memory_improvement_loop.py.
"""

from __future__ import annotations

import ast
import atexit
import random
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class OracleResult:
    """Result of grading one agent run against the oracle test suite."""

    passed: bool
    score: float
    feedback: str | None = None


@dataclass
class TaskSpec:
    """One benchmark task ready to be given to an agent."""

    task_id: str
    prompt: str
    oracle: Callable[[str], OracleResult]
    expected_skills: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_test_paths(test_file: Path) -> dict[str, str]:
    """Return {constant_name: path} for module-level /root/ string constants."""
    source = test_file.read_text()
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith("/root/"):
                result[target.id] = value.value
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "Path"
                and value.args
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
                and value.args[0].value.startswith("/root/")
            ):
                result[target.id] = value.args[0].value
    return result


def _build_path_map(
    instruction: str,
    test_paths: dict[str, str],
    sandbox: Path,
) -> dict[str, str]:
    """Return {original_path: sandbox_path} for every /root/ path."""
    raw: set[str] = set(re.findall(r"/root/[\w./\-]+", instruction))
    raw.update(test_paths.values())
    result: dict[str, str] = {}
    for path in raw:
        name = Path(path).name
        if "input" in path.lower():
            result[path] = str(sandbox / "input" / name)
        else:
            result[path] = str(sandbox / "output" / name)
    return result


def _make_oracle(
    task_dir: Path,
    test_paths: dict[str, str],
    path_map: dict[str, str],
) -> Callable[[str], OracleResult]:
    """Return an oracle that grades a run by executing test_outputs.py."""

    def oracle(_transcript: str) -> OracleResult:
        test_file = task_dir / "tests" / "test_outputs.py"
        if not test_file.exists():
            return OracleResult(
                passed=False,
                score=0.0,
                feedback=f"No test_outputs.py found in {task_dir / 'tests'}",
            )

        for original, sandbox_path in path_map.items():
            if "input" not in original.lower():
                Path(sandbox_path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy2(test_file, tmp_path / "test_outputs.py")

            lines = [
                "from pathlib import Path\n",
                "def pytest_collection_modifyitems(session, config, items):\n",
                "    import test_outputs\n",
            ]
            for const_name, original_path in test_paths.items():
                sandbox_path = path_map.get(original_path, original_path)
                lines.append(f"    test_outputs.{const_name} = Path({sandbox_path!r})\n")
            (tmp_path / "conftest.py").write_text("".join(lines))

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(tmp_path / "test_outputs.py"),
                    "--tb=short",
                    "-q",
                    "--no-header",
                ],
                capture_output=True,
                text=True,
                cwd=tmp,
            )

        stdout = proc.stdout
        passed_m = re.search(r"(\d+) passed", stdout)
        failed_m = re.search(r"(\d+) failed", stdout)
        error_m = re.search(r"(\d+) error", stdout)
        n_passed = int(passed_m.group(1)) if passed_m else 0
        n_failed = int(failed_m.group(1)) if failed_m else 0
        n_error = int(error_m.group(1)) if error_m else 0
        total = n_passed + n_failed + n_error
        score = n_passed / total if total > 0 else 0.0
        run_passed = n_failed == 0 and n_error == 0 and total > 0
        feedback: str | None = (stdout + proc.stderr).strip() or None
        return OracleResult(passed=run_passed, score=score, feedback=feedback)

    return oracle


# ---------------------------------------------------------------------------
# SkillsBenchSource
# ---------------------------------------------------------------------------


class SkillsBenchSource:
    """Load tasks from a SkillsBench benchmark directory.

    Each ``<benchmark_dir>/tasks/<task_id>/`` sub-directory becomes a
    :class:`TaskSpec`.  Input files are copied into a per-task sandbox at
    construction time; ``/root/`` paths in the instruction are rewritten to
    point into the sandbox.  Skills are available via :meth:`install_skills`.

    Args:
        benchmark_dir: Root of a SkillsBench checkout; must contain ``tasks/``.
        sandbox_root: Parent for per-task sandboxes.  Defaults to a temp dir
            cleaned up on process exit.
        task_ids: When given, only load tasks in this set.
    """

    def __init__(
        self,
        benchmark_dir: Path,
        *,
        sandbox_root: Path | None = None,
        task_ids: frozenset[str] | None = None,
    ) -> None:
        if not benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
        tasks_dir = benchmark_dir / "tasks"
        if not tasks_dir.exists():
            raise FileNotFoundError(f"No tasks/ sub-directory in {benchmark_dir}")

        if sandbox_root is None:
            tmpdir = tempfile.mkdtemp(prefix="aifunc-skillsbench-")
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
            self._sandbox_root = Path(tmpdir)
        else:
            self._sandbox_root = sandbox_root

        task_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())
        if task_ids is not None:
            task_dirs = [d for d in task_dirs if d.name in task_ids]
        if not task_dirs:
            raise FileNotFoundError(f"No matching task directories found in {tasks_dir}")

        self._skills: dict[str, dict[str, str]] = {}
        for task_dir in task_dirs:
            skills_dir = task_dir / "environment" / "skills"
            task_skills: dict[str, str] = {}
            if skills_dir.exists():
                for skill_dir in sorted(skills_dir.iterdir()):
                    if skill_dir.is_dir():
                        skill_md = skill_dir / "SKILL.md"
                        if skill_md.exists():
                            task_skills[skill_dir.name] = skill_md.read_text()
            self._skills[task_dir.name] = task_skills

        self._tasks: list[TaskSpec] = [self._parse(d) for d in task_dirs]

    def _parse(self, task_dir: Path) -> TaskSpec:
        task_id = task_dir.name
        sandbox = self._sandbox_root / task_id

        env_input_dir = task_dir / "environment" / "input"
        input_dir = sandbox / "input"
        output_dir = sandbox / "output"
        if env_input_dir.is_dir():
            shutil.copytree(env_input_dir, input_dir, dirs_exist_ok=True)
        else:
            for f in sorted((task_dir / "environment").iterdir()):
                if f.is_file() and f.name.startswith("input"):
                    input_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, input_dir / f.name)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        instruction = (task_dir / "instruction.md").read_text()
        test_file = task_dir / "tests" / "test_outputs.py"
        test_paths = _scan_test_paths(test_file) if test_file.exists() else {}
        path_map = _build_path_map(instruction, test_paths, sandbox)

        prompt = instruction
        for original, replacement in sorted(path_map.items(), key=lambda x: len(x[0]), reverse=True):
            prompt = prompt.replace(original, replacement)

        expected_skills = frozenset(self._skills.get(task_id, {}).keys())

        metadata: dict[str, object] = {}
        task_toml = task_dir / "task.toml"
        if task_toml.exists():
            with open(task_toml, "rb") as f:
                metadata = tomllib.load(f)  # pyright: ignore[reportAssignmentType]

        return TaskSpec(
            task_id=task_id,
            prompt=prompt,
            oracle=_make_oracle(task_dir, test_paths, path_map),
            expected_skills=expected_skills,
            metadata=metadata,
        )

    def all(self) -> list[TaskSpec]:
        """Return all loaded tasks."""
        return list(self._tasks)

    def sample(self, n: int) -> list[TaskSpec]:
        """Return ``n`` tasks sampled uniformly without replacement."""
        if n > len(self._tasks):
            raise ValueError(f"Cannot sample {n} tasks from {len(self._tasks)} available")
        return random.sample(self._tasks, n)

    def skill_content(self, task_id: str, skill_name: str) -> str:
        """Return the text of a skill's SKILL.md file."""
        if task_id not in self._skills:
            raise KeyError(f"Task {task_id!r} not in this source")
        skills = self._skills[task_id]
        if skill_name not in skills:
            raise KeyError(f"Skill {skill_name!r} not found for task {task_id!r}")
        return skills[skill_name]

    def install_skills(self, target_dir: Path) -> None:
        """Write each skill's SKILL.md into target_dir/<skill_name>/SKILL.md."""
        target_dir.mkdir(parents=True, exist_ok=True)
        for task_skills in self._skills.values():
            for skill_name, content in task_skills.items():
                skill_dir = target_dir / skill_name
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(content)
