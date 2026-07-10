"""SkillsBench improvement loop — self-improving agent via textual gradients.

Runs five SkillsBench tasks with ten skills available (four relevant + six
distractors), built on the ai-functions native API:

- ``JSONMemoryBackend``   replaces the old plain-markdown AgentMemory
- ``TextGradOptimizer``   replaces the custom TextGrad wrapper
- Synthetic computation graph (``ParameterNode`` + ``ThreadNode``) substitutes
  for the ``@ai_function.trace()`` graph, because tasks run via ``ClaudeAgent``
  (not an ``@ai_function``).

Pipeline::

    SkillsBenchSource (5 tasks, 10 skills)
          │
          ▼
    ClaudeAgent  (one thread per task, run in parallel)
          │
          ▼
    Oracle  (pytest test_outputs.py against sandbox output files)
          │
          ▼
    TextGradOptimizer.backward()   (one call per task → synthetic graph)
    TextGradOptimizer.consolidate() (merge all gradients → memory update)
          │
          ▼
    JSONMemoryBackend  (workflow_guidelines updated for round 1)

The mental model is PyTorch autograd:

- ``recall()``      ≈ reading a learnable weight into the forward pass
- ``backward()``    ≈ ``loss.backward()`` — accumulates textual gradients
- ``consolidate()`` ≈ ``optimizer.step()`` — writes improvements to memory

Setup and usage: see README.md in this directory.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from task_source import OracleResult, SkillsBenchSource, TaskSpec

from ai_functions import (
    InMemoryCoordinator,
    JSONMemoryBackend,
    LocalWorker,
    TextGradOptimizer,
)
from ai_functions.claude_code import ClaudeAgent
from ai_functions.types.graph import ParameterNode, ThreadNode

# Bedrock model for memory consolidation and backward pass
_OPT_MODEL = "us.anthropic.claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Display helpers (same look as the top-level examples' _utils.py)
# ---------------------------------------------------------------------------

_console = Console()


def display(title: str, content: str, lang: str = "markdown") -> None:
    """Render ``content`` in a titled panel with syntax highlighting."""
    body = Syntax(content, lang, theme="monokai", word_wrap=True)
    _console.print(Panel(body, title=title, border_style="cyan", expand=True))


def rule(title: str) -> None:
    """Print a horizontal divider to mark a step."""
    _console.print(Rule(title, style="cyan"))


# ---------------------------------------------------------------------------
# Task and skill configuration
# ---------------------------------------------------------------------------

_TARGET_TASKS: dict[str, list[str]] = {
    "exceltable-in-ppt": ["pptx", "xlsx"],
    "financial-modeling-qa": ["pdf", "xlsx"],
    "offer-letter-generator": ["docx"],
    "pptx-reference-formatting": ["pptx"],
    "court-form-filling": ["pdf"],
}

_DISTRACTOR_SOURCES: dict[str, list[str]] = {
    "edit-pdf": ["pdf-editing", "text-parser"],
    "jpg-ocr-stat": ["image-ocr"],
    "latex-formula-extraction": ["marker"],
    "invoice-fraud-detection": ["fuzzy-match"],
    "xlsx-recover-data": ["data-reconciliation"],
}

_ALL_SKILLS: list[str] = sorted(
    {skill for skills in _TARGET_TASKS.values() for skill in skills}
    | {skill for skills in _DISTRACTOR_SOURCES.values() for skill in skills}
)

# ---------------------------------------------------------------------------
# Memory schema
# ---------------------------------------------------------------------------


class AgentMemory(BaseModel):
    workflow_guidelines: str = Field(
        default="No specific guidelines yet.",
        description=(
            "Task-to-skill mapping and workflow rules for the agent. "
            "Maps each task type to the correct skill(s) to invoke, and notes "
            "any workflow steps (file paths, output formats) the agent must follow. "
            "Updated round-over-round from evaluation feedback. "
            "Keep concise (max 30 lines) and actionable."
        ),
    )


# ---------------------------------------------------------------------------
# Task execution helpers
# ---------------------------------------------------------------------------


def _install_distractor_skills(
    skillsbench: Path,
    distractor_sources: dict[str, list[str]],
    target_dir: Path,
) -> None:
    for source_task, skill_names in distractor_sources.items():
        source = SkillsBenchSource(
            benchmark_dir=skillsbench,
            task_ids=frozenset([source_task]),
        )
        source.install_skills(target_dir=target_dir)
        for skill_name in skill_names:
            print(f"  Installed distractor skill '{skill_name}' (from '{source_task}')")


async def _run_one_task(
    task: TaskSpec,
    agent: ClaudeAgent,
    worker: LocalWorker,
) -> OracleResult:
    """Spawn one ClaudeAgent thread, run the task, return the oracle result."""
    handle = await worker.spawn_locally(agent)
    try:
        response = await handle.run(task.prompt)
    except Exception as exc:
        return OracleResult(passed=False, score=0.0, feedback=repr(exc))
    finally:
        await handle.terminate_now()
    return task.oracle(response)


async def _run_round(
    tasks: list[TaskSpec],
    memory: JSONMemoryBackend,
    worker: LocalWorker,
    skills_cwd: Path,
) -> list[tuple[TaskSpec, OracleResult, ThreadNode]]:
    """Run all tasks concurrently; return (task, oracle, task_node) triples."""
    # Recall guidelines once — all tasks in this round share the same snapshot.
    guidelines_view = await memory.recall("workflow_guidelines")
    guidelines = str(guidelines_view)

    system_prompt: str | None = None
    if guidelines and guidelines.strip() != "No specific guidelines yet.":
        system_prompt = f"# Workflow Guidelines\n\n{guidelines}"

    # One agent template per round; ClaudeAgent is an immutable template so
    # each spawn_locally call creates a fresh independent subprocess.
    agent = ClaudeAgent(
        options=ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            skills=_ALL_SKILLS,
            cwd=str(skills_cwd),
        )
    )

    raw = await asyncio.gather(
        *(_run_one_task(task, agent, worker) for task in tasks),
        return_exceptions=True,
    )

    results: list[tuple[TaskSpec, OracleResult, ThreadNode]] = []
    for task, res in zip(tasks, raw, strict=True):
        oracle = OracleResult(passed=False, score=0.0, feedback=repr(res)) if isinstance(res, Exception) else res
        # Build a synthetic computation graph node for this task.
        # The ParameterNode wraps the shared JSONMemoryBackend so the optimizer
        # can accumulate gradients and consolidate them in one pass at the end.
        param_node = ParameterNode(
            node_id=f"param:{task.task_id}:workflow_guidelines",
            value=guidelines,  # current value — gives backward LLM context
            name="workflow_guidelines",
            backend=memory,
            requires_grad=True,
            description=AgentMemory.model_fields["workflow_guidelines"].description or "",
        )
        task_node = ThreadNode(
            node_id=f"thread:{task.task_id}",
            thread_id=f"thread:{task.task_id}",
            parameters=[param_node],
        )
        results.append((task, oracle, task_node))
    return results


# ---------------------------------------------------------------------------
# Feedback builder
# ---------------------------------------------------------------------------


def _build_feedback(task: TaskSpec, oracle: OracleResult) -> str:
    if oracle.passed:
        return (
            f"Task '{task.task_id}' PASSED (score {oracle.score:.2f}). "
            "The skill selection and workflow were correct. Preserve this approach."
        )
    expected = ", ".join(sorted(task.expected_skills)) if task.expected_skills else "unknown"
    detail = (oracle.feedback or "No oracle details available.")[:1500]
    return (
        f"Task '{task.task_id}' FAILED (score {oracle.score:.2f}).\n\n"
        f"Expected skill(s): {expected}\n\n"
        f"Oracle feedback:\n{detail}\n\n"
        "Update workflow_guidelines to specify which skill(s) this task type "
        "requires and how to complete it correctly."
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_round(
    label: str,
    results: list[tuple[TaskSpec, OracleResult, ThreadNode]],
) -> None:
    n_passed = sum(1 for _, o, _ in results if o.passed)
    rule(f"{label}: {n_passed}/{len(results)} passed")
    for task, oracle, _ in results:
        status = "PASS ✓" if oracle.passed else "FAIL ✗"
        detail = (oracle.feedback or "").strip()[:300]
        body = f"score: {oracle.score:.2f}"
        if detail:
            body += f"\n\n{detail}"
        display(f"[{status}] {task.task_id}", body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(skillsbench: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tasks ────────────────────────────────────────────────────────────
    print(f"Loading {len(_TARGET_TASKS)} tasks from {skillsbench}")
    main_source = SkillsBenchSource(
        benchmark_dir=skillsbench,
        task_ids=frozenset(_TARGET_TASKS.keys()),
    )
    tasks = main_source.all()
    print(f"  Loaded: {[t.task_id for t in tasks]}")

    # ── Install skills ────────────────────────────────────────────────────────
    # Claude discovers skills in .claude/skills/ relative to its CWD.
    # We set cwd=output_dir for the agent, so install there.
    skills_dir = output_dir / ".claude" / "skills"
    main_source.install_skills(target_dir=skills_dir)
    n_target = len({s for ss in _TARGET_TASKS.values() for s in ss})
    print(f"Installed {n_target} target skill(s) to {skills_dir}")

    print("Installing distractor skills:")
    _install_distractor_skills(skillsbench, _DISTRACTOR_SOURCES, skills_dir)
    print(f"\nAll {len(_ALL_SKILLS)} available skills: {_ALL_SKILLS}")

    # ── Memory + optimizer ────────────────────────────────────────────────────
    memory_path = output_dir / "memory.json"
    memory = JSONMemoryBackend(AgentMemory, actor_id="demo", path=memory_path, model=_OPT_MODEL)
    optimizer = TextGradOptimizer(model=_OPT_MODEL)

    display("Initial Memory", str(memory))

    # ── Coordinator + worker for ClaudeAgent threads ──────────────────────────
    coordinator = InMemoryCoordinator()
    worker = LocalWorker(coordinator)
    await worker.register()

    try:
        # ── Round 0: no guidelines ────────────────────────────────────────────
        rule("Round 0 — no guidelines")
        round_0 = await _run_round(tasks, memory, worker, skills_cwd=output_dir)
        _print_round("Round 0", round_0)

        # ── Backward: accumulate gradients on each task's ParameterNode ───────
        all_nodes: list[ThreadNode] = []
        n_failing = 0
        for task, oracle, task_node in round_0:
            feedback = _build_feedback(task, oracle)
            # backward() makes a blocking LLM call to route feedback into
            # task_node.parameters[0].gradients (workflow_guidelines).
            optimizer.backward(task_node, feedback)
            all_nodes.append(task_node)
            if not oracle.passed:
                n_failing += 1

        if n_failing == 0:
            rule("All tasks passed in round 0 — nothing to optimize")
            memory.close()
            return

        # Gather all task nodes under a synthetic root so one consolidate()
        # call merges every workflow_guidelines gradient (grouped by backend +
        # name) into a single update of the shared JSONMemoryBackend.
        root = ThreadNode(
            node_id="improvement_loop:consolidate",
            thread_id="improvement_loop:consolidate",
            child_threads=all_nodes,
        )
        for node in all_nodes:
            node.parent = root

        rule(f"Optimizer: consolidating feedback from {n_failing} failing task(s)")
        optimizer.consolidate(root)
        memory.close()

        # ── Round 1: with learned guidelines ──────────────────────────────────
        memory = JSONMemoryBackend(AgentMemory, actor_id="demo", path=memory_path, model=_OPT_MODEL)
        display("Learned Memory", str(memory))

        rule("Round 1 — with learned guidelines")
        round_1 = await _run_round(tasks, memory, worker, skills_cwd=output_dir)
        _print_round("Round 1", round_1)
        memory.close()

    finally:
        await worker.close()
        # terminate_now() schedules each thread's teardown (which disconnects
        # its claude subprocess) as a background task. Wait for those tasks
        # here — otherwise asyncio.run() cancels them mid-disconnect and the
        # orphaned subprocess transports raise "RuntimeError: Event loop is
        # closed" when the garbage collector finalizes them after exit.
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.wait(pending, timeout=30)

    # ── Round-over-round comparison ───────────────────────────────────────────
    rule("Round-over-round comparison")
    header = f"{'Task':<35} {'Round 0':>10} {'Round 1':>10}  Change"
    lines = [header, "-" * len(header)]
    for (task, o0, _), (_, o1, _) in zip(round_0, round_1, strict=True):
        r0 = f"{o0.score:.2f}" + (" ✓" if o0.passed else " ✗")
        r1 = f"{o1.score:.2f}" + (" ✓" if o1.passed else " ✗")
        if not o0.passed and o1.passed:
            change = "  FAIL → PASS ✓"
        elif o0.passed and not o1.passed:
            change = "  PASS → FAIL ✗"
        else:
            change = ""
        lines.append(f"{task.task_id:<35} {r0:>10} {r1:>10}{change}")
    display("Results", "\n".join(lines), lang="text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkillsBench improvement loop demo")
    parser.add_argument(
        "--skillsbench",
        type=Path,
        required=True,
        help="Path to the SkillsBench benchmark directory (must contain a tasks/ sub-directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/aifunc-improvement-demo"),
        help="Directory to save memory.json and installed skill files (default: /tmp/aifunc-improvement-demo)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.skillsbench, args.output_dir))
