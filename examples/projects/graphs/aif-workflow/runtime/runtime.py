"""Workflow runtime — copied verbatim into every generated .aif/<slug>/ package.

This file is the canonical copy; the skill copies it, never rewrites it. That is
deliberate: an agent re-deriving this logic per task gets it silently wrong in
ways nothing catches — a missing ``terminate_now`` leaks a ``kiro-cli``
subprocess, a missing sink makes the run invisible, and a *sync* wrapper around
an async post-condition makes every AI judge pass. Copying keeps each workflow
package self-contained without re-deriving any of that.

It owns everything about *executing* a workflow, so a generated ``workflow.py``
only declares what its steps are:

  - one agent thread per step — Kiro, Claude Code, Codex or the library's own
    ``AIThread`` — on a real coordinator + worker
  - pre-conditions checked before a step spawns (the library has no such hook)
  - post-conditions wrapped so each evaluation lands in the event log
  - every event appended to ``runs/<run-id>/events.jsonl``

Wire format written here and read by ``uv run python graphs.py view`` — keep the two in step:

    .aif/<slug>/manifest.json          the workflow's shape, prompts, condition source
    .aif/<slug>/runs/<run-id>/manifest.json  this run's graph snapshot
    .aif/<slug>/runs/<run-id>/run.json      metadata: task, session, status
    .aif/<slug>/runs/<run-id>/events.jsonl  one JSON event per line
    .aif/<slug>/runs/<run-id>/result.json   each step's outcome
"""

from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import itertools
import json
import logging
import os
import re
import textwrap
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ai_functions.ai_thread.postcondition import PostConditionResult
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import CustomEvent, Event, ThreadId
from ai_functions.types.context import current_thread_scope, thread_scope
from pydantic import Field

RUNTIME_VERSION = 4
"""Bumped when the wire format changes, so the UX can flag a stale copy."""

logger = logging.getLogger("aif.runtime")

try:  # ``cli.print_event`` landed after 0.4.0; keep the released package usable.
    from ai_functions.cli import print_event
except ImportError:  # pragma: no cover — exercised only on an older ai-functions

    def print_event(event: Any) -> None:
        """Log one event as a single line; a stand-in for the library's console printer.

        Args:
            event: Any coordinator event; only ``kind`` and the thread label are read.
        """
        kind = getattr(event, "kind", "?")
        thread = getattr(event, "thread_name", None) or getattr(event, "thread_id", "?")
        logger.info("  %s %s", thread, kind)


Condition = Callable[..., Any]
"""A pre- or post-condition. Returns ``PostConditionResult`` or ``None`` to pass,
raises to fail, and may be async — ``@ai_function`` judges are awaited."""

ConditionKind = Literal["code", "ai"]

# ── Event kinds this runtime adds to a run (read by the workflows TUI) ─────────
PRECONDITION = "aif_precondition"
POSTCONDITION = "aif_postcondition"
STEP_STARTED = "aif_step_started"
STEP_FINISHED = "aif_step_finished"


class WorkflowEvent(CustomEvent):
    """A custom event carrying the runtime's own ``aif_*`` kinds.

    ``Coordinator.append_event`` rejects events without a ``thread_id``. Current
    releases declare routing fields on ``CustomEvent`` and forbid redeclaring them;
    older ones lack them, so they are added here only when missing.
    """

    # Older releases need these fields; current releases forbid redeclaring them.
    if "thread_id" not in CustomEvent.model_fields:
        id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex}")
        timestamp: float = Field(default_factory=time.time)
        thread_id: str | None = None
        thread_name: str | None = None


class Backend(enum.StrEnum):
    """Which agent runs a step's prompt. All four are ``Thread[[str], str]``: a prompt in, text out."""

    KIRO = "kiro"
    """``kiro-cli`` over ACP, with ``auto_approve``. Needs the ``kiro`` extra."""
    CLAUDE = "claude"
    """Claude Code via the Claude Agent SDK, ``permission_mode="bypassPermissions"``. Needs ``claude-code``."""
    CODEX = "codex"
    """OpenAI Codex via its SDK, ``workspace_write`` sandbox. Needs the ``codex`` extra."""
    AI_THREAD = "ai_thread"
    """The library's own Strands-based ``AIThread`` with file and shell tools. Needs
    ``strands-agents-tools`` unless the step supplies ``tools`` itself, and model credentials."""


BACKEND_ENV = "AIF_BACKEND"
"""Environment variable naming the default backend when neither the step nor the run sets one."""

MODEL_ENV = "AIF_MODEL"
"""Environment variable naming the model for every backend; a step's ``model`` overrides it."""

_EXTRAS: dict[Backend, str] = {
    Backend.KIRO: "uv add 'strands-ai-functions[kiro]'",
    Backend.CLAUDE: "uv add 'strands-ai-functions[claude-code]'",
    Backend.CODEX: "uv add 'strands-ai-functions[codex]'",
    Backend.AI_THREAD: "uv add strands-agents-tools   # or pass tools=(...) on the Step",
}


class BackendUnavailableError(ImportError):
    """The chosen backend's optional dependency is not installed."""

    def __init__(self, backend: Backend, cause: BaseException) -> None:
        """Record the backend and build a message naming its install command.

        Args:
            backend: The backend whose dependency failed to import.
            cause: The original ``ImportError``.
        """
        self.backend = backend
        super().__init__(f"backend {backend.value!r} is not available ({cause}). Install it with: {_EXTRAS[backend]}")


def resolve_backend(step: Step | None = None, default: Backend | str | None = None) -> Backend:
    """Pick the backend a step runs on.

    The most specific choice wins: the step's own ``backend``, then the run's
    default, then ``$AIF_BACKEND``, then Kiro.

    Args:
        step: The step being resolved; ``None`` resolves the run's default.
        default: The run's default backend, if one was given.

    Returns:
        The resolved backend.

    Raises:
        ValueError: The chosen name is not a known backend.
    """
    chosen = (step.backend if step is not None else None) or default or os.environ.get(BACKEND_ENV) or Backend.KIRO
    return Backend(chosen)


@dataclass
class Step:
    """One node of a workflow graph: its prompt, its conditions, and what it waits for.

    A step's prompt runs on one of the :class:`Backend` agents — Kiro by default — each
    a ``Thread[[str], str]``: a prompt in, its final answer out. Set ``function`` to run a
    typed ``@ai_function`` instead: it is also a ``Spawnable``, so it hosts on the same
    coordinator, and its return type can be a Pydantic model rather than text. The
    prompt-in, text-out agents cannot do that; their protocols are text conversations.

    Edges are implicit: each name in ``depends_on`` is an edge from that step to this one.

    Attributes:
        name: Unique step name; also the thread name and the ``prompt.md`` section.
        prompt: Text sent to the agent; ``{other_step}`` is replaced by that step's result.
        pre_conditions: Zero-argument checks evaluated before the agent spawns.
        post_conditions: ``(condition, kind)`` pairs checked after each attempt.
        depends_on: Names of steps that must pass before this one starts.
        max_attempts: Attempt budget for satisfying ``post_conditions``.
    """

    name: str
    prompt: str = ""
    pre_conditions: tuple[Condition, ...] = ()
    post_conditions: tuple[tuple[Condition, ConditionKind], ...] = ()
    depends_on: tuple[str, ...] = ()
    max_attempts: int = 4
    function: Any = None
    """A typed ``AIFunction`` to run instead of a prompt-driven agent. Its post-conditions and
    attempt budget are taken from this Step, so both kinds are declared the same way."""
    arguments: dict[str, Any] = field(default_factory=dict[str, Any])
    """Keyword arguments for a typed ``function`` whose signature is not one prompt."""
    backend: Backend | str | None = None
    """Which agent runs this step's prompt; ``None`` defers to the run, then ``$AIF_BACKEND``, then Kiro."""
    model: str | None = None
    """Model for this step's agent; ``None`` defers to ``$AIF_MODEL``, then the backend's default."""
    tools: tuple[Any, ...] = ()
    """Tools for the ``ai_thread`` backend; empty means the default file and shell tools."""

    @property
    def is_typed(self) -> bool:
        """Whether this step runs a typed function rather than a prompt-driven agent."""
        return self.function is not None

    @property
    def thread_id(self) -> ThreadId:
        """Stable per-step thread id, so conditions can emit routed events."""
        return ThreadId(f"th-{self.name}")


@dataclass
class StepResult:
    """How a step ended, including why it never ran when it did not.

    Attributes:
        name: The step's name.
        status: ``passed``, ``failed``, ``blocked`` (a pre-condition failed) or
            ``skipped`` (a dependency did not pass).
        result: The agent's final answer, stringified, when the step passed.
        blocked_by: Failed pre-conditions, or unmet dependencies when skipped.
        error: The exception text when the step failed.
    """

    name: str
    status: str
    result: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    error: str | None = None


# ── Conditions ────────────────────────────────────────────────────────────────


def condition_name(condition: Condition) -> str:
    """Return the best display name for a condition callable.

    Args:
        condition: A function, an ``@ai_function`` or any other callable.

    Returns:
        Its ``__name__``, else its ``name``, else its type's name.
    """
    for attribute in ("__name__", "name"):
        value = getattr(condition, attribute, None)
        if isinstance(value, str):
            return value
    return type(condition).__name__


async def evaluate(condition: Condition, *args: object, **kwargs: object) -> tuple[bool, str | None]:
    """Call a condition and normalize its outcome to ``(passed, message)``.

    Every way a condition can report is accepted: returning ``None`` passes,
    a ``PostConditionResult`` or ``bool`` is taken at its word, raising fails
    with the exception text, and any other return value fails. A coroutine is
    awaited first, which is what makes ``@ai_function`` judges work.

    Args:
        condition: The pre- or post-condition to call.
        *args: Positional arguments; a post-condition receives the step's result.
        **kwargs: Keyword arguments forwarded unchanged.

    Returns:
        Whether it passed, and a failure message when it did not.
    """
    try:
        outcome = condition(*args, **kwargs)
        # An ``@ai_function`` judge returns a coroutine; an un-awaited one would be truthy.
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
    except Exception as exc:  # noqa: BLE001 — any failure is a failed condition
        return False, str(exc)
    # Map each supported return shape onto (passed, message).
    if outcome is None:
        return True, None
    if isinstance(outcome, PostConditionResult):
        return outcome.passed, outcome.message
    if isinstance(outcome, bool):
        return outcome, None if outcome else f"{condition_name(condition)} returned False"
    return False, f"{condition_name(condition)} returned {type(outcome).__name__}"


def _emit(event: WorkflowEvent) -> None:
    """Append ``event`` to the ambient thread's log; drop it if there is no scope.

    The worker opens a ``thread_scope`` around every cycle, so a post-condition
    can recover its coordinator and thread id with no wiring. Losing an
    observability event must never fail a workflow, so this warns rather than
    raising.

    Args:
        event: The event to record; its ``thread_id`` is set from the scope.
    """
    # A condition receives only the result string, so the scope is its one route to the log.
    scope = current_thread_scope()
    if scope is None or scope.thread_id is None:
        logger.warning("dropping %s: no ambient thread scope", event.kind)
        return
    scope.coordinator.append_event(event.model_copy(update={"thread_id": scope.thread_id}))


def observed(condition: Condition, *, step: str, kind: ConditionKind) -> Condition:
    """Wrap a post-condition so every evaluation is recorded as an event.

    Always async: ``evaluate_post_conditions`` awaits a coroutine return value, so
    an async wrapper covers plain predicates and ``@ai_function`` judges alike — a
    sync wrapper would treat an un-awaited judge coroutine as a pass. Attempts are
    counted per wrapper, which is what gives the UX its attempt numbers.

    Args:
        condition: The post-condition to wrap.
        step: Name of the step it belongs to, recorded on each event.
        kind: ``"code"`` or ``"ai"``, recorded for display only.

    Returns:
        An async condition with the same name and docstring as ``condition``.

    Emits:
        aif_postcondition — one per evaluation, with the attempt number and verdict.
    """
    # One counter per wrapper: the library calls this once per attempt.
    attempts = itertools.count(1)
    name = condition_name(condition)

    @functools.wraps(condition)
    async def wrapper(result: Any, **bound: Any) -> PostConditionResult:
        # ``AIThread`` passes the cycle's call arguments to conditions whose parameter
        # names match, so the wrapper forwards whatever it is handed.
        attempt = next(attempts)
        passed, message = await evaluate(condition, result, **bound)
        # Record the verdict before returning it, so the log matches what the agent sees.
        _emit(
            WorkflowEvent(
                kind=POSTCONDITION,
                thread_name=step,
                payload={
                    "step": step,
                    "condition": name,
                    "condition_kind": kind,
                    "attempt": attempt,
                    "passed": passed,
                    "message": message,
                },
            ),
        )
        # The library feeds this message back to the agent as its next turn.
        if passed:
            return PostConditionResult(passed=True)
        return PostConditionResult(passed=False, message=message or f"{name} failed")

    return wrapper


async def check_pre_conditions(step: Step) -> list[str]:
    """Evaluate every pre-condition, emitting one event each; return the failures.

    Evaluates all of them rather than stopping at the first, so the UX shows
    everything that blocked a step. The library has no pre-condition hook; these
    run before the step's thread is spawned.

    Args:
        step: The step about to run.

    Returns:
        Names of the pre-conditions that failed; empty means the step may start.

    Emits:
        aif_precondition — one per pre-condition, on the step's thread id.
    """
    failures: list[str] = []
    for condition in step.pre_conditions:
        passed, message = await evaluate(condition)
        name = condition_name(condition)
        if not passed:
            failures.append(name)
        _emit(
            WorkflowEvent(
                kind=PRECONDITION,
                thread_id=str(step.thread_id),
                thread_name=step.name,
                payload={"step": step.name, "condition": name, "passed": passed, "message": message},
            ),
        )
    return failures


# ── Package layout and prompts ────────────────────────────────────────────────


def package_dir() -> Path:
    """Return the ``.aif/<slug>/`` directory this runtime was copied into.

    The runtime locates its package from its own file, which is why it is copied
    into every package rather than shared.
    """
    return Path(__file__).resolve().parent


def prompt_section(name: str, *, path: Path | None = None) -> str:
    """Read one ``## <name>`` section out of the package's ``prompt.md``.

    Args:
        name: The step name, matching a ``## <name>`` heading.
        path: The prompt file; defaults to this package's ``prompt.md``.

    Returns:
        The section's text, stripped, up to the next ``## `` heading.

    Raises:
        KeyError: The file has no section for ``name``.
    """
    source = path or package_dir() / "prompt.md"
    text = source.read_text(encoding="utf-8")
    # Everything after the heading up to the next ``## `` heading or the end of the file.
    match = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise KeyError(f"{source.name} has no section for step {name!r}")
    return match.group(1).strip()


def session_label() -> str:
    """Label which shell this run came from, so concurrent terminals stay distinguishable.

    Returns:
        ``$AIF_SESSION`` if set, else the controlling terminal, else the process id.
    """
    explicit = os.environ.get("AIF_SESSION")
    if explicit:
        return explicit
    try:
        return f"terminal {Path(os.ttyname(0)).name}"
    except OSError:
        return f"process {os.getpid()}"


def describe_condition(condition: Condition, kind: str) -> dict[str, str]:
    """Everything the UX shows about a condition: its docstring, type and source.

    The signature and source are read off the live callable, so what the UX displays
    is what actually ran — no separate declaration to drift out of step.

    Args:
        condition: The condition callable to describe.
        kind: ``"code"`` or ``"ai"``.

    Returns:
        Its name, kind, first docstring line, full docstring, signature and
        source (truncated to 4000 characters), as written to ``manifest.json``.
    """
    doc = (condition.__doc__ or "").strip()
    # Prefer resolved annotations; fall back to the raw signature, then to a placeholder.
    try:
        # eval_str resolves the string annotations that ``from __future__ import
        # annotations`` leaves behind, so the reader sees `str` not `'str'`. Module
        # paths are then stripped: `PostConditionResult` reads better than its FQN.
        rendered = str(inspect.signature(condition, eval_str=True))
        signature = f"{condition_name(condition)}{re.sub(r'[A-Za-z_][A-Za-z0-9_.]*\.', '', rendered)}"
    except (TypeError, ValueError, NameError):
        try:
            signature = f"{condition_name(condition)}{inspect.signature(condition)}"
        except (TypeError, ValueError):
            signature = f"{condition_name(condition)}(...)"
    # Built-ins and some wrappers have no retrievable source; show none rather than fail.
    try:
        source = textwrap.dedent(inspect.getsource(condition)).strip()
    except (OSError, TypeError):
        source = ""
    return {
        "name": condition_name(condition),
        "kind": kind,
        "description": doc.splitlines()[0] if doc else "",
        "docstring": doc,
        "signature": signature,
        "source": source[:4000],
    }


def _type_name(annotation: object) -> str:
    """Render a type annotation the way a reader would write it.

    Args:
        annotation: A type, a typing construct, or ``Signature.empty``.

    Returns:
        A short name such as ``TriageReport`` or ``list[str]``, with module paths stripped.
    """
    if annotation is inspect.Signature.empty or annotation is None:
        return "None"
    name = getattr(annotation, "__name__", None)
    return str(name) if isinstance(name, str) else re.sub(r"[A-Za-z_][A-Za-z0-9_.]*\.", "", str(annotation))


def _input_type(step: Step) -> str:
    """Describe what a step consumes, for the node's type contract in the viewer.

    Args:
        step: The step to describe.

    Returns:
        ``"str"`` for a prompt-driven agent, or the typed function's parameters.
    """
    if not step.is_typed:
        return "str"
    try:
        signature = inspect.signature(step.function.prompt_fn)
    except (AttributeError, TypeError, ValueError):
        return "..."
    return (
        ", ".join(f"{name}: {_type_name(parameter.annotation)}" for name, parameter in signature.parameters.items())
        or "()"
    )


def _output_type(step: Step) -> str:
    """Describe what a step returns, for the node's type contract and its outgoing edges.

    Args:
        step: The step to describe.

    Returns:
        ``"str"`` for a prompt-driven agent, or the typed function's return type.
    """
    if not step.is_typed:
        return "str"
    try:
        return _type_name(inspect.signature(step.function.prompt_fn).return_annotation)
    except (AttributeError, TypeError, ValueError):
        return "?"


def _backend_name(step: Step, default: Backend) -> str:
    """Name what runs a step, as recorded for the viewer.

    Args:
        step: The step to describe.
        default: The run's default backend.

    Returns:
        The resolved backend's name, or ``"ai_function"`` for a typed step.
    """
    return "ai_function" if step.is_typed else resolve_backend(step, default).value


def _manifest(
    slug: str, task: str, objective: str, workspace: Path, steps: Sequence[Step], backend: Backend
) -> dict[str, Any]:
    """Build the workflow's shape, as the viewer reads it from ``manifest.json``.

    The viewer rebuilds the graph from each step's ``depends_on``, and shows each
    condition's docstring, signature and source from this record.

    Args:
        slug: The package name.
        task: The user's request, verbatim.
        objective: What success means.
        workspace: The resolved directory the agents work in.
        steps: The steps, in declaration order.
        backend: The run's default backend.

    Returns:
        A JSON-serializable manifest.
    """
    return {
        "runtime_version": RUNTIME_VERSION,
        "slug": slug,
        "task": task,
        "objective": objective,
        "workspace": str(workspace),
        "backend": backend.value,
        "steps": [
            {
                "name": s.name,
                "prompt": s.prompt,
                "thread_id": str(s.thread_id),
                "backend": _backend_name(s, backend),
                # An agent thread is ``Thread[[str], str]``; a typed function reports its
                # own annotations, so the UX shows the real contract either way.
                "input_type": _input_type(s),
                "output_type": _output_type(s),
                "pre_conditions": [describe_condition(c, "code") for c in s.pre_conditions],
                "post_conditions": [describe_condition(c, k) for c, k in s.post_conditions],
                "depends_on": list(s.depends_on),
                "max_attempts": s.max_attempts,
            }
            for s in steps
        ],
    }


# ── The run ───────────────────────────────────────────────────────────────────


async def run_workflow(
    *,
    task: str,
    objective: str,
    steps: Sequence[Step],
    workspace: Path,
    slug: str | None = None,
    echo: bool = True,
    record_into: Path | None = None,
    spawn: Callable[[Step, Path, tuple[Condition, ...]], Any] | None = None,
    backend: Backend | str | None = None,
) -> dict[str, StepResult]:
    """Run every step on its own agent thread and record the whole run to disk.

    Steps run in dependency waves: everything whose dependencies have passed runs
    concurrently, then the next wave. A step whose pre-conditions fail is blocked
    without starting an agent; a step whose dependency failed is skipped.

    Args:
        task: The user's request, verbatim.
        objective: What success means, in checkable terms.
        steps: The steps to run, in declaration order.
        workspace: Absolute directory each agent works in.
        slug: Package name; defaults to this package's directory name.
        echo: Stream events to the console as they happen.
        record_into: Directory to write ``manifest.json`` and ``runs/`` into; defaults
            to this package. A skill promoted out of ``.aif/`` passes the ``.aif``
            directory it was promoted from, so its runs appear on the same board.
        spawn: Builds the spawnable for a step. Defaults to the step's backend agent;
            tests pass a fake so orchestration is exercisable without a live agent.
        backend: Default :class:`Backend` for steps that do not set their own. ``None``
            reads ``$AIF_BACKEND`` and falls back to Kiro.

    Returns:
        Each step's ``StepResult``, keyed by step name.

    Raises:
        ValueError: ``backend`` or a step's backend is not a known :class:`Backend`.
    """
    package = record_into or package_dir()
    package.mkdir(parents=True, exist_ok=True)
    slug = slug or package.name
    workspace = workspace.expanduser().resolve()
    default_backend = resolve_backend(None, backend)

    # Write the graph snapshot and a "running" record first, so even a crashed run is visible.
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = package / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = json.dumps(_manifest(slug, task, objective, workspace, steps, default_backend), indent=2)
    (run_dir / "manifest.json").write_text(manifest, encoding="utf-8")
    (package / "manifest.json").write_text(manifest, encoding="utf-8")
    started = time.time()
    _write_run_json(
        run_dir, run_id=run_id, slug=slug, task=task, started=started, status="running", backend=default_backend
    )

    # Subscribe the recorder before anything spawns: every event passes append_event (I2).
    coordinator = InMemoryCoordinator()
    events_path = run_dir / "events.jsonl"
    coordinator.on(_sink(events_path))
    if echo:
        coordinator.on(print_event)

    worker = LocalWorker(coordinator)
    await worker.register()
    results: dict[str, StepResult] = {}
    logger.info("run %s: %d step(s) in %s, backend %s", run_id, len(steps), workspace, default_backend.value)

    try:
        # Each wave runs concurrently; the next wave starts once the whole wave has finished.
        for wave in _waves(steps, results):
            outcomes = await asyncio.gather(
                *(_run_step(s, results, worker, coordinator, workspace, spawn, default_backend) for s in wave),
            )
            for outcome in outcomes:
                results[outcome.name] = outcome
        # A step that was never reached (for example, after an error) still gets a result.
        for step in steps:
            results.setdefault(step.name, StepResult(name=step.name, status="skipped"))
    finally:
        # Close out the record even if a step raised, so the viewer never shows a stale "running".
        await worker.close()
        status = _roll_up(results)
        _write_run_json(
            run_dir,
            run_id=run_id,
            slug=slug,
            task=task,
            started=started,
            status=status,
            finished=time.time(),
            backend=default_backend,
        )
        (run_dir / "result.json").write_text(
            json.dumps({name: vars(outcome) for name, outcome in results.items()}, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("run %s finished: %s — %s", run_id, status, run_dir)

    return results


def _sink(path: Path) -> Callable[[Event], None]:
    """A coordinator subscriber that appends every event to ``path`` as JSONL.

    Deliberately dumb: it serializes and interprets nothing, so a recorded run
    stays re-interpretable after the UX changes its mind about the view.

    Args:
        path: The run's ``events.jsonl``.

    Returns:
        A synchronous subscriber for ``Coordinator.on``.
    """

    def sink(event: Event) -> None:
        """Append one event as a JSON line; log, never raise, on failure."""
        # Reopen per event so each line is flushed and a reader sees it immediately.
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.model_dump(mode="json")) + "\n")
        except Exception:
            logger.exception("failed to record a %s event", getattr(event, "kind", "?"))

    return sink


def _write_run_json(
    run_dir: Path,
    *,
    run_id: str,
    slug: str,
    task: str,
    started: float,
    status: str,
    finished: float | None = None,
    backend: Backend = Backend.KIRO,
) -> None:
    """Write the run's ``run.json``; called once at the start and once when the run closes.

    Args:
        run_dir: The run's directory.
        run_id: The run's id.
        slug: The package name.
        task: The user's request, verbatim.
        started: Start time, as a UNIX timestamp.
        status: ``running`` at the start; the rolled-up status at the end.
        finished: End time, or ``None`` while running.
        backend: The run's default backend.
    """
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "runtime_version": RUNTIME_VERSION,
                "run_id": run_id,
                "slug": slug,
                "task": task,
                "session": session_label(),
                "started_at": started,
                "finished_at": finished,
                "status": status,
                "backend": backend.value,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _waves(steps: Sequence[Step], results: dict[str, StepResult]) -> list[list[Step]]:
    """Group steps into dependency waves; each wave runs concurrently.

    A step joins the first wave after all of its dependencies have been placed.
    Waves are computed up front, so they do not depend on outcomes: a step whose
    dependency fails is still scheduled, and ``_run_step`` skips it.

    Args:
        steps: The steps, in declaration order.
        results: Results already known; their steps count as placed.

    Returns:
        The waves in execution order. Steps in a dependency cycle, or depending
        on an unknown step, form one final wave and are skipped when they run.
    """
    remaining, waves = list(steps), []
    satisfied = set(results)
    while remaining:
        # Everything whose dependencies are all placed can run together.
        wave = [s for s in remaining if set(s.depends_on) <= satisfied]
        if not wave:
            logger.error("unsatisfiable dependencies among: %s", [s.name for s in remaining])
            waves.append(list(remaining))
            break
        waves.append(wave)
        satisfied |= {s.name for s in wave}
        remaining = [s for s in remaining if s not in wave]
    return waves


async def _run_step(
    step: Step,
    results: dict[str, StepResult],
    worker: LocalWorker,
    coordinator: InMemoryCoordinator,
    workspace: Path,
    spawn: Callable[[Step, Path, tuple[Condition, ...]], Any] | None,
    default_backend: Backend = Backend.KIRO,
) -> StepResult:
    """Run one step: check its gates, then run its agent to completion.

    The agent's retry loop is the library's: the wrapped post-conditions and
    ``max_attempts`` are handed to the agent, which re-prompts with each failure
    message until the conditions pass or the attempts run out.

    Args:
        step: The step to run.
        results: Results of the steps that have finished so far.
        worker: The worker that hosts the step's thread.
        coordinator: The run's coordinator, for the step's own events.
        workspace: The directory the agent works in.
        spawn: Optional factory for the step's spawnable; tests pass a fake agent.
        default_backend: The run's default backend.

    Returns:
        ``skipped`` if a dependency did not pass, ``blocked`` if a pre-condition
        failed, ``passed`` with the result, or ``failed`` with the error.

    Emits:
        aif_precondition (via ``check_pre_conditions``), aif_step_started,
        aif_postcondition (via ``observed``), and aif_step_finished.
    """
    # Skip without starting anything when an upstream step did not pass.
    unmet = [d for d in step.depends_on if results.get(d) is None or results[d].status != "passed"]
    if unmet:
        return StepResult(name=step.name, status="skipped", blocked_by=unmet)

    # Pre-conditions run before the thread exists; the scope routes their events to its id (I4).
    with thread_scope(coordinator, step.thread_id):
        blocking = await check_pre_conditions(step)
    if blocking:
        logger.warning("step %s blocked by pre-conditions: %s", step.name, blocking)
        return StepResult(name=step.name, status="blocked", blocked_by=blocking)

    # Build the agent with recording post-conditions, then host it on the worker.
    conditions = tuple(observed(c, step=step.name, kind=k) for c, k in step.post_conditions)
    template = (
        spawn(step, workspace, conditions) if spawn else _template_for(step, workspace, conditions, default_backend)
    )
    handle = await worker.spawn_locally(template, thread_id=step.thread_id, thread_name=step.name)
    coordinator.append_event(
        WorkflowEvent(
            kind=STEP_STARTED,
            thread_id=str(step.thread_id),
            thread_name=step.name,
            payload={"step": step.name, "depends_on": list(step.depends_on), "max_attempts": step.max_attempts},
        ),
    )
    try:
        # Blocks until the post-conditions pass; raises when the attempts run out.
        if step.is_typed:
            # A typed thread takes its own arguments and returns its own type; the
            # result is a real object, stringified only for the record.
            result = await handle.run(**_fill_arguments(step, results))
        else:
            result = await handle.run(_fill(step, results))
        outcome = StepResult(name=step.name, status="passed", result=str(result))
    except Exception as exc:
        logger.exception("step %s failed", step.name)
        outcome = StepResult(name=step.name, status="failed", error=str(exc))
    finally:
        # Always terminate the thread: a live Kiro or Codex session is a leaked subprocess.
        coordinator.append_event(
            WorkflowEvent(
                kind=STEP_FINISHED,
                thread_id=str(step.thread_id),
                thread_name=step.name,
                payload={"step": step.name},
            ),
        )
        await handle.terminate_now()
    return outcome


def _template_for(
    step: Step, workspace: Path, conditions: tuple[Condition, ...], default: Backend = Backend.KIRO
) -> Any:
    """Build the spawnable a step runs on: its typed function, or the agent for its backend.

    Args:
        step: The step to build for.
        workspace: The directory the agent works in.
        conditions: The step's post-conditions, already wrapped by ``observed``.
        default: The run's default backend.

    Returns:
        A ``Spawnable`` ready for ``LocalWorker.spawn_locally``.

    Raises:
        BackendUnavailableError: The backend's optional dependency is missing.
    """
    if step.is_typed:
        # ``AIFunction.replace`` returns a new template, so the caller's function is
        # never mutated — two steps may share one function with different budgets.
        return step.function.replace(post_conditions=conditions, max_attempts=step.max_attempts)
    return _FACTORIES[resolve_backend(step, default)](step, workspace, conditions)


def _model(step: Step) -> str | None:
    """Pick the model for a step's agent.

    Args:
        step: The step being built.

    Returns:
        The step's ``model``, else ``$AIF_MODEL``, else ``None`` for the backend's default.
    """
    return step.model or os.environ.get(MODEL_ENV) or None


# Each backend's dependency is imported here, not at module top, so a workflow on one
# backend does not need the extras of the other three; a missing one fails with the
# install command instead of an ImportError at load time.


def _import_kiro() -> Any:
    """Import ``KiroAgent`` on demand.

    Returns:
        The ``KiroAgent`` class.

    Raises:
        BackendUnavailableError: The ``kiro`` extra is not installed.
    """
    try:
        from ai_functions.kiro import KiroAgent
    except ImportError as exc:
        raise BackendUnavailableError(Backend.KIRO, exc) from exc
    return KiroAgent


def _import_claude() -> tuple[Any, Any]:
    """Import the Claude Code agent and its options class on demand.

    Returns:
        ``(ClaudeAgent, ClaudeAgentOptions)``.

    Raises:
        BackendUnavailableError: The ``claude-code`` extra is not installed.
    """
    try:
        from ai_functions.claude_code import ClaudeAgent
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise BackendUnavailableError(Backend.CLAUDE, exc) from exc
    return ClaudeAgent, ClaudeAgentOptions


def _import_codex() -> tuple[Any, Any]:
    """Import the Codex agent and its sandbox type on demand.

    Returns:
        ``(CodexAgent, Sandbox)``.

    Raises:
        BackendUnavailableError: The ``codex`` extra is not installed.
    """
    try:
        from ai_functions.codex import CodexAgent
        from openai_codex import Sandbox
    except ImportError as exc:
        raise BackendUnavailableError(Backend.CODEX, exc) from exc
    return CodexAgent, Sandbox


def _default_ai_thread_tools() -> tuple[Any, ...]:
    """File and shell tools for the ``ai_thread`` backend, or raise naming ``strands-agents-tools``.

    ``BYPASS_TOOL_CONSENT`` is the strands-tools equivalent of Kiro's ``auto_approve``:
    without it the shell tool stops to ask, which a workflow cannot answer.

    Returns:
        ``(file_read, file_write, editor, shell)`` from ``strands_tools``.

    Raises:
        BackendUnavailableError: ``strands-agents-tools`` is not installed.
    """
    try:
        from strands_tools import editor, file_read, file_write, shell
    except ImportError as exc:
        raise BackendUnavailableError(Backend.AI_THREAD, exc) from exc
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")
    return (file_read, file_write, editor, shell)


def _kiro(step: Step, workspace: Path, conditions: tuple[Condition, ...]) -> Any:
    """A Kiro ACP agent scoped to ``workspace``.

    ``KIRO_MODEL`` is honoured alongside ``AIF_MODEL``. Worth setting: a workflow spawns
    one session per step, so the default model is paid for several times over;
    ``kiro-cli chat --list-models`` shows the options and their credit multipliers.

    Args:
        step: The step to build for.
        workspace: The ACP session's working directory.
        conditions: The step's wrapped post-conditions.

    Returns:
        A ``KiroAgent`` that approves its own tool calls.
    """
    kiro_agent = _import_kiro()
    return kiro_agent(
        name=step.name,
        cwd=str(workspace),
        model=_model(step) or os.environ.get("KIRO_MODEL") or None,
        auto_approve=True,
        post_conditions=conditions,
        max_attempts=step.max_attempts,
    )


def _claude(step: Step, workspace: Path, conditions: tuple[Condition, ...]) -> Any:
    """Build a Claude Code agent working in ``workspace`` without permission prompts.

    Args:
        step: The step to build for.
        workspace: The session's working directory.
        conditions: The step's wrapped post-conditions.

    Returns:
        A ``ClaudeAgent`` in ``bypassPermissions`` mode.
    """
    claude_agent, claude_options = _import_claude()
    options = {"cwd": str(workspace), "permission_mode": "bypassPermissions"}
    if _model(step):
        options["model"] = _model(step)
    return claude_agent(
        name=step.name,
        options=claude_options(**options),
        post_conditions=conditions,
        max_attempts=step.max_attempts,
    )


def _codex(step: Step, workspace: Path, conditions: tuple[Condition, ...]) -> Any:
    """Build a Codex agent working in ``workspace`` with a workspace-write sandbox.

    Args:
        step: The step to build for.
        workspace: The session's working directory.
        conditions: The step's wrapped post-conditions.

    Returns:
        A ``CodexAgent`` allowed to write inside ``workspace``.
    """
    codex_agent, sandbox = _import_codex()
    return codex_agent(
        name=step.name,
        cwd=str(workspace),
        model=_model(step),
        sandbox=getattr(sandbox, "workspace_write", None) or sandbox("workspace_write"),
        post_conditions=conditions,
        max_attempts=step.max_attempts,
    )


def _ai_thread(step: Step, workspace: Path, conditions: tuple[Condition, ...]) -> Any:
    """The library's own ``AIThread``: a Strands agent with file and shell tools, told where to work.

    Unlike the CLI-backed agents it has no working directory of its own, so the system
    prompt names the workspace and asks for absolute paths under it.

    Args:
        step: The step to build for.
        workspace: The directory named in the system prompt.
        conditions: The step's wrapped post-conditions.

    Returns:
        An ``AIFunction`` whose docstring template is the prompt itself.

    Raises:
        BackendUnavailableError: No ``tools`` were given and ``strands-agents-tools`` is missing.
    """
    from ai_functions.ai_thread import ai_function

    # An AI function whose prompt is its argument: the step's text goes to the model as-is.
    def prompt(prompt: str) -> str:  # noqa: D103 — the prompt is the prompt
        return prompt

    prompt.__name__ = step.name
    settings: dict[str, Any] = {
        "system_prompt": (
            f"You are carrying out one step of a workflow. Work only inside the directory {workspace}, "
            "reading and writing files by absolute path under it. When you are done, reply with a short, "
            "factual account of what you changed."
        ),
        "tools": tuple(step.tools) or _default_ai_thread_tools(),
        "post_conditions": conditions,
        "max_attempts": step.max_attempts,
    }
    if _model(step):
        settings["model"] = _model(step)
    return ai_function(**settings)(prompt)


_FACTORIES: dict[Backend, Callable[[Step, Path, tuple[Condition, ...]], Any]] = {
    Backend.KIRO: _kiro,
    Backend.CLAUDE: _claude,
    Backend.CODEX: _codex,
    Backend.AI_THREAD: _ai_thread,
}


def _interpolate(text: str, results: dict[str, StepResult]) -> str:
    """Replace each ``{step_name}`` in ``text`` with that finished step's result.

    Only names of steps that have finished are substituted, and every other brace is
    left exactly as written, so a prompt can contain a JSON example or code without
    escaping. ``str.format`` would reject such a prompt outright.

    Args:
        text: A prompt or string argument that may name upstream steps.
        results: Results of the steps that have finished so far.

    Returns:
        ``text`` with every ``{name}`` of a finished step replaced by its result.
    """
    if not results:
        return text
    names = "|".join(re.escape(name) for name in results)
    return re.sub(rf"\{{({names})\}}", lambda match: results[match.group(1)].result or "", text)


def _fill(step: Step, results: dict[str, StepResult]) -> str:
    """Substitute ``{upstream_step}`` in a step's prompt with that step's result.

    Args:
        step: The step whose prompt is filled.
        results: Results of the steps that have finished so far.

    Returns:
        The prompt with upstream results substituted; other braces are untouched.
    """
    return _interpolate(step.prompt, results)


def _fill_arguments(step: Step, results: dict[str, StepResult]) -> dict[str, Any]:
    """Resolve a typed step's arguments just before it runs.

    A callable argument is called now, which is how a step reads a *file* an earlier
    step wrote: the file does not exist when the workflow is declared. A string argument
    gets ``{upstream_step}`` substituted from the results so far.

    Args:
        step: The typed step whose ``arguments`` are resolved.
        results: Results of the steps that have finished so far.

    Returns:
        Keyword arguments for the typed function's ``run``.
    """
    filled: dict[str, Any] = {}
    for key, value in step.arguments.items():
        # Callables are deferred reads; strings are templates; anything else passes through.
        if callable(value):
            filled[key] = value()
        elif isinstance(value, str):
            filled[key] = _interpolate(value, results)
        else:
            filled[key] = value
    return filled


def _roll_up(results: dict[str, StepResult]) -> str:
    """Roll step outcomes into one run status.

    Args:
        results: Every step's result.

    Returns:
        ``passed`` when every step passed, ``failed`` when any failed or was
        blocked, and ``error`` otherwise (no steps, or only skips).
    """
    if not results:
        return "error"
    statuses = {outcome.status for outcome in results.values()}
    if statuses == {"passed"}:
        return "passed"
    if statuses & {"failed", "blocked"}:
        return "failed"
    return "error"


__all__ = [
    "BACKEND_ENV",
    "MODEL_ENV",
    "POSTCONDITION",
    "PRECONDITION",
    "RUNTIME_VERSION",
    "Backend",
    "BackendUnavailableError",
    "Step",
    "StepResult",
    "WorkflowEvent",
    "condition_name",
    "describe_condition",
    "evaluate",
    "observed",
    "package_dir",
    "prompt_section",
    "resolve_backend",
    "run_workflow",
]
