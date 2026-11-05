"""``@skillify`` on ScienceWorld: skills learned from the function's own calls.

An ``@skillify``-decorated ``play`` function is simply *called* on
ScienceWorld tasks, the way an application would call it. Nobody asks it to
learn anything. Every ``--every`` calls, a background round mines candidate
skills from the recorded calls and keeps a candidate only if it raises the
completion rate when recorded calls are replayed with it, against the
library as it stands. The final library is then frozen and measured, with
and without, on held-out tasks (``evaluation.py``).

A replay needs the environment the call acted in, and a ScienceWorld session
is stateful, so the decorator is given an ``environment=``
(``scienceworld_env.make_episode_factory``): every call and every replay
rebuilds a fresh session from the call's own arguments, and a replay whose
rebuilt environment differs from the recorded one fails loudly
(``ReplayDriftError``) rather than measuring a different task.

The tasks are fixed ScienceWorld variation ids (``evaluation.py``), so this
runs from a clean clone and every reader measures the same tasks: the
``--traffic-tasks`` / ``--eval-tasks`` counts take a prefix of those lists,
and nothing else decides which tasks run.

Run it from this directory (it is a workspace member, so uv is pointed at
the repository root)::

    DEMO="uv run --project ../.. --package ai-functions-skillify-demo-example"

    $DEMO main.py --out ./out-skillify            # full run
    $DEMO main.py --smoke --out ./out-smoke       # quick end-to-end check
    $DEMO main.py --evaluate-only ./out-skillify/library --repeats 4 --out ./out-recheck
        # re-run just the held-out evaluation against a library an earlier run already
        # learned (e.g. with a larger --repeats), skipping traffic and rounds entirely

Requires Bedrock credentials, a JVM and ``scienceworld``. See README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation import (
    DEFAULT_MODEL,
    EVAL_VARIATIONS,
    PLAY_PROMPT,
    SKILLS_SLOT,
    TASK,
    TRAFFIC_VARIATIONS,
    AIFunctionEvalDriver,
    _report,
    _spec,
    report_evaluation,
    report_selection,
)
from full_pipeline import FullDiscoveryMiner, PrefilteredGate
from scienceworld_env import (
    ReplayDriftError,
    ScienceWorldEpisode,
    Variation,
    live_jvms,
    load_variations,
    make_episode_factory,
    opening_observation,
)

from ai_functions import ai_function
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.experimental.skills import SkillifiedFunction, skillify
from ai_functions.experimental.skills.evaluation import CompletionGate
from ai_functions.experimental.skills.skill_library import LibraryStore, SkillDeliveryError, load_skill

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_logger = logging.getLogger("skillify_demo")
logging.getLogger("py4j").setLevel(logging.WARNING)

SMOKE = {
    "traffic_tasks": 4,
    "traffic_repeats": 1,
    "every": 4,
    "eval_tasks": 2,
    "repeats": 1,
    "step_limit": 12,
    "trial_inputs": 2,
    "trial_repeats": 1,
}
"""``--smoke``: every stage once, small. Explicit flags still override it."""


@dataclasses.dataclass
class EpisodeStats:
    """How often the episode factory ran, to show replays really rebuilt the environment."""

    factory_calls: int = 0
    drift_errors: int = 0
    live_attempts: int = 0
    live_failed: int = 0
    completed: int = 0
    max_jvms: int = 0

    @property
    def replay_episodes(self) -> int:
        """Factory calls not made by a live call: the rounds' replays."""
        return self.factory_calls - self.live_attempts


def make_play(variations: Mapping[str, Variation], args: argparse.Namespace, stats: EpisodeStats) -> SkillifiedFunction:
    """The decorated ``play``: the shared prompt, a fresh ScienceWorld episode per call and per replay."""
    factory = make_episode_factory(variations, args.step_limit)

    async def environment(arguments: Mapping[str, Any]) -> ScienceWorldEpisode:
        stats.factory_calls += 1
        try:
            return await factory(arguments)
        except ReplayDriftError:
            stats.drift_errors += 1
            _logger.error("REPLAY DRIFT on %s: ScienceWorld reset is not deterministic", arguments["variation_id"])
            raise

    # coordinator_tools_enabled=False, as in evaluation.py: no peers to message.
    def play(variation_id: str, task_description: str, opening_observation: str) -> str: ...

    # evaluation.py's prompt, less the skill-index line: ``@skillify`` puts the index
    # in the system prompt. ``variation_id`` is not shown to the model; it is
    # what the episode factory rebuilds the environment from.
    play.__doc__ = PLAY_PROMPT.replace(SKILLS_SLOT, "")
    base = ai_function(model=args.model, coordinator_tools_enabled=False)(play)
    return skillify(
        library=Path(args.out).resolve() / "library",
        replay=True,
        environment=environment,
        every=args.every,
        trial_inputs=args.trial_inputs,
        max_concurrency=args.concurrency,
        # The pipeline behind the published result, plugged in explicitly: the full
        # discovery pipeline as the miner; a Tier-1 prefilter in front of the
        # library's with/without completion gate.
        miner=FullDiscoveryMiner(),
        gate=PrefilteredGate(
            CompletionGate(min_gain=args.min_gain, repeats=args.trial_repeats, budget_calls=args.budget_calls),
            min_tier1=0.5,
        ),
    )(base)


async def live_call(
    play: SkillifiedFunction,
    variation: Variation,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    stats: EpisodeStats,
    label: str,
) -> str | None:
    """One traffic call, retried on transient failures like ``AIFunctionEvalDriver``.

    Every attempt is recorded by the decorator (a failed one unscored), so a
    call that still fails after the last attempt is on record and the demo
    goes on. ``AIFunctionError`` is the function failing, not the service:
    the decorator scores it 0.0 and it is not retried.
    """
    async with semaphore:
        opening = await opening_observation(variation, args.step_limit)
        for attempt in range(1, args.max_attempts + 1):
            stats.live_attempts += 1
            try:
                result = await play(
                    variation_id=variation.task_id,
                    task_description=variation.description,
                    opening_observation=opening,
                )
            except (SkillDeliveryError, ReplayDriftError):
                raise
            except AIFunctionError as exc:
                _logger.warning("%s: %s (scored 0.0)", label, exc)
                stats.live_failed += 1
                return None
            except Exception as exc:  # noqa: BLE001 - transient infra failures are the norm here
                if attempt == args.max_attempts:
                    _logger.error("%s failed after %d attempts: %s: %s", label, attempt, type(exc).__name__, exc)
                    stats.live_failed += 1
                    return None
                delay = args.retry_backoff * 2 ** (attempt - 1)
                _logger.warning(
                    "%s failed (attempt %d/%d): %s: %s -- retrying in %ds",
                    label,
                    attempt,
                    args.max_attempts,
                    type(exc).__name__,
                    str(exc)[:160],
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            _logger.info("%s done (%d calls pending for the next round)", label, play.learner.pending)
            stats.completed += 1
            return result
    return None


async def settle_rounds(play: SkillifiedFunction, every: int) -> None:
    """Wait for any in-flight round; run the one a busy learner could not start.

    A round starts after a call only if none is running, so the round due at
    call 24 is skipped when round 1 is still replaying then. If a full window
    is left pending, it is run now; if a round already consumed it, nothing is.
    """
    learner = play.learner
    # ``_task`` is the learner's background round; there is no public handle to await.
    while (task := learner._task) is not None and not task.done():  # noqa: SLF001
        _logger.info("waiting for the in-flight round")
        await asyncio.wait({task})
    if learner.pending >= every:
        _logger.info("%d calls pending with no round running; running that round now", learner.pending)
        await play.refresh_skills()


async def watch_jvms(stats: EpisodeStats) -> None:
    """Track the peak number of live ScienceWorld JVMs (one per open environment)."""
    while True:
        stats.max_jvms = max(stats.max_jvms, live_jvms())
        await asyncio.sleep(2)


def print_rounds(play: SkillifiedFunction) -> list[dict[str, Any]]:
    """Every round report and rollback, then the library's versions."""
    history = play.history()
    for entry in history:
        if entry.get("type") != "round":
            print(f"\n{entry}")
            continue
        rate = entry["window_success_rate"]
        rows = [
            ("status", entry["status"] + (f"  ({entry['error']})" if entry["error"] else "")),
            ("window", f"{entry['window_calls']} calls, success rate {'n/a' if rate is None else f'{rate:.2f}'}"),
            ("mined", ", ".join(entry["mined"]) or "-"),
            ("dropped duplicates", ", ".join(entry["dropped_duplicates"]) or "-"),
            ("trial inputs overlap mining", str(entry["trials_overlap_mining"])),
            ("gate replay runs", str(entry["extra_calls"])),
            ("budget cuts", "; ".join(entry["budget_cuts"]) or "-"),
            ("version", f"{entry['version_before']} -> {entry['version_after']}"),
        ]
        for verdicts, mark in ((entry["admitted"], "ADMIT "), (entry["rejected"], "REJECT")):
            for name, verdict in verdicts.items():
                scores = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in verdict["scores"].items()
                )
                rows.append((f"{mark} [{verdict['stage']}]", f"{name}  {scores}"))
        _report(f"Round {entry['round']}", rows)

    store = play.learner.store
    current = store.current_version()
    rows = [
        (f"v{v}{' (current)' if v == current else ''}", ", ".join(d.name for d in store.skill_dirs(v)) or "(empty)")
        for v in store.versions()
    ]
    _report("Library versions", rows or [("v0 (current)", "(empty)")])
    return history


def _select_tasks(args: argparse.Namespace) -> tuple[list[Variation], list[Variation], dict[str, Variation]]:
    """``(traffic, evaluation, variations)``: the fixed variation ids, loaded from ScienceWorld.

    ``--traffic-tasks`` and ``--eval-tasks`` take a prefix of
    ``TRAFFIC_VARIATIONS`` and ``EVAL_VARIATIONS``, which is all a smaller
    run changes. Shared by the learning path and ``--evaluate-only``, so both
    measure the same held-out tasks; ``--evaluate-only`` makes no calls, so
    it loads no traffic variations.
    """
    traffic_ids = [] if args.evaluate_only else TRAFFIC_VARIATIONS[: args.traffic_tasks]
    eval_ids = EVAL_VARIATIONS[: args.eval_tasks]
    loaded = load_variations(TASK, [*traffic_ids, *eval_ids])
    traffic, evaluation = loaded[: len(traffic_ids)], loaded[len(traffic_ids) :]
    report_selection(traffic, evaluation)
    return traffic, evaluation, {v.task_id: v for v in loaded}


def _open_library_store(path: Path) -> LibraryStore:
    """Open ``path`` as an existing skill library for ``--evaluate-only``, refusing anything else.

    ``LibraryStore.__init__`` creates ``versions/``, ``records/`` etc. for
    *any* path handed to it -- harmless for a library ``@skillify`` already
    owns, but wrong for an arbitrary or mistyped path, which would otherwise
    silently start growing library scaffolding there. Checked before
    construction.
    """
    if not path.is_dir() or not (path / "versions").is_dir():
        raise SystemExit(
            f"--evaluate-only {path} is not a skill library: no CURRENT/versions found there. Pass the "
            "library directory a previous run of this demo wrote, e.g. <out>/library"
        )
    return LibraryStore(path)


async def run_evaluate_only(
    args: argparse.Namespace,
    out_root: Path,
    variations: Mapping[str, Variation],
    evaluation: list[Variation],
    started: float,
) -> None:
    """``--evaluate-only``: skip traffic and rounds; evaluate an existing library's current version.

    Exactly the evaluation section of the learning path -- same task
    selection, same ``AIFunctionEvalDriver``, same ``report_evaluation`` --
    aimed at a library a previous run already learned, so it can be re-run
    with more ``--repeats`` for a tighter confidence interval with no new
    mining.
    """
    library_path = Path(args.evaluate_only).resolve()
    store = _open_library_store(library_path)
    final_version = store.current_version()
    final_dirs = store.skill_dirs(final_version)
    if not final_dirs:
        raise SystemExit(
            f"--evaluate-only {library_path} v{final_version} has no skills: the skill_on and skill_off arms "
            "would be identical. Point --evaluate-only at a library whose current version has learned skills."
        )
    skill_names = [load_skill(d).name for d in final_dirs]
    print(f"\nEvaluate-only: library {library_path} v{final_version}: {', '.join(skill_names)}")

    eval_driver = AIFunctionEvalDriver(
        variations,
        model=args.model,
        max_concurrency=args.concurrency,
        run_root=out_root / "runs",
        step_limit=args.step_limit,
        observation={"library": str(library_path), "version": final_version},
    )
    specs = [
        dataclasses.replace(_spec(v, ("skill_on", "skill_off"), args.repeats), skills=list(final_dirs))
        for v in evaluation
    ]
    results = await eval_driver.run(specs)
    analysis = report_evaluation(results, evaluation, out_root, repeats=args.repeats)
    finished = time.monotonic()

    summary = {
        "mode": "evaluate_only",
        "args": vars(args),
        "library_path": str(library_path),
        "evaluation_tasks": [v.task_id for v in evaluation],
        "final_version": final_version,
        "final_skills": skill_names,
        "minutes": {"evaluation": round((finished - started) / 60, 1), "total": round((finished - started) / 60, 1)},
        "analysis": analysis,
    }
    (out_root / "skillify_summary.json").write_text(json.dumps(summary, indent=2, default=str))


async def main(args: argparse.Namespace) -> None:
    started = time.monotonic()
    out_root = Path(args.out).resolve()
    library = out_root / "library"
    if not args.evaluate_only and (library / "records").exists():
        raise SystemExit(f"{library} already holds recorded calls; use a fresh --out")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(exist_ok=True)

    traffic, evaluation, variations = _select_tasks(args)

    if args.evaluate_only:
        await run_evaluate_only(args, out_root, variations, evaluation, started)
        return

    stats = EpisodeStats()
    watcher = asyncio.create_task(watch_jvms(stats))
    play = make_play(variations, args, stats)

    # --- Traffic: the function is just called; rounds fire on their own ---
    _logger.info(
        "traffic: %d tasks x %d = %d calls, a round every %d",
        len(traffic),
        args.traffic_repeats,
        len(traffic) * args.traffic_repeats,
        args.every,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(
        *(
            live_call(play, v, args, semaphore, stats, f"call {v.task_id}/{r}")
            for r in range(args.traffic_repeats)
            for v in traffic
        )
    )
    traffic_done = time.monotonic()
    await settle_rounds(play, args.every)
    rounds_done = time.monotonic()
    history = print_rounds(play)

    # --- Evaluation: the final library, frozen, on the held-out tasks ---
    store = play.learner.store
    final_version = store.current_version()
    final_dirs = store.skill_dirs(final_version)
    if not final_dirs:
        print("\nThe final library is EMPTY: the skill_on arm below runs with no skills and equals skill_off.")
    else:
        print(f"\nEvaluating library v{final_version}: {', '.join(load_skill(d).name for d in final_dirs)}")
    eval_driver = AIFunctionEvalDriver(
        variations,
        model=args.model,
        max_concurrency=args.concurrency,
        run_root=out_root / "runs",
        step_limit=args.step_limit,
        observation={"library": str(store.root.resolve()), "version": final_version},
    )
    specs = [
        dataclasses.replace(_spec(v, ("skill_on", "skill_off"), args.repeats), skills=list(final_dirs))
        for v in evaluation
    ]
    results = await eval_driver.run(specs)
    analysis = report_evaluation(results, evaluation, out_root, repeats=args.repeats)
    finished = time.monotonic()

    await asyncio.sleep(3)  # let the last environments' JVM shutdowns land
    watcher.cancel()
    jvms_left = live_jvms()
    summary = {
        "args": vars(args),
        "traffic_tasks": [v.task_id for v in traffic],
        "evaluation_tasks": [v.task_id for v in evaluation],
        "rounds": [e for e in history if e.get("type") == "round"],
        "final_version": final_version,
        "final_skills": [d.name for d in final_dirs],
        "episodes": {**dataclasses.asdict(stats), "replay_episodes": stats.replay_episodes, "jvms_left": jvms_left},
        "minutes": {
            "traffic": round((traffic_done - started) / 60, 1),
            "rounds_after_traffic": round((rounds_done - traffic_done) / 60, 1),
            "evaluation": round((finished - rounds_done) / 60, 1),
            "total": round((finished - started) / 60, 1),
        },
        "analysis": analysis,
    }
    (out_root / "skillify_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _report(
        "Episodes (the environment= factory)",
        [
            ("factory calls", str(stats.factory_calls)),
            ("  by live calls (attempts)", str(stats.live_attempts)),
            ("  by round replays", str(stats.replay_episodes)),
            ("replay drift errors", str(stats.drift_errors)),
            ("live calls failed", str(stats.live_failed)),
            ("peak live JVMs", str(stats.max_jvms)),
            ("JVMs left at exit", str(jvms_left)),
            ("minutes", json.dumps(summary["minutes"])),
        ],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help=f"a quick end-to-end check: {SMOKE}")
    parser.add_argument(
        "--evaluate-only",
        metavar="LIBRARY",
        default=None,
        help="skip traffic and all rounds; evaluate this existing library's current version instead "
        "(e.g. <out>/library from a previous run) -- for re-measuring with a bigger --repeats",
    )
    parser.add_argument("--out", default="./out-skillify", help="library, run dirs and results")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--step-limit", type=int, default=60, help="ScienceWorld steps per episode")
    parser.add_argument(
        "--traffic-tasks",
        type=int,
        default=len(TRAFFIC_VARIATIONS),
        help=f"train tasks the function is called on: this many of evaluation.TRAFFIC_VARIATIONS ({TASK})",
    )
    parser.add_argument("--traffic-repeats", type=int, default=2, help="calls per traffic task")
    parser.add_argument("--every", type=int, default=12, help="recorded calls between rounds")
    parser.add_argument("--trial-inputs", type=int, default=4, help="recorded calls replayed per gate trial")
    parser.add_argument("--trial-repeats", type=int, default=2, help="runs per arm per trial input")
    parser.add_argument("--min-gain", type=float, default=0.02, help="completion gain a skill must add")
    parser.add_argument("--budget-calls", type=int, default=200, help="cap on replay calls per round")
    parser.add_argument(
        "--eval-tasks",
        type=int,
        default=len(EVAL_VARIATIONS),
        help="held-out tasks to measure: this many of evaluation.EVAL_VARIATIONS",
    )
    parser.add_argument("--repeats", type=int, default=2, help="runs per arm per evaluation task")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="calls, replays and evaluation runs in flight (each holds a ScienceWorld JVM); traffic and a round's "
        "replays can overlap, so up to twice this many environments are open at once",
    )
    parser.add_argument("--max-attempts", type=int, default=4, help="tries per traffic call on transient failures")
    parser.add_argument("--retry-backoff", type=int, default=20, help="seconds before the first retry; doubles")
    if parser.parse_known_args(argv)[0].smoke:
        parser.set_defaults(**SMOKE)
    args = parser.parse_args(argv)
    # The task set is fixed, so asking for more tasks than there are ids is a
    # mistake, not a request to be quietly truncated: it would report fewer
    # tasks than the caller believes were measured.
    for flag, count, ids in (
        ("--traffic-tasks", args.traffic_tasks, TRAFFIC_VARIATIONS),
        ("--eval-tasks", args.eval_tasks, EVAL_VARIATIONS),
    ):
        if not 0 <= count <= len(ids):
            parser.error(f"{flag} {count}: the demo has {len(ids)} fixed variations to take from (see evaluation.py)")
    return args


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
