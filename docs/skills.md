# Skills that Learn from Calls

A *skill* is an ability: an inference function that maps past and present data to a decision or action, and that improves with experience because it depends on memory experience can change — structural (model weights, system prompts), procedural (code and procedures), factual (stored facts) and fading (working context). An AI Function is such a function. `@skillify` improves one by growing its **procedural memory**: a library of reusable procedures mined from the function's own calls and kept only when they measurably make it better.

In this package each procedure is currently stored as a `SKILL.md` file (frontmatter plus Markdown instructions) and handed to the agent on demand through a `Skill` tool; that is why the API calls them skills (`Skill`, `SkillLibraryBackend`, the `Skill` tool, following the Claude Code convention), and why the rest of this guide does too. A file of instructions alone only conditions the model's inference; what the library measures and improves is the function itself.

The `ai_functions.experimental.skills` module does this in the background: stack `@skillify` on the function, keep calling it, and the library mines candidate procedures from the recorded calls and publishes only the ones that measurably help. One rule governs the whole module — **a stored skill earns its place only by measurably improving the function over the library it would join** — so every candidate is replayed with and without it before it is published.

This enables:

- **A skill library with no induction step** — the function is just called; recording, mining, gating and publishing happen in background rounds the caller never waits for.
- **Pluggable mining and gating** — a round asks a `Miner` for candidates and a `Gate` for verdicts. Small defaults ship with the library; a heavier pipeline plugs in through the same two protocols.
- **Stateful tools** — an `environment=` factory rebuilds a game, simulator or sandbox from each call's arguments, so recorded calls can be replayed faithfully.
- **Skills as graph parameters** — a stored procedure the agent loads is a parameter node of the traced call: procedural memory the [optimization loop](tutorial.md#memory-and-optimization) can rewrite; the rewrite is staged and gated like any new candidate, never published directly.

Note: the module is experimental — the API may change in future releases.

For transparent inspection, wrap local calls and rounds in
`with observe_skills(recorder): ...`. The callback receives isolated facts about
executions, candidate generation, gating, and publication; it cannot mutate the
live objects, and callback failures do not affect execution. Custom miners and
gates can report additional phases with `skill_event(...)`.
The [shared example viewer](../examples/skills_viz/README.md) records these facts,
shows procedure blocks with exact source lines, and replays observed loads and
model trajectories. Its UI and recording dependencies stay under `examples/`.

## Contents

- [`@skillify`](#skillify)
- [What a round does](#what-a-round-does)
- [Mining and gating: `Miner` and `Gate`](#mining-and-gating-miner-and-gate)
- [Stateful tools: `environment=`](#stateful-tools-environment)
- [Skills as graph parameters](#skills-as-graph-parameters)
- [Limits](#limits)
- [Examples](#examples)

## `@skillify`

`@skillify` stacks on `@ai_function` the way `@routed` does:

```python
from ai_functions import ai_function
from ai_functions.experimental.skills import skillify


@skillify(library="skills/solve", replay=True, every=20)
@ai_function(tools=[...], post_conditions=[check_report])
async def solve(task: str) -> Report:
    """Solve this task: {task}"""


report = await solve(task="...")          # called like any AI Function
round_report = await solve.refresh_skills()  # or run a round now instead of waiting
print(round_report.status, list(round_report.admitted))
```

Every call runs with the current library version — unchanged while it is empty, otherwise with the skill index in the system prompt and a `Skill` tool added — then records its outcome, arguments and event log. Once `every` new calls have landed and no round is already running, a round starts in the background.

The agent uses skills the way Claude Code does: the index lists each skill's name and one-line description, and the `Skill` tool, taking a `skill` argument, returns the full body. Nothing is preloaded, so the agent can decline a skill that does not fit. Pasting keyword-retrieved bodies into the prompt instead is simpler but worse: near-identical tasks retrieve the same skills every time, whether or not they apply.

A call's outcome is `score(arguments, result)` if given, else 1.0 when it returned and 0.0 when it raised `AIFunctionError` (post-conditions exhausted). Any other exception leaves the call unscored. Every exception reaches the caller unchanged; recording never breaks a call.


| Setting           | Meaning                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `library`         | Directory holding skills, versions, call records and history                                                                                                                                                                                                                                                                                                                                     |
| `replay`          | Required: confirms the function's tools are safe to re-run                                                                                                                                                                                                                                                                                                                                       |
| `every`           | New recorded calls between rounds                                                                                                                                                                                                                                                                                                                                                                |
| `score`           | Optional `(arguments, result) -> float in [0, 1]`, where `arguments` is the call's arguments bound by name; default is post-condition pass/fail                                                                                                                                                                                                                                                  |
| `pass_at`         | Outcome at or above which a call counts as a success                                                                                                                                                                                                                                                                                                                                             |
| `trial_inputs`    | Recorded calls replayed per gate trial                                                                                                                                                                                                                                                                                                                                                           |
| `trial_repeats`   | Runs per arm per trial input (default gate only)                                                                                                                                                                                                                                                                                                                                                 |
| `min_gain`        | Completion-rate gain a skill must add over the current library (default gate only)                                                                                                                                                                                                                                                                                                               |
| `max_skills`      | Cap on the installed library                                                                                                                                                                                                                                                                                                                                                                     |
| `budget_calls`    | Cap on gate replay runs per round (default gate only)                                                                                                                                                                                                                                                                                                                                            |
| `keep_versions`   | Library versions retained on disk                                                                                                                                                                                                                                                                                                                                                                |
| `max_concurrency` | Replays in flight during a round                                                                                                                                                                                                                                                                                                                                                                 |
| `environment`     | Optional `arguments -> Episode` for stateful tools; exclusive with `score`                                                                                                                                                                                                                                                                                                                       |
| `miner`           | Where candidates come from; default `PromptMiner` on the decorated function's model                                                                                                                                                                                                                                                                                                              |
| `gate`            | Which candidates join; default `CompletionGate` built from `min_gain`, `trial_repeats`, `budget_calls` and `pass_at`. Passing `gate=` with a non-default value of any of those three raises `ValueError`, since they configure the default gate only. A gate you pass gets none of these settings, `pass_at` included: write `CompletionGate(..., pass_at=0.8)` yourself if it should not be 1.0 |


All of these can also be passed at once as `config=SkillifyConfig(...)`.


| Member                                 | Behavior                                                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `await solve(...)`, `solve.trace(...)` | unchanged calling contract                                                                                                           |
| `await solve.refresh_skills()`         | runs a round now (after any running one), waits for it, returns its `RoundReport`                                                    |
| `solve.skills`                         | the skills in the current version                                                                                                    |
| `solve.skill_backend`                  | the current version as a staged `SkillLibraryBackend`; `None` while the library is empty                                             |
| `solve.history()`                      | a list of dicts, one per round and per rollback, oldest first; `RoundReport.from_json` turns a round entry back into a `RoundReport` |
| `solve.rollback_skills(to=None)`       | switches to the previous version, or to version `to`                                                                                 |




## What a round does

A round turns the calls recorded since the last one into, possibly, a new library version:

1. **Trigger.** It looks at the window's outcomes. Fewer than three scored calls is a transcript, not a pattern, and nothing is mined; nor is anything mined from a window with no failure and no spread between the best and worst score — the at-ceiling case, where nothing is left to learn. Skills nobody needed are not free: every description occupies the agent's context on every later call. Such a round still gates any staged revisions (see [below](#skills-as-graph-parameters)), and is declined when there are none.
2. **Mine.** The window's scored calls go to the miner as `RecordedCall` objects, each with its event log and `passed` (outcome at or above `pass_at`). Failed calls are the repair material. An unscored call counts toward the window but is never mined.
3. **Deduplicate.** A candidate named like an installed skill is dropped, and so is one whose name differs from an installed skill's, or an earlier candidate's, only by `-` versus `_` (both map to one parameter name). A candidate named `skill-index`, the backend's reserved index parameter, is rejected as `"invalid"`. Revisions staged by an optimizer join the candidates here.
4. **Choose trial inputs.** Replayable, scored calls from *outside* the window, failed ones first. When there are none — always the case in the first round — it uses the window's own calls and says so (`trials_overlap_mining=True`).
5. **Gate.** The gate returns one verdict per candidate, replaying trial inputs with and without each one against the *current* library.
6. **Publish.** The current skills, with admitted revisions swapped in, plus the admitted new ones — capped at `max_skills` by gain — become a new version.

Every round that runs, declined or not, appends a `RoundReport` to the library's `history.jsonl`: its `status` (`"declined"`, `"admitted"`, `"none_admitted"` or `"failed"`), what was `mined`, `admitted`, `revised` and `rejected` with each verdict's scores and reason, the versions before and after, and the replay runs it cost. A filter whose rejections are invisible is indistinguishable from no filter, so every gated candidate that is rejected names the stage that decided it. A mined candidate that duplicates an installed skill is never gated and is listed under `dropped_duplicates` instead; drafts `PromptMiner` drops are only logged.

A round that raises — including a custom gate returning the wrong number of verdicts — is reported as `"failed"` and leaves the current version unchanged, and so does a round that finds the library rolled back under it before it publishes; either way, the staged revisions it took go back to `pending/` for the next round, unless a newer one was staged meanwhile. A failed round still consumes its window, so it cannot re-trigger on every call. Two kinds of round consume nothing and leave their calls pending for the next one: a cancelled round, and one refused because another round is already running in this process or another process holds the library's round lock (returned as `"failed"` but not written to the history).

## Mining and gating: `Miner` and `Gate`

A round asks two questions, and each is a protocol you can replace:

```python
class Miner(Protocol):
    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]: ...

class Gate(Protocol):
    async def __call__(
        self,
        candidates: Sequence[Candidate],   # skill, plus `replaces` for a revision
        current: Sequence[Path],           # the current library's skill directories
        trials: Sequence[RecordedCall],    # the calls to try them on
        replay: Replay,                    # replay(skill_dirs, calls, repeats) -> outcomes
    ) -> list[Verdict]: ...                # one per candidate, in order
```

All of these types are importable from `ai_functions.experimental.skills`. A miner writes each candidate under `out_dir` and returns it; whatever it returns is still gated. A gate measures through `replay`, which re-runs recorded calls with exactly the skill directories it is given — rebuilding each call's environment, if there is one — and returns one outcome per (call, repeat), `None` when unscored. The learner publishes only what the gate admits.

The library ships a small default for each:

- `PromptMiner(model=None, max_drafts=5)` makes one model call per round: it reads the calls as compact transcripts, failed ones marked (a long call keeps its beginning and its end), and asks for at most `max_drafts` reusable procedures. `@skillify`'s default miner runs on the decorated function's model; `model=None` on a `PromptMiner` you build yourself uses the library default. A draft with no name, no description or no body, a refusal-shaped body, a name already taken in the same round, or the reserved `skill-index` is dropped and logged, and writes nothing to disk. If the model's structured output fails altogether, that is logged and the round mines nothing.
- `CompletionGate(min_gain=0.02, repeats=2, budget_calls=200, pass_at=1.0)` replays the trial calls under the current library alone and under the current library plus the candidate (for a revision, with the skill it replaces swapped out), and admits the candidate only if the completion rate rises by at least `min_gain`. Over `budget_calls` replays, candidates are cut from the end of the list first, then trial inputs, so the order of the list decides what survives.

**The built-in defaults have no measured result.** They are small, offline-tested reference implementations of the two protocols. The published result below was produced by the example's own pipeline — a five-stage discovery miner and a quality prefilter in front of `CompletionGate` — plugged in through these same protocols:

```python
# FullDiscoveryMiner and PrefilteredGate live in examples/skillify_demo/full_pipeline/, not in the library.
@skillify(
    library="out/library", replay=True, environment=environment,
    miner=FullDiscoveryMiner(),
    gate=PrefilteredGate(CompletionGate(min_gain=0.02, repeats=2, budget_calls=200), min_tier1=0.5),
)
```

Why completion and not mean score: on a benchmark that pays partial credit, a skill that makes the agent race through the mechanics and stop before answering can beat a baseline that sometimes finishes. Counting a run only when it finished rejects that trade. If your headline is a success rate, the gate should judge completion too. A tie (`min_gain=0.0`) admits a skill that showed no benefit, and a single replay per arm decides on noise; a small positive `min_gain` and two repeats are cheap insurance.

## Stateful tools: `environment=`

A replay has only the recorded arguments, and a function's tools are fixed. When a tool drives a stateful environment (a game, a simulator, a sandbox), that environment must be rebuilt from the arguments on every run, and the outcome usually lives in the environment rather than in the return value. Pass a factory:

```python
class Episode(Protocol):
    tools: Sequence[Any]                        # tools bound to this run's environment
    def outcome(self, result: Any) -> float: ...  # in [0, 1]
    def close(self) -> None: ...

def environment(arguments: Mapping[str, Any]) -> Episode: ...  # or async

@skillify(library="skills/play", replay=True, environment=environment)
@ai_function(...)
async def play(level: str) -> str: ...
```

Every call and every gate replay — each arm, each repeat — calls `environment(arguments)` for a fresh episode, runs with the episode's tools added for that run only, and is scored by `episode.outcome(result)`. `AIFunctionError` still scores 0.0 and any other exception leaves the call unscored; an outcome that raises or falls outside [0, 1] is unscored too. `close()` always runs, and a close that raises is logged without masking the call's result. A factory that raises on a live call reaches the caller and nothing is recorded; on a replay it is an infrastructure failure, retried and then failing the round. `score` and `environment` are exclusive, since an outcome has one source.

## Skills as graph parameters

A library version is the function's procedural memory exposed as an ordinary `MemoryBackend`: each stored skill is a scalar parameter holding its body, and a synthetic `skill_index` parameter carries one `"<slug>: <description>"` line per skill. When the agent loads a skill through the `Skill` tool during `solve.trace(...)`, that load is a recall edge, so the stored procedure is a parameter node of the traced call's graph and feedback on the final output propagates back to it. Skills the agent never loaded receive no gradient.

`solve.skill_backend` is the supported way to hand the library to an optimizer:

```python
from ai_functions.optimizer import TextGradOptimizer

result = await solve.trace(task="...")
await TextGradOptimizer(model=model).step(
    result, "the report skipped the verification step", backends=[solve.skill_backend]
)
```

The backend is *staged*: a rewrite goes to the library's `pending/` directory, never into a published version, and `delete` raises `TypeError`; an index edit stages only the skills whose description it changes. The next round takes every staged revision, gates it as a `Candidate` with `replaces` set — `CompletionGate` measures the library with the revision swapped in against the library as published — and publishes it only if it shows a gain of at least `min_gain`. Admitted revisions are listed under `RoundReport.revised` and do not count against `max_skills`; a revision that makes things worse is rejected and the published text stands. Two rewrites of one skill before a round leave only the last one staged. A revision whose skill is no longer installed (rolled back, say) is discarded as `"stale"`, one that does not load as `"invalid"`. A staged revision never triggers a round by itself, but the next round that runs gates it, even one whose calls give it nothing to mine.

## Limits

- **A Unix-like system is required.** The skills library uses `fcntl` file locks, so it currently runs on systems such as Linux and macOS. Native Windows is not supported.
- `replay=True` **is required** because gating re-runs recorded calls, which re-runs the function's tools. Pass it only once that is safe.
- **No signal, no rounds.** With no post-conditions, no `score` and no `environment`, every call scores 1.0 and rounds always decline; the decorator warns.
- **Background rounds need a long-lived event loop.** `run_sync` records the call but starts no round; the pending count is on disk, so the next async call or `refresh_skills()` runs it. Calling through a fresh `asyncio.run(solve(...))` per call has the same limit: the round starts, but is cancelled when that loop shuts down (logged as a warning; its calls stay pending). Call from a long-lived loop, or run rounds explicitly with `refresh_skills()`.
- `max_skills` **is a ceiling, not an eviction policy.** Installed skills are never removed by a round, so once the library holds `max_skills` skills, no further new skill is admitted; revisions still are.
- **The gate's evidence is in-sample.** Trial inputs are recorded calls of the same function, and a round with no replayable calls outside its window — the first round, at least — replays the very calls it mined from. A round's reported gain says the skill helped on those calls; measure on held-out tasks before quoting a number.
- **Gate replays run the undecorated base function** with the model it was configured with, including when `@routed` is stacked on top: a round measures skills on that function, not on whichever model the router would pick.

**Choosing tasks** matters more than any setting here. A task whose baseline already completes every time has no headroom, and one that never completes under any condition has no signal. Check that the no-skill baseline sits off both ceiling and floor before spending a run budget, and report a rate with its spread over repeats, not a single run.

## Examples


| Example          | Shows                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `skillify_demo/` | `@skillify` on ScienceWorld with the example's full pipeline plugged in as `miner=`/`gate=`: an AI Function is called 24 times on training tasks, and two background rounds mine and gate skills from those calls. With a 60-step budget, the learned library raised Haiku 4.5's success rate on 15 held-out tasks from 46.7% to 81.7% (+35.0 pts, one-sided p < 0.001, 4 runs per arm) and cut steps per episode by 36% |
| `skillsbench/`   | The optimizer loop over a `workflow_guidelines` memory parameter with pytest oracles, on real SkillsBench tasks                                                                                                                                                                                                                                                                                                          |


Run the `@skillify` example from `examples/skillify_demo/` with `uv run --project ../.. --package ai-functions-skillify-demo-example main.py` (it is a workspace member, so `uv` is pointed at the repository root); it needs Bedrock credentials and a JVM. See its [README](../examples/skillify_demo/README.md) for the design, the flags and the full result.
