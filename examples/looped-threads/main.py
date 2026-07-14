"""Looped-Threads — an AI-Thread that learns to price houses.

A modify -> run -> keep/discard -> repeat loop: a *Looped-Thread* (an AI-Thread
that continuously edits and re-runs one script) searches for the best model on the
Kaggle "House Prices" (Ames) dataset. Starting from a one-line linear baseline it
rewrites ``build_model()`` to EXPLORE model families (linear, tree ensembles, a
1-hidden-layer neural net, stacking), SELECTS the best, and REFINES it. A locked,
code-measured post-condition scores a held-out validation split and, via the
``AIFunction.replace`` ratchet, only accepts edits that beat the best so far — so
overfitting cannot win.

Files: ``seed_model.py`` is the tunable script the thread edits; ``plotting.py``
draws the learning-curve figure; this file is the loop. See ``README.md``.
"""

import asyncio
import csv
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reach examples/_utils.py

from _utils import display, rule  # noqa: E402
from plotting import plot_learning_curve, read_experiments  # noqa: E402
from strands import tool  # noqa: E402

from ai_functions import ai_function  # noqa: E402
from ai_functions.ai_thread import AIFunctionError, PostConditionResult  # noqa: E402

model = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

ROUNDS = 3
MAX_ATTEMPTS = 5

SEED_MODEL = Path(__file__).parent / "seed_model.py"
# The scratch lab (model.py + notes.md + experiments.csv) lives beside this file
# and is wiped at the start of every run, so each run starts fresh from the seed.
LAB = Path(__file__).parent / ".lab"
SEED_NOTES = """\
# House Prices — Looped-Threads notes

The lab notebook. Each round reads it first to avoid repeating tried ideas, then
appends one line per experiment. The metric of record is the validation RMSE of
log(SalePrice) — LOWER is better; a low train_rmse beside a high validation RMSE
means overfitting.

Format: `- val_rmse=.. train_rmse=.. | MODEL_NAME=.. | <what changed / lesson>`

## Log

(no experiments yet)
"""

EXPERIMENTS_COLUMNS = ["exp", "model", "train_rmse", "val_rmse"]


# ── The locked evaluator: run the script, read the metrics it prints ──


def _parse_metrics(stdout: str) -> dict[str, float]:
    """Parse every numeric ``key=value`` token printed by model.py."""
    metrics: dict[str, float] = {}
    for token in stdout.split():
        key, sep, value = token.partition("=")
        if sep:
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    return metrics


def _parse_model_name(stdout: str) -> str:
    """Return the agent-set MODEL_NAME (the whole ``model=`` line, spaces allowed)."""
    for line in stdout.splitlines():
        if line.startswith("model="):
            return line.removeprefix("model=").strip()
    return "?"


def measure(model_path: Path) -> float:
    """Run model.py and return the validation RMSE it prints."""
    proc = subprocess.run([sys.executable, str(model_path)], capture_output=True, text=True, timeout=300)
    metrics = _parse_metrics(proc.stdout)
    if "rmse" not in metrics:
        raise RuntimeError(f"model.py printed no rmse= line:\n{proc.stdout}\n{proc.stderr}")
    return metrics["rmse"]


def make_gate(model_path: Path, target: float):
    """Post-condition that accepts the edit only if validation RMSE beats ``target``."""

    def val_rmse_below_target(_result: object, **_kwargs: object) -> PostConditionResult | None:
        measured = measure(model_path)
        if measured < target:
            return None
        return PostConditionResult(
            passed=False,
            message=textwrap.dedent(f"""\
                Validation RMSE {measured:.6f} does not beat the target {target:.6f}.
                The gate scores the held-out validation RMSE, not train_rmse: if train_rmse
                is far below it your model is overfitting. Try a simpler model or stronger
                regularization (e.g. a higher-alpha linear model, a shallower/less-greedy
                tree ensemble, or more weight decay in the MLP), then re-run."""),
        )

    return val_rmse_below_target


def _run_and_log(model_path: Path) -> subprocess.CompletedProcess[str]:
    """Run model.py once and append the experiment (config + metrics) to experiments.csv."""
    proc = subprocess.run([sys.executable, str(model_path)], capture_output=True, text=True, timeout=300)
    metrics = _parse_metrics(proc.stdout)
    if "rmse" not in metrics:  # broken edit — surface the error to the thread, log nothing
        return proc
    csv_path = model_path.parent / "experiments.csv"
    rows = read_experiments(csv_path)
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if not rows:
            writer.writerow(EXPERIMENTS_COLUMNS)
        writer.writerow(
            [
                len(rows) + 1,
                _parse_model_name(proc.stdout),
                f"{metrics.get('train_rmse', float('nan')):.6f}",
                f"{metrics['rmse']:.6f}",
            ]
        )
    return proc


# ── The thread's file tools (run_python doubles as the experiment recorder) ──
# The tools are confined to the lab directory: a path that resolves outside LAB
# (e.g. via ``..`` or an absolute path elsewhere) is refused, so the thread can
# only read, write, and run files within its own working directory.


def _resolve_in_lab(path: str) -> Path | None:
    """Resolve `path` and return it only if it stays inside LAB, else None."""
    resolved = Path(path).resolve()
    lab = LAB.resolve()
    return resolved if (resolved == lab or resolved.is_relative_to(lab)) else None


@tool
def read_file(path: str) -> str:
    """Read and return the full text of the file at `path` (must be inside the lab)."""
    target = _resolve_in_lab(path)
    if target is None:
        return f"refused: {path} is outside the working directory {LAB}"
    return target.read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Overwrite the file at `path` (must be inside the lab) and confirm the byte count."""
    target = _resolve_in_lab(path)
    if target is None:
        return f"refused: {path} is outside the working directory {LAB}"
    target.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


@tool
def run_python(path: str) -> str:
    """Run the Python script at `path` (must be inside the lab); return its stdout/stderr."""
    target = _resolve_in_lab(path)
    if target is None:
        return f"refused: {path} is outside the working directory {LAB}"
    proc = _run_and_log(target)
    return f"exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"


# ── The Looped-Thread: explores model families, then refines the winner ──


@ai_function[Literal["done"]](model=model, tools=[read_file, write_file, run_python], max_attempts=MAX_ATTEMPTS)
def optimize(model_path: str, notes_path: str, target: float, guidance: str):
    """
    You are an ML researcher improving a house-price regressor at {model_path},
    working one round at a time toward a lower validation RMSE (of log-price).

    FIRST, read the lab notebook at {notes_path} for what has already been tried;
    do not repeat a failed idea. Then read {model_path}. Edit ONLY the block marked
    "EDIT THIS": ADD the model you want to try to the `MODELS` catalog (keep every
    entry already there, each in its best configuration — never delete one), add
    whatever imports it needs, and set `MODEL_NAME` to the entry to score, using a
    short label that captures the model and its key settings (e.g. "gb n=300 d=3
    lr=.08"). Do NOT touch anything below the "DO NOT EDIT" line — the data, split,
    and metric are fixed. Run the script with run_python after each edit; every run
    is one logged experiment. By the end, `MODELS` should hold every family you
    tried in its best config, with `MODEL_NAME` pointing at the winner.

    Models worth trying (all in scikit-learn):
      - Linear, regularized:  from sklearn.linear_model import Ridge, Lasso, ElasticNet
      - Tree ensembles:       from sklearn.ensemble import RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor
      - Small neural net:     from sklearn.neural_network import MLPRegressor
                              (e.g. hidden_layer_sizes=(128, 64), max_iter=2000 — try a
                              couple of hidden layers)
      - Ensemble of the best: from sklearn.ensemble import StackingRegressor, VotingRegressor

    {guidance}

    If `train_rmse` is far below the validation `rmse` the model is OVERFITTING —
    a bigger/deeper model will make it worse. Reach for a simpler model or stronger
    regularization instead.

    BEFORE finishing, append one line per experiment to {notes_path}:
      - val_rmse=.. train_rmse=.. | MODEL_NAME=.. | <what changed / lesson>

    The current best validation RMSE to beat is {target:.6f}. Reply 'done' once the
    script prints a validation RMSE below that and you have updated the notebook.
    """


EXPLORE_GUIDANCE = (
    "This is the EXPLORATION phase. Run 5-7 experiments, no more: add SEVERAL "
    "DIFFERENT model families to MODELS (a regularized linear model, a random forest, "
    "gradient boosting, and a small MLP), note each one's validation RMSE, then set "
    "MODEL_NAME to the MOST PROMISING entry and reply 'done'. Do not sweep."
)
REFINE_GUIDANCE = (
    "This is the REFINEMENT phase. Run 4-6 experiments, no more: improve the best "
    "entry's config (or add a StackingRegressor that blends the top few), updating its "
    "MODELS entry in place and pointing MODEL_NAME at it, then reply 'done'."
)


def _seed_lab() -> Path:
    """Wipe any previous lab, then seed a fresh one and warm the data cache."""
    shutil.rmtree(LAB, ignore_errors=True)  # start every run fresh from the seed
    LAB.mkdir(parents=True)
    (LAB / "model.py").write_text(SEED_MODEL.read_text())
    (LAB / "notes.md").write_text(SEED_NOTES)
    from sklearn.datasets import fetch_openml  # warm the OpenML cache so runs are offline

    fetch_openml("house_prices", as_frame=True, parser="auto")
    return LAB


async def main() -> None:
    lab = _seed_lab()
    model_path, notes_path = lab / "model.py", lab / "notes.md"

    rule("Looped-Threads — house-price prediction")
    _run_and_log(model_path)  # log the seed as experiment 1
    best = measure(model_path)
    print(f"baseline val_rmse = {best:.6f}  (lab: {lab})")

    for r in range(ROUNDS):
        guidance = EXPLORE_GUIDANCE if r == 0 else REFINE_GUIDANCE
        gated = optimize.replace(post_conditions=[make_gate(model_path, best)])
        try:
            await gated(model_path=str(model_path), notes_path=str(notes_path), target=best, guidance=guidance)
        except AIFunctionError:  # nothing beat the best within max_attempts — converged
            print(f"round {r}: no edit beat {best:.6f} — converged")
            break
        best = measure(model_path)
        print(f"round {r}: new best val_rmse = {best:.6f}")

    figure = Path(__file__).parent / "house_prices_search.png"
    plot_learning_curve(lab / "experiments.csv", figure)
    print(f"\nfinal val_rmse = {best:.6f}")
    display("Learning curve saved", str(figure), lang="text")
    display("Lab notebook", notes_path.read_text(), lang="markdown")


if __name__ == "__main__":
    asyncio.run(main())
