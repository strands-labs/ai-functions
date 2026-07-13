"""The learning-curve figure for the Looped-Threads example.

Kept separate from ``main.py`` so the orchestrator stays small. Nothing here
touches the runtime — it only reads ``experiments.csv`` and draws matplotlib.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

_BLUE = "#1f77b4"
_DARKBLUE = "#12507a"


def read_experiments(csv_path: Path) -> list[dict[str, str]]:
    """Return the logged experiments as row dicts (empty if the file is absent)."""
    if not csv_path.exists():
        return []
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


@dataclass
class Step:
    """A best-so-far improvement: the experiment that set a new RMSE low."""

    exp: int
    rmse: float
    name: str  # the agent-set MODEL_NAME for this experiment


def best_so_far(rows: list[dict[str, str]], key: str = "val_rmse") -> list[Step]:
    """Running minimum of ``key``: each experiment that set a new low, in order."""
    steps: list[Step] = []
    best = float("inf")
    for row in rows:
        rmse = float(row[key])
        if rmse < best - 1e-9:
            best = rmse
            steps.append(Step(int(row["exp"]), rmse, row["model"]))
    return steps


def plot_learning_curve(csv_path: Path, out_path: Path) -> None:
    """Best validation RMSE vs. experiment number, every step labelled.

    Log y-axis to zoom the packed low-RMSE region. Each best-so-far step gets a
    slanted label placed at the point, showing the agent-set MODEL_NAME.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_experiments(csv_path)
    exps = [int(r["exp"]) for r in rows]
    rmse = [float(r["val_rmse"]) for r in rows]
    steps = best_so_far(rows)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(exps, rmse, s=22, color="#9ecae1", alpha=0.7, label="each experiment", zorder=1)
    ax.step(
        [s.exp for s in steps],
        [s.rmse for s in steps],
        where="post",
        color=_BLUE,
        lw=2.2,
        marker="o",
        label="best so far",
        zorder=2,
    )

    ax.set_yscale("log")
    # Fit to the trajectory: best-so-far to the baseline (the first, highest step).
    # A diverged experiment (e.g. an unconverged MLP) is left to clip off the top
    # rather than squashing the whole plot.
    top = steps[0].rmse if steps else max(rmse)
    ax.set_ylim(min(rmse) * 0.96, top * 1.3)
    ax.set_xlim(0, max(exps) * 1.08)

    # Slanted label at each point showing the agent's MODEL_NAME for that step.
    for s in steps:
        ax.text(
            s.exp,
            s.rmse,
            f" {s.name}",
            ha="left",
            va="bottom",
            fontsize=8,
            rotation=30,
            rotation_mode="anchor",
            color=_DARKBLUE,
        )

    ax.set_title("Looped-Threads — house-price prediction", fontsize=13, y=1.04)
    ax.set_xlabel("experiment number")
    ax.set_ylabel("validation RMSE of log(SalePrice)")
    ax.legend(loc="lower left", fontsize=9)
    caption = (
        "A Looped-Thread is an AI-Thread that rewrites one model script, running each candidate as an\n"
        "experiment (dot); the stepped line tracks the best validation RMSE so far. It starts from a plain\n"
        "linear baseline and searches across model families — regularized linear models (ridge/lasso/\n"
        "elastic-net), tree ensembles (random forest, extra-trees, gradient boosting), a small neural\n"
        "net, and stacked ensembles — keeping only edits that beat the best. RMSE is of\n"
        "log(SalePrice) — Kaggle's official metric for this task; lower is better."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
