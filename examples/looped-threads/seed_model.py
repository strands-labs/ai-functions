"""Predict Ames house sale prices — grow the MODELS catalog to try different models.

You (the AI-Thread) edit ONLY the block marked "EDIT THIS": add each model you try
to the ``MODELS`` catalog (an unfitted sklearn regressor per entry), keeping the
ones already there in their best configuration, and set ``MODEL_NAME`` to the entry
you want scored this run. Everything below the "DO NOT EDIT" line — the data, the
train/validation split, the metric, and the printing — is fixed so the score cannot
be gamed. The target is the validation ``rmse=`` (RMSE of log price); LOWER is better.
"""

# ===== EDIT THIS: add models to MODELS (keep the ones already there), set MODEL_NAME =====
from sklearn.linear_model import LinearRegression

# Every model tried so far, each in its best configuration. The key is a short
# label shown on the plot; grow this catalog as you explore — do not drop entries.
MODELS = {
    "linear": LinearRegression(),
}
MODEL_NAME = "linear"  # the entry to score this run; build_model() returns MODELS[MODEL_NAME]


def build_model():
    """Return the currently selected model from the MODELS catalog above."""
    return MODELS[MODEL_NAME]


# ===== DO NOT EDIT BELOW: data, split, metric, and reporting are fixed =====

import numpy as np  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

TRAIN_FRACTION = 0.75  # first 75% trains, last 25% is the held-out validation split


def main():
    bundle = fetch_openml("house_prices", as_frame=True, parser="auto")
    numeric = bundle.data.select_dtypes("number").drop(columns=["Id"], errors="ignore")
    X = numeric.fillna(numeric.median(numeric_only=True)).to_numpy(dtype=float)
    y = np.log(bundle.target.to_numpy(dtype=float))  # always fit on log price (Kaggle's metric)

    split = int(len(X) * TRAIN_FRACTION)
    Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]

    # Standardize on TRAIN statistics only (no leakage from the validation split).
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    model = build_model().fit(Xtr, ytr)

    def rmse(features, target):
        return float(mean_squared_error(target, model.predict(features)) ** 0.5)

    print(f"model={MODEL_NAME}")
    print(f"n_train={len(Xtr)} n_val={len(Xva)}")
    print(f"train_rmse={rmse(Xtr, ytr):.6f}")
    print(f"rmse={rmse(Xva, yva):.6f}")  # the optimization target is VALIDATION rmse


if __name__ == "__main__":
    main()
