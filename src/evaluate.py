"""Evaluate baselines and neural models on the held-out TEST garments.

Run from the repo root:  python -m src.evaluate
Writes reports/results.csv, reports/pred_vs_actual.png, reports/residuals_by_size.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

import json
from src.features.preprocess import ARTIFACT_DIR, PROCESSED_DIR, load_splits, to_pct
from src.train import REPORT_DIR, inputs_for

ORDER = ["mean_baseline", "coverage_linreg", "formula_linreg", "mlp_only", "cnn_only", "fusion"]


def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def coverage(split):
    """Fraction of the grid covered by pattern pieces (single number per mask)."""
    return split["mask"].mean(axis=(1, 2, 3)).reshape(-1, 1)


def formula_feature(split, scale, width):
    """Utilization (%) computed directly from the mask: area / (width x last filled row)."""
    m = split["mask"][..., 0]
    area = m.reshape(len(m), -1).sum(axis=1) * scale ** 2
    last_row = np.array([np.where(x.any(axis=1))[0].max() + 1 for x in m])
    return (100 * area / (width * last_row * scale)).reshape(-1, 1)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    splits, prep = load_splits()
    train = splits["train"]

    preds = {name: {} for name in ORDER}
    for sp in ["val", "test"]:
        s = splits[sp]
        # baseline 1: always predict the training mean
        preds["mean_baseline"][sp] = np.full(len(s["y_pct"]), train["y_pct"].mean())
        # baseline 2: linear regression on mean pixel coverage only
        lr = LinearRegression().fit(coverage(train), train["y_pct"])
        for sp in ["val", "test"]:
            preds["coverage_linreg"][sp] = lr.predict(coverage(splits[sp]))

        # baseline 3: formula computed from the mask, linearly calibrated on train
        with open(PROCESSED_DIR / "meta.json") as f:
            meta = json.load(f)
        scale, width = meta["cm_per_pixel"], meta["fabric_width_cm"]
        lr2 = LinearRegression().fit(formula_feature(train, scale, width), train["y_pct"])
        for sp in ["val", "test"]:
            preds["formula_linreg"][sp] = lr2.predict(formula_feature(splits[sp], scale, width))
    # neural models
    for name in ["mlp_only", "cnn_only", "fusion"]:
        model = keras.models.load_model(ARTIFACT_DIR / f"{name}.keras")
        for sp in ["val", "test"]:
            preds[name][sp] = to_pct(model.predict(inputs_for(model, splits[sp]), verbose=0), prep)

    rows = []
    for name in ORDER:
        for sp in ["val", "test"]:
            rows.append({"model": name, "split": sp, **metrics(splits[sp]["y_pct"], preds[name][sp])})
    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "results.csv", index=False)
    print("TEST SET (utilization in % points; MSE in %^2)")
    print(results[results["split"] == "test"].drop(columns="split").round(3).to_string(index=False))
    print("\nVALIDATION SET")
    print(results[results["split"] == "val"].drop(columns="split").round(3).to_string(index=False))

    # ---- predicted vs actual (test) ----
    y = splits["test"]["y_pct"]
    fig, axes = plt.subplots(1, len(ORDER), figsize=(4 * len(ORDER), 4), sharex=True, sharey=True)
    lo, hi = y.min() - 3, y.max() + 3
    for ax, name in zip(axes, ORDER):
        ax.scatter(y, preds[name]["test"], s=14, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name}\nR2 = {r2_score(y, preds[name]['test']):.2f}")
        ax.set_xlabel("actual utilization (%)")
    axes[0].set_ylabel("predicted utilization (%)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "pred_vs_actual.png", dpi=150)
    plt.close()

    # ---- residuals by size (fusion) ----
    res = pd.DataFrame({"size": splits["test"]["size"], "residual": preds["fusion"]["test"] - y})
    order = [s for s in ["XS", "S", "M", "L", "XL", "XXL"] if s in set(res["size"])]
    res.boxplot(column="residual", by="size", positions=range(len(order)), figsize=(6, 4))
    plt.suptitle("")
    plt.title("Fusion model residuals by size (test)")
    plt.ylabel("predicted - actual (%)")
    plt.savefig(REPORT_DIR / "residuals_by_size.png", dpi=150)
    plt.close()
    print("\nFusion mean absolute error by size (test):")
    print(res.assign(abs_err=res["residual"].abs()).groupby("size")["abs_err"].mean().reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
