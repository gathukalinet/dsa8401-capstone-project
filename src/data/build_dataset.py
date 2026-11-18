"""Build the model-ready dataset.

Outputs (in data/processed/):
  tabular.csv  one row per layout: observed (noisy, partly missing) features,
               true values, target, group id and train/val/test split
  masks.npy    uint8 array (N, 128, 128, 1), row i matches tabular.csv row i

Run from the repo root:  python -m src.data.build_dataset
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_patterns import CSV_PATH
from src.data.layout import load_config, iter_layouts
from src.data.rasterize import to_mask, mask_area_cm2

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_COLS = ["waist_circ", "hip_circ", "total_length"]


def make_observed(df, sd_cm, missing_rate, seed):
    """Messiness, applied ONCE per garment: tape-measure noise + missing values.

    The pattern was cut from the TRUE values; the model only sees these.
    """
    rng = np.random.default_rng(seed)
    obs = df[["sample_id"]].copy()
    for col in FEATURE_COLS:
        noisy = (df[col] + rng.normal(0.0, sd_cm, len(df))).round(1)
        missing = rng.random(len(df)) < missing_rate
        obs[col] = noisy.where(~missing)              # NaN where missing
    obs["waist_hip_ratio"] = obs["waist_circ"] / obs["hip_circ"]   # NaN if either is missing
    return obs


def split_by_garment(df, test_size, val_size, seed):
    """Stratified-by-size split at GARMENT level: all layouts of a garment stay together."""
    rng = np.random.default_rng(seed)
    split = {}
    for _, g in df.groupby("size_label"):
        ids = g["sample_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_size))
        n_val = int(round(len(ids) * val_size))
        for i, sid in enumerate(ids):
            split[sid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return split


def main():
    cfg = load_config()
    scale = cfg["fabric"]["cm_per_pixel"]
    df_true = pd.read_csv(CSV_PATH)

    masks, rows, area_err = [], [], []
    for row, l_idx, res in iter_layouts(cfg, df_true):
        mask = to_mask(res["placed"], scale)
        masks.append(mask)
        area_err.append(100 * (mask_area_cm2(mask, scale) - res["pattern_area"]) / res["pattern_area"])
        rows.append({
            "mask_idx": len(masks) - 1,
            "sample_id": row["sample_id"],
            "layout_idx": l_idx,
            "size_label": row["size_label"],
            "pattern_area": res["pattern_area"],
            "marker_bbox_area": res["marker_bbox_area"],
            "used_length": res["used_length"],
            "utilization_pct": res["utilization_pct"],            # TARGET
        })
    layouts = pd.DataFrame(rows)

    # true measurements (kept for audit only) and observed (noisy / missing) features
    true_cols = df_true[["sample_id"] + FEATURE_COLS].rename(columns={c: c + "_true" for c in FEATURE_COLS})
    obs = make_observed(df_true, cfg["noise"]["measurement_sd_cm"], cfg["noise"]["missing_rate"], cfg["seed"])
    table = layouts.merge(obs, on="sample_id").merge(true_cols, on="sample_id")

    split_map = split_by_garment(df_true, cfg["split"]["test_size"], cfg["split"]["val_size"], cfg["seed"])
    table["split"] = table["sample_id"].map(split_map)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X = np.stack(masks).astype(np.uint8)
    np.save(PROCESSED_DIR / "masks.npy", X)
    table.to_csv(PROCESSED_DIR / "tabular.csv", index=False)
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump({"cm_per_pixel": scale, "mask_shape": list(X.shape[1:]),
                   "fabric_width_cm": cfg["fabric"]["width_cm"]}, f, indent=2)

    # ---- checks and summary ----
    # no garment may appear in more than one split
    assert table.groupby("sample_id")["split"].nunique().max() == 1, "garment split across sets"
    # mask must be row-aligned with the table
    assert len(X) == len(table) and (table["mask_idx"].to_numpy() == np.arange(len(table))).all()

    coverage = X.reshape(len(X), -1).mean(axis=1)
    print(f"samples: {len(table)} | masks: {X.shape} | dtype {X.dtype}")
    print("rows per split:\n", table["split"].value_counts().to_string())
    print("garments per split:\n", table.groupby("split")["sample_id"].nunique().to_string())
    print("missing cells in observed features:\n", table[FEATURE_COLS + ["waist_hip_ratio"]].isna().sum().to_string())
    print(f"rasterisation area error (mask vs polygon): mean {np.mean(area_err):+.2f}% | worst {np.max(np.abs(area_err)):.2f}%")
    print(f"corr(mask coverage, utilization) = {np.corrcoef(coverage, table['utilization_pct'])[0, 1]:.3f}")


if __name__ == "__main__":
    main()