"""Preprocessing shared by training, evaluation and the API.

Everything that is "learned" (medians, target mean/std) is fitted on the TRAIN split only.
Feature normalisation happens inside the models (keras Normalization layer, adapted on train).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

RAW_FEATS = ["waist_circ", "hip_circ", "total_length"]
TAB_FEATS = RAW_FEATS + ["waist_hip_ratio"]          # the 4 MLP inputs


def fit_preprocessor(train_df):
    """Train-only statistics: feature medians (for imputation) and target mean/std."""
    y = train_df["utilization_pct"]
    return {
        "medians": {c: float(train_df[c].median()) for c in RAW_FEATS},
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }


def make_tabular(df, prep):
    """Impute missing raw measurements, then derive waist_hip_ratio. Returns float32 (N, 4)."""
    out = df[RAW_FEATS].copy()
    for c in RAW_FEATS:
        out[c] = out[c].fillna(prep["medians"][c])
    out["waist_hip_ratio"] = out["waist_circ"] / out["hip_circ"]
    return out[TAB_FEATS].to_numpy(dtype="float32")


def to_pct(z, prep):
    """Convert a model output (standardised target) back to utilization %."""
    return np.asarray(z).reshape(-1) * prep["y_std"] + prep["y_mean"]


def save_preprocessor(prep):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "preprocessor.json", "w") as f:
        json.dump(prep, f, indent=2)


def load_preprocessor():
    with open(ARTIFACT_DIR / "preprocessor.json") as f:
        return json.load(f)


def load_splits():
    """Return ({'train'|'val'|'test': dict of arrays}, prep)."""
    df = pd.read_csv(PROCESSED_DIR / "tabular.csv")
    masks = np.load(PROCESSED_DIR / "masks.npy")
    prep = fit_preprocessor(df[df["split"] == "train"])

    splits = {}
    for name in ["train", "val", "test"]:
        d = df[df["split"] == name]
        splits[name] = {
            "mask": masks[d["mask_idx"].to_numpy()].astype("float32"),   # (N,128,128,1), 0/1
            "tab": make_tabular(d, prep),
            "y": ((d["utilization_pct"].to_numpy() - prep["y_mean"]) / prep["y_std"]).astype("float32"),
            "y_pct": d["utilization_pct"].to_numpy(),
            "size": d["size_label"].to_numpy(),
        }
    return splits, prep