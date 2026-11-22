"""Prediction with the saved fusion model (command-line).

Run from the repo root:
  python -m src.predict --payload examples/payload.json   # one layout, same format as the API
  python -m src.predict --split test                      # every layout of a split, with metrics
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

from src.features.preprocess import ARTIFACT_DIR, RAW_FEATS, load_splits, make_tabular, to_pct


def load_model():
    return keras.models.load_model(ARTIFACT_DIR / "fusion.keras")


def predict_one(model, prep, payload):
    """payload: optional waist_circ / hip_circ / total_length (cm) and a 128 x 128 'mask' of 0/1."""
    values = {c: payload.get(c) for c in RAW_FEATS}                 # missing -> imputed with train medians
    tab = make_tabular(pd.DataFrame([values]), prep)
    mask = np.asarray(payload["mask"], dtype="float32").reshape(1, 128, 128, 1)
    return float(to_pct(model.predict({"mask": mask, "tab": tab}, verbose=0), prep)[0])


def predict_split(model, prep, split):
    """Predict utilization (%) for every layout in a split; returns (predicted, actual)."""
    z = model.predict({"mask": split["mask"], "tab": split["tab"]}, verbose=0)
    return to_pct(z, prep), split["y_pct"]


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--payload", help="path to a JSON file in the API request format")
    group.add_argument("--split", choices=["train", "val", "test"], help="predict a whole data split")
    args = ap.parse_args()

    splits, prep = load_splits()
    model = load_model()

    if args.payload:
        with open(args.payload) as f:
            payload = json.load(f)
        pct = predict_one(model, prep, payload)
        print(f"predicted utilization: {pct:.2f}%  (waste {100 - pct:.2f}%)")
        if "expected_utilization_pct" in payload:
            print(f"expected (from the data): {payload['expected_utilization_pct']}%")
    else:
        pred, actual = predict_split(model, prep, splits[args.split])
        print(f"{args.split}: n={len(pred)} | MAE {mean_absolute_error(actual, pred):.3f} "
              f"| R2 {r2_score(actual, pred):.3f}")


if __name__ == "__main__":
    main()