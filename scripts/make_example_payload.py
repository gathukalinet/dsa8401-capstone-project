"""Write examples/payload.json: one real test-set sample in the API's request format.

Run from the repo root:  python -m scripts.make_example_payload
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    t = pd.read_csv(ROOT / "data" / "processed" / "tabular.csv")
    X = np.load(ROOT / "data" / "processed" / "masks.npy")
    row = t[(t["split"] == "test") & t[["waist_circ", "hip_circ", "total_length"]].notna().all(axis=1)].iloc[0]
    payload = {
        "waist_circ": float(row["waist_circ"]),
        "hip_circ": float(row["hip_circ"]),
        "total_length": float(row["total_length"]),
        "mask": X[int(row["mask_idx"]), :, :, 0].astype(int).tolist(),     # 128 x 128 of 0/1
        "expected_utilization_pct": round(float(row["utilization_pct"]), 2),  # ignored by the API
    }
    out = ROOT / "examples" / "payload.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out} | sample {row['sample_id']} layout {int(row['layout_idx'])} | "
          f"true utilization {payload['expected_utilization_pct']}%")


if __name__ == "__main__":
    main()