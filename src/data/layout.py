"""Layout engine: arranges the 12 pieces of each garment on a fixed-width fabric.

Run from the repo root:  python -m src.data.layout
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from shapely import affinity

from src.data.generate_patterns import build_garment_pieces, CSV_PATH

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def normalise(poly):
    """Translate a polygon so its bounding box starts at (0, 0)."""
    minx, miny, _, _ = poly.bounds
    return affinity.translate(poly, xoff=-minx, yoff=-miny)


def maybe_flip(poly, rng, allow_rotation):
    """With probability 0.5 rotate 180 degrees (keeps the grain direction)."""
    if allow_rotation and rng.random() < 0.5:
        poly = affinity.rotate(poly, 180, origin="centroid")
    return normalise(poly)


def shelf_pack(pieces, fabric_width, gap, rng, allow_rotation):
    """First-fit shelf packing with a jittered tallest-first order.

    pieces: list of (name, polygon). Returns (placed, used_length) where
    placed is a list of (name, polygon) in fabric coordinates.
    """
    prepared = []
    for name, poly in pieces:
        p = maybe_flip(poly, rng, allow_rotation)
        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny
        if w > fabric_width:
            raise ValueError(f"{name} is {w:.1f} cm wide, wider than the fabric")
        prepared.append((name, p, w, h))

    # layout variation: tallest first, but with a random jitter on the height
    prepared.sort(key=lambda t: t[3] * rng.uniform(0.7, 1.3), reverse=True)

    shelves = []   # each shelf: {"y": bottom edge, "h": height, "x": next free x}
    placed = []
    for name, p, w, h in prepared:
        target = None
        for s in shelves:                         # first-fit into an existing shelf
            if s["x"] + w <= fabric_width and h <= s["h"]:
                target = s
                break
        if target is None:                        # otherwise open a new shelf
            y = 0.0 if not shelves else shelves[-1]["y"] + shelves[-1]["h"] + gap
            target = {"y": y, "h": h, "x": 0.0}
            shelves.append(target)
        placed.append((name, affinity.translate(p, xoff=target["x"], yoff=target["y"])))
        target["x"] += w + gap

    used_length = max(p.bounds[3] for _, p in placed)
    return placed, used_length


def build_layout(pieces, fabric_width, gap, allow_rotation, seed_parts):
    """Build one layout and compute the target (utilization %)."""
    rng = np.random.default_rng(seed_parts)       # e.g. [seed, garment_idx, layout_idx]
    placed, used_length = shelf_pack(pieces, fabric_width, gap, rng, allow_rotation)
    pattern_area = sum(p.area for _, p in placed)
    marker_bbox_area = fabric_width * used_length
    return {
        "placed": placed,
        "used_length": used_length,
        "pattern_area": pattern_area,
        "marker_bbox_area": marker_bbox_area,
        "utilization_pct": 100.0 * pattern_area / marker_bbox_area,
    }


def check_layout(placed, fabric_width, expected_area):
    """Assert: all pieces inside the fabric, none overlapping, no area lost."""
    for name, p in placed:
        minx, miny, maxx, maxy = p.bounds
        assert minx >= -1e-6 and miny >= -1e-6, f"{name} outside fabric (negative)"
        assert maxx <= fabric_width + 1e-6, f"{name} outside fabric width"
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            overlap = placed[i][1].intersection(placed[j][1]).area
            assert overlap < 1e-6, f"{placed[i][0]} overlaps {placed[j][0]}"
    total = sum(p.area for _, p in placed)
    assert abs(total - expected_area) < 1e-3, "pattern area changed during layout"


def iter_layouts(cfg=None, df=None):
    """Yield (row, layout_idx, layout_dict) for every garment x layout.

    Deterministic: the same config and seed always give the same layouts,
    so rasterize.py can call this instead of storing polygons.
    """
    cfg = cfg or load_config()
    df = df if df is not None else pd.read_csv(CSV_PATH)
    fab = cfg["fabric"]
    n_layouts = cfg["data"]["layouts_per_garment"]
    for g_idx, (_, row) in enumerate(df.iterrows()):
        pieces = [(name, poly) for poly, name in build_garment_pieces(row)]
        base_area = sum(p.area for _, p in pieces)
        for l_idx in range(n_layouts):
            result = build_layout(
                pieces, fab["width_cm"], fab["gap_cm"], fab["allow_rotation"],
                [cfg["seed"], g_idx, l_idx],
            )
            check_layout(result["placed"], fab["width_cm"], base_area)
            yield row, l_idx, result


if __name__ == "__main__":
    cfg = load_config()
    records = []
    for row, l_idx, res in iter_layouts(cfg):
        records.append({
            "sample_id": row["sample_id"],
            "size_label": row["size_label"],
            "layout_idx": l_idx,
            "used_length": res["used_length"],
            "pattern_area": res["pattern_area"],
            "marker_bbox_area": res["marker_bbox_area"],
            "utilization_pct": res["utilization_pct"],
        })
    out = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / "layouts.csv", index=False)

    L, U = out["used_length"], out["utilization_pct"]
    print(f"layouts built and checked: {len(out)}  (all passed check_layout)")
    print(f"used_length  cm : max {L.max():.1f} | mean {L.mean():.1f} | min {L.min():.1f}")
    print(f"utilization  %  : min {U.min():.1f} | mean {U.mean():.1f} | max {U.max():.1f} | sd {U.std():.2f}")
    print(out.groupby("size_label")[["used_length", "utilization_pct"]].mean().round(1))
    need = max(L.max(), cfg["fabric"]["width_cm"]) * 1.10 / 128
    print(f"minimum cm_per_pixel (10% margin): {need:.2f}  -> round UP, e.g. {np.ceil(need*10)/10:.1f}")