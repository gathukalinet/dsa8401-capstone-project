"""Rasterise a placed layout into a fixed-scale binary mask (128 x 128 x 1).

Fixed scale: 1 pixel = cm_per_px cm, the SAME for every sample. We never rescale
to the layout's own bounding box, because that would leak the target.
1 = pattern piece, 0 = empty fabric.
"""
import numpy as np
from PIL import Image, ImageDraw


def to_mask(placed, cm_per_px, size=(128, 128), supersample=4):
    """placed: list of (name, shapely polygon) in cm. Returns uint8 array (H, W, 1)."""
    h_px, w_px = size

    # fail loudly if a layout would be clipped by the grid
    max_x = max(p.bounds[2] for _, p in placed)
    max_y = max(p.bounds[3] for _, p in placed)
    assert max_x <= w_px * cm_per_px and max_y <= h_px * cm_per_px, (
        f"layout {max_x:.0f} x {max_y:.0f} cm does not fit the grid; increase cm_per_pixel"
    )

    # draw at higher resolution, then average down and threshold (cleaner edges)
    big = Image.new("L", (w_px * supersample, h_px * supersample), 0)
    draw = ImageDraw.Draw(big)
    s = supersample / cm_per_px
    for _, poly in placed:
        draw.polygon([(x * s, y * s) for x, y in poly.exterior.coords], fill=1)
    arr = np.asarray(big, dtype=np.float32)
    arr = arr.reshape(h_px, supersample, w_px, supersample).mean(axis=(1, 3))
    return (arr >= 0.5).astype(np.uint8)[..., None]


def mask_area_cm2(mask, cm_per_px):
    """Area covered by the mask in cm^2 (used to check rasterisation error)."""
    return float(mask.sum()) * cm_per_px ** 2