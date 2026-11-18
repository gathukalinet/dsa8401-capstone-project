import csv
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import svgwrite


from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "body_measurements_true.csv"
PATTERN_DIR = RAW_DIR / "patterns"

# ==========================================
# 1. GENERATE / VERIFY CSV WITH SIZE LABELS
# ==========================================
def assign_size_label(waist_cm):
    if waist_cm < 68.0:
        return "XS"
    elif waist_cm < 76.0:
        return "S"
    elif waist_cm < 84.0:
        return "M"
    elif waist_cm < 92.0:
        return "L"
    elif waist_cm < 100.0:
        return "XL"
    else:
        return "XXL"

SIZE_PLAN = {            # size: (n_sets, base_waist_cm)
    "XS": (17, 64.0), "S": (17, 72.0), "M": (17, 80.0),
    "L": (17, 88.0), "XL": (16, 96.0), "XXL": (16, 104.0),
}

def create_sized_csv(filename, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for size, (n, base_w) in SIZE_PLAN.items():
        for _ in range(n):
            waist = round(float(np.clip(rng.normal(base_w, 1.5), base_w - 3.5, base_w + 3.5)), 1)
            hip = round(waist + float(rng.normal(24, 2)), 1)
            waist_to_hip = round(float(rng.normal(20, 1.5)), 1)
            total_length = round(float(rng.normal(98, 3)), 1)
            rows.append((size, waist, hip, waist_to_hip, total_length))
    df = pd.DataFrame(rows, columns=["size_label", "waist_circ", "hip_circ", "waist_to_hip", "total_length"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"PAT_{i+1:03d}" for i in range(len(df))])
    df.to_csv(filename, index=False)

# ==========================================
# 2. PARAMETRIC DRAFTING FUNCTIONS
# ==========================================
def interpolate_bezier(p0, p1, p2, num_points=20):
    t = np.linspace(0, 1, num_points)
    return [( (1-ti)**2 * p0[0] + 2*(1-ti)*ti * p1[0] + ti**2 * p2[0],
              (1-ti)**2 * p0[1] + 2*(1-ti)*ti * p1[1] + ti**2 * p2[1] ) for ti in t]

def generate_front_panel(waist_circ, hip_circ, waist_to_hip, total_length, inseam):
    seam = 1.5
    w_q = (waist_circ / 4) + seam
    h_q = (hip_circ / 4) + seam
    crotch_depth = total_length - inseam
    crotch_ext = h_q * 0.15
    knee_depth = crotch_depth + (inseam * 0.5)
    grain_x = h_q * 0.5

    p_center_waist = (0.0, 0.0)
    p_side_waist = (w_q + 1.0, 0.5)
    p_side_hip = (h_q, -waist_to_hip)
    p_thigh_outer = (h_q, -crotch_depth)
    p_knee_outer = (grain_x + 11.0, -knee_depth)
    p_hem_outer = (grain_x + 10.0, -total_length)
    p_hem_inner = (grain_x - 10.0, -total_length)
    p_knee_inner = (grain_x - 11.0, -knee_depth)
    p_crotch_point = (-crotch_ext, -crotch_depth)

    waist_curve = interpolate_bezier(p_center_waist, (grain_x, -0.5), p_side_waist)
    hip_curve = interpolate_bezier(p_side_waist, (h_q - 0.5, -waist_to_hip * 0.5), p_side_hip)
    crotch_curve = interpolate_bezier(p_center_waist, (0.0, -crotch_depth * 0.7), p_crotch_point)

    boundary = []
    boundary.extend(waist_curve)
    boundary.extend(hip_curve[1:])
    boundary.extend([p_thigh_outer, p_knee_outer, p_hem_outer, p_hem_inner, p_knee_inner, p_crotch_point])
    boundary.extend(crotch_curve[::-1][1:])
    return Polygon(boundary)

def generate_back_panel(waist_circ, hip_circ, waist_to_hip, total_length, inseam):
    seam = 1.5
    w_q = (waist_circ / 4) + seam + 3.0
    h_q = (hip_circ / 4) + seam + 1.0
    crotch_depth = total_length - inseam
    back_crotch_ext = (h_q * 0.15) * 2.5
    knee_depth = crotch_depth + (inseam * 0.5)
    grain_x = h_q * 0.5

    p_center_back_waist = (2.5, 3.5)
    p_side_waist = (w_q + 2.5, 0.5)
    p_side_hip = (h_q + 2.0, -waist_to_hip)
    p_thigh_outer = (h_q + 2.0, -crotch_depth)
    p_knee_outer = (grain_x + 12.5, -knee_depth)
    p_hem_outer = (grain_x + 11.5, -total_length)
    p_hem_inner = (grain_x - 11.5, -total_length)
    p_knee_inner = (grain_x - 12.5, -knee_depth)
    p_back_crotch_point = (-back_crotch_ext, -crotch_depth)

    waist_curve = interpolate_bezier(p_center_back_waist, (grain_x + 2.0, 1.5), p_side_waist)
    hip_curve = interpolate_bezier(p_side_waist, (h_q + 1.5, -waist_to_hip * 0.5), p_side_hip)
    crotch_curve = interpolate_bezier(p_center_back_waist, (1.0, -crotch_depth * 0.6), p_back_crotch_point)

    boundary = []
    boundary.extend(waist_curve)
    boundary.extend(hip_curve[1:])
    boundary.extend([p_thigh_outer, p_knee_outer, p_hem_outer, p_hem_inner, p_knee_inner, p_back_crotch_point])
    boundary.extend(crotch_curve[::-1][1:])
    return Polygon(boundary)

def generate_waistband_panel(waist_circ, band_width=4.0, extension=4.0, seam=1.5):
    total_height = (band_width * 2) + (2 * seam)
    total_length = waist_circ + extension + (2 * seam)
    return Polygon([(0.0, 0.0), (total_length, 0.0), (total_length, total_height), (0.0, total_height)])

def generate_fly_assembly(crotch_depth, fly_length=18.0, fly_width=4.0, seam=1.5):
    """Generates fly components with Bézier curves on Facing and Shield."""
    actual_length = min(fly_length, crotch_depth * 0.75) + seam
    total_width = fly_width + seam
    
    # 1. Outer Fly Facing (J-Curve)
    j_curve = interpolate_bezier(
        (total_width, -actual_length + 4.5), 
        (total_width, -actual_length), 
        (0.0, -actual_length), 
        num_points=15
    )
    facing_pts = [(0.0, 0.0), (total_width, 0.0)] + j_curve + [(0.0, 0.0)]
    
    # 2. Fly Shield (Bézier rounded bottom fold)
    shield_width = (fly_width * 2) + (2 * seam)
    # Rounded J-curve corners at the bottom of the fly shield
    corner_radius = 2.5
    bottom_y = -actual_length
    
    # Right corner J-curve
    right_j = interpolate_bezier(
        (shield_width, bottom_y + corner_radius),
        (shield_width, bottom_y),
        (shield_width - corner_radius, bottom_y),
        num_points=10
    )
    # Left corner J-curve
    left_j = interpolate_bezier(
        (corner_radius, bottom_y),
        (0.0, bottom_y),
        (0.0, bottom_y + corner_radius),
        num_points=10
    )
    
    shield_bottom_curve = right_j + left_j
    
    shield_pts = [(0.0, 0.0), (shield_width, 0.0)] + shield_bottom_curve + [(0.0, 0.0)]
    
    # 3. Straight Inner Zipper Guard
    guard_width = fly_width + (2 * seam)
    guard_pts = [(0.0, 0.0), (guard_width, 0.0), (guard_width, -actual_length), (0.0, -actual_length), (0.0, 0.0)]
    
    return Polygon(facing_pts), Polygon(shield_pts), Polygon(guard_pts)

def generate_pocket_assembly(hip_circ, waist_to_hip, opening_width=14.0, pocket_depth=22.0, pocket_width=14.0, seam=1.5):
    """Generates proportionally constrained pocket bags with smooth lower basins."""
    
    # Cap maximum pocket width so it never crosses the front center grainline
    max_allowed_width = (hip_circ / 4.0) * 0.55  # Never wider than 55% of the front hip
    adj_width = min(pocket_width + seam, max_allowed_width)
    adj_depth = pocket_depth + seam
    slant_x = 3.5
    
    # Curved lower pouch basin
    curve_start_y = max(waist_to_hip * 0.8, opening_width + 1.0)
    pocket_bottom_curve = interpolate_bezier(
        (adj_width, -curve_start_y),
        (adj_width, -adj_depth),
        (0.0, -adj_depth),
        num_points=15
    )
    
    # Outer Facing Bag
    facing_pts = [(0.0, 0.0), (adj_width, 0.0)] + pocket_bottom_curve + [(0.0, 0.0)]
    
    # Inner Bag (with slant cut-out)
    inner_pts = [(0.0, 0.0), (adj_width - slant_x, 0.0), (adj_width, -opening_width)] + pocket_bottom_curve + [(0.0, 0.0)]
    
    return Polygon(facing_pts), Polygon(inner_pts)

# ==========================================
# 3. NON-OVERLAPPING DYNAMIC SVG EXPORTER
# ==========================================
def export_fitted_annotated_svg(polygon, filepath, sample_id, size_lbl, piece_name, padding=5.0):
    minx, miny, maxx, maxy = polygon.bounds
    poly_width = maxx - minx
    poly_height = maxy - miny
    
    width = poly_width + (2 * padding)
    height = poly_height + (2 * padding)
    
    normalized_pts = [
        (x - minx + padding, maxy - y + padding) 
        for x, y in polygon.exterior.coords
    ]
    
    centroid = polygon.centroid
    cx_norm = centroid.x - minx + padding
    cy_norm = maxy - centroid.y + padding

    dwg = svgwrite.Drawing(filepath, size=(f"{width * 10}px", f"{height * 10}px"), viewBox=f"0 0 {width} {height}")
    
    # Render Pattern Polygon
    dwg.add(dwg.polygon(
        points=normalized_pts, 
        fill="#e9ecef", 
        stroke="#111111", 
        stroke_width=0.3,
        id=f"{sample_id}_{size_lbl}_{piece_name}"
    ))
    
    # Adaptive Orientation & Formatting
    orient_vertical = poly_height > (poly_width * 1.5)
    usable_span = poly_height if orient_vertical else poly_width
    
    # Proportional font size and exact line spacing calculation
    if usable_span < 12.0:  # Very narrow pieces (Fly Guard, Fly Facing)
        font_sz = 0.55
        line_spacing = font_sz * 1.25
        lines = [sample_id, f"SIZE:{size_lbl}", piece_name.replace("_", " ")]
    elif usable_span < 25.0:  # Small/Medium pieces (Pockets, Fly Shield)
        font_sz = 0.85
        line_spacing = font_sz * 1.25
        lines = [f"{sample_id} | {size_lbl}", piece_name.replace("_", " ")]
    else:  # Large panels (Legs, Waistband)
        font_sz = 1.60
        line_spacing = font_sz * 1.25
        lines = [f"{sample_id} | SIZE:{size_lbl}", piece_name.replace("_", " ")]

    text_group = dwg.g(id="pattern_annotation")
    
    # Calculate initial Y offset to prevent overlapping
    total_block_height = (len(lines) - 1) * line_spacing
    start_y = cy_norm - (total_block_height / 2.0)
    
    for i, line in enumerate(lines):
        y_pos = start_y + (i * line_spacing)
        txt = dwg.text(
            line,
            insert=(cx_norm, y_pos),
            font_size=f"{font_sz}px",
            font_family="Arial",
            font_weight="bold",
            text_anchor="middle",
            fill="#b23b3b"
        )
        
        # Rotate text 90 degrees if piece is tall and thin
        if orient_vertical and poly_width < 10.0:
            txt.rotate(90, (cx_norm, y_pos))
            
        text_group.add(txt)
        
    dwg.add(text_group)
    dwg.save()

# ==========================================
# 4. MASTER BATCH GENERATION ENGINE
# ==========================================
def main():
    print(ROOT)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    create_sized_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"\nGenerating 12-piece sets with curved pocket/fly shapes and non-overlapping labels...")

    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        size_lbl = row['size_label']
        waist = row['waist_circ']
        hip = row['hip_circ']
        w2h = row['waist_to_hip']
        length = row['total_length']
        crotch_depth = length - 25.0
        inseam = crotch_depth

        # Generate Geometries
        front_left = generate_front_panel(waist, hip, w2h, length, inseam)
        front_right = generate_front_panel(waist, hip, w2h, length, inseam)
        back_left = generate_back_panel(waist, hip, w2h, length, inseam)
        back_right = generate_back_panel(waist, hip, w2h, length, inseam)
        waistband = generate_waistband_panel(waist)
        fly_facing, fly_shield, fly_guard = generate_fly_assembly(crotch_depth)
        pocket_facing_l, pocket_inner_l = generate_pocket_assembly(hip, w2h)
        pocket_facing_r, pocket_inner_r = generate_pocket_assembly(hip, w2h)

        garment_pieces = [
            (front_left, "FRONT_LEFT"),
            (front_right, "FRONT_RIGHT"),
            (back_left, "BACK_LEFT"),
            (back_right, "BACK_RIGHT"),
            (waistband, "WAISTBAND"),
            (fly_facing, "FLY_FACING"),
            (fly_shield, "FLY_SHIELD"),
            (fly_guard, "FLY_GUARD"),
            (pocket_facing_l, "POCKET_FACING_LEFT"),
            (pocket_inner_l, "POCKET_INNER_LEFT"),
            (pocket_facing_r, "POCKET_FACING_RIGHT"),
            (pocket_inner_r, "POCKET_INNER_RIGHT"),
        ]

        sample_dir = os.path.join(PATTERN_DIR, f"Size_{size_lbl}", sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        for poly, piece_name in garment_pieces:
            filename = f"{sample_id}_{size_lbl}_{piece_name.lower()}.svg"
            export_fitted_annotated_svg(
                polygon=poly, 
                filepath=os.path.join(sample_dir, filename), 
                sample_id=sample_id, 
                size_lbl=size_lbl, 
                piece_name=piece_name
            )

    print("\nSUCCESS! Updated curves and clean, non-overlapping labels applied across all SVGs.")

if __name__ == "__main__":
    main()