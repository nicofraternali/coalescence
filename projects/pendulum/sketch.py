"""
Double pendulum — chaotic trajectory traced and painted as topology.

A double pendulum runs under gravity and damping; its endpoint traces a
path. After MAX_POINTS samples, the trace is treated as a wall structure
and the resulting enclosed regions are flood-filled and colored using
greedy graph coloring (no two adjacent regions share a color, when
possible).

Each composition is determined by:
  - Six random initial conditions (m1, m2, r1, r2, a1_init, a2_init),
    sampled from fixed ranges using the seeded RNG.
  - The integration step count (MAX_POINTS).
  - The chosen theme (palette of region colors).

A note on reproducibility: the double pendulum is famously chaotic, but
the *integrator* is deterministic. Given the same seed, the same code,
and the same NumPy version, the trajectory reproduces bit-for-bit. The
chaotic sensitivity only matters if you tried to reconstruct a piece by
typing initial angles back from a sidecar — at which point precision
loss would diverge the trajectory.

Interactive keys:
    r       New random composition (new seed).
    space   New random theme + reset.
    s       Save the current piece (trace and art views).

Usage:
    uv run python projects/pendulum/sketch.py
    uv run python projects/pendulum/sketch.py --seed 4823 --theme JAPAN
    uv run python projects/pendulum/sketch.py --seed 4823 --max-points 1000
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import py5
from scipy.ndimage import label as ndi_label

from coalescence.io import save_artwork
from coalescence.palettes import PENDULUM_THEMES, list_pendulum_themes
from coalescence.seeds import init_seed


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

VIEW_W = 1200
VIEW_H = 600
RES_SCALE = 3.0  # 3.0 -> 3600x1800 high-res buffers

# Physics
DEFAULT_MAX_POINTS = 750
R_MIN_BASE, R_MAX_BASE = 100, 300
M_MIN, M_MAX = 10, 40
G = 1
DAMPING = 0.9995

DEFAULT_THEME = "JAPAN"
PROJECT_NAME = "pendulum"


# ---------------------------------------------------------------------------
# Module state.
# ---------------------------------------------------------------------------

# CLI-driven config
seed: int = 0
theme_name: str = DEFAULT_THEME
max_points: int = DEFAULT_MAX_POINTS
auto_save_and_exit: bool = False

# Theme
colors: dict = {}

# Physics state
r1: float = 0
r2: float = 0
m1: float = 0
m2: float = 0
a1: float = 0
a2: float = 0
a1_v: float = 0
a2_v: float = 0
px2: float = 0
py2: float = 0
initial_a1: float = 0
initial_a2: float = 0

# Trace
path_points: list[tuple[float, float]] = []

# Topology results
regions: list[list[tuple[int, int]]] = []
region_colors: list[str] = []
paint_index: int = 0
quality_metrics: dict = {}

# Lifecycle state machine
current_state: str = "RUNNING"  # RUNNING -> ANALYZING -> PAINTING -> DONE

# Buffers
pg_physics = None
pg_art = None
pg_analysis = None
f_reg = None
f_italic = None
f_bold = None

# Derived dimensions
buff_w = int((VIEW_W // 2) * RES_SCALE)
buff_h = int(VIEW_H * RES_SCALE)
TEXT_MARGIN = 90 * RES_SCALE
CY_POS = (buff_h - TEXT_MARGIN) / 2


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Double pendulum generative art.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic reproduction.")
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME,
                        choices=list_pendulum_themes(),
                        help=f"Theme name. Default: {DEFAULT_THEME}.")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS,
                        dest="max_points",
                        help=f"Number of trace points before topology pass. "
                             f"Default: {DEFAULT_MAX_POINTS}.")
    parser.add_argument("--save-and-exit", action="store_true",
                        help="With --seed, save both views and exit when DONE.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Composition reset.
# ---------------------------------------------------------------------------

def reset_composition(use_seed: int | None = None) -> None:
    """Seed RNG, sample new physical parameters, clear all state."""
    global seed
    global a1, a2, a1_v, a2_v, path_points, px2, py2
    global r1, r2, m1, m2, initial_a1, initial_a2
    global regions, region_colors, paint_index, current_state, quality_metrics

    seed = init_seed(use_seed)

    current_state = "RUNNING"
    path_points = []
    regions = []
    region_colors = []
    paint_index = 0
    quality_metrics = {}

    # Sample physics parameters from the seeded RNG.
    raw_r1 = random.uniform(R_MIN_BASE, R_MAX_BASE) * RES_SCALE
    raw_r2 = random.uniform(R_MIN_BASE, R_MAX_BASE) * RES_SCALE
    m1 = random.uniform(M_MIN, M_MAX)
    m2 = random.uniform(M_MIN, M_MAX)

    # Scale arms so the maximum reach fits within the safe drawing radius.
    max_reach = raw_r1 + raw_r2
    safe_radius = min(buff_w / 2, CY_POS) * 0.95
    scale_factor = safe_radius / max_reach if max_reach > safe_radius else 1.0
    r1 = raw_r1 * scale_factor
    r2 = raw_r2 * scale_factor

    initial_a1 = random.uniform(0, py5.TWO_PI)
    initial_a2 = random.uniform(0, py5.TWO_PI)
    a1, a2 = initial_a1, initial_a2
    a1_v, a2_v = 0, 0

    cx, cy = buff_w / 2, CY_POS
    x1 = cx + r1 * py5.sin(a1)
    y1 = cy + r1 * py5.cos(a1)
    px2 = x1 + r2 * py5.sin(a2)
    py2 = y1 + r2 * py5.cos(a2)
    path_points.append((px2, py2))

    # Clear buffers.
    for buf in (pg_physics, pg_art, pg_analysis):
        buf.begin_draw()
        buf.background(colors["bg"])
        buf.end_draw()

    print(f"Composition: seed={seed}, theme={theme_name}, max_points={max_points}")


# ---------------------------------------------------------------------------
# Physics.
# ---------------------------------------------------------------------------

def update_physics_step() -> tuple[float, float, float, float] | None:
    """Advance one Euler step. Return new line segment if the bob moved."""
    global a1, a2, a1_v, a2_v, px2, py2

    num1 = -G * (2 * m1 + m2) * py5.sin(a1)
    num2 = -m2 * G * py5.sin(a1 - 2 * a2)
    num3 = -2 * py5.sin(a1 - a2) * m2
    num4 = a2_v * a2_v * r2 + a1_v * a1_v * r1 * py5.cos(a1 - a2)
    den = r1 * (2 * m1 + m2 - m2 * py5.cos(2 * a1 - 2 * a2))
    a1_a = (num1 + num2 + num3 * num4) / den

    num1 = 2 * py5.sin(a1 - a2)
    num2 = a1_v * a1_v * r1 * (m1 + m2)
    num3 = G * (m1 + m2) * py5.cos(a1)
    num4 = a2_v * a2_v * r2 * m2 * py5.cos(a1 - a2)
    den = r2 * (2 * m1 + m2 - m2 * py5.cos(2 * a1 - 2 * a2))
    a2_a = (num1 * (num2 + num3 + num4)) / den

    a1_v += a1_a
    a2_v += a2_a
    a1 += a1_v
    a2 += a2_v
    a1_v *= DAMPING
    a2_v *= DAMPING

    cx, cy = buff_w / 2, CY_POS
    x1 = cx + r1 * py5.sin(a1)
    y1 = cy + r1 * py5.cos(a1)
    x2 = x1 + r2 * py5.sin(a2)
    y2 = y1 + r2 * py5.cos(a2)

    segment = None
    if abs(x2 - px2) > 0.1 or abs(y2 - py2) > 0.1:
        path_points.append((x2, y2))
        segment = (px2, py2, x2, y2)

    px2, py2 = x2, y2
    return segment


def draw_segment(segment: tuple[float, float, float, float]) -> None:
    """Draw one physics segment onto both the trace and analysis buffers."""
    x1, y1, x2, y2 = segment

    pg_physics.begin_draw()
    pg_physics.stroke(py5.color(colors["trace"]))
    pg_physics.stroke_weight(1.25 * RES_SCALE)
    pg_physics.stroke_cap(py5.ROUND)
    pg_physics.stroke_join(py5.ROUND)
    pg_physics.line(x1, y1, x2, y2)
    pg_physics.end_draw()

    pg_analysis.begin_draw()
    pg_analysis.stroke(py5.color(colors["trace"]))
    pg_analysis.stroke_weight(1.1 * RES_SCALE)
    pg_analysis.stroke_cap(py5.ROUND)
    pg_analysis.stroke_join(py5.ROUND)
    pg_analysis.line(x1, y1, x2, y2)
    pg_analysis.end_draw()


def draw_pendulum_overlay() -> None:
    cx, cy = buff_w / 2, CY_POS
    x1 = cx + r1 * py5.sin(a1)
    y1 = cy + r1 * py5.cos(a1)
    x2 = x1 + r2 * py5.sin(a2)
    y2 = y1 + r2 * py5.cos(a2)
    s = 1.0 / RES_SCALE

    py5.stroke(colors["arm"])
    py5.stroke_weight(3)
    py5.line(cx * s, cy * s, x1 * s, y1 * s)
    py5.fill(colors["arm"])
    py5.circle(x1 * s, y1 * s, 8)
    py5.line(x1 * s, y1 * s, x2 * s, y2 * s)
    py5.fill(colors["arm"])
    py5.circle(x2 * s, y2 * s, 8)


# ---------------------------------------------------------------------------
# Topology — region discovery, adjacency, coloring.
# ---------------------------------------------------------------------------

def recolor_regions() -> None:
    """
    Re-run greedy graph coloring on the existing region map with the
    current theme's palette. Repaints pg_art from scratch.

    Cheaper than a full reset because the physics simulation and flood-fill
    are skipped — only the coloring and painting passes re-run.
    """
    global region_colors

    # Rebuild adjacency from scratch. We don't store it across composition
    # state, so we have to recompute. Cheap relative to the full topology
    # solve since the regions list is already in memory.
    pg_analysis.load_pixels()
    raw_pixels = np.array(pg_analysis.pixels, dtype=np.int32)
    w, h = pg_analysis.width, pg_analysis.height
    bg_int = py5.color(colors["bg"])
    is_empty = (raw_pixels == bg_int).reshape((h, w)).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, _ = ndi_label(is_empty, structure=structure)

    # Map our regions list back to label IDs by checking each region's
    # first pixel against the labeled array.
    region_ids: list[int] = []
    for pixel_list in regions:
        if not pixel_list:
            region_ids.append(0)
            continue
        x, y = pixel_list[0]
        region_ids.append(int(labeled[y, x]))

    # Compute adjacency among the labels we care about.
    valid = set(region_ids)
    valid.discard(0)
    adjacency: dict[int, set[int]] = {i: set() for i in valid}
    for shift_axis, shift_amount in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        rolled = np.roll(labeled, shift=shift_amount, axis=shift_axis)
        mask = (labeled > 0) & (rolled > 0) & (labeled != rolled)
        if not mask.any():
            continue
        pairs = np.stack([labeled[mask], rolled[mask]], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for a, b in unique_pairs.tolist():
            if a in valid and b in valid:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Greedy coloring with the new palette.
    palette_pool = [c for c in colors["palette"] if c != colors["bg"]]
    sorted_indices = sorted(range(len(regions)), key=lambda i: len(regions[i]), reverse=True)
    assigned: dict[int, str] = {}
    new_colors: list[str] = [""] * len(regions)
    for idx in sorted_indices:
        r_id = region_ids[idx]
        if r_id == 0:
            new_colors[idx] = random.choice(palette_pool)
            continue
        used = {assigned[n] for n in adjacency[r_id] if n in assigned}
        candidates = [c for c in palette_pool if c not in used]
        chosen = random.choice(candidates if candidates else palette_pool)
        assigned[r_id] = chosen
        new_colors[idx] = chosen
    region_colors[:] = new_colors

    # Repaint pg_art from scratch in a single batched pass.
    pg_art.begin_draw()
    pg_art.background(colors["bg"])
    pg_art.end_draw()

    paint_regions_batch(regions, region_colors)
    finish_painting()

    # Also repaint pg_physics: same trajectory, but redraw it with the new
    # theme's background and trace colors.
    pg_physics.begin_draw()
    pg_physics.background(colors["bg"])
    pg_physics.stroke(py5.color(colors["trace"]))
    pg_physics.stroke_weight(1.25 * RES_SCALE)
    pg_physics.stroke_cap(py5.ROUND)
    pg_physics.stroke_join(py5.ROUND)
    pg_physics.no_fill()
    pg_physics.begin_shape()
    for x, y in path_points:
        pg_physics.vertex(x, y)
    pg_physics.end_shape()
    pg_physics.end_draw()

def solve_topology() -> None:
    """
    Discover regions, compute adjacency, assign colors.

    Optimized vs. the original pure-Python flood-fill: uses
    scipy.ndimage.label for connected-component labeling, and numpy roll
    operations for adjacency detection.
    """
    global regions, region_colors, quality_metrics

    pg_analysis.load_pixels()
    raw_pixels = np.array(pg_analysis.pixels, dtype=np.int32)
    w, h = pg_analysis.width, pg_analysis.height
    bg_int = py5.color(colors["bg"])

    # Walls are anything that isn't background. 1 = empty (region candidate),
    # 0 = wall (will be skipped by labeling).
    is_empty = (raw_pixels == bg_int).reshape((h, w)).astype(np.uint8)

    # 8-connectivity so diagonals don't artificially split regions.
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = ndi_label(is_empty, structure=structure)

    # Identify the "outside" region: any region touching the image border.
    border_labels = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(np.unique(edge).tolist())
    border_labels.discard(0)  # 0 is wall, not a region

    # Build the {label_id: list_of_pixels} dict for non-outside regions.
    found: dict[int, list[tuple[int, int]]] = {}
    for label_id in range(1, n_labels + 1):
        if label_id in border_labels:
            continue
        ys, xs = np.where(labeled == label_id)
        if len(ys) == 0:
            continue
        found[label_id] = list(zip(xs.tolist(), ys.tolist()))

    print(f"Topology: {n_labels} components, {len(found)} interior regions.")

    # Adjacency: two regions are adjacent if any wall pixel has both labels
    # in its immediate neighborhood. Vectorized via np.roll: shift the
    # labeled array in 4 directions, find pairs of distinct positive labels.
    adjacency: dict[int, set[int]] = {i: set() for i in found.keys()}
    valid = set(found.keys())

    for shift_axis, shift_amount in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        rolled = np.roll(labeled, shift=shift_amount, axis=shift_axis)
        # Wherever both labeled and rolled are positive but different,
        # record adjacency.
        mask = (labeled > 0) & (rolled > 0) & (labeled != rolled)
        if not mask.any():
            continue
        pairs = np.stack([labeled[mask], rolled[mask]], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for a, b in unique_pairs.tolist():
            if a in valid and b in valid:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Greedy coloring: largest regions first, picking colors not used by
    # any already-colored neighbor.
    sorted_ids = sorted(found.keys(), key=lambda k: len(found[k]), reverse=True)
    palette_pool = [c for c in colors["palette"] if c != colors["bg"]]
    assigned: dict[int, str] = {}
    for r_id in sorted_ids:
        used = {assigned[n] for n in adjacency[r_id] if n in assigned}
        candidates = [c for c in palette_pool if c not in used]
        assigned[r_id] = random.choice(candidates if candidates else palette_pool)

    # Build painting lists.
    regions.clear()
    region_colors.clear()
    export_order = list(found.keys())
    random.shuffle(export_order)
    for r_id in export_order:
        regions.append(found[r_id])
        region_colors.append(assigned[r_id])

    # Compute quality metrics for the metadata.
    sizes = [len(p) for p in found.values()]
    if sizes:
        quality_metrics = {
            "n_components_total": int(n_labels),
            "n_regions_interior": len(found),
            "n_regions_outside": len(border_labels),
            "region_size_mean": float(np.mean(sizes)),
            "region_size_std": float(np.std(sizes)),
            "region_size_min": int(np.min(sizes)),
            "region_size_max": int(np.max(sizes)),
        }
    else:
        quality_metrics = {
            "n_components_total": int(n_labels),
            "n_regions_interior": 0,
            "n_regions_outside": len(border_labels),
        }


def _hex_to_argb(color_hex: str) -> tuple[int, int, int, int]:
    """
    Convert '#RRGGBB' to a (A, R, G, B) tuple of uint8 values for use
    with py5's np_pixels[] buffer.

    py5.np_pixels is a 3D array shaped (height, width, 4) with channels
    ordered A, R, G, B and values in 0..255. No signed/unsigned weirdness.
    """
    h = color_hex.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (255, r, g, b)


def paint_regions_batch(
    pixel_lists: list[list[tuple[int, int]]],
    color_hexes: list[str],
) -> None:
    """
    Paint all regions at once via direct numpy pixel manipulation.

    Uses py5's np_pixels[] interface: a (H, W, 4) uint8 array with ARGB
    channel order. Much faster than per-pixel circle calls because all
    assignment happens in numpy with a single load/update pair.

    Wraps the pixel operations in begin_draw/end_draw to ensure proper
    state transitions when this is followed by vector drawing on the
    same buffer (e.g., finish_painting overlaying the trace line).
    """
    pg_art.begin_draw()
    pg_art.load_np_pixels()
    np_pixels = pg_art.np_pixels  # shape (H, W, 4), uint8, ARGB

    for pixel_list, color_hex in zip(pixel_lists, color_hexes):
        if not pixel_list:
            continue
        argb = _hex_to_argb(color_hex)
        coords = np.array(pixel_list, dtype=np.int64)
        xs = coords[:, 0]
        ys = coords[:, 1]
        # np_pixels is indexed [y, x, channel]; broadcast the 4-tuple across rows.
        np_pixels[ys, xs] = argb

    pg_art.update_np_pixels()
    pg_art.end_draw()


def paint_region_step(pixel_list: list[tuple[int, int]], color_hex: str) -> None:
    """Single-region wrapper for the streaming PAINTING state in draw()."""
    paint_regions_batch([pixel_list], [color_hex])


def finish_painting() -> None:
    """
    Overlay the crisp continuous trace and burn the physics metadata
    into pg_art so the saved PNG carries the values that produced it.
    """
    pg_art.begin_draw()

    # Trace overlay.
    pg_art.no_fill()
    pg_art.stroke(colors["trace"])
    pg_art.stroke_weight(1.25 * RES_SCALE)
    pg_art.stroke_cap(py5.ROUND)
    pg_art.stroke_join(py5.ROUND)
    pg_art.begin_shape()
    for x, y in path_points:
        pg_art.vertex(x, y)
    pg_art.end_shape()

    _draw_metadata_into_buffer(pg_art)
    pg_art.end_draw()


def _draw_metadata_into_buffer(buf) -> None:
    """
    Draw the physics metadata (m, r, a for both bobs) into a high-res
    buffer at the bottom-center, matching the layout used by the
    on-screen draw_metadata_columns(). All sizes scale with RES_SCALE
    so the burned text matches the apparent size of the live overlay.
    """
    # Match the on-screen font; scale up for the high-res buffer.
    font_size = int(11 * RES_SCALE)
    font_italic = py5.create_font("Consolas Italic", font_size)
    buf.text_font(font_italic)
    buf.fill(colors["text"])
    buf.text_align(py5.LEFT, py5.BOTTOM)

    dr1 = r1 / RES_SCALE
    dr2 = r2 / RES_SCALE

    labels1 = ["m\u2081=", "r\u2081=", "α\u2081="]
    vals1 = [f"{m1:.2f}", f"{dr1:.2f}", f"{initial_a1:.2f}"]
    labels2 = ["m\u2082=", "r\u2082=", "α\u2082="]
    vals2 = [f"{m2:.2f}", f"{dr2:.2f}", f"{initial_a2:.2f}"]

    max_label_w = []
    max_val_w = []
    for i in range(3):
        lw = max(buf.text_width(labels1[i]), buf.text_width(labels2[i]))
        vw = max(buf.text_width(vals1[i]), buf.text_width(vals2[i]))
        max_label_w.append(lw)
        max_val_w.append(vw)

    # Spacing scaled to match the on-screen version proportionally.
    inner_gap = 6 * RES_SCALE
    col_gap = 18 * RES_SCALE
    col_widths = [max_label_w[i] + inner_gap + max_val_w[i] for i in range(3)]

    # Center horizontally on the art buffer (buf is the full pg_art width).
    center_x = buf.width / 2
    base_x = center_x - (sum(col_widths) + col_gap * 2) / 2

    # Bottom rows, scaled from the on-screen offsets (22 and 36 pixels).
    y2 = buf.height - 22 * RES_SCALE
    y1 = buf.height - 36 * RES_SCALE

    x = base_x
    for i in range(3):
        label_x = x
        # Bottom row (subscript 2).
        lbl_x2 = label_x + (max_label_w[i] - buf.text_width(labels2[i]))
        buf.text(labels2[i], lbl_x2, y2)
        buf.text(vals2[i], label_x + max_label_w[i] + inner_gap, y2)
        # Top row (subscript 1).
        lbl_x1 = label_x + (max_label_w[i] - buf.text_width(labels1[i]))
        buf.text(labels1[i], lbl_x1, y1)
        buf.text(vals1[i], label_x + max_label_w[i] + inner_gap, y1)
        x += col_widths[i] + col_gap


# ---------------------------------------------------------------------------
# Metadata overlay.
# ---------------------------------------------------------------------------

def draw_metadata_columns() -> None:
    dr1 = r1 / RES_SCALE
    dr2 = r2 / RES_SCALE

    center_x = VIEW_W * 0.75
    py5.fill(colors["text"])
    py5.text_font(f_italic)
    py5.text_align(py5.LEFT, py5.BOTTOM)

    labels1 = ["m\u2081=", "r\u2081=", "α\u2081="]
    vals1 = [f"{m1:.2f}", f"{dr1:.2f}", f"{initial_a1:.2f}"]
    labels2 = ["m\u2082=", "r\u2082=", "α\u2082="]
    vals2 = [f"{m2:.2f}", f"{dr2:.2f}", f"{initial_a2:.2f}"]

    max_label_w = []
    max_val_w = []
    for i in range(3):
        lw = max(py5.text_width(labels1[i]), py5.text_width(labels2[i]))
        vw = max(py5.text_width(vals1[i]), py5.text_width(vals2[i]))
        max_label_w.append(lw)
        max_val_w.append(vw)

    inner_gap = 6
    col_gap = 18
    col_widths = [max_label_w[i] + inner_gap + max_val_w[i] for i in range(3)]
    base_x = center_x - (sum(col_widths) + col_gap * 2) / 2

    y2 = VIEW_H - 22
    y1 = VIEW_H - 36

    x = base_x
    for i in range(3):
        label_x = x
        lbl_x2 = label_x + (max_label_w[i] - py5.text_width(labels2[i]))
        py5.text(labels2[i], lbl_x2, y2)
        py5.text(vals2[i], label_x + max_label_w[i] + inner_gap, y2)
        lbl_x1 = label_x + (max_label_w[i] - py5.text_width(labels1[i]))
        py5.text(labels1[i], lbl_x1, y1)
        py5.text(vals1[i], label_x + max_label_w[i] + inner_gap, y1)
        x += col_widths[i] + col_gap


# ---------------------------------------------------------------------------
# Save.
# ---------------------------------------------------------------------------

def save_current() -> None:
    """Save both the trace view and the painted art view, with full metadata."""
    params = {
        "max_points": max_points,
        "m1": m1, "m2": m2,
        "r1": r1 / RES_SCALE, "r2": r2 / RES_SCALE,
        "a1_init": initial_a1, "a2_init": initial_a2,
        "g": G, "damping": DAMPING,
        "res_scale": RES_SCALE,
        "quality_metrics": quality_metrics,
    }
    trace_path = save_artwork(pg_physics, PROJECT_NAME, seed, theme_name, params, suffix="trace")
    art_path = save_artwork(pg_art, PROJECT_NAME, seed, theme_name, params, suffix="art")
    print(f"Saved trace: {trace_path}")
    print(f"Saved art:   {art_path}")


# ---------------------------------------------------------------------------
# py5 lifecycle.
# ---------------------------------------------------------------------------

def setup() -> None:
    global pg_physics, pg_art, pg_analysis, f_reg, f_italic, f_bold, colors
    py5.size(VIEW_W, VIEW_H, py5.P2D)
    py5.smooth(8)

    pg_physics = py5.create_graphics(buff_w, buff_h, py5.P2D)
    pg_art = py5.create_graphics(buff_w, buff_h, py5.P2D)
    pg_analysis = py5.create_graphics(buff_w, buff_h, py5.P2D)
    pg_physics.smooth(8)
    pg_art.smooth(8)

    f_reg = py5.create_font("Consolas", 11)
    f_italic = py5.create_font("Consolas Italic", 11)
    f_bold = py5.create_font("Consolas Bold", 9)

    colors = PENDULUM_THEMES[theme_name]
    reset_composition(use_seed=seed if seed != 0 else None)


def draw() -> None:
    global current_state, paint_index

    if current_state == "RUNNING":
        seg = update_physics_step()
        if seg:
            draw_segment(seg)
        if len(path_points) >= max_points:
            current_state = "ANALYZING"

    elif current_state == "ANALYZING":
        solve_topology()
        current_state = "PAINTING"

    elif current_state == "PAINTING":
        if regions:
            for _ in range(50):
                if paint_index < len(regions):
                    paint_region_step(regions[paint_index], region_colors[paint_index])
                    paint_index += 1
                else:
                    finish_painting()
                    current_state = "DONE"
                    break
        else:
            finish_painting()
            current_state = "DONE"

    py5.image(pg_physics, 0, 0, VIEW_W // 2, VIEW_H)
    py5.image(pg_art, VIEW_W // 2, 0, VIEW_W // 2, VIEW_H)
    draw_metadata_columns()

    if current_state == "RUNNING":
        draw_pendulum_overlay()

    if auto_save_and_exit and current_state == "DONE":
        save_current()
        py5.exit_sketch()


def key_pressed() -> None:
    global colors, theme_name, regions, region_colors, paint_index, current_state
    if py5.key == "r":
        reset_composition()
    elif py5.key == " ":
        # Re-color the existing regions with a new theme.
        # Only meaningful once the topology has been solved (i.e., we have
        # a `regions` list to recolor). Before that, fall back to reset.
        if current_state == "DONE" and regions:
            theme_name = random.choice(list_pendulum_themes())
            colors = PENDULUM_THEMES[theme_name]
            recolor_regions()
            print(f"Theme: {theme_name} (recolored existing regions)")
        else:
            theme_name = random.choice(list_pendulum_themes())
            colors = PENDULUM_THEMES[theme_name]
            reset_composition()
            print(f"Theme: {theme_name} (full reset)")
    elif py5.key == "s":
        save_current()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    theme_name = args.theme
    max_points = args.max_points
    auto_save_and_exit = args.save_and_exit
    seed = args.seed if args.seed is not None else 0
    py5.run_sketch()