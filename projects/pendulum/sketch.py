"""
Double pendulum — chaotic trajectory traced and painted as topology.

A double pendulum in SI units (meters, kilograms, seconds, g = 9.81) is
integrated with RK4 (see pendulum_physics.py), optionally with viscous
friction at both joints. Its lower bob traces a path for a fixed span of
simulated time. The trace is then treated as a wall structure: the
enclosed regions are labeled and colored with greedy graph coloring (no
two adjacent regions share a color, when possible).

The art view keeps only closed shapes. Trace segments with the same area
on both sides (loose strands, such as the fall from the release point and
the end of the run) are dropped and their gap is filled with the
surrounding color. The trace view on the left keeps the full path.

Each composition is determined by:
  - Six random initial conditions (l1, l2, m1, m2, a1_init, a2_init),
    sampled from fixed ranges using the seeded RNG. The pendulum is
    released from rest.
  - The simulated duration (--duration) and joint friction (--damping).
  - The chosen theme (palette of region colors).

A note on reproducibility: the double pendulum is famously chaotic, but
the integrator is deterministic. Given the same seed, the same code, and
the same Python version, the trajectory reproduces bit-for-bit. Typing
the rounded initial angles back in from a sidecar would not: chaos
amplifies the lost precision.

Interactive keys:
    r       New random composition (new seed).
    space   New random theme (recolors once finished, otherwise resets).
    s       Save the current piece (trace and art views).

Usage:
    uv run python projects/pendulum/sketch.py
    uv run python projects/pendulum/sketch.py --seed 4823 --theme JAPAN
    uv run python projects/pendulum/sketch.py --seed 4823 --duration 3 --damping 0.5
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import py5
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label

from coalescence.io import save_artwork
from coalescence.palettes import PENDULUM_THEMES, list_pendulum_themes
from coalescence.seeds import init_seed
from pendulum_physics import G, Pendulum, bob_positions, energy, rk4_step


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

VIEW_W = 1200
VIEW_H = 600
RES_SCALE = 3.0  # 3.0 -> 3600x1800 high-res buffers

# Physics (SI units)
DT = 0.002                    # s per RK4 step; one trace point per step
DEFAULT_DURATION = 5.0        # s of simulated time per piece
DEFAULT_DAMPING = 0.0         # joint friction c, N*m*s/rad (0 = undamped)
L_MIN, L_MAX = 0.5, 2.0       # arm lengths, m
M_MIN, M_MAX = 1.0, 5.0       # bob masses, kg

# Physics steps per frame. Only changes playback speed; the trajectory (and
# so the artwork for a given seed) is the same for any value. 8 steps of
# 2 ms is ~16 ms of simulated time per frame: roughly real time at 60 fps.
STEPS_PER_FRAME = 8

# How far (high-res pixels) to look on each side of a trace segment when
# deciding whether it bounds two different areas. Must exceed half the
# analysis wall width (1.1 * RES_SCALE / 2).
STRAND_SAMPLE_PX = 4.0

DEFAULT_THEME = "JAPAN"
PROJECT_NAME = "pendulum"


# ---------------------------------------------------------------------------
# Module state.
# ---------------------------------------------------------------------------

# CLI-driven config
seed: int = 0
theme_name: str = DEFAULT_THEME
duration: float = DEFAULT_DURATION
damping: float = DEFAULT_DAMPING
auto_save_and_exit: bool = False

# Theme
colors: dict = {}

# Physics state
pend: Pendulum | None = None
state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
initial_a1: float = 0
initial_a2: float = 0
pixels_per_meter: float = 0
steps_done: int = 0
n_steps: int = 0
energy_initial: float = 0

# Trace, in high-res pixels
path_points: list[tuple[float, float]] = []

# Topology results
labeled_map: np.ndarray | None = None   # region label per pixel, 0 = wall
filled_map: np.ndarray | None = None    # same, with walls given the nearest label
interior_labels: list[int] = []          # labels of enclosed regions, largest first
adjacency: dict[int, set[int]] = {}
region_colors: dict[int, str] = {}       # label -> hex color
boundary_segments: np.ndarray | None = None  # bool per trace segment
segment_cuts: dict[int, tuple] = {}          # segment index -> (start, end) cut at a crossing
quality_metrics: dict = {}

# Lifecycle state machine
current_state: str = "RUNNING"  # RUNNING -> ANALYZING -> DONE

# Buffers
pg_physics = None
pg_art = None
pg_analysis = None
analysis_bg: str = ""  # background pg_analysis was cleared with (walls = anything else)
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
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Seconds of simulated motion before the topology pass. "
                             f"Default: {DEFAULT_DURATION}.")
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING,
                        help=f"Joint friction coefficient c in N*m*s/rad; 0 = undamped. "
                             f"Default: {DEFAULT_DAMPING}.")
    parser.add_argument("--save-and-exit", action="store_true",
                        help="With --seed, save both views and exit when DONE.")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.damping < 0:
        parser.error("--damping cannot be negative")
    return args


# ---------------------------------------------------------------------------
# Composition reset.
# ---------------------------------------------------------------------------

def reset_composition(use_seed: int | None = None) -> None:
    """Seed RNG, sample new physical parameters, clear all state."""
    global seed, pend, state, initial_a1, initial_a2, pixels_per_meter
    global steps_done, n_steps, energy_initial, path_points
    global labeled_map, filled_map, interior_labels, adjacency, region_colors
    global boundary_segments, segment_cuts, current_state, quality_metrics, analysis_bg

    seed = init_seed(use_seed)

    current_state = "RUNNING"
    path_points = []
    labeled_map = filled_map = boundary_segments = None
    segment_cuts = {}
    interior_labels = []
    adjacency = {}
    region_colors = {}
    quality_metrics = {}

    # Sample physics parameters from the seeded RNG.
    l1 = random.uniform(L_MIN, L_MAX)
    l2 = random.uniform(L_MIN, L_MAX)
    m1 = random.uniform(M_MIN, M_MAX)
    m2 = random.uniform(M_MIN, M_MAX)
    initial_a1 = random.uniform(0, math.tau)
    initial_a2 = random.uniform(0, math.tau)

    pend = Pendulum(m1=m1, m2=m2, l1=l1, l2=l2, damping=damping)
    state = (initial_a1, initial_a2, 0.0, 0.0)
    energy_initial = energy(pend, state)
    steps_done = 0
    n_steps = round(duration / DT)

    # Scale meters to pixels so the maximum reach fills the safe drawing radius.
    safe_radius = min(buff_w / 2, CY_POS) * 0.95
    pixels_per_meter = safe_radius / (l1 + l2)
    path_points.append(_bob_pixels()[2:])

    # Clear buffers.
    for buf in (pg_physics, pg_art, pg_analysis):
        buf.begin_draw()
        buf.background(colors["bg"])
        buf.end_draw()
    analysis_bg = colors["bg"]

    print(f"Composition: seed={seed}, theme={theme_name}, "
          f"duration={duration}s, damping={damping}")


# ---------------------------------------------------------------------------
# Physics.
# ---------------------------------------------------------------------------

def _bob_pixels() -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) of both bobs in high-res buffer pixels."""
    cx, cy = buff_w / 2, CY_POS
    x1, y1, x2, y2 = bob_positions(pend, state)
    s = pixels_per_meter
    return cx + x1 * s, cy + y1 * s, cx + x2 * s, cy + y2 * s


def update_physics_step() -> tuple[float, float, float, float]:
    """Advance one RK4 step and return the new trace segment."""
    global state, steps_done
    state = rk4_step(pend, state, DT)
    steps_done += 1
    x2, y2 = _bob_pixels()[2:]
    px2, py2 = path_points[-1]
    path_points.append((x2, y2))
    return px2, py2, x2, y2


def draw_segments(segments: list[tuple[float, float, float, float]]) -> None:
    """
    Draw this frame's physics segments onto both the trace and analysis
    buffers. One begin_draw/end_draw per buffer per frame: opening a
    high-res GPU buffer is the expensive part, not the lines themselves.
    """
    if not segments:
        return

    for buf, weight in ((pg_physics, 1.25), (pg_analysis, 1.1)):
        buf.begin_draw()
        buf.stroke(py5.color(colors["trace"]))
        buf.stroke_weight(weight * RES_SCALE)
        buf.stroke_cap(py5.ROUND)
        buf.stroke_join(py5.ROUND)
        for x1, y1, x2, y2 in segments:
            buf.line(x1, y1, x2, y2)
        buf.end_draw()


def draw_pendulum_overlay() -> None:
    cx, cy = buff_w / 2, CY_POS
    x1, y1, x2, y2 = _bob_pixels()
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
# Topology — region discovery, adjacency, coloring, loose strands.
# ---------------------------------------------------------------------------

def _label_empty_regions() -> tuple[np.ndarray, int]:
    """
    Read pg_analysis back and label its enclosed empty regions.

    Uses np_pixels: one bulk copy into a (H, W, 4) ARGB uint8 array.
    (Reading `pixels` and wrapping it in np.array crosses the Java bridge
    once per pixel, which took ~33 s at 3600x1800.)

    Compares against `analysis_bg`, the background the buffer was cleared
    with, so labeling still works after a theme change.
    """
    pg_analysis.load_np_pixels()
    bg_argb = np.array(_hex_to_argb(analysis_bg), dtype=np.uint8)

    # Walls are anything that isn't background. 1 = empty (region candidate),
    # 0 = wall (will be skipped by labeling).
    is_empty = np.all(pg_analysis.np_pixels == bg_argb, axis=2).astype(np.uint8)

    # 8-connectivity so diagonals don't artificially split regions.
    structure = np.ones((3, 3), dtype=np.uint8)
    return ndi_label(is_empty, structure=structure)


def _region_adjacency(labeled: np.ndarray, valid: set[int]) -> dict[int, set[int]]:
    """
    Two regions are adjacent if any pixel of one touches a pixel of the other
    in the 4-neighborhood. Vectorized via np.roll: shift the labeled array in
    4 directions, find pairs of distinct positive labels.
    """
    result: dict[int, set[int]] = {i: set() for i in valid}
    for shift_axis, shift_amount in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        rolled = np.roll(labeled, shift=shift_amount, axis=shift_axis)
        mask = (labeled > 0) & (rolled > 0) & (labeled != rolled)
        if not mask.any():
            continue
        pairs = np.stack([labeled[mask], rolled[mask]], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for a, b in unique_pairs.tolist():
            if a in valid and b in valid:
                result[a].add(b)
                result[b].add(a)
    return result


def _fill_walls(labeled: np.ndarray) -> np.ndarray:
    """
    Give every wall pixel (label 0) the label of its nearest region pixel.

    Painting this map leaves no background-colored gaps where loose strands
    were removed; real boundaries are covered by the trace overlay anyway.
    """
    indices = distance_transform_edt(
        labeled == 0, return_distances=False, return_indices=True
    )
    return labeled[indices[0], indices[1]]


def _boundary_segment_mask(filled: np.ndarray) -> np.ndarray:
    """
    For each trace segment, True if it separates two different areas.

    Samples the filled label map STRAND_SAMPLE_PX to the left and right of
    each segment's midpoint. A loose strand (the fall from the release
    point, the end of the run, a dead-end loop) has the same area on both
    sides. Segments with no usable direction are kept.
    """
    pts = np.asarray(path_points, dtype=np.float64)
    n = len(pts) - 1
    if n < 1:
        return np.zeros(0, dtype=bool)

    mid = (pts[:-1] + pts[1:]) / 2
    # Tangent over the neighboring points: steadier than a single tiny segment.
    i = np.arange(n)
    tangent = pts[np.minimum(i + 2, n)] - pts[np.maximum(i - 1, 0)]
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    usable = length > 1e-9
    normal = np.zeros_like(tangent)
    normal[usable, 0] = -tangent[usable, 1] / length[usable]
    normal[usable, 1] = tangent[usable, 0] / length[usable]

    h, w = filled.shape

    def sample(points: np.ndarray) -> np.ndarray:
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
        return filled[ys, xs]

    left = sample(mid + normal * STRAND_SAMPLE_PX)
    right = sample(mid - normal * STRAND_SAMPLE_PX)
    return (left != right) | ~usable


def _crossing_params(p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Positions t in [0, 1] along segment p0->p1 where it crosses any q segment."""
    r = p1 - p0
    s = q1 - q0
    qp = q0 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return t[hit]


def _trim_free_ends(keep: np.ndarray) -> tuple[np.ndarray, dict[int, tuple]]:
    """
    Cut the trace's two free ends exactly at its first and last self-crossing.

    The side-sampling test can't judge the few pixels next to a crossing, so
    a short stub could survive where a tail meets the shape. Before the first
    crossing and after the last one, the path can't enclose anything.

    Returns the updated keep mask and {segment index: (start, end)} overrides
    for the two segments that get cut at the crossing point.
    """
    pts = np.asarray(path_points, dtype=np.float64)
    p0, p1 = pts[:-1], pts[1:]
    n = len(p0)

    first = None
    for i in range(n):
        ts = _crossing_params(p0[i], p1[i], p0[i + 2:], p1[i + 2:])
        if len(ts):
            first = (i, p0[i] + ts.min() * (p1[i] - p0[i]))
            break
    if first is None:
        return keep, {}  # never crosses itself: nothing to trim against

    last = None
    for i in range(n - 1, -1, -1):
        ts = _crossing_params(p0[i], p1[i], p0[:max(i - 1, 0)], p1[:max(i - 1, 0)])
        if len(ts):
            last = (i, p0[i] + ts.max() * (p1[i] - p0[i]))
            break

    keep = keep.copy()
    keep[:first[0]] = False
    keep[last[0] + 1:] = False
    keep[first[0]] = keep[last[0]] = True

    cuts = {first[0]: (tuple(first[1]), tuple(p1[first[0]]))}
    start = cuts.get(last[0], (tuple(p0[last[0]]),))[0]
    cuts[last[0]] = (start, tuple(last[1]))
    return keep, cuts


def _greedy_colors() -> dict[int, str]:
    """Largest regions first, each picking a color unused by colored neighbors."""
    palette_pool = [c for c in colors["palette"] if c != colors["bg"]]
    assigned: dict[int, str] = {}
    for r_id in interior_labels:
        used = {assigned[n] for n in adjacency[r_id] if n in assigned}
        candidates = [c for c in palette_pool if c not in used]
        assigned[r_id] = random.choice(candidates if candidates else palette_pool)
    return assigned


def solve_topology() -> None:
    """Discover regions, adjacency, colors, and which trace segments to keep."""
    global labeled_map, filled_map, interior_labels, adjacency, region_colors
    global boundary_segments, segment_cuts, quality_metrics

    labeled, n_labels = _label_empty_regions()

    # Identify the "outside" region: any region touching the image border.
    border_labels = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(np.unique(edge).tolist())
    border_labels.discard(0)  # 0 is wall, not a region

    sizes = np.bincount(labeled.ravel(), minlength=n_labels + 1)
    interior = [i for i in range(1, n_labels + 1) if i not in border_labels and sizes[i] > 0]
    interior_labels = sorted(interior, key=lambda i: sizes[i], reverse=True)
    print(f"Topology: {n_labels} components, {len(interior_labels)} interior regions.")

    labeled_map = labeled
    adjacency = _region_adjacency(labeled, set(interior_labels))
    region_colors = _greedy_colors()
    filled_map = _fill_walls(labeled)
    boundary_segments, segment_cuts = _trim_free_ends(_boundary_segment_mask(filled_map))

    # Quality metrics for the metadata.
    interior_sizes = sizes[interior_labels] if interior_labels else np.zeros(0)
    quality_metrics = {
        "n_components_total": int(n_labels),
        "n_regions_interior": len(interior_labels),
        "n_regions_outside": len(border_labels),
        "trace_segments_total": int(len(boundary_segments)),
        "trace_segments_dropped": int((~boundary_segments).sum()),
    }
    if len(interior_sizes):
        quality_metrics.update({
            "region_size_mean": float(np.mean(interior_sizes)),
            "region_size_std": float(np.std(interior_sizes)),
            "region_size_min": int(np.min(interior_sizes)),
            "region_size_max": int(np.max(interior_sizes)),
        })


def recolor_regions() -> None:
    """
    Re-run greedy graph coloring on the existing region map with the
    current theme's palette and repaint both views. Physics and topology
    are kept from the last solve.
    """
    global region_colors
    region_colors = _greedy_colors()

    paint_regions()
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


def paint_regions() -> None:
    """
    Paint every pixel of pg_art from the filled label map in one pass.

    A lookup table maps each label to its ARGB color (outside and walls
    next to the outside stay background). One load/update pair: each pair
    copies the full high-res buffer from and back to the GPU.
    """
    lut = np.empty((int(filled_map.max()) + 1, 4), dtype=np.uint8)
    lut[:] = _hex_to_argb(colors["bg"])
    for r_id, color_hex in region_colors.items():
        lut[r_id] = _hex_to_argb(color_hex)

    pg_art.begin_draw()
    pg_art.load_np_pixels()
    pg_art.np_pixels[:] = lut[filled_map]
    pg_art.update_np_pixels()
    pg_art.end_draw()


def finish_painting() -> None:
    """
    Overlay the trace segments that bound regions, and burn the physics
    metadata into pg_art so the saved PNG carries the values that produced it.
    """
    pg_art.begin_draw()

    # Trace overlay: one polyline per run of consecutive boundary segments.
    pg_art.no_fill()
    pg_art.stroke(colors["trace"])
    pg_art.stroke_weight(1.25 * RES_SCALE)
    pg_art.stroke_cap(py5.ROUND)
    pg_art.stroke_join(py5.ROUND)
    in_run = False
    for i, keep in enumerate(boundary_segments):
        start, end = segment_cuts.get(i, (path_points[i], path_points[i + 1]))
        if keep and not in_run:
            pg_art.begin_shape()
            pg_art.vertex(*start)
            in_run = True
        if keep:
            pg_art.vertex(*end)
        elif in_run:
            pg_art.end_shape()
            in_run = False
    if in_run:
        pg_art.end_shape()

    _draw_metadata_into_buffer(pg_art)
    pg_art.end_draw()


def _metadata_columns() -> tuple[list[str], list[str], list[str], list[str]]:
    """Labels and values for the two metadata rows (bob 1 on top, bob 2 below)."""
    labels1 = ["m₁=", "r₁=", "α₁="]
    vals1 = [f"{pend.m1:.2f}", f"{pend.l1:.2f}", f"{initial_a1:.2f}"]
    labels2 = ["m₂=", "r₂=", "α₂="]
    vals2 = [f"{pend.m2:.2f}", f"{pend.l2:.2f}", f"{initial_a2:.2f}"]
    return labels1, vals1, labels2, vals2


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

    labels1, vals1, labels2, vals2 = _metadata_columns()

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
    center_x = VIEW_W * 0.75
    py5.fill(colors["text"])
    py5.text_font(f_italic)
    py5.text_align(py5.LEFT, py5.BOTTOM)

    labels1, vals1, labels2, vals2 = _metadata_columns()

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
    energy_final = energy(pend, state)
    params = {
        "duration_s": duration,
        "dt_s": DT,
        "damping": damping,
        "m1_kg": pend.m1, "m2_kg": pend.m2,
        "l1_m": pend.l1, "l2_m": pend.l2,
        "a1_init": initial_a1, "a2_init": initial_a2,
        "g": G,
        "integrator": "rk4",
        "energy_initial_J": energy_initial,
        "energy_final_J": energy_final,
        "pixels_per_meter": pixels_per_meter,
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
    global current_state

    if current_state == "RUNNING":
        # Several physics steps per frame, drawn in one pass. Stops at exactly
        # n_steps, so the trajectory doesn't depend on STEPS_PER_FRAME.
        segments = []
        for _ in range(STEPS_PER_FRAME):
            segments.append(update_physics_step())
            if steps_done >= n_steps:
                current_state = "ANALYZING"
                break
        draw_segments(segments)

    elif current_state == "ANALYZING":
        solve_topology()
        paint_regions()
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
    global colors, theme_name
    if py5.key == "r":
        reset_composition()
    elif py5.key == " ":
        # Re-color the existing regions with a new theme.
        # Only meaningful once the topology has been solved. Before that,
        # fall back to reset.
        theme_name = random.choice(list_pendulum_themes())
        colors = PENDULUM_THEMES[theme_name]
        if current_state == "DONE" and interior_labels:
            recolor_regions()
            print(f"Theme: {theme_name} (recolored existing regions)")
        else:
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
    duration = args.duration
    damping = args.damping
    auto_save_and_exit = args.save_and_exit
    seed = args.seed if args.seed is not None else 0
    py5.run_sketch()
