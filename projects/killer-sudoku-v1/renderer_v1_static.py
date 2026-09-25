"""
renderer_v1_static.py — Static final-frame renderer for a killer sudoku
solver trace.

Reads a trace JSON file (produced by the solver) and renders the
complete dependency graph as a single high-resolution image:

  - One node per CellResolved event, positioned at the cell's grid
    location (grid-aligned layout).
  - Givens are distinguished from inferred resolutions.
  - Cell-to-cell cause edges drawn as thin lines with arrowheads
    at the target end. Edges fade with age (older = lighter).
  - Structural causes (cages, houses) shown as a single faint tint
    on the cage region or house, applied to every region that
    participated causally in any event. Density of overlap encodes
    structural importance.

Two node-coloring modes (toggle via COLOR_MODE at top of file, or 'm'):
  COLOR_MODE = "inference"
    Nodes colored by which rule produced them.
  COLOR_MODE = "step"
    Nodes colored by their position in the temporal sequence,
    cool (early) to warm (late).

Usage:
  Edit TRACE_PATH below to point at the trace file, then run with:
    uv run python renderer_v1_static.py
  Press 's' to save a hi-res PNG (filename includes current step).
  Press 'm' to toggle COLOR_MODE between "inference" and "step".
  Press 'o' to toggle structural overlays (cell-background tints) on/off.
  Press 'c' to toggle cage outlines and sums on/off.
  Press 'r' to re-load the trace and re-render.

  Step navigation (lets you scrub through the solve interactively):
    Left / Right     step backward / forward by 1
    Down  / Up       step backward / forward by 10
    Home             jump to step 0 (just before first event)
    End              jump to final state (full solve)
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import py5


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

TRACE_PATH = "traces/001_hard.json"

HI_RES_SIZE = 2400
PREVIEW_SIZE = 800
MARGIN_RATIO = 0.08

# Visual weights (relative to grid cell size `s`)
NODE_RADIUS_RATIO = 0.16
GIVEN_RADIUS_RATIO = 0.18
EDGE_WEIGHT_RATIO = 0.014
EDGE_AGE_FADE = 0.45
OVERLAY_OPACITY = 22
DIGIT_TEXT_RATIO = 0.30

SIZE_BY_IMPORTANCE = True
IMPORTANCE_MIN_SCALE = 1.0
IMPORTANCE_MAX_SCALE = 1.5

SHOW_OVERLAYS = True
COLOR_MODE = "step"   # "inference" or "step"

# Cage outlines and sums — show the puzzle's constraint structure.
# Particularly useful for intermediate frames where the dependency graph
# alone doesn't convey enough context. Toggle at runtime with 'c'.
SHOW_CAGES = True
CAGE_LINE_WEIGHT_RATIO = 0.012    # cage boundary stroke weight
CAGE_DASH_LEN_RATIO = 0.06        # length of each dash
CAGE_DASH_GAP_RATIO = 0.045       # gap between dashes
CAGE_SUM_TEXT_RATIO = 0.13        # cage sum font size relative to cell
CAGE_SUM_INSET_RATIO = 0.08       # offset from cell corner for sum text

# Render only events with step <= MAX_STEP. Set to None to render all
# events (the full final frame). Used for generating intermediate frames
# to validate the static design as a progression.
#
# Example: setting MAX_STEP = 30 produces the image that would result if
# the solver had stopped after step 30 — the partial dependency graph
# at that point, with overlay participation computed only from those
# 30 events.
MAX_STEP = None

DRAW_ARROWS = True
ARROW_SIZE_RATIO = 0.07
ARROW_INSET = 1.0

STEP_GRADIENT = [
    "#29617A",   # muted blue (cool — earliest)
    "#7A9A8E",   # sage
    "#E37E52",   # orange
    "#BC4A31",   # terracotta (warm — latest)
]

THEME = {
    "bg": "#F0EADC",
    "grid_minor": "#D9D2C0",
    "grid_major": "#9B9079",
    "cage_line": "#7A6B52",      # cage dashed outline — between minor and major
    "cage_sum_text": "#5A4D38",  # cage sum text — darker than cage line
    "edge": "#3A332A",
    "node_text": "#1A1612",      # single digit color used for ALL nodes
    "overlay": "#6B5D4A",
    "inf_given":                          "#26625B",
    "inf_cage_initial_pruning":           "#E37E52",
    "inf_locked_cage_elimination":        "#BC4A31",
    "inf_cage_combination_elimination":   "#C68E3C",
    "inf_hidden_single_cage":             "#A47148",
    "inf_hidden_single_house":            "#7A9A8E",
    "inf_naked_single":                   "#F2C14E",
    "inf_forty_five_basic":               "#5A7A8E",
    "inf_forty_five_innie":               "#365C6E",
    "inf_forty_five_outie":               "#4C7186",
}


# ----------------------------------------------------------------------------
# Trace loading & preprocessing
# ----------------------------------------------------------------------------

trace_data = None
raw_events = []   # full unfiltered event list, set once by load_trace
events = []       # current filtered view, set by apply_filter
puzzle = None
cage_of = {}
cage_cells = {}
cage_sum = {}
givens_set = set()
cages_used_as_cause = set()
houses_used_as_cause = set()
event_by_step = {}
out_degree = {}
full_solve_length = 0   # total events in the underlying solve; used for color stability
pg = None


def load_trace():
    """Read the trace from disk and cache the raw events plus all
    static puzzle metadata. Called once at startup. After this, changing
    MAX_STEP only requires apply_filter() — no disk re-read.
    """
    global trace_data, puzzle, full_solve_length
    global raw_events, _puzzle_loaded

    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    puzzle = trace_data["puzzle"]
    raw_events = [e for e in trace_data["events"] if e["type"] == "cell_resolved"]
    full_solve_length = max((e["step"] for e in raw_events), default=0) + 1

    cage_of.clear()
    cage_cells.clear()
    cage_sum.clear()
    for cage in puzzle["cages"]:
        cells = [tuple(cell) for cell in cage["cells"]]
        cage_cells[cage["id"]] = cells
        cage_sum[cage["id"]] = cage["sum"]
        for cell in cells:
            cage_of[cell] = cage["id"]

    givens_set.clear()
    for g in puzzle.get("givens", []):
        givens_set.add(tuple(g["cell"]))

    print(f"Loaded trace: {TRACE_PATH}")
    print(f"  Puzzle: {puzzle.get('puzzle_id')}")
    print(f"  Total events in solve: {full_solve_length}")
    print(f"  Givens: {len(givens_set)}")

    apply_filter()


def apply_filter():
    """Compute the filtered event list and all derived structures based
    on the current MAX_STEP. Called on every step change.
    """
    global events, event_by_step

    if MAX_STEP is not None:
        events = [e for e in raw_events if e["step"] <= MAX_STEP]
    else:
        events = list(raw_events)

    event_by_step = {e["step"]: e for e in events}

    cages_used_as_cause.clear()
    houses_used_as_cause.clear()
    out_degree.clear()
    for ev in events:
        out_degree.setdefault(ev["step"], 0)
        for cause in ev.get("causes", ()):
            t = cause.get("type")
            if t == "cage":
                cages_used_as_cause.add(cause["cage_id"])
            elif t == "house":
                houses_used_as_cause.add(
                    (cause["house_type"], cause["index"])
                )
            elif t == "cell":
                src = cause["step"]
                out_degree[src] = out_degree.get(src, 0) + 1

    if MAX_STEP is not None:
        print(f"  step={MAX_STEP} → showing {len(events)}/{full_solve_length} events")
    else:
        print(f"  step=FINAL → showing all {len(events)} events")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _lerp_gradient(t, stops):
    if not stops:
        return py5.color(0)
    if len(stops) == 1:
        return py5.color(stops[0])
    t = max(0.0, min(1.0, t))
    n_segs = len(stops) - 1
    pos = t * n_segs
    i = int(pos)
    if i >= n_segs:
        return py5.color(stops[-1])
    frac = pos - i
    return py5.lerp_color(py5.color(stops[i]), py5.color(stops[i + 1]), frac)


def _node_fill(ev, n_events):
    if COLOR_MODE == "step":
        # Normalize against full solve length so gradient is stable across
        # truncated views.
        denom = max(1, full_solve_length - 1)
        t = ev["step"] / denom
        return _lerp_gradient(t, STEP_GRADIENT)
    inf = ev["inference_type"]
    return py5.color(THEME.get(f"inf_{inf}", "#888888"))


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def setup():
    global pg
    py5.size(PREVIEW_SIZE, PREVIEW_SIZE, py5.P2D)
    pg = py5.create_graphics(HI_RES_SIZE, HI_RES_SIZE, py5.P2D)
    load_trace()
    render()


def render():
    pg.begin_draw()
    pg.background(THEME["bg"])

    margin = HI_RES_SIZE * MARGIN_RATIO
    grid_size = HI_RES_SIZE - 2 * margin
    s = grid_size / 9.0

    pg.push_matrix()
    pg.translate(margin, margin)

    _draw_cage_overlays(s)
    _draw_house_overlays(s)
    _draw_grid(s)
    _draw_cages(s)
    _draw_edges(s)
    _draw_nodes(s)

    pg.pop_matrix()
    pg.end_draw()
    print(f"Render complete ({COLOR_MODE} mode, overlays {'on' if SHOW_OVERLAYS else 'off'}).")


def _draw_grid(s):
    pg.no_fill()
    pg.stroke(THEME["grid_minor"])
    pg.stroke_weight(s * 0.012)
    for i in range(10):
        x = i * s
        pg.line(x, 0, x, 9 * s)
        pg.line(0, x, 9 * s, x)

    pg.stroke(THEME["grid_major"])
    pg.stroke_weight(s * 0.030)
    for i in [0, 3, 6, 9]:
        x = i * s
        pg.line(x, 0, x, 9 * s)
        pg.line(0, x, 9 * s, x)


def _draw_cages(s):
    """Draw dashed cage boundaries between cells in different cages,
    plus the cage's sum in the top-left corner of each cage.

    Skips edges that coincide with 3x3 box boundaries — those are
    already drawn solid by the grid major lines.
    """
    if not SHOW_CAGES:
        return

    import math

    line_col = py5.color(THEME["cage_line"])
    text_col = py5.color(THEME["cage_sum_text"])
    weight = s * CAGE_LINE_WEIGHT_RATIO
    dash_len = s * CAGE_DASH_LEN_RATIO
    gap_len = s * CAGE_DASH_GAP_RATIO

    pg.stroke(line_col)
    pg.stroke_weight(weight)
    pg.no_fill()
    pg.stroke_cap(py5.SQUARE)

    # Iterate every cell, check right neighbor and bottom neighbor.
    # If they belong to different cages AND the edge isn't a 3x3 box
    # boundary, draw a dashed line along the boundary.
    for r in range(9):
        for c in range(9):
            x, y = c * s, r * s
            my_cage = cage_of.get((r, c))
            if my_cage is None:
                continue

            # Right edge — between (r, c) and (r, c+1)
            if c < 8:
                neighbor_cage = cage_of.get((r, c + 1))
                if neighbor_cage != my_cage:
                    if (c + 1) % 3 != 0:   # not a 3x3 box boundary
                        _dashed_line(x + s, y, x + s, y + s, dash_len, gap_len)

            # Bottom edge — between (r, c) and (r+1, c)
            if r < 8:
                neighbor_cage = cage_of.get((r + 1, c))
                if neighbor_cage != my_cage:
                    if (r + 1) % 3 != 0:
                        _dashed_line(x, y + s, x + s, y + s, dash_len, gap_len)

    # Cage sums — drawn in the top-left cell of each cage.
    text_size = s * CAGE_SUM_TEXT_RATIO
    inset = s * CAGE_SUM_INSET_RATIO
    pg.no_stroke()
    pg.fill(text_col)
    pg.text_size(text_size)
    pg.text_align(py5.LEFT, py5.TOP)

    # Find top-left cell of each cage: smallest row, then smallest col.
    for cage_id, cells in cage_cells.items():
        sorted_cells = sorted(cells)   # sorts by row then col
        tl_r, tl_c = sorted_cells[0]
        x = tl_c * s + inset
        y = tl_r * s + inset
        pg.text(str(cage_sum[cage_id]), x, y)


def _dashed_line(x1, y1, x2, y2, dash_len, gap_len):
    """Draw a dashed line from (x1, y1) to (x2, y2)."""
    import math
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash_len, length)
        sx = x1 + ux * pos
        sy = y1 + uy * pos
        ex = x1 + ux * end
        ey = y1 + uy * end
        pg.line(sx, sy, ex, ey)
        pos = end + gap_len


def _draw_cage_overlays(s):
    if not SHOW_OVERLAYS:
        return
    pg.no_stroke()
    fill_col = py5.color(THEME["overlay"])
    for cage_id in cages_used_as_cause:
        cells = cage_cells[cage_id]
        pg.fill(fill_col, OVERLAY_OPACITY)
        for r, c in cells:
            pg.rect(c * s, r * s, s, s)


def _draw_house_overlays(s):
    if not SHOW_OVERLAYS:
        return
    fill_col = py5.color(THEME["overlay"])
    pg.no_stroke()
    for (house_type, index) in houses_used_as_cause:
        pg.fill(fill_col, OVERLAY_OPACITY)
        if house_type == "row":
            pg.rect(0, index * s, 9 * s, s)
        elif house_type == "col":
            pg.rect(index * s, 0, s, 9 * s)
        else:
            br = (index // 3) * 3
            bc = (index % 3) * 3
            pg.rect(bc * s, br * s, 3 * s, 3 * s)


def _draw_edges(s):
    import math

    pg.stroke_cap(py5.ROUND)
    base_col = py5.color(THEME["edge"])
    weight = s * EDGE_WEIGHT_RATIO
    arrow_size = s * ARROW_SIZE_RATIO
    max_out = max(out_degree.values()) if out_degree else 1
    edge_denom = max(1, full_solve_length - 1)

    for ev in events:
        step = ev["step"]
        tr, tc = ev["cell"]
        tx, ty = (tc + 0.5) * s, (tr + 0.5) * s

        is_given = (tr, tc) in givens_set
        base = s * (GIVEN_RADIUS_RATIO if is_given else NODE_RADIUS_RATIO)
        if SIZE_BY_IMPORTANCE and max_out > 0:
            od = out_degree.get(step, 0)
            t = math.sqrt(od / max_out)
            target_scale = IMPORTANCE_MIN_SCALE + t * (IMPORTANCE_MAX_SCALE - IMPORTANCE_MIN_SCALE)
        else:
            target_scale = 1.0
        target_radius = base * target_scale
        inset = target_radius * ARROW_INSET

        age_norm = step / edge_denom
        alpha = int(255 * (EDGE_AGE_FADE + (1 - EDGE_AGE_FADE) * age_norm))

        for cause in ev.get("causes", ()):
            if cause.get("type") != "cell":
                continue
            sr, sc = cause["cell"]
            sx, sy = (sc + 0.5) * s, (sr + 0.5) * s

            dx, dy = tx - sx, ty - sy
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            ux, uy = dx / length, dy / length

            end_x = tx - ux * inset
            end_y = ty - uy * inset

            pg.stroke(base_col, alpha)
            pg.stroke_weight(weight)
            pg.no_fill()
            pg.line(sx, sy, end_x, end_y)

            if DRAW_ARROWS:
                px, py_ = -uy, ux
                tip_x, tip_y = end_x, end_y
                base_cx = end_x - ux * arrow_size
                base_cy = end_y - uy * arrow_size
                left_x  = base_cx + px * arrow_size * 0.45
                left_y  = base_cy + py_ * arrow_size * 0.45
                right_x = base_cx - px * arrow_size * 0.45
                right_y = base_cy - py_ * arrow_size * 0.45
                pg.no_stroke()
                pg.fill(base_col, alpha)
                pg.triangle(tip_x, tip_y, left_x, left_y, right_x, right_y)


def _draw_nodes(s):
    import math

    given_base = s * GIVEN_RADIUS_RATIO
    node_base = s * NODE_RADIUS_RATIO
    text_size_base = s * DIGIT_TEXT_RATIO
    n_events = len(events)
    max_out = max(out_degree.values()) if out_degree else 1

    pg.text_align(py5.CENTER, py5.CENTER)

    for ev in events:
        r, c = ev["cell"]
        x, y = (c + 0.5) * s, (r + 0.5) * s
        digit = ev["digit"]
        is_given = (r, c) in givens_set

        if SIZE_BY_IMPORTANCE and max_out > 0:
            od = out_degree.get(ev["step"], 0)
            t = math.sqrt(od / max_out)
            scale = IMPORTANCE_MIN_SCALE + t * (IMPORTANCE_MAX_SCALE - IMPORTANCE_MIN_SCALE)
        else:
            scale = 1.0

        base = given_base if is_given else node_base
        radius = base * scale

        pg.text_size(text_size_base * scale)

        fill_color = _node_fill(ev, n_events)

        pg.no_stroke()
        pg.fill(fill_color)
        pg.circle(x, y, 2 * radius)

        pg.no_fill()
        pg.stroke(THEME["node_text"], 110)
        pg.stroke_weight(s * (0.010 if is_given else 0.005))
        pg.circle(x, y, 2 * radius)

        # Single digit color — no luminance-based switching.
        pg.no_stroke()
        pg.fill(THEME["node_text"])
        pg.text(str(digit), x, y - s * 0.01)


# ----------------------------------------------------------------------------
# Main loop and interaction
# ----------------------------------------------------------------------------

def draw():
    py5.image(pg, 0, 0, py5.width, py5.height)


def _set_max_step(new_value):
    """Update MAX_STEP, clamp to valid range, apply filter, re-render."""
    global MAX_STEP
    # Clamp. A value past the end becomes the FINAL state (None).
    if new_value is None or new_value >= full_solve_length - 1:
        MAX_STEP = None
    elif new_value < 0:
        MAX_STEP = 0
    else:
        MAX_STEP = new_value
    apply_filter()
    render()


def key_pressed():
    global COLOR_MODE, SHOW_OVERLAYS, SHOW_CAGES, MAX_STEP

    # Save / mode toggles
    if py5.key == "s":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = puzzle.get("puzzle_id", "trace") if puzzle else "trace"
        step_tag = "final" if MAX_STEP is None else f"step{MAX_STEP:03d}"
        filename = f"depgraph_{pid}_{COLOR_MODE}_{step_tag}_{timestamp}.png"
        pg.save(filename)
        print(f"Saved: {filename}")
    elif py5.key == "r":
        load_trace()
        render()
    elif py5.key == "m":
        COLOR_MODE = "inference" if COLOR_MODE == "step" else "step"
        print(f"Switched COLOR_MODE → {COLOR_MODE}")
        render()
    elif py5.key == "o":
        SHOW_OVERLAYS = not SHOW_OVERLAYS
        print(f"Structural overlays: {'on' if SHOW_OVERLAYS else 'off'}")
        render()
    elif py5.key == "c":
        SHOW_CAGES = not SHOW_CAGES
        print(f"Cage outlines: {'on' if SHOW_CAGES else 'off'}")
        render()

    # Step navigation. py5 reports arrow keys via key_code, not key.
    elif py5.key_code == py5.LEFT or py5.key_code == py5.RIGHT \
         or py5.key_code == py5.UP or py5.key_code == py5.DOWN \
         or py5.key_code == py5.HOME or py5.key_code == py5.END:

        # Determine current position. None = "final" = past the last index.
        current = MAX_STEP if MAX_STEP is not None else full_solve_length - 1

        # Step size: 1 by default, 10 with shift held.
        try:
            shifted = py5.is_key_pressed() and py5.key_code == py5.SHIFT
        except Exception:
            shifted = False
        # py5 doesn't expose modifier state portably; instead we use Up/Down
        # for big steps (+/- 10) and Left/Right for single steps. Cleaner
        # and works the same across platforms.

        if py5.key_code == py5.LEFT:
            _set_max_step(current - 1)
        elif py5.key_code == py5.RIGHT:
            _set_max_step(current + 1)
        elif py5.key_code == py5.DOWN:
            _set_max_step(current - 10)
        elif py5.key_code == py5.UP:
            _set_max_step(current + 10)
        elif py5.key_code == py5.HOME:
            _set_max_step(0)
        elif py5.key_code == py5.END:
            _set_max_step(None)


py5.run_sketch()
