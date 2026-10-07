"""
Killer sudoku — static renderer for a solver trace.

Reads a trace JSON file (produced by solve.py) and renders the
complete dependency graph as a single high-resolution image:

  - One node per CellResolved event, positioned at the cell's grid
    location (grid-aligned layout).
  - Node size = downstream importance: how many cells eventually
    depended on this one, following the chain of deductions.
  - Where the solve started is drawn as a rounded square: the givens,
    or the first solved cell when the puzzle has none (e.g. the app's "Killer" level).
  - Cell-to-cell cause edges drawn as thin lines with arrowheads
    at the target end. Edges fade with age (older = lighter).

Two node-coloring modes (--color-mode, or 'm'):
  "inference"
    Nodes colored by which rule produced them.
  "step"
    Nodes colored by their position in the temporal sequence,
    cool (early) to warm (late).

Usage:
  First solve the puzzle to produce its trace, then render it:
    uv run python projects/killer_sudoku/solve.py --puzzle 001_hard
    uv run python projects/killer_sudoku/sketch.py --puzzle 001_hard
    uv run python projects/killer_sudoku/sketch.py --puzzle 001_hard --step 30 --color-mode inference --save-and-exit

Interactive keys:
  Press 's' to save a hi-res PNG with metadata (via save_artwork).
  Press 'm' to toggle the color mode between "inference" and "step".
  Press 'c' to toggle cage outlines and sums on/off.
  Press 'r' to re-load the trace and re-render.

  Step navigation (lets you scrub through the solve interactively):
    Left / Right     step backward / forward by 1
    Down  / Up       step backward / forward by 10
    Home             jump to step 0 (just before first event)
    End              jump to final state (full solve)
"""

import argparse
import json
import sys
import math
from pathlib import Path

import py5

from coalescence.io import save_artwork


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_NAME = "killer_sudoku"
PROJECT_DIR = Path(__file__).resolve().parent
TRACES_DIR = PROJECT_DIR / "traces"
DEFAULT_PUZZLE = "001_hard"

# Set from --puzzle in the entry point.
puzzle_name = DEFAULT_PUZZLE
auto_save_and_exit = False

HI_RES_SIZE = 2400
PREVIEW_SIZE = 800
MARGIN_RATIO = 0.08

# Visual weights (relative to grid cell size `s`)
NODE_RADIUS_RATIO = 0.16
EDGE_WEIGHT_RATIO = 0.014
EDGE_AGE_FADE = 0.45
DIGIT_TEXT_RATIO = 0.30

# Node size from downstream importance (sqrt keeps big cascades from
# dwarfing everything else).
SIZE_BY_IMPORTANCE = True
IMPORTANCE_MIN_SCALE = 1.0
IMPORTANCE_MAX_SCALE = 1.5

# Start marker: a rounded square of the same area as the circle.
START_CORNER_RATIO = 0.25   # corner radius relative to the square's half-side

COLOR_MODE = "step"   # "inference" or "step"

# Cage outlines and sums — show the puzzle's constraint structure.
# Particularly useful for intermediate frames where the dependency graph
# alone doesn't convey enough context. Toggle at runtime with 'c'.
SHOW_CAGES = True
CAGE_LINE_WEIGHT_RATIO = 0.012    # cage boundary stroke weight
CAGE_DASH_LEN_RATIO = 0.06        # length of each dash
CAGE_DASH_GAP_RATIO = 0.045       # gap between dashes
CAGE_OUTLINE_INSET_RATIO = 0.05   # each cage's outline sits this far inside its cells
CAGE_SUM_TEXT_RATIO = 0.13        # cage sum font size relative to cell
CAGE_SUM_PAD_RATIO = 0.02         # background patch around the sum label

# Render only events with step <= MAX_STEP. Set to None to render all
# events (the full final frame). Used for generating intermediate frames
# to validate the static design as a progression.
#
# Example: setting MAX_STEP = 30 produces the image that would result if
# the solver had stopped after step 30 — the partial dependency graph
# at that point, with importance computed only from those 30 events.
MAX_STEP = None

# Key codes py5 doesn't name (Java's KeyEvent.VK_HOME / VK_END).
KEY_HOME = 36
KEY_END = 35

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
event_by_step = {}
downstream = {}     # step -> number of shown cells that eventually depended on it
start_steps = set()  # where the solve started: the givens, or the first solved cell
full_solve_length = 0   # total events in the underlying solve; used for color stability
pg = None


def _trace_path():
    return TRACES_DIR / f"{puzzle_name}.json"


def load_trace():
    """Read the trace from disk and cache the raw events plus all
    static puzzle metadata. Called once at startup. After this, changing
    MAX_STEP only requires apply_filter() — no disk re-read.
    """
    global trace_data, puzzle, full_solve_length
    global raw_events, _puzzle_loaded

    with open(_trace_path(), "r", encoding="utf-8") as f:
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

    # Not "every cell with no cell cause": the solver doesn't record
    # eliminations, so mid-solve cage deductions would look like starts too.
    start_steps.clear()
    start_steps.update(e["step"] for e in raw_events if e["inference_type"] == "given")
    if not start_steps and raw_events:
        start_steps.add(min(e["step"] for e in raw_events))

    print(f"Loaded trace: {_trace_path()}")
    print(f"  Solver commit: {trace_data.get('git_commit', 'unknown')}")
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

    # Direct children of each step (cells it was a cause of).
    children = {e["step"]: [] for e in events}
    for ev in events:
        cell_causes = [c["step"] for c in ev.get("causes", ()) if c.get("type") == "cell"]
        for src in cell_causes:
            if src in children:
                children[src].append(ev["step"])

    # Downstream importance: every cell reachable by following the arrows.
    # Causes always come earlier, so walking steps from last to first means
    # each child's set is complete before its parents need it. A cell with
    # several causes counts once for each ancestor.
    reachable = {}
    for step in sorted(children, reverse=True):
        acc = set()
        for child in children[step]:
            acc.add(child)
            acc |= reachable[child]
        reachable[step] = acc
    downstream.clear()
    downstream.update({step: len(r) for step, r in reachable.items()})

    if MAX_STEP is not None:
        print(f"  step={MAX_STEP} -> showing {len(events)}/{full_solve_length} events")
    else:
        print(f"  step=FINAL -> showing all {len(events)} events")


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
    _set_max_step(MAX_STEP)   # clamps a CLI --step to the solve length and renders


def render():
    pg.begin_draw()
    pg.background(THEME["bg"])

    margin = HI_RES_SIZE * MARGIN_RATIO
    grid_size = HI_RES_SIZE - 2 * margin
    s = grid_size / 9.0

    pg.push_matrix()
    pg.translate(margin, margin)

    _draw_grid(s)
    _draw_cages(s)
    _draw_edges(s)
    _draw_nodes(s)

    pg.pop_matrix()
    pg.end_draw()
    print(f"Render complete ({COLOR_MODE} mode).")


def _node_radius(ev, s):
    """Node radius: base size scaled by downstream importance."""
    scale = 1.0
    max_down = max(downstream.values(), default=0)
    if SIZE_BY_IMPORTANCE and max_down > 0:
        t = math.sqrt(downstream.get(ev["step"], 0) / max_down)
        scale = IMPORTANCE_MIN_SCALE + t * (IMPORTANCE_MAX_SCALE - IMPORTANCE_MIN_SCALE)
    return s * NODE_RADIUS_RATIO * scale


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
    """Draw each cage as its own dashed outline, inset inside its cells,
    plus the cage's sum in its top-left cell.

    Like the puzzle apps: neighboring cages never share a line, so their
    common border shows two parallel dashed lines with a gap between them.
    """
    if not SHOW_CAGES:
        return

    inset = s * CAGE_OUTLINE_INSET_RATIO
    dash_len = s * CAGE_DASH_LEN_RATIO
    gap_len = s * CAGE_DASH_GAP_RATIO

    pg.stroke(py5.color(THEME["cage_line"]))
    pg.stroke_weight(s * CAGE_LINE_WEIGHT_RATIO)
    pg.no_fill()
    pg.stroke_cap(py5.SQUARE)
    for cells in cage_cells.values():
        for loop in _cage_outline_loops(cells, s, inset):
            _dashed_path(loop, dash_len, gap_len)

    # Cage sums in the top-left cell (smallest row, then column), on a small
    # background patch that interrupts the dashed corner, as in the apps.
    text_size = s * CAGE_SUM_TEXT_RATIO
    pad = s * CAGE_SUM_PAD_RATIO
    pg.text_size(text_size)
    pg.text_align(py5.LEFT, py5.TOP)
    for cage_id, cells in cage_cells.items():
        tl_r, tl_c = min(cells)
        label = str(cage_sum[cage_id])
        x = tl_c * s + inset
        y = tl_r * s + inset
        pg.no_stroke()
        pg.fill(py5.color(THEME["bg"]))
        pg.rect(x - pad, y - pad, pg.text_width(label) + 2 * pad, text_size + 2 * pad)
        pg.fill(py5.color(THEME["cage_sum_text"]))
        pg.text(label, x, y)


def _cage_outline_loops(cells, s, d):
    """
    Closed outlines of a cage, inset by d inside its cells, as point lists.

    Each cell edge on the cage border becomes one inset segment. Its ends
    depend on the neighbors along that edge: a convex corner (no cage cell
    beside it) pulls the end in by d; a concave corner (cage cell beside it
    and diagonally ahead) pushes it out by d; otherwise the line runs on
    into the next cell. The segments are then chained end to end.
    """
    cs = set(cells)

    def inside(r, c):
        return (r, c) in cs

    def span(lo, hi, before, before_diag, after, after_diag):
        a = lo + d if not before else (lo - d if before_diag else lo)
        b = hi - d if not after else (hi + d if after_diag else hi)
        return a, b

    segments = []
    for r, c in cells:
        x0, y0, x1, y1 = c * s, r * s, (c + 1) * s, (r + 1) * s
        if not inside(r - 1, c):   # top
            a, b = span(x0, x1, inside(r, c - 1), inside(r - 1, c - 1),
                        inside(r, c + 1), inside(r - 1, c + 1))
            segments.append(((a, y0 + d), (b, y0 + d)))
        if not inside(r + 1, c):   # bottom
            a, b = span(x0, x1, inside(r, c - 1), inside(r + 1, c - 1),
                        inside(r, c + 1), inside(r + 1, c + 1))
            segments.append(((a, y1 - d), (b, y1 - d)))
        if not inside(r, c - 1):   # left
            a, b = span(y0, y1, inside(r - 1, c), inside(r - 1, c - 1),
                        inside(r + 1, c), inside(r + 1, c - 1))
            segments.append(((x0 + d, a), (x0 + d, b)))
        if not inside(r, c + 1):   # right
            a, b = span(y0, y1, inside(r - 1, c), inside(r - 1, c + 1),
                        inside(r + 1, c), inside(r + 1, c + 1))
            segments.append(((x1 - d, a), (x1 - d, b)))

    def key(p):
        return (round(p[0], 3), round(p[1], 3))

    at_point = {}
    for i, (p, q) in enumerate(segments):
        at_point.setdefault(key(p), []).append(i)
        at_point.setdefault(key(q), []).append(i)

    loops = []
    used = set()
    for start in range(len(segments)):
        if start in used:
            continue
        used.add(start)
        p, q = segments[start]
        loop = [p, q]
        while True:
            nxt = [i for i in at_point[key(loop[-1])] if i not in used]
            if not nxt:
                break
            used.add(nxt[0])
            a, b = segments[nxt[0]]
            loop.append(b if key(a) == key(loop[-1]) else a)
        loops.append(loop)
    return loops


def _dashed_path(points, dash_len, gap_len):
    """Draw a dashed polyline whose dash pattern runs on across corners."""
    eps = 1e-9
    period = dash_len + gap_len
    phase = 0.0   # distance into the current dash+gap period
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < eps:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length - eps:
            drawing = phase < dash_len - eps
            end = min(length, pos + (dash_len if drawing else period) - phase)
            if drawing:
                pg.line(x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end)
            # Snap to the dash/gap boundaries so rounding can't stall the loop.
            phase += end - pos
            if drawing and phase >= dash_len - eps:
                phase = dash_len
            if phase >= period - eps:
                phase = 0.0
            pos = end


def _draw_edges(s):
    pg.stroke_cap(py5.ROUND)
    base_col = py5.color(THEME["edge"])
    weight = s * EDGE_WEIGHT_RATIO
    arrow_size = s * ARROW_SIZE_RATIO
    edge_denom = max(1, full_solve_length - 1)

    for ev in events:
        step = ev["step"]
        tr, tc = ev["cell"]
        tx, ty = (tc + 0.5) * s, (tr + 0.5) * s
        inset = _node_radius(ev, s) * ARROW_INSET

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
    text_size_base = s * DIGIT_TEXT_RATIO
    n_events = len(events)
    base_radius = s * NODE_RADIUS_RATIO

    pg.text_align(py5.CENTER, py5.CENTER)

    for ev in events:
        r, c = ev["cell"]
        x, y = (c + 0.5) * s, (r + 0.5) * s
        radius = _node_radius(ev, s)
        fill_color = _node_fill(ev, n_events)

        # Fill plus a thin ink outline. Start cells: a rounded square of the
        # same area as the circle.
        pg.fill(fill_color)
        pg.stroke(THEME["node_text"], 110)
        pg.stroke_weight(s * 0.005)
        if ev["step"] in start_steps:
            half = radius * math.sqrt(math.pi) / 2
            pg.rect_mode(py5.CENTER)
            pg.rect(x, y, 2 * half, 2 * half, half * START_CORNER_RATIO)
            pg.rect_mode(py5.CORNER)
        else:
            pg.circle(x, y, 2 * radius)

        # Single digit color — no luminance-based switching.
        pg.no_stroke()
        pg.fill(THEME["node_text"])
        pg.text_size(text_size_base * radius / base_radius)
        pg.text(str(ev["digit"]), x, y - s * 0.01)


# ----------------------------------------------------------------------------
# Main loop and interaction
# ----------------------------------------------------------------------------

def draw():
    py5.image(pg, 0, 0, py5.width, py5.height)
    if auto_save_and_exit and py5.frame_count == 1:
        save_current()
        py5.exit_sketch()


def save_current():
    step_tag = "final" if MAX_STEP is None else f"step{MAX_STEP:03d}"
    params = {
        "puzzle": puzzle_name,
        "puzzle_id": puzzle.get("puzzle_id") if puzzle else None,
        "trace_git_commit": trace_data.get("git_commit") if trace_data else None,
        "max_step": MAX_STEP,
        "color_mode": COLOR_MODE,
        "node_size": "downstream" if SIZE_BY_IMPORTANCE else "constant",
        "start_marker": "givens" if givens_set else "first_cell",
        "show_cages": SHOW_CAGES,
        "hi_res_size": HI_RES_SIZE,
        "margin_ratio": MARGIN_RATIO,
    }
    path = save_artwork(pg, PROJECT_NAME, params=params,
                        label=puzzle_name, suffix=f"{COLOR_MODE}_{step_tag}")
    print(f"Saved: {path}")


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
    global COLOR_MODE, SHOW_CAGES, MAX_STEP

    # Save / mode toggles
    if py5.key == "s":
        save_current()
    elif py5.key == "r":
        load_trace()
        render()
    elif py5.key == "m":
        COLOR_MODE = "inference" if COLOR_MODE == "step" else "step"
        print(f"Switched COLOR_MODE -> {COLOR_MODE}")
        render()
    elif py5.key == "c":
        SHOW_CAGES = not SHOW_CAGES
        print(f"Cage outlines: {'on' if SHOW_CAGES else 'off'}")
        render()

    # Step navigation. py5 reports arrow keys via key_code, not key.
    # Up/Down step by 10, Left/Right by 1.
    elif py5.key_code in (py5.LEFT, py5.RIGHT, py5.UP, py5.DOWN, KEY_HOME, KEY_END):

        # Determine current position. None = "final" = past the last index.
        current = MAX_STEP if MAX_STEP is not None else full_solve_length - 1

        if py5.key_code == py5.LEFT:
            _set_max_step(current - 1)
        elif py5.key_code == py5.RIGHT:
            _set_max_step(current + 1)
        elif py5.key_code == py5.DOWN:
            _set_max_step(current - 10)
        elif py5.key_code == py5.UP:
            _set_max_step(current + 10)
        elif py5.key_code == KEY_HOME:
            _set_max_step(0)
        elif py5.key_code == KEY_END:
            _set_max_step(None)


# ----------------------------------------------------------------------------
# CLI and entry point
# ----------------------------------------------------------------------------

def parse_args():
    available = sorted(p.stem for p in TRACES_DIR.glob("*.json"))
    parser = argparse.ArgumentParser(description="Killer sudoku trace renderer.")
    parser.add_argument("--puzzle", type=str, default=DEFAULT_PUZZLE,
                        help=f"Puzzle name; reads traces/<name>.json. Default: {DEFAULT_PUZZLE}. "
                             f"Available traces: {', '.join(available) or 'none'}.")
    parser.add_argument("--step", type=int, default=None,
                        help="Render only events up to this step. Default: final state.")
    parser.add_argument("--color-mode", type=str, default=COLOR_MODE,
                        choices=["step", "inference"],
                        help=f"Node coloring. Default: {COLOR_MODE}.")
    parser.add_argument("--cages", action=argparse.BooleanOptionalAction,
                        default=SHOW_CAGES, help="Cage outlines and sums.")
    parser.add_argument("--save-and-exit", action="store_true",
                        help="Render once, save, and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    puzzle_name = args.puzzle
    if not _trace_path().exists():
        sys.exit(f"No trace at {_trace_path()}.\n"
                 f"Run first: uv run python projects/killer_sudoku/solve.py --puzzle {puzzle_name}")
    MAX_STEP = args.step
    COLOR_MODE = args.color_mode
    SHOW_CAGES = args.cages
    auto_save_and_exit = args.save_and_exit
    py5.run_sketch()
