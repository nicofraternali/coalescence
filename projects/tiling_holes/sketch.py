"""
Tiling holes — Truchet-style tiling with V-cut trenches.

Same combinatorial substrate as tiling_squares, but the rendering inverts
the figure-ground relationship: where tiling_squares fills diamonds with
a flat color, tiling_holes cuts V-shaped trenches into the surface, with
per-vertex Gouraud-shaded gradients giving the impression of depth.
The diagonal "wires" are drawn over the surface, with breaks where they
cross trench connections.

Interactive keys:
    r       New random structure (new seed), keep current theme.
    space   New random theme, keep current structure.
    s       Save the current piece.

Usage:
    uv run python projects/tiling_holes/sketch.py
    uv run python projects/tiling_holes/sketch.py --seed 4823 --theme JAPAN
    uv run python projects/tiling_holes/sketch.py --seed 4823 --theme JAPAN --save-and-exit
"""

from __future__ import annotations

import argparse
import random

import py5

from coalescence.io import save_artwork
from coalescence.palettes import TILING_THEMES, list_tiling_themes
from coalescence.seeds import init_seed
from coalescence.tiling import GridGeometry, find_diamonds, generate_grid


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

DEFAULT_L = 4
DEFAULT_THEME = "ORIGINAL"
HI_RES_SIZE = 2400
PREVIEW_SIZE = 600
MARGIN_RATIO = 0.08

PROJECT_NAME = "tiling_holes"


# ---------------------------------------------------------------------------
# Module state.
# ---------------------------------------------------------------------------

L: int = DEFAULT_L
seed: int = 0
theme_name: str = DEFAULT_THEME
palette: dict = {}
grid: list[list[int]] = []
diamond_set: set[tuple[int, int]] = set()
geom: GridGeometry | None = None
pg = None
auto_save_and_exit: bool = False


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiling holes generative art.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic reproduction.")
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME,
                        choices=list_tiling_themes(),
                        help=f"Theme name. Default: {DEFAULT_THEME}.")
    parser.add_argument("--L", type=int, default=DEFAULT_L,
                        help=f"Grid resolution exponent. Default: {DEFAULT_L}.")
    parser.add_argument("--save-and-exit", action="store_true",
                        help="With --seed, render once, save, and exit.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Composition.
# ---------------------------------------------------------------------------

def _draw_trench_wall(
    cx: float, cy: float, tip1: tuple[float, float], tip2: tuple[float, float],
    neighbor_offset: tuple[int, int], r: int, c: int,
    s: float, surf_col: int, deep_col: int,
) -> None:
    """
    Draw one of the four V-cut walls around a diamond center.

    If the neighbor at (r + dr, c + dc) is also a diamond, the wall connects
    to the midpoint between centers (continuous trench). Otherwise the wall
    closes off as an isolated pit at this diamond.
    """
    dr, dc = neighbor_offset
    if (r + dr, c + dc) in diamond_set:
        ncx = (c + dc + 1) * s
        ncy = (r + dr + 1) * s
        mx = (cx + ncx) / 2
        my = (cy + ncy) / 2
        pg.fill(deep_col); pg.vertex(cx, cy)
        pg.fill(surf_col); pg.vertex(*tip1)
        pg.fill(deep_col); pg.vertex(mx, my)
        pg.fill(deep_col); pg.vertex(cx, cy)
        pg.fill(surf_col); pg.vertex(*tip2)
        pg.fill(deep_col); pg.vertex(mx, my)
    else:
        pg.fill(deep_col); pg.vertex(cx, cy)
        pg.fill(surf_col); pg.vertex(*tip1)
        pg.fill(surf_col); pg.vertex(*tip2)


def render_art() -> None:
    """Paint the current grid into the high-res buffer."""
    if not grid or geom is None:
        return

    s = geom.cell_size
    surf_col = py5.color(palette["bg"])
    deep_col = py5.color(palette["deep"])

    pg.begin_draw()
    pg.background(palette["bg"])
    pg.push_matrix()
    pg.translate(geom.margin, geom.margin)

    # --- Layer 1: V-cut trenches with Gouraud-shaded triangles. ---
    pg.no_stroke()
    pg.begin_shape(py5.TRIANGLES)

    for r, c in diamond_set:
        cx = (c + 1) * s
        cy = (r + 1) * s
        # Four walls: TL, TR, BR, BL — defined by their two outer tips
        # and the offset to check for a connecting neighbor diamond.
        walls = [
            ((cx - s, cy), (cx, cy - s), (-1, -1)),  # top-left
            ((cx, cy - s), (cx + s, cy), (-1,  1)),  # top-right
            ((cx + s, cy), (cx, cy + s), ( 1,  1)),  # bottom-right
            ((cx, cy + s), (cx - s, cy), ( 1, -1)),  # bottom-left
        ]
        for tip1, tip2, offset in walls:
            _draw_trench_wall(cx, cy, tip1, tip2, offset, r, c, s, surf_col, deep_col)

    pg.end_shape()

    # --- Layer 2: wire mesh over the surface. ---
    pg.stroke(palette["line"])
    pg.stroke_weight(s * 0.15)
    pg.stroke_cap(py5.ROUND)

    for r in range(geom.grid_size):
        for c in range(geom.grid_size):
            x = c * s
            y = r * s
            skip_wire = False
            if grid[r][c] == 0:
                if (r, c - 1) in diamond_set and (r - 1, c) in diamond_set:
                    skip_wire = True
                if not skip_wire:
                    pg.line(x, y, x + s, y + s)
            else:
                if (r, c) in diamond_set and (r - 1, c - 1) in diamond_set:
                    skip_wire = True
                if not skip_wire:
                    pg.line(x, y + s, x + s, y)

    pg.pop_matrix()
    pg.end_draw()


def new_composition(use_seed: int | None = None) -> None:
    global seed, grid, diamond_set
    seed = init_seed(use_seed)
    grid = generate_grid(L)
    diamond_set = find_diamonds(grid)
    render_art()
    print(f"Composition: seed={seed}, theme={theme_name}, L={L}, diamonds={len(diamond_set)}")


def cycle_theme() -> None:
    global theme_name, palette
    theme_name = random.choice(list_tiling_themes())
    palette = TILING_THEMES[theme_name]
    render_art()
    print(f"Theme: {theme_name}")


def save_current() -> None:
    params = {
        "L": L,
        "margin_ratio": MARGIN_RATIO,
        "hi_res_size": HI_RES_SIZE,
        "n_diamonds": len(diamond_set),
    }
    path = save_artwork(pg, PROJECT_NAME, seed, theme_name, params)
    print(f"Saved: {path}")
    print(f"  seed={seed}  theme={theme_name}  L={L}")


# ---------------------------------------------------------------------------
# py5 lifecycle.
# ---------------------------------------------------------------------------

def setup() -> None:
    global pg, palette, geom
    py5.size(PREVIEW_SIZE, PREVIEW_SIZE, py5.P2D)
    pg = py5.create_graphics(HI_RES_SIZE, HI_RES_SIZE, py5.P2D)

    palette = TILING_THEMES[theme_name]
    geom = GridGeometry(L=L, canvas_size=HI_RES_SIZE, margin_ratio=MARGIN_RATIO)
    new_composition(use_seed=seed if seed != 0 else None)


def draw() -> None:
    py5.image(pg, 0, 0, py5.width, py5.height)
    if auto_save_and_exit and py5.frame_count == 1:
        save_current()
        py5.exit_sketch()


def key_pressed() -> None:
    if py5.key == "s":
        save_current()
    elif py5.key == "r":
        new_composition()
    elif py5.key == " ":
        cycle_theme()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    L = args.L
    theme_name = args.theme
    auto_save_and_exit = args.save_and_exit
    seed = args.seed if args.seed is not None else 0
    py5.run_sketch()