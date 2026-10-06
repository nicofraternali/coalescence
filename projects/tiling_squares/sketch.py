"""
Tiling squares — Truchet-style tiling with filled diamonds.

A 2^L by 2^L grid of diagonal lines, with diamonds drawn where a 2x2
neighborhood matches the "1001" pattern.

Interactive keys:
    r       New random structure (new seed), keep current theme.
    space   New random theme, keep current structure.
    s       Save the current piece.

Usage:
    uv run python projects/tiling_squares/sketch.py
    uv run python projects/tiling_squares/sketch.py --seed 4823 --theme JAPAN
    uv run python projects/tiling_squares/sketch.py --seed 4823 --theme JAPAN --save-and-exit
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

PROJECT_NAME = "tiling_squares"


# ---------------------------------------------------------------------------
# Module state.
# ---------------------------------------------------------------------------

L: int = DEFAULT_L
seed: int = 0
theme_name: str = DEFAULT_THEME
palette: dict = {}
grid: list[list[int]] = []
geom: GridGeometry | None = None
pg = None
auto_save_and_exit: bool = False


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiling squares generative art.")
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

def render_art() -> None:
    """Paint the current grid into the high-res buffer using the current palette."""
    if not grid or geom is None:
        return

    s = geom.cell_size
    pg.begin_draw()
    pg.background(palette["bg"])
    pg.push_matrix()
    pg.translate(geom.margin, geom.margin)

    # Phase A: filled diamonds.
    pg.no_stroke()
    pg.fill(palette["accent"])
    for r, c in find_diamonds(grid):
        cx = (c + 1) * s
        cy = (r + 1) * s
        pg.quad(cx, cy - s, cx + s, cy, cx, cy + s, cx - s, cy)

    # Phase B: diagonal lines per cell.
    pg.stroke(palette["line"])
    pg.stroke_weight(HI_RES_SIZE * 0.005)
    pg.stroke_cap(py5.ROUND)
    for r in range(geom.grid_size):
        for c in range(geom.grid_size):
            x = c * s
            y = r * s
            if grid[r][c] == 0:
                pg.line(x, y, x + s, y + s)
            else:
                pg.line(x, y + s, x + s, y)

    pg.pop_matrix()
    pg.end_draw()


def new_composition(use_seed: int | None = None) -> None:
    """Reseed RNG, generate a new grid, render."""
    global seed, grid
    seed = init_seed(use_seed)
    grid = generate_grid(L)
    render_art()
    print(f"Composition: seed={seed}, theme={theme_name}, L={L}")


def cycle_theme() -> None:
    """Pick a new random theme; re-render existing grid."""
    global theme_name, palette
    theme_name = random.choice(list_tiling_themes())
    palette = TILING_THEMES[theme_name]
    render_art()
    print(f"Theme: {theme_name}")


def save_current() -> None:
    """Save with full metadata."""
    params = {"L": L, "margin_ratio": MARGIN_RATIO, "hi_res_size": HI_RES_SIZE}
    path = save_artwork(pg, PROJECT_NAME, seed, theme_name, params)
    print(f"Saved: {path}")
    print(f"  seed={seed}  theme={theme_name}  L={L}")


# ---------------------------------------------------------------------------
# py5 lifecycle.
# ---------------------------------------------------------------------------

def setup() -> None:
    global pg, palette, geom
    py5.size(PREVIEW_SIZE, PREVIEW_SIZE)
    pg = py5.create_graphics(HI_RES_SIZE, HI_RES_SIZE)

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