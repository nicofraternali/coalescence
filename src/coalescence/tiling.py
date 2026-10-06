"""
Truchet-style tiling primitives.

Both `tiling_squares` and `tiling_holes` build on the same combinatorial
substrate: a 2^L by 2^L grid where each cell carries a 0 or 1, drawn as
a diagonal line in one of two orientations. Where a 2x2 neighborhood
matches the "1001" pattern (top-left and bottom-right are 1, top-right
and bottom-left are 0), a diamond is implied at the cell intersection.

This module exposes the shared logic. The two sketches differ only in
how they render the diamonds and lines — `tiling_squares` fills diamonds
and draws plain lines, while `tiling_holes` cuts V-shaped trenches and
draws thicker wires that interrupt at trench centers.

Usage:

    from coalescence.tiling import generate_grid, find_diamonds, GridGeometry

    grid = generate_grid(L=4)              # 16x16 grid of 0s and 1s
    diamonds = find_diamonds(grid)         # set of (row, col) tuples
    geom = GridGeometry(L=4, canvas_size=2400, margin_ratio=0.08)
    s = geom.cell_size                     # pixels per cell
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Grid generation and analysis.
# ---------------------------------------------------------------------------

def generate_grid(L: int) -> list[list[int]]:
    """
    Return a fresh 2^L by 2^L grid of random 0/1 values.

    Uses Python's `random` module, so callers should seed it via
    `coalescence.seeds.init_seed` before calling for reproducibility.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    size = 2**L
    return [[random.choice([0, 1]) for _ in range(size)] for _ in range(size)]


def find_diamonds(grid: list[list[int]]) -> set[tuple[int, int]]:
    """
    Return the set of (row, col) coordinates where a diamond should appear.

    A diamond is centered at the intersection between cells (r, c), (r, c+1),
    (r+1, c), (r+1, c+1) when those cells form the pattern:

        1 0
        0 1

    The returned coordinate (r, c) is the top-left cell of the matching
    2x2 neighborhood. The diamond's geometric center is at pixel position
    ((c + 1) * cell_size, (r + 1) * cell_size).
    """
    diamonds: set[tuple[int, int]] = set()
    rows = len(grid)
    if rows == 0:
        return diamonds
    cols = len(grid[0])
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (
                grid[r][c] == 1
                and grid[r][c + 1] == 0
                and grid[r + 1][c] == 0
                and grid[r + 1][c + 1] == 1
            ):
                diamonds.add((r, c))
    return diamonds


# ---------------------------------------------------------------------------
# Geometry helper.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridGeometry:
    """
    Pre-computed geometry for rendering a tiling onto a canvas.

    Encapsulates the conversion from grid coordinates (r, c) to pixel
    coordinates, accounting for the active drawing area inside a margin.

    Attributes
    ----------
    L : int
        Grid resolution exponent. Grid is 2^L by 2^L.
    canvas_size : float
        Side length of the square canvas in pixels (e.g., 2400).
    margin_ratio : float
        Fraction of canvas_size used as margin on each side (e.g., 0.08).
    """
    L: int
    canvas_size: float
    margin_ratio: float

    @property
    def grid_size(self) -> int:
        return 2**self.L

    @property
    def margin(self) -> float:
        return self.canvas_size * self.margin_ratio

    @property
    def active_width(self) -> float:
        return self.canvas_size - 2 * self.margin

    @property
    def cell_size(self) -> float:
        """Side length of one grid cell in pixels."""
        return self.active_width / self.grid_size