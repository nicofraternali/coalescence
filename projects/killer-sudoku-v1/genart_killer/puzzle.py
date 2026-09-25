"""
puzzle.py — Killer sudoku puzzle representation and loading.

This module is responsible for:
  - Defining the data structures (Cage, Puzzle, SolveState).
  - Loading puzzles from YAML files.
  - Validating structural correctness.
  - Constructing the initial solve state (unpruned candidate sets).

It deliberately does NOT apply any solving rules — not even cage initial
pruning. All inferences must be visible to the trace, so they belong in
the solver, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterator

import yaml


# A cell coordinate. Row-major, zero-indexed: (row, col).
Cell = tuple[int, int]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Cage:
    """A killer sudoku cage: a sum and a set of cells.

    Frozen because cages don't change once a puzzle is loaded. The set of
    *valid combinations* a cage can take is mutable solver state and lives
    in SolveState, not here.
    """
    id: int
    sum: int
    cells: tuple[Cell, ...]   # tuple, not list — needed for frozen=True

    @property
    def size(self) -> int:
        return len(self.cells)

    def valid_combinations(self) -> list[frozenset[int]]:
        """All sets of `size` distinct digits in 1..9 summing to `sum`.

        Computed on demand. Cheap (cage sizes are small) and pure, so we
        don't bother caching at this layer — the solver will cache.
        """
        return [
            frozenset(combo)
            for combo in combinations(range(1, 10), self.size)
            if sum(combo) == self.sum
        ]


@dataclass(frozen=True)
class Given:
    """A pre-filled digit in the initial puzzle. Rare in killer sudoku."""
    cell: Cell
    digit: int


@dataclass(frozen=True)
class Puzzle:
    """A complete killer sudoku puzzle, as loaded from disk.

    This is the static description — what the puzzle *is*. The dynamic
    state during solving is SolveState (below).
    """
    puzzle_id: str
    source: str
    difficulty: str
    cages: tuple[Cage, ...]
    givens: tuple[Given, ...]

    def cage_of(self, cell: Cell) -> Cage:
        """Return the cage containing the given cell."""
        for cage in self.cages:
            if cell in cage.cells:
                return cage
        raise ValueError(f"Cell {cell} is not in any cage")


@dataclass
class SolveState:
    """The mutable solving state. Not frozen — the solver mutates this.

    `candidates[r][c]` is the set of digits still possible at cell (r, c).
    A resolved cell has a singleton candidate set; we use that uniform
    representation rather than a separate "resolved digit" field so rules
    don't need to special-case resolved cells.

    `cage_combinations[cage_id]` is the set of combinations still valid
    for that cage. Initially every cage's full combination list; pruned
    by the solver as cells resolve.

    `eliminated_by[(r, c, d)]` records the *causes* (not step numbers)
    that eliminated digit d from cell (r, c). A cause is either a
    structural constraint (CauseCage, CauseHouse) or a prior
    cell_resolved event (CauseCell). Rules read this directly when
    building their own event's cause list — no event-by-step lookup
    needed, so causes stay sound even when no event was emitted for
    an internal elimination (like the silent ones during cage
    initial pruning).

    The value is `tuple[Any, ...]` here to avoid importing Cause from
    trace.py and creating a circular import. The solver and rules
    treat these tuples as opaque Cause sequences.
    """
    candidates: list[list[set[int]]]
    cage_combinations: dict[int, set[frozenset[int]]]
    eliminated_by: dict[tuple[int, int, int], tuple] = field(default_factory=dict)

    def is_resolved(self, cell: Cell) -> bool:
        r, c = cell
        return len(self.candidates[r][c]) == 1

    def digit_at(self, cell: Cell) -> int | None:
        r, c = cell
        if len(self.candidates[r][c]) == 1:
            return next(iter(self.candidates[r][c]))
        return None

    def is_complete(self) -> bool:
        return all(
            len(self.candidates[r][c]) == 1
            for r in range(9) for c in range(9)
        )

    def eliminate(self, cell: Cell, digit: int, causes: tuple) -> bool:
        """Remove `digit` from `cell`'s candidates, recording the causes.

        Returns True if the digit was actually removed (it was a candidate),
        False if it was already absent. Idempotent and safe to call
        repeatedly.
        """
        r, c = cell
        if digit in self.candidates[r][c]:
            self.candidates[r][c].discard(digit)
            self.eliminated_by[(r, c, digit)] = causes
            return True
        return False


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load(path: str | Path) -> Puzzle:
    """Load a puzzle from a YAML file and validate it.

    Raises ValueError on any structural problem. We fail loudly here — a
    malformed puzzle should never reach the solver.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cages = tuple(
        Cage(
            id=c["id"],
            sum=c["sum"],
            cells=tuple((r, col) for r, col in c["cells"]),
        )
        for c in data["cages"]
    )

    givens = tuple(
        Given(cell=(g["cell"][0], g["cell"][1]), digit=g["digit"])
        for g in data.get("givens", [])
    )

    puzzle = Puzzle(
        puzzle_id=data["puzzle_id"],
        source=data.get("source", ""),
        difficulty=data.get("difficulty", "unknown"),
        cages=cages,
        givens=givens,
    )

    _validate(puzzle)
    return puzzle


def _validate(puzzle: Puzzle) -> None:
    """Check that the puzzle is structurally sound.

    Collects ALL errors before raising, so transcription mistakes can
    be diagnosed in a single pass. Stopping at the first error makes
    fix-rerun-fix-rerun cycles unnecessary.

    We check:
      - Every cell (r, c) for r, c in 0..8 belongs to exactly one cage.
      - No cell appears in more than one cage.
      - Each cage's sum is achievable given its size.
      - Givens reference valid cells with digits in 1..9.
      - Givens do not duplicate a cell.
    """
    errors: list[str] = []

    # Build cell -> list-of-cages membership map (lists, not single ids,
    # so we can report every cage that claims a duplicated cell).
    membership: dict[Cell, list[int]] = {}
    for cage in puzzle.cages:
        if cage.size == 0:
            errors.append(f"Cage {cage.id} has no cells")
            continue
        if not cage.valid_combinations():
            errors.append(
                f"Cage {cage.id} sum={cage.sum} size={cage.size} "
                "has no valid digit combinations"
            )
        for cell in cage.cells:
            r, c = cell
            if not (0 <= r < 9 and 0 <= c < 9):
                errors.append(f"Cage {cage.id}: cell {cell} out of bounds")
                continue
            membership.setdefault(cell, []).append(cage.id)

    # Report all overlaps.
    for cell, cage_ids in membership.items():
        if len(cage_ids) > 1:
            errors.append(f"Cell {cell} is in cages {cage_ids}")

    # Every cell must be covered.
    missing = [
        (r, c) for r in range(9) for c in range(9)
        if (r, c) not in membership
    ]
    if missing:
        errors.append(f"Cells not in any cage: {missing}")

    # The sum of all cage sums must equal 9 * 45 = 405.
    total = sum(cage.sum for cage in puzzle.cages)
    if total != 405:
        errors.append(
            f"Cage sums total {total}, expected 405 (9 rows of 45)"
        )

    # Givens. Check bounds, range, and duplicates.
    given_cells: set[Cell] = set()
    for given in puzzle.givens:
        r, c = given.cell
        if not (0 <= r < 9 and 0 <= c < 9):
            errors.append(f"Given cell {given.cell} out of bounds")
            continue
        if given.cell in given_cells:
            errors.append(f"Given cell {given.cell} is listed more than once")
        given_cells.add(given.cell)
        if not (1 <= given.digit <= 9):
            errors.append(f"Given digit {given.digit} not in 1..9 (cell {given.cell})")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"Puzzle validation failed with {len(errors)} error(s):\n  - {joined}"
        )


# ----------------------------------------------------------------------------
# Initial state construction
# ----------------------------------------------------------------------------

def initial_state(puzzle: Puzzle) -> SolveState:
    """Construct the initial solve state.

    Every cell starts with candidates {1..9}. Givens are placed (their
    candidate sets reduced to singletons) but no other pruning happens —
    cage constraints, house constraints, etc. are the solver's job, so
    they appear in the trace.
    """
    candidates: list[list[set[int]]] = [
        [set(range(1, 10)) for _ in range(9)]
        for _ in range(9)
    ]
    for given in puzzle.givens:
        r, c = given.cell
        candidates[r][c] = {given.digit}

    cage_combinations: dict[int, set[frozenset[int]]] = {
        cage.id: set(cage.valid_combinations())
        for cage in puzzle.cages
    }

    return SolveState(
        candidates=candidates,
        cage_combinations=cage_combinations,
    )


# ----------------------------------------------------------------------------
# House iteration helpers
# ----------------------------------------------------------------------------
#
# Houses (rows, columns, boxes) are not stored — they're computed on
# demand. These helpers are used heavily by the solver's rules, so they
# live here in puzzle.py rather than being scattered across rules.

def row_cells(r: int) -> Iterator[Cell]:
    for c in range(9):
        yield (r, c)


def col_cells(c: int) -> Iterator[Cell]:
    for r in range(9):
        yield (r, c)


def box_cells(b: int) -> Iterator[Cell]:
    """Box index b in 0..8, reading left-to-right, top-to-bottom."""
    br, bc = (b // 3) * 3, (b % 3) * 3
    for dr in range(3):
        for dc in range(3):
            yield (br + dr, bc + dc)


def all_houses() -> Iterator[tuple[str, int, list[Cell]]]:
    """Yield (house_type, index, cells) for all 27 houses."""
    for r in range(9):
        yield ("row", r, list(row_cells(r)))
    for c in range(9):
        yield ("col", c, list(col_cells(c)))
    for b in range(9):
        yield ("box", b, list(box_cells(b)))
