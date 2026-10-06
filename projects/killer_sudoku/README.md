# killer_sudoku

A killer sudoku solver that records every deduction it makes, rendered as a
dependency graph over the grid: one node per resolved cell, arrows from the
cells that caused each deduction, tints on the cages and houses involved.

Input-driven rather than random: there is no seed and a single fixed style.
A piece is reproduced from the puzzle file, the render settings, and the
commit (all recorded in the PNG metadata).

## Run

Two stages. First solve the puzzle, which writes `traces/<name>.json`:

```powershell
uv run python projects/killer_sudoku/solve.py --puzzle 001_hard
```

Then render the trace:

```powershell
uv run python projects/killer_sudoku/sketch.py --puzzle 001_hard
```

Reproduce a specific piece:

```powershell
uv run python projects/killer_sudoku/sketch.py --puzzle 001_hard --step 30 --color-mode inference --save-and-exit
```

Tests:

```powershell
uv run pytest
```

## Keys

- `s` — save current piece (via `save_artwork`)
- `m` — toggle color mode (`step` / `inference`)
- `o` — toggle structural overlays
- `c` — toggle cage outlines and sums
- `r` — reload the trace and re-render
- `Left` / `Right` — step back / forward by 1
- `Down` / `Up` — step back / forward by 10
- `Home` / `End` — first step / final state

## Parameters

- `--puzzle` — puzzle name (a file in `puzzles/`, without `.yaml`). Default `001_hard`.
- `--step` — render only events up to this step. Default: final state.
- `--color-mode` — `step` (cool to warm over time) or `inference` (by rule).
- `--overlays` / `--no-overlays`, `--cages` / `--no-cages`.
- `--save-and-exit` — save once and exit without interaction.

## Layout

- `killer_solver/` — puzzle loading, deduction rules, solver, trace format.
  Local to this project (no other project reuses it).
- `puzzles/` — input puzzles (YAML).
- `traces/` — solver output (JSON, gitignored, regenerable with `solve.py`).
  Each trace records the commit of the solver that produced it.
