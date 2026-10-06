"""
Solve a killer sudoku puzzle and save its trace as JSON.

First stage of the killer_sudoku pipeline:

    puzzles/<name>.yaml  --solve.py-->  traces/<name>.json  --sketch.py-->  image

The trace records every deduction the solver made and the git commit of the
solver code that produced it. The solver is deterministic, so the same puzzle
and commit always give the same trace. Traces are plain JSON so other tools
(e.g., a JavaScript animation on the blog) can read them too.

Usage:
    uv run python projects/killer_sudoku/solve.py --puzzle 001_hard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coalescence.io import git_provenance
from killer_solver import puzzle as pz
from killer_solver import solver

PROJECT_DIR = Path(__file__).resolve().parent
PUZZLES_DIR = PROJECT_DIR / "puzzles"
TRACES_DIR = PROJECT_DIR / "traces"


def parse_args() -> argparse.Namespace:
    available = sorted(p.stem for p in PUZZLES_DIR.glob("*.yaml"))
    parser = argparse.ArgumentParser(description="Solve a killer sudoku and save its trace.")
    parser.add_argument("--puzzle", type=str, required=True, choices=available,
                        help="Puzzle name (a file in puzzles/, without .yaml).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle = pz.load(PUZZLES_DIR / f"{args.puzzle}.yaml")
    result = solver.solve(puzzle)

    provenance = git_provenance()
    result.trace.git_commit = provenance["git_commit"]
    result.trace.git_dirty = provenance["git_dirty"]

    trace_path = TRACES_DIR / f"{args.puzzle}.json"
    result.trace.save(trace_path)

    print(f"Puzzle: {puzzle.puzzle_id}")
    print(f"  Solved: {result.solved}  (stuck cells: {len(result.stuck_cells)})")
    print(f"  Trace events: {len(result.trace.events)}")
    print(f"  Commit: {provenance['git_commit']}{' (dirty)' if provenance['git_dirty'] else ''}")
    print(f"Trace saved to {trace_path}")


if __name__ == "__main__":
    main()
