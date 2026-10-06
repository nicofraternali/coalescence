from pathlib import Path

from killer_solver import puzzle as p
from killer_solver import solver as s

PUZZLES_DIR = Path(__file__).resolve().parent.parent / "puzzles"


def test_hard_puzzle_solves_completely():
    result = s.solve(p.load(PUZZLES_DIR / "001_hard.yaml"))
    assert result.solved
    assert result.stuck_cells == []
    assert len(result.trace.events) == 81


def test_solver_is_deterministic():
    # Traces are only reproducible from puzzle + commit if this holds.
    pz = p.load(PUZZLES_DIR / "001_hard.yaml")
    assert s.solve(pz).trace.to_dict() == s.solve(pz).trace.to_dict()
