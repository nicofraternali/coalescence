from pathlib import Path

from killer_solver import puzzle as p

PUZZLES_DIR = Path(__file__).resolve().parent.parent / "puzzles"


def test_puzzle_loads_with_complete_cage_sums():
    pz = p.load(PUZZLES_DIR / "001_hard.yaml")
    assert pz.puzzle_id
    # Every digit 1-9 appears 9 times: 9 * 45 = 405.
    assert sum(c.sum for c in pz.cages) == 405


def test_initial_state_cage_combinations():
    pz = p.load(PUZZLES_DIR / "001_hard.yaml")
    state = p.initial_state(pz)
    # Cage 3 is a 2-cell cage summing to 16: only 7 + 9 works.
    assert sorted(sorted(c) for c in state.cage_combinations[3]) == [[7, 9]]
    # Cage 4 is a 2-cell cage summing to 11.
    assert sorted(sorted(c) for c in state.cage_combinations[4]) == [[2, 9], [3, 8], [4, 7], [5, 6]]
