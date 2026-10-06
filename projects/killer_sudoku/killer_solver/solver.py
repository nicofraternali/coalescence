"""
solver.py — The main constraint-propagation loop.

Given a Puzzle, the solver:
  1. Builds the initial SolveState.
  2. Emits CellResolved events for any givens.
  3. Iterates: try rules in order; on the first one that fires, restart
     from rule 0. Stop when no rule fires.
  4. Returns the trace and the final state.

The solver is short because all the logic lives in rules.py. This is
intentional: the solver is just orchestration, and adding/removing
rules is a one-line change to rules.ALL_RULES.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import rules
from .puzzle import Puzzle, SolveState, initial_state
from .trace import CellResolved, Trace


@dataclass
class SolveResult:
    trace: Trace
    state: SolveState
    solved: bool
    stuck_cells: list[tuple[int, int]]


def _puzzle_to_dict(puzzle: Puzzle) -> dict:
    """Convert a Puzzle to the dict shape expected by Trace.puzzle_data."""
    return {
        "puzzle_id": puzzle.puzzle_id,
        "source": puzzle.source,
        "difficulty": puzzle.difficulty,
        "cages": [
            {"id": c.id, "sum": c.sum, "cells": [list(cell) for cell in c.cells]}
            for c in puzzle.cages
        ],
        "givens": [
            {"cell": list(g.cell), "digit": g.digit} for g in puzzle.givens
        ],
    }


def solve(puzzle: Puzzle, seed: int = 0) -> SolveResult:
    """Run the solver to completion (or until stuck)."""
    rules.initialize(puzzle)
    state = initial_state(puzzle)
    trace = Trace(
        puzzle_id=puzzle.puzzle_id,
        puzzle_data=_puzzle_to_dict(puzzle),
        seed=seed,
    )

    # Emit a CellResolved event for each given. Givens have no causes;
    # they're the roots of the dependency graph. We also propagate the
    # given digit to peers, attributing each peer elimination to this
    # given's resolution event.
    for given in puzzle.givens:
        step = trace.next_step()
        trace.append(CellResolved(
            step=step,
            cell=given.cell,
            digit=given.digit,
            inference_type="given",
            causes=(),
        ))
        # Lock the cell to the given digit.
        r, c = given.cell
        state.candidates[r][c] = {given.digit}
        # Propagate to peers.
        from .trace import CauseCell
        propagation_cause = (CauseCell(cell=given.cell, step=step),)
        cage_id = rules._cage_id_of(given.cell)
        for peer in rules._peers(given.cell, cage_id):
            pr, pc = peer
            if given.digit in state.candidates[pr][pc]:
                state.candidates[pr][pc].discard(given.digit)
                state.eliminated_by[(pr, pc, given.digit)] = propagation_cause

    # Main loop.
    max_iters = 10000  # safety net; a real solve won't come close
    iters = 0
    while iters < max_iters:
        iters += 1
        fired = False
        for rule in rules.ALL_RULES:
            if rule(state, puzzle, trace):
                fired = True
                break
        if not fired:
            break

    solved = state.is_complete()
    stuck = [
        (r, c) for r in range(9) for c in range(9)
        if len(state.candidates[r][c]) != 1
    ]
    return SolveResult(trace=trace, state=state, solved=solved, stuck_cells=stuck)
