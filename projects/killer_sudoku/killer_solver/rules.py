"""
rules.py — Constraint-propagation rules for the killer sudoku v1 solver.

Each rule is a function (state, puzzle, trace) -> bool. It returns True
if it fired (produced at least one event or elimination) and False
otherwise. The solver's main loop tries rules in order from cheapest to
most expensive, and restarts from the top on the first success.

Rules mutate `state` directly: they eliminate candidates (recording the
causes in state.eliminated_by), narrow cage_combinations, and append
CellResolved events to the trace. v1 only emits cell_resolved events.

Cause construction is per-rule. The pattern is:
  - The rule builds its own event's causes from the structural
    constraints it consulted (CauseHouse, CauseCage).
  - When the rule eliminates a digit from a cell, it records the
    elimination's causes in state.eliminated_by. Later rules read
    these directly when building causes for naked-single resolutions.

This keeps causes sound even for internal eliminations that don't
produce a v1 event.
"""

from __future__ import annotations

from .puzzle import (
    Cage,
    Cell,
    Puzzle,
    SolveState,
    all_houses,
    box_cells,
    col_cells,
    row_cells,
)
from .trace import (
    Cause,
    CauseCage,
    CauseCell,
    CauseHouse,
    CellResolved,
    Trace,
)


# ----------------------------------------------------------------------------
# Cage membership cache
# ----------------------------------------------------------------------------

_cage_membership: dict[Cell, int] = {}


def initialize(puzzle: Puzzle) -> None:
    _cage_membership.clear()
    for cage in puzzle.cages:
        for cell in cage.cells:
            _cage_membership[cell] = cage.id


def _cage_id_of(cell: Cell) -> int:
    return _cage_membership[cell]


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _peers(cell: Cell, cage_id: int) -> set[Cell]:
    """Every cell that shares a row, column, box, or cage with `cell`."""
    r, c = cell
    peers: set[Cell] = set()
    peers.update(row_cells(r))
    peers.update(col_cells(c))
    box = 3 * (r // 3) + (c // 3)
    peers.update(box_cells(box))
    for other_cell, other_cage in _cage_membership.items():
        if other_cage == cage_id:
            peers.add(other_cell)
    peers.discard(cell)
    return peers


def _digits_in_combinations(combos: set[frozenset[int]]) -> set[int]:
    out: set[int] = set()
    for combo in combos:
        out.update(combo)
    return out


def _digits_required(combos: set[frozenset[int]]) -> set[int]:
    if not combos:
        return set()
    required = set(next(iter(combos)))
    for combo in combos:
        required &= combo
    return required


def _house_cells(house_type: str, index: int) -> list[Cell]:
    if house_type == "row":
        return list(row_cells(index))
    if house_type == "col":
        return list(col_cells(index))
    return list(box_cells(index))


def _resolve(
    state: SolveState,
    trace: Trace,
    cell: Cell,
    digit: int,
    inference_type: str,
    causes: tuple[Cause, ...],
) -> None:
    """Mark a cell as resolved, emit the event, and propagate the digit
    to peers (recording the resolution as the cause of each peer
    elimination).
    """
    r, c = cell
    step = trace.next_step()

    other_digits = state.candidates[r][c] - {digit}
    state.candidates[r][c] = {digit}
    for d in other_digits:
        state.eliminated_by[(r, c, d)] = causes

    trace.append(CellResolved(
        step=step,
        cell=cell,
        digit=digit,
        inference_type=inference_type,
        causes=causes,
    ))

    propagation_cause = (CauseCell(cell=cell, step=step),)
    cage_id = _cage_id_of(cell)
    for peer in _peers(cell, cage_id):
        pr, pc = peer
        if digit in state.candidates[pr][pc]:
            state.candidates[pr][pc].discard(digit)
            state.eliminated_by[(pr, pc, digit)] = propagation_cause


def _collect_causes(
    state: SolveState,
    cell: Cell,
    surviving_digit: int,
) -> tuple[Cause, ...]:
    """Union the causes that eliminated every other digit from this cell.
    Deduplicates while preserving a stable order.
    """
    r, c = cell
    seen: set = set()
    out: list[Cause] = []
    for d in range(1, 10):
        if d == surviving_digit:
            continue
        causes = state.eliminated_by.get((r, c, d), ())
        for cause in causes:
            if cause not in seen:
                seen.add(cause)
                out.append(cause)
    return tuple(out)


def _already_resolved(trace: Trace, cell: Cell) -> bool:
    for event in trace.events:
        if isinstance(event, CellResolved) and event.cell == cell:
            return True
    return False


# ----------------------------------------------------------------------------
# Rule 1: Cage initial pruning
# ----------------------------------------------------------------------------

def cage_initial_pruning(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for cage in puzzle.cages:
        valid_digits = _digits_in_combinations(state.cage_combinations[cage.id])
        for cell in cage.cells:
            r, c = cell
            if state.is_resolved(cell):
                continue
            invalid = state.candidates[r][c] - valid_digits
            if not invalid:
                continue
            cage_cause = (CauseCage(cage_id=cage.id),)
            for d in invalid:
                state.candidates[r][c].discard(d)
                state.eliminated_by[(r, c, d)] = cage_cause
            if len(state.candidates[r][c]) == 1:
                digit = next(iter(state.candidates[r][c]))
                _resolve(
                    state, trace, cell, digit,
                    "cage_initial_pruning", cage_cause,
                )
            return True
    return False


# ----------------------------------------------------------------------------
# Rule 2: Naked single
# ----------------------------------------------------------------------------

def naked_single(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for r in range(9):
        for c in range(9):
            cell = (r, c)
            cands = state.candidates[r][c]
            if len(cands) != 1:
                continue
            if _already_resolved(trace, cell):
                continue
            digit = next(iter(cands))
            has_eliminations = any(
                (r, c, d) in state.eliminated_by
                for d in range(1, 10) if d != digit
            )
            if not has_eliminations:
                continue
            causes = _collect_causes(state, cell, digit)
            _resolve(state, trace, cell, digit, "naked_single", causes)
            return True
    return False


# ----------------------------------------------------------------------------
# Rule 3: Hidden single in house
# ----------------------------------------------------------------------------

def hidden_single_house(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for house_type, index, cells in all_houses():
        for digit in range(1, 10):
            candidates_for_digit = [
                cell for cell in cells
                if digit in state.candidates[cell[0]][cell[1]]
            ]
            if len(candidates_for_digit) != 1:
                continue
            target = candidates_for_digit[0]
            if state.is_resolved(target):
                continue
            cause_set: set[Cause] = set()
            cause_list: list[Cause] = [
                CauseHouse(house_type=house_type, index=index)
            ]
            cause_set.add(cause_list[0])
            for cell in cells:
                if cell == target:
                    continue
                r, c = cell
                for cause in state.eliminated_by.get((r, c, digit), ()):
                    if cause not in cause_set:
                        cause_set.add(cause)
                        cause_list.append(cause)
            _resolve(
                state, trace, target, digit,
                "hidden_single_house", tuple(cause_list),
            )
            return True
    return False


# ----------------------------------------------------------------------------
# Rule 4: Hidden single in cage
# ----------------------------------------------------------------------------

def hidden_single_cage(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for cage in puzzle.cages:
        # A hidden-single-in-cage is only sound when the digit MUST appear
        # in the cage — i.e., it's present in every valid combination. If
        # the digit is present in only some combinations, it's optional,
        # and we can't conclude the only-candidate cell must hold it (the
        # cage could choose a combination that excludes the digit).
        required = _digits_required(state.cage_combinations[cage.id])
        for digit in required:
            candidates_for_digit = [
                cell for cell in cage.cells
                if digit in state.candidates[cell[0]][cell[1]]
            ]
            if len(candidates_for_digit) != 1:
                continue
            target = candidates_for_digit[0]
            if state.is_resolved(target):
                continue
            cause_set: set[Cause] = set()
            cause_list: list[Cause] = [CauseCage(cage_id=cage.id)]
            cause_set.add(cause_list[0])
            for cell in cage.cells:
                if cell == target:
                    continue
                r, c = cell
                for cause in state.eliminated_by.get((r, c, digit), ()):
                    if cause not in cause_set:
                        cause_set.add(cause)
                        cause_list.append(cause)
            _resolve(
                state, trace, target, digit,
                "hidden_single_cage", tuple(cause_list),
            )
            return True
    return False


# ----------------------------------------------------------------------------
# Rule 5: Cage combination elimination
# ----------------------------------------------------------------------------

def cage_combination_elimination(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for cage in puzzle.cages:
        feasible: set[frozenset[int]] = set()
        for combo in state.cage_combinations[cage.id]:
            if _combination_feasible(combo, cage.cells, state):
                feasible.add(combo)
        if feasible == state.cage_combinations[cage.id]:
            continue
        state.cage_combinations[cage.id] = feasible
        valid_digits = _digits_in_combinations(feasible)

        cause_list: list[Cause] = [CauseCage(cage_id=cage.id)]
        for other_cell in cage.cells:
            if state.is_resolved(other_cell):
                for event in trace.events:
                    if (isinstance(event, CellResolved)
                            and event.cell == other_cell):
                        cause_list.append(CauseCell(
                            cell=other_cell, step=event.step
                        ))
                        break
        narrowing_causes = tuple(cause_list)

        for cell in cage.cells:
            r, c = cell
            if state.is_resolved(cell):
                continue
            invalid = state.candidates[r][c] - valid_digits
            if not invalid:
                continue
            for d in invalid:
                state.candidates[r][c].discard(d)
                state.eliminated_by[(r, c, d)] = narrowing_causes
            if len(state.candidates[r][c]) == 1:
                digit = next(iter(state.candidates[r][c]))
                _resolve(
                    state, trace, cell, digit,
                    "cage_combination_elimination", narrowing_causes,
                )
            return True
    return False


def _combination_feasible(
    combo: frozenset[int],
    cells: tuple[Cell, ...],
    state: SolveState,
) -> bool:
    return _match(list(combo), cells, 0, set(), state)


def _match(
    digits: list[int],
    cells: tuple[Cell, ...],
    i: int,
    used_cells: set[int],
    state: SolveState,
) -> bool:
    if i == len(digits):
        return True
    d = digits[i]
    for j, cell in enumerate(cells):
        if j in used_cells:
            continue
        r, c = cell
        if d in state.candidates[r][c]:
            used_cells.add(j)
            if _match(digits, cells, i + 1, used_cells, state):
                return True
            used_cells.discard(j)
    return False


# ----------------------------------------------------------------------------
# Rule 6: Locked cage elimination
# ----------------------------------------------------------------------------

def locked_cage_elimination(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for cage in puzzle.cages:
        required = _digits_required(state.cage_combinations[cage.id])
        if not required:
            continue
        containing_houses = _houses_containing_cage(cage)
        if not containing_houses:
            continue
        for digit in required:
            for house_type, index in containing_houses:
                house_cells = _house_cells(house_type, index)
                lock_causes = (
                    CauseCage(cage_id=cage.id),
                    CauseHouse(house_type=house_type, index=index),
                )
                eliminated_any = False
                resolved_cell: Cell | None = None
                for cell in house_cells:
                    if cell in cage.cells:
                        continue
                    r, c = cell
                    if digit in state.candidates[r][c]:
                        state.candidates[r][c].discard(digit)
                        state.eliminated_by[(r, c, digit)] = lock_causes
                        eliminated_any = True
                        if len(state.candidates[r][c]) == 1:
                            resolved_cell = cell
                if not eliminated_any:
                    continue
                if resolved_cell is not None:
                    r, c = resolved_cell
                    new_digit = next(iter(state.candidates[r][c]))
                    _resolve(
                        state, trace, resolved_cell, new_digit,
                        "locked_cage_elimination", lock_causes,
                    )
                return True
    return False


def _houses_containing_cage(cage: Cage) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    rows = {r for r, _ in cage.cells}
    cols = {c for _, c in cage.cells}
    boxes = {3 * (r // 3) + (c // 3) for r, c in cage.cells}
    if len(rows) == 1:
        out.append(("row", next(iter(rows))))
    if len(cols) == 1:
        out.append(("col", next(iter(cols))))
    if len(boxes) == 1:
        out.append(("box", next(iter(boxes))))
    return out


# ----------------------------------------------------------------------------
# Rule 7: 45 rule (basic)
# ----------------------------------------------------------------------------

def forty_five_basic(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for house_type, index, cells in all_houses():
        unresolved = [cell for cell in cells if not state.is_resolved(cell)]
        if len(unresolved) != 1:
            continue
        target = unresolved[0]
        resolved_sum = sum(
            state.digit_at(cell) for cell in cells if state.is_resolved(cell)
        )
        digit = 45 - resolved_sum
        if not (1 <= digit <= 9):
            continue
        r, c = target
        if digit not in state.candidates[r][c]:
            continue
        cause_list: list[Cause] = [CauseHouse(house_type=house_type, index=index)]
        for cell in cells:
            if cell != target and state.is_resolved(cell):
                for event in trace.events:
                    if (isinstance(event, CellResolved)
                            and event.cell == cell):
                        cause_list.append(CauseCell(
                            cell=cell, step=event.step
                        ))
                        break
        _resolve(
            state, trace, target, digit,
            "forty_five_basic", tuple(cause_list),
        )
        return True
    return False


# ----------------------------------------------------------------------------
# Rule 8: Innies and outies
# ----------------------------------------------------------------------------

def innies_and_outies(
    state: SolveState, puzzle: Puzzle, trace: Trace
) -> bool:
    for house_type, index, cells in all_houses():
        house_set = set(cells)
        fully_in: list[Cage] = []
        crossing: list[tuple[Cage, list[Cell], list[Cell]]] = []
        seen_cage_ids = set()
        for cell in cells:
            cid = _cage_id_of(cell)
            if cid in seen_cage_ids:
                continue
            seen_cage_ids.add(cid)
            cage = next(c for c in puzzle.cages if c.id == cid)
            in_cells = [cc for cc in cage.cells if cc in house_set]
            out_cells = [cc for cc in cage.cells if cc not in house_set]
            if not out_cells:
                fully_in.append(cage)
            else:
                crossing.append((cage, in_cells, out_cells))

        if len(crossing) != 1:
            continue
        cage, in_cells, out_cells = crossing[0]
        sum_in = sum(c.sum for c in fully_in)

        if len(out_cells) == 1:
            outie_value = cage.sum - (45 - sum_in)
            outie = out_cells[0]
            if not (1 <= outie_value <= 9):
                continue
            r, c = outie
            if state.is_resolved(outie):
                continue
            if outie_value not in state.candidates[r][c]:
                continue
            causes_list = [
                CauseHouse(house_type=house_type, index=index),
                CauseCage(cage_id=cage.id),
            ] + [CauseCage(cage_id=c.id) for c in fully_in]
            _resolve(
                state, trace, outie, outie_value,
                "forty_five_outie", tuple(causes_list),
            )
            return True
        if len(in_cells) == 1:
            innie_value = 45 - sum_in
            innie = in_cells[0]
            if not (1 <= innie_value <= 9):
                continue
            r, c = innie
            if state.is_resolved(innie):
                continue
            if innie_value not in state.candidates[r][c]:
                continue
            causes_list = [
                CauseHouse(house_type=house_type, index=index),
                CauseCage(cage_id=cage.id),
            ] + [CauseCage(cage_id=c.id) for c in fully_in]
            _resolve(
                state, trace, innie, innie_value,
                "forty_five_innie", tuple(causes_list),
            )
            return True
    return False


# ----------------------------------------------------------------------------
# Rule registry
# ----------------------------------------------------------------------------

ALL_RULES = [
    # Elimination-only rules first — they narrow candidate sets but don't
    # resolve cells. Resolution rules below depend on candidate sets being
    # sound (over-approximations are fine; under-approximations are not).
    # Naked-single in particular fires when one candidate remains: if a
    # later elimination-only rule would have removed the surviving digit,
    # the resolution is unsound. So all such rules must run first.
    cage_initial_pruning,
    locked_cage_elimination,
    cage_combination_elimination,

    # Resolution rules — these place a digit in a cell. Ordered roughly
    # by inference strength: hidden-single in cage/house is more informative
    # than naked-single (it identifies which specific cell holds a digit,
    # rather than which digit a cell must hold). Forty-five and innies/outies
    # use arithmetic across a whole house, the strongest form.
    hidden_single_cage,
    hidden_single_house,
    naked_single,
    forty_five_basic,
    innies_and_outies,
]
