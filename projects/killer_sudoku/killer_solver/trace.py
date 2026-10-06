"""
trace.py — Event recording and serialization for the killer sudoku solver.

The trace is the artistic substrate of this whole project: it's what the
renderer consumes. So we care about three things here:

  1. Faithful recording — every event the solver emits is preserved
     verbatim, with its causes intact.
  2. Forward compatibility — the schema accommodates v1's cell_resolved
     events plus v2's candidate_eliminated events plus v3's
     inference_applied events, even though v1 only uses the first.
  3. Self-contained output — a trace file includes the puzzle definition
     it was solved from, so the renderer (or anyone reading the trace
     later) doesn't need the original YAML.

This module is purely declarative. It defines types and provides a
container that serializes to JSON. No solving logic, no validation of
causal relationships — those are the solver's concern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


# A cell coordinate, matching puzzle.py.
Cell = tuple[int, int]


# ----------------------------------------------------------------------------
# Cause types
# ----------------------------------------------------------------------------
#
# A cause is a typed reference to something earlier in the trace (or to a
# structural feature of the puzzle, like a cage or a house). Every event
# carries a list of causes that explain why it fired.
#
# The "proximate causes" design decision (from earlier) means each cause
# is what the firing rule *directly* consulted, not the full transitive
# chain back to the givens. Transitive reconstruction happens at render
# time by walking the cause graph backward.

@dataclass(frozen=True)
class CauseGiven:
    """A digit given in the initial puzzle."""
    type: Literal["given"] = field(default="given", init=False)
    cell: Cell = (0, 0)


@dataclass(frozen=True)
class CauseCell:
    """A previously resolved cell."""
    cell: Cell
    step: int
    type: Literal["cell"] = field(default="cell", init=False)


@dataclass(frozen=True)
class CauseCage:
    """A cage's constraint (sum + membership + current combinations)."""
    cage_id: int
    type: Literal["cage"] = field(default="cage", init=False)


@dataclass(frozen=True)
class CauseHouse:
    """A row/column/box constraint (distinct digits 1..9, sums to 45)."""
    house_type: Literal["row", "col", "box"]
    index: int
    type: Literal["house"] = field(default="house", init=False)


@dataclass(frozen=True)
class CauseInference:
    """(v3) A previously applied inference event."""
    step: int
    type: Literal["inference"] = field(default="inference", init=False)


# Union type for all causes. Used in event signatures.
Cause = CauseGiven | CauseCell | CauseCage | CauseHouse | CauseInference


# ----------------------------------------------------------------------------
# Inference types
# ----------------------------------------------------------------------------
#
# Each cell_resolved or candidate_eliminated event names the rule that
# fired. Renderers can use this to color-code edges, filter the trace,
# produce statistics. The set below matches the v1 rule set; v1.5 and
# later will extend it.

InferenceType = Literal[
    "given",
    "cage_initial_pruning",
    "naked_single",
    "hidden_single_house",
    "hidden_single_cage",
    "cage_combination_elimination",
    "locked_cage_elimination",
    "forty_five_basic",
    "forty_five_innie",
    "forty_five_outie",
]


# ----------------------------------------------------------------------------
# Event types
# ----------------------------------------------------------------------------
#
# Three event types span all three versions:
#
#   - cell_resolved        — v1, v2, v3
#   - candidate_eliminated — v2, v3
#   - inference_applied    — v3
#
# v1 only emits cell_resolved. The other two are declared here so the
# schema is stable across versions.

@dataclass(frozen=True)
class CellResolved:
    """A cell was determined to hold a specific digit."""
    step: int
    cell: Cell
    digit: int
    inference_type: InferenceType
    causes: tuple[Cause, ...]
    type: Literal["cell_resolved"] = field(default="cell_resolved", init=False)


@dataclass(frozen=True)
class CandidateEliminated:
    """A digit was removed from a cell's candidate set (not a resolution)."""
    step: int
    cell: Cell
    digit: int
    inference_type: InferenceType
    causes: tuple[Cause, ...]
    type: Literal["candidate_eliminated"] = field(
        default="candidate_eliminated", init=False
    )


@dataclass(frozen=True)
class InferenceApplied:
    """(v3) A rule fired, producing one or more eliminations/resolutions.

    The downstream events (cell_resolved, candidate_eliminated) reference
    this event's step in their causes. Useful for grouping fine-grained
    eliminations under their generating inference.
    """
    step: int
    inference_type: InferenceType
    causes: tuple[Cause, ...]
    description: str = ""   # optional human-readable summary
    type: Literal["inference_applied"] = field(
        default="inference_applied", init=False
    )


Event = CellResolved | CandidateEliminated | InferenceApplied


# ----------------------------------------------------------------------------
# Trace container
# ----------------------------------------------------------------------------

@dataclass
class Trace:
    """The full record of a solve: puzzle metadata + ordered events.

    The trace is built incrementally by the solver. Step numbers are
    assigned automatically via `next_step()` — the solver doesn't track
    its own step counter.
    """
    puzzle_id: str
    puzzle_data: dict[str, Any]   # the original puzzle as a dict (for self-containment)
    seed: int = 0                  # for renderer-side stochastic choices
    version: str = "1.0"           # trace schema version
    git_commit: str | None = None  # solver code that produced the trace (set by solve.py)
    git_dirty: bool | None = None
    events: list[Event] = field(default_factory=list)

    _step_counter: int = field(default=0, init=False, repr=False)

    def next_step(self) -> int:
        """Allocate and return the next step number. Monotonically increasing."""
        s = self._step_counter
        self._step_counter += 1
        return s

    def append(self, event: Event) -> None:
        """Add an event to the trace.

        We don't validate causal references here — the solver is
        responsible for producing well-formed events. Trace is passive.
        """
        self.events.append(event)

    # -------- Serialization --------

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire trace to a JSON-serializable dict."""
        return {
            "version": self.version,
            "puzzle_id": self.puzzle_id,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "puzzle": self.puzzle_data,
            "events": [_event_to_dict(e) for e in self.events],
        }

    def save(self, path: str | Path) -> None:
        """Write the trace to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)


# ----------------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------------
#
# dataclasses.asdict almost does what we want, but it converts tuples to
# lists indiscriminately. For cells we want lists [r, c] in JSON (this is
# how they appear in the puzzle YAML), so the conversion is fine — but we
# still need a custom path for nested causes, since asdict handles those
# transparently and we want to preserve the `type` discriminator.

def _event_to_dict(event: Event) -> dict[str, Any]:
    """Convert an event to a JSON-friendly dict, preserving type tags."""
    d = asdict(event)
    # asdict already includes the `type` field because it's a dataclass field.
    # Causes are also handled recursively. We just need to ensure cells become
    # lists (asdict already does this since tuples convert to lists in JSON).
    return d


def _json_default(obj: Any) -> Any:
    """Fallback for json.dump when it encounters something unexpected.

    Tuples are already handled by json (converted to lists). Frozensets
    and sets aren't — they appear in causes for v3 inferences. Convert
    to sorted lists for stable output.
    """
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ----------------------------------------------------------------------------
# Loading (for the renderer)
# ----------------------------------------------------------------------------
#
# The renderer needs to read traces back from disk. We don't reconstruct
# the typed dataclasses on load — the renderer can work with raw dicts,
# which are simpler and more flexible. If a future tool needs the typed
# form, it can convert.

def load(path: str | Path) -> dict[str, Any]:
    """Load a trace from disk and return it as a raw dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
