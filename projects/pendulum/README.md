# pendulum

A double pendulum in real units (meters, kilograms, seconds, g = 9.81),
integrated with RK4 and optionally slowed by friction at both joints. Its
lower bob traces a chaotic path for `--duration` seconds of simulated time.
The trace is then treated as a wall structure: the enclosed regions are
labeled (via `scipy.ndimage.label`), adjacency is computed (via numpy roll
operations), and regions are colored with a greedy graph-coloring algorithm.

The art view keeps only closed shapes. Loose strands, such as the fall from
the release point, the end of the run, or dead ends in between, are removed
and their gap is filled with the surrounding color. The trace view on the
left keeps the full path.

## Physics

The model lives in `pendulum_physics.py`, independent of py5:

- Equations of motion from the Lagrangian, solved in mass-matrix form.
- Joint friction as generalized forces: torque `-c·ω₁` at the pivot and
  `-c·(ω₂-ω₁)` at the elbow. `c = 0` is the ideal, energy-conserving pendulum.
- Classic RK4 with a 2 ms step. Undamped, energy drifts by about one part
  per million over 15 s.
- Each piece samples arm lengths of 0.5–2 m, masses of 1–5 kg, and release
  angles anywhere on the circle, released from rest.

Tests (`uv run pytest`) check energy conservation, energy loss with friction,
the small-swing period against `2π√(l/g)`, and determinism.

## A note on reproducibility and chaos

Double pendulums are paradigmatic chaotic systems: sensitivity to initial
conditions is the textbook property. But the integrator used here is
deterministic: same seed + same code + same Python version = bit-identical
output, every time. Reproducibility via `--seed` works perfectly.

The chaotic sensitivity only matters if you tried to reconstruct a piece
by typing initial angles back in from the metadata sidecar, at which point
floating-point precision loss would diverge the trajectory. The seed-based
contract avoids this entirely.

Pieces made before October 2026 used a different integrator (Euler, pixel
units, per-step damping), so their seeds produce different pieces now. They
stay reproducible from the commit recorded in their metadata.

## Run

Interactive (random seed each session):

```powershell
uv run python projects/pendulum/sketch.py
```

Reproduce a specific piece:

```powershell
uv run python projects/pendulum/sketch.py --seed 4823 --theme JAPAN
```

Longer run, with joint friction:

```powershell
uv run python projects/pendulum/sketch.py --seed 4823 --duration 15 --damping 0.5
```

## Parameters

- `--seed` — seed for deterministic reproduction.
- `--theme` — color theme. See `coalescence.palettes.PENDULUM_THEMES`.
- `--duration` — seconds of simulated motion. Default 5.
- `--damping` — joint friction coefficient `c` in N·m·s/rad. Default 0 (undamped).
- `--save-and-exit` — with `--seed`, save both views and exit when done.

`STEPS_PER_FRAME` in `sketch.py` sets playback speed only (8 ≈ real time).

## Keys

- `r` — new random composition (new seed)
- `space` — new random theme (recolors once finished, otherwise resets)
- `s` — save current piece (saves both trace and art views)

## Each save produces two files

- `pendulum_<seed>_<timestamp>_trace.png` — the raw simulation view
- `pendulum_<seed>_<timestamp>_art.png` — the painted topology view

Plus matching `.json` sidecars for each, recording the physical parameters,
initial and final energy, and quality metrics.

## Quality metrics

Each saved piece records region statistics in its sidecar:
`n_components_total`, `n_regions_interior`, `n_regions_outside`,
`region_size_mean`, `region_size_std`, `region_size_min`, `region_size_max`,
plus `trace_segments_total` and `trace_segments_dropped` (loose strands).
Over time these can inform which durations and damping values work for
which physical setups.
