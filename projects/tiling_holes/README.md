# tiling_holes

Truchet-style tiling rendered as V-cut trenches with Gouraud-shaded depth,
plus a wire mesh on top. Same combinatorial substrate as `tiling_squares`
but with inverted figure-ground.

## Run

Interactive (random seed each session):

```powershell
uv run python projects/tiling_holes/sketch.py
```

Reproduce a specific piece:

```powershell
uv run python projects/tiling_holes/sketch.py --seed 4823 --theme JAPAN
```

## Keys

- `r` — new random structure, keep theme
- `space` — new random theme, keep structure
- `s` — save current piece

## Parameters

- `--L` — grid resolution exponent. Default 4.
- `--theme` — color theme. See `coalescence.palettes.TILING_THEMES`.
- `--seed` — seed for deterministic reproduction.
- `--save-and-exit` — with `--seed`, save once and exit without interaction.