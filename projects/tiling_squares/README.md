# tiling_squares

Truchet-style tiling: a 2^L by 2^L grid of diagonal lines, with diamonds
drawn where a 2x2 neighborhood matches the "1001" pattern.

## Run

Interactive (random seed each session):

```powershell
uv run python projects/tiling_squares/sketch.py
```

Deterministic (reproduce a specific piece from a sidecar):

```powershell
uv run python projects/tiling_squares/sketch.py --seed 4823 --theme JAPAN
```

## Keys

- `r` — new random structure, keep theme
- `space` — new random theme, keep structure
- `s` — save current piece (prints seed/theme to console)

## Parameters

- `--L` — grid resolution exponent. Default 4 (16x16 grid). Higher values
  produce finer detail at the same canvas size.
- `--theme` — color theme. See `coalescence.palettes.TILING_THEMES`.
- `--seed` — seed for deterministic reproduction. If given, the sketch
  renders, saves once, and exits (no interactive window).