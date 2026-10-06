# coalescence

A monorepo for generative art experiments by [Nico Fraternali](https://github.com/nicofraternali).
Each project is a self-contained py5 sketch; shared utilities live in the `coalescence` package.

## Repository structure

```
coalescence/
├── pyproject.toml          # uv-managed dependencies
├── uv.lock                 # pinned exact versions, committed
├── src/coalescence/        # shared importable package
│   └── palettes.py         # unified color themes
├── projects/               # individual sketches (one folder each)
├── curated/                # hand-picked outputs + metadata sidecars (committed)
├── notebooks/              # exploration notebooks (outputs stripped on commit)
└── scripts/                # tooling: curation, gallery generation
```

## Reproducibility contract

Every curated piece can be regenerated bit-for-bit from three things:

1. **The seed** recorded in the piece's JSON sidecar
2. **The Git commit hash** also recorded in the sidecar — pins the exact code
3. **`uv.lock`** at that commit — pins exact dependency versions

To reproduce a curated piece:

```powershell
git checkout <commit-hash-from-sidecar>
uv sync
uv run python projects/<project-name>/sketch.py --seed <seed-from-sidecar>
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```powershell
git clone https://github.com/nicofraternali/coalescence.git
cd coalescence
uv sync --all-groups
```

## License

All code in this repository is released under the MIT License.
Curated artworks are © Nico Fraternali, all rights reserved.
