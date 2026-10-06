"""
Saving artwork with embedded metadata and JSON sidecar.

The save_artwork function is the only supported way to write a finished
piece. It does three things atomically:

  1. Saves the high-resolution PNG to the project's `output/` folder.
  2. Embeds metadata (seed, theme, parameters, git commit, timestamp,
     library versions) as PNG text chunks inside the file itself.
  3. Writes a JSON sidecar with the same metadata next to the PNG.

The redundancy is deliberate. PNG metadata survives file moves, copies,
and separation from the sidecar. The JSON sidecar is trivially scriptable
for downstream tools like the gallery builder.

Usage from a sketch:

    from coalescence.io import save_artwork

    filepath = save_artwork(
        pg,                         # the py5 graphics buffer to save
        project_name="pendulum",
        seed=4823,
        theme_name="JAPAN",
        params={"m1": 24.3, "m2": 31.7, "r1": 220.5, "r2": 185.2,
                "a1_init": 2.14, "a2_init": 4.78, "max_points": 750},
    )
    print(f"Saved: {filepath}")

The function returns the path of the saved PNG. The sidecar is at the
same path with a `.json` extension instead of `.png`.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin


# Prefix for PNG text-chunk keys. Pieces saved before the repo was renamed
# to `coalescence` use the legacy `genart:` prefix; reading accepts both.
METADATA_PREFIX = "coalescence:"
LEGACY_METADATA_PREFIXES = ("genart:",)


# ---------------------------------------------------------------------------
# Path resolution.
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    """Walk upwards from `start` until a `.git` directory is found."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find repo root (no .git directory above {start}). "
        "save_artwork() must be called from a sketch inside the coalescence repo."
    )


def _project_output_dir(project_name: str, repo_root: Path) -> Path:
    """Return the output/ directory for a project, creating it if needed."""
    output_dir = repo_root / "projects" / project_name / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Metadata gathering.
# ---------------------------------------------------------------------------

def _git_commit_hash(repo_root: Path) -> str:
    """Return the short Git commit hash, or 'unknown' if anything goes wrong."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _git_is_dirty(repo_root: Path) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _library_versions() -> dict[str, str]:
    """Return versions of libraries that affect rendering output."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for lib_name, import_name in [("py5", "py5"), ("numpy", "numpy"), ("pillow", "PIL")]:
        try:
            mod = __import__(import_name)
            versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def save_artwork(
    pg: Any,
    project_name: str,
    seed: int,
    theme_name: str,
    params: dict[str, Any] | None = None,
    suffix: str | None = None,
) -> Path:
    """
    Save a py5 graphics buffer to the project's output folder with metadata.

    Parameters
    ----------
    pg : py5 graphics buffer (Py5Graphics)
        The off-screen buffer holding the finished artwork.
    project_name : str
        Name of the project (e.g., "pendulum"). Determines the output folder.
    seed : int
        The seed used for this composition. Recorded in metadata.
    theme_name : str
        Name of the color theme (e.g., "JAPAN"). Recorded in metadata.
    params : dict, optional
        Project-specific parameters (masses, radii, grid size, etc.).
        Will be JSON-serialized, so values must be JSON-compatible
        (numbers, strings, lists, dicts, booleans, None).
    suffix : str, optional
        Extra string appended to the filename before the extension.
        Useful for distinguishing variants (e.g., "trace" vs "art").

    Returns
    -------
    Path
        The absolute path to the saved PNG. The sidecar is at the same
        path with `.json` extension.
    """
    if params is None:
        params = {}

    # Locate the repo and the project's output folder.
    # We resolve from this file's location, walking up to find .git.
    repo_root = _find_repo_root(Path(__file__))
    output_dir = _project_output_dir(project_name, repo_root)

    # Build a filename: <project>_<seed>_<timestamp>[_<suffix>].png
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [project_name, str(seed), timestamp]
    if suffix:
        parts.append(suffix)
    base_name = "_".join(parts)
    png_path = output_dir / f"{base_name}.png"
    json_path = output_dir / f"{base_name}.json"

    # Gather metadata.
    metadata = {
        "project": project_name,
        "seed": seed,
        "theme": theme_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(repo_root),
        "git_dirty": _git_is_dirty(repo_root),
        "library_versions": _library_versions(),
        "params": params,
    }

    # Step 1: have py5 save the PNG to a temporary path.
    # py5's pg.save() handles the actual rendering; we then re-open
    # with Pillow to embed metadata.
    temp_path = output_dir / f"{base_name}.tmp.png"
    pg.save(str(temp_path))

    # Step 2: re-open with Pillow and rewrite with embedded metadata.
    with Image.open(temp_path) as img:
        png_info = PngImagePlugin.PngInfo()
        # PNG text chunks must be strings.
        png_info.add_text(f"{METADATA_PREFIX}project", project_name)
        png_info.add_text(f"{METADATA_PREFIX}seed", str(seed))
        png_info.add_text(f"{METADATA_PREFIX}theme", theme_name)
        png_info.add_text(f"{METADATA_PREFIX}timestamp_utc", metadata["timestamp_utc"])
        png_info.add_text(f"{METADATA_PREFIX}git_commit", metadata["git_commit"])
        png_info.add_text(f"{METADATA_PREFIX}metadata_json", json.dumps(metadata))
        img.save(png_path, "PNG", pnginfo=png_info)

    # Step 3: remove the temp file.
    temp_path.unlink()

    # Step 4: write the JSON sidecar.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return png_path


def read_artwork_metadata(png_path: Path | str) -> dict[str, Any]:
    """
    Read embedded metadata from a PNG saved by `save_artwork`.

    Useful for verifying a piece's provenance or reading parameters back
    when no sidecar is available.
    """
    png_path = Path(png_path)
    with Image.open(png_path) as img:
        text_chunks = img.text if hasattr(img, "text") else {}

    for prefix in (METADATA_PREFIX, *LEGACY_METADATA_PREFIXES):
        if f"{prefix}metadata_json" in text_chunks:
            return json.loads(text_chunks[f"{prefix}metadata_json"])

    # Fallback: reconstruct from individual chunks.
    for prefix in (METADATA_PREFIX, *LEGACY_METADATA_PREFIXES):
        chunks = {
            key.removeprefix(prefix): value
            for key, value in text_chunks.items()
            if key.startswith(prefix)
        }
        if chunks:
            return chunks
    return {}
