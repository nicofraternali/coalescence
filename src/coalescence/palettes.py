"""
Color palettes for generative art projects.

This module is the single source of truth for color themes used across
all sketches. Each theme is a dictionary mapping semantic role names
(e.g. "bg", "line") to hex color strings.

Themes are organized into two layers:

  - SHARED_PALETTES: a curated list of colors used as the base for each
    theme, plus optional auxiliary colors for projects that need more
    than three (e.g. the pendulum's region-coloring solver).

  - THEMES: a dictionary keyed by theme name, where each value is a dict
    of role -> hex color. Role names are NOT standardized across project
    types — the tiling sketches use {bg, line, fill} or {bg, wire, deep},
    while the pendulum uses {bg, arm, trace, text, palette}.

When adding a new theme, define it once here and import it in your sketch.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tiling-style themes (used by tiling_squares and tiling_holes sketches).
#
# Tiling sketches use three roles:
#   - bg:   background color
#   - line: foreground stroke / wire color
#   - fill: secondary color used for diamonds (squares variant) or for
#           the deep "trench" color (holes variant)
#
# To support both variants from a single theme definition, we use a
# generic key "accent" and let each sketch map it to its own role.
# ---------------------------------------------------------------------------

TILING_THEMES: dict[str, dict[str, str]] = {
    "ORIGINAL": {"bg": "#E37E52", "line": "#26625B", "accent": "#F2C14E", "deep": "#000000"},
    "ATI_1":    {"bg": "#F0EADC", "line": "#29617A", "accent": "#BC4A31", "deep": "#BC4A31"},

    "BLUEPRINT": {"bg": "#2B2D42", "line": "#EDF2F4", "accent": "#8D99AE", "deep": "#0B0C15"},
    "BAUHAUS":   {"bg": "#F0F0F0", "line": "#111111", "accent": "#D93025", "deep": "#D93025"},
    "GREYSCALE": {"bg": "#121212", "line": "#FFFFFF", "accent": "#444444", "deep": "#000000"},
    "SWISS":     {"bg": "#E5E5E5", "line": "#000000", "accent": "#FF4400", "deep": "#FF4400"},

    "MATRIX":    {"bg": "#0D0D0D", "line": "#00FF41", "accent": "#003B00", "deep": "#003B00"},
    "CYBERPUNK": {"bg": "#0B1026", "line": "#00F3FF", "accent": "#FF0099", "deep": "#FF0099"},
    "TERMINAL":  {"bg": "#1E1E1E", "line": "#FFB000", "accent": "#593E00", "deep": "#593E00"},
    "VAPORWAVE": {"bg": "#2E2157", "line": "#00F0FF", "accent": "#FF0055", "deep": "#FF0055"},

    "FOREST":  {"bg": "#1A2F1A", "line": "#8FBC8F", "accent": "#D2B48C", "deep": "#D2B48C"},
    "OCEANIC": {"bg": "#003049", "line": "#669BBC", "accent": "#FDF0D5", "deep": "#FDF0D5"},
    "DESERT":  {"bg": "#A47148", "line": "#FAE1DF", "accent": "#6F4E37", "deep": "#6F4E37"},

    "CANDY":     {"bg": "#FFCAD4", "line": "#9D8189", "accent": "#FFF0F5", "deep": "#540B0E"},
    "SUNSET":    {"bg": "#2D1E2F", "line": "#FC2F00", "accent": "#FFD400", "deep": "#FFD400"},
    "RETRO_POP": {"bg": "#4ECDC4", "line": "#FFE66D", "accent": "#FF6B6B", "deep": "#FF6B6B"},
    "ROYAL":     {"bg": "#191970", "line": "#FFD700", "accent": "#C0C0C0", "deep": "#C0C0C0"},
}


# ---------------------------------------------------------------------------
# Pendulum-style themes (used by the double pendulum sketch).
#
# Pendulum sketches use:
#   - bg:      background color
#   - arm:     pendulum arm color (overlay during simulation)
#   - trace:   trace line color
#   - text:    metadata text color
#   - palette: list of region-fill colors (>= 6 distinct colors recommended
#              so the greedy graph-coloring solver has options)
# ---------------------------------------------------------------------------

PENDULUM_THEMES: dict[str, dict[str, str | list[str]]] = {
    "ORIGINAL": {
        "bg": "#E37E52", "arm": "#26625B", "trace": "#FFFFFF", "text": "#000000",
        "palette": ["#F2C14E", "#26625B", "#F0EADC", "#BC4A31", "#5A3A31", "#335C67", "#FFFFFF"],
    },
    "JAPAN": {
        "bg": "#FDFFFC", "arm": "#003049", "trace": "#D62828", "text": "#003049",
        "palette": ["#003049", "#F77F00", "#FCBF49", "#EAE2B7", "#2A9D8F", "#264653", "#8D99AE"],
    },
    "NEON": {
        "bg": "#120024", "arm": "#00FFC2", "trace": "#FF00AA", "text": "#00FFC2",
        "palette": ["#00FFC2", "#FF00AA", "#FAFF00", "#BD00FF", "#00F0FF", "#FF3D00", "#76FF03"],
    },
    "TROPIC": {
        "bg": "#FF9F1C", "arm": "#FFFFFF", "trace": "#2EC4B6", "text": "#FFFFFF",
        "palette": ["#2EC4B6", "#E71D36", "#FDFFFC", "#011627", "#B9E769", "#4ECDC4", "#F15BB5"],
    },
}


# ---------------------------------------------------------------------------
# Convenience accessors.
# ---------------------------------------------------------------------------

def get_tiling_theme(name: str) -> dict[str, str]:
    """Return the tiling theme with the given name. Raises KeyError if missing."""
    return TILING_THEMES[name]


def get_pendulum_theme(name: str) -> dict[str, str | list[str]]:
    """Return the pendulum theme with the given name. Raises KeyError if missing."""
    return PENDULUM_THEMES[name]


def list_tiling_themes() -> list[str]:
    """Return all tiling theme names, sorted."""
    return sorted(TILING_THEMES.keys())


def list_pendulum_themes() -> list[str]:
    """Return all pendulum theme names, sorted."""
    return sorted(PENDULUM_THEMES.keys())