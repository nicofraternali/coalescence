"""
screenshot.py — Read a killer sudoku from an app screenshot.

Classic image processing, tuned to one app's look (beige board, dotted
cage outlines drawn inside the cells, sums in each cage's top-left cell,
givens as large digits). No OCR engine and no network: digits are
recognized by comparing them with reference images in reader_templates/.

The reader never guesses. Anything it isn't certain about is returned as
a Question; the caller (read_screenshot.py) asks the user and passes the
answers back in. Pipeline:

  1. Ink mask: pixels darker than INK_THRESHOLD. Lines and digits are far
     darker than any background tint (plain cells, the selected cage's
     teal, the shaded row/column/box), so selection highlighting is
     ignored automatically. Thin grid lines are lighter than the
     threshold and drop out too.
  2. Board: the four thick lines in each direction are the rows/columns
     that are mostly ink. Cells are the 9x9 subdivision between them.
  3. Blobs: connected components of ink, with the thick lines blanked.
     Tiny blobs are the dots of cage outlines; short ones are sum
     digits; tall ones are givens. A blob too tall for a given is a sum
     label touching a given, split by subtracting the best given match.
  4. Cage borders: for each pair of neighboring cells, look for dots
     just inside each cell along their shared edge. Dots on both sides =
     border, on neither = same cage, on one side only = question.
  5. Digits: each glyph, scaled to a standard size, is compared with the
     templates by normalized correlation. Accepted only if the best match
     is near-perfect and clearly beats every other digit; else a question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

INK_THRESHOLD = 170          # brightness; ink is darker
THICK_LINE_FILL = 0.5        # a thick line row/column is at least this fraction ink
THICK_LINE_HALF_WIDTH = 5    # px blanked around each thick line

# Blob classes, as fractions of the cell size s.
DOT_MAX = 0.09               # dots of the cage outlines: both sides below this
SUM_HEIGHT = (0.12, 0.28)    # sum digits
GIVEN_HEIGHT = (0.40, 0.70)  # givens
GIVEN_GLYPH_HEIGHT = 0.56    # measured height of a given digit (about 63 px at s = 112)
LABEL_ZONE = (0.62, 0.40)    # a sum glyph starts within this (x, y) of the cell's top-left

# Cage border detection: dots counted in a strip just inside each cell side.
STRIP_DEPTH = (0.03, 0.13)   # from the cell edge inward, fraction of s (dotted lines sit at ~0.07-0.09)
STRIP_SPAN = (0.45, 0.85)    # along the side: past the sum label corner, short of the far dotted line
DOTS_PRESENT = 40            # px of dots: at least this many = a dotted line (a full strip has ~100)
DOTS_ABSENT = 5              # at most this many = no line; in between = question

# Glyph matching.
GLYPH_SIZE = (32, 20)        # (h, w) every glyph is resized to before comparing
MATCH_ACCEPT = 0.90          # best correlation must reach this...
MATCH_MARGIN = 0.12          # ...and beat the best other digit by this much

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "reader_templates"


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

@dataclass
class Question:
    """Something the reader isn't sure about. `key` identifies the answer."""
    key: str
    kind: str                  # "border", "sum", "given"
    prompt: str
    picture: str = ""          # ASCII rendering of the pixels in doubt
    glyphs: list[np.ndarray] = field(default_factory=list)  # for learning templates


@dataclass
class Reading:
    cages: list[dict]                       # {"id", "sum", "cells"} in reading order
    givens: list[dict]                      # {"cell", "digit"}
    questions: list[Question]
    problems: list[str]                     # structural issues found by validation


# ----------------------------------------------------------------------------
# Image → ink, board, blobs
# ----------------------------------------------------------------------------

def _line_centers(profile: np.ndarray) -> list[float]:
    idx = np.where(profile > THICK_LINE_FILL)[0]
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [float(g.mean()) for g in groups]


def _find_board(ink: np.ndarray) -> tuple[float, float, float]:
    """Return (left, top, cell size) from the four thick lines each way."""
    ys = _line_centers(ink.mean(axis=1))
    if len(ys) != 4:
        raise ValueError(f"Expected 4 thick horizontal lines, found {len(ys)}: is this a board screenshot?")
    cols = ink[int(ys[0]):int(ys[-1]) + 1].mean(axis=0)
    xs = _line_centers(cols)
    if len(xs) != 4:
        raise ValueError(f"Expected 4 thick vertical lines, found {len(xs)}")
    s_x, s_y = (xs[-1] - xs[0]) / 9, (ys[-1] - ys[0]) / 9
    if abs(s_x - s_y) > 2 or max(np.ptp(np.diff(xs)), np.ptp(np.diff(ys))) > 4:
        raise ValueError("Board lines are not evenly spaced; the screenshot may be cropped or distorted")
    return xs[0], ys[0], (s_x + s_y) / 2


@dataclass
class _Blob:
    y0: int
    x0: int
    mask: np.ndarray      # boolean, bbox-sized
    dark: np.ndarray      # darkness 0..1, bbox-sized (for matching)

    @property
    def h(self):
        return self.mask.shape[0]

    @property
    def w(self):
        return self.mask.shape[1]


def _blobs(ink: np.ndarray, dark: np.ndarray, left: float, top: float, s: float):
    """Connected ink components inside the board, thick lines blanked."""
    x0, y0, x1, y1 = int(round(left)), int(round(top)), int(round(left + 9 * s)), int(round(top + 9 * s))
    sub = ink[y0:y1 + 1, x0:x1 + 1].copy()
    for k in (0, 3, 6, 9):
        p = int(round(k * s))
        sub[:, max(0, p - THICK_LINE_HALF_WIDTH):p + THICK_LINE_HALF_WIDTH + 1] = False
        sub[max(0, p - THICK_LINE_HALF_WIDTH):p + THICK_LINE_HALF_WIDTH + 1, :] = False
    labels, _ = ndi.label(sub, structure=np.ones((3, 3), dtype=int))
    blobs = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        m = labels[sl] == i
        d = np.where(m, dark[y0:y1 + 1, x0:x1 + 1][sl], 0.0)
        blobs.append(_Blob(sl[0].start, sl[1].start, m, d))   # board coordinates
    return blobs


# ----------------------------------------------------------------------------
# Glyph matching
# ----------------------------------------------------------------------------

def _normalize(dark: np.ndarray) -> np.ndarray:
    img = Image.fromarray((np.clip(dark, 0, 1) * 255).astype(np.uint8))
    v = np.asarray(img.resize(GLYPH_SIZE[::-1], Image.BILINEAR), dtype=np.float64)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_templates(directory: Path = TEMPLATES_DIR) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    """{"sum"|"given": [(digit, normalized glyph, aspect ratio), ...]} from PNGs named <kind>_<digit>_<n>.png."""
    out: dict[str, list] = {"sum": [], "given": []}
    for p in sorted(directory.glob("*.png")):
        kind, digit, _ = p.stem.split("_")
        dark = 1.0 - np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255
        out[kind].append((int(digit), _normalize(dark), dark.shape[1] / dark.shape[0]))
    return out


def save_template(kind: str, digit: int, dark: np.ndarray, directory: Path = TEMPLATES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    n = len(list(directory.glob(f"{kind}_{digit}_*.png")))
    path = directory / f"{kind}_{digit}_{n}.png"
    Image.fromarray((255 - np.clip(dark, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def _match(dark: np.ndarray, templates: list) -> tuple[int | None, float, float]:
    """(digit, best score, margin over the best other digit), or None if no templates."""
    if not templates:
        return None, 0.0, 0.0
    v = _normalize(dark)
    aspect = dark.shape[1] / dark.shape[0]
    best: dict[int, float] = {}
    for digit, t, t_aspect in templates:
        score = float((v * t).sum())
        if not (0.75 < aspect / t_aspect < 1.33):
            score -= 0.3          # very different proportions: not the same glyph
        best[digit] = max(best.get(digit, -1.0), score)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    top_digit, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return top_digit, top, top - second


def _confident(score: float, margin: float) -> bool:
    return score >= MATCH_ACCEPT and margin >= MATCH_MARGIN


def ascii_art(dark: np.ndarray, width: int = 24) -> str:
    """Small text picture of a glyph, for questions in the terminal."""
    h, w = dark.shape
    scale = width / max(w, 1)
    rows = max(1, int(h * scale * 0.5))
    img = Image.fromarray((np.clip(dark, 0, 1) * 255).astype(np.uint8)).resize((width, rows))
    a = np.asarray(img)
    return "\n".join("".join("#" if v > 128 else ("+" if v > 50 else " ") for v in row) for row in a)


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------

def read(image_path: str | Path, answers: dict[str, object] | None = None,
         templates: dict | None = None) -> Reading:
    """
    Read a screenshot. `answers` resolves earlier Questions by key:
    "border:r,c:r,c" -> bool, "sum:r,c" -> int, "given:r,c" -> int or 0 (no given).
    """
    answers = answers or {}
    templates = templates if templates is not None else load_templates()

    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    ink = gray < INK_THRESHOLD
    dark = (255.0 - gray) / 255.0
    left, top, s = _find_board(ink)
    blobs = _blobs(ink, dark, left, top, s)

    questions: list[Question] = []

    def cell_of(y, x):
        return min(8, max(0, int(y // s))), min(8, max(0, int(x // s)))

    # --- Classify blobs.
    dots = np.zeros((int(round(9 * s)) + 1, int(round(9 * s)) + 1), dtype=bool)
    sum_glyphs: dict[tuple, list[_Blob]] = {}
    given_glyphs: dict[tuple, _Blob] = {}
    for b in blobs:
        if b.h < DOT_MAX * s and b.w < DOT_MAX * s:
            dots[b.y0:b.y0 + b.h, b.x0:b.x0 + b.w] |= b.mask
        elif SUM_HEIGHT[0] * s <= b.h <= SUM_HEIGHT[1] * s:
            sum_glyphs.setdefault(cell_of(b.y0, b.x0), []).append(b)
        elif GIVEN_HEIGHT[0] * s <= b.h <= GIVEN_HEIGHT[1] * s:
            given_glyphs[cell_of(b.y0 + b.h / 2, b.x0 + b.w / 2)] = b
        elif b.h > GIVEN_HEIGHT[1] * s:
            label, given = _split_label_and_given(b, s, templates["given"])
            cell = cell_of(b.y0 + b.h - 1, b.x0 + b.w / 2)
            if given is not None:
                given_glyphs[cell] = given
                sum_glyphs.setdefault(cell, []).extend(label)
            else:
                questions.append(Question(
                    key=f"sum:{cell[0]},{cell[1]}", kind="sum",
                    prompt=f"Cell row {cell[0] + 1}, column {cell[1] + 1} has a sum touching a given digit. "
                           "Type the cage SUM shown in its corner:",
                    picture=ascii_art(b.dark)))
                questions.append(Question(
                    key=f"given:{cell[0]},{cell[1]}", kind="given",
                    prompt=f"...and the large GIVEN digit in that cell (row {cell[0] + 1}, column {cell[1] + 1}):"))
        # Anything else (stray specks, odd shapes) is ignored.

    # --- Sums: glyphs starting in a cell's label corner, read left to right.
    sums: dict[tuple, int] = {}
    for (r, c), glyphs in sum_glyphs.items():
        key = f"sum:{r},{c}"
        if key in answers:
            sums[(r, c)] = int(answers[key])
            continue
        glyphs = [g for g in glyphs
                  if g.x0 - c * s < LABEL_ZONE[0] * s and g.y0 - r * s < LABEL_ZONE[1] * s]
        if not glyphs:
            continue
        glyphs.sort(key=lambda g: g.x0)
        digits = []
        for g in glyphs:
            d, score, margin = _match(g.dark, templates["sum"])
            if d is None or not _confident(score, margin):
                digits = None
                break
            digits.append(str(d))
        if digits is None:
            picture = ascii_art(_stack([g.dark for g in glyphs]))
            questions.append(Question(
                key=key, kind="sum",
                prompt=f"Cage sum in row {r + 1}, column {c + 1} is unclear. Type the number shown:",
                picture=picture, glyphs=[g.dark for g in glyphs]))
        else:
            sums[(r, c)] = int("".join(digits))

    # --- Givens.
    givens: dict[tuple, int] = {}
    for (r, c), g in given_glyphs.items():
        key = f"given:{r},{c}"
        if key in answers:
            if int(answers[key]):
                givens[(r, c)] = int(answers[key])
            continue
        d, score, margin = _match(g.dark, templates["given"])
        if d is None or not _confident(score, margin):
            questions.append(Question(
                key=key, kind="given",
                prompt=f"Large digit in row {r + 1}, column {c + 1} is unclear. Type the digit (0 if it is not a given):",
                picture=ascii_art(g.dark), glyphs=[g.dark]))
        else:
            givens[(r, c)] = d
    for key, value in answers.items():
        if key.startswith("given:") and int(value):
            r, c = map(int, key.split(":")[1].split(","))
            givens[(r, c)] = int(value)

    # --- Cage borders between neighbors.
    def dots_in_strip(r, c, side):
        d0, d1 = (int(STRIP_DEPTH[0] * s), int(STRIP_DEPTH[1] * s))
        a0, a1 = (int(STRIP_SPAN[0] * s), int(STRIP_SPAN[1] * s))
        cy, cx = int(round(r * s)), int(round(c * s))
        e = int(round(s))
        if side == "left":
            return int(dots[cy + a0:cy + a1, cx + d0:cx + d1].sum())
        if side == "right":
            return int(dots[cy + a0:cy + a1, cx + e - d1:cx + e - d0].sum())
        if side == "top":
            return int(dots[cy + d0:cy + d1, cx + a0:cx + a1].sum())
        return int(dots[cy + e - d1:cy + e - d0, cx + a0:cx + a1].sum())   # bottom

    def line_state(count):
        return True if count >= DOTS_PRESENT else (False if count <= DOTS_ABSENT else None)

    parent = {(r, c): (r, c) for r in range(9) for c in range(9)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in range(9):
        for c in range(9):
            for (nr, nc), side, other in (((r, c + 1), "right", "left"), ((r + 1, c), "bottom", "top")):
                if nr > 8 or nc > 8:
                    continue
                key = f"border:{r},{c}:{nr},{nc}"
                if key in answers:
                    border = bool(answers[key])
                else:
                    a = line_state(dots_in_strip(r, c, side))
                    b = line_state(dots_in_strip(nr, nc, other))
                    if a is not None and a == b:
                        border = a
                    else:
                        where = "right of" if side == "right" else "below"
                        questions.append(Question(
                            key=key, kind="border",
                            prompt=f"Is there a cage border between row {r + 1}, column {c + 1} and the cell "
                                   f"{where} it? [y/n]"))
                        continue
                if not border:
                    parent[find((r, c))] = find((nr, nc))

    # --- Assemble cages in reading order of their top-left cell.
    groups: dict[tuple, list] = {}
    for r in range(9):
        for c in range(9):
            groups.setdefault(find((r, c)), []).append((r, c))
    cages = []
    problems = []
    for cells in sorted(groups.values(), key=lambda cs: min(cs)):
        first = min(cells)
        labels_in = [cell for cell in cells if cell in sums]
        if labels_in != [first]:
            problems.append(
                f"Cage with cells {_cells_text(cells)} has sum labels at {_cells_text(labels_in) or 'none'}; "
                f"expected exactly one, at {_cells_text([first])}")
        cages.append({"id": len(cages) + 1,
                      "sum": sums.get(first, 0),
                      "cells": [list(cell) for cell in cells]})

    problems += validate(cages, [{"cell": list(k), "digit": v} for k, v in givens.items()])
    return Reading(cages=cages,
                   givens=[{"cell": list(k), "digit": v} for k, v in sorted(givens.items())],
                   questions=questions, problems=problems if not questions else [])


def _split_label_and_given(b: _Blob, s: float, given_templates: list):
    """
    Split a sum label that touches a given. Slide each given template along
    the bottom of the blob; the best fit is the given, the rest is the label.
    Returns (label blobs, given blob), or ([], None) if no template fits.
    """
    if not given_templates:
        return [], None
    gh = int(round(GIVEN_GLYPH_HEIGHT * s))
    best = (-2.0, None)
    region = b.dark[b.h - gh:, :]
    for x in range(0, max(1, b.w - int(0.15 * s))):
        for w in range(int(0.18 * s), int(0.45 * s) + 1, 2):
            if x + w > b.w:
                break
            window = region[:, x:x + w]
            d, score, margin = _match(window, given_templates)
            if score > best[0]:
                best = (score, (x, w, d))
    score, fit = best
    if fit is None or score < MATCH_ACCEPT:
        return [], None
    x, w, _ = fit
    given_mask = np.zeros_like(b.mask)
    given_mask[b.h - gh:, x:x + w] = b.mask[b.h - gh:, x:x + w]
    given_mask = ndi.binary_dilation(given_mask, iterations=1) & b.mask
    rest = b.mask & ~given_mask
    given_blob = _crop_blob(b, given_mask)
    labels, _ = ndi.label(rest, structure=np.ones((3, 3), dtype=int))
    label_blobs = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        part = _crop_blob(b, labels == i)
        if part is not None and part.h >= SUM_HEIGHT[0] * s:
            label_blobs.append(part)
    return label_blobs, given_blob


def _crop_blob(b: _Blob, mask: np.ndarray) -> _Blob | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    return _Blob(b.y0 + y0, b.x0 + x0, m, np.where(m, b.dark[y0:y1, x0:x1], 0.0))


def _stack(darks: list[np.ndarray]) -> np.ndarray:
    h = max(d.shape[0] for d in darks)
    parts = [np.pad(d, ((0, h - d.shape[0]), (0, 2))) for d in darks]
    return np.concatenate(parts, axis=1)


def _cells_text(cells) -> str:
    return ", ".join(f"r{r + 1}c{c + 1}" for r, c in cells)


# ----------------------------------------------------------------------------
# Validation and text output
# ----------------------------------------------------------------------------

def validate(cages: list[dict], givens: list[dict]) -> list[str]:
    """Structural checks beyond what the image reading guarantees."""
    from itertools import combinations
    problems = []
    total = sum(c["sum"] for c in cages)
    if total != 405:
        problems.append(f"Cage sums total {total}, not 405 (= 9 x 45)")
    for cage in cages:
        n, t = len(cage["cells"]), cage["sum"]
        if n > 9:
            problems.append(f"Cage {cage['id']} has {n} cells (max 9)")
        elif not any(sum(combo) == t for combo in combinations(range(1, 10), n)):
            problems.append(f"Cage {cage['id']}: sum {t} is impossible with {n} different digits")
    where = {tuple(cell): cage for cage in cages for cell in cage["cells"]}
    seen: dict[tuple, int] = {}
    for g in givens:
        r, c = g["cell"]
        d = g["digit"]
        for unit in (("row", r), ("col", c), ("box", (r // 3) * 3 + c // 3), ("cage", where[(r, c)]["id"])):
            if (unit, d) in seen:
                problems.append(f"Given {d} appears twice in the same {unit[0]}")
            seen[(unit, d)] = 1
    return problems


def text_grid(cages: list[dict], givens: list[dict]) -> str:
    """Cages as letters with their sums, and givens, for checking against the app."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    letter_of = {}
    for i, cage in enumerate(cages):
        for cell in cage["cells"]:
            letter_of[tuple(cell)] = letters[i % len(letters)]
    given_at = {tuple(g["cell"]): g["digit"] for g in givens}
    lines = ["Cages (each letter is one cage)        Givens"]
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------     ------+-------+------")
        left = " | ".join(" ".join(letter_of[(r, c)] for c in range(b, b + 3)) for b in (0, 3, 6))
        right = " | ".join(" ".join(str(given_at.get((r, c), ".")) for c in range(b, b + 3)) for b in (0, 3, 6))
        lines.append(f"{left}     {right}")
    lines.append("")
    sums = [f"{letters[i % len(letters)]}={cage['sum']}" for i, cage in enumerate(cages)]
    for i in range(0, len(sums), 10):
        lines.append("  ".join(sums[i:i + 10]))
    return "\n".join(lines)
