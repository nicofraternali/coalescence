"""
Turn an app screenshot of a killer sudoku into a puzzle file for the solver.

    uv run python projects/killer_sudoku/read_screenshot.py <screenshot>
    uv run python projects/killer_sudoku/read_screenshot.py <screenshot> --name my_puzzle

A bare file name is also looked up in screenshots/, so a screenshot dropped
there can be read with just its name.

The reader (killer_solver/screenshot.py) never guesses: anything it isn't
certain about becomes a question here. Answers about digits are saved as new
reference images in reader_templates/, so the same glyph reads cleanly next
time. Before anything is written, the puzzle is shown as a text grid to check
against the app. The file goes to puzzles/<name>.yaml; by default <name> is
the next free number (002, 003, ...).

Then, as usual:
    uv run python projects/killer_sudoku/solve.py --puzzle <name>
    uv run python projects/killer_sudoku/sketch.py --puzzle <name>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from killer_solver import screenshot as ss

PROJECT_DIR = Path(__file__).resolve().parent
PUZZLES_DIR = PROJECT_DIR / "puzzles"
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"


def next_free_name() -> str:
    numbers = [int(m.group()) for p in PUZZLES_DIR.glob("*.yaml") if (m := re.match(r"\d+", p.stem))]
    return f"{max(numbers, default=0) + 1:03d}"


def ask(question: ss.Question):
    """Ask until the answer has the right form."""
    print()
    if question.picture:
        print(question.picture)
    while True:
        raw = input(question.prompt + " ").strip().lower()
        if question.kind == "border" and raw in ("y", "n"):
            return raw == "y"
        if question.kind == "sum" and raw.isdigit() and 1 <= int(raw) <= 45:
            return int(raw)
        if question.kind == "given" and raw.isdigit() and 0 <= int(raw) <= 9:
            return int(raw)
        print("  Please answer " + {"border": "y or n", "sum": "a number from 1 to 45",
                                     "given": "a digit from 0 to 9"}[question.kind] + ".")


def learn(question: ss.Question, answer) -> None:
    """Save the glyphs of an answered digit question as new reference images."""
    if question.kind not in ("sum", "given") or not question.glyphs:
        return
    digits = str(answer)
    if question.kind == "given" and answer == 0:
        return
    if len(digits) != len(question.glyphs):
        return   # can't tell which glyph is which digit; don't learn from it
    for d, glyph in zip(digits, question.glyphs):
        path = ss.save_template(question.kind, int(d), glyph)
        print(f"  Learned: {path.name}")


def puzzle_yaml(name: str, source: str, reading: ss.Reading) -> str:
    data = {
        "puzzle_id": name,
        "source": source,
        "cages": reading.cages,
        "givens": reading.givens,
    }
    header = (f"# Read from a screenshot by read_screenshot.py ({source}).\n"
              "# Cells are [row, col], zero-indexed; cages in reading order of their\n"
              "# top-left cell, where the app shows the sum.\n")
    return header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a killer sudoku screenshot into a puzzle file.")
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--name", type=str, default=None,
                        help="Puzzle name (file puzzles/<name>.yaml). Default: next free number.")
    args = parser.parse_args()

    # A bare file name also works for screenshots dropped into screenshots/.
    if not args.screenshot.exists():
        fallback = SCREENSHOTS_DIR / args.screenshot.name
        if not fallback.exists():
            sys.exit(f"Screenshot not found. Looked for:\n  {args.screenshot.resolve()}\n  {fallback}")
        args.screenshot = fallback

    name = args.name or next_free_name()
    out_path = PUZZLES_DIR / f"{name}.yaml"
    if out_path.exists():
        sys.exit(f"{out_path} already exists; choose another --name.")

    answers: dict[str, object] = {}
    reading = ss.read(args.screenshot)
    while reading.questions:
        print(f"\n{len(reading.questions)} thing(s) to confirm:")
        for q in reading.questions:
            answers[q.key] = ask(q)
            learn(q, answers[q.key])
        reading = ss.read(args.screenshot, answers=answers)

    if reading.problems:
        print("\nThe puzzle read from the screenshot is not valid:")
        for p in reading.problems:
            print(f"  - {p}")
        sys.exit("No file written. Check the screenshot, or report the problem above.")

    print()
    print(ss.text_grid(reading.cages, reading.givens))
    print()
    if input("Does this match the app? [y/n] ").strip().lower() != "y":
        sys.exit("No file written.")

    out_path.write_text(puzzle_yaml(name, args.screenshot.name, reading), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Next: uv run python projects/killer_sudoku/solve.py --puzzle {name}")


if __name__ == "__main__":
    main()
