from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from killer_solver import screenshot as ss

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS = PROJECT_DIR / "screenshots"

# Each example screenshot and the puzzle file it was read into (checked by hand
# against the app).
EXAMPLES = [
    ("Screenshot_20261007_192556_Killer Sudoku.jpg", "002.yaml"),   # "Killer" level, no givens
    ("Screenshot_20261007_193709_Killer Sudoku.jpg", "003.yaml"),   # "Hard" level, givens; a sum touches a given
]


@pytest.mark.parametrize("shot, puzzle", EXAMPLES)
def test_screenshot_reads_exactly_without_questions(shot, puzzle):
    reading = ss.read(SCREENSHOTS / shot)
    expected = yaml.safe_load((PROJECT_DIR / "puzzles" / puzzle).read_text(encoding="utf-8"))
    assert reading.questions == []
    assert reading.problems == []
    assert reading.cages == expected["cages"]
    assert reading.givens == expected["givens"]


def test_without_templates_it_asks_instead_of_guessing():
    reading = ss.read(SCREENSHOTS / EXAMPLES[1][0], templates={"sum": [], "given": []})
    kinds = {q.kind for q in reading.questions}
    assert kinds == {"sum", "given"}
    assert reading.problems == []   # nothing is validated while questions are open


def test_validation_catches_a_wrong_sum():
    expected = yaml.safe_load((PROJECT_DIR / "puzzles" / EXAMPLES[0][1]).read_text(encoding="utf-8"))
    cages = expected["cages"]
    cages[0]["sum"] += 1
    assert any("not 405" in p for p in ss.validate(cages, []))


def test_rejects_an_image_without_a_board(tmp_path):
    blank = tmp_path / "blank.png"
    Image.fromarray(np.full((800, 400), 240, dtype=np.uint8)).save(blank)
    with pytest.raises(ValueError, match="thick horizontal lines"):
        ss.read(blank)
