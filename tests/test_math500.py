from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.tasks.math500 import (
    extract_prediction_answer,
    extract_reference_answer,
    normalize_answer,
    score_prediction,
)


class Math500ParsingTests(unittest.TestCase):
    def test_normalize_answer_strips_latex_spacing(self) -> None:
        self.assertEqual(normalize_answer(r"\left( 3, \frac{1}{2} \right)"), r"(3,(1)/(2))")

    def test_extract_reference_answer(self) -> None:
        self.assertEqual(extract_reference_answer(r"\frac{3}{4}"), r"(3)/(4)")

    def test_extract_prediction_prefers_boxed_answer(self) -> None:
        completion = r"We solve it and get \boxed{\frac{5}{6}}."
        self.assertEqual(extract_prediction_answer(completion), r"(5)/(6)")

    def test_extract_prediction_uses_last_nonempty_line(self) -> None:
        completion = "Final answer: x=7\n\n7"
        self.assertEqual(extract_prediction_answer(completion), "7")

    def test_score_prediction(self) -> None:
        predicted, correct = score_prediction(r"\boxed{\frac{9}{2}}", r"(9)/(2)")
        self.assertEqual(predicted, r"(9)/(2)")
        self.assertTrue(correct)


if __name__ == "__main__":
    unittest.main()
