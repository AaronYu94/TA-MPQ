from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.tasks.gsm8k import (
    extract_prediction_answer,
    extract_reference_answer,
    normalize_answer,
    score_prediction,
)


class GSM8KParsingTests(unittest.TestCase):
    def test_extract_reference_answer(self) -> None:
        answer = "reasoning\n#### 1,234"
        self.assertEqual(extract_reference_answer(answer), "1234")

    def test_extract_prediction_prefers_boxed_answer(self) -> None:
        completion = "We solve it and get \\boxed{42}. Another number is 7."
        self.assertEqual(extract_prediction_answer(completion), "42")

    def test_extract_prediction_falls_back_to_last_number(self) -> None:
        completion = "First 10, then 11, so the answer is 12."
        self.assertEqual(extract_prediction_answer(completion), "12")

    def test_score_prediction(self) -> None:
        predicted, correct = score_prediction("Final: \\boxed{256}", "256")
        self.assertEqual(predicted, "256")
        self.assertTrue(correct)

    def test_normalize_answer(self) -> None:
        self.assertEqual(normalize_answer("$1,000."), "1000")


if __name__ == "__main__":
    unittest.main()
