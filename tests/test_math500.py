from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.tasks.math500 import (
    MATH500Example,
    build_messages,
    extract_prediction_details,
    extract_prediction_answer,
    extract_reference_answer,
    normalize_answer,
    select_examples_by_id,
    select_examples_for_split,
    score_prediction_detailed,
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

    def test_extract_prediction_details_reports_boxed_metadata(self) -> None:
        completion = r"We solve it and get \boxed{\frac{5}{6}}."
        predicted, source, has_boxed = extract_prediction_details(completion)
        self.assertEqual(predicted, r"(5)/(6)")
        self.assertEqual(source, "boxed")
        self.assertTrue(has_boxed)

    def test_extract_prediction_uses_last_nonempty_line(self) -> None:
        completion = "Final answer: x=7\n\n7"
        self.assertEqual(extract_prediction_answer(completion), "7")

    def test_extract_prediction_details_reports_fallback_metadata(self) -> None:
        completion = "Reasoning...\nFinal answer: x=7\n\n7"
        predicted, source, has_boxed = extract_prediction_details(completion)
        self.assertEqual(predicted, "7")
        self.assertEqual(source, "last_line")
        self.assertFalse(has_boxed)

    def test_score_prediction(self) -> None:
        predicted, correct = score_prediction(r"\boxed{\frac{9}{2}}", r"(9)/(2)")
        self.assertEqual(predicted, r"(9)/(2)")
        self.assertTrue(correct)

    def test_score_prediction_detailed_reports_metadata(self) -> None:
        predicted, correct, metadata = score_prediction_detailed(
            r"\boxed{\frac{9}{2}}",
            r"(9)/(2)",
        )
        self.assertEqual(predicted, r"(9)/(2)")
        self.assertTrue(correct)
        self.assertEqual(metadata["answer_extraction_source"], "boxed")
        self.assertTrue(metadata["has_boxed_answer"])

    def test_build_messages_reasoning_boxed_prompt(self) -> None:
        messages = build_messages("Find x.", prompt_style="reasoning_boxed")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Please reason step by step", messages[0]["content"])
        self.assertIn(r"\boxed{}", messages[0]["content"])

    def test_build_messages_defaults_to_simple_evals_prompt(self) -> None:
        messages = build_messages("Find x.")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Please reason step by step", messages[0]["content"])
        self.assertIn(r"\boxed{}", messages[0]["content"])

    def test_build_messages_supports_smoke_test_prompt_aliases(self) -> None:
        for prompt_style in ("simple_evals_nonthinking", "qwen_prompt_thinking"):
            messages = build_messages("Find x.", prompt_style=prompt_style)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")
            self.assertIn("Please reason step by step", messages[0]["content"])
            self.assertIn(r"\boxed{}", messages[0]["content"])

    def test_named_subsets_are_stable_and_disjoint(self) -> None:
        examples = [
            MATH500Example(
                example_id=f"example-{index:03d}",
                question=f"Question {index}",
                answer=str(index),
            )
            for index in range(500)
        ]

        dev = select_examples_for_split(examples, "dev100")
        accept = select_examples_for_split(examples, "accept200")
        train = select_examples_for_split(examples, "train200")
        test = select_examples_for_split(examples, "test300")
        first300 = select_examples_for_split(examples, "first300")
        hard = select_examples_for_split(examples, "hard50")
        last100 = select_examples_for_split(examples, "last100")
        full = select_examples_for_split(examples, "full500")

        self.assertEqual(len(dev), 100)
        self.assertEqual(len(accept), 200)
        self.assertEqual(len(train), 200)
        self.assertEqual(len(test), 300)
        self.assertEqual(len(first300), 300)
        self.assertEqual(len(hard), 50)
        self.assertEqual(len(last100), 100)
        self.assertEqual(len(full), 500)
        self.assertEqual(len({example.example_id for example in dev}), 100)
        self.assertEqual(len({example.example_id for example in accept}), 200)
        self.assertEqual(len({example.example_id for example in train}), 200)
        self.assertEqual(len({example.example_id for example in test}), 300)
        self.assertEqual(len({example.example_id for example in first300}), 300)
        self.assertEqual(len({example.example_id for example in hard}), 50)
        self.assertEqual(len({example.example_id for example in last100}), 100)
        self.assertEqual(
            {example.example_id for example in dev}.intersection(
                {example.example_id for example in accept}
            ),
            set(),
        )
        self.assertEqual(
            {example.example_id for example in train}.intersection(
                {example.example_id for example in test}
            ),
            set(),
        )
        self.assertEqual(
            {example.example_id for example in train}.intersection(
                {example.example_id for example in hard}
            ),
            set(),
        )
        self.assertEqual(
            {example.example_id for example in first300}.intersection(
                {example.example_id for example in last100}
            ),
            set(),
        )
        self.assertEqual(
            {example.example_id for example in train}.union(
                {example.example_id for example in test}
            ),
            {example.example_id for example in full},
        )
        self.assertEqual(
            {example.example_id for example in first300},
            {example.example_id for example in full[:300]},
        )
        self.assertEqual(
            {example.example_id for example in last100},
            {example.example_id for example in full[400:500]},
        )
        self.assertEqual(
            {example.example_id for example in hard},
            {example.example_id for example in test[-50:]},
        )
        self.assertEqual(
            [example.example_id for example in dev],
            [example.example_id for example in select_examples_for_split(examples, "dev100")],
        )
        self.assertEqual(
            [example.example_id for example in hard],
            [example.example_id for example in select_examples_for_split(examples, "hard50")],
        )

    def test_select_examples_by_id_preserves_requested_order(self) -> None:
        examples = [
            MATH500Example(
                example_id=f"example-{index:03d}",
                question=f"Question {index}",
                answer=str(index),
            )
            for index in range(8)
        ]
        selected = select_examples_by_id(
            examples,
            ["example-005", "example-001", "example-007"],
        )
        self.assertEqual(
            [example.example_id for example in selected],
            ["example-005", "example-001", "example-007"],
        )


if __name__ == "__main__":
    unittest.main()
