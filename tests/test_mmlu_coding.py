from __future__ import annotations

import types
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.tasks import load_task_adapter
from ta_mpq.tasks.mmlu_coding import (
    MMLUCodingExample,
    MMLU_CODING_SUBJECTS,
    build_messages,
    extract_prediction_details,
    extract_prediction_answer,
    extract_reference_answer,
    load_examples,
    select_examples_by_id,
    select_examples_for_split,
    score_prediction_detailed,
)


class MMLUCodingTaskTests(unittest.TestCase):
    def test_task_adapter_registration(self) -> None:
        task = load_task_adapter("mmlu-coding")
        self.assertEqual(task.__name__, "ta_mpq.tasks.mmlu_coding")

    def test_build_messages_defaults_to_openai_simple_evals_prompt(self) -> None:
        messages = build_messages("What is 2+2?\nA. 1\nB. 2\nC. 3\nD. 4")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Answer the following multiple choice question", messages[0]["content"])
        self.assertIn("Answer: $LETTER", messages[0]["content"])
        self.assertIn("Think step by step before answering", messages[0]["content"])

    def test_build_messages_keeps_legacy_letter_only_prompt(self) -> None:
        messages = build_messages(
            "Question\nA. alpha\nB. beta\nC. gamma\nD. delta",
            prompt_style="simple_evals_nonthinking",
        )
        self.assertIn("Respond with only the single letter", messages[0]["content"])

    def test_build_messages_supports_reasoning_prompt(self) -> None:
        messages = build_messages(
            "Question\nA. alpha\nB. beta\nC. gamma\nD. delta",
            prompt_style="reasoning_boxed",
        )
        self.assertIn("Think step by step", messages[0]["content"])
        self.assertIn(r"\boxed{}", messages[0]["content"])

    def test_extract_reference_answer_from_letter(self) -> None:
        self.assertEqual(extract_reference_answer("c"), "C")

    def test_extract_prediction_prefers_boxed_letter(self) -> None:
        self.assertEqual(extract_prediction_answer(r"Reasoning... \boxed{b}"), "B")

    def test_extract_prediction_prefers_official_answer_colon(self) -> None:
        predicted, source, has_boxed = extract_prediction_details("Reasoning...\nAnswer: $C")
        self.assertEqual(predicted, "C")
        self.assertEqual(source, "answer_colon")
        self.assertFalse(has_boxed)

    def test_extract_prediction_details_falls_back_to_answer_prefix(self) -> None:
        predicted, source, has_boxed = extract_prediction_details("The answer is (D).")
        self.assertEqual(predicted, "D")
        self.assertEqual(source, "answer_prefix")
        self.assertFalse(has_boxed)

    def test_score_prediction_detailed_reports_metadata(self) -> None:
        predicted, correct, metadata = score_prediction_detailed(r"\boxed{A}", "A")
        self.assertEqual(predicted, "A")
        self.assertTrue(correct)
        self.assertEqual(metadata["answer_extraction_source"], "boxed")
        self.assertTrue(metadata["has_boxed_answer"])

    def test_named_subsets_are_stable_and_disjoint(self) -> None:
        examples = [
            MMLUCodingExample(
                example_id=f"subject-{index % 4}:example-{index:03d}",
                question=f"Question {index}\nA. a\nB. b\nC. c\nD. d",
                answer="A",
            )
            for index in range(400)
        ]

        first300 = select_examples_for_split(examples, "first300")
        last100 = select_examples_for_split(examples, "last100")
        full = select_examples_for_split(examples, "full400")

        self.assertEqual(len(first300), 300)
        self.assertEqual(len(last100), 100)
        self.assertEqual(len(full), 400)
        self.assertEqual(
            {example.example_id for example in first300}.intersection(
                {example.example_id for example in last100}
            ),
            set(),
        )
        self.assertEqual(
            {example.example_id for example in first300}.union(
                {example.example_id for example in last100}
            ),
            {example.example_id for example in full},
        )

    def test_select_examples_by_id_preserves_requested_order(self) -> None:
        examples = [
            MMLUCodingExample(
                example_id=f"subject-0:example-{index:03d}",
                question=f"Question {index}\nA. a\nB. b\nC. c\nD. d",
                answer="A",
            )
            for index in range(8)
        ]
        selected = select_examples_by_id(
            examples,
            ["subject-0:example-005", "subject-0:example-001", "subject-0:example-007"],
        )
        self.assertEqual(
            [example.example_id for example in selected],
            ["subject-0:example-005", "subject-0:example-001", "subject-0:example-007"],
        )

    def test_load_examples_combines_subjects_and_applies_named_split(self) -> None:
        subject_rows = [
            {
                "question": "Which option is correct?",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer": 2,
                "id": str(index),
            }
            for index in range(100)
        ]
        load_calls: list[tuple[str, str, str]] = []

        def fake_load_dataset(dataset_name: str, subject: str, *, split: str) -> list[dict[str, object]]:
            load_calls.append((dataset_name, subject, split))
            return list(subject_rows)

        fake_datasets_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets_module}):
            examples = load_examples(split="last100")

        self.assertEqual(len(examples), 100)
        self.assertEqual(
            {subject for _, subject, _ in load_calls},
            set(MMLU_CODING_SUBJECTS),
        )
        self.assertTrue(all(split == "test" for _, _, split in load_calls))
        self.assertTrue(all(example.answer == "C" for example in examples))
        self.assertTrue(all("\nA) alpha" in example.question for example in examples))


if __name__ == "__main__":
    unittest.main()
