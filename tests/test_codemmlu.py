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
from ta_mpq.tasks.codemmlu import (
    CODEMMLU_CONFIGS,
    CodeMMLUExample,
    build_messages,
    extract_prediction_answer,
    extract_reference_answer,
    load_examples,
    select_examples_by_id,
    select_examples_for_split,
)


class CodeMMLUTaskTests(unittest.TestCase):
    def test_task_adapter_registration(self) -> None:
        task = load_task_adapter("codemmlu")
        self.assertEqual(task.__name__, "ta_mpq.tasks.codemmlu")

    def test_build_messages_uses_simple_evals_multichoice_prompt(self) -> None:
        messages = build_messages("Question\nA) alpha\nB) beta\nC) gamma\nD) delta")
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Answer the following multiple choice question", messages[0]["content"])
        self.assertIn("Answer: $LETTER", messages[0]["content"])
        self.assertIn("listed option letters", messages[0]["content"])
        self.assertIn("Do not include reasoning", messages[0]["content"])
        self.assertNotIn("Think step by step", messages[0]["content"])

    def test_build_messages_keeps_explicit_cot_prompt_available(self) -> None:
        messages = build_messages(
            "Question\nA) alpha\nB) beta\nC) gamma\nD) delta",
            prompt_style="simple_evals_cot",
        )
        self.assertIn("Think step by step", messages[0]["content"])
        self.assertIn("Answer: $LETTER", messages[0]["content"])

    def test_extract_supports_extended_answer_letters(self) -> None:
        self.assertEqual(extract_reference_answer("f"), "F")
        self.assertEqual(extract_prediction_answer("Reasoning...\nAnswer: $G"), "G")

    def test_named_subsets_are_stable_and_disjoint(self) -> None:
        examples = [
            CodeMMLUExample(
                example_id=f"config-{index % 4}:task-{index:03d}",
                question=f"Question {index}\nA) a\nB) b\nC) c\nD) d",
                answer="A",
            )
            for index in range(400)
        ]

        first300 = select_examples_for_split(examples, "first300")
        last100 = select_examples_for_split(examples, "last100")
        full = select_examples_for_split(examples, "full")

        self.assertEqual(len(first300), 300)
        self.assertEqual(len(last100), 100)
        self.assertEqual(len(full), 400)
        self.assertEqual(
            {example.example_id for example in first300}.intersection(
                {example.example_id for example in last100}
            ),
            set(),
        )

    def test_select_examples_by_id_preserves_requested_order(self) -> None:
        examples = [
            CodeMMLUExample(
                example_id=f"config:task-{index:03d}",
                question=f"Question {index}\nA) a\nB) b\nC) c\nD) d",
                answer="A",
            )
            for index in range(8)
        ]
        selected = select_examples_by_id(
            examples,
            ["config:task-005", "config:task-001", "config:task-007"],
        )
        self.assertEqual(
            [example.example_id for example in selected],
            ["config:task-005", "config:task-001", "config:task-007"],
        )

    def test_load_examples_combines_configs_and_applies_named_split(self) -> None:
        config_rows = [
            {
                "task_id": str(index),
                "question": "Which option is correct?",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer": "C",
            }
            for index in range(100)
        ]
        load_calls: list[tuple[str, str, str]] = []

        def fake_load_dataset(dataset_name: str, config_name: str, *, split: str) -> list[dict[str, object]]:
            load_calls.append((dataset_name, config_name, split))
            rows = list(config_rows)
            if config_name == "fill_in_the_middle":
                rows[0] = {
                    **rows[0],
                    "problem_description": "Fill in the missing code.",
                    "choices": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
                    "answer": "F",
                }
            return rows

        fake_datasets_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets_module}):
            examples = load_examples(split="last100")

        self.assertEqual(len(examples), 100)
        self.assertEqual(
            {config for _, config, _ in load_calls},
            set(CODEMMLU_CONFIGS),
        )
        self.assertTrue(all(split == "test" for _, _, split in load_calls))
        self.assertTrue(all(example.answer in {"A", "B", "C", "D", "E", "F", "G"} for example in examples))
        self.assertTrue(all("\nA) alpha" in example.question for example in examples))


if __name__ == "__main__":
    unittest.main()
