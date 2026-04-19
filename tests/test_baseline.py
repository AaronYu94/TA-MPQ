from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.baseline import _render_prompt, _resolve_task_evaluation_mode
from ta_mpq.metrics import ExampleResult, summarize_results


class BaselineEvaluationModeTests(unittest.TestCase):
    def test_math500_defaults_to_simple_evals_with_thinking(self) -> None:
        mode = _resolve_task_evaluation_mode("math500", "")
        self.assertEqual(mode["task_prompt_style"], "simple_evals")
        self.assertTrue(mode["enable_thinking"])
        self.assertEqual(mode["thinking_mode"], "enabled")
        self.assertEqual(mode["generation_mode"], "greedy")

    def test_math500_nonthinking_smoke_mode_disables_thinking(self) -> None:
        mode = _resolve_task_evaluation_mode("math500", "simple_evals_nonthinking")
        self.assertEqual(mode["task_prompt_style"], "simple_evals_nonthinking")
        self.assertFalse(mode["enable_thinking"])
        self.assertEqual(mode["thinking_mode"], "disabled")
        self.assertEqual(mode["generation_mode"], "greedy")

    def test_math500_qwen_prompt_thinking_keeps_thinking_enabled(self) -> None:
        mode = _resolve_task_evaluation_mode("math500", "qwen_prompt_thinking")
        self.assertEqual(mode["task_prompt_style"], "qwen_prompt_thinking")
        self.assertTrue(mode["enable_thinking"])
        self.assertEqual(mode["thinking_mode"], "enabled")
        self.assertEqual(mode["generation_mode"], "greedy")

    def test_non_math_task_keeps_thinking_disabled(self) -> None:
        mode = _resolve_task_evaluation_mode("gsm8k", "")
        self.assertEqual(mode["task_prompt_style"], "")
        self.assertFalse(mode["enable_thinking"])
        self.assertEqual(mode["thinking_mode"], "disabled")
        self.assertEqual(mode["generation_mode"], "greedy")

    def test_render_prompt_passes_thinking_flag(self) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        result = _render_prompt(
            tokenizer,
            [{"role": "user", "content": "Find x."}],
            enable_thinking=True,
        )
        self.assertEqual(result, "prompt")
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Find x."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )


class ResultSummaryTests(unittest.TestCase):
    def test_summary_tracks_truncation_and_extraction_counts(self) -> None:
        results = [
            ExampleResult(
                example_id="ex-1",
                model_id="demo-model",
                reference_answer="1",
                predicted_answer="1",
                raw_completion=r"\boxed{1}",
                is_correct=True,
                latency_sec=1.0,
                prompt_tokens=10,
                completion_tokens=20,
                resident_memory_mb=100.0,
                generation_peak_delta_mb=5.0,
                total_peak_memory_mb=105.0,
                answer_extraction_source="boxed",
                has_boxed_answer=True,
                length_capped=False,
            ),
            ExampleResult(
                example_id="ex-2",
                model_id="demo-model",
                reference_answer="2",
                predicted_answer="2",
                raw_completion="Final answer: 2",
                is_correct=True,
                latency_sec=2.0,
                prompt_tokens=12,
                completion_tokens=2048,
                resident_memory_mb=100.0,
                generation_peak_delta_mb=7.0,
                total_peak_memory_mb=107.0,
                answer_extraction_source="last_line",
                has_boxed_answer=False,
                length_capped=True,
            ),
            ExampleResult(
                example_id="ex-3",
                model_id="demo-model",
                reference_answer="3",
                predicted_answer=None,
                raw_completion="Reasoning only",
                is_correct=False,
                latency_sec=3.0,
                prompt_tokens=14,
                completion_tokens=5,
                resident_memory_mb=100.0,
                generation_peak_delta_mb=8.0,
                total_peak_memory_mb=108.0,
                answer_extraction_source="none",
                has_boxed_answer=False,
                length_capped=False,
            ),
        ]

        summary = summarize_results(model_id="demo-model", task_name="math500", results=results)

        self.assertEqual(summary["length_capped_count"], 1)
        self.assertEqual(summary["boxed_answer_count"], 1)
        self.assertEqual(summary["fallback_answer_count"], 1)
        self.assertEqual(summary["no_answer_count"], 1)
        self.assertEqual(summary["max_completion_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
