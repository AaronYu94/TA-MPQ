from __future__ import annotations

import unittest
from unittest import mock

from ta_mpq.modal_feasibility_app import (
    _advance_search_deck_for_resume,
    _build_evaluation_payload,
    _candidate_key_sort_key,
    _candidate_round_snapshot,
    _consume_search_turn_examples,
    _evaluation_output_path,
    _initialize_search_deck,
    _merge_staged_evaluation_payload,
    _resolve_default_max_new_tokens,
    _resolve_remote_model_ref,
    _round_index_from_candidate_key,
    _task_limit_for_split,
)


class ResolveRemoteModelRefTests(unittest.TestCase):
    def test_non_artifact_path_returns_unchanged(self) -> None:
        with mock.patch("ta_mpq.modal_feasibility_app.artifact_volume.reload") as reload_mock:
            resolved = _resolve_remote_model_ref("Qwen/Qwen3.5-27B")

        self.assertEqual(resolved, "Qwen/Qwen3.5-27B")
        reload_mock.assert_not_called()

    def test_artifact_path_reloads_until_path_exists(self) -> None:
        with mock.patch(
            "ta_mpq.modal_feasibility_app.Path.exists",
            side_effect=[False, False, True],
        ):
            with mock.patch("ta_mpq.modal_feasibility_app.artifact_volume.reload") as reload_mock:
                with mock.patch("time.sleep") as sleep_mock:
                    resolved = _resolve_remote_model_ref(
                        "/artifacts/test-model",
                        reload_attempts=4,
                        sleep_seconds=0.01,
                    )

        self.assertEqual(resolved, "/artifacts/test-model")
        self.assertEqual(reload_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_artifact_path_raises_when_never_visible(self) -> None:
        with mock.patch("ta_mpq.modal_feasibility_app.Path.exists", return_value=False):
            with mock.patch("ta_mpq.modal_feasibility_app.artifact_volume.reload") as reload_mock:
                with mock.patch("time.sleep") as sleep_mock:
                    with self.assertRaises(FileNotFoundError):
                        _resolve_remote_model_ref(
                            "/artifacts/missing-model",
                            reload_attempts=3,
                            sleep_seconds=0.01,
                        )

        self.assertEqual(reload_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 3)


class DefaultMaxNewTokensTests(unittest.TestCase):
    def test_math500_uses_2048_default(self) -> None:
        self.assertEqual(_resolve_default_max_new_tokens("math500", 0), 2048)
        self.assertEqual(_resolve_default_max_new_tokens("math-500", 0), 2048)

    def test_non_math_task_keeps_small_default(self) -> None:
        self.assertEqual(_resolve_default_max_new_tokens("gsm8k", 0), 64)
        self.assertEqual(_resolve_default_max_new_tokens("gsm8k", 256), 256)


class TaskLimitForSplitTests(unittest.TestCase):
    def test_first300_and_last100_limits_are_exposed(self) -> None:
        self.assertEqual(_task_limit_for_split("first300"), 300)
        self.assertEqual(_task_limit_for_split("last100"), 100)


class ResumeHelperTests(unittest.TestCase):
    def test_round_index_from_candidate_key(self) -> None:
        self.assertEqual(_round_index_from_candidate_key("round-03-candidate-02"), 3)
        self.assertIsNone(_round_index_from_candidate_key("seed-01-uniform_int4_seed"))

    def test_candidate_key_sort_key_orders_seeds_then_rounds(self) -> None:
        ordered = sorted(
            [
                "round-02-candidate-01",
                "seed-02-compression_first_seed",
                "round-01-candidate-02",
                "seed-01-uniform_int4_seed",
                "round-01-candidate-01",
            ],
            key=_candidate_key_sort_key,
        )
        self.assertEqual(
            ordered,
            [
                "seed-01-uniform_int4_seed",
                "seed-02-compression_first_seed",
                "round-01-candidate-01",
                "round-01-candidate-02",
                "round-02-candidate-01",
            ],
        )

    def test_staged_evaluation_payload_combines_raw_counts(self) -> None:
        stage1_path = _evaluation_output_path("slug", "round-01-candidate-01", "first300", "stage1")
        stage2_path = _evaluation_output_path("slug", "round-01-candidate-01", "first300", "stage2")
        payload = _merge_staged_evaluation_payload(
            None,
            "stage1",
            _build_evaluation_payload(
                {
                    "accuracy": 0.8,
                    "num_correct": 8,
                    "num_examples": 10,
                    "evaluated_example_ids": [f"q{i}" for i in range(10)],
                },
                "first300",
                stage1_path,
                {"deck_cursor_after": 10},
            ),
        )
        payload = _merge_staged_evaluation_payload(
            payload,
            "stage2",
            _build_evaluation_payload(
                {
                    "accuracy": 12 / 15,
                    "num_correct": 12,
                    "num_examples": 15,
                    "evaluated_example_ids": [f"q{i}" for i in range(10, 25)],
                },
                "first300",
                stage2_path,
                {"deck_cursor_after": 25},
            ),
        )

        self.assertAlmostEqual(payload["combined_accuracy"], 20 / 25)
        self.assertEqual(payload["num_correct"], 20)
        self.assertEqual(payload["num_examples"], 25)
        self.assertEqual(len(payload["evaluated_example_ids"]), 25)
        snapshot = _candidate_round_snapshot(
            {
                "candidate_key": "round-01-candidate-01",
                "provenance": "cluster_flip",
                "proposal_score": 0.1,
                "integrity_clean": True,
                "smoke_test_passed": True,
                "matched_linear_weight_footprint_gb": 1.0,
                "estimated_full_model_weight_footprint_gb": 2.0,
                "evaluations": {"first300": payload},
            },
            "first300",
        )
        self.assertAlmostEqual(snapshot["accuracy"], 20 / 25)
        self.assertAlmostEqual(snapshot["stage1_accuracy"], 0.8)
        self.assertAlmostEqual(snapshot["stage2_accuracy"], 12 / 15)

    def test_resume_deck_replay_only_consumes_shared_stage_slices_once_per_round(self) -> None:
        with mock.patch(
            "ta_mpq.modal_feasibility_app.load_task_example_ids_remote.remote",
            return_value=[f"q{i:03d}" for i in range(300)],
        ):
            deck_state = _initialize_search_deck("math500", "first300", "resume-unit")
            stage1_turn = _consume_search_turn_examples(deck_state, 10)
            stage2_turn = _consume_search_turn_examples(deck_state, 15)

            replay_deck = _initialize_search_deck("math500", "first300", "resume-unit")
            resumed_records = [
                {
                    "candidate_key": "round-01-candidate-01",
                    "evaluations": {
                        "first300": {
                            "stage1_evaluated_example_ids": list(stage1_turn["example_ids"]),
                            "stage2_evaluated_example_ids": list(stage2_turn["example_ids"]),
                        }
                    },
                },
                {
                    "candidate_key": "round-01-candidate-02",
                    "evaluations": {
                        "first300": {
                            "stage1_evaluated_example_ids": list(stage1_turn["example_ids"]),
                            "stage2_evaluated_example_ids": list(stage2_turn["example_ids"]),
                        }
                    },
                },
            ]

            _advance_search_deck_for_resume(
                deck_state=replay_deck,
                resumed_records=resumed_records,
                stage1_turn_limit=10,
                stage2_turn_limit=15,
                split_name="first300",
            )

            self.assertEqual(replay_deck["cursor"], 25)
            self.assertEqual(replay_deck["reshuffle_count"], 0)


if __name__ == "__main__":
    unittest.main()
