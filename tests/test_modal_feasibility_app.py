from __future__ import annotations

import unittest
from unittest import mock

from ta_mpq.modal_feasibility_app import _resolve_remote_model_ref


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


if __name__ == "__main__":
    unittest.main()
