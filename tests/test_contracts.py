from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.contracts import load_contract


class ContractTests(unittest.TestCase):
    def test_contract_loads_expected_models(self) -> None:
        contract = load_contract(PROJECT_ROOT / "configs" / "experiment_contract.json")
        self.assertEqual(contract.compressed_source_model_id, "Qwen/Qwen3.5-9B")
        self.assertEqual(contract.native_baseline_model_id, "Qwen/Qwen3.5-4B")
        self.assertEqual(contract.resolve_model_role("upper_bound"), "Qwen/Qwen3.5-9B")

    def test_math500_contract_loads_uniform8_search_defaults(self) -> None:
        contract = load_contract(PROJECT_ROOT / "configs" / "experiment_contract_27b_9b_math500.json")
        self.assertEqual(contract.quantization_bits, (4, 8, 16))
        self.assertAlmostEqual(contract.search_target_budget_gb or 0.0, 23.8623046875)
        self.assertEqual(contract.surrogate_target_metric, "accuracy_advantage_over_uniform")
        self.assertEqual(contract.surrogate_uniform_baseline_bit_width, 8)


if __name__ == "__main__":
    unittest.main()
