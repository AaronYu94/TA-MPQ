from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    name: str
    task_name: str
    compressed_source_model_id: str
    native_baseline_model_id: str
    upper_bound_model_id: str
    comparison_rule: str
    budget_rule: str
    quantization_bits: tuple[int, ...]
    calibration_samples: int
    baseline_eval_limit: int
    generation_max_new_tokens: int
    search_target_budget_gb: float | None = None
    surrogate_target_metric: str | None = None
    surrogate_uniform_baseline_bit_width: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentContract":
        return cls(
            name=str(payload["name"]),
            task_name=str(payload["task_name"]),
            compressed_source_model_id=str(payload["compressed_source_model_id"]),
            native_baseline_model_id=str(payload["native_baseline_model_id"]),
            upper_bound_model_id=str(payload["upper_bound_model_id"]),
            comparison_rule=str(payload["comparison_rule"]),
            budget_rule=str(payload["budget_rule"]),
            quantization_bits=tuple(int(bit) for bit in payload["quantization_bits"]),
            calibration_samples=int(payload["calibration_samples"]),
            baseline_eval_limit=int(payload["baseline_eval_limit"]),
            generation_max_new_tokens=int(payload["generation_max_new_tokens"]),
            search_target_budget_gb=(
                float(payload["search_target_budget_gb"])
                if payload.get("search_target_budget_gb") is not None
                else None
            ),
            surrogate_target_metric=(
                str(payload["surrogate_target_metric"])
                if payload.get("surrogate_target_metric") is not None
                else None
            ),
            surrogate_uniform_baseline_bit_width=(
                int(payload["surrogate_uniform_baseline_bit_width"])
                if payload.get("surrogate_uniform_baseline_bit_width") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_model_role(self, role: str) -> str:
        role_to_model = {
            "compressed_source": self.compressed_source_model_id,
            "native_baseline": self.native_baseline_model_id,
            "upper_bound": self.upper_bound_model_id,
        }
        if role not in role_to_model:
            raise KeyError(f"Unknown model role: {role}")
        return role_to_model[role]


def load_contract(path: str | Path) -> ExperimentContract:
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentContract.from_dict(payload)


def save_contract(path: str | Path, contract: ExperimentContract) -> None:
    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    with contract_path.open("w", encoding="utf-8") as handle:
        json.dump(contract.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
