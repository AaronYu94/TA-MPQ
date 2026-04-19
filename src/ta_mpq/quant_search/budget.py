from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ta_mpq.quant_search.group_registry import GroupInfo, group_lookup, total_param_count


@dataclass(frozen=True, slots=True)
class PolicyBudgetStats:
    num_2bit: int
    num_4bit: int
    num_8bit: int
    twobit_param_count: int
    twobit_mass_fraction: float
    fourbit_param_count: int
    fourbit_mass_fraction: float
    realized_8bit_param_mass_fraction: float
    realized_2bit_param_mass_fraction: float
    realized_4bit_param_mass_fraction: float
    raw_weight_bits: int
    target_int4_bits: int
    budget_slack_bits: int
    budget_slack_fraction: float
    promotion_mass_fraction: float
    promoted_param_count: int
    total_param_count: int
    changed_groups_vs_uniform_int4: int = 0
    path_index: int | None = None
    actual_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["actual_bytes"] is None:
            payload.pop("actual_bytes")
        if payload["path_index"] is None:
            payload.pop("path_index")
        return payload


def target_int4_bits(groups: list[GroupInfo]) -> int:
    return sum(4 * group.param_count for group in groups)


def policy_bits(
    bitwidths: dict[str, int],
    groups: list[GroupInfo],
) -> int:
    lookup = group_lookup(groups)
    return sum(int(bitwidths[group_id]) * lookup[group_id].param_count for group_id in lookup)


def compute_policy_budget_stats(
    bitwidths: dict[str, int],
    groups: list[GroupInfo],
    actual_bytes: int | None = None,
    path_index: int | None = None,
    uniform_anchor_bitwidth: int = 4,
) -> PolicyBudgetStats:
    lookup = group_lookup(groups)
    missing = sorted(set(lookup) - set(bitwidths))
    if missing:
        raise ValueError(f"Policy is missing {len(missing)} group assignments")

    raw_bits = policy_bits(bitwidths, groups)
    target_bits = target_int4_bits(groups)
    slack_bits = target_bits - raw_bits
    total_params = total_param_count(groups)
    promoted_params = sum(
        lookup[group_id].param_count
        for group_id, bitwidth in bitwidths.items()
        if int(bitwidth) == 8
    )
    twobit_param_count = sum(
        lookup[group_id].param_count
        for group_id, bitwidth in bitwidths.items()
        if int(bitwidth) == 2
    )
    fourbit_param_count = sum(
        lookup[group_id].param_count
        for group_id, bitwidth in bitwidths.items()
        if int(bitwidth) == 4
    )
    counts = {
        2: 0,
        4: 0,
        8: 0,
    }
    for bitwidth in bitwidths.values():
        bit = int(bitwidth)
        if bit in counts:
            counts[bit] += 1
    return PolicyBudgetStats(
        num_2bit=counts[2],
        num_4bit=counts[4],
        num_8bit=counts[8],
        twobit_param_count=twobit_param_count,
        twobit_mass_fraction=(twobit_param_count / total_params) if total_params else 0.0,
        fourbit_param_count=fourbit_param_count,
        fourbit_mass_fraction=(fourbit_param_count / total_params) if total_params else 0.0,
        realized_8bit_param_mass_fraction=(promoted_params / total_params) if total_params else 0.0,
        realized_2bit_param_mass_fraction=(twobit_param_count / total_params) if total_params else 0.0,
        realized_4bit_param_mass_fraction=(fourbit_param_count / total_params) if total_params else 0.0,
        raw_weight_bits=raw_bits,
        target_int4_bits=target_bits,
        budget_slack_bits=slack_bits,
        budget_slack_fraction=(slack_bits / target_bits) if target_bits else 0.0,
        promotion_mass_fraction=(promoted_params / total_params) if total_params else 0.0,
        promoted_param_count=promoted_params,
        total_param_count=total_params,
        changed_groups_vs_uniform_int4=sum(
            1
            for bitwidth in bitwidths.values()
            if int(bitwidth) != int(uniform_anchor_bitwidth)
        ),
        path_index=path_index,
        actual_bytes=actual_bytes,
    )


def measure_artifact_size_bytes(path: str | Path) -> int:
    artifact_path = Path(path)
    if artifact_path.is_file():
        return artifact_path.stat().st_size
    return sum(
        child.stat().st_size
        for child in artifact_path.rglob("*")
        if child.is_file()
    )
