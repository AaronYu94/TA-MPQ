from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ta_mpq.feasibility import LinearLayerStat, collect_linear_layer_stats
from ta_mpq.search import build_search_groups, layer_stats_from_report


@dataclass(frozen=True, slots=True)
class GroupInfo:
    group_id: str
    module_path: str
    block_idx: int | None
    component: str
    param_count: int
    is_quantizable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupInfo":
        return cls(
            group_id=str(payload["group_id"]),
            module_path=str(payload["module_path"]),
            block_idx=(
                int(payload["block_idx"])
                if payload.get("block_idx") is not None
                else None
            ),
            component=str(payload["component"]),
            param_count=int(payload["param_count"]),
            is_quantizable=bool(payload.get("is_quantizable", True)),
        )


def build_group_registry(
    layer_stats: list[LinearLayerStat],
    grouping: str = "per_block_component",
) -> list[GroupInfo]:
    search_groups = build_search_groups(layer_stats, grouping=grouping)
    registry: list[GroupInfo] = []
    for group in search_groups:
        if len(group.layer_names) != 1:
            raise ValueError(
                "Group registry currently requires a 1:1 group-to-module mapping; "
                f"group {group.name} expanded to {len(group.layer_names)} modules under {grouping}"
            )
        module_path = str(group.layer_names[0])
        registry.append(
            GroupInfo(
                group_id=str(group.name),
                module_path=module_path,
                block_idx=_block_idx_from_group_id(group.name),
                component=str(group.component_type),
                param_count=int(group.parameter_count),
                is_quantizable=True,
            )
        )
    return sorted(registry, key=lambda item: item.group_id)


def build_group_registry_from_report(
    report_payload: dict[str, Any],
    grouping: str = "per_block_component",
) -> list[GroupInfo]:
    return build_group_registry(layer_stats_from_report(report_payload), grouping=grouping)


def build_group_registry_from_model(
    model_id: str,
    grouping: str = "per_block_component",
) -> list[GroupInfo]:
    return build_group_registry(collect_linear_layer_stats(model_id), grouping=grouping)


def save_group_registry(path: str | Path, groups: list[GroupInfo]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for group in sorted(groups, key=lambda item: item.group_id):
            json.dump(group.to_dict(), handle, sort_keys=True)
            handle.write("\n")


def load_group_registry(path: str | Path) -> list[GroupInfo]:
    groups: list[GroupInfo] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            groups.append(GroupInfo.from_dict(json.loads(line)))
    return sorted(groups, key=lambda item: item.group_id)


def group_lookup(groups: list[GroupInfo]) -> dict[str, GroupInfo]:
    return {group.group_id: group for group in groups}


def total_param_count(groups: list[GroupInfo]) -> int:
    return sum(group.param_count for group in groups)


def to_layer_stats(groups: list[GroupInfo]) -> list[LinearLayerStat]:
    return [
        LinearLayerStat(
            name=group.module_path,
            parameter_count=group.param_count,
            in_features=0,
            out_features=0,
        )
        for group in sorted(groups, key=lambda item: item.group_id)
    ]


def build_report_payload(
    groups: list[GroupInfo],
    model_id: str | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "layer_stats": [layer.to_dict() for layer in to_layer_stats(groups)],
    }


def _block_idx_from_group_id(group_id: str) -> int | None:
    if not group_id.startswith("block:"):
        return None
    _, raw_block_idx, _ = group_id.split(":", 2)
    return int(raw_block_idx)
