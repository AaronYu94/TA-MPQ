from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


DEFAULT_BUDGET_RULE = "raw_weight_int4"
DEFAULT_GROUPING = "per_block_component"


def canonical_assignment_hash(
    bitwidths: dict[str, int],
    *,
    grouping: str = DEFAULT_GROUPING,
    budget_rule: str = DEFAULT_BUDGET_RULE,
    registry_hash: str | None = None,
    quantizer_config_hash: str | None = None,
) -> str:
    payload = {
        "grouping": str(grouping),
        "budget_rule": str(budget_rule),
        "registry_hash": registry_hash,
        "quantizer_config_hash": quantizer_config_hash,
        "assignment": sorted((str(group_id), int(bitwidth)) for group_id, bitwidth in bitwidths.items()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_registry_hash(groups: Iterable[Any]) -> str:
    payload = [
        (
            str(group.group_id),
            str(group.module_path),
            int(group.param_count),
            bool(group.is_quantizable),
        )
        for group in groups
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dedupe_by_policy_hash(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        policy_hash = str(item["policy_hash"])
        if policy_hash in seen:
            continue
        deduped.append(item)
        seen.add(policy_hash)
    return deduped


def duplicate_policy_hashes(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        policy_hash = str(item["policy_hash"])
        grouped.setdefault(policy_hash, []).append(str(item.get("policy_id", policy_hash)))
    return {
        policy_hash: policy_ids
        for policy_hash, policy_ids in grouped.items()
        if len(policy_ids) > 1
    }
