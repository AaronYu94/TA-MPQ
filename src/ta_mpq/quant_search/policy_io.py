from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ta_mpq.policy_export import export_candidate_from_group_bits
from ta_mpq.quant_search.group_registry import GroupInfo, build_report_payload
from ta_mpq.quant_search.policy_builder import BuiltPolicy
from ta_mpq.quant_search.policy_hash import canonical_assignment_hash, canonical_registry_hash


def write_built_policy(
    path: str | Path,
    policy: BuiltPolicy,
    groups: list[GroupInfo],
    model_id: str | None = None,
) -> dict[str, Any]:
    payload = _candidate_payload_from_built_policy(
        policy=policy,
        groups=groups,
        model_id=model_id,
    )
    save_policy_payload(path, payload)
    return payload


def save_policy_payload(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_policy_payload(payload)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_policy_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_policy_payload(payload)


def canonical_policy_hash(payload: dict[str, Any]) -> str:
    normalized = _normalize_policy_payload(payload)
    return canonical_assignment_hash(
        normalized["bitwidths"],
        grouping=str(normalized.get("grouping") or "per_block_component"),
        budget_rule=str(normalized.get("budget_rule") or "raw_weight_int4"),
        registry_hash=normalized.get("registry_hash"),
        quantizer_config_hash=normalized.get("quantizer_config_hash"),
    )


def list_policy_files(
    policy_dir: str | Path,
    policy_ids: list[str] | None = None,
    policy_hashes: list[str] | None = None,
    builder_names: set[str] | None = None,
) -> list[Path]:
    directory = Path(policy_dir)
    selected_ids = set(policy_ids or [])
    selected_hashes = set(policy_hashes or [])
    files: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if "policy_id" not in raw_payload:
            continue
        payload = _normalize_policy_payload(raw_payload)
        if selected_ids and str(payload.get("policy_id")) not in selected_ids:
            continue
        if selected_hashes and str(payload.get("policy_hash")) not in selected_hashes:
            continue
        if builder_names and str(payload.get("source", {}).get("builder")) not in builder_names:
            continue
        files.append(path)
    return files


def _candidate_payload_from_built_policy(
    policy: BuiltPolicy,
    groups: list[GroupInfo],
    model_id: str | None = None,
) -> dict[str, Any]:
    report_payload = build_report_payload(groups, model_id=model_id)
    candidate_payload = export_candidate_from_group_bits(
        report_payload=report_payload,
        grouping="per_block_component",
        group_bits=policy.bitwidth_dict(),
        rank=0,
        metadata={
            "fitness": float(policy.proxy_score),
            "proxy_quality_score": float(policy.proxy_score),
            "provenance": str(policy.source.get("builder", "manual")),
            "estimated_average_bit_width": None,
            "estimated_weight_footprint_gb": None,
        },
    )
    candidate_payload.update(policy.to_dict())
    candidate_payload["grouping"] = "per_block_component"
    candidate_payload["budget_rule"] = "raw_weight_int4"
    candidate_payload["registry_hash"] = canonical_registry_hash(groups)
    candidate_payload["quantizer_config_hash"] = None
    candidate_payload["group_bit_assignments"] = dict(policy.bitwidths)
    candidate_payload["candidate_path"] = None
    return _normalize_policy_payload(candidate_payload)


def _normalize_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["policy_id"] = str(normalized["policy_id"])
    normalized["method"] = str(normalized.get("method", "task_sensitivity_exact_budget"))
    normalized["grouping"] = str(normalized.get("grouping", "per_block_component"))
    normalized["budget_rule"] = str(normalized.get("budget_rule", "raw_weight_int4"))
    normalized["registry_hash"] = normalized.get("registry_hash")
    normalized["quantizer_config_hash"] = normalized.get("quantizer_config_hash")
    normalized["bitwidths"] = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in sorted(dict(normalized.get("bitwidths", {})).items())
    }
    normalized["group_bit_assignments"] = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in sorted(
            dict(normalized.get("group_bit_assignments", normalized["bitwidths"])).items()
        )
    }
    normalized["stats"] = dict(normalized.get("stats", {}))
    normalized["source"] = dict(normalized.get("source", {}))
    normalized["policy_hash"] = str(
        normalized.get("policy_hash")
        or canonical_assignment_hash(
            normalized["bitwidths"],
            grouping=normalized["grouping"],
            budget_rule=normalized["budget_rule"],
            registry_hash=normalized.get("registry_hash"),
            quantizer_config_hash=normalized.get("quantizer_config_hash"),
        )
    )
    return normalized
