from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def candidate_signature_from_payload(
    candidate_payload: dict[str, Any],
    policy_source: str = "llmcompressor",
) -> str:
    assignments = candidate_assignments_from_payload(
        candidate_payload,
        policy_source=policy_source,
    )
    canonical_payload = {
        "policy_source": policy_source,
        "assignments": sorted(
            (str(module_name), int(bit_width))
            for module_name, bit_width in assignments.items()
        ),
    }
    digest = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest


def candidate_assignments_from_payload(
    candidate_payload: dict[str, Any],
    policy_source: str = "llmcompressor",
) -> dict[str, int]:
    if policy_source == "llmcompressor":
        raw_assignments = candidate_payload["backend_projections"]["llmcompressor"][
            "projected_module_bit_assignments"
        ]
    elif policy_source == "project":
        raw_assignments = candidate_payload["module_bit_assignments"]
    else:
        raise ValueError(f"Unknown policy_source: {policy_source}")
    return {
        str(module_name): int(bit_width)
        for module_name, bit_width in raw_assignments.items()
    }


def candidate_assignments_from_path(
    candidate_path: str | Path,
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
) -> dict[str, int]:
    payload = _load_json(_resolve_path(candidate_path, base_dir))
    return candidate_assignments_from_payload(payload, policy_source=policy_source)


def normalized_policy_distance(
    left_assignments: dict[str, int],
    right_assignments: dict[str, int],
) -> float:
    all_modules = sorted(set(left_assignments) | set(right_assignments))
    if not all_modules:
        return 0.0
    mismatches = 0
    for module_name in all_modules:
        if left_assignments.get(module_name) != right_assignments.get(module_name):
            mismatches += 1
    return mismatches / len(all_modules)


def estimate_candidate_novelty(
    candidate_assignments: dict[str, int],
    existing_assignments: list[dict[str, int]],
) -> float:
    if not existing_assignments:
        return 1.0
    distances = [
        normalized_policy_distance(candidate_assignments, prior_assignments)
        for prior_assignments in existing_assignments
    ]
    return min(distances)


def estimate_candidate_acquisition_score(
    candidate_payload: dict[str, Any],
    novelty_score: float,
    *,
    exploration_weight: float = 0.5,
    diversity_weight: float = 0.15,
) -> dict[str, float]:
    exploitation_value = candidate_payload.get("conservative_prediction")
    if exploitation_value is None:
        exploitation_value = candidate_payload.get("fitness", 0.0)
    exploitation = float(exploitation_value)
    uncertainty = float(candidate_payload.get("prediction_uncertainty") or 0.0)
    acquisition = exploitation + exploration_weight * uncertainty + diversity_weight * novelty_score
    return {
        "selection_exploitation_score": exploitation,
        "selection_novelty_score": novelty_score,
        "selection_acquisition_score": acquisition,
    }


def candidate_signature_from_path(
    candidate_path: str | Path,
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
) -> str:
    payload = _load_json(_resolve_path(candidate_path, base_dir))
    return candidate_signature_from_payload(payload, policy_source=policy_source)


def collect_manifest_signatures(
    manifest_payload: dict[str, Any],
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
) -> set[str]:
    signatures: set[str] = set()
    for record in manifest_payload.get("records", []):
        signature = record_signature(
            record_payload=record,
            base_dir=base_dir,
            policy_source=policy_source,
        )
        if signature:
            signatures.add(signature)
    return signatures


def record_signature(
    record_payload: dict[str, Any],
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
) -> str | None:
    candidate_path = record_payload.get("candidate_path")
    if candidate_path:
        resolved_candidate_path = _resolve_path(candidate_path, base_dir)
        if resolved_candidate_path.exists():
            return candidate_signature_from_path(
                resolved_candidate_path,
                base_dir=base_dir,
                policy_source=policy_source,
            )

    uniform_bit_width = record_payload.get("uniform_bit_width")
    if uniform_bit_width is not None:
        return f"uniform::{int(uniform_bit_width)}"

    return None


def select_novel_candidates(
    executed_manifest_payload: dict[str, Any],
    candidate_manifest_payload: dict[str, Any],
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
    limit: int = 1,
    exploration_weight: float = 0.5,
    diversity_weight: float = 0.15,
) -> list[dict[str, Any]]:
    existing_signatures = collect_manifest_signatures(
        executed_manifest_payload,
        base_dir=base_dir,
        policy_source=policy_source,
    )
    existing_assignment_pool: list[dict[str, int]] = []
    for record in executed_manifest_payload.get("records", []):
        candidate_path = record.get("candidate_path")
        if not candidate_path:
            continue
        resolved_candidate_path = _resolve_path(candidate_path, base_dir)
        if not resolved_candidate_path.exists():
            continue
        existing_assignment_pool.append(
            candidate_assignments_from_path(
                resolved_candidate_path,
                base_dir=base_dir,
                policy_source=policy_source,
            )
        )

    candidate_pool: list[dict[str, Any]] = []
    for item in candidate_manifest_payload.get("candidates", []):
        candidate_path = _resolve_path(item["path"], base_dir)
        signature = candidate_signature_from_path(
            candidate_path,
            base_dir=base_dir,
            policy_source=policy_source,
        )
        if signature in existing_signatures:
            continue
        candidate_pool.append(
            {
                "rank": int(item["rank"]),
                "path": str(candidate_path),
                "signature": signature,
                "fitness": float(item["fitness"]),
                "conservative_prediction": item.get("conservative_prediction"),
                "budget_alignment_score": item.get("budget_alignment_score"),
                "reference_accuracy": item.get("reference_accuracy"),
                "reference_advantage_score": item.get("reference_advantage_score"),
                "estimated_average_bit_width": float(item["estimated_average_bit_width"]),
                "estimated_weight_footprint_gb": float(item["estimated_weight_footprint_gb"]),
                "prediction_uncertainty": item.get("prediction_uncertainty"),
                "bit_counts": item.get("bit_counts", {}),
                "_assignments": candidate_assignments_from_path(
                    candidate_path,
                    base_dir=base_dir,
                    policy_source=policy_source,
                ),
            }
        )

    selected: list[dict[str, Any]] = []
    selected_signatures: set[str] = set()
    selected_assignment_pool: list[dict[str, int]] = []
    remaining_candidates = list(candidate_pool)

    while remaining_candidates and len(selected) < limit:
        best_index = -1
        best_score = float("-inf")
        best_payload: dict[str, Any] | None = None

        comparison_pool = existing_assignment_pool + selected_assignment_pool
        for index, item in enumerate(remaining_candidates):
            if item["signature"] in selected_signatures:
                continue
            novelty_score = estimate_candidate_novelty(item["_assignments"], comparison_pool)
            acquisition = estimate_candidate_acquisition_score(
                item,
                novelty_score,
                exploration_weight=exploration_weight,
                diversity_weight=diversity_weight,
            )
            acquisition_score = acquisition["selection_acquisition_score"]
            if acquisition_score > best_score:
                best_score = acquisition_score
                best_index = index
                best_payload = {
                    **item,
                    **acquisition,
                }

        if best_index < 0 or best_payload is None:
            break

        selected.append(
            {
                key: value
                for key, value in best_payload.items()
                if key != "_assignments"
            }
        )
        selected_signatures.add(best_payload["signature"])
        selected_assignment_pool.append(best_payload["_assignments"])
        del remaining_candidates[best_index]

    return selected


def append_record_if_novel(
    manifest_payload: dict[str, Any],
    record_payload: dict[str, Any],
    base_dir: str | Path,
    policy_source: str = "llmcompressor",
) -> dict[str, Any]:
    manifest_copy = copy.deepcopy(manifest_payload)
    records = list(manifest_copy.get("records", []))
    new_signature = record_signature(record_payload, base_dir=base_dir, policy_source=policy_source)
    existing_signatures = collect_manifest_signatures(
        manifest_copy,
        base_dir=base_dir,
        policy_source=policy_source,
    )
    if new_signature and new_signature in existing_signatures:
        return manifest_copy
    records.append(record_payload)
    manifest_copy["records"] = records
    return manifest_copy


def build_executed_record(
    *,
    policy_id: str,
    task_name: str,
    candidate_path: str | Path,
    report_path: str | Path,
    evaluation_path: str | Path,
    sensitivity_profile_path: str | Path,
    sensitivity_field: str,
    provenance: str,
    search_result_path: str | Path | None = None,
    surrogate_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "policy_id": policy_id,
        "task_name": task_name,
        "candidate_path": str(candidate_path),
        "report_path": str(report_path),
        "evaluation_path": str(evaluation_path),
        "sensitivity_profile_path": str(sensitivity_profile_path),
        "sensitivity_field": sensitivity_field,
        "provenance": provenance,
    }
    if search_result_path is not None:
        record["search_result_path"] = str(search_result_path)
    if surrogate_summary_path is not None:
        record["surrogate_summary_path"] = str(surrogate_summary_path)
    return record


def best_record_by_accuracy(
    manifest_payload: dict[str, Any],
    base_dir: str | Path,
    *,
    task_name: str | None = None,
    provenance_prefix: str | None = None,
) -> dict[str, Any] | None:
    best_record: dict[str, Any] | None = None
    best_accuracy = float("-inf")
    for record in manifest_payload.get("records", []):
        if task_name and record.get("task_name") != task_name:
            continue
        if provenance_prefix and not str(record.get("provenance", "")).startswith(provenance_prefix):
            continue
        evaluation_path = record.get("evaluation_path")
        if not evaluation_path:
            continue
        resolved_path = _resolve_path(evaluation_path, base_dir)
        if not resolved_path.exists():
            continue
        evaluation_payload = _load_json(resolved_path)
        accuracy = float(evaluation_payload.get("accuracy", 0.0))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_record = copy.deepcopy(record)
            best_record["accuracy"] = accuracy
    return best_record


def artifact_dir_from_record(
    record_payload: dict[str, Any],
    base_dir: str | Path,
) -> str | None:
    report_path = record_payload.get("report_path")
    if not report_path:
        return None
    resolved_report_path = _resolve_path(report_path, base_dir)
    if not resolved_report_path.exists():
        return None
    report_payload = _load_json(resolved_report_path)
    output_dir = report_payload.get("output_dir")
    if not output_dir:
        return None
    return str(output_dir)


def to_relative_path(path: str | Path, base_dir: str | Path) -> str:
    resolved = _resolve_path(path, base_dir)
    root = Path(base_dir).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir).resolve() / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
