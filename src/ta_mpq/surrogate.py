from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.search import (
    SearchGroup,
    build_search_groups,
    estimate_candidate_average_bit_width,
    estimate_candidate_weight_footprint_gb,
)
from ta_mpq.sensitivity import group_sensitivity_overrides_from_profile


BIT_ORDER = (2, 3, 4, 8, 16)
BIT_NORMALIZATION = {
    2: 0.0,
    3: 0.25,
    4: 0.5,
    8: 1.0,
    16: 1.25,
}
DEFAULT_SURROGATE_TARGET = "accuracy"
COMPONENT_FEATURES = (
    "linear_attn.in_proj_qkv",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_z",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)


@dataclass(frozen=True, slots=True)
class SurrogateExample:
    policy_id: str
    task_name: str
    feature_values: dict[str, float]
    target_values: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SurrogateDataset:
    task_name: str
    grouping: str
    target_metric: str
    feature_names: tuple[str, ...]
    examples: tuple[SurrogateExample, ...]
    baseline_metrics: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "grouping": self.grouping,
            "target_metric": self.target_metric,
            "feature_names": list(self.feature_names),
            "num_features": len(self.feature_names),
            "num_examples": len(self.examples),
            "baseline_metrics": self.baseline_metrics,
            "examples": [example.to_dict() for example in self.examples],
        }


def build_surrogate_dataset_from_manifest(
    manifest_payload: dict[str, Any],
    target_metric: str = DEFAULT_SURROGATE_TARGET,
    uniform_baseline_bit_width: int | None = None,
) -> SurrogateDataset:
    grouping = str(manifest_payload.get("grouping", "per_block_component"))
    task_name = str(manifest_payload["task_name"])
    baseline_metrics = _collect_task_baselines(
        manifest_payload,
        uniform_baseline_bit_width=uniform_baseline_bit_width,
    )
    if "advantage_over_uniform" in target_metric and baseline_metrics.get("uniform_accuracy") is None:
        requested_uniform = (
            f"uniform {uniform_baseline_bit_width}-bit"
            if uniform_baseline_bit_width is not None
            else "a uniform baseline"
        )
        raise ValueError(
            f"Manifest does not contain a usable record for {requested_uniform}, "
            f"so target_metric={target_metric!r} cannot be computed."
        )
    examples: list[SurrogateExample] = []

    for record in manifest_payload["records"]:
        examples.append(
            build_surrogate_example_from_record(
                record_payload=record,
                grouping=grouping,
                baseline_metrics=baseline_metrics,
            )
        )

    feature_names = tuple(
        sorted(
            {
                feature_name
                for example in examples
                for feature_name in example.feature_values
            }
        )
    )
    return SurrogateDataset(
        task_name=task_name,
        grouping=grouping,
        target_metric=target_metric,
        feature_names=feature_names,
        examples=tuple(examples),
        baseline_metrics=baseline_metrics,
    )


def resolve_manifest_paths(
    manifest_payload: dict[str, Any],
    base_dir: str | Path,
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest_payload))
    root = Path(base_dir)
    for record in resolved.get("records", []):
        for field_name in (
            "candidate_path",
            "report_path",
            "evaluation_path",
            "sensitivity_profile_path",
        ):
            value = record.get(field_name)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                record[field_name] = str(root / path)
    return resolved


def build_surrogate_example_from_record(
    record_payload: dict[str, Any],
    grouping: str,
    baseline_metrics: dict[str, float | None] | None = None,
) -> SurrogateExample:
    report_payload = _load_json(record_payload["report_path"])
    evaluation_payload = _load_json(record_payload["evaluation_path"])
    sensitivity_payload = _load_json(record_payload["sensitivity_profile_path"])
    layer_stats = _layer_stats_from_report(report_payload)
    groups = _groups_from_record(
        record_payload=record_payload,
        layer_stats=layer_stats,
        grouping=grouping,
        sensitivity_payload=sensitivity_payload,
    )
    group_bits = _group_bits_from_record(
        record_payload=record_payload,
        groups=groups,
    )

    if set(group_bits) != {group.name for group in groups}:
        missing = sorted({group.name for group in groups} - set(group_bits))
        if missing:
            raise ValueError(f"Record is missing group assignments for: {missing[:5]}")

    feature_values = extract_surrogate_features(
        groups=groups,
        group_bits=group_bits,
        report_payload=report_payload,
        sensitivity_payload=sensitivity_payload,
    )
    target_values = extract_surrogate_targets(
        evaluation_payload,
        native_accuracy=_resolve_native_accuracy_for_record(record_payload, baseline_metrics),
        uniform_accuracy=(
            float(baseline_metrics["uniform_accuracy"])
            if baseline_metrics and baseline_metrics.get("uniform_accuracy") is not None
            else None
        ),
    )
    metadata = {
        "record_label": str(record_payload["policy_id"]),
        "candidate_path": record_payload.get("candidate_path"),
        "report_path": record_payload["report_path"],
        "evaluation_path": record_payload["evaluation_path"],
        "sensitivity_profile_path": record_payload["sensitivity_profile_path"],
        "provenance": record_payload.get("provenance", "executed_policy"),
        "model_id": report_payload.get("model_id"),
        "group_bit_assignments": group_bits,
    }
    return SurrogateExample(
        policy_id=str(record_payload["policy_id"]),
        task_name=str(record_payload["task_name"]),
        feature_values=feature_values,
        target_values=target_values,
        metadata=metadata,
    )


def extract_surrogate_features(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    report_payload: dict[str, Any],
    sensitivity_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    total_groups = len(groups)
    total_params = sum(group.parameter_count for group in groups)
    sorted_groups = sorted(groups, key=lambda group: group.sensitivity, reverse=True)
    top_quartile_cutoff = max(1, total_groups // 4)
    high_sensitivity_groups = {group.name for group in sorted_groups[:top_quartile_cutoff]}
    low_sensitivity_groups = {group.name for group in sorted_groups[-top_quartile_cutoff:]}
    profile_group_lookup = _profile_group_lookup(sensitivity_payload)

    feature_values: dict[str, float] = {
        "num_groups": float(total_groups),
        "total_parameters_billions": total_params / 1_000_000_000,
        "estimated_average_bit_width": estimate_candidate_average_bit_width(groups, group_bits),
        "estimated_weight_footprint_gb": estimate_candidate_weight_footprint_gb(groups, group_bits),
        "report_estimated_average_bit_width": float(
            report_payload.get("estimated_average_bit_width", 0.0)
        ),
        "report_estimated_weight_footprint_gb": float(
            report_payload.get("estimated_weight_footprint_gb", 0.0)
        ),
        "rule_hit_rate": _rule_hit_rate(report_payload),
        "policy_alignment_score": _policy_alignment_score(groups, group_bits),
        "avg_8bit_group_sensitivity": _mean_sensitivity_for_bit(groups, group_bits, 8),
        "avg_4bit_group_sensitivity": _mean_sensitivity_for_bit(groups, group_bits, 4),
        "avg_16bit_group_sensitivity": _mean_sensitivity_for_bit(groups, group_bits, 16),
        "high_sensitivity_8bit_fraction": _sensitivity_bucket_fraction(
            groups, group_bits, high_sensitivity_groups, 8
        ),
        "low_sensitivity_8bit_fraction": _sensitivity_bucket_fraction(
            groups, group_bits, low_sensitivity_groups, 8
        ),
        "high_sensitivity_16bit_fraction": _sensitivity_bucket_fraction(
            groups, group_bits, high_sensitivity_groups, 16
        ),
        "activation_saliency_alignment_score": _profile_signal_alignment_score(
            groups,
            group_bits,
            profile_group_lookup,
            field_name="normalized_activation_score",
        ),
        "kl_divergence_alignment_score": _profile_signal_alignment_score(
            groups,
            group_bits,
            profile_group_lookup,
            field_name="normalized_kl_divergence_score",
        ),
        "avg_8bit_activation_saliency": _profile_signal_mean_for_bit(
            groups,
            group_bits,
            profile_group_lookup,
            bit_width=8,
            field_name="normalized_activation_score",
        ),
        "avg_16bit_activation_saliency": _profile_signal_mean_for_bit(
            groups,
            group_bits,
            profile_group_lookup,
            bit_width=16,
            field_name="normalized_activation_score",
        ),
        "avg_8bit_kl_divergence": _profile_signal_mean_for_bit(
            groups,
            group_bits,
            profile_group_lookup,
            bit_width=8,
            field_name="normalized_kl_divergence_score",
        ),
        "avg_16bit_kl_divergence": _profile_signal_mean_for_bit(
            groups,
            group_bits,
            profile_group_lookup,
            bit_width=16,
            field_name="normalized_kl_divergence_score",
        ),
    }

    for bit_width in BIT_ORDER:
        count, param_fraction = _bit_statistics(groups, group_bits, bit_width)
        feature_values[f"bit_count_{bit_width}"] = float(count)
        feature_values[f"param_fraction_{bit_width}"] = param_fraction

    for component_name in COMPONENT_FEATURES:
        sanitized = _sanitize_component_name(component_name)
        feature_values[f"{sanitized}_8bit_fraction"] = _component_bit_fraction(
            groups,
            group_bits,
            component_name,
            8,
        )
        feature_values[f"{sanitized}_16bit_fraction"] = _component_bit_fraction(
            groups,
            group_bits,
            component_name,
            16,
        )
        feature_values[f"{sanitized}_4bit_fraction"] = _component_bit_fraction(
            groups,
            group_bits,
            component_name,
            4,
        )

    downgraded = 0.0
    backend_projection = report_payload.get("policy")
    if backend_projection:
        downgraded = float(_count_unsupported_bits(group_bits))
    feature_values["unsupported_low_bit_groups"] = downgraded
    return feature_values


def extract_surrogate_targets(
    evaluation_payload: dict[str, Any],
    native_accuracy: float | None = None,
    uniform_accuracy: float | None = None,
) -> dict[str, float]:
    accuracy = float(evaluation_payload["accuracy"])
    target_values = {
        "accuracy": accuracy,
        "mean_latency_sec": float(evaluation_payload["mean_latency_sec"]),
        "mean_total_peak_memory_mb": float(evaluation_payload["mean_total_peak_memory_mb"]),
    }
    if native_accuracy is not None:
        target_values["accuracy_advantage_over_native"] = accuracy - native_accuracy
    if uniform_accuracy is not None:
        target_values["accuracy_advantage_over_uniform"] = accuracy - uniform_accuracy
    best_baseline = _best_available_baseline(native_accuracy, uniform_accuracy)
    if best_baseline is not None:
        target_values["accuracy_advantage_over_best_baseline"] = accuracy - best_baseline
    return target_values


def build_group_value_prior_from_dataset(
    dataset_payload: dict[str, Any],
    target_metric: str = "accuracy_advantage_over_best_baseline",
    min_support: int = 2,
) -> dict[str, Any]:
    group_targets: dict[str, dict[int, list[float]]] = {}
    component_targets: dict[str, dict[int, list[float]]] = {}
    examples = dataset_payload.get("examples", [])

    for example in examples:
        target_values = example.get("target_values", {})
        target_value = target_values.get(target_metric)
        if target_value is None:
            target_value = target_values.get("accuracy")
        target = float(target_value)
        group_bit_assignments = (
            example.get("metadata", {}).get("group_bit_assignments")
            or {}
        )
        for group_name, bit_width in group_bit_assignments.items():
            bit = int(bit_width)
            if bit not in (4, 8, 16):
                continue
            group_targets.setdefault(str(group_name), {4: [], 8: [], 16: []})[bit].append(target)
            component_name = _component_name_from_group_name(str(group_name))
            component_targets.setdefault(component_name, {4: [], 8: [], 16: []})[bit].append(target)

    group_scores = {
        group_name: _summarize_value_score(bit_targets, min_support=min_support)
        for group_name, bit_targets in group_targets.items()
    }
    component_scores = {
        component_name: _summarize_value_score(bit_targets, min_support=min_support)
        for component_name, bit_targets in component_targets.items()
    }
    return {
        "task_name": dataset_payload.get("task_name"),
        "target_metric": target_metric,
        "num_examples": len(examples),
        "baseline_metrics": dataset_payload.get("baseline_metrics", {}),
        "group_scores": group_scores,
        "component_scores": component_scores,
    }


def build_ablation_adjusted_group_value_prior(
    base_prior_payload: dict[str, Any],
    ablation_profile_payloads: list[dict[str, Any]],
    *,
    zero_drop_tolerance: float = 1e-9,
    no_evidence_four_bit_penalty: float = 0.05,
    no_evidence_sixteen_bit_cap: float = 0.01,
) -> dict[str, Any]:
    adjusted_payload = copy.deepcopy(base_prior_payload)
    group_scores = adjusted_payload.setdefault("group_scores", {})
    applied_adjustments: list[dict[str, Any]] = []

    for profile in ablation_profile_payloads:
        for group_payload in profile.get("groups", []):
            group_name = str(group_payload["name"])
            reference_bit_width = int(group_payload["reference_bit_width"])
            ablated_bit_width = int(group_payload["ablated_bit_width"])
            accuracy_drop = float(group_payload.get("accuracy_drop", 0.0))
            positive_drop = float(group_payload.get("positive_accuracy_drop", max(accuracy_drop, 0.0)))

            score_payload = copy.deepcopy(group_scores.get(group_name, {}))
            original_score = float(score_payload.get("score", 0.0))
            original_uplift_8 = float(score_payload.get("uplift_8_over_4", original_score))
            original_uplift_16 = float(score_payload.get("uplift_16_over_8", 0.0))
            adjusted_score = original_score
            adjusted_uplift_8 = original_uplift_8
            adjusted_uplift_16 = original_uplift_16
            adjustment_reason = "no_change"

            if reference_bit_width == 8 and ablated_bit_width == 4:
                if positive_drop <= zero_drop_tolerance:
                    adjusted_uplift_8 = min(
                        original_uplift_8,
                        -max(no_evidence_four_bit_penalty, abs(original_score)),
                    )
                    adjustment_reason = "8_to_4_no_drop"
                else:
                    adjusted_uplift_8 = max(original_uplift_8, positive_drop)
                    adjustment_reason = "8_to_4_positive_drop"
            elif reference_bit_width == 16 and ablated_bit_width == 8:
                if positive_drop <= zero_drop_tolerance:
                    if original_uplift_16 > 0:
                        adjusted_uplift_16 = min(original_uplift_16, no_evidence_sixteen_bit_cap)
                        adjustment_reason = "16_to_8_no_drop"
                else:
                    adjusted_uplift_16 = max(original_uplift_16, positive_drop)
                    adjustment_reason = "16_to_8_positive_drop"

            adjusted_score = _overall_value_score_from_uplifts(
                adjusted_uplift_8,
                adjusted_uplift_16,
            )
            preferred_bit = _preferred_bit_from_uplifts(
                adjusted_uplift_8,
                adjusted_uplift_16,
                fallback=int(score_payload.get("preferred_bit", 4)),
            )
            score_payload["score"] = adjusted_score
            score_payload["uplift_8_over_4"] = adjusted_uplift_8
            score_payload["uplift_16_over_8"] = adjusted_uplift_16
            score_payload["uplift_16_over_4"] = adjusted_uplift_8 + adjusted_uplift_16
            score_payload["preferred_bit"] = float(preferred_bit)
            score_payload["ablation_reference_bit_width"] = reference_bit_width
            score_payload["ablation_ablated_bit_width"] = ablated_bit_width
            score_payload["ablation_accuracy_drop"] = accuracy_drop
            score_payload["ablation_positive_accuracy_drop"] = positive_drop
            score_payload["ablation_adjustment_reason"] = adjustment_reason
            group_scores[group_name] = score_payload

            applied_adjustments.append(
                {
                    "name": group_name,
                    "reference_bit_width": reference_bit_width,
                    "ablated_bit_width": ablated_bit_width,
                    "accuracy_drop": accuracy_drop,
                    "original_score": original_score,
                    "adjusted_score": adjusted_score,
                    "original_uplift_8_over_4": original_uplift_8,
                    "adjusted_uplift_8_over_4": adjusted_uplift_8,
                    "original_uplift_16_over_8": original_uplift_16,
                    "adjusted_uplift_16_over_8": adjusted_uplift_16,
                    "reason": adjustment_reason,
                }
            )

    adjusted_payload["ablation_adjustments"] = applied_adjustments
    adjusted_payload["ablation_adjustment_config"] = {
        "zero_drop_tolerance": zero_drop_tolerance,
        "no_evidence_four_bit_penalty": no_evidence_four_bit_penalty,
        "no_evidence_sixteen_bit_cap": no_evidence_sixteen_bit_cap,
    }
    return adjusted_payload


def train_surrogate_model(
    dataset_payload: dict[str, Any],
    target_metric: str = DEFAULT_SURROGATE_TARGET,
    random_seed: int = 0,
    ensemble_size: int = 8,
) -> dict[str, Any]:
    examples = dataset_payload["examples"]
    feature_names = list(dataset_payload["feature_names"])
    rows = [example["feature_values"] for example in examples]
    targets = [float(example["target_values"][target_metric]) for example in examples]
    matrix = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]

    if not matrix:
        raise ValueError("Surrogate dataset is empty")

    if len(matrix) < 3:
        return train_mean_baseline_surrogate(dataset_payload, target_metric=target_metric)

    try:
        import xgboost as xgb
    except ModuleNotFoundError:
        return train_mean_baseline_surrogate(dataset_payload, target_metric=target_metric)

    if len(matrix) >= 5 and ensemble_size > 1:
        return _train_xgboost_ensemble_surrogate(
            dataset_payload=dataset_payload,
            matrix=matrix,
            targets=targets,
            feature_names=feature_names,
            target_metric=target_metric,
            random_seed=random_seed,
            ensemble_size=ensemble_size,
        )

    return _train_single_xgboost_surrogate(
        dataset_payload=dataset_payload,
        matrix=matrix,
        targets=targets,
        feature_names=feature_names,
        target_metric=target_metric,
        random_seed=random_seed,
    )


def _train_single_xgboost_surrogate(
    dataset_payload: dict[str, Any],
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    target_metric: str,
    random_seed: int,
) -> dict[str, Any]:
    import xgboost as xgb

    examples = dataset_payload["examples"]
    booster = _fit_xgboost_booster(
        matrix=matrix,
        targets=targets,
        feature_names=feature_names,
        random_seed=random_seed,
    )
    training_matrix = xgb.DMatrix(matrix, label=targets, feature_names=feature_names)
    predictions = [float(value) for value in booster.predict(training_matrix)]
    validation_predictions = _leave_one_out_predictions(
        matrix=matrix,
        targets=targets,
        feature_names=feature_names,
        random_seed=random_seed,
    )
    target_mean, target_std, target_range, target_scale = _target_distribution_stats(targets)
    validation_mae = (
        _mean_absolute_error(targets, validation_predictions) if validation_predictions else None
    )
    validation_rmse = (
        _root_mean_squared_error(targets, validation_predictions)
        if validation_predictions
        else None
    )
    validation_spearman = (
        _spearman_rank_correlation(targets, validation_predictions)
        if validation_predictions
        else None
    )
    validation_top1_hit_rate = (
        _top_k_hit_rate(targets, validation_predictions, k=1)
        if validation_predictions
        else None
    )
    validation_top3_hit_rate = (
        _top_k_hit_rate(targets, validation_predictions, k=3)
        if validation_predictions
        else None
    )
    validation_pairwise_accuracy = (
        _pairwise_ranking_accuracy(targets, validation_predictions)
        if validation_predictions
        else None
    )
    calibration_weight = _calibration_weight(
        num_examples=len(examples),
        validation_rmse=validation_rmse,
        validation_spearman=validation_spearman,
        validation_top1_hit_rate=validation_top1_hit_rate,
        validation_top3_hit_rate=validation_top3_hit_rate,
        validation_pairwise_accuracy=validation_pairwise_accuracy,
        target_scale=target_scale,
    )
    uncertainty_floor = _uncertainty_floor(
        validation_rmse=validation_rmse,
        target_std=target_std,
        calibration_weight=calibration_weight,
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        booster.save_model(str(temp_path))
        model_json = temp_path.read_text(encoding="utf-8")
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "backend": "xgboost",
        "target_metric": target_metric,
        "feature_names": feature_names,
        "num_examples": len(examples),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_range": target_range,
        "target_scale": target_scale,
        "training_mae": _mean_absolute_error(targets, predictions),
        "training_rmse": _root_mean_squared_error(targets, predictions),
        "validation_strategy": "leave_one_out" if len(examples) >= 3 else "none",
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_spearman": validation_spearman,
        "validation_top1_hit_rate": validation_top1_hit_rate,
        "validation_top3_hit_rate": validation_top3_hit_rate,
        "validation_pairwise_accuracy": validation_pairwise_accuracy,
        "calibration_weight": calibration_weight,
        "uncertainty_floor": uncertainty_floor,
        "feature_importances": _sorted_feature_importances_from_booster(
            feature_names,
            booster,
        ),
        "predictions": [
            {
                "policy_id": example["policy_id"],
                "target": target,
                "prediction": prediction,
            }
            for example, target, prediction in zip(examples, targets, predictions, strict=True)
        ],
        "validation_predictions": [
            {
                "policy_id": example["policy_id"],
                "target": target,
                "prediction": prediction,
            }
            for example, target, prediction in zip(
                examples,
                targets,
                validation_predictions,
                strict=True,
            )
        ]
        if validation_predictions
        else [],
        "model_json": model_json,
        "task_name": dataset_payload["task_name"],
    }


def _train_xgboost_ensemble_surrogate(
    dataset_payload: dict[str, Any],
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    target_metric: str,
    random_seed: int,
    ensemble_size: int,
) -> dict[str, Any]:
    examples = dataset_payload["examples"]
    boosters = _fit_xgboost_ensemble(
        matrix=matrix,
        targets=targets,
        feature_names=feature_names,
        random_seed=random_seed,
        ensemble_size=ensemble_size,
    )
    predictions, prediction_uncertainties = _ensemble_predictions(
        boosters=boosters,
        matrix=matrix,
        feature_names=feature_names,
    )
    validation_predictions, validation_uncertainties = _leave_one_out_ensemble_predictions(
        matrix=matrix,
        targets=targets,
        feature_names=feature_names,
        random_seed=random_seed,
        ensemble_size=ensemble_size,
    )
    target_mean, target_std, target_range, target_scale = _target_distribution_stats(targets)
    validation_mae = (
        _mean_absolute_error(targets, validation_predictions) if validation_predictions else None
    )
    validation_rmse = (
        _root_mean_squared_error(targets, validation_predictions)
        if validation_predictions
        else None
    )
    validation_spearman = (
        _spearman_rank_correlation(targets, validation_predictions)
        if validation_predictions
        else None
    )
    validation_top1_hit_rate = (
        _top_k_hit_rate(targets, validation_predictions, k=1)
        if validation_predictions
        else None
    )
    validation_top3_hit_rate = (
        _top_k_hit_rate(targets, validation_predictions, k=3)
        if validation_predictions
        else None
    )
    validation_pairwise_accuracy = (
        _pairwise_ranking_accuracy(targets, validation_predictions)
        if validation_predictions
        else None
    )
    calibration_weight = _calibration_weight(
        num_examples=len(examples),
        validation_rmse=validation_rmse,
        validation_spearman=validation_spearman,
        validation_top1_hit_rate=validation_top1_hit_rate,
        validation_top3_hit_rate=validation_top3_hit_rate,
        validation_pairwise_accuracy=validation_pairwise_accuracy,
        target_scale=target_scale,
    )
    uncertainty_floor = _uncertainty_floor(
        validation_rmse=validation_rmse,
        target_std=target_std,
        calibration_weight=calibration_weight,
    )
    model_json = json.dumps(
        {
            "ensemble_models": [_serialize_booster(booster) for booster in boosters],
        },
        sort_keys=True,
    )
    return {
        "backend": "xgboost_ensemble",
        "target_metric": target_metric,
        "feature_names": feature_names,
        "num_examples": len(examples),
        "ensemble_size": ensemble_size,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_range": target_range,
        "target_scale": target_scale,
        "training_mae": _mean_absolute_error(targets, predictions),
        "training_rmse": _root_mean_squared_error(targets, predictions),
        "training_prediction_std_mean": statistics.fmean(prediction_uncertainties),
        "validation_strategy": "leave_one_out" if len(examples) >= 3 else "none",
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_spearman": validation_spearman,
        "validation_top1_hit_rate": validation_top1_hit_rate,
        "validation_top3_hit_rate": validation_top3_hit_rate,
        "validation_pairwise_accuracy": validation_pairwise_accuracy,
        "validation_prediction_std_mean": statistics.fmean(validation_uncertainties)
        if validation_uncertainties
        else None,
        "calibration_weight": calibration_weight,
        "uncertainty_floor": uncertainty_floor,
        "feature_importances": _sorted_feature_importances_from_boosters(
            feature_names=feature_names,
            boosters=boosters,
        ),
        "predictions": [
            {
                "policy_id": example["policy_id"],
                "target": target,
                "prediction": prediction,
                "prediction_std": prediction_std,
            }
            for example, target, prediction, prediction_std in zip(
                examples,
                targets,
                predictions,
                prediction_uncertainties,
                strict=True,
            )
        ],
        "validation_predictions": [
            {
                "policy_id": example["policy_id"],
                "target": target,
                "prediction": prediction,
                "prediction_std": prediction_std,
            }
            for example, target, prediction, prediction_std in zip(
                examples,
                targets,
                validation_predictions,
                validation_uncertainties,
                strict=True,
            )
        ]
        if validation_predictions
        else [],
        "model_json": model_json,
        "task_name": dataset_payload["task_name"],
    }


def train_mean_baseline_surrogate(
    dataset_payload: dict[str, Any],
    target_metric: str = DEFAULT_SURROGATE_TARGET,
) -> dict[str, Any]:
    examples = dataset_payload["examples"]
    targets = [float(example["target_values"][target_metric]) for example in examples]
    mean_target = sum(targets) / len(targets)
    predictions = [mean_target for _ in targets]
    target_mean, target_std, target_range, target_scale = _target_distribution_stats(targets)
    return {
        "backend": "mean_baseline",
        "target_metric": target_metric,
        "feature_names": list(dataset_payload["feature_names"]),
        "num_examples": len(examples),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_range": target_range,
        "target_scale": target_scale,
        "training_mae": _mean_absolute_error(targets, predictions),
        "training_rmse": _root_mean_squared_error(targets, predictions),
        "validation_strategy": "none",
        "validation_mae": None,
        "validation_rmse": None,
        "validation_spearman": None,
        "validation_top1_hit_rate": None,
        "validation_top3_hit_rate": None,
        "validation_pairwise_accuracy": None,
        "calibration_weight": 0.0,
        "uncertainty_floor": 0.0,
        "feature_importances": [],
        "predictions": [
            {
                "policy_id": example["policy_id"],
                "target": target,
                "prediction": prediction,
            }
            for example, target, prediction in zip(examples, targets, predictions, strict=True)
        ],
        "model_json": "",
        "task_name": dataset_payload["task_name"],
    }


def predict_surrogate_target(
    feature_values: dict[str, float],
    surrogate_summary_payload: dict[str, Any],
    model_json: str,
) -> float:
    mean_prediction, _ = predict_surrogate_distribution(
        feature_values=feature_values,
        surrogate_summary_payload=surrogate_summary_payload,
        model_json=model_json,
    )
    return mean_prediction


def predict_surrogate_distribution(
    feature_values: dict[str, float],
    surrogate_summary_payload: dict[str, Any],
    model_json: str,
) -> tuple[float, float]:
    predictor = build_surrogate_distribution_predictor(
        surrogate_summary_payload=surrogate_summary_payload,
        model_json=model_json,
    )
    return predictor(feature_values)


def predict_surrogate_targets(
    feature_rows: list[dict[str, float]],
    surrogate_summary_payload: dict[str, Any],
    model_json: str,
) -> list[float]:
    predictor = build_surrogate_distribution_predictor(
        surrogate_summary_payload=surrogate_summary_payload,
        model_json=model_json,
    )
    return [predictor(feature_values)[0] for feature_values in feature_rows]


def build_surrogate_predictor(
    surrogate_summary_payload: dict[str, Any],
    model_json: str,
) -> Any:
    distribution_predictor = build_surrogate_distribution_predictor(
        surrogate_summary_payload=surrogate_summary_payload,
        model_json=model_json,
    )

    def predict_mean(feature_values: dict[str, float]) -> float:
        mean_prediction, _ = distribution_predictor(feature_values)
        return mean_prediction

    return predict_mean


def build_surrogate_distribution_predictor(
    surrogate_summary_payload: dict[str, Any],
    model_json: str,
) -> Any:
    backend = str(surrogate_summary_payload.get("backend", "mean_baseline"))
    feature_names = [str(name) for name in surrogate_summary_payload.get("feature_names", [])]
    if backend == "mean_baseline" or not model_json:
        fallback = _mean_prediction_from_summary(surrogate_summary_payload)

        def predict_mean_baseline(feature_values: dict[str, float]) -> tuple[float, float]:
            return fallback, 0.0

        return predict_mean_baseline

    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost is required to run surrogate predictions") from exc

    if backend == "xgboost_ensemble":
        ensemble_payload = json.loads(model_json)
        boosters = [
            _load_booster_from_json(model_payload)
            for model_payload in ensemble_payload.get("ensemble_models", [])
        ]

        def predict_xgboost_ensemble(feature_values: dict[str, float]) -> tuple[float, float]:
            matrix = xgb.DMatrix(
                [[float(feature_values.get(feature_name, 0.0)) for feature_name in feature_names]],
                feature_names=feature_names,
            )
            predictions = [float(booster.predict(matrix)[0]) for booster in boosters]
            mean_prediction = statistics.fmean(predictions)
            prediction_std = statistics.pstdev(predictions) if len(predictions) > 1 else 0.0
            return _apply_surrogate_calibration(
                mean_prediction,
                prediction_std,
                surrogate_summary_payload,
            )

        return predict_xgboost_ensemble

    booster = _load_booster_from_json(model_json)

    def predict_xgboost(feature_values: dict[str, float]) -> tuple[float, float]:
        matrix = xgb.DMatrix(
            [[float(feature_values.get(feature_name, 0.0)) for feature_name in feature_names]],
            feature_names=feature_names,
        )
        return _apply_surrogate_calibration(
            float(booster.predict(matrix)[0]),
            0.0,
            surrogate_summary_payload,
        )

    return predict_xgboost


def save_surrogate_dataset(path: str | Path, dataset: SurrogateDataset) -> None:
    _save_json(Path(path), dataset.to_dict())


def save_group_value_prior(path: str | Path, payload: dict[str, Any]) -> None:
    _save_json(Path(path), payload)


def _groups_from_record(
    record_payload: dict[str, Any],
    layer_stats: list[LinearLayerStat],
    grouping: str,
    sensitivity_payload: dict[str, Any],
) -> list[SearchGroup]:
    return build_search_groups(
        layer_stats,
        grouping=grouping,
        sensitivity_overrides=group_sensitivity_overrides_from_profile(
            sensitivity_payload,
            field=record_payload.get("sensitivity_field", "combined_sensitivity"),
        ),
    )


def _group_bits_from_record(
    record_payload: dict[str, Any],
    groups: list[SearchGroup],
) -> dict[str, int]:
    candidate_path = record_payload.get("candidate_path")
    if candidate_path:
        candidate_payload = _load_json(candidate_path)
        return {
            str(name): int(bit)
            for name, bit in candidate_payload["group_bit_assignments"].items()
        }

    uniform_bit_width = record_payload.get("uniform_bit_width")
    if uniform_bit_width is None:
        raise ValueError("Record requires either candidate_path or uniform_bit_width")
    return {group.name: int(uniform_bit_width) for group in groups}


def _layer_stats_from_report(report_payload: dict[str, Any]) -> list[LinearLayerStat]:
    return [
        LinearLayerStat(
            name=str(item["name"]),
            parameter_count=int(item["parameter_count"]),
            in_features=int(item.get("in_features", 0)),
            out_features=int(item.get("out_features", 0)),
        )
        for item in report_payload["layer_stats"]
    ]


def _bit_statistics(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    bit_width: int,
) -> tuple[int, float]:
    matching = [group for group in groups if group_bits[group.name] == bit_width]
    if not groups:
        return 0, 0.0
    total_params = sum(group.parameter_count for group in groups)
    param_fraction = sum(group.parameter_count for group in matching) / total_params
    return len(matching), param_fraction


def _policy_alignment_score(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        denominator += group.parameter_count * group.sensitivity
        numerator += (
            group.parameter_count
            * group.sensitivity
            * BIT_NORMALIZATION.get(group_bits[group.name], 0.0)
        )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean_sensitivity_for_bit(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    bit_width: int,
) -> float:
    matching = [group.sensitivity for group in groups if group_bits[group.name] == bit_width]
    if not matching:
        return 0.0
    return sum(matching) / len(matching)


def _sensitivity_bucket_fraction(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    bucket_names: set[str],
    bit_width: int,
) -> float:
    bucket_groups = [group for group in groups if group.name in bucket_names]
    if not bucket_groups:
        return 0.0
    matching = [group for group in bucket_groups if group_bits[group.name] == bit_width]
    return len(matching) / len(bucket_groups)


def _component_bit_fraction(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    component_name: str,
    bit_width: int,
) -> float:
    matching_groups = [group for group in groups if group.component_type.endswith(component_name)]
    if not matching_groups:
        return 0.0
    selected = [group for group in matching_groups if group_bits[group.name] == bit_width]
    return len(selected) / len(matching_groups)


def _count_unsupported_bits(group_bits: dict[str, int]) -> int:
    return sum(1 for bit in group_bits.values() if bit not in (4, 8))


def _collect_task_baselines(
    manifest_payload: dict[str, Any],
    uniform_baseline_bit_width: int | None = None,
) -> dict[str, float | None]:
    uniform_accuracies: list[float] = []
    native_accuracies: list[float] = []
    for record in manifest_payload.get("records", []):
        evaluation_path = record.get("evaluation_path")
        if not evaluation_path:
            continue
        try:
            evaluation_payload = _load_json(evaluation_path)
        except FileNotFoundError:
            continue
        accuracy = float(evaluation_payload["accuracy"])
        if record.get("provenance") == "uniform_baseline" or record.get("uniform_bit_width") is not None:
            record_uniform_bit_width = record.get("uniform_bit_width")
            if uniform_baseline_bit_width is None or (
                record_uniform_bit_width is not None
                and int(record_uniform_bit_width) == uniform_baseline_bit_width
            ):
                uniform_accuracies.append(accuracy)
        native_path = _infer_native_evaluation_path(evaluation_path)
        if native_path and native_path.exists():
            native_payload = _load_json(native_path)
            native_accuracies.append(float(native_payload["accuracy"]))
    return {
        "uniform_accuracy": statistics.fmean(uniform_accuracies) if uniform_accuracies else None,
        "native_accuracy": statistics.fmean(native_accuracies) if native_accuracies else None,
        "uniform_baseline_bit_width": float(uniform_baseline_bit_width)
        if uniform_baseline_bit_width is not None
        else None,
    }


def _resolve_native_accuracy_for_record(
    record_payload: dict[str, Any],
    baseline_metrics: dict[str, float | None] | None,
) -> float | None:
    native_path = _infer_native_evaluation_path(record_payload["evaluation_path"])
    if native_path and native_path.exists():
        native_payload = _load_json(native_path)
        return float(native_payload["accuracy"])
    if baseline_metrics:
        native_accuracy = baseline_metrics.get("native_accuracy")
        if native_accuracy is not None:
            return float(native_accuracy)
    return None


def _infer_native_evaluation_path(evaluation_path: str | Path) -> Path | None:
    path = Path(evaluation_path)
    if not path.name.endswith("-quantized.json"):
        return None
    return path.with_name(path.name.replace("-quantized.json", "-native.json"))


def _best_available_baseline(
    native_accuracy: float | None,
    uniform_accuracy: float | None,
) -> float | None:
    available = [
        value
        for value in (native_accuracy, uniform_accuracy)
        if value is not None
    ]
    if not available:
        return None
    return max(available)


def _component_name_from_group_name(group_name: str) -> str:
    if group_name.startswith("block:"):
        return group_name.split(":", 2)[-1]
    if group_name.startswith("component:"):
        return group_name.split(":", 1)[-1]
    return group_name


def _summarize_value_score(
    bit_targets: dict[int, list[float]],
    min_support: int,
) -> dict[str, float]:
    supported_bits = (4, 8, 16)
    mean_targets = {
        bit_width: (
            statistics.fmean(float(value) for value in bit_targets.get(bit_width, []))
            if bit_targets.get(bit_width)
            else 0.0
        )
        for bit_width in supported_bits
    }
    support_counts = {
        bit_width: len(bit_targets.get(bit_width, []))
        for bit_width in supported_bits
    }

    support_8_over_4 = min(support_counts[4], support_counts[8])
    confidence_8_over_4 = min(1.0, support_8_over_4 / max(1, min_support))
    raw_uplift_8_over_4 = mean_targets[8] - mean_targets[4] if support_8_over_4 else 0.0
    uplift_8_over_4 = raw_uplift_8_over_4 * confidence_8_over_4

    support_16_over_8 = min(support_counts[8], support_counts[16])
    confidence_16_over_8 = min(1.0, support_16_over_8 / max(1, min_support))
    raw_uplift_16_over_8 = mean_targets[16] - mean_targets[8] if support_16_over_8 else 0.0
    uplift_16_over_8 = raw_uplift_16_over_8 * confidence_16_over_8

    confidence = max(confidence_8_over_4, confidence_16_over_8)
    support = max(support_8_over_4, support_16_over_8)
    score = _overall_value_score_from_uplifts(uplift_8_over_4, uplift_16_over_8)
    preferred_bit = _preferred_bit_from_means(
        mean_targets=mean_targets,
        support_counts=support_counts,
        min_support=min_support,
    )
    return {
        "score": score,
        "raw_uplift": raw_uplift_8_over_4,
        "confidence": confidence,
        "support": float(support),
        "mean_4bit_target": mean_targets[4],
        "mean_8bit_target": mean_targets[8],
        "mean_16bit_target": mean_targets[16],
        "support_4bit": float(support_counts[4]),
        "support_8bit": float(support_counts[8]),
        "support_16bit": float(support_counts[16]),
        "uplift_8_over_4": uplift_8_over_4,
        "uplift_16_over_8": uplift_16_over_8,
        "uplift_16_over_4": uplift_8_over_4 + uplift_16_over_8,
        "confidence_8_over_4": confidence_8_over_4,
        "confidence_16_over_8": confidence_16_over_8,
        "preferred_bit": float(preferred_bit),
    }


def _overall_value_score_from_uplifts(
    uplift_8_over_4: float,
    uplift_16_over_8: float,
) -> float:
    return uplift_8_over_4 + 0.65 * uplift_16_over_8


def _preferred_bit_from_means(
    *,
    mean_targets: dict[int, float],
    support_counts: dict[int, int],
    min_support: int,
) -> int:
    ranked_bits = [
        bit_width
        for bit_width, _ in sorted(
            mean_targets.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if support_counts.get(bit_width, 0) >= min_support
    ]
    if ranked_bits:
        return ranked_bits[0]
    fallback_supported_bits = [
        bit_width
        for bit_width, count in support_counts.items()
        if count > 0
    ]
    if fallback_supported_bits:
        return max(
            fallback_supported_bits,
            key=lambda bit_width: (mean_targets[bit_width], bit_width),
        )
    return 4


def _preferred_bit_from_uplifts(
    uplift_8_over_4: float,
    uplift_16_over_8: float,
    *,
    fallback: int = 4,
) -> int:
    if uplift_16_over_8 > 0:
        return 16
    if uplift_8_over_4 > 0:
        return 8
    return fallback if fallback in (4, 8, 16) else 4


def _rule_hit_rate(report_payload: dict[str, Any]) -> float:
    rule_hits = report_payload.get("rule_hits", [])
    if not rule_hits:
        policy_rules = report_payload.get("policy", {}).get("rules", [])
        return 1.0 if not policy_rules else 0.0
    if isinstance(rule_hits, dict):
        policy_rules = report_payload.get("policy", {}).get("rules", [])
        if not policy_rules:
            return 1.0
        matched_rule_names = {
            str(rule_name)
            for rule_name, hit_count in rule_hits.items()
            if int(hit_count) > 0
        }
        expected_rule_names = {
            str(rule.get("name"))
            for rule in policy_rules
            if rule.get("name")
        }
        if not expected_rule_names:
            return 1.0
        return len(matched_rule_names & expected_rule_names) / len(expected_rule_names)
    matched = sum(1 for item in rule_hits if item.get("matched"))
    return matched / len(rule_hits)


def _profile_group_lookup(
    sensitivity_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not sensitivity_payload:
        return {}
    groups_payload = sensitivity_payload.get("groups", [])
    return {
        str(group_payload["name"]): dict(group_payload)
        for group_payload in groups_payload
        if group_payload.get("name") is not None
    }


def _profile_signal_alignment_score(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    profile_group_lookup: dict[str, dict[str, Any]],
    *,
    field_name: str,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        group_payload = profile_group_lookup.get(group.name)
        if not group_payload:
            continue
        signal_value = float(group_payload.get(field_name, 0.0))
        if signal_value <= 0.0:
            continue
        numerator += (
            group.parameter_count
            * signal_value
            * BIT_NORMALIZATION.get(group_bits[group.name], 0.0)
        )
        denominator += group.parameter_count * signal_value
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _profile_signal_mean_for_bit(
    groups: list[SearchGroup],
    group_bits: dict[str, int],
    profile_group_lookup: dict[str, dict[str, Any]],
    *,
    bit_width: int,
    field_name: str,
) -> float:
    matching_values = []
    for group in groups:
        if group_bits[group.name] != bit_width:
            continue
        group_payload = profile_group_lookup.get(group.name)
        if not group_payload:
            continue
        matching_values.append(float(group_payload.get(field_name, 0.0)))
    if not matching_values:
        return 0.0
    return statistics.fmean(matching_values)


def _sorted_feature_importances(
    feature_names: list[str],
    raw_importances: Any,
) -> list[dict[str, float]]:
    pairs = [
        {
            "feature_name": feature_name,
            "importance": float(importance),
        }
        for feature_name, importance in zip(feature_names, raw_importances, strict=True)
    ]
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)


def _sorted_feature_importances_from_booster(
    feature_names: list[str],
    booster: Any,
) -> list[dict[str, float]]:
    gain_scores = booster.get_score(importance_type="gain")
    pairs = [
        {
            "feature_name": feature_name,
            "importance": float(gain_scores.get(feature_name, 0.0)),
        }
        for feature_name in feature_names
    ]
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)


def _sorted_feature_importances_from_boosters(
    feature_names: list[str],
    boosters: list[Any],
) -> list[dict[str, float]]:
    if not boosters:
        return []
    aggregate = {feature_name: 0.0 for feature_name in feature_names}
    for booster in boosters:
        gain_scores = booster.get_score(importance_type="gain")
        for feature_name in feature_names:
            aggregate[feature_name] += float(gain_scores.get(feature_name, 0.0))
    pairs = [
        {
            "feature_name": feature_name,
            "importance": aggregate[feature_name] / len(boosters),
        }
        for feature_name in feature_names
    ]
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)


def _fit_xgboost_booster(
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    random_seed: int,
) -> Any:
    import xgboost as xgb

    training_matrix = xgb.DMatrix(matrix, label=targets, feature_names=feature_names)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.08,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "seed": random_seed,
    }
    return xgb.train(
        params=params,
        dtrain=training_matrix,
        num_boost_round=64,
    )


def _fit_xgboost_ensemble(
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    random_seed: int,
    ensemble_size: int,
) -> list[Any]:
    rng = random.Random(random_seed)
    boosters: list[Any] = []
    for ensemble_index in range(ensemble_size):
        bootstrap_indices = [rng.randrange(len(matrix)) for _ in range(len(matrix))]
        bootstrap_matrix = [matrix[index] for index in bootstrap_indices]
        bootstrap_targets = [targets[index] for index in bootstrap_indices]
        boosters.append(
            _fit_xgboost_booster(
                matrix=bootstrap_matrix,
                targets=bootstrap_targets,
                feature_names=feature_names,
                random_seed=random_seed + ensemble_index,
            )
        )
    return boosters


def _leave_one_out_predictions(
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    random_seed: int,
) -> list[float]:
    if len(matrix) < 3:
        return []

    import xgboost as xgb

    predictions: list[float] = []
    for held_out_index in range(len(matrix)):
        train_matrix = [row for index, row in enumerate(matrix) if index != held_out_index]
        train_targets = [
            target for index, target in enumerate(targets) if index != held_out_index
        ]
        booster = _fit_xgboost_booster(
            matrix=train_matrix,
            targets=train_targets,
            feature_names=feature_names,
            random_seed=random_seed,
        )
        held_out = xgb.DMatrix(
            [matrix[held_out_index]],
            feature_names=feature_names,
        )
        predictions.append(float(booster.predict(held_out)[0]))
    return predictions


def _leave_one_out_ensemble_predictions(
    matrix: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    random_seed: int,
    ensemble_size: int,
) -> tuple[list[float], list[float]]:
    if len(matrix) < 3:
        return [], []

    predictions: list[float] = []
    uncertainties: list[float] = []
    for held_out_index in range(len(matrix)):
        train_matrix = [row for index, row in enumerate(matrix) if index != held_out_index]
        train_targets = [
            target for index, target in enumerate(targets) if index != held_out_index
        ]
        boosters = _fit_xgboost_ensemble(
            matrix=train_matrix,
            targets=train_targets,
            feature_names=feature_names,
            random_seed=random_seed + held_out_index * 17,
            ensemble_size=ensemble_size,
        )
        fold_predictions, fold_uncertainties = _ensemble_predictions(
            boosters=boosters,
            matrix=[matrix[held_out_index]],
            feature_names=feature_names,
        )
        predictions.append(fold_predictions[0])
        uncertainties.append(fold_uncertainties[0])
    return predictions, uncertainties


def _ensemble_predictions(
    boosters: list[Any],
    matrix: list[list[float]],
    feature_names: list[str],
) -> tuple[list[float], list[float]]:
    import xgboost as xgb

    design = xgb.DMatrix(matrix, feature_names=feature_names)
    per_model_predictions = [
        [float(value) for value in booster.predict(design)]
        for booster in boosters
    ]
    means: list[float] = []
    stds: list[float] = []
    for row_predictions in zip(*per_model_predictions, strict=True):
        row_predictions = tuple(row_predictions)
        means.append(statistics.fmean(row_predictions))
        stds.append(statistics.pstdev(row_predictions) if len(row_predictions) > 1 else 0.0)
    return means, stds


def _serialize_booster(booster: Any) -> str:
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        booster.save_model(str(temp_path))
        return temp_path.read_text(encoding="utf-8")
    finally:
        temp_path.unlink(missing_ok=True)


def _load_booster_from_json(model_json: str) -> Any:
    import xgboost as xgb

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        temp_path.write_text(model_json, encoding="utf-8")
        booster = xgb.Booster()
        booster.load_model(str(temp_path))
        return booster
    finally:
        temp_path.unlink(missing_ok=True)


def _mean_prediction_from_summary(surrogate_summary_payload: dict[str, Any]) -> float:
    target_mean = surrogate_summary_payload.get("target_mean")
    if target_mean is not None:
        return float(target_mean)
    predictions = surrogate_summary_payload.get("predictions", [])
    if not predictions:
        return 0.0
    return sum(float(item["prediction"]) for item in predictions) / len(predictions)


def _apply_surrogate_calibration(
    prediction_mean: float,
    prediction_std: float,
    surrogate_summary_payload: dict[str, Any],
) -> tuple[float, float]:
    anchor = _mean_prediction_from_summary(surrogate_summary_payload)
    calibration_weight = float(surrogate_summary_payload.get("calibration_weight", 1.0))
    calibration_weight = max(0.0, min(1.0, calibration_weight))
    uncertainty_floor = max(
        0.0,
        float(surrogate_summary_payload.get("uncertainty_floor", 0.0)),
    )
    calibrated_mean = anchor + calibration_weight * (prediction_mean - anchor)
    calibrated_std = (prediction_std**2 + uncertainty_floor**2) ** 0.5
    return calibrated_mean, calibrated_std


def _top_k_hit_rate(
    targets: list[float],
    predictions: list[float],
    k: int,
) -> float:
    if not targets or not predictions:
        return 0.0
    top_k = max(1, min(k, len(targets)))
    actual_best_index = max(range(len(targets)), key=lambda index: (targets[index], -index))
    predicted_ranking = sorted(
        range(len(predictions)),
        key=lambda index: (predictions[index], targets[index]),
        reverse=True,
    )
    return 1.0 if actual_best_index in predicted_ranking[:top_k] else 0.0


def _pairwise_ranking_accuracy(targets: list[float], predictions: list[float]) -> float:
    if len(targets) != len(predictions) or len(targets) < 2:
        return 0.0

    pair_count = 0
    correct = 0.0
    for left_index in range(len(targets) - 1):
        for right_index in range(left_index + 1, len(targets)):
            target_delta = targets[left_index] - targets[right_index]
            if target_delta == 0:
                continue
            prediction_delta = predictions[left_index] - predictions[right_index]
            pair_count += 1
            if prediction_delta == 0:
                correct += 0.5
            elif target_delta * prediction_delta > 0:
                correct += 1.0

    if pair_count == 0:
        return 0.0
    return correct / pair_count


def _spearman_rank_correlation(targets: list[float], predictions: list[float]) -> float:
    if len(targets) != len(predictions) or len(targets) < 2:
        return 0.0
    target_ranks = _average_ranks(targets)
    prediction_ranks = _average_ranks(predictions)
    return _pearson_correlation(target_ranks, prediction_ranks)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        tie_end = index
        while tie_end + 1 < len(indexed) and indexed[tie_end + 1][1] == indexed[index][1]:
            tie_end += 1
        average_rank = (index + tie_end) / 2 + 1
        for tie_index in range(index, tie_end + 1):
            ranks[indexed[tie_index][0]] = average_rank
        index = tie_end + 1
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if left_variance <= 0 or right_variance <= 0:
        return 0.0
    return numerator / ((left_variance ** 0.5) * (right_variance ** 0.5))


def _mean_absolute_error(targets: list[float], predictions: list[float]) -> float:
    return sum(abs(target - prediction) for target, prediction in zip(targets, predictions, strict=True)) / len(targets)


def _root_mean_squared_error(targets: list[float], predictions: list[float]) -> float:
    mse = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions, strict=True)) / len(targets)
    return mse ** 0.5


def _target_distribution_stats(targets: list[float]) -> tuple[float, float, float, float]:
    if not targets:
        return 0.0, 0.0, 0.0, 1.0
    target_mean = statistics.fmean(targets)
    target_std = statistics.pstdev(targets) if len(targets) > 1 else 0.0
    target_range = (max(targets) - min(targets)) if len(targets) > 1 else 0.0
    target_scale = max(target_std, target_range / 4.0, 1e-6)
    return target_mean, target_std, target_range, target_scale


def _calibration_weight(
    *,
    num_examples: int,
    validation_rmse: float | None,
    validation_spearman: float | None,
    validation_top1_hit_rate: float | None,
    validation_top3_hit_rate: float | None,
    validation_pairwise_accuracy: float | None,
    target_scale: float,
) -> float:
    if num_examples < 3:
        return 0.0

    data_score = max(0.0, min(1.0, (num_examples - 3) / 12.0))
    spearman_score = (
        max(0.0, min(1.0, (float(validation_spearman) + 1.0) / 2.0))
        if validation_spearman is not None
        else 0.5
    )
    top1_score = float(validation_top1_hit_rate or 0.0)
    top3_score = float(validation_top3_hit_rate or 0.0)
    pairwise_score = (
        max(0.0, min(1.0, float(validation_pairwise_accuracy)))
        if validation_pairwise_accuracy is not None
        else 0.5
    )
    if validation_rmse is None:
        rmse_score = 0.5
    else:
        rmse_score = max(0.0, min(1.0, 1.0 - (float(validation_rmse) / max(target_scale, 1e-6))))

    quality_score = (
        0.30 * spearman_score
        + 0.25 * pairwise_score
        + 0.20 * rmse_score
        + 0.10 * top1_score
        + 0.15 * top3_score
    )
    return max(0.25, min(1.0, 0.25 + 0.75 * data_score * quality_score))


def _uncertainty_floor(
    *,
    validation_rmse: float | None,
    target_std: float,
    calibration_weight: float,
) -> float:
    base_floor = max(0.0, float(validation_rmse)) if validation_rmse is not None else 0.0
    if target_std > 0:
        base_floor = max(base_floor, 0.25 * target_std)
    return base_floor * max(0.0, 1.0 - calibration_weight)


def _sanitize_component_name(component_name: str) -> str:
    return component_name.replace(".", "_")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
