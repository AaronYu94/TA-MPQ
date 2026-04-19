from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_COARSE_PROMOTION_MASS_GRID = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20,
    0.24,
    0.28,
    0.32,
]


FRONTIER_RESULTS_FIELDNAMES = [
    "policy_id",
    "policy_hash",
    "stage",
    "builder",
    "was_evaluated",
    "duplicate_of_policy_hash",
    "path_index",
    "endpoint_kind",
    "num_eval_examples",
    "score",
    "accuracy",
    "correct",
    "total",
    "proxy_score",
    "twobit_param_count",
    "twobit_mass_fraction",
    "num_2bit",
    "num_4bit",
    "num_8bit",
    "raw_weight_bits",
    "target_int4_bits",
    "budget_slack_bits",
    "budget_slack_fraction",
    "promotion_mass_fraction",
    "realized_8bit_param_mass_fraction",
    "realized_2bit_param_mass_fraction",
    "realized_4bit_param_mass_fraction",
    "policy_path",
    "eval_artifact_path",
    "failure_reason",
]


def save_frontier_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRONTIER_RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FRONTIER_RESULTS_FIELDNAMES})


def load_frontier_results_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "policy_id": row["policy_id"],
                    "policy_hash": str(row.get("policy_hash") or ""),
                    "stage": row["stage"],
                    "builder": row.get("builder", ""),
                    "was_evaluated": _parse_bool(row.get("was_evaluated"), default=True),
                    "duplicate_of_policy_hash": str(row.get("duplicate_of_policy_hash") or ""),
                    "path_index": _parse_int(row.get("path_index"), default=0),
                    "endpoint_kind": str(row.get("endpoint_kind") or ""),
                    "num_eval_examples": _parse_int(row.get("num_eval_examples"), default=0),
                    "score": _parse_float(row.get("score"), default=-1.0),
                    "accuracy": _parse_float(row.get("accuracy"), default=_parse_float(row.get("score"), default=-1.0)),
                    "correct": _parse_int(row.get("correct"), default=0),
                    "total": _parse_int(row.get("total"), default=0),
                    "proxy_score": _parse_float(row.get("proxy_score"), default=0.0),
                    "twobit_param_count": _parse_int(row.get("twobit_param_count"), default=0),
                    "twobit_mass_fraction": _parse_float(row.get("twobit_mass_fraction"), default=0.0),
                    "num_2bit": _parse_int(row.get("num_2bit"), default=0),
                    "num_4bit": _parse_int(row.get("num_4bit"), default=0),
                    "num_8bit": _parse_int(row.get("num_8bit"), default=0),
                    "raw_weight_bits": _parse_int(row.get("raw_weight_bits"), default=0),
                    "target_int4_bits": _parse_int(row.get("target_int4_bits"), default=0),
                    "budget_slack_bits": _parse_int(row.get("budget_slack_bits"), default=0),
                    "budget_slack_fraction": _parse_float(row.get("budget_slack_fraction"), default=0.0),
                    "promotion_mass_fraction": _parse_float(
                        row.get("promotion_mass_fraction"),
                        default=_parse_float(row.get("realized_8bit_param_mass_fraction"), default=0.0),
                    ),
                    "realized_8bit_param_mass_fraction": _parse_float(
                        row.get("realized_8bit_param_mass_fraction"),
                        default=_parse_float(row.get("promotion_mass_fraction"), default=0.0),
                    ),
                    "realized_2bit_param_mass_fraction": _parse_float(
                        row.get("realized_2bit_param_mass_fraction"),
                        default=_parse_float(row.get("twobit_mass_fraction"), default=0.0),
                    ),
                    "realized_4bit_param_mass_fraction": _parse_float(
                        row.get("realized_4bit_param_mass_fraction"),
                        default=0.0,
                    ),
                    "policy_path": row.get("policy_path", ""),
                    "eval_artifact_path": row.get("eval_artifact_path", ""),
                    "failure_reason": str(row.get("failure_reason") or ""),
                }
            )
    return rows


def write_duplicate_policy_report(
    path: str | Path,
    duplicates: dict[str, list[str]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(duplicates, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_refinement_grid(
    coarse_results: list[dict[str, Any]],
    tie_band_correct_answers: int = 1,
    tie_break: list[str] | tuple[str, ...] = (
        "higher_proxy_score",
        "lower_promotion_mass_fraction",
        "lower_twobit_mass",
        "smaller_budget_slack",
    ),
    num_threshold_subpoints: int = 4,
    include_winner_fraction: bool = True,
    top_k_coarse_candidates: int = 1,
    min_fraction: float = 0.0,
    max_fraction: float = 0.32,
) -> list[float]:
    threshold_rows = [
        row
        for row in coarse_results
        if row.get("builder") == "coarse_grid"
    ]
    if not threshold_rows:
        return []
    ordered = sorted(
        threshold_rows,
        key=lambda row: (
            float(row["promotion_mass_fraction"]),
            str(row["policy_id"]),
        ),
    )
    winners = select_top_rows(
        ordered,
        top_k=max(1, int(top_k_coarse_candidates)),
        tie_band_correct_answers=tie_band_correct_answers,
        tie_break=tie_break,
    )
    fractions = [float(row["promotion_mass_fraction"]) for row in ordered]
    refined: set[float] = set()
    for winner in winners:
        winner_fraction = float(winner["promotion_mass_fraction"])
        winner_index = fractions.index(winner_fraction)
        if len(fractions) == 1:
            refined.add(round(_clamp(winner_fraction, min_fraction, max_fraction), 4))
            continue

        if winner_index == 0:
            low = winner_fraction
            high = fractions[1]
        elif winner_index == len(fractions) - 1:
            low = fractions[-2]
            high = winner_fraction
        else:
            low = fractions[winner_index - 1]
            high = fractions[winner_index + 1]

        if include_winner_fraction:
            refined.add(round(_clamp(winner_fraction, min_fraction, max_fraction), 4))
        for index in range(num_threshold_subpoints):
            midpoint = low + (((index + 0.5) / num_threshold_subpoints) * (high - low))
            refined.add(round(_clamp(midpoint, min_fraction, max_fraction), 4))
    return sorted(refined)


def select_best_policy_id(
    rows: list[dict[str, Any]],
    preferred_stage: str | None = None,
    tie_band_correct_answers: int = 0,
    tie_break: list[str] | tuple[str, ...] = (),
) -> str:
    filtered = [
        row
        for row in rows
        if preferred_stage is None or str(row.get("stage")) == preferred_stage
    ]
    if not filtered:
        raise ValueError("No results available to select a best policy")
    best = select_best_row(
        filtered,
        tie_band_correct_answers=tie_band_correct_answers,
        tie_break=tie_break,
    )
    return str(best["policy_id"])


def select_best_policy_hash(
    rows: list[dict[str, Any]],
    preferred_stage: str | None = None,
    tie_band_correct_answers: int = 0,
    tie_break: list[str] | tuple[str, ...] = (),
) -> str:
    filtered = [
        row
        for row in rows
        if preferred_stage is None or str(row.get("stage")) == preferred_stage
    ]
    if not filtered:
        raise ValueError("No results available to select a best policy hash")
    best = select_best_row(
        filtered,
        tie_band_correct_answers=tie_band_correct_answers,
        tie_break=tie_break,
    )
    return str(best["policy_hash"])


def select_finalist_policy_ids(
    refined_rows: list[dict[str, Any]],
    top_k_threshold: int = 3,
    include_baseline_builders: tuple[str, ...] = (
        "uniform_int4_baseline",
        "inverse_sensitivity_baseline",
        "random_exact_budget_baseline",
    ),
) -> list[str]:
    threshold_rows = [
        row
        for row in refined_rows
        if row.get("builder") == "refined_grid"
    ]
    ranked_threshold = sorted(
        threshold_rows,
        key=lambda row: (
            float(row["score"]),
            -float(row["budget_slack_fraction"]),
            -float(row["promotion_mass_fraction"]),
            str(row["policy_id"]),
        ),
        reverse=True,
    )
    finalists = [str(row["policy_id"]) for row in ranked_threshold[:top_k_threshold]]
    for builder in include_baseline_builders:
        builder_rows = [row for row in refined_rows if row.get("builder") == builder]
        if not builder_rows:
            continue
        best_builder = max(
            builder_rows,
            key=lambda row: (
                float(row["score"]),
                str(row["policy_id"]),
            ),
        )
        finalists.append(str(best_builder["policy_id"]))
    deduped: list[str] = []
    for policy_id in finalists:
        if policy_id not in deduped:
            deduped.append(policy_id)
    return deduped


def resolve_policy_parallelism(
    *,
    stage: str,
    num_policies: int,
    cfg: dict[str, Any],
) -> int:
    policy_cfg = dict(dict(cfg.get("execution", {})).get("policy_parallelism", {}))
    stage_overrides = dict(policy_cfg.get("stage_overrides", {}))
    stage_override_payload = stage_overrides.get(stage)
    stage_override: Any = None
    if isinstance(stage_override_payload, dict):
        stage_override = stage_override_payload.get("max_parallel_policies")
    else:
        stage_override = stage_override_payload
    global_cap = policy_cfg.get("max_parallel_policies")
    cap = stage_override if stage_override is not None else global_cap
    if cap in {None, "", "auto"}:
        return max(1, int(num_policies))
    return max(1, min(int(num_policies), int(cap)))


def select_best_row(
    rows: list[dict[str, Any]],
    tie_band_correct_answers: int = 0,
    tie_break: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    return select_top_rows(
        rows,
        top_k=1,
        tie_band_correct_answers=tie_band_correct_answers,
        tie_break=tie_break,
    )[0]


def select_top_rows(
    rows: list[dict[str, Any]],
    top_k: int,
    tie_band_correct_answers: int = 0,
    tie_break: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No rows available for selection")
    if top_k <= 0:
        return []
    best_correct = max(int(row.get("correct", 0)) for row in rows)
    finalists = [
        row
        for row in rows
        if int(row.get("correct", 0)) >= (best_correct - int(tie_band_correct_answers))
    ]
    ordered = sorted(
        finalists,
        key=lambda row: tuple(_tie_break_value(row, rule) for rule in tie_break) + (str(row["policy_id"]),),
    )
    return ordered[:top_k]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value in {None, ""}:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return bool(default)


def _parse_int(value: Any, *, default: int) -> int:
    if value in {None, ""}:
        return int(default)
    return int(value)


def _parse_float(value: Any, *, default: float) -> float:
    if value in {None, ""}:
        return float(default)
    return float(value)


def _tie_break_value(row: dict[str, Any], rule: str) -> Any:
    normalized = str(rule)
    if normalized == "higher_proxy_score":
        return -float(row.get("proxy_score", 0.0))
    if normalized == "higher_accuracy":
        return -float(row.get("accuracy", row.get("score", 0.0)))
    if normalized == "lower_promotion_mass_fraction":
        return float(
            row.get(
                "realized_8bit_param_mass_fraction",
                row.get("promotion_mass_fraction", 0.0),
            )
        )
    if normalized == "lower_twobit_mass":
        return float(
            row.get(
                "realized_2bit_param_mass_fraction",
                row.get("twobit_mass_fraction", 0.0),
            )
        )
    if normalized == "smaller_budget_slack":
        return abs(float(row.get("budget_slack_fraction", 0.0)))
    raise ValueError(f"Unsupported tie-break rule: {rule}")
