#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.baseline import save_summary
from ta_mpq.contracts import ExperimentContract
from ta_mpq.metrics import ExampleResult, summarize_results
from ta_mpq.quant_search.budget import compute_policy_budget_stats
from ta_mpq.quant_search.config import load_config
from ta_mpq.quant_search.frontier_search import (
    load_frontier_results_csv,
    resolve_policy_parallelism,
    save_frontier_results_csv,
    select_best_policy_hash,
    select_best_row,
    write_duplicate_policy_report,
)
from ta_mpq.quant_search.greedy_path import METHOD_NAME as GREEDY_PATH_METHOD_NAME
from ta_mpq.quant_search.group_registry import load_group_registry
from ta_mpq.quant_search.policy_hash import duplicate_policy_hashes
from ta_mpq.quant_search.policy_io import list_policy_files, load_policy_payload
from ta_mpq.modal_feasibility_app import (
    A100_40GB_MODAL_GPU,
    DEFAULT_MODAL_GPU,
    app as modal_app,
    evaluate_task_source_model,
    load_task_example_ids_remote,
    probe_mixed_precision_feasibility_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--policy-dir", type=str, default="")
    parser.add_argument("--results-csv", type=str, default="")
    parser.add_argument("--cache-evals", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    task_name = str(config.get("task") or "")
    model_id = str(config.get("model_id") or "")
    grouping = str(config.get("grouping") or "per_block_component")
    if not task_name or not model_id:
        raise ValueError("Config must define model_id and task")

    frontier_config = dict(config.get("frontier_search", {}))
    if not frontier_config:
        raise ValueError("Config must define frontier_search")
    frontier_mode = str(frontier_config.get("mode") or "sector_coarse_to_fine_budgeted")
    stage_config = _resolve_stage_config(args.stage, frontier_config)
    sensitivity_config = dict(config.get("sensitivity", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    task_prompt_style = str(evaluation_config.get("eval_mode") or "simple_evals_nonthinking")
    max_new_tokens = int(evaluation_config.get("max_new_tokens") or 4096)
    cache_evals = bool(args.cache_evals or evaluation_config.get("cache_evals", True))
    hard_limits = dict(frontier_config.get("hard_limits", {}))

    registry_path = Path(
        sensitivity_config.get("group_registry_path")
        or PROJECT_ROOT / "artifacts" / "group_registry" / f"{grouping}.jsonl"
    )
    groups = load_group_registry(registry_path)
    policy_dir = Path(
        args.policy_dir
        or dict(config.get("policy_builder", {})).get("output_dir")
        or PROJECT_ROOT / "artifacts" / "policies" / "task_sensitivity_exact_budget"
    )
    results_root = PROJECT_ROOT / "artifacts" / "results" / "task_sensitivity_exact_budget"
    results_root.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv or (results_root / f"frontier_{args.stage}.csv"))

    stage_policy_records = _select_stage_policies(
        stage=args.stage,
        policy_dir=policy_dir,
        results_root=results_root,
        frontier_config=frontier_config,
        frontier_mode=frontier_mode,
    )
    unique_policy_records, duplicate_rows = _dedupe_stage_policy_records(
        stage_policy_records,
        stage=args.stage,
        results_root=results_root,
    )
    if (
        hard_limits.get("max_unique_quantized_policies") is not None
        and len(unique_policy_records) > int(hard_limits["max_unique_quantized_policies"])
    ):
        raise ValueError(
            f"Stage {args.stage} selected {len(unique_policy_records)} unique policies, exceeding "
            f"max_unique_quantized_policies={hard_limits['max_unique_quantized_policies']}"
        )

    contract = _build_contract(config)
    example_ids = _resolve_stage_example_ids(
        task_name=task_name,
        frontier_config=frontier_config,
        stage=args.stage,
    )
    rows = list(duplicate_rows)
    resolved_parallelism = _resolve_stage_parallelism(
        stage=args.stage,
        num_policies=len(unique_policy_records),
        stage_config=stage_config,
        config=config,
    )
    print(
        f"[frontier:{args.stage}] evaluating {len(unique_policy_records)} unique policies "
        f"with parallelism={resolved_parallelism}",
        flush=True,
    )

    if resolved_parallelism <= 1:
        for policy_record in unique_policy_records:
            rows.append(
                _run_policy_eval_row(
                    stage=args.stage,
                    policy_record=policy_record,
                    contract=contract,
                    groups=groups,
                    split=str(stage_config["split"]),
                    limit=int(stage_config["num_examples"]),
                    max_new_tokens=max_new_tokens,
                    task_prompt_style=task_prompt_style,
                    cache_evals=cache_evals,
                    results_root=results_root,
                    calibration_limit=int(sensitivity_config.get("num_calibration_prompts") or 64),
                    example_ids=example_ids,
                    gpu_type=str(stage_config.get("gpu_type") or DEFAULT_MODAL_GPU),
                    cache_policy_artifacts_by_hash=bool(hard_limits.get("cache_policy_artifacts_by_hash", True)),
                    cache_per_example_outputs=bool(hard_limits.get("cache_per_example_outputs", True)),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=resolved_parallelism) as executor:
            futures = [
                executor.submit(
                    _run_policy_eval_row,
                    stage=args.stage,
                    policy_record=policy_record,
                    contract=contract,
                    groups=groups,
                    split=str(stage_config["split"]),
                    limit=int(stage_config["num_examples"]),
                    max_new_tokens=max_new_tokens,
                    task_prompt_style=task_prompt_style,
                    cache_evals=cache_evals,
                    results_root=results_root,
                    calibration_limit=int(sensitivity_config.get("num_calibration_prompts") or 64),
                    example_ids=example_ids,
                    gpu_type=str(stage_config.get("gpu_type") or DEFAULT_MODAL_GPU),
                    cache_policy_artifacts_by_hash=bool(hard_limits.get("cache_policy_artifacts_by_hash", True)),
                    cache_per_example_outputs=bool(hard_limits.get("cache_per_example_outputs", True)),
                )
                for policy_record in unique_policy_records
            ]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: (str(row["policy_id"]), str(row.get("duplicate_of_policy_hash") or "")))

    save_frontier_results_csv(results_csv, rows)
    max_generated_question_evals = hard_limits.get("max_generated_question_evals_default")
    if max_generated_question_evals is not None:
        total_question_evals = sum(int(row["num_eval_examples"]) for row in rows if bool(row.get("was_evaluated", True)))
        if args.stage not in {"final", "endpoint_diagnostic"} and total_question_evals > int(max_generated_question_evals):
            raise ValueError(
                f"Stage {args.stage} used {total_question_evals} generated question-evals, exceeding "
                f"max_generated_question_evals_default={max_generated_question_evals}"
            )
    if args.stage == "final":
        _write_final_summary(
            results_root=results_root,
            final_rows=[row for row in rows if bool(row.get("was_evaluated", True))],
            policy_dir=policy_dir,
        )
    print(results_csv)


def _resolve_stage_config(stage: str, frontier_config: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "final_last100": "final",
    }
    lookup_key = aliases.get(stage, stage)
    resolved = dict(frontier_config.get(lookup_key, {}))
    if not resolved:
        raise ValueError(f"Config did not define frontier_search.{lookup_key}")
    return {
        "split": resolved["eval_split"],
        "num_examples": int(resolved["num_questions"]),
        "question_set_name": str(resolved.get("question_set_name") or ""),
        "gpu_type": str(resolved.get("gpu_type") or DEFAULT_MODAL_GPU),
        "max_parallel_policies": resolved.get("max_parallel_policies"),
    }


def _select_stage_policies(
    *,
    stage: str,
    policy_dir: Path,
    results_root: Path,
    frontier_config: dict[str, Any],
    frontier_mode: str,
) -> list[dict[str, Any]]:
    if frontier_mode == GREEDY_PATH_METHOD_NAME:
        return _select_greedy_path_stage_policies(
            stage=stage,
            policy_dir=policy_dir,
            results_root=results_root,
            frontier_config=frontier_config,
        )

    policy_paths = _select_legacy_policy_paths(
        stage=stage,
        policy_dir=policy_dir,
        results_root=results_root,
        frontier_config=frontier_config,
    )
    return [
        {
            "policy_path": path,
            "payload": load_policy_payload(path),
        }
        for path in policy_paths
    ]


def _select_greedy_path_stage_policies(
    *,
    stage: str,
    policy_dir: Path,
    results_root: Path,
    frontier_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if stage == "coarse":
        return _load_policy_records(
            list_policy_files(policy_dir, builder_names={"coarse_path"}),
        )
    if stage == "refine":
        return _load_policy_records(
            list_policy_files(policy_dir, builder_names={"refined_path"}),
        )
    if stage in {"final", "final_last100"}:
        refined_rows = _evaluated_rows(load_frontier_results_csv(results_root / "frontier_refine.csv"))
        refine_config = dict(frontier_config.get("refine", {}))
        best_policy_hash = select_best_policy_hash(
            refined_rows,
            tie_band_correct_answers=int(refine_config.get("tie_band_correct_answers") or 0),
            tie_break=list(refine_config.get("tie_break", [])),
        )
        best_row = next(row for row in refined_rows if str(row.get("policy_hash") or "") == best_policy_hash)
        return _load_policy_records([Path(best_row["policy_path"])])
    if stage == "endpoint_diagnostic":
        return _load_policy_records(
            list_policy_files(policy_dir, builder_names={"endpoint_diagnostic"}),
        )
    raise ValueError(f"Unsupported stage: {stage}")


def _select_legacy_policy_paths(
    *,
    stage: str,
    policy_dir: Path,
    results_root: Path,
    frontier_config: dict[str, Any],
) -> list[Path]:
    if stage == "coarse":
        return list_policy_files(
            policy_dir,
            builder_names={
                "coarse_grid",
            },
        )
    if stage == "refine":
        return list_policy_files(
            policy_dir,
            builder_names={
                "refined_grid",
            },
        )
    if stage in {"final", "final_last100"}:
        refined_rows = load_frontier_results_csv(results_root / "frontier_refine.csv")
        refine_config = dict(frontier_config.get("refine", {}))
        best_policy_id = select_best_row(
            refined_rows,
            tie_band_correct_answers=int(refine_config.get("tie_band_correct_answers") or 0),
            tie_break=list(refine_config.get("tie_break", [])),
        )["policy_id"]
        return list_policy_files(policy_dir, policy_ids=[str(best_policy_id)])
    raise ValueError(f"Unsupported stage: {stage}")


def _load_policy_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "policy_path": path,
            "payload": load_policy_payload(path),
        }
        for path in paths
    ]


def _dedupe_stage_policy_records(
    policy_records: list[dict[str, Any]],
    *,
    stage: str,
    results_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicates = duplicate_policy_hashes([record["payload"] for record in policy_records])
    if duplicates:
        write_duplicate_policy_report(
            results_root / "duplicate_reports" / f"{stage}.json",
            duplicates,
        )

    unique_records: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for record in policy_records:
        payload = record["payload"]
        policy_hash = str(payload["policy_hash"])
        if policy_hash not in seen:
            seen[policy_hash] = record
            unique_records.append(record)
            continue
        duplicate_rows.append(
            _make_duplicate_row(
                stage=stage,
                payload=payload,
                policy_path=record["policy_path"],
                canonical_hash=policy_hash,
            )
        )
    return unique_records, duplicate_rows


def _build_contract(config: dict[str, Any]) -> ExperimentContract:
    model_id = str(config["model_id"])
    task_name = str(config["task"])
    evaluation = dict(config.get("evaluation", {}))
    sensitivity = dict(config.get("sensitivity", {}))
    return ExperimentContract(
        name="task-sensitivity-exact-budget",
        task_name=task_name,
        compressed_source_model_id=model_id,
        native_baseline_model_id=str(config.get("native_baseline_model_id") or model_id),
        upper_bound_model_id=str(config.get("upper_bound_model_id") or model_id),
        comparison_rule="accuracy_under_int4_budget",
        budget_rule="raw_weight_int4",
        quantization_bits=(2, 4, 8),
        calibration_samples=int(sensitivity.get("num_calibration_prompts") or 64),
        baseline_eval_limit=int(evaluation.get("baseline_eval_limit") or 100),
        generation_max_new_tokens=int(evaluation.get("max_new_tokens") or 4096),
    )


def _evaluate_policy_payload(
    *,
    contract: ExperimentContract,
    policy_payload: dict[str, Any],
    policy_path: Path,
    groups: list[Any],
    split: str,
    limit: int,
    max_new_tokens: int,
    task_prompt_style: str,
    cache_evals: bool,
    results_root: Path,
    calibration_limit: int,
    example_ids: list[str] | None,
    gpu_type: str,
    cache_policy_artifacts_by_hash: bool,
    cache_per_example_outputs: bool,
) -> dict[str, Any]:
    policy_hash = str(policy_payload["policy_hash"])
    requested_example_ids = list(example_ids or [])
    if not requested_example_ids:
        requested_example_ids = list(
            load_task_example_ids_remote.remote(task_name=contract.task_name, split=split)
        )[:limit]
    requested_example_ids = [str(example_id) for example_id in requested_example_ids[:limit]]
    question_set_hash = _question_set_hash(requested_example_ids)
    decode_config_hash = _decode_config_hash(max_new_tokens=max_new_tokens)
    prompt_version = task_prompt_style or "default"
    eval_mode = task_prompt_style or "default"
    policy_eval_dir = results_root / "evaluations_by_hash"
    exact_summary_path = (
        policy_eval_dir
        / split
        / f"{policy_hash}-{decode_config_hash}-{_slug(prompt_version)}-{_slug(eval_mode)}-{question_set_hash}.json"
    )
    if cache_evals and exact_summary_path.exists():
        return json.loads(exact_summary_path.read_text(encoding="utf-8"))

    precomputed_report = {
        "model_id": contract.compressed_source_model_id,
        "layer_stats": [
            {
                "name": group.module_path,
                "parameter_count": group.param_count,
                "in_features": 0,
                "out_features": 0,
            }
            for group in groups
        ],
    }
    feasibility_remote = _resolve_feasibility_remote(gpu_type)
    eval_remote = _resolve_eval_remote(gpu_type)

    artifact_cache_path = results_root / "quantized_artifacts" / f"{policy_hash}.json"
    report: dict[str, Any] | None = None
    if cache_policy_artifacts_by_hash and artifact_cache_path.exists():
        report = json.loads(artifact_cache_path.read_text(encoding="utf-8"))
    if report is None:
        report = feasibility_remote.remote(
            contract.to_dict(),
            calibration_limit=calibration_limit,
            dry_run=False,
            policy_payload=policy_payload["backend_projections"]["llmcompressor"]["projected_policy"],
            policy_label=policy_hash,
            precomputed_report=precomputed_report,
        )
        artifact_cache_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_cache_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_dir = str(report.get("output_dir") or "")
    if not artifact_dir or not report.get("quantized_model_runnable", False):
        failed_summary = {
            "policy_id": policy_payload["policy_id"],
            "policy_hash": policy_hash,
            "accuracy": -1.0,
            "num_correct": 0,
            "num_examples": 0,
            "status": report.get("status", "quantization_failed"),
            "policy_path": str(policy_path),
            "failure_reason": str(report.get("status", "quantization_failed")),
            "results": [],
            "evaluated_example_ids": [],
        }
        exact_summary_path.parent.mkdir(parents=True, exist_ok=True)
        exact_summary_path.write_text(json.dumps(failed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return failed_summary

    per_example_cache_path = (
        results_root
        / "per_example_cache"
        / split
        / f"{policy_hash}-{decode_config_hash}-{_slug(prompt_version)}-{_slug(eval_mode)}.json"
    )
    cached_example_map = (
        _load_per_example_cache(per_example_cache_path)
        if cache_evals and cache_per_example_outputs
        else {}
    )
    cached_hits = {
        example_id: cached_example_map[example_id]
        for example_id in requested_example_ids
        if example_id in cached_example_map
    }
    missing_ids = [
        example_id
        for example_id in requested_example_ids
        if example_id not in cached_hits
    ]

    remote_summary: dict[str, Any] | None = None
    if missing_ids:
        remote_summary = eval_remote.remote(
            model_ref=artifact_dir,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=f"{contract.compressed_source_model_id}-{policy_hash}",
            task_name=contract.task_name,
            limit=len(missing_ids),
            max_new_tokens=max_new_tokens,
            split=split,
            example_ids=missing_ids,
            load_dtype="auto",
            task_prompt_style=task_prompt_style,
        )
        if cache_evals and cache_per_example_outputs:
            _update_per_example_cache(per_example_cache_path, remote_summary)
            cached_example_map = _load_per_example_cache(per_example_cache_path)

    complete_result_map = {
        example_id: cached_example_map[example_id]
        for example_id in requested_example_ids
        if example_id in cached_example_map
    }
    if len(complete_result_map) != len(requested_example_ids):
        missing_after_eval = [
            example_id
            for example_id in requested_example_ids
            if example_id not in complete_result_map
        ]
        raise RuntimeError(
            f"Per-example cache did not cover requested ids after remote eval: {missing_after_eval[:5]}"
        )

    merged_summary = _summarize_cached_results(
        model_id=str((remote_summary or {}).get("model_id") or f"{contract.compressed_source_model_id}-{policy_hash}"),
        task_name=contract.task_name,
        split=split,
        ordered_example_ids=requested_example_ids,
        result_map=complete_result_map,
        task_prompt_style=task_prompt_style,
        max_new_tokens=max_new_tokens,
    )
    exact_summary_path.parent.mkdir(parents=True, exist_ok=True)
    exact_summary_path.write_text(json.dumps(merged_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged_summary


def _run_policy_eval_row(
    *,
    stage: str,
    policy_record: dict[str, Any],
    contract: ExperimentContract,
    groups: list[Any],
    split: str,
    limit: int,
    max_new_tokens: int,
    task_prompt_style: str,
    cache_evals: bool,
    results_root: Path,
    calibration_limit: int,
    example_ids: list[str] | None,
    gpu_type: str,
    cache_policy_artifacts_by_hash: bool,
    cache_per_example_outputs: bool,
) -> dict[str, Any]:
    policy_path = Path(policy_record["policy_path"])
    payload = dict(policy_record["payload"])
    summary = _evaluate_policy_payload(
        contract=contract,
        policy_payload=payload,
        policy_path=policy_path,
        groups=groups,
        split=split,
        limit=limit,
        max_new_tokens=max_new_tokens,
        task_prompt_style=task_prompt_style,
        cache_evals=cache_evals,
        results_root=results_root,
        calibration_limit=calibration_limit,
        example_ids=example_ids,
        gpu_type=gpu_type,
        cache_policy_artifacts_by_hash=cache_policy_artifacts_by_hash,
        cache_per_example_outputs=cache_per_example_outputs,
    )
    stats = compute_policy_budget_stats(
        dict(payload.get("group_bit_assignments", payload["bitwidths"])),
        groups,
        path_index=int(payload.get("stats", {}).get("path_index")) if payload.get("stats", {}).get("path_index") is not None else None,
    )
    failure_reason = str(summary.get("failure_reason") or summary.get("status") or "")
    decode_config_hash = _decode_config_hash(max_new_tokens=max_new_tokens)
    prompt_version = task_prompt_style or "default"
    eval_mode = task_prompt_style or "default"
    requested_example_ids = list(example_ids or [])[:limit]
    question_set_hash = _question_set_hash([str(example_id) for example_id in requested_example_ids])
    eval_artifact_path = (
        results_root
        / "evaluations_by_hash"
        / split
        / f"{payload['policy_hash']}-{decode_config_hash}-{_slug(prompt_version)}-{_slug(eval_mode)}-{question_set_hash}.json"
    )
    return {
        "policy_id": payload["policy_id"],
        "policy_hash": payload["policy_hash"],
        "stage": stage,
        "builder": payload.get("source", {}).get("builder", ""),
        "was_evaluated": True,
        "duplicate_of_policy_hash": "",
        "path_index": int(payload.get("stats", {}).get("path_index") or 0),
        "endpoint_kind": str(payload.get("source", {}).get("endpoint_kind") or ""),
        "num_eval_examples": int(summary.get("num_examples", 0)),
        "score": float(summary.get("accuracy", -1.0)),
        "accuracy": float(summary.get("accuracy", -1.0)),
        "correct": int(summary.get("num_correct", 0)),
        "total": int(summary.get("num_examples", 0)),
        "proxy_score": float(payload.get("proxy_score", payload.get("proxy_quality_score", 0.0))),
        "twobit_param_count": int(stats.twobit_param_count),
        "twobit_mass_fraction": float(stats.twobit_mass_fraction),
        "num_2bit": stats.num_2bit,
        "num_4bit": stats.num_4bit,
        "num_8bit": stats.num_8bit,
        "raw_weight_bits": stats.raw_weight_bits,
        "target_int4_bits": stats.target_int4_bits,
        "budget_slack_bits": stats.budget_slack_bits,
        "budget_slack_fraction": stats.budget_slack_fraction,
        "promotion_mass_fraction": stats.promotion_mass_fraction,
        "realized_8bit_param_mass_fraction": stats.realized_8bit_param_mass_fraction,
        "realized_2bit_param_mass_fraction": stats.realized_2bit_param_mass_fraction,
        "realized_4bit_param_mass_fraction": stats.realized_4bit_param_mass_fraction,
        "policy_path": str(policy_path),
        "eval_artifact_path": str(eval_artifact_path),
        "failure_reason": failure_reason,
    }


def _write_final_summary(
    *,
    results_root: Path,
    final_rows: list[dict[str, Any]],
    policy_dir: Path,
) -> None:
    if not final_rows:
        return
    best_policy_id = final_rows[0]["policy_id"]
    policy_payload = load_policy_payload(policy_dir / f"{best_policy_id}.json")
    top_8bit = [
        group_id
        for group_id, bitwidth in policy_payload["bitwidths"].items()
        if int(bitwidth) == 8
    ][:20]
    top_2bit = [
        group_id
        for group_id, bitwidth in policy_payload["bitwidths"].items()
        if int(bitwidth) == 2
    ][:20]
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "task_sensitivity_exact_budget" / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"best_policy_id: {best_policy_id}",
        f"best_policy_hash: {final_rows[0]['policy_hash']}",
        f"score_last100: {final_rows[0]['score']:.4f}",
        (
            "num_2bit / num_4bit / num_8bit: "
            f"{final_rows[0]['num_2bit']} / {final_rows[0]['num_4bit']} / {final_rows[0]['num_8bit']}"
        ),
        f"realized_8bit_param_mass_fraction: {final_rows[0]['realized_8bit_param_mass_fraction']:.6f}",
        f"budget_slack_fraction: {final_rows[0]['budget_slack_fraction']:.6f}",
        "top 20 8-bit groups:",
        *top_8bit,
        "top 20 2-bit groups:",
        *top_2bit,
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_stage_example_ids(
    *,
    task_name: str,
    frontier_config: dict[str, Any],
    stage: str,
) -> list[str] | None:
    resolved_stage = "final" if stage == "final_last100" else stage
    stage_config = dict(frontier_config.get(resolved_stage, {}))
    split = str(stage_config["eval_split"])
    num_questions = int(stage_config["num_questions"])
    examples = [str(example_id) for example_id in load_task_example_ids_remote.remote(task_name=task_name, split=split)]
    question_set_name = str(stage_config.get("question_set_name") or "")
    if "disjoint_from_coarse" not in question_set_name:
        return examples[:num_questions]

    consumed = 0
    for predecessor in ("coarse",):
        predecessor_config = dict(frontier_config.get(predecessor, {}))
        if str(predecessor_config.get("eval_split") or "") == split:
            consumed += int(predecessor_config.get("num_questions") or 0)
    return examples[consumed : consumed + num_questions]


def _resolve_eval_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        from ta_mpq.modal_feasibility_app import evaluate_task_source_model_a100

        return evaluate_task_source_model_a100
    return evaluate_task_source_model


def _resolve_feasibility_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        from ta_mpq.modal_feasibility_app import probe_mixed_precision_feasibility_source_a100

        return probe_mixed_precision_feasibility_source_a100
    return probe_mixed_precision_feasibility_source


def _load_per_example_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(example_id): dict(result)
        for example_id, result in dict(payload.get("results_by_example_id", {})).items()
    }


def _update_per_example_cache(path: Path, summary: dict[str, Any]) -> None:
    payload = {
        "results_by_example_id": _load_per_example_cache(path),
    }
    for item in summary.get("results", []):
        payload["results_by_example_id"][str(item["example_id"])] = dict(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_cached_results(
    *,
    model_id: str,
    task_name: str,
    split: str,
    ordered_example_ids: list[str],
    result_map: dict[str, dict[str, Any]],
    task_prompt_style: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    results = [
        ExampleResult(**result_map[example_id])
        for example_id in ordered_example_ids
    ]
    summary = summarize_results(model_id=model_id, task_name=task_name, results=results)
    summary["task_split"] = split
    summary["evaluated_example_ids"] = [result.example_id for result in results]
    summary["task_prompt_style"] = task_prompt_style or None
    normalized_task_name = str(task_name).strip().lower()
    if normalized_task_name in {"math500", "math-500"}:
        summary["thinking_mode"] = "disabled" if task_prompt_style == "simple_evals_nonthinking" else "enabled"
    else:
        summary["thinking_mode"] = "disabled"
    summary["generation_mode"] = "greedy"
    summary["max_new_tokens"] = max_new_tokens
    return summary


def _decode_config_hash(*, max_new_tokens: int) -> str:
    payload = json.dumps({"max_new_tokens": int(max_new_tokens)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _question_set_hash(example_ids: list[str]) -> str:
    payload = json.dumps(list(example_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-") or "default"


def _resolve_stage_parallelism(
    *,
    stage: str,
    num_policies: int,
    stage_config: dict[str, Any],
    config: dict[str, Any],
) -> int:
    has_policy_parallelism = bool(dict(config.get("execution", {})).get("policy_parallelism"))
    if has_policy_parallelism:
        return resolve_policy_parallelism(stage=stage, num_policies=num_policies, cfg=config)
    configured_cap = stage_config.get("max_parallel_policies")
    if configured_cap in {None, "", "auto"}:
        return max(1, int(num_policies))
    return max(1, min(int(num_policies), int(configured_cap)))


def _make_duplicate_row(
    *,
    stage: str,
    payload: dict[str, Any],
    policy_path: Path,
    canonical_hash: str,
) -> dict[str, Any]:
    raw_stats = dict(payload.get("stats", {}))
    return {
        "policy_id": payload["policy_id"],
        "policy_hash": payload["policy_hash"],
        "stage": stage,
        "builder": payload.get("source", {}).get("builder", ""),
        "was_evaluated": False,
        "duplicate_of_policy_hash": canonical_hash,
        "path_index": int(raw_stats.get("path_index") or 0),
        "endpoint_kind": str(payload.get("source", {}).get("endpoint_kind") or ""),
        "num_eval_examples": 0,
        "score": -1.0,
        "accuracy": -1.0,
        "correct": 0,
        "total": 0,
        "proxy_score": float(payload.get("proxy_score", payload.get("proxy_quality_score", 0.0))),
        "twobit_param_count": int(raw_stats.get("twobit_param_count") or 0),
        "twobit_mass_fraction": float(raw_stats.get("twobit_mass_fraction") or 0.0),
        "num_2bit": int(raw_stats.get("num_2bit") or 0),
        "num_4bit": int(raw_stats.get("num_4bit") or 0),
        "num_8bit": int(raw_stats.get("num_8bit") or 0),
        "raw_weight_bits": int(raw_stats.get("raw_weight_bits") or 0),
        "target_int4_bits": int(raw_stats.get("target_int4_bits") or 0),
        "budget_slack_bits": int(raw_stats.get("budget_slack_bits") or 0),
        "budget_slack_fraction": float(raw_stats.get("budget_slack_fraction") or 0.0),
        "promotion_mass_fraction": float(raw_stats.get("promotion_mass_fraction") or 0.0),
        "realized_8bit_param_mass_fraction": float(raw_stats.get("realized_8bit_param_mass_fraction") or 0.0),
        "realized_2bit_param_mass_fraction": float(raw_stats.get("realized_2bit_param_mass_fraction") or 0.0),
        "realized_4bit_param_mass_fraction": float(raw_stats.get("realized_4bit_param_mass_fraction") or 0.0),
        "policy_path": str(policy_path),
        "eval_artifact_path": "",
        "failure_reason": "duplicate_policy_hash",
    }


def _evaluated_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("was_evaluated", True))]


if __name__ == "__main__":
    with modal.enable_output():
        with modal_app.run():
            main()
