#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.quant_search.config import load_config
from ta_mpq.quant_search.group_registry import (
    build_group_registry_from_model,
    GroupInfo,
    load_group_registry,
    save_group_registry,
)
from ta_mpq.quant_search.sensitivity import profile_paper_task_sensitivity, profile_taq_kl_lite
from ta_mpq.modal_feasibility_app import (
    A100_40GB_MODAL_GPU,
    app as modal_app,
    collect_paper_task_sensitivity_profile_remote,
    collect_paper_task_sensitivity_profile_remote_a100,
    collect_taq_kl_lite_profile_remote,
    collect_taq_kl_lite_profile_remote_a100,
)


SUPPORTED_METHODS = {
    "taq_kl_lite",
    "paper_task_activation_profile",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument("--grouping", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-modal", action="store_true")
    parser.add_argument("--gpu-type", type=str, default=A100_40GB_MODAL_GPU)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    model_id = args.model or str(config.get("model_id") or "")
    task_name = args.task or str(config.get("task") or "")
    grouping = args.grouping or str(config.get("grouping") or "per_block_component")
    sensitivity_config = dict(config.get("sensitivity", {}))
    method = str(args.method or sensitivity_config.get("method") or "taq_kl_lite")
    split = args.split or str(sensitivity_config.get("calibration_split") or "first300")
    num_prompts = args.num_prompts or int(sensitivity_config.get("num_calibration_prompts") or 64)
    seed = args.seed if args.seed >= 0 else int(config.get("seed") or 42)
    output_path = Path(
        args.output
        or sensitivity_config.get("output")
        or PROJECT_ROOT / "artifacts" / "sensitivity" / f"{task_name}_{_method_slug(method)}.json"
    )
    registry_path = Path(
        sensitivity_config.get("group_registry_path")
        or PROJECT_ROOT / "artifacts" / "group_registry" / f"{grouping}.jsonl"
    )
    task_prompt_style = str(config.get("evaluation", {}).get("eval_mode") or "simple_evals_nonthinking")
    noise_bits = tuple(int(bit) for bit in sensitivity_config.get("noise_bits", [2, 4, 8]))
    max_prompt_tokens = int(sensitivity_config.get("max_prompt_tokens") or 1024)
    activation_weight = float(sensitivity_config.get("activation_weight") or 0.55)
    batch_size = int(sensitivity_config.get("batch_size") or 8)

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported sensitivity method: {method}")
    if not model_id or not task_name:
        raise ValueError("Both model_id and task must be set via --config or CLI flags")

    if args.use_modal:
        payload = _profile_via_modal(
            method=method,
            model_id=model_id,
            task_name=task_name,
            grouping=grouping,
            split=split,
            num_prompts=num_prompts,
            temperature=float(sensitivity_config.get("temperature") or 1.0),
            seed=seed,
            noise_bits=noise_bits,
            max_prompt_tokens=max_prompt_tokens,
            activation_weight=activation_weight,
            batch_size=batch_size,
            task_prompt_style=task_prompt_style,
            gpu_type=str(args.gpu_type or A100_40GB_MODAL_GPU),
        )
        groups = [GroupInfo.from_dict(item) for item in payload["group_registry"]]
        save_group_registry(registry_path, groups)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload["sensitivity_profile"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        if registry_path.exists():
            groups = load_group_registry(registry_path)
        else:
            groups = build_group_registry_from_model(model_id=model_id, grouping=grouping)
            save_group_registry(registry_path, groups)

        if method == "taq_kl_lite":
            profile_taq_kl_lite(
                model_id=model_id,
                task_name=task_name,
                groups=groups,
                split=split,
                num_prompts=num_prompts,
                output_path=output_path,
                temperature=float(sensitivity_config.get("temperature") or 1.0),
                seed=seed,
                noise_bits=noise_bits,
                max_prompt_tokens=max_prompt_tokens,
                task_prompt_style=task_prompt_style,
                resume=bool(args.resume or sensitivity_config.get("resume", True)),
                batch_size=batch_size,
            )
        else:
            profile_paper_task_sensitivity(
                model_id=model_id,
                task_name=task_name,
                groups=groups,
                split=split,
                num_prompts=num_prompts,
                output_path=output_path,
                activation_weight=activation_weight,
                max_prompt_tokens=max_prompt_tokens,
                task_prompt_style=task_prompt_style,
            )
    print(output_path)


def _profile_via_modal(
    method: str,
    model_id: str,
    task_name: str,
    grouping: str,
    split: str,
    num_prompts: int,
    temperature: float,
    seed: int,
    noise_bits: tuple[int, ...],
    max_prompt_tokens: int,
    activation_weight: float,
    batch_size: int,
    task_prompt_style: str,
    gpu_type: str,
) -> dict[str, object]:
    if method == "taq_kl_lite":
        remote = collect_taq_kl_lite_profile_remote
        if gpu_type == A100_40GB_MODAL_GPU:
            remote = collect_taq_kl_lite_profile_remote_a100
    else:
        remote = collect_paper_task_sensitivity_profile_remote
        if gpu_type == A100_40GB_MODAL_GPU:
            remote = collect_paper_task_sensitivity_profile_remote_a100

    with modal.enable_output():
        with modal_app.run():
            if method == "taq_kl_lite":
                return remote.remote(
                    model_id=model_id,
                    task_name=task_name,
                    split=split,
                    num_prompts=num_prompts,
                    grouping=grouping,
                    temperature=temperature,
                    seed=seed,
                    noise_bits=noise_bits,
                    max_prompt_tokens=max_prompt_tokens,
                    task_prompt_style=task_prompt_style,
                    batch_size=batch_size,
                )
            return remote.remote(
                model_id=model_id,
                task_name=task_name,
                split=split,
                num_prompts=num_prompts,
                grouping=grouping,
                activation_weight=activation_weight,
                max_prompt_tokens=max_prompt_tokens,
                task_prompt_style=task_prompt_style,
            )


def _method_slug(method: str) -> str:
    return method.replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
