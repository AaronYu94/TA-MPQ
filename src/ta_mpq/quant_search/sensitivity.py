from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ta_mpq.quant_search.group_registry import GroupInfo, to_layer_stats
from ta_mpq.sensitivity import build_task_sensitivity_profile, collect_task_activation_stats
from ta_mpq.tasks import load_task_adapter
from ta_mpq.transformers_compat import apply_qwen3_5_fast_path_compat_patch


@dataclass(frozen=True, slots=True)
class SensitivityRecord:
    param_count: int
    score: float
    risk_2: float | None = None
    risk_4: float | None = None
    risk_8: float | None = None
    benefit_8_over_4: float | None = None
    demotion_cost_4_to_2: float | None = None
    extras: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extras = dict(payload.pop("extras") or {})
        normalized = {
            key: value
            for key, value in payload.items()
            if value is not None
        }
        normalized.update(extras)
        return normalized

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensitivityRecord":
        known_keys = {
            "param_count",
            "score",
            "risk_2",
            "risk_4",
            "risk_8",
            "benefit_8_over_4",
            "demotion_cost_4_to_2",
        }
        return cls(
            param_count=int(payload["param_count"]),
            score=float(payload["score"]),
            risk_2=_optional_float(payload.get("risk_2")),
            risk_4=_optional_float(payload.get("risk_4")),
            risk_8=_optional_float(payload.get("risk_8")),
            benefit_8_over_4=_optional_float(payload.get("benefit_8_over_4")),
            demotion_cost_4_to_2=_optional_float(payload.get("demotion_cost_4_to_2")),
            extras={
                key: value
                for key, value in dict(payload).items()
                if key not in known_keys
            }
            or None,
        )


def load_sensitivity_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_sensitivity_payload(payload)


def save_sensitivity_profile(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_sensitivity_payload(payload)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")


def profile_paper_task_sensitivity(
    model_id: str,
    task_name: str,
    groups: list[GroupInfo],
    split: str,
    num_prompts: int,
    output_path: str | Path | None = None,
    activation_weight: float = 0.55,
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
) -> dict[str, Any]:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")

    activation_stats = collect_task_activation_stats(
        model_id=model_id,
        task_name=task_name,
        limit=num_prompts,
        max_prompt_tokens=max_prompt_tokens,
        split=split,
        task_prompt_style=task_prompt_style,
    )
    paper_profile = build_task_sensitivity_profile(
        layer_stats=to_layer_stats(groups),
        activation_stats=activation_stats,
        grouping="per_block_component",
        activation_weight=activation_weight,
    )
    payload = convert_task_sensitivity_profile(
        groups=groups,
        task_sensitivity_profile=paper_profile,
        model_id=model_id,
        task_name=task_name,
        split=split,
        num_prompts=num_prompts,
        activation_weight=activation_weight,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
    )
    if output_path:
        save_sensitivity_profile(output_path, payload)
    return payload


def convert_task_sensitivity_profile(
    groups: list[GroupInfo],
    task_sensitivity_profile: dict[str, Any],
    *,
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    activation_weight: float,
    max_prompt_tokens: int,
    task_prompt_style: str,
    method_name: str = "paper_task_activation_profile",
) -> dict[str, Any]:
    raw_groups = list(task_sensitivity_profile.get("groups", []))
    grouped_payloads = {
        str(group_payload["name"]): dict(group_payload)
        for group_payload in raw_groups
    }
    missing = [
        group.group_id
        for group in groups
        if group.group_id not in grouped_payloads
    ]
    if missing:
        raise ValueError(
            "Task sensitivity profile missing quant-search groups: "
            + ", ".join(missing[:5])
        )

    converted_groups: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_payload = grouped_payloads[group.group_id]
        converted_groups[group.group_id] = SensitivityRecord(
            param_count=group.param_count,
            score=float(group_payload.get("combined_sensitivity", group_payload.get("prior_sensitivity", 0.0))),
            extras={
                "component_type": group_payload.get("component_type"),
                "layer_names": list(group_payload.get("layer_names", [])),
                "prior_sensitivity": _optional_float(group_payload.get("prior_sensitivity")),
                "activation_score": _optional_float(group_payload.get("activation_score")),
                "normalized_activation_score": _optional_float(group_payload.get("normalized_activation_score")),
                "combined_sensitivity": _optional_float(group_payload.get("combined_sensitivity")),
            },
        ).to_dict()

    payload = {
        "metadata": {
            "method": method_name,
            "model_id": model_id,
            "task": task_name,
            "calibration_split": split,
            "num_calibration_prompts": num_prompts,
            "activation_weight": activation_weight,
            "max_prompt_tokens": max_prompt_tokens,
            "task_prompt_style": task_prompt_style,
            "paper_profile_kind": "activation_plus_prior_blend",
        },
        "groups": converted_groups,
        "module_activation_stats": list(task_sensitivity_profile.get("module_activation_stats", [])),
        "paper_profile": {
            key: value
            for key, value in dict(task_sensitivity_profile).items()
            if key != "module_activation_stats"
        },
    }
    return _normalize_sensitivity_payload(payload)


def profile_taq_kl_lite(
    model_id: str,
    task_name: str,
    groups: list[GroupInfo],
    split: str,
    num_prompts: int,
    output_path: str | Path | None = None,
    temperature: float = 1.0,
    seed: int = 42,
    noise_bits: tuple[int, ...] = (2, 4, 8),
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
    resume: bool = True,
    batch_size: int = 8,
    log_every_groups: int = 8,
) -> dict[str, Any]:
    import torch

    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    existing_groups: dict[str, dict[str, Any]] = {}
    if output_path and resume and Path(output_path).exists():
        existing_groups = dict(load_sensitivity_profile(output_path).get("groups", {}))

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task = load_task_adapter(task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    try:
        device = _infer_model_device(model)
        module_lookup = dict(model.named_modules())
        missing_modules = [
            group.module_path
            for group in groups
            if group.module_path not in module_lookup
        ]
        if missing_modules:
            raise ValueError(
                "Group registry referenced unknown modules: " + ", ".join(missing_modules[:5])
            )

        encoded_prompts = _prepare_encoded_prompts(
            tokenizer=tokenizer,
            task=task,
            split=split,
            num_prompts=num_prompts,
            max_prompt_tokens=max_prompt_tokens,
            task_prompt_style=task_prompt_style,
        )
        print(
            f"[taq_kl_lite] starting profile: groups={len(groups)} prompts={len(encoded_prompts)} "
            f"noise_bits={list(noise_bits)} batch_size={batch_size}",
            flush=True,
        )
        baseline_logits, output_ranges = _collect_baseline_logits_and_ranges(
            model=model,
            device=device,
            encoded_prompts=encoded_prompts,
            groups=groups,
            module_lookup=module_lookup,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        results = dict(existing_groups)
        started_at = time.time()
        completed_groups = len(results)
        for group in groups:
            if group.group_id in results:
                continue

            per_bit_risks: dict[int, float] = {}
            range_value = float(output_ranges.get(group.group_id, 0.0))
            for bit_width in noise_bits:
                if range_value <= 0.0:
                    per_bit_risks[int(bit_width)] = 0.0
                    continue
                delta = range_value / max((2**int(bit_width)) - 1, 1)
                per_bit_risks[int(bit_width)] = _estimate_group_perturbation_kl(
                    model=model,
                    device=device,
                    encoded_prompts=encoded_prompts,
                    baseline_logits=baseline_logits,
                    module=module_lookup[group.module_path],
                    delta=delta,
                    temperature=temperature,
                    seed=seed,
                    group_id=group.group_id,
                    bit_width=int(bit_width),
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                )

            score = (
                per_bit_risks.get(4)
                if 4 in per_bit_risks
                else next(iter(per_bit_risks.values()), 0.0)
            )
            risk_2 = per_bit_risks.get(2)
            risk_4 = per_bit_risks.get(4)
            risk_8 = per_bit_risks.get(8)
            benefit_8_over_4 = (
                max(0.0, float(risk_4) - float(risk_8))
                if risk_4 is not None and risk_8 is not None
                else score
            )
            demotion_cost_4_to_2 = (
                max(0.0, float(risk_2) - float(risk_4))
                if risk_2 is not None and risk_4 is not None
                else score
            )
            results[group.group_id] = SensitivityRecord(
                param_count=group.param_count,
                score=float(score),
                risk_2=_optional_float(risk_2),
                risk_4=_optional_float(risk_4),
                risk_8=_optional_float(risk_8),
                benefit_8_over_4=float(benefit_8_over_4),
                demotion_cost_4_to_2=float(demotion_cost_4_to_2),
            ).to_dict()
            completed_groups += 1
            if (
                completed_groups == 1
                or completed_groups == len(groups)
                or completed_groups % max(1, int(log_every_groups)) == 0
            ):
                elapsed = max(time.time() - started_at, 1e-6)
                groups_done = max(completed_groups - len(existing_groups), 1)
                avg_seconds_per_group = elapsed / groups_done
                remaining_groups = max(len(groups) - completed_groups, 0)
                eta_minutes = (remaining_groups * avg_seconds_per_group) / 60.0
                print(
                    f"[taq_kl_lite] progress {completed_groups}/{len(groups)} "
                    f"group={group.group_id} risk4={float(risk_4 or 0.0):.6f} "
                    f"eta_min={eta_minutes:.1f}",
                    flush=True,
                )

            if output_path:
                save_sensitivity_profile(
                    output_path,
                    {
                        "metadata": {
                            "method": "taq_kl_lite",
                            "model_id": model_id,
                            "task": task_name,
                            "calibration_split": split,
                            "num_calibration_prompts": num_prompts,
                            "temperature": temperature,
                            "seed": seed,
                            "task_prompt_style": task_prompt_style,
                            "noise_bits": list(noise_bits),
                            "batch_size": batch_size,
                        },
                        "groups": results,
                    },
                )

        final_payload = {
            "metadata": {
                "method": "taq_kl_lite",
                "model_id": model_id,
                "task": task_name,
                "calibration_split": split,
                "num_calibration_prompts": num_prompts,
                "temperature": temperature,
                "seed": seed,
                "task_prompt_style": task_prompt_style,
                "noise_bits": list(noise_bits),
                "batch_size": batch_size,
            },
            "groups": results,
        }
        if output_path:
            save_sensitivity_profile(output_path, final_payload)
        return final_payload
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prepare_encoded_prompts(
    tokenizer: Any,
    task: Any,
    split: str,
    num_prompts: int,
    max_prompt_tokens: int,
    task_prompt_style: str,
) -> list[dict[str, Any]]:
    examples = task.load_examples(limit=num_prompts, split=split)
    encoded_prompts: list[dict[str, Any]] = []
    for example in examples:
        messages = _build_task_messages(
            task=task,
            question=example.question,
            task_prompt_style=task_prompt_style,
        )
        prompt = _render_prompt(tokenizer, messages)
        encoded_prompts.append(
            {
                key: value.squeeze(0)
                for key, value in tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_tokens,
                ).items()
            }
        )
    return encoded_prompts


def _collect_baseline_logits_and_ranges(
    model: Any,
    device: Any,
    encoded_prompts: list[dict[str, Any]],
    groups: list[GroupInfo],
    module_lookup: dict[str, Any],
    tokenizer: Any,
    batch_size: int,
) -> tuple[list[Any], dict[str, float]]:
    import torch

    range_accumulators = {
        group.group_id: {
            "sum": 0.0,
            "count": 0,
        }
        for group in groups
    }
    hooks = []
    for group in groups:
        module = module_lookup[group.module_path]
        hooks.append(
            module.register_forward_hook(
                _build_range_hook(
                    group_id=group.group_id,
                    accumulators=range_accumulators,
                )
            )
        )

    baseline_logits: list[Any] = []
    try:
        for batch in _iter_encoded_batches(encoded_prompts, batch_size=batch_size):
            inputs = _move_inputs_to_device(_pad_encoded_batch(tokenizer, batch), device)
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[:, -1, :].detach().float().cpu()
            for row in logits:
                baseline_logits.append(row.unsqueeze(0))
    finally:
        for hook in hooks:
            hook.remove()

    output_ranges = {
        group_id: (
            accumulator["sum"] / accumulator["count"]
            if accumulator["count"] > 0
            else 0.0
        )
        for group_id, accumulator in range_accumulators.items()
    }
    return baseline_logits, output_ranges


def _estimate_group_perturbation_kl(
    model: Any,
    device: Any,
    encoded_prompts: list[dict[str, Any]],
    baseline_logits: list[Any],
    module: Any,
    delta: float,
    temperature: float,
    seed: int,
    group_id: str,
    bit_width: int,
    tokenizer: Any,
    batch_size: int,
) -> float:
    import torch

    total_kl = 0.0
    total_examples = 0
    for batch_start, batch in _iter_encoded_batches_with_start(encoded_prompts, batch_size=batch_size):
        noise_seed = _stable_seed(seed, group_id, bit_width, batch_start)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
            generator.manual_seed(noise_seed)
            return _add_noise_to_output(output, delta=delta, generator=generator)

        handle = module.register_forward_hook(hook)
        try:
            inputs = _move_inputs_to_device(_pad_encoded_batch(tokenizer, batch), device)
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            perturbed_logits = outputs.logits[:, -1, :].detach().float().cpu()
        finally:
            handle.remove()
        batch_baseline = torch.cat(
            baseline_logits[batch_start : batch_start + len(batch)],
            dim=0,
        )
        total_kl += float(
            kl_p_to_q(
                batch_baseline,
                perturbed_logits,
                temperature=temperature,
            )
        ) * len(batch)
        total_examples += len(batch)
    return total_kl / max(total_examples, 1)


def kl_p_to_q(
    baseline_logits: Any,
    perturbed_logits: Any,
    temperature: float = 1.0,
) -> Any:
    import torch

    p_log = torch.log_softmax(baseline_logits / temperature, dim=-1)
    q_log = torch.log_softmax(perturbed_logits / temperature, dim=-1)
    p = p_log.exp()
    return torch.sum(p * (p_log - q_log), dim=-1).mean()


def _build_range_hook(
    group_id: str,
    accumulators: dict[str, dict[str, float]],
):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = _extract_first_tensor(output)
        if tensor is None:
            return
        values = tensor.detach().float()
        accumulators[group_id]["sum"] += float(values.max().item() - values.min().item())
        accumulators[group_id]["count"] += 1

    return hook


def _build_task_messages(
    task: Any,
    question: str,
    task_prompt_style: str,
) -> list[dict[str, str]]:
    try:
        return task.build_messages(question, prompt_style=task_prompt_style)
    except TypeError:
        return task.build_messages(question)


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _infer_model_device(model: Any) -> Any:
    import torch

    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return torch.device(device)
    return next(model.parameters()).device


def _move_inputs_to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return dict(inputs)
    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def _iter_encoded_batches(
    encoded_prompts: list[dict[str, Any]],
    *,
    batch_size: int,
):
    for batch_start in range(0, len(encoded_prompts), max(1, int(batch_size))):
        yield encoded_prompts[batch_start : batch_start + max(1, int(batch_size))]


def _iter_encoded_batches_with_start(
    encoded_prompts: list[dict[str, Any]],
    *,
    batch_size: int,
):
    for batch_start in range(0, len(encoded_prompts), max(1, int(batch_size))):
        yield batch_start, encoded_prompts[batch_start : batch_start + max(1, int(batch_size))]


def _pad_encoded_batch(tokenizer: Any, batch: list[dict[str, Any]]) -> dict[str, Any]:
    return tokenizer.pad(batch, padding=True, return_tensors="pt")


def _extract_first_tensor(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _extract_first_tensor(item)
            if resolved is not None:
                return resolved
    if isinstance(value, dict):
        for item in value.values():
            resolved = _extract_first_tensor(item)
            if resolved is not None:
                return resolved
    return None


def _add_noise_to_output(output: Any, delta: float, generator: Any) -> Any:
    import torch

    if torch.is_tensor(output):
        return output + _uniform_noise_like(output, delta=delta, generator=generator)
    if isinstance(output, tuple) and output:
        first = output[0]
        if torch.is_tensor(first):
            return (
                first + _uniform_noise_like(first, delta=delta, generator=generator),
                *output[1:],
            )
    raise TypeError(f"Unsupported module output type for perturbation: {type(output)!r}")


def _uniform_noise_like(tensor: Any, delta: float, generator: Any) -> Any:
    import torch

    noise = torch.empty(
        tensor.shape,
        device=tensor.device,
        dtype=torch.float32,
    ).uniform_(-delta / 2.0, delta / 2.0, generator=generator)
    return noise.to(dtype=tensor.dtype)


def _stable_seed(seed: int, group_id: str, bit_width: int, prompt_index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{group_id}:{bit_width}:{prompt_index}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_sensitivity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in dict(payload).items()
        if key not in {"metadata", "groups"}
    }
    normalized["metadata"] = dict(payload.get("metadata", {}))
    normalized["groups"] = {
        str(group_id): SensitivityRecord.from_dict(group_payload).to_dict()
        for group_id, group_payload in sorted(dict(payload.get("groups", {})).items())
    }
    return normalized
