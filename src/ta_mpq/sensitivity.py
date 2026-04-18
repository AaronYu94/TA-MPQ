from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.tasks import load_task_adapter


@dataclass(frozen=True, slots=True)
class ModuleActivationStat:
    name: str
    parameter_count: int
    mean_abs_input: float
    mean_abs_output: float
    num_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModuleKLDivergenceStat:
    name: str
    parameter_count: int
    mean_output_kl: float
    num_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskSensitivityGroupStat:
    name: str
    component_type: str
    layer_names: tuple[str, ...]
    parameter_count: int
    prior_sensitivity: float
    activation_score: float
    normalized_activation_score: float
    kl_divergence_score: float
    normalized_kl_divergence_score: float
    combined_sensitivity: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_names"] = list(self.layer_names)
        return payload


def collect_task_activation_stats(
    model_id: str,
    task_name: str,
    limit: int,
    max_prompt_tokens: int = 1024,
) -> list[ModuleActivationStat]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task = load_task_adapter(task_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = _infer_model_device(model)
    module_param_counts: dict[str, int] = {}
    module_accumulators: dict[str, dict[str, float]] = {}
    hooks = []

    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        module_param_counts[module_name] = int(weight.numel())
        module_accumulators[module_name] = {
            "input_abs_sum": 0.0,
            "output_abs_sum": 0.0,
            "count": 0.0,
        }
        hooks.append(module.register_forward_hook(_build_activation_hook(module_name, module_accumulators)))

    try:
        for example in task.load_examples(limit=limit):
            prompt = _render_prompt(tokenizer, task.build_messages(example.question))
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_tokens,
            )
            if device.type == "cuda":
                inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                model(**inputs, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    activation_stats: list[ModuleActivationStat] = []
    for module_name in sorted(module_param_counts):
        accumulator = module_accumulators[module_name]
        num_observations = int(accumulator["count"])
        if num_observations == 0:
            mean_abs_input = 0.0
            mean_abs_output = 0.0
        else:
            mean_abs_input = accumulator["input_abs_sum"] / num_observations
            mean_abs_output = accumulator["output_abs_sum"] / num_observations
        activation_stats.append(
            ModuleActivationStat(
                name=module_name,
                parameter_count=module_param_counts[module_name],
                mean_abs_input=mean_abs_input,
                mean_abs_output=mean_abs_output,
                num_observations=num_observations,
            )
        )
    return activation_stats


def build_task_sensitivity_profile(
    layer_stats: list[LinearLayerStat],
    activation_stats: list[ModuleActivationStat],
    grouping: str,
    activation_weight: float = 0.55,
    kl_stats: list[ModuleKLDivergenceStat] | None = None,
    kl_weight: float = 0.0,
) -> dict[str, Any]:
    from ta_mpq.search import build_search_groups

    if activation_weight < 0.0:
        raise ValueError("activation_weight must be non-negative")
    if kl_weight < 0.0:
        raise ValueError("kl_weight must be non-negative")
    if activation_weight + kl_weight > 1.0:
        raise ValueError("activation_weight + kl_weight must not exceed 1.0")

    base_groups = build_search_groups(layer_stats, grouping=grouping)
    activation_lookup = {stat.name: stat for stat in activation_stats}
    kl_lookup = {stat.name: stat for stat in (kl_stats or [])}

    raw_activation_scores: dict[str, float] = {}
    raw_kl_scores: dict[str, float] = {}
    for group in base_groups:
        raw_activation_scores[group.name] = _group_activation_score(group.layer_names, activation_lookup)
        raw_kl_scores[group.name] = _group_kl_score(group.layer_names, kl_lookup)

    normalized_scores = _normalize_scores(raw_activation_scores)
    normalized_kl_scores = _normalize_scores(raw_kl_scores)
    prior_weight = 1.0 - activation_weight - kl_weight

    groups_payload: list[dict[str, Any]] = []
    for group in base_groups:
        activation_score = raw_activation_scores[group.name]
        normalized_activation = normalized_scores[group.name]
        kl_score = raw_kl_scores[group.name]
        normalized_kl = normalized_kl_scores[group.name]
        combined = (
            prior_weight * group.sensitivity
            + activation_weight * normalized_activation
            + kl_weight * normalized_kl
        )
        groups_payload.append(
            TaskSensitivityGroupStat(
                name=group.name,
                component_type=group.component_type,
                layer_names=group.layer_names,
                parameter_count=group.parameter_count,
                prior_sensitivity=group.sensitivity,
                activation_score=activation_score,
                normalized_activation_score=normalized_activation,
                kl_divergence_score=kl_score,
                normalized_kl_divergence_score=normalized_kl,
                combined_sensitivity=combined,
            ).to_dict()
        )

    return {
        "grouping": grouping,
        "activation_weight": activation_weight,
        "kl_weight": kl_weight,
        "prior_weight": prior_weight,
        "num_layer_stats": len(layer_stats),
        "num_activation_stats": len(activation_stats),
        "num_kl_stats": len(kl_stats or []),
        "groups": groups_payload,
        "module_activation_stats": [stat.to_dict() for stat in activation_stats],
        "module_kl_stats": [stat.to_dict() for stat in (kl_stats or [])],
    }


def build_task_kl_sensitivity_profile(
    layer_stats: list[LinearLayerStat],
    kl_stats: list[ModuleKLDivergenceStat],
    grouping: str,
    kl_weight: float = 0.55,
) -> dict[str, Any]:
    return build_task_sensitivity_profile(
        layer_stats=layer_stats,
        activation_stats=[],
        grouping=grouping,
        activation_weight=0.0,
        kl_stats=kl_stats,
        kl_weight=kl_weight,
    )


def group_sensitivity_overrides_from_profile(
    profile_payload: dict[str, Any],
    field: str = "combined_sensitivity",
) -> dict[str, float]:
    groups_payload = profile_payload.get("groups")
    if not groups_payload:
        raise ValueError("Sensitivity profile does not include groups")
    overrides: dict[str, float] = {}
    for group_payload in groups_payload:
        overrides[str(group_payload["name"])] = float(group_payload[field])
    return overrides


def _build_activation_hook(
    module_name: str,
    accumulators: dict[str, dict[str, float]],
):
    def hook(_: Any, inputs: tuple[Any, ...], output: Any) -> None:
        input_tensor = _extract_first_tensor(inputs)
        output_tensor = _extract_first_tensor(output)
        if input_tensor is None and output_tensor is None:
            return

        accumulator = accumulators[module_name]
        if input_tensor is not None:
            accumulator["input_abs_sum"] += _mean_abs(input_tensor)
        if output_tensor is not None:
            accumulator["output_abs_sum"] += _mean_abs(output_tensor)
        accumulator["count"] += 1.0

    return hook


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

    first_parameter = next(model.parameters())
    return first_parameter.device


def _extract_first_tensor(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _mean_abs(tensor: Any) -> float:
    return float(tensor.detach().float().abs().mean().cpu())


def _group_activation_score(
    layer_names: tuple[str, ...],
    activation_lookup: dict[str, ModuleActivationStat],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for layer_name in layer_names:
        activation_stat = activation_lookup.get(layer_name)
        if activation_stat is None:
            continue
        weight = activation_stat.parameter_count
        numerator += weight * ((activation_stat.mean_abs_input + activation_stat.mean_abs_output) / 2.0)
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _group_kl_score(
    layer_names: tuple[str, ...],
    kl_lookup: dict[str, ModuleKLDivergenceStat],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for layer_name in layer_names:
        kl_stat = kl_lookup.get(layer_name)
        if kl_stat is None:
            continue
        weight = kl_stat.parameter_count
        numerator += weight * kl_stat.mean_output_kl
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum <= minimum:
        fill_value = 0.0 if maximum == 0.0 else 1.0
        return {name: fill_value for name in scores}
    scale = maximum - minimum
    return {name: (value - minimum) / scale for name, value in scores.items()}
