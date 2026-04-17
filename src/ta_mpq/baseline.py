from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from ta_mpq.metrics import ExampleResult, summarize_results
from ta_mpq.tasks import load_task_adapter


def evaluate_gsm8k_baseline(
    model_id: str,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return evaluate_task_baseline(
        model_ref=model_id,
        task_name="gsm8k",
        limit=limit,
        max_new_tokens=max_new_tokens,
        tokenizer_source=model_id,
        model_label=model_id,
        load_dtype="bfloat16",
    )


def evaluate_gsm8k_model(
    model_ref: str,
    limit: int,
    max_new_tokens: int,
    tokenizer_source: str | None = None,
    model_label: str | None = None,
    load_dtype: str = "auto",
) -> dict[str, Any]:
    return evaluate_task_model(
        model_ref=model_ref,
        task_name="gsm8k",
        limit=limit,
        max_new_tokens=max_new_tokens,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        load_dtype=load_dtype,
    )


def evaluate_task_baseline(
    model_ref: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    tokenizer_source: str | None = None,
    model_label: str | None = None,
    load_dtype: str = "auto",
) -> dict[str, Any]:
    return evaluate_task_model(
        model_ref=model_ref,
        task_name=task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        load_dtype=load_dtype,
    )


def evaluate_task_model(
    model_ref: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    tokenizer_source: str | None = None,
    model_label: str | None = None,
    load_dtype: str = "auto",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_tokenizer_source = tokenizer_source or model_ref
    resolved_model_label = model_label or model_ref
    task = load_task_adapter(task_name)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(0)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if load_dtype == "auto":
        model_kwargs["dtype"] = "auto"
    elif load_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"Unsupported load_dtype: {load_dtype}")

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    model.eval()

    device = _infer_model_device(model)
    examples = task.load_examples(limit=limit)

    resident_memory_mb = _current_cuda_memory_mb(device)
    results: list[ExampleResult] = []

    for example in examples:
        prompt = _render_prompt(tokenizer, task.build_messages(example.question))
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.type == "cuda":
            inputs = {key: value.to(device) for key, value in inputs.items()}

        prompt_tokens = int(inputs["input_ids"].shape[-1])

        base_memory_mb = _current_cuda_memory_mb(device)
        _reset_peak_memory(device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_sec = time.perf_counter() - start

        generated_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        reference_answer = task.extract_reference_answer(example.answer)
        predicted_answer, is_correct = task.score_prediction(completion, reference_answer)

        total_peak_memory_mb = _peak_cuda_memory_mb(device)
        generation_peak_delta_mb = max(total_peak_memory_mb - base_memory_mb, 0.0)

        results.append(
            ExampleResult(
                example_id=example.example_id,
                model_id=resolved_model_label,
                reference_answer=reference_answer,
                predicted_answer=predicted_answer,
                raw_completion=completion,
                is_correct=is_correct,
                latency_sec=latency_sec,
                prompt_tokens=prompt_tokens,
                completion_tokens=int(generated_tokens.shape[-1]),
                resident_memory_mb=resident_memory_mb,
                generation_peak_delta_mb=generation_peak_delta_mb,
                total_peak_memory_mb=total_peak_memory_mb,
            )
        )

    return summarize_results(model_id=resolved_model_label, task_name=task_name, results=results)


def save_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def _current_cuda_memory_mb(device: Any) -> float:
    import torch

    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**2)


def _peak_cuda_memory_mb(device: Any) -> float:
    import torch

    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _reset_peak_memory(device: Any) -> None:
    import torch

    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)
