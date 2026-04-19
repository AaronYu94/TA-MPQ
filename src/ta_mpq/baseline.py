from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from ta_mpq.metrics import ExampleResult, summarize_results
from ta_mpq.tasks import load_task_adapter
from ta_mpq.transformers_compat import apply_qwen3_5_fast_path_compat_patch


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
    split: str = "test",
    example_ids: list[str] | None = None,
    tokenizer_source: str | None = None,
    model_label: str | None = None,
    load_dtype: str = "auto",
    task_prompt_style: str = "",
) -> dict[str, Any]:
    return evaluate_task_model(
        model_ref=model_ref,
        task_name=task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        split=split,
        example_ids=example_ids,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        load_dtype=load_dtype,
        task_prompt_style=task_prompt_style,
    )


def evaluate_task_model(
    model_ref: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    split: str = "test",
    example_ids: list[str] | None = None,
    tokenizer_source: str | None = None,
    model_label: str | None = None,
    load_dtype: str = "auto",
    task_prompt_style: str = "",
) -> dict[str, Any]:
    import torch

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_tokenizer_source = tokenizer_source or model_ref
    resolved_model_label = model_label or model_ref
    task = load_task_adapter(task_name)
    evaluation_mode = _resolve_task_evaluation_mode(
        task_name=task_name,
        task_prompt_style=task_prompt_style,
    )
    resolved_task_prompt_style = evaluation_mode["task_prompt_style"]

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
        model_kwargs["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"Unsupported load_dtype: {load_dtype}")

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    device = _infer_model_device(model)
    examples = task.load_examples(limit=None if example_ids else limit, split=split)
    if example_ids is not None:
        selector = getattr(task, "select_examples_by_id", None)
        if not callable(selector):
            raise ValueError(f"Task {task_name} does not support explicit example-id evaluation")
        examples = selector(examples, example_ids)
        if limit is not None:
            examples = examples[: min(limit, len(examples))]

    resident_memory_mb = _current_cuda_memory_mb(device)
    results: list[ExampleResult] = []

    for example in examples:
        prompt = _render_prompt(
            tokenizer,
            _build_task_messages(
                task=task,
                question=example.question,
                task_prompt_style=resolved_task_prompt_style,
            ),
            enable_thinking=evaluation_mode["enable_thinking"],
        )
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
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        latency_sec = time.perf_counter() - start

        generated_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        reference_answer = task.extract_reference_answer(example.answer)
        predicted_answer, is_correct, prediction_metadata = _score_task_prediction(
            task=task,
            completion=completion,
            reference_answer=reference_answer,
        )
        completion_token_count = int(generated_tokens.shape[-1])

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
                completion_tokens=completion_token_count,
                resident_memory_mb=resident_memory_mb,
                generation_peak_delta_mb=generation_peak_delta_mb,
                total_peak_memory_mb=total_peak_memory_mb,
                answer_extraction_source=prediction_metadata["answer_extraction_source"],
                has_boxed_answer=prediction_metadata["has_boxed_answer"],
                length_capped=completion_token_count >= max_new_tokens,
            )
        )

        completed = len(results)
        if completed % 5 == 0 or completed == len(examples):
            num_correct = sum(1 for item in results if item.is_correct)
            print(
                (
                    f"[eval:{task_name}] completed {completed}/{len(examples)} "
                    f"examples, accuracy_so_far={num_correct / completed:.3f}, "
                    f"last_latency_sec={latency_sec:.2f}"
                ),
                flush=True,
            )

    summary = summarize_results(model_id=resolved_model_label, task_name=task_name, results=results)
    summary["task_split"] = split
    summary["evaluated_example_ids"] = [result.example_id for result in results]
    summary["task_prompt_style"] = resolved_task_prompt_style or None
    summary["thinking_mode"] = evaluation_mode["thinking_mode"]
    summary["generation_mode"] = evaluation_mode["generation_mode"]
    summary["max_new_tokens"] = max_new_tokens
    return summary


def save_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_task_messages(
    task: Any,
    question: str,
    task_prompt_style: str,
) -> list[dict[str, str]]:
    if task_prompt_style:
        try:
            return task.build_messages(question, prompt_style=task_prompt_style)
        except TypeError:
            return task.build_messages(question)
    return task.build_messages(question)


def _resolve_task_prompt_style(task_name: str, task_prompt_style: str) -> str:
    normalized_task_name = str(task_name or "").strip().lower()
    if task_prompt_style:
        return task_prompt_style
    if normalized_task_name in {"math500", "math-500"}:
        return "simple_evals"
    return ""


def _resolve_task_evaluation_mode(
    task_name: str,
    task_prompt_style: str,
) -> dict[str, Any]:
    normalized_task_name = str(task_name or "").strip().lower()
    resolved_task_prompt_style = _resolve_task_prompt_style(task_name, task_prompt_style)
    if normalized_task_name in {"math500", "math-500"}:
        if resolved_task_prompt_style == "simple_evals_nonthinking":
            return {
                "task_prompt_style": resolved_task_prompt_style,
                "enable_thinking": False,
                "thinking_mode": "disabled",
                "generation_mode": "greedy",
            }
        return {
            "task_prompt_style": resolved_task_prompt_style,
            "enable_thinking": True,
            "thinking_mode": "enabled",
            "generation_mode": "greedy",
        }
    return {
        "task_prompt_style": resolved_task_prompt_style,
        "enable_thinking": False,
        "thinking_mode": "disabled",
        "generation_mode": "greedy",
    }


def _score_task_prediction(
    task: Any,
    completion: str,
    reference_answer: str,
) -> tuple[str | None, bool, dict[str, Any]]:
    detailed_scorer = getattr(task, "score_prediction_detailed", None)
    if callable(detailed_scorer):
        predicted_answer, is_correct, metadata = detailed_scorer(completion, reference_answer)
        return predicted_answer, is_correct, {
            "answer_extraction_source": metadata.get("answer_extraction_source"),
            "has_boxed_answer": metadata.get("has_boxed_answer"),
        }

    predicted_answer, is_correct = task.score_prediction(completion, reference_answer)
    return predicted_answer, is_correct, {
        "answer_extraction_source": None,
        "has_boxed_answer": None,
    }


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
