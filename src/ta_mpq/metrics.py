from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics


@dataclass(frozen=True, slots=True)
class ExampleResult:
    example_id: str
    model_id: str
    reference_answer: str
    predicted_answer: str | None
    raw_completion: str
    is_correct: bool
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    resident_memory_mb: float
    generation_peak_delta_mb: float
    total_peak_memory_mb: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_results(
    model_id: str,
    task_name: str,
    results: list[ExampleResult],
) -> dict[str, object]:
    if not results:
        return {
            "model_id": model_id,
            "task_name": task_name,
            "num_examples": 0,
            "accuracy": 0.0,
            "mean_latency_sec": 0.0,
            "median_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
            "mean_resident_memory_mb": 0.0,
            "mean_generation_peak_delta_mb": 0.0,
            "mean_total_peak_memory_mb": 0.0,
            "results": [],
        }

    latencies = [result.latency_sec for result in results]
    resident_memories = [result.resident_memory_mb for result in results]
    generation_peak_deltas = [result.generation_peak_delta_mb for result in results]
    total_peaks = [result.total_peak_memory_mb for result in results]
    num_correct = sum(1 for result in results if result.is_correct)

    return {
        "model_id": model_id,
        "task_name": task_name,
        "num_examples": len(results),
        "num_correct": num_correct,
        "accuracy": num_correct / len(results),
        "mean_latency_sec": statistics.fmean(latencies),
        "median_latency_sec": statistics.median(latencies),
        "p95_latency_sec": _percentile(latencies, 95),
        "mean_resident_memory_mb": statistics.fmean(resident_memories),
        "mean_generation_peak_delta_mb": statistics.fmean(generation_peak_deltas),
        "mean_total_peak_memory_mb": statistics.fmean(total_peaks),
        "results": [result.to_dict() for result in results],
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = round((percentile / 100) * (len(ordered) - 1))
    return ordered[rank]
