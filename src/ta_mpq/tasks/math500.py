from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re


INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+answer\s+is|answer\s+is|final\s+answer\s+is|final\s+answer|answer)\s*[:=]?\s*",
    flags=re.IGNORECASE,
)
FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")


@dataclass(frozen=True, slots=True)
class MATH500Example:
    example_id: str
    question: str
    answer: str


def load_examples(limit: int | None = None, split: str = "test") -> list[MATH500Example]:
    return load_math500_examples(limit=limit, split=split)


def load_math500_examples(limit: int | None = None, split: str = "test") -> list[MATH500Example]:
    from datasets import load_dataset

    requested_split = str(split or "test")
    dataset_split = "test" if requested_split in _NAMED_SUBSET_SPECS else requested_split
    dataset = None
    last_error: Exception | None = None
    split_candidates = (dataset_split, "test", "train")
    dataset_candidates = ("HuggingFaceH4/MATH-500", "ankner/math-500")

    for dataset_name in dataset_candidates:
        for split_name in split_candidates:
            try:
                dataset = load_dataset(dataset_name, split=split_name)
                break
            except Exception as exc:  # pragma: no cover - remote dataset fallback
                last_error = exc
        if dataset is not None:
            break

    if dataset is None:
        raise RuntimeError("Unable to load a MATH-500 dataset split") from last_error

    examples: list[MATH500Example] = []
    for index, row in enumerate(dataset):
        example_id = str(row.get("unique_id") or row.get("id") or index)
        question = str(row.get("problem") or row.get("question") or "")
        answer = str(row.get("answer") or row.get("final_answer") or "")
        examples.append(
            MATH500Example(
                example_id=example_id,
                question=question,
                answer=answer,
            )
        )
    examples = select_examples_for_split(examples, requested_split)
    if limit is not None:
        examples = examples[: min(limit, len(examples))]
    return examples


_NAMED_SUBSET_SPECS = {
    "dev100": (0, 100),
    "accept200": (100, 300),
    "train200": (0, 200),
    "test300": (200, 500),
    "first300": (0, 300),
    "hard50": (450, 500),
    "last100": (400, 500),
    "full500": (0, None),
}


def select_examples_for_split(
    examples: list[MATH500Example],
    split: str,
) -> list[MATH500Example]:
    requested_split = str(split or "test")
    subset_spec = _NAMED_SUBSET_SPECS.get(requested_split)
    if subset_spec is None:
        return list(examples)

    ordered = sorted(
        examples,
        key=lambda example: (
            _stable_example_sort_key(example.example_id),
            str(example.example_id),
        ),
    )
    start_index, end_index = subset_spec
    return ordered[start_index:end_index]


def select_examples_by_id(
    examples: list[MATH500Example],
    example_ids: Iterable[str],
) -> list[MATH500Example]:
    selected_ids = [str(example_id) for example_id in example_ids]
    if not selected_ids:
        return []

    lookup = {str(example.example_id): example for example in examples}
    missing_ids = [example_id for example_id in selected_ids if example_id not in lookup]
    if missing_ids:
        raise ValueError(
            "Unknown MATH-500 example ids requested: " + ", ".join(missing_ids[:5])
        )
    return [lookup[example_id] for example_id in selected_ids]


def _stable_example_sort_key(example_id: str) -> str:
    return hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()


def build_messages(
    question: str,
    prompt_style: str = "simple_evals",
) -> list[dict[str, str]]:
    normalized_style = str(prompt_style or "simple_evals").strip().lower()
    if normalized_style in {
        "simple_evals",
        "simple_evals_nonthinking",
        "qwen_prompt_thinking",
        "reasoning_boxed",
        "deepseek_math",
    }:
        return [
            {
                "role": "user",
                "content": (
                    question.strip()
                    + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
                ),
            },
        ]
    raise ValueError(f"Unsupported MATH-500 prompt_style: {prompt_style}")


def extract_reference_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError(f"Could not parse MATH-500 reference answer: {answer!r}")
    return normalized


def extract_prediction_answer(completion: str) -> str | None:
    predicted, _, _ = extract_prediction_details(completion)
    return predicted


def extract_prediction_details(completion: str) -> tuple[str | None, str, bool]:
    boxed = _extract_last_boxed_content(completion)
    has_boxed_answer = boxed is not None
    if boxed is not None:
        normalized = normalize_answer(boxed)
        if normalized:
            return normalized, "boxed", has_boxed_answer

    inline_math = INLINE_MATH_RE.findall(completion)
    if inline_math:
        normalized = normalize_answer(inline_math[-1])
        if normalized:
            return normalized, "inline_math", has_boxed_answer

    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = ANSWER_PREFIX_RE.sub("", line)
        normalized = normalize_answer(candidate)
        if normalized:
            return normalized, "last_line", has_boxed_answer
    return None, "none", has_boxed_answer


def normalize_answer(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""

    candidate = candidate.replace("$", "")
    candidate = candidate.replace("\\left", "")
    candidate = candidate.replace("\\right", "")
    candidate = candidate.replace("\\!", "")
    candidate = candidate.replace("\\,", "")
    candidate = candidate.replace("\\ ", "")
    candidate = candidate.replace("−", "-")
    candidate = candidate.rstrip(".")
    candidate = _normalize_fracs(candidate)
    candidate = re.sub(r"\s+", "", candidate)
    return candidate


def score_prediction(completion: str, reference_answer: str) -> tuple[str | None, bool]:
    predicted = extract_prediction_answer(completion)
    if predicted is None:
        return None, False
    return predicted, predicted == normalize_answer(reference_answer)


def score_prediction_detailed(
    completion: str,
    reference_answer: str,
) -> tuple[str | None, bool, dict[str, object]]:
    predicted, extraction_source, has_boxed_answer = extract_prediction_details(completion)
    if predicted is None:
        return None, False, {
            "answer_extraction_source": extraction_source,
            "has_boxed_answer": has_boxed_answer,
        }
    return predicted, predicted == normalize_answer(reference_answer), {
        "answer_extraction_source": extraction_source,
        "has_boxed_answer": has_boxed_answer,
    }


def _normalize_fracs(text: str) -> str:
    candidate = text
    while True:
        updated = FRAC_RE.sub(r"(\1)/(\2)", candidate)
        if updated == candidate:
            return candidate
        candidate = updated


def _extract_last_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    index = start + len(marker)
    depth = 1
    content_chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            content_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars)
            content_chars.append(char)
        else:
            content_chars.append(char)
        index += 1
    return None
