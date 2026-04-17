from __future__ import annotations

from dataclasses import dataclass
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

    dataset = None
    last_error: Exception | None = None
    split_candidates = (split, "test", "train")
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

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

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
    return examples


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve competition-style math problems. "
                "Return only the final answer as a concise mathematical expression. "
                "Do not include reasoning, analysis, or extra words."
            ),
        },
        {
            "role": "user",
            "content": question.strip() + "\n\nRespond with only the final answer.",
        },
    ]


def extract_reference_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError(f"Could not parse MATH-500 reference answer: {answer!r}")
    return normalized


def extract_prediction_answer(completion: str) -> str | None:
    boxed = _extract_last_boxed_content(completion)
    if boxed is not None:
        normalized = normalize_answer(boxed)
        if normalized:
            return normalized

    inline_math = INLINE_MATH_RE.findall(completion)
    if inline_math:
        normalized = normalize_answer(inline_math[-1])
        if normalized:
            return normalized

    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = ANSWER_PREFIX_RE.sub("", line)
        normalized = normalize_answer(candidate)
        if normalized:
            return normalized
    return None


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
