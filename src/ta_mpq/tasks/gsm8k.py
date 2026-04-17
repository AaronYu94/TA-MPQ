from __future__ import annotations

from dataclasses import dataclass
import re


ANSWER_RE = re.compile(r"####\s*([^\n]+)")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True, slots=True)
class GSM8KExample:
    example_id: str
    question: str
    answer: str


def load_examples(limit: int | None = None, split: str = "test") -> list[GSM8KExample]:
    return load_gsm8k_examples(limit=limit, split=split)


def load_gsm8k_examples(limit: int | None = None, split: str = "test") -> list[GSM8KExample]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples: list[GSM8KExample] = []
    for index, row in enumerate(dataset):
        examples.append(
            GSM8KExample(
                example_id=str(row.get("id") or index),
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
        )
    return examples


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve grade-school math word problems. "
                "Return only the final numeric answer. "
                "Do not include reasoning, analysis, or extra words."
            ),
        },
        {
            "role": "user",
            "content": question.strip() + "\n\nRespond with only the final answer as a number.",
        },
    ]


def extract_reference_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    if not match:
        raise ValueError(f"Could not parse GSM8K reference answer: {answer!r}")
    return normalize_answer(match.group(1))


def extract_prediction_answer(completion: str) -> str | None:
    boxed = BOXED_RE.findall(completion)
    if boxed:
        return normalize_answer(boxed[-1])

    numbers = NUMBER_RE.findall(completion)
    if not numbers:
        return None
    return normalize_answer(numbers[-1])


def normalize_answer(text: str) -> str:
    candidate = text.strip()
    candidate = candidate.replace("$", "")
    candidate = candidate.replace(",", "")
    candidate = candidate.rstrip(".")
    candidate = candidate.strip()
    return candidate


def score_prediction(completion: str, reference_answer: str) -> tuple[str | None, bool]:
    predicted = extract_prediction_answer(completion)
    if predicted is None:
        return None, False
    return predicted, predicted == normalize_answer(reference_answer)
