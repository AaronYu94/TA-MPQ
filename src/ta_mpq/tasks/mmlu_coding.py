from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re


CHOICE_LETTERS = ("A", "B", "C", "D")
OPENAI_SIMPLE_EVALS_MULTICHOICE_PROMPT_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}
""".strip()
ANSWER_COLON_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
BOXED_CHOICE_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}", flags=re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?\s*([ABCD])\s*\)?",
    flags=re.IGNORECASE,
)
STANDALONE_CHOICE_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)

MMLU_CODING_SUBJECTS = (
    "college_computer_science",
    "computer_security",
    "high_school_computer_science",
    "machine_learning",
)


@dataclass(frozen=True, slots=True)
class MMLUCodingExample:
    example_id: str
    question: str
    answer: str


def load_examples(limit: int | None = None, split: str = "test") -> list[MMLUCodingExample]:
    return load_mmlu_coding_examples(limit=limit, split=split)


def load_mmlu_coding_examples(
    limit: int | None = None,
    split: str = "test",
) -> list[MMLUCodingExample]:
    from datasets import load_dataset

    requested_split = str(split or "test")
    dataset_split = "test" if requested_split in _NAMED_SUBSET_SPECS else requested_split
    last_error: Exception | None = None
    examples: list[MMLUCodingExample] = []

    for subject in MMLU_CODING_SUBJECTS:
        dataset = None
        for split_name in _split_candidates(dataset_split):
            try:
                dataset = load_dataset("cais/mmlu", subject, split=split_name)
                break
            except Exception as exc:  # pragma: no cover - remote dataset fallback
                last_error = exc
        if dataset is None:
            raise RuntimeError(
                f"Unable to load MMLU-coding subject '{subject}'"
            ) from last_error

        for index, row in enumerate(dataset):
            question = str(row.get("question") or row.get("input") or "").strip()
            choices = _extract_choices(row)
            answer_letter = _extract_reference_letter(row)
            example_key = str(row.get("id") or index)
            example_id = f"{subject}:{example_key}"
            examples.append(
                MMLUCodingExample(
                    example_id=example_id,
                    question=_format_question(question, choices),
                    answer=answer_letter,
                )
            )

    examples = select_examples_for_split(examples, requested_split)
    if limit is not None:
        examples = examples[: min(limit, len(examples))]
    return examples


_NAMED_SUBSET_SPECS = {
    "first100": ("head", 100),
    "first200": ("head", 200),
    "first300": ("head", 300),
    "last100": ("tail", 100),
    "full": ("all", None),
    "full400": ("all", None),
}


def select_examples_for_split(
    examples: list[MMLUCodingExample],
    split: str,
) -> list[MMLUCodingExample]:
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
    mode, value = subset_spec
    if mode == "head":
        return ordered[: int(value)]
    if mode == "tail":
        return ordered[-int(value) :]
    return ordered


def select_examples_by_id(
    examples: list[MMLUCodingExample],
    example_ids: Iterable[str],
) -> list[MMLUCodingExample]:
    selected_ids = [str(example_id) for example_id in example_ids]
    if not selected_ids:
        return []

    lookup = {str(example.example_id): example for example in examples}
    missing_ids = [example_id for example_id in selected_ids if example_id not in lookup]
    if missing_ids:
        raise ValueError(
            "Unknown MMLU-coding example ids requested: " + ", ".join(missing_ids[:5])
        )
    return [lookup[example_id] for example_id in selected_ids]


def build_messages(
    question: str,
    prompt_style: str = "simple_evals",
) -> list[dict[str, str]]:
    normalized_style = str(prompt_style or "simple_evals").strip().lower()
    if normalized_style in {"simple_evals", "openai_simple_evals", "simple_evals_cot"}:
        return [
            {
                "role": "user",
                "content": OPENAI_SIMPLE_EVALS_MULTICHOICE_PROMPT_TEMPLATE.format(
                    question=question.strip()
                ),
            },
        ]
    if normalized_style in {"simple_evals_nonthinking", "multiple_choice_letter"}:
        return [
            {
                "role": "user",
                "content": (
                    question.strip()
                    + "\n\nRespond with only the single letter of the correct answer (A, B, C, or D)."
                ),
            },
        ]
    if normalized_style in {"reasoning_boxed", "qwen_prompt_thinking"}:
        return [
            {
                "role": "user",
                "content": (
                    question.strip()
                    + "\n\nThink step by step, then give the final answer as a single letter in \\boxed{}."
                ),
            },
        ]
    raise ValueError(f"Unsupported MMLU-coding prompt_style: {prompt_style}")


def extract_reference_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if normalized not in CHOICE_LETTERS:
        raise ValueError(f"Could not parse MMLU-coding reference answer: {answer!r}")
    return normalized


def extract_prediction_answer(completion: str) -> str | None:
    predicted, _, _ = extract_prediction_details(completion)
    return predicted


def extract_prediction_details(completion: str) -> tuple[str | None, str, bool]:
    has_boxed_answer = bool(BOXED_CHOICE_RE.search(completion))
    normalized_completion = _normalize_response_for_simple_evals(completion)
    answer_colon = ANSWER_COLON_RE.findall(normalized_completion)
    if answer_colon:
        return normalize_answer(answer_colon[-1]), "answer_colon", has_boxed_answer

    boxed = BOXED_CHOICE_RE.findall(completion)
    if boxed:
        return normalize_answer(boxed[-1]), "boxed", has_boxed_answer

    prefixed = ANSWER_PREFIX_RE.findall(normalized_completion)
    if prefixed:
        return normalize_answer(prefixed[-1]), "answer_prefix", has_boxed_answer

    lines = [line.strip() for line in normalized_completion.splitlines() if line.strip()]
    for line in reversed(lines):
        if normalize_answer(line) in CHOICE_LETTERS:
            return normalize_answer(line), "last_line", has_boxed_answer
        standalone = STANDALONE_CHOICE_RE.findall(line)
        if standalone:
            return normalize_answer(standalone[-1]), "standalone", has_boxed_answer
    return None, "none", has_boxed_answer


def normalize_answer(text: str) -> str:
    return str(text or "").strip().upper().rstrip(".")


def _normalize_response_for_simple_evals(response: str) -> str:
    """Mirror OpenAI simple-evals response cleanup before answer extraction."""
    return (
        str(response or "")
        .replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


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


def _split_candidates(dataset_split: str) -> tuple[str, ...]:
    primary = str(dataset_split or "test")
    candidates = [primary]
    for fallback in ("test", "validation"):
        if fallback not in candidates:
            candidates.append(fallback)
    return tuple(candidates)


def _extract_choices(row: dict[str, object]) -> list[str]:
    raw_choices = row.get("choices")
    if isinstance(raw_choices, list) and len(raw_choices) >= 4:
        return [str(choice) for choice in raw_choices[:4]]

    options = [row.get(letter) for letter in CHOICE_LETTERS]
    if all(option is not None for option in options):
        return [str(option) for option in options]

    raise ValueError(f"Could not extract 4-way choices from row: {row!r}")


def _extract_reference_letter(row: dict[str, object]) -> str:
    for key in ("answer", "target"):
        raw_answer = row.get(key)
        if raw_answer is None:
            continue
        if isinstance(raw_answer, int):
            if 0 <= raw_answer < len(CHOICE_LETTERS):
                return CHOICE_LETTERS[raw_answer]
        normalized = normalize_answer(str(raw_answer))
        if normalized in CHOICE_LETTERS:
            return normalized
        if normalized.isdigit():
            answer_index = int(normalized)
            if 0 <= answer_index < len(CHOICE_LETTERS):
                return CHOICE_LETTERS[answer_index]
    raise ValueError(f"Could not parse MMLU-coding answer from row: {row!r}")


def _format_question(question: str, choices: list[str]) -> str:
    option_lines = [f"{letter}) {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)]
    return question.strip() + "\n" + "\n".join(option_lines)


def _stable_example_sort_key(example_id: str) -> str:
    return hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()
