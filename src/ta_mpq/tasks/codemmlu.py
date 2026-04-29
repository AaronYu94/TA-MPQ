from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re


CODEMMLU_DATASET = "Fsoft-AIC/CodeMMLU"
CODEMMLU_CONFIGS = (
    "api_frameworks",
    "code_completion",
    "code_repair",
    "dbms_sql",
    "execution_prediction",
    "fill_in_the_middle",
    "others",
    "programming_syntax",
    "software_principles",
)
CODEMMLU_CHOICE_LETTERS = tuple("ABCDEFG")
CODEMMLU_SIMPLE_EVALS_MULTICHOICE_PROMPT_TEMPLATE = """
Answer the following multiple choice question. Reply with exactly one line in this format:
Answer: $LETTER

LETTER must be one of the listed option letters. Do not include reasoning, explanations, code fences, or any extra text.

{question}
""".strip()
CODEMMLU_COT_MULTICHOICE_PROMPT_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the listed option letters. Think step by step before answering.

{question}
""".strip()
ANSWER_COLON_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-G])\$?")
BOXED_CHOICE_RE = re.compile(r"\\boxed\{\s*([A-G])\s*\}", flags=re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?\s*([A-G])\s*\)?",
    flags=re.IGNORECASE,
)
STANDALONE_CHOICE_RE = re.compile(r"\b([A-G])\b", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CodeMMLUExample:
    example_id: str
    question: str
    answer: str


def load_examples(limit: int | None = None, split: str = "test") -> list[CodeMMLUExample]:
    return load_codemmlu_examples(limit=limit, split=split)


def load_codemmlu_examples(
    limit: int | None = None,
    split: str = "test",
) -> list[CodeMMLUExample]:
    from datasets import load_dataset

    requested_split = str(split or "test")
    dataset_split = "test" if requested_split in _NAMED_SUBSET_SPECS else requested_split
    last_error: Exception | None = None
    examples: list[CodeMMLUExample] = []

    for config_name in CODEMMLU_CONFIGS:
        dataset = None
        for split_name in _split_candidates(dataset_split):
            try:
                dataset = load_dataset(CODEMMLU_DATASET, config_name, split=split_name)
                break
            except Exception as exc:  # pragma: no cover - remote dataset fallback
                last_error = exc
        if dataset is None:
            raise RuntimeError(
                f"Unable to load CodeMMLU config '{config_name}'"
            ) from last_error

        for index, row in enumerate(dataset):
            question = str(row.get("question") or "").strip()
            choices = _extract_choices(row)
            answer_letter = extract_reference_answer(str(row.get("answer") or ""))
            task_id = str(row.get("task_id") or index)
            example_id = f"{config_name}:{task_id}"
            problem_description = row.get("problem_description")
            examples.append(
                CodeMMLUExample(
                    example_id=example_id,
                    question=_format_question(
                        question=question,
                        choices=choices,
                        problem_description=(
                            str(problem_description).strip()
                            if problem_description is not None
                            else ""
                        ),
                    ),
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
}


def select_examples_for_split(
    examples: list[CodeMMLUExample],
    split: str,
) -> list[CodeMMLUExample]:
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
    examples: list[CodeMMLUExample],
    example_ids: Iterable[str],
) -> list[CodeMMLUExample]:
    selected_ids = [str(example_id) for example_id in example_ids]
    if not selected_ids:
        return []

    lookup = {str(example.example_id): example for example in examples}
    missing_ids = [example_id for example_id in selected_ids if example_id not in lookup]
    if missing_ids:
        raise ValueError(
            "Unknown CodeMMLU example ids requested: " + ", ".join(missing_ids[:5])
        )
    return [lookup[example_id] for example_id in selected_ids]


def build_messages(
    question: str,
    prompt_style: str = "simple_evals",
) -> list[dict[str, str]]:
    normalized_style = str(prompt_style or "simple_evals").strip().lower()
    if normalized_style in {"simple_evals", "openai_simple_evals"}:
        return [
            {
                "role": "user",
                "content": CODEMMLU_SIMPLE_EVALS_MULTICHOICE_PROMPT_TEMPLATE.format(
                    question=question.strip()
                ),
            },
        ]
    if normalized_style in {"simple_evals_cot", "openai_simple_evals_cot"}:
        return [
            {
                "role": "user",
                "content": CODEMMLU_COT_MULTICHOICE_PROMPT_TEMPLATE.format(
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
                    + "\n\nRespond with only the single letter of the correct answer."
                ),
            },
        ]
    raise ValueError(f"Unsupported CodeMMLU prompt_style: {prompt_style}")


def extract_reference_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if normalized not in CODEMMLU_CHOICE_LETTERS:
        raise ValueError(f"Could not parse CodeMMLU reference answer: {answer!r}")
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
        if normalize_answer(line) in CODEMMLU_CHOICE_LETTERS:
            return normalize_answer(line), "last_line", has_boxed_answer
        standalone = STANDALONE_CHOICE_RE.findall(line)
        if standalone:
            return normalize_answer(standalone[-1]), "standalone", has_boxed_answer
    return None, "none", has_boxed_answer


def normalize_answer(text: str) -> str:
    return str(text or "").strip().upper().rstrip(".")


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
    if "test" not in candidates:
        candidates.append("test")
    return tuple(candidates)


def _extract_choices(row: dict[str, object]) -> list[str]:
    raw_choices = row.get("choices")
    if isinstance(raw_choices, list) and 1 <= len(raw_choices) <= len(CODEMMLU_CHOICE_LETTERS):
        return [str(choice) for choice in raw_choices]
    raise ValueError(f"Could not extract CodeMMLU choices from row: {row!r}")


def _format_question(
    *,
    question: str,
    choices: list[str],
    problem_description: str = "",
) -> str:
    parts: list[str] = []
    if problem_description:
        parts.append("Problem description:\n" + problem_description.strip())
    parts.append(question.strip())
    option_lines = [
        f"{letter}) {choice}"
        for letter, choice in zip(CODEMMLU_CHOICE_LETTERS, choices)
    ]
    parts.append("\n".join(option_lines))
    return "\n\n".join(part for part in parts if part)


def _normalize_response_for_simple_evals(response: str) -> str:
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


def _stable_example_sort_key(example_id: str) -> str:
    return hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()
