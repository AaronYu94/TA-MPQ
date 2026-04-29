from __future__ import annotations

from typing import Any


def load_task_adapter(task_name: str) -> Any:
    normalized = task_name.lower()
    if normalized == "gsm8k":
        from ta_mpq.tasks import gsm8k as task

        return task
    if normalized in {"math500", "math-500"}:
        from ta_mpq.tasks import math500 as task

        return task
    if normalized in {"mmlu_coding", "mmlu-coding", "mmlucoding"}:
        from ta_mpq.tasks import mmlu_coding as task

        return task
    if normalized in {"codemmlu", "code_mmlu", "code-mmlu"}:
        from ta_mpq.tasks import codemmlu as task

        return task
    raise ValueError(f"Unsupported task_name: {task_name}")
