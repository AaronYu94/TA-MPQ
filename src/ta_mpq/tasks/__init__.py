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
    raise ValueError(f"Unsupported task_name: {task_name}")
