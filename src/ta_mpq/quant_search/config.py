from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Loading YAML configs requires PyYAML. Install pyyaml or use a JSON config."
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {config_path} must decode to a mapping")
    config = deepcopy(payload)
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(Path.cwd().resolve())
    return config


def artifact_root(config: dict[str, Any]) -> Path:
    root = config.get("artifacts_root") or DEFAULT_ARTIFACT_ROOT
    return Path(root)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(
    config: dict[str, Any],
    *parts: str,
    create_parent: bool = False,
) -> Path:
    path = artifact_root(config).joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_float_list(value: str | list[Any] | tuple[Any, ...]) -> list[float]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        return [float(token) for token in tokens]
    return [float(item) for item in value]


def parse_int_list(value: str | list[Any] | tuple[Any, ...]) -> list[int]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        return [int(token) for token in tokens]
    return [int(item) for item in value]
