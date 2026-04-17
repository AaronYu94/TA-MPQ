from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import re
from typing import Any


PROJECT_SUPPORTED_BITS = (2, 3, 4, 8, 16)
LLM_COMPRESSOR_SUPPORTED_BITS = (4, 8)
HIGH_PRECISION_BIT = 16


@dataclass(frozen=True, slots=True)
class BitRule:
    name: str
    targets: tuple[str, ...]
    bit_width: int
    group_size: int = 128
    symmetric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BitRule":
        return cls(
            name=str(payload["name"]),
            targets=tuple(str(target) for target in payload["targets"]),
            bit_width=int(payload["bit_width"]),
            group_size=int(payload.get("group_size", 128)),
            symmetric=bool(payload.get("symmetric", True)),
        )


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    default_bit_width: int = 4
    default_targets: tuple[str, ...] = ("Linear",)
    ignore: tuple[str, ...] = ("lm_head",)
    rules: tuple[BitRule, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_bit_width": self.default_bit_width,
            "default_targets": list(self.default_targets),
            "ignore": list(self.ignore),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MixedPrecisionPolicy":
        return cls(
            default_bit_width=int(payload["default_bit_width"]),
            default_targets=tuple(str(target) for target in payload.get("default_targets", ("Linear",))),
            ignore=tuple(str(target) for target in payload.get("ignore", ("lm_head",))),
            rules=tuple(BitRule.from_dict(rule) for rule in payload.get("rules", ())),
        )


def validate_policy(policy: MixedPrecisionPolicy) -> None:
    all_bits = {policy.default_bit_width, *(rule.bit_width for rule in policy.rules)}
    unsupported = sorted(bit for bit in all_bits if bit not in PROJECT_SUPPORTED_BITS)
    if unsupported:
        raise ValueError(f"Unsupported project bit-widths: {unsupported}")


def validate_backend_support(policy: MixedPrecisionPolicy, backend: str = "llmcompressor") -> None:
    if backend != "llmcompressor":
        raise ValueError(f"Unknown backend: {backend}")

    all_bits = {policy.default_bit_width, *(rule.bit_width for rule in policy.rules)}
    unsupported = sorted(bit for bit in all_bits if bit not in LLM_COMPRESSOR_SUPPORTED_BITS)
    if unsupported:
        raise ValueError(
            "Backend llmcompressor currently only supports a runtime bridge for bit-widths "
            f"{LLM_COMPRESSOR_SUPPORTED_BITS}; policy requested {unsupported}"
        )


def assign_bits_to_modules(
    module_names: list[str],
    policy: MixedPrecisionPolicy,
) -> dict[str, int]:
    validate_policy(policy)
    assignments: dict[str, int] = {}
    for module_name in module_names:
        if any(_matches_target(module_name, pattern) for pattern in policy.ignore):
            assignments[module_name] = HIGH_PRECISION_BIT
            continue
        assigned = policy.default_bit_width
        for rule in policy.rules:
            if any(_matches_target(module_name, pattern) for pattern in rule.targets):
                assigned = rule.bit_width
                break
        assignments[module_name] = assigned
    return assignments


def estimate_average_bit_width(
    param_counts: dict[str, int],
    assignments: dict[str, int],
) -> float:
    weighted_bits = 0
    total_params = 0
    for module_name, param_count in param_counts.items():
        weighted_bits += param_count * assignments[module_name]
        total_params += param_count
    if total_params == 0:
        return 0.0
    return weighted_bits / total_params


def estimate_weight_footprint_gb(
    param_counts: dict[str, int],
    assignments: dict[str, int],
) -> float:
    total_bits = 0
    for module_name, param_count in param_counts.items():
        total_bits += param_count * assignments[module_name]
    return total_bits / 8 / (1024**3)


def to_llmcompressor_recipe_config(
    policy: MixedPrecisionPolicy,
) -> dict[str, Any]:
    validate_backend_support(policy, backend="llmcompressor")

    config_groups: dict[str, Any] = {
        "default": {
            "targets": list(policy.default_targets),
            "weights": _weight_args(policy.default_bit_width, group_size=128, symmetric=True),
            "format": _format_name(policy.default_bit_width),
        }
    }

    for rule in policy.rules:
        config_groups[rule.name] = {
            "targets": list(rule.targets),
            "weights": _weight_args(rule.bit_width, rule.group_size, rule.symmetric),
            "format": _format_name(rule.bit_width),
        }

    return {
        "ignore": list(policy.ignore),
        "config_groups": config_groups,
    }


def default_feasibility_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        default_bit_width=4,
        rules=(
            BitRule(
                name="down_proj_int8",
                targets=("re:.*down_proj$",),
                bit_width=8,
            ),
        ),
    )


def _weight_args(bit_width: int, group_size: int, symmetric: bool) -> dict[str, Any]:
    return {
        "num_bits": bit_width,
        "type": "int",
        "strategy": "group",
        "group_size": group_size,
        "symmetric": symmetric,
        "dynamic": False,
    }


def _format_name(bit_width: int) -> str:
    # In current llm-compressor releases, weight-only W4A16/W8A16 schemes map to
    # the pack-quantized compressor format.
    if bit_width in (4, 8):
        return "pack-quantized"
    return "int-quantized"


def _matches_target(module_name: str, pattern: str) -> bool:
    if pattern == "Linear":
        return True
    if pattern.startswith("re:"):
        return re.search(pattern[3:], module_name) is not None
    return fnmatch.fnmatch(module_name, pattern)
