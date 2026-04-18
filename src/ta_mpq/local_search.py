from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ta_mpq.policy_export import export_candidate_from_group_bits
from ta_mpq.search import (
    SearchGroup,
    build_search_groups,
    estimate_budget_alignment_score,
    estimate_candidate_average_bit_width,
    estimate_candidate_weight_footprint_gb,
    estimate_group_value_alignment_score,
    layer_stats_from_report,
    resolve_group_value_scores,
)


@dataclass(frozen=True, slots=True)
class LocalSearchMove:
    group_name: str
    component_type: str
    from_bit: int
    to_bit: int
    direction: str
    evidence_score: float
    estimated_footprint_delta_gb: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalSearchCandidate:
    source_candidate_path: str
    proposal_score: float
    estimated_average_bit_width: float
    estimated_weight_footprint_gb: float
    budget_alignment_score: float
    group_value_alignment_score: float
    group_bits: tuple[tuple[str, int], ...]
    moves: tuple[LocalSearchMove, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["group_bits"] = dict(self.group_bits)
        payload["moves"] = [move.to_dict() for move in self.moves]
        return payload


@dataclass(frozen=True, slots=True)
class LocalSearchRoundResult:
    strategy: str
    grouping: str
    target_budget_gb: float
    allowed_bits: tuple[int, ...]
    beam_size: int
    max_candidates: int
    base_candidate_paths: tuple[str, ...]
    ablation_profile_paths: tuple[str, ...]
    group_value_prior_path: str | None
    candidates: tuple[LocalSearchCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "grouping": self.grouping,
            "target_budget_gb": self.target_budget_gb,
            "allowed_bits": list(self.allowed_bits),
            "beam_size": self.beam_size,
            "max_candidates": self.max_candidates,
            "base_candidate_paths": list(self.base_candidate_paths),
            "ablation_profile_paths": list(self.ablation_profile_paths),
            "group_value_prior_path": self.group_value_prior_path,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def build_no_surrogate_local_search_round(
    *,
    report_payload: dict[str, Any],
    base_candidate_payloads: list[dict[str, Any]],
    base_candidate_paths: list[str],
    target_budget_gb: float,
    output_dir: str | Path,
    allowed_bits: tuple[int, ...] = (4, 8, 16),
    beam_size: int = 3,
    max_candidates: int = 8,
    group_value_prior_payload: dict[str, Any] | None = None,
    group_value_prior_path: str | None = None,
    ablation_profile_payloads: list[dict[str, Any]] | None = None,
    ablation_profile_paths: list[str] | None = None,
) -> dict[str, Any]:
    if not base_candidate_payloads:
        raise ValueError("At least one base candidate payload is required")

    grouping = str(base_candidate_payloads[0]["grouping"])
    groups = build_search_groups(layer_stats_from_report(report_payload), grouping=grouping)
    allowed_bits = tuple(sorted({int(bit) for bit in allowed_bits}))
    group_lookup = {group.name: group for group in groups}
    group_value_scores = resolve_group_value_scores(groups, group_value_prior_payload)
    move_evidence = build_move_evidence(
        groups=groups,
        group_value_scores=group_value_scores,
        ablation_profile_payloads=ablation_profile_payloads or [],
    )

    candidate_pool: list[LocalSearchCandidate] = []
    seen_group_bits: set[tuple[tuple[str, int], ...]] = set()
    for source_path, payload in zip(base_candidate_paths, base_candidate_payloads, strict=True):
        base_group_bits = {
            str(group_name): int(bit_width)
            for group_name, bit_width in payload["group_bit_assignments"].items()
        }
        proposals = generate_local_search_candidates(
            groups=groups,
            base_group_bits=base_group_bits,
            source_candidate_path=source_path,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            group_value_scores=group_value_scores,
            move_evidence=move_evidence,
            max_candidates=max(beam_size * max_candidates, max_candidates),
        )
        for candidate in proposals:
            signature = candidate.group_bits
            if signature in seen_group_bits:
                continue
            seen_group_bits.add(signature)
            candidate_pool.append(candidate)

    ranked_candidates = sorted(
        candidate_pool,
        key=lambda candidate: (
            candidate.proposal_score,
            candidate.budget_alignment_score,
            candidate.group_value_alignment_score,
            -candidate.estimated_weight_footprint_gb,
        ),
        reverse=True,
    )[:max_candidates]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(ranked_candidates, start=1):
        exported = export_candidate_from_group_bits(
            report_payload=report_payload,
            grouping=grouping,
            group_bits=dict(candidate.group_bits),
            rank=index,
            metadata={
                "proposal_score": candidate.proposal_score,
                "budget_alignment_score": candidate.budget_alignment_score,
                "group_value_alignment_score": candidate.group_value_alignment_score,
                "estimated_average_bit_width": candidate.estimated_average_bit_width,
                "estimated_weight_footprint_gb": candidate.estimated_weight_footprint_gb,
                "source_candidate_path": candidate.source_candidate_path,
                "move_count": len(candidate.moves),
                "moves": [move.to_dict() for move in candidate.moves],
                "provenance": "no_surrogate_local_search",
            },
        )
        candidate_path = output_root / f"candidate-{index:02d}.json"
        _save_json(candidate_path, exported)
        manifest_candidates.append(
            {
                "rank": index,
                "path": str(candidate_path),
                "proposal_score": candidate.proposal_score,
                "estimated_average_bit_width": candidate.estimated_average_bit_width,
                "estimated_weight_footprint_gb": candidate.estimated_weight_footprint_gb,
                "budget_alignment_score": candidate.budget_alignment_score,
                "group_value_alignment_score": candidate.group_value_alignment_score,
                "bit_counts": _bit_counts(dict(candidate.group_bits).values()),
                "source_candidate_path": candidate.source_candidate_path,
                "moves": [move.to_dict() for move in candidate.moves],
            }
        )

    result = LocalSearchRoundResult(
        strategy="ablation_guided_beam_local_search",
        grouping=grouping,
        target_budget_gb=float(target_budget_gb),
        allowed_bits=allowed_bits,
        beam_size=beam_size,
        max_candidates=max_candidates,
        base_candidate_paths=tuple(base_candidate_paths),
        ablation_profile_paths=tuple(ablation_profile_paths or []),
        group_value_prior_path=group_value_prior_path,
        candidates=tuple(ranked_candidates),
    )
    manifest = result.to_dict()
    manifest["report_path"] = report_payload.get("source_report_path") or report_payload.get("report_path")
    manifest["num_exported_candidates"] = len(manifest_candidates)
    manifest["candidates"] = manifest_candidates
    _save_json(output_root / "manifest.json", manifest)
    return manifest


def generate_local_search_candidates(
    *,
    groups: list[SearchGroup],
    base_group_bits: dict[str, int],
    source_candidate_path: str,
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    group_value_scores: dict[str, Any] | None,
    move_evidence: dict[tuple[str, int, int], dict[str, float | str]],
    max_candidates: int,
) -> list[LocalSearchCandidate]:
    lower_bits = {bit: _next_lower_bit(bit, allowed_bits) for bit in allowed_bits}
    higher_bits = {bit: _next_higher_bit(bit, allowed_bits) for bit in allowed_bits}

    downgrade_moves: list[tuple[LocalSearchMove, float]] = []
    upgrade_moves: list[tuple[LocalSearchMove, float]] = []
    group_lookup = {group.name: group for group in groups}
    for group in groups:
        current_bit = int(base_group_bits[group.name])
        lower_bit = lower_bits[current_bit]
        if lower_bit is not None:
            evidence = _move_evidence_payload(move_evidence, group.name, current_bit, lower_bit)
            downgrade_moves.append(
                (
                    LocalSearchMove(
                        group_name=group.name,
                        component_type=group.component_type,
                        from_bit=current_bit,
                        to_bit=lower_bit,
                        direction="downgrade",
                        evidence_score=float(evidence["score"]),
                        estimated_footprint_delta_gb=estimate_bit_delta_footprint_gb(
                            group=group,
                            from_bit=current_bit,
                            to_bit=lower_bit,
                        ),
                        rationale=str(evidence["rationale"]),
                    ),
                    float(evidence["score"]),
                )
            )
        higher_bit = higher_bits[current_bit]
        if higher_bit is not None:
            evidence = _move_evidence_payload(move_evidence, group.name, current_bit, higher_bit)
            upgrade_moves.append(
                (
                    LocalSearchMove(
                        group_name=group.name,
                        component_type=group.component_type,
                        from_bit=current_bit,
                        to_bit=higher_bit,
                        direction="upgrade",
                        evidence_score=float(evidence["score"]),
                        estimated_footprint_delta_gb=estimate_bit_delta_footprint_gb(
                            group=group,
                            from_bit=current_bit,
                            to_bit=higher_bit,
                        ),
                        rationale=str(evidence["rationale"]),
                    ),
                    float(evidence["score"]),
                )
            )

    downgrade_moves.sort(key=lambda item: item[1], reverse=True)
    upgrade_moves.sort(key=lambda item: item[1], reverse=True)

    proposals: list[LocalSearchCandidate] = []
    seen_group_bits: set[tuple[tuple[str, int], ...]] = set()

    def maybe_add_candidate(moves: list[LocalSearchMove]) -> None:
        assignments = dict(base_group_bits)
        for move in moves:
            assignments[move.group_name] = move.to_bit
        signature = tuple(sorted(assignments.items()))
        if signature in seen_group_bits:
            return
        seen_group_bits.add(signature)
        footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
        average_bit_width = estimate_candidate_average_bit_width(groups, assignments)
        budget_alignment = estimate_budget_alignment_score(footprint, target_budget_gb)
        value_alignment = estimate_group_value_alignment_score(
            groups=groups,
            assignments=assignments,
            group_value_scores=group_value_scores,
        )
        proposal_score = (
            sum(move.evidence_score for move in moves)
            + 0.25 * budget_alignment
            + 0.20 * value_alignment
        )
        proposals.append(
            LocalSearchCandidate(
                source_candidate_path=source_candidate_path,
                proposal_score=proposal_score,
                estimated_average_bit_width=average_bit_width,
                estimated_weight_footprint_gb=footprint,
                budget_alignment_score=budget_alignment,
                group_value_alignment_score=value_alignment,
                group_bits=signature,
                moves=tuple(moves),
            )
        )

    for move, score in downgrade_moves[: max(2, max_candidates)]:
        if score <= 0.0:
            continue
        maybe_add_candidate([move])

    for move, score in upgrade_moves[: max(2, max_candidates)]:
        if score <= 0.0:
            continue
        if estimate_candidate_weight_footprint_gb(
            groups,
            {**base_group_bits, move.group_name: move.to_bit},
        ) > target_budget_gb * 1.01:
            continue
        maybe_add_candidate([move])

    downgrade_pool = [move for move, score in downgrade_moves[: max(3, max_candidates)] if score > 0.0]
    upgrade_pool = [move for move, score in upgrade_moves[: max(3, max_candidates)] if score > 0.0]
    for downgrade_move in downgrade_pool:
        for upgrade_move in upgrade_pool:
            if downgrade_move.group_name == upgrade_move.group_name:
                continue
            assignments = dict(base_group_bits)
            assignments[downgrade_move.group_name] = downgrade_move.to_bit
            assignments[upgrade_move.group_name] = upgrade_move.to_bit
            footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
            if footprint > target_budget_gb * 1.01:
                continue
            maybe_add_candidate([downgrade_move, upgrade_move])

    proposals.sort(
        key=lambda candidate: (
            candidate.proposal_score,
            candidate.budget_alignment_score,
            candidate.group_value_alignment_score,
            -candidate.estimated_weight_footprint_gb,
        ),
        reverse=True,
    )
    return proposals[:max_candidates]


def build_move_evidence(
    *,
    groups: list[SearchGroup],
    group_value_scores: dict[str, Any] | None,
    ablation_profile_payloads: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, float | str]]:
    evidence: dict[tuple[str, int, int], dict[str, float | str]] = {}
    for group in groups:
        value_profile = _group_value_profile(group_value_scores.get(group.name) if group_value_scores else None)
        uplift_8 = float(value_profile.get("uplift_8_over_4", 0.0))
        uplift_16 = float(value_profile.get("uplift_16_over_8", 0.0))
        preferred_bit = int(value_profile.get("preferred_bit", 4))
        evidence[(group.name, 8, 4)] = {
            "score": max(-uplift_8, 0.0) + (0.04 if preferred_bit <= 4 else 0.0),
            "rationale": "prior-encourages-4bit" if preferred_bit <= 4 else "prior-neutral-8to4",
        }
        evidence[(group.name, 4, 8)] = {
            "score": max(uplift_8, 0.0) + (0.04 if preferred_bit >= 8 else 0.0),
            "rationale": "prior-encourages-8bit" if preferred_bit >= 8 else "prior-neutral-4to8",
        }
        evidence[(group.name, 16, 8)] = {
            "score": max(-uplift_16, 0.0) + (0.04 if preferred_bit <= 8 else 0.0),
            "rationale": "prior-encourages-8bit" if preferred_bit <= 8 else "prior-neutral-16to8",
        }
        evidence[(group.name, 8, 16)] = {
            "score": max(uplift_16, 0.0) + (0.04 if preferred_bit >= 16 else 0.0),
            "rationale": "prior-encourages-16bit" if preferred_bit >= 16 else "prior-neutral-8to16",
        }

    for payload in ablation_profile_payloads:
        for item in payload.get("groups", []):
            group_name = str(item["name"])
            from_bit = int(item["reference_bit_width"])
            to_bit = int(item["ablated_bit_width"])
            positive_drop = float(item.get("positive_accuracy_drop", 0.0))
            improvement = float(item.get("improvement_if_downgraded", 0.0))
            combined_sensitivity = float(item.get("combined_sensitivity", 0.0))
            downgrade_score = 0.06 + improvement + 0.20 * combined_sensitivity - 1.5 * positive_drop
            upgrade_score = positive_drop + 0.25 * combined_sensitivity
            downgrade_reason = "ablation-safe"
            if improvement > 0:
                downgrade_reason = "ablation-improves-when-downgraded"
            elif positive_drop > 0:
                downgrade_reason = "ablation-penalizes-downgrade"
            evidence[(group_name, from_bit, to_bit)] = {
                "score": downgrade_score,
                "rationale": downgrade_reason,
            }
            evidence[(group_name, to_bit, from_bit)] = {
                "score": max(upgrade_score, 0.0),
                "rationale": "ablation-restore-precision" if positive_drop > 0 else "ablation-no-upgrade-evidence",
            }
    return evidence


def estimate_bit_delta_footprint_gb(
    *,
    group: SearchGroup,
    from_bit: int,
    to_bit: int,
) -> float:
    return (group.parameter_count * (to_bit - from_bit)) / 8 / (1024**3)


def select_best_candidate_from_evaluation_manifest(
    evaluation_manifest_payload: dict[str, Any],
) -> dict[str, Any] | None:
    best_record: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for candidate in evaluation_manifest_payload.get("candidates", []):
        accuracy = float(candidate.get("accuracy", 0.0))
        footprint = float(candidate.get("estimated_weight_footprint_gb", float("inf")))
        proposal_score = float(candidate.get("proposal_score", 0.0))
        key = (accuracy, -footprint, proposal_score)
        if best_key is None or key > best_key:
            best_key = key
            best_record = dict(candidate)
    return best_record


def _move_evidence_payload(
    evidence: dict[tuple[str, int, int], dict[str, float | str]],
    group_name: str,
    from_bit: int,
    to_bit: int,
) -> dict[str, float | str]:
    return evidence.get(
        (group_name, from_bit, to_bit),
        {
            "score": 0.0,
            "rationale": "no-evidence",
        },
    )


def _next_lower_bit(bit_width: int, allowed_bits: tuple[int, ...]) -> int | None:
    allowed = sorted(set(int(bit) for bit in allowed_bits))
    try:
        index = allowed.index(int(bit_width))
    except ValueError:
        return None
    if index == 0:
        return None
    return allowed[index - 1]


def _next_higher_bit(bit_width: int, allowed_bits: tuple[int, ...]) -> int | None:
    allowed = sorted(set(int(bit) for bit in allowed_bits))
    try:
        index = allowed.index(int(bit_width))
    except ValueError:
        return None
    if index >= len(allowed) - 1:
        return None
    return allowed[index + 1]


def _group_value_profile(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {
            "score": float(value.get("score", 0.0)),
            "uplift_8_over_4": float(value.get("uplift_8_over_4", value.get("score", 0.0))),
            "uplift_16_over_8": float(value.get("uplift_16_over_8", 0.0)),
            "preferred_bit": float(value.get("preferred_bit", 4)),
            "confidence": float(value.get("confidence", 0.0)),
        }
    if value is None:
        return {
            "score": 0.0,
            "uplift_8_over_4": 0.0,
            "uplift_16_over_8": 0.0,
            "preferred_bit": 4.0,
            "confidence": 0.0,
        }
    score = float(value)
    return {
        "score": score,
        "uplift_8_over_4": score,
        "uplift_16_over_8": 0.0,
        "preferred_bit": 8.0 if score > 0 else 4.0,
        "confidence": abs(score),
    }


def _bit_counts(bit_widths: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bit_width in bit_widths:
        key = str(int(bit_width))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
