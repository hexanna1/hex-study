"""Local coordinate orbits, ablation calibration, and representative pooling."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import hex_board as hb
from hex_symmetry import apply_transform_ax, inverse_transform_id


PASS_PROXY_CANONICAL_KEY = "pass_proxy"
TENUKI_CANONICAL_KEY = "tenuki"

@dataclass(slots=True)
class _AblationPair:
    row: dict[str, Any]
    signal: float
    delta_without: float

    @property
    def is_pass(self) -> bool:
        return self.row["candidate_key_local"] == PASS_PROXY_CANONICAL_KEY

def stone_fraction_for_importance(*, stone_fraction: float, child_ply: int) -> float:
    return stone_fraction if child_ply <= 1 else stone_fraction ** (child_ply / (child_ply + 1.0))


def base_relative_point(point: hb.Point, exp_meta: dict[str, Any]) -> hb.Point:
    shift_q, shift_r = exp_meta["orientation_norm_shift"]
    offset_q, offset_r = exp_meta["board_offset"]
    unnormalized = (point[0] - offset_q + shift_q, point[1] - offset_r + shift_r)
    return apply_transform_ax(unnormalized, inverse_transform_id(exp_meta["orientation_transform_id"]))


def canonicalize_base_rel_under_orbit(point: hb.Point, exp_meta: dict[str, Any]) -> hb.Point:
    equivalents = [point]
    for entry in exp_meta.get("local_key_orbit", []):
        q, r = apply_transform_ax(point, entry["transform_id"])
        sq, sr = entry["norm_shift"]
        equivalents.append((q - sq, r - sr))
    return min(equivalents)


def candidate_key(move: str, exp_meta: dict[str, Any] | None) -> str | None:
    if not exp_meta:
        return None
    try:
        point = hb.parse_cell(move)
    except ValueError:
        return None
    q, r = canonicalize_base_rel_under_orbit(base_relative_point(point, exp_meta), exp_meta)
    return f"{q},{r}"


def logit_clamped(winrate: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, float(winrate)))
    return math.log(p / (1.0 - p))


def attach_stone_fractions(
    rows: list[dict[str, Any]],
    *,
    position_input: str,
    allow_first_row_proxy_fallback: bool = False,
) -> None:
    for row in rows:
        row["stone_fraction"] = None

    valid = [r for r in rows if isinstance(r.get("mean_winrate"), (int, float))]
    if not valid:
        return

    board_size, _red, _blue, to_play = hb.position_state(position_input)
    canonical_proxy = hb.canonical_pass_proxy_move(board_size, to_play)

    proxy_row: dict[str, Any] | None = None
    if canonical_proxy is not None:
        for row in valid:
            if str(row.get("move") or "").lower() == canonical_proxy:
                proxy_row = row
                break
    if proxy_row is None and allow_first_row_proxy_fallback:
        proxy_row = min(valid, key=lambda r: int(r.get("_idx", 0)))
    if proxy_row is None:
        return

    best_row = max(valid, key=lambda r: float(r["mean_winrate"]))
    l_proxy = logit_clamped(float(proxy_row["mean_winrate"]))
    l_best = logit_clamped(float(best_row["mean_winrate"]))
    denom = l_best - l_proxy
    if abs(denom) < 1e-12:
        return

    for row in valid:
        l_row = logit_clamped(float(row["mean_winrate"]))
        row["stone_fraction"] = (l_row - l_proxy) / denom


def pool_candidates(
    summary_rows: list[dict[str, Any]], *, total_representatives: int, value_field: str = "corrected_value"
) -> list[dict[str, Any]]:
    corrected = value_field == "corrected_value"
    mean_key, stdev_key, min_key, max_key = (f"{stat}_{value_field}" for stat in ("mean", "stdev", "min", "max"))

    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = row.get("candidate_key_local")
        value = row.get(value_field)
        exp_name = str(row.get("experiment") or "")
        # Intentional: pool all retained row-level samples for a local key.
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(value, (int, float)):
            continue
        by_key.setdefault(key, []).append(
            {
                "experiment": exp_name,
                "candidate_abs": str(row.get("candidate_abs") or ""),
                value_field: float(value),
            }
        )

    out: list[dict[str, Any]] = []
    for key, rows in by_key.items():
        values = [float(r[value_field]) for r in rows]
        exp_set = {r["experiment"] for r in rows if r["experiment"]}
        sample_abs = next((r["candidate_abs"] for r in rows if r["candidate_abs"]), "")
        n = len(exp_set) if exp_set else len(rows)
        coverage = (n / total_representatives) if total_representatives > 0 else 0.0
        out.append(
            {
                "candidate_key_local": key,
                "sample_candidate_abs": sample_abs,
                "n": n,
                "coverage": coverage,
                mean_key: mean(values),
                stdev_key: pstdev(values),
                min_key: min(values),
                max_key: max(values),
            }
        )

    if corrected:
        denom = max((row[mean_key] for row in out), default=0.0)
        for row in out:
            for stat in ("mean", "stdev", "min", "max"):
                row[f"{stat}_stone_fraction"] = row[f"{stat}_corrected_value"] / denom if denom > 1e-12 else 0.0

    out.sort(
        key=lambda r: (
            -r["mean_stone_fraction"], -r.get("mean_corrected_value", float("-inf")),
            -r["n"], r["candidate_key_local"],
        )
    )
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def write_pooled_candidates_json(path: Path, pooled_rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(pooled_rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def apply_ablation_calibration(
    *,
    summary_rows_with: list[dict[str, Any]],
    summary_rows_without: list[dict[str, Any]],
    root_ablation_by_experiment: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    without_by_key = defaultdict(list)
    for row in summary_rows_without:
        without_by_key[row["experiment"], row["candidate_key_local"]].append(row)
    by_experiment = defaultdict(list)
    for row in summary_rows_with:
        by_experiment[row["experiment"]].append(row)

    out, diagnostics = [], []
    for experiment, rows in by_experiment.items():
        roots = root_ablation_by_experiment[experiment]
        root_with = logit_clamped(roots["with_root_winrate"])
        root_without = logit_clamped(roots["without_root_winrate"])
        root_effect = root_with - root_without
        numeric_rows = [row for row in rows if row["mean_winrate"] is not None]
        by_key = defaultdict(list)
        for row in numeric_rows:
            by_key[row["candidate_key_local"]].append(row)

        paired = []
        for key, with_rows in sorted(by_key.items()):
            without_rows = without_by_key[experiment, key]
            with_wrs = [row["mean_winrate"] for row in with_rows]
            without_wrs = [row["mean_winrate"] for row in without_rows if row["mean_winrate"] is not None]
            if not without_wrs:
                raise ValueError(f"Ablation calibration failed for {experiment}: missing paired without-pattern eval for local key {key}")
            # Average symmetry-equivalent samples in logit space before scoring.
            delta_with = mean(map(logit_clamped, with_wrs)) - root_with
            delta_without = mean(map(logit_clamped, without_wrs)) - root_without
            interaction = delta_with - delta_without
            pre_values = [row["stone_fraction"] for row in with_rows if row.get("stone_fraction") is not None]
            display = min(with_rows, key=lambda row: row["move"])
            row_out = {
                "experiment": experiment,
                "label": display.get("label", ""),
                "mode": display.get("mode", ""),
                "move": display["move"],
                "candidate_abs": display.get("candidate_abs") or display["move"],
                "candidate_key_local": key,
                "stone_fraction_pre_ablation": mean(pre_values) if pre_values else None,
                "corrected_value": None,
                "n": sum(row["n"] for row in with_rows),
                "mean_winrate": mean(with_wrs),
                "stdev_winrate": pstdev(with_wrs),
                "min_winrate": min(with_wrs),
                "max_winrate": max(with_wrs),
                "mean_winrate_without_pattern": mean(without_wrs),
                "ablation_root_pattern_effect_logit": root_effect,
                "ablation_interaction_lift_logit": interaction,
            }
            out.append(row_out)
            paired.append(_AblationPair(row_out, interaction, delta_without))

        pass_pair = next((pair for pair in paired if pair.is_pass), None)
        if pass_pair is None:
            raise ValueError(f"Ablation calibration failed for {experiment}: missing paired pass_proxy anchor")
        generic_value = max(pair.delta_without - pass_pair.delta_without for pair in paired)
        for pair in paired:
            pair.row["corrected_value"] = (
                generic_value * (0.0 if pair.is_pass else 1.0) + (pair.signal - pass_pair.signal)
            )
        best = max(paired, key=lambda pair: pair.row["corrected_value"])
        pre_values = [pair.row["stone_fraction_pre_ablation"] for pair in paired if pair.row["stone_fraction_pre_ablation"] is not None]
        diagnostics.append({
            "experiment": experiment,
            "numeric_row_count": len(numeric_rows),
            "paired_row_count": len(paired),
            "root_pattern_effect_logit": root_effect,
            "generic_move_value_logit_hat": generic_value,
            "anchor_pass_move": pass_pair.row["move"] or None,
            "anchor_best_local_move": best.row["move"] or None,
            "max_pre_fraction": max(pre_values, default=None),
            "max_corrected_value": best.row["corrected_value"],
        })
    return out, diagnostics
