#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hex_symmetry import hex_distance_squared
import pattern_output_utils as sout
import hex_board as hb
import pattern_scoring as scoring
import raw_nn as nn
import study_runtime as runtime
from hex_board import serialize_position
from pattern_evaluation import PatternEvaluator
from pattern_scoring import PASS_PROXY_CANONICAL_KEY, TENUKI_CANONICAL_KEY
from local_pattern_representative import (
    build_study_spec,
    extract_pattern,
    generate_representatives,
    parse_balance_profiles,
)

CANONICAL_BALANCE_RED_MOVE = "a1"
CANONICAL_BALANCE_BLUE_MOVE = "d2"
RUN_SPEC_CANDIDATE_DISTANCE_KEY = "candidate_Δ_max"
RUN_SPEC_TENUKI_MIN_DISTANCE_KEY = "tenuki_Δ_min"


def _artifact_slug_for_out_dir(out_dir: Path) -> str:
    name = out_dir.name
    parts = name.split("-", 2)
    if len(parts) == 3 and len(parts[0]) == 8 and len(parts[1]) == 6 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return name


def _default_debug_dir_name(
    *,
    stamp: str,
    board_size: int,
    candidate_Δ_max: int,
    movelist_slug: str,
) -> str:
    # Keep auto-generated artifact paths ASCII-only so local file links remain stable.
    return f"{stamp}-s{int(board_size)}-d{int(candidate_Δ_max)}-{movelist_slug}"


def _default_json_artifact_path(*, run_slug: str) -> Path:
    return Path("artifacts") / f"{run_slug}.json"


def _resolve_nondebug_output_path(*, out_dir_arg: str | None, run_slug: str) -> Path:
    if not out_dir_arg:
        return _default_json_artifact_path(run_slug=run_slug)
    out_path = Path(out_dir_arg)
    if out_path.suffix.lower() == ".json":
        return out_path
    return out_path / f"{run_slug}.json"


def _mirror_pooled_map_pngs(*, out_dir: Path, scoring_outputs: dict[str, Any]) -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_slug_for_out_dir(out_dir)
    for key, suffix in (("default", ""), ("pre_ablation", "-pre-ablation")):
        output = scoring_outputs.get(key)
        if not isinstance(output, dict):
            continue
        pooled_map_artifacts = output.get("pooled_map_artifacts")
        if not isinstance(pooled_map_artifacts, dict):
            continue
        png_name = pooled_map_artifacts.get("png")
        if not isinstance(png_name, str) or not png_name.strip():
            continue
        src = out_dir / png_name
        if not src.exists():
            continue
        shutil.copy2(src, artifacts_dir / f"{stem}{suffix}.png")


def _select_root_tenuki_move(
    payload: dict[str, Any],
    base_local_candidates: set[str],
    *,
    pattern_cells_abs: set[tuple[int, int]],
    tenuki_Δ_min: int,
) -> str | None:
    for move, _prior in nn.policy_moves(payload):
        move = move.strip().lower()
        if move in base_local_candidates:
            continue
        try:
            point = hb.parse_cell(move)
        except ValueError:
            continue
        if all(hex_distance_squared(point, stone) >= tenuki_Δ_min for stone in pattern_cells_abs):
            return move
    return None


def _canonical_key_for_move(
    *,
    move: str,
    exp_meta: dict[str, Any] | None,
    pass_proxy_move: str | None,
    tenuki_move: str | None,
) -> str | None:
    mv = str(move).strip().lower()
    if not mv:
        return None
    if pass_proxy_move and mv == pass_proxy_move:
        return PASS_PROXY_CANONICAL_KEY
    if tenuki_move and mv == tenuki_move:
        return TENUKI_CANONICAL_KEY
    return scoring.candidate_key(mv, exp_meta)


def _rewrite_row_candidate_keys(
    *,
    rows: list[dict[str, Any]],
    exp_meta: dict[str, Any] | None,
    pass_proxy_move: str | None,
    tenuki_move: str | None,
) -> None:
    for row in rows:
        move = row["move"]
        row["candidate_abs"] = move
        row["candidate_key_local"] = _canonical_key_for_move(
            move=move,
            exp_meta=exp_meta,
            pass_proxy_move=pass_proxy_move,
            tenuki_move=tenuki_move,
        )


def _sanitize_persisted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    obj = json.loads(json.dumps(payload, ensure_ascii=True))
    if isinstance(obj, dict):
        meta = obj.get("meta")
        if isinstance(meta, dict):
            meta.pop("elapsed_ms", None)
    return obj if isinstance(obj, dict) else {}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-off local pattern study runner with a1,d2 balance only")
    ap.add_argument("--hexworld", required=True, help="Source HexWorld URL/hash to infer local pattern")
    ap.add_argument(
        "--delta",
        dest="candidate_Δ_max",
        type=int,
        default=7,
        help="Max candidate Δ-distance threshold (default: 7)",
    )
    ap.add_argument(
        "--board-size",
        type=int,
        default=None,
        help="Optional target board size override (default: inferred from --hexworld)",
    )
    ap.add_argument(
        "--symmetry",
        choices=["identity", "rotations", "d6"],
        default=None,
        help="Representative symmetry policy (default: d6)",
    )
    ap.add_argument(
        "--tenuki-delta-min",
        dest="tenuki_Δ_min",
        type=int,
        default=21,
        help="Minimum squared Δ-distance from all pattern stones for tenuki eligibility (default: 21)",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Write the full debug directory, raw payloads, manifests, and pre-ablation outputs",
    )
    return ap.parse_args()


def _append_summary_rows_from_agg(
    *,
    out_rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    agg: list[dict[str, Any]],
) -> None:
    fields = (
        "move", "candidate_abs", "candidate_key_local", "stone_fraction", "n",
        "mean_winrate", "stdev_winrate", "min_winrate", "max_winrate",
    )
    out_rows.extend({**base_row, **{key: row[key] for key in fields}} for row in agg)


def _raw_nn_cache_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "raw_nn_cache.json"


def _write_raw_payload(
    *,
    exp_dir: Path | None,
    filename: str,
    experiment_name: str,
    command: list[str],
    ok: bool,
    err: str,
    payload: dict[str, Any],
) -> None:
    if isinstance(exp_dir, Path):
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / str(filename)).write_text(
            json.dumps(
                runtime.redact_personal_obj(
                    {
                        "experiment": experiment_name,
                        "ok": ok,
                        "error": err,
                        "command": command,
                        "payload": _sanitize_persisted_payload(payload),
                    }
                ),
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _baseline_position(
    evaluator: PatternEvaluator, experiments: list[dict[str, Any]], raw_dir: Path | None,
) -> tuple[str, set[str]]:
    red = hb.parse_cell(CANONICAL_BALANCE_RED_MOVE)
    blue = hb.parse_cell(CANONICAL_BALANCE_BLUE_MOVE)
    position = serialize_position(
        board_size=evaluator.board_size, red_cells=(red,), blue_cells=(blue,), to_play="red",
    )
    if evaluator.to_play == "red":
        return position, set()
    payloads, _hits, elapsed = evaluator.roots(
        position_inputs=[position], persistent_keys={hb.position_cache_key(position)},
    )
    payload = payloads[position]
    _write_raw_payload(
        exp_dir=raw_dir, filename="canonical_blue_probe.json", experiment_name="canonical_blue_probe",
        command=nn.batch_command(board_size=evaluator.board_size, move_limit=evaluator.board_size**2 + 1),
        ok=True, err="",
        payload={"source": "raw_nn_cache_batchrawnn", "hexworld": position, "payload": payload},
    )
    forbidden = {CANONICAL_BALANCE_RED_MOVE, CANONICAL_BALANCE_BLUE_MOVE}
    forbidden.update(exp["candidates"][0] for exp in experiments)
    eligible = []
    for rank, (move, prior) in enumerate(nn.policy_moves(payload)):
        if move in forbidden:
            continue
        try:
            point = hb.parse_cell(move)
        except ValueError:
            continue
        eligible.append((-prior, rank, move, point))
    if not eligible:
        raise ValueError("Canonical move3 probe failed: no eligible place-cell move")
    _prior, _rank, move, point = min(eligible)
    runtime.log(f"[canonical blue baseline] move3={move} probe_elapsed={runtime.format_duration(elapsed)}")
    return serialize_position(
        board_size=evaluator.board_size, red_cells=tuple(sorted({red, point})),
        blue_cells=(blue,), to_play="blue",
    ), {CANONICAL_BALANCE_RED_MOVE, CANONICAL_BALANCE_BLUE_MOVE, move}


def _prepare_experiments(
    evaluator: PatternEvaluator, experiments: list[dict[str, Any]], representatives: list,
    without_position: str, baseline_occupied: set[str], tenuki_Δ_min: int,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    positions = {}
    for exp in experiments:
        positions.setdefault(hb.position_cache_key(exp["position"]), exp["position"])
        positions.setdefault(hb.position_cache_key(without_position), without_position)
    payloads, hits, _elapsed = evaluator.roots(
        position_inputs=list(positions.values()), persistent_keys={hb.position_cache_key(without_position)},
    )
    prepared = []
    for index, (exp, rep) in enumerate(zip(experiments, representatives), start=1):
        payload = payloads[exp["position"]]
        base_candidates = exp["candidates"]
        pass_move = base_candidates[0]
        tenuki = _select_root_tenuki_move(
            payload, set(base_candidates), pattern_cells_abs=set(rep.plus_abs) | set(rep.minus_abs),
            tenuki_Δ_min=tenuki_Δ_min,
        )
        candidates = [move for move in base_candidates + ([tenuki] if tenuki else []) if move not in baseline_occupied]
        if pass_move not in candidates:
            raise ValueError(f"Pass-proxy candidate was removed by occupancy filtering for {exp['name']}: {pass_move}")
        with_wr, without_wr = payload["r"], payloads[without_position]["r"]
        prepared.append({
            "index": index, "name": exp["name"], "label": exp["label"],
            "with_position": exp["position"], "without_position": without_position,
            "base_candidates": base_candidates, "candidates": candidates,
            "root_ablation_with_winrate": with_wr if evaluator.to_play == "red" else 1.0 - with_wr,
            "root_ablation_without_winrate": without_wr if evaluator.to_play == "red" else 1.0 - without_wr,
            "tenuki_move": tenuki if tenuki in candidates else None, "pass_proxy_move": pass_move,
        })
    runtime.log(f"[probe] cache={hits}/{len(positions)} elapsed={runtime.format_duration(time.monotonic() - started)}")
    return prepared


def main() -> int:
    args = _parse_args()
    debug_mode = args.debug
    if args.candidate_Δ_max < 0:
        raise ValueError("--delta must be >= 0")
    if args.tenuki_Δ_min < 0:
        raise ValueError("--tenuki-delta-min must be >= 0")
    tenuki_Δ_min = int(args.tenuki_Δ_min)

    extracted = extract_pattern(args.hexworld)
    board_size = int(args.board_size) if args.board_size is not None else int(extracted.board_size_source)
    balance_moves = ["a1,d2"]
    balances = parse_balance_profiles(balance_moves)
    symmetry_policy = str(args.symmetry or "d6")
    representatives = generate_representatives(
        extracted=extracted,
        board_size=board_size,
        symmetry=symmetry_policy,
        balance_profiles=balances,
    )
    base_spec = build_study_spec(
        extracted=extracted,
        representatives=representatives,
        board_size=board_size,
        symmetry=symmetry_policy,
        candidate_mode="auto-near-pattern",
        explicit_candidates=[],
        candidate_Δ_max=args.candidate_Δ_max,
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    movelist_slug = sout.movelist_slug_from_hexworld(args.hexworld)
    run_slug = _default_debug_dir_name(
        stamp=stamp,
        board_size=int(board_size),
        candidate_Δ_max=int(args.candidate_Δ_max),
        movelist_slug=movelist_slug,
    )
    out_dir = (Path(args.out_dir) if args.out_dir else Path("debug") / run_slug) if debug_mode else None
    raw_dir = (out_dir / "raw") if isinstance(out_dir, Path) else None
    final_json_path = _resolve_nondebug_output_path(out_dir_arg=args.out_dir, run_slug=run_slug)
    final_output_hint = out_dir if isinstance(out_dir, Path) else final_json_path
    if isinstance(raw_dir, Path):
        raw_dir.mkdir(parents=True, exist_ok=True)
    evaluator = PatternEvaluator(board_size, extracted.to_play_at_cursor, _raw_nn_cache_path())

    def fail() -> int:
        print(runtime.redact_personal_text(str(final_output_hint)))
        return 1

    try:
        generator_meta = base_spec["generator_meta"]
        exp_meta_map = generator_meta["experiment_meta"]
        base_exps = base_spec["experiments"]
        study_start = time.monotonic()
        shared_without_position, baseline_occupied = _baseline_position(evaluator, base_exps, raw_dir)
        prepared = _prepare_experiments(
            evaluator, base_exps, representatives, shared_without_position,
            baseline_occupied, tenuki_Δ_min,
        )

        if isinstance(out_dir, Path):
            run_spec = {
                "source_hexworld": args.hexworld,
                "board_size": board_size,
                "symmetry": symmetry_policy,
                "balance": list(balance_moves),
                RUN_SPEC_CANDIDATE_DISTANCE_KEY: int(args.candidate_Δ_max),
                RUN_SPEC_TENUKI_MIN_DISTANCE_KEY: tenuki_Δ_min,
                "generator_meta": generator_meta,
                "representatives": [
                    {
                        "index": int(p["index"]),
                        "name": str(p["name"]),
                        "label": str(p["label"]),
                        "with_position": str(p["with_position"]),
                        "without_position": str(p["without_position"]),
                        "base_candidates": list(p["base_candidates"]),
                        "candidates": list(p["candidates"]),
                        "pass_proxy_move": p.get("pass_proxy_move"),
                        "tenuki_move": p.get("tenuki_move"),
                        "root_with_winrate": p.get("root_ablation_with_winrate"),
                        "root_without_winrate": p.get("root_ablation_without_winrate"),
                    }
                    for p in prepared
                ],
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_spec.json").write_text(
                json.dumps(run_spec, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )

        summary_rows_with: list[dict[str, Any]] = []
        summary_rows_without: list[dict[str, Any]] = []
        root_diag_rows: list[dict[str, Any]] = []
        manifest = (
            {
                "raw_nn_backend": "batchrawnn",
                "source_hexworld": args.hexworld,
                "experiments": [],
            }
            if isinstance(out_dir, Path)
            else None
        )

        batch_raw_nn_cmd = nn.batch_command(board_size=board_size, move_limit=1)
        candidate_request_specs = [
            (p[f"{side}_position"], p["candidates"]) for p in prepared for side in ("with", "without")
        ]
        candidate_batch_t0 = time.monotonic()
        total_cache_miss_moves = evaluator.fill_candidates(
            candidate_request_specs,
            persistent_parent_positions={hb.position_cache_key(shared_without_position)},
        )
        for p in prepared:
            name = str(p["name"])
            label = str(p["label"])
            exp_dir = (raw_dir / f"{int(p['index']):02d}-{runtime.safe_name(name)}") if isinstance(raw_dir, Path) else None
            if isinstance(exp_dir, Path):
                exp_dir.mkdir(parents=True, exist_ok=True)

            for side, summary in (("with", summary_rows_with), ("without", summary_rows_without)):
                rows, payload = evaluator.candidate_rows(
                    position_input=p[f"{side}_position"], moves=p["candidates"],
                )
                _write_raw_payload(
                    exp_dir=exp_dir, filename=f"candidate_{side}.json", experiment_name=name,
                    command=batch_raw_nn_cmd, ok=True, err="", payload=payload,
                )
                _rewrite_row_candidate_keys(
                    rows=rows, exp_meta=exp_meta_map[name],
                    pass_proxy_move=p["pass_proxy_move"], tenuki_move=p["tenuki_move"],
                )
                _append_summary_rows_from_agg(
                    out_rows=summary, base_row={"experiment": name, "label": label, "mode": "candidate"}, agg=rows,
                )

            root_diag_rows.append(
                {
                    "experiment": name,
                    "tenuki_move": p.get("tenuki_move"),
                    "base_candidate_count": len(p["base_candidates"]),
                    "candidate_count": len(p["candidates"]),
                    "ablation_root_with_winrate": p.get("root_ablation_with_winrate"),
                    "ablation_root_without_winrate": p.get("root_ablation_without_winrate"),
                }
            )

            if isinstance(manifest, dict) and isinstance(exp_dir, Path) and isinstance(out_dir, Path):
                manifest["experiments"].append(
                    {
                        "name": name,
                        "label": label,
                        "raw_dir": str(exp_dir.relative_to(out_dir)),
                    }
                )

        total_candidate_cache_hits = sum(len(moves) for _position, moves in candidate_request_specs) - total_cache_miss_moves
        total_candidate_requests = sum(len(moves) for _position, moves in candidate_request_specs)
        runtime.log(
            f"[candidate]"
            f" reps={len(prepared)}"
            f" cache={total_candidate_cache_hits}/{total_candidate_requests}"
            f" elapsed={runtime.format_duration(time.monotonic() - candidate_batch_t0)}"
            f" total_elapsed={runtime.format_duration(time.monotonic() - study_start)}"
        )

        first_exp_meta = exp_meta_map[prepared[0]["name"]]
        artifact_common = {
            "out_dir": out_dir,
            "first_rep": representatives[0],
            "first_exp_meta": first_exp_meta,
            "total_representatives": len(prepared),
        }

        root_ablation_by_experiment = {
            str(p["name"]): {
                "with_root_winrate": p.get("root_ablation_with_winrate"),
                "without_root_winrate": p.get("root_ablation_without_winrate"),
            }
            for p in prepared
        }
        post_summary_rows, ablation_diag_rows = scoring.apply_ablation_calibration(
            summary_rows_with=summary_rows_with,
            summary_rows_without=summary_rows_without,
            root_ablation_by_experiment=root_ablation_by_experiment,
        )
        if isinstance(out_dir, Path):
            scoring_outputs: dict[str, Any] = {}
            pre_outputs = sout._write_scored_outputs(
                summary_rows=summary_rows_with,
                file_suffix="_pre_ablation",
                value_field="stone_fraction",
                **artifact_common,
            )
            default_outputs = sout._write_scored_outputs(
                summary_rows=post_summary_rows,
                file_suffix="",
                value_field="corrected_value",
                **artifact_common,
            )
            scoring_outputs["pre_ablation"] = dict(pre_outputs)
            scoring_outputs["default"] = dict(default_outputs)
            _mirror_pooled_map_pngs(out_dir=out_dir, scoring_outputs=scoring_outputs)

            (out_dir / "root_probe_diagnostics.json").write_text(
                json.dumps(root_diag_rows, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if isinstance(manifest, dict):
                manifest["scoring_outputs"] = runtime.redact_personal_obj(scoring_outputs)
                manifest["pooled_map_artifacts"] = (
                    scoring_outputs.get("default", {}).get("pooled_map_artifacts")
                    if isinstance(scoring_outputs.get("default"), dict)
                    else None
                )
                manifest["ablation_calibration"] = runtime.redact_personal_obj(
                    {
                        "method": "pattern_ablation_v1",
                        "root_eval_mode": "batchrawnn/full_policy",
                        "candidate_eval_mode": "batchrawnn/child_positions",
                        "anchor_policy": "pass-class-restored ablation: g_hat=max_c[u(c)-u(pass_proxy)], V(c)=g_hat*m(c)+[I(c)-I(pass_proxy)], m(pass_proxy)=0, best_local=argmax V",
                        "experiment_stats": ablation_diag_rows,
                        "default_outputs_are_post_ablation": True,
                    }
                )
                (out_dir / "manifest.json").write_text(
                    json.dumps(runtime.redact_personal_obj(manifest), ensure_ascii=True, indent=2) + "\n",
                    encoding="utf-8",
                )
        else:
            pooled_rows = scoring.pool_candidates(
                post_summary_rows,
                total_representatives=len(prepared),
                value_field="corrected_value",
            )
            local_map_spec = sout._build_local_map_spec(
                first_rep=representatives[0],
                first_exp_meta=first_exp_meta,
                pooled_rows=pooled_rows,
            )
            if not isinstance(local_map_spec, dict):
                raise ValueError("Failed to build local map spec")
            final_json_path.parent.mkdir(parents=True, exist_ok=True)
            sout._write_local_map_spec_json(
                final_json_path,
                local_map_spec,
            )
        return 0
    except Exception as exc:
        runtime.log(f"Study failed: {exc}")
        return fail()
    finally:
        evaluator.checkpoint()


if __name__ == "__main__":
    raise SystemExit(main())
