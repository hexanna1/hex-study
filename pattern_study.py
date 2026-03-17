#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pattern_output_utils as sout
import study_common as lps
from local_pattern_representative import (
    Representative,
    _serialize_position,
    build_study_spec,
    extract_pattern,
    generate_representatives,
    parse_balance_profiles,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - native Windows fallback
    fcntl = None

PASS_PROXY_CANONICAL_KEY = "pass_proxy"
TENUKI_CANONICAL_KEY = "tenuki"
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
    placement_tag: str,
    candidate_Δ_max: int,
    movelist_slug: str,
) -> str:
    # Keep auto-generated artifact paths ASCII-only so local file links remain stable.
    return f"{stamp}-s{int(board_size)}{placement_tag}-d{int(candidate_Δ_max)}-{movelist_slug}"


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

def _Δ(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return int(dq * dq + dq * dr + dr * dr)


def _select_root_tenuki_move(
    payload: dict[str, Any],
    base_local_candidates: set[str],
    *,
    pattern_cells_abs: set[tuple[int, int]],
    tenuki_Δ_min: int,
) -> str | None:
    moves = payload.get("moves")
    if isinstance(moves, list):
        ranked_outside: list[tuple[int, float, int, str]] = []
        for i, entry in enumerate(moves):
            if not isinstance(entry, dict):
                continue
            mv = entry.get("move")
            if not isinstance(mv, str):
                continue
            move = mv.strip().lower()
            if not move or move in base_local_candidates:
                continue
            try:
                # Keep tenuki selection to real place-cell moves.
                col, row = lps._cell_to_col_row(move)
            except Exception:
                continue
            if pattern_cells_abs:
                min_Δ = min(_Δ((col, row), p) for p in pattern_cells_abs)
                if min_Δ < int(tenuki_Δ_min):
                    continue
            rank_val = entry.get("rank")
            rank = int(rank_val) if isinstance(rank_val, (int, float)) else (i + 1)
            prior_val = entry.get("prior")
            prior = float(prior_val) if isinstance(prior_val, (int, float)) else float("-inf")
            ranked_outside.append((rank, -prior, i, move))
        if ranked_outside:
            ranked_outside.sort(key=lambda t: (t[0], t[1], t[2]))
            return ranked_outside[0][3]
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
    return lps._candidate_key_local_for_move(mv, exp_meta)


def _rewrite_row_candidate_keys(
    *,
    rows: list[dict[str, Any]],
    exp_meta: dict[str, Any] | None,
    pass_proxy_move: str | None,
    tenuki_move: str | None,
) -> None:
    for row in rows:
        move = str(row.get("move") or "")
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
        choices=["identity", "edge-bilateral", "rotations", "d6"],
        default=None,
        help="Representative symmetry policy (default: d6, or edge-bilateral when --placement edge)",
    )
    ap.add_argument(
        "--placement",
        choices=["centered", "edge"],
        default="centered",
        help="Representative placement policy: centered (default) or fixed right-edge mode",
    )
    ap.add_argument(
        "--edge-anchor-col-from-right",
        type=int,
        default=1,
        help="For --placement edge, place the canonical anchor stone on this 1-based column from the right edge",
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
    if not agg:
        out_rows.append(
            {
                **base_row,
                "move": "",
                "candidate_abs": "",
                "candidate_key_local": None,
                "stone_fraction": None,
                "n": 0,
                "mean_winrate": None,
                "stdev_winrate": None,
                "min_winrate": None,
                "max_winrate": None,
            }
        )
        return
    for row in agg:
        out_rows.append(
            {
                **base_row,
                "move": row["move"],
                "candidate_abs": row.get("candidate_abs", ""),
                "candidate_key_local": row.get("candidate_key_local"),
                "stone_fraction": row.get("stone_fraction"),
                "n": row["n"],
                "mean_winrate": row["mean_winrate"],
                "stdev_winrate": row["stdev_winrate"],
                "min_winrate": row["min_winrate"],
                "max_winrate": row["max_winrate"],
            }
        )


def _raw_nn_cache_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "raw_nn_cache.json"

def _raw_nn_cache_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")

def _load_raw_nn_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Raw-NN cache file is not a JSON object: {path}")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _save_raw_nn_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        # Keep native Windows importable. Without advisory file locks this path
        # uses a plain atomic rewrite and does not protect concurrent writers
        # from overwriting each other.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps({k: cache[k] for k in sorted(cache)}, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return
    lock_path = _raw_nn_cache_lock_path(path)
    with lock_path.open("a", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        merged = _load_raw_nn_cache(path)
        merged.update({k: float(v) for k, v in cache.items()})
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps({k: merged[k] for k in sorted(merged)}, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)


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
                sout._redact_personal_obj(
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
def _apply_ablation_calibration_to_summary_rows(
    *,
    summary_rows_with: list[dict[str, Any]],
    summary_rows_without: list[dict[str, Any]],
    root_ablation_by_experiment: dict[str, dict[str, Any]],
    debug_top_n_per_experiment: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    without_by_exp_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows_without:
        exp = str(row.get("experiment") or "")
        key = str(row.get("candidate_key_local") or "").strip()
        if exp and key:
            without_by_exp_key.setdefault((exp, key), []).append(row)

    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows_with:
        exp = str(row.get("experiment") or "")
        by_experiment.setdefault(exp, []).append(row)

    diagnostics: list[dict[str, Any]] = []
    for exp, rows in by_experiment.items():
        root_meta = root_ablation_by_experiment.get(exp) if isinstance(root_ablation_by_experiment, dict) else None
        root_with_wr = (
            float(root_meta["with_root_winrate"]) if isinstance(root_meta, dict) and isinstance(root_meta.get("with_root_winrate"), (int, float)) else None
        )
        root_without_wr = (
            float(root_meta["without_root_winrate"]) if isinstance(root_meta, dict) and isinstance(root_meta.get("without_root_winrate"), (int, float)) else None
        )
        root_pattern_effect_logit = None
        root_with_logit = None
        root_without_logit = None
        if isinstance(root_with_wr, (int, float)) and isinstance(root_without_wr, (int, float)):
            root_with_logit = lps._logit_clamped(root_with_wr)
            root_without_logit = lps._logit_clamped(root_without_wr)
            root_pattern_effect_logit = float(root_with_logit) - float(root_without_logit)
        if not isinstance(root_pattern_effect_logit, (int, float)):
            raise ValueError(f"Ablation calibration failed for {exp}: missing root with/without evals")

        paired_entries: list[dict[str, Any]] = []
        raw_numeric_rows = [row for row in rows if isinstance(row.get("mean_winrate"), (int, float))]
        by_key_with: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("candidate_key_local") or "").strip()
            with_wr = row.get("mean_winrate")
            if isinstance(with_wr, (int, float)) and not key:
                mv = str(row.get("move") or "").strip().lower()
                raise ValueError(f"Ablation calibration failed for {exp}: missing local key for move {mv}")
            if key:
                by_key_with.setdefault(key, []).append(row)

        for key in sorted(by_key_with):
            with_rows = by_key_with[key]
            without_rows = without_by_exp_key.get((exp, key), [])
            with_wrs = [
                float(r["mean_winrate"])
                for r in with_rows
                if isinstance(r.get("mean_winrate"), (int, float))
            ]
            if not with_wrs:
                continue
            without_wrs = [
                float(r["mean_winrate"])
                for r in without_rows
                if isinstance(r.get("mean_winrate"), (int, float))
            ]
            if not without_wrs:
                raise ValueError(
                    f"Ablation calibration failed for {exp}: missing paired without-pattern eval for local key {key}"
                )
            with_logits = [float(lps._logit_clamped(v)) for v in with_wrs]
            without_logits = [float(lps._logit_clamped(v)) for v in without_wrs]
            l_with = float(mean(with_logits))
            l_without = float(mean(without_logits))
            delta_w = float(l_with) - float(root_with_logit)
            delta_u = float(l_without) - float(root_without_logit)
            interaction = delta_w - delta_u

            pre_vals = [
                float(r["stone_fraction"])
                for r in with_rows
                if isinstance(r.get("stone_fraction"), (int, float))
            ]
            display_row = min(with_rows, key=lambda r: str(r.get("move") or ""))
            move_display = str(display_row.get("move") or "").strip().lower()
            cand_display = str(display_row.get("candidate_abs") or move_display).strip().lower()
            n_total = sum(int(r.get("n") or 0) for r in with_rows)
            row_out = {
                "experiment": exp,
                "label": str(display_row.get("label") or ""),
                "mode": str(display_row.get("mode") or ""),
                "move": move_display,
                "candidate_abs": cand_display,
                "candidate_key_local": key,
                "stone_fraction_pre_ablation": (float(mean(pre_vals)) if pre_vals else None),
                "corrected_value": None,
                "n": n_total,
                "mean_winrate": float(mean(with_wrs)),
                "stdev_winrate": (float(pstdev(with_wrs)) if len(with_wrs) >= 2 else 0.0),
                "min_winrate": min(with_wrs),
                "max_winrate": max(with_wrs),
                "mean_winrate_without_pattern": float(mean(without_wrs)),
                "ablation_root_pattern_effect_logit": root_pattern_effect_logit,
                "ablation_interaction_lift_logit": interaction,
            }
            out.append(row_out)
            paired_entries.append(
                {
                    "row": row_out,
                    "move": move_display,
                    "candidate_key_local": key,
                    "signal": interaction,
                    "delta_u": delta_u,
                    "is_pass_anchor": key == PASS_PROXY_CANONICAL_KEY,
                    "pre": row_out["stone_fraction_pre_ablation"],
                    "corrected_value": None,
                }
            )

        pass_entry = next((e for e in paired_entries if bool(e.get("is_pass_anchor"))), None)
        pass_move = str(pass_entry.get("move") or "").strip().lower() if isinstance(pass_entry, dict) else ""

        if not paired_entries:
            raise ValueError(f"Ablation calibration failed for {exp}: no paired candidate rows")
        if not isinstance(pass_entry, dict):
            raise ValueError(f"Ablation calibration failed for {exp}: missing paired pass_proxy anchor")
        pass_delta_u = float(pass_entry["delta_u"])
        g_hat = max(float(e["delta_u"]) - pass_delta_u for e in paired_entries)
        pass_signal = float(pass_entry["signal"])
        for e in paired_entries:
            m_class = 0.0 if bool(e.get("is_pass_anchor")) else 1.0
            corrected = g_hat * m_class + (float(e["signal"]) - pass_signal)
            e["corrected_value"] = corrected
            e["row"]["corrected_value"] = float(corrected)

        best_entry = max(paired_entries, key=lambda e: float(e.get("corrected_value") or 0.0))
        best_move = str(best_entry.get("move") or "").strip().lower() if isinstance(best_entry, dict) else ""
        if not isinstance(best_entry, dict):
            raise ValueError(f"Ablation calibration failed for {exp}: missing local-best ablation anchor")

        max_pre = max((float(e["pre"]) for e in paired_entries if isinstance(e.get("pre"), (int, float))), default=None)
        max_corrected = max((float(e["corrected_value"]) for e in paired_entries), default=None)
        diag = {
            "experiment": exp,
            "numeric_row_count": len(raw_numeric_rows),
            "paired_row_count": len(paired_entries),
            "root_pattern_effect_logit": root_pattern_effect_logit,
            "generic_move_value_logit_hat": g_hat,
            "anchor_pass_move": (pass_move or None),
            "anchor_best_local_move": (best_move or None),
            "max_pre_fraction": max_pre,
            "max_corrected_value": max_corrected,
        }
        diagnostics.append(diag)

    return out, diagnostics


def main() -> int:
    args = _parse_args()
    debug_mode = bool(getattr(args, "debug", False))
    if args.candidate_Δ_max < 0:
        raise ValueError("--delta must be >= 0")
    if args.tenuki_Δ_min < 0:
        raise ValueError("--tenuki-delta-min must be >= 0")
    tenuki_Δ_min = int(args.tenuki_Δ_min)

    extracted = extract_pattern(args.hexworld)
    board_size = int(args.board_size) if args.board_size is not None else int(extracted.board_size_source)
    balance_moves = ["a1,d2"]
    balances = parse_balance_profiles(balance_moves)
    symmetry_policy = "edge-bilateral" if args.placement == "edge" and args.symmetry is None else str(args.symmetry or "d6")
    placement_policy = "edge" if args.placement == "edge" else "centered"
    if args.placement != "edge" and int(args.edge_anchor_col_from_right) != 1:
        raise ValueError("--edge-anchor-col-from-right requires --placement edge")
    representatives = generate_representatives(
        extracted=extracted,
        board_size=board_size,
        symmetry=symmetry_policy,
        balance_profiles=balances,
        placement=placement_policy,
        edge_anchor_col_from_right=int(args.edge_anchor_col_from_right),
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
    movelist_slug = sout._movelist_slug_from_hexworld(args.hexworld)
    placement_tag = f"-e{int(args.edge_anchor_col_from_right)}" if args.placement == "edge" else ""
    run_slug = _default_debug_dir_name(
        stamp=stamp,
        board_size=int(board_size),
        placement_tag=placement_tag,
        candidate_Δ_max=int(args.candidate_Δ_max),
        movelist_slug=movelist_slug,
    )
    out_dir = (Path(args.out_dir) if args.out_dir else Path("debug") / run_slug) if debug_mode else None
    raw_dir = (out_dir / "raw") if isinstance(out_dir, Path) else None
    final_json_path = _resolve_nondebug_output_path(out_dir_arg=args.out_dir, run_slug=run_slug)
    final_output_hint = out_dir if isinstance(out_dir, Path) else final_json_path
    if isinstance(raw_dir, Path):
        raw_dir.mkdir(parents=True, exist_ok=True)
    raw_nn_cache_path = _raw_nn_cache_path()
    raw_nn_cache = _load_raw_nn_cache(raw_nn_cache_path)
    def _finish(code: int) -> int:
        _save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
        return code
    def fail() -> int:
        print(sout._redact_personal_text(str(final_output_hint)))
        return _finish(1)
    def run_raw(*, cmd: list[str], exp_dir: Path | None, filename: str, experiment_name: str) -> tuple[bool, dict[str, Any], str, float]:
        t0 = time.time()
        ok, payload, err = lps._run_once(cmd)
        elapsed = time.time() - t0
        _write_raw_payload(
            exp_dir=exp_dir,
            filename=filename,
            experiment_name=experiment_name,
            command=cmd,
            ok=ok,
            err=err,
            payload=payload,
        )
        return ok, payload, err, elapsed

    def candidate_rows_from_cache(
        *,
        position_input: str,
        moves: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        red_by_move: dict[str, float] = {}
        for move in moves:
            cached = raw_nn_cache.get(lps._cache_key(position_input, move))
            if not isinstance(cached, (int, float)):
                raise ValueError(f"Missing cached child winrate for move {move!r} from {position_input!r}")
            red_by_move[move] = float(cached)
        rows = [
            {
                "move": move,
                "n": 1,
                "mean_winrate": (red_by_move[move] if extracted.to_play_at_cursor == "red" else (1.0 - red_by_move[move])),
                "stdev_winrate": 0.0,
                "min_winrate": (red_by_move[move] if extracted.to_play_at_cursor == "red" else (1.0 - red_by_move[move])),
                "max_winrate": (red_by_move[move] if extracted.to_play_at_cursor == "red" else (1.0 - red_by_move[move])),
                "_idx": idx,
            }
            for idx, move in enumerate(moves)
        ]
        lps._attach_stone_fractions(
            {"position": {"input": position_input, "to_play": extracted.to_play_at_cursor}, "mode": "candidate"},
            rows,
            position_input=position_input,
        )
        rows.sort(
            key=lambda r: (
                -(r["mean_winrate"] if r["mean_winrate"] is not None else float("-inf")),
                r["_idx"],
                r["move"],
            )
        )
        debug_payload = {
            "source": "raw_nn_cache_batched_analyze",
            "position": {"input": position_input, "to_play": extracted.to_play_at_cursor},
            "rows": rows,
        }
        return rows, debug_payload

    def batch_fill_candidate_cache(
        request_specs: list[tuple[tuple[int, str], str, list[str]]]
    ) -> int:
        child_position_by_key: dict[str, str] = {}
        total_cache_miss_moves = 0
        for request_id, position_input, moves in request_specs:
            for move in moves:
                child_key = lps._cache_key(position_input, move)
                cached = raw_nn_cache.get(child_key)
                if isinstance(cached, (int, float)):
                    continue
                total_cache_miss_moves += 1
                child_position = lps._position_after_move(position_input, move)
                child_position_by_key.setdefault(child_key, child_position)
        if not child_position_by_key:
            return total_cache_miss_moves
        key_items = list(child_position_by_key.items())
        for start in range(0, len(key_items), 250):
            chunk = key_items[start : start + 250]
            payload_by_position = lps._run_multi_position_analyze(
                hexata_main=hexata_main,
                position_inputs=[position for _key, position in chunk],
            )
            for child_key, child_position in chunk:
                payload = payload_by_position.get(child_position)
                root_eval = payload.get("root_eval") if isinstance(payload, dict) else None
                red_wr = root_eval.get("red_winrate") if isinstance(root_eval, dict) else None
                if not isinstance(red_wr, (int, float)):
                    raise ValueError(f"Raw-NN analyze payload missing root winrate for {child_position!r}")
                raw_nn_cache[child_key] = float(red_wr)
        return total_cache_miss_moves

    hexata_main = lps._hexata_main_path()
    if not hexata_main.exists():
        raise FileNotFoundError(f"Missing hexata CLI entrypoint: {hexata_main}")

    exp_meta_map = (base_spec.get("generator_meta") or {}).get("experiment_meta") or {}
    generator_meta = base_spec.get("generator_meta")
    base_exps = list(base_spec["experiments"])
    if len(base_exps) != len(representatives):
        raise RuntimeError(
            "Representative/spec experiment count mismatch:"
            f" reps={len(representatives)} exps={len(base_exps)}"
        )

    study_start = time.time()

    shared_without_position: str
    canonical_blue_without_occupied: set[str] | None = None
    canonical_red_point = tuple(int(x) for x in lps._cell_to_col_row(CANONICAL_BALANCE_RED_MOVE))
    canonical_blue_point = tuple(int(x) for x in lps._cell_to_col_row(CANONICAL_BALANCE_BLUE_MOVE))
    if extracted.to_play_at_cursor == "blue":
        pass_proxy_forbidden = {
            str((list(exp.get("candidates") or [""])[0])).strip().lower()
            for exp in base_exps
        }
        if not pass_proxy_forbidden or any(not mv for mv in pass_proxy_forbidden):
            raise ValueError("Missing pass-proxy candidate at index 0")
        canonical_blue_probe_position = _serialize_position(
            board_size=int(board_size),
            red_cells=(canonical_red_point,),
            blue_cells=(canonical_blue_point,),
            to_play="red",
        )
        ok_canon, payload_canon, err_canon, elapsed_canon = run_raw(
            cmd=lps._build_cmd({"position": canonical_blue_probe_position}, {}, hexata_main=hexata_main),
            exp_dir=raw_dir,
            filename="canonical_blue_probe.json",
            experiment_name="canonical_blue_probe",
        )
        if not ok_canon:
            return fail()
        moves = payload_canon.get("moves")
        if not isinstance(moves, list):
            raise ValueError("Canonical move3 probe failed: missing moves list")
        forbidden_moves = {CANONICAL_BALANCE_RED_MOVE, CANONICAL_BALANCE_BLUE_MOVE} | pass_proxy_forbidden
        ranked_move3: list[tuple[float, int, int, str]] = []
        for i, entry in enumerate(moves):
            if not isinstance(entry, dict):
                continue
            mv = str(entry.get("move") or "").strip().lower()
            if not mv or mv in forbidden_moves:
                continue
            try:
                lps._cell_to_col_row(mv)
            except Exception:
                continue
            rank_val = entry.get("rank")
            rank = int(rank_val) if isinstance(rank_val, (int, float)) else (i + 1)
            prior_val = entry.get("prior")
            prior = float(prior_val) if isinstance(prior_val, (int, float)) else float("-inf")
            ranked_move3.append((-prior, rank, i, mv))
        if not ranked_move3:
            raise ValueError("Canonical move3 probe failed: no eligible place-cell move")
        ranked_move3.sort(key=lambda t: (t[0], t[1], t[2]))
        canonical_blue_move3 = ranked_move3[0][3]
        canonical_blue_move3_point = tuple(int(x) for x in lps._cell_to_col_row(canonical_blue_move3))
        for col, row in (canonical_red_point, canonical_blue_point, canonical_blue_move3_point):
            if not (1 <= col <= int(board_size) and 1 <= row <= int(board_size)):
                raise ValueError(f"Canonical baseline has out-of-bounds move for board size {int(board_size)}")
        if canonical_blue_move3_point == canonical_blue_point:
            raise ValueError("Canonical baseline overlap between red/blue stones")
        shared_without_position = _serialize_position(
            board_size=int(board_size),
            red_cells=tuple(sorted({canonical_red_point, canonical_blue_move3_point})),
            blue_cells=(canonical_blue_point,),
            to_play="blue",
        )
        canonical_blue_without_occupied = {
            CANONICAL_BALANCE_RED_MOVE,
            CANONICAL_BALANCE_BLUE_MOVE,
            str(canonical_blue_move3).strip().lower(),
        }
        lps._log(
            "[canonical blue baseline]"
            f" move3={canonical_blue_move3}"
            f" probe_elapsed={lps._fmt_s(elapsed_canon)}"
        )
    else:
        shared_without_position = _serialize_position(
            board_size=int(board_size),
            red_cells=(canonical_red_point,),
            blue_cells=(canonical_blue_point,),
            to_play="red",
        )
    prepared: list[dict[str, Any]] = []
    probe_start = time.time()
    probe_rows = [
        (
            exp,
            rep,
            shared_without_position,
        )
        for exp, rep in zip(base_exps, representatives)
    ]
    positions_to_analyze: list[str] = []
    seen_probe_keys: set[str] = set()
    probe_reused_roots = 0
    for exp, _rep, without_position in probe_rows:
        with_position = str(exp["position"])
        with_key = lps._cache_key(with_position)
        if with_key not in seen_probe_keys:
            seen_probe_keys.add(with_key)
            positions_to_analyze.append(with_position)
        without_key = lps._cache_key(without_position)
        if without_key not in seen_probe_keys:
            seen_probe_keys.add(without_key)
            if isinstance(raw_nn_cache.get(without_key), (int, float)):
                probe_reused_roots += 1
            else:
                positions_to_analyze.append(without_position)

    try:
        root_payload_by_position = lps._run_multi_position_analyze(
            hexata_main=hexata_main,
            position_inputs=positions_to_analyze,
        )
    except Exception as exc:
        lps._log(f"[probe] batch failed: {exc}")
        return fail()
    for position, payload in root_payload_by_position.items():
        root_eval = payload.get("root_eval") if isinstance(payload, dict) else None
        red_wr = root_eval.get("red_winrate") if isinstance(root_eval, dict) else None
        if isinstance(red_wr, (int, float)):
            raw_nn_cache[lps._cache_key(position)] = float(red_wr)

    for i, (exp, rep, without_position) in enumerate(probe_rows, start=1):
        name = str(exp["name"])
        label = str(exp.get("label") or "")
        position = str(exp["position"])
        payload = root_payload_by_position.get(position)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing batched root probe payload for {name}")
        base_candidates = [str(x).strip().lower() for x in list(exp["candidates"])]
        if not base_candidates or not base_candidates[0]:
            raise ValueError(f"Missing pass-proxy candidate at index 0 for {name}")
        pass_proxy_move = str(base_candidates[0]).strip().lower()

        root_red_wr_without = raw_nn_cache.get(lps._cache_key(without_position))
        if not isinstance(root_red_wr_without, (int, float)):
            raise ValueError(f"Missing cached root probe winrate for {without_position}")

        base_local_set = set(base_candidates)
        pattern_cells_abs = set(rep.plus_abs) | set(rep.minus_abs)
        tenuki_move = _select_root_tenuki_move(
            payload,
            base_local_set,
            pattern_cells_abs=pattern_cells_abs,
            tenuki_Δ_min=tenuki_Δ_min,
        )
        aug_candidates = list(base_candidates)
        if tenuki_move:
            aug_candidates.append(tenuki_move)

        # run_candidates must retain the pass-proxy anchor for calibration.
        run_candidates = (
            [c for c in aug_candidates if c not in canonical_blue_without_occupied]
            if isinstance(canonical_blue_without_occupied, set)
            else list(aug_candidates)
        )
        if pass_proxy_move not in run_candidates:
            raise ValueError(
                "Pass-proxy candidate was removed by occupancy filtering"
                f" for {name}: {pass_proxy_move}"
            )
        if tenuki_move and tenuki_move not in run_candidates:
            tenuki_move = None

        root_ablation_with_wr = lps._root_eval_side_to_play_winrate(payload)
        root_ablation_without_wr = float(root_red_wr_without) if extracted.to_play_at_cursor == "red" else (1.0 - float(root_red_wr_without))
        prepared.append(
            {
                "index": i,
                "name": name,
                "label": label,
                "with_position": position,
                "without_position": without_position,
                "base_candidates": list(base_candidates),
                "candidates": list(run_candidates),
                "root_ablation_with_winrate": root_ablation_with_wr,
                "root_ablation_without_winrate": root_ablation_without_wr,
                "tenuki_move": tenuki_move,
                "pass_proxy_move": pass_proxy_move,
            }
        )
    lps._log(
        f"[probe] cache={probe_reused_roots}/{len(seen_probe_keys)}"
        f" elapsed={lps._fmt_s(time.time() - probe_start)}"
    )

    if isinstance(out_dir, Path):
        run_spec = {
            "version": "raw_nn_v1",
            "source_hexworld": args.hexworld,
            "board_size": board_size,
            "symmetry": symmetry_policy,
            "placement": placement_policy,
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
        if placement_policy == "edge":
            run_spec["edge_anchor_col_from_right"] = int(args.edge_anchor_col_from_right)
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
            "hexata_main": str(hexata_main),
            "source_hexworld": args.hexworld,
            "experiments": [],
        }
        if isinstance(out_dir, Path)
        else None
    )

    batch_analyze_cmd = ["python3", str(hexata_main), "cli", "analyze", "-"]
    candidate_request_specs: list[tuple[tuple[int, str], str, list[str]]] = []
    for p in prepared:
        candidate_request_specs.append(((int(p["index"]), "with"), str(p["with_position"]), list(p["candidates"])))
        candidate_request_specs.append(((int(p["index"]), "without"), str(p["without_position"]), list(p["candidates"])))
    candidate_batch_t0 = time.time()
    try:
        total_cache_miss_moves = batch_fill_candidate_cache(candidate_request_specs)
    except Exception as exc:
        batch_err = str(exc)
        lps._log(f"[candidate batch] failed: {batch_err}")
        return fail()
    for p in prepared:
        name = str(p["name"])
        label = str(p["label"])
        exp_dir = (raw_dir / f"{int(p['index']):02d}-{lps._safe_name(name)}") if isinstance(raw_dir, Path) else None
        if isinstance(exp_dir, Path):
            exp_dir.mkdir(parents=True, exist_ok=True)

        agg_with, payload_with = candidate_rows_from_cache(
            position_input=str(p["with_position"]),
            moves=list(p["candidates"]),
        )
        _write_raw_payload(
            exp_dir=exp_dir,
            filename="candidate_with.json",
            experiment_name=name,
            command=batch_analyze_cmd,
            ok=True,
            err="",
            payload=payload_with,
        )

        agg_without, payload_without = candidate_rows_from_cache(
            position_input=str(p["without_position"]),
            moves=list(p["candidates"]),
        )
        _write_raw_payload(
            exp_dir=exp_dir,
            filename="candidate_without.json",
            experiment_name=name,
            command=batch_analyze_cmd,
            ok=True,
            err="",
            payload=payload_without,
        )

        mode = "candidate"
        exp_meta = exp_meta_map.get(name) if isinstance(exp_meta_map, dict) else None
        lps._attach_candidate_keys(agg_with, exp_meta if isinstance(exp_meta, dict) else None)
        lps._attach_candidate_keys(agg_without, exp_meta if isinstance(exp_meta, dict) else None)
        _rewrite_row_candidate_keys(
            rows=agg_with,
            exp_meta=(exp_meta if isinstance(exp_meta, dict) else None),
            pass_proxy_move=(str(p.get("pass_proxy_move") or "").strip().lower() or None),
            tenuki_move=(str(p.get("tenuki_move") or "").strip().lower() or None),
        )
        _rewrite_row_candidate_keys(
            rows=agg_without,
            exp_meta=(exp_meta if isinstance(exp_meta, dict) else None),
            pass_proxy_move=(str(p.get("pass_proxy_move") or "").strip().lower() or None),
            tenuki_move=(str(p.get("tenuki_move") or "").strip().lower() or None),
        )

        base_row = {"experiment": name, "label": label, "mode": mode}
        _append_summary_rows_from_agg(out_rows=summary_rows_with, base_row=base_row, agg=agg_with)
        _append_summary_rows_from_agg(out_rows=summary_rows_without, base_row=base_row, agg=agg_without)

        root_diag_rows.append(
            {
                "experiment": name,
                "tenuki_move": p.get("tenuki_move"),
                "base_candidate_count": len(list(p["base_candidates"])),
                "candidate_count": len(list(p["candidates"])),
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

    total_candidate_cache_hits = sum(len(spec[2]) for spec in candidate_request_specs) - total_cache_miss_moves
    total_candidate_requests = sum(len(spec[2]) for spec in candidate_request_specs)
    lps._log(
        f"[candidate]"
        f" reps={len(prepared)}"
        f" cache={total_candidate_cache_hits}/{total_candidate_requests}"
        f" elapsed={lps._fmt_s(time.time() - candidate_batch_t0)}"
        f" total_elapsed={lps._fmt_s(time.time() - study_start)}"
    )

    first_exp_name = str(prepared[0]["name"]) if prepared else ""
    first_exp_meta = exp_meta_map.get(first_exp_name) if isinstance(exp_meta_map, dict) else None
    first_exp_meta_dict = first_exp_meta if isinstance(first_exp_meta, dict) else None
    artifact_common = {
        "out_dir": out_dir,
        "first_rep": representatives[0],
        "first_exp_meta": first_exp_meta_dict,
        "total_representatives": len(prepared),
    }

    root_ablation_by_experiment = {
        str(p["name"]): {
            "with_root_winrate": p.get("root_ablation_with_winrate"),
            "without_root_winrate": p.get("root_ablation_without_winrate"),
        }
        for p in prepared
    }
    post_summary_rows, ablation_diag_rows = _apply_ablation_calibration_to_summary_rows(
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
            manifest["scoring_outputs"] = sout._redact_personal_obj(scoring_outputs)
            manifest["pooled_map_artifacts"] = (
                scoring_outputs.get("default", {}).get("pooled_map_artifacts")
                if isinstance(scoring_outputs.get("default"), dict)
                else None
            )
            manifest["ablation_calibration"] = sout._redact_personal_obj(
                {
                    "method": "pattern_ablation_v1",
                    "root_eval_mode": "analyze/raw_nn",
                    "candidate_eval_mode": "analyze/raw_nn_child_positions",
                    "anchor_policy": "pass-class-restored ablation: g_hat=max_c[u(c)-u(pass_proxy)], V(c)=g_hat*m(c)+[I(c)-I(pass_proxy)], m(pass_proxy)=0, best_local=argmax V",
                    "experiment_stats": ablation_diag_rows,
                    "default_outputs_are_post_ablation": True,
                }
            )
            (out_dir / "manifest.json").write_text(
                json.dumps(sout._redact_personal_obj(manifest), ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        pooled_rows = lps._build_pooled_candidates(
            post_summary_rows,
            total_representatives=len(prepared),
            value_field="corrected_value",
        )
        local_map_spec = sout._build_local_map_spec(
            first_rep=representatives[0],
            first_exp_meta=first_exp_meta_dict,
            pooled_rows=pooled_rows,
        )
        if not isinstance(local_map_spec, dict):
            raise ValueError("Failed to build local map spec")
        final_json_path.parent.mkdir(parents=True, exist_ok=True)
        sout._write_local_map_spec_json(
            final_json_path,
            local_map_spec,
        )
    return _finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
