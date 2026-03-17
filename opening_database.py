#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dead_region_rules import canonicalize_acute_equivalent_move, should_exclude_acute_dead_region_move
import local_pattern_representative as lpr
import study_common as lps


TOP_K_BY_PLY: dict[int, int] = {
    1: 6,
    2: 5,
    3: 4,
    4: 3,
}
DEFAULT_TOP_K = 2
DEFAULT_SMALL_BOARD_IMPORTANCE_MIN = 0.87
DEFAULT_LARGE_BOARD_IMPORTANCE_MIN = 0.89
IMPORTANCE_MIN_BY_BOARD_SIZE: dict[int, float] = {
    13: 0.86,
}
PLY_DECAY = 0.994
EXTRA_CANDIDATE_PRIOR_MIN = 0.20
WINRATE_CLAMP_EPS = 1e-6
RAW_NN_CACHE_CHUNK_SIZE = 250
RAW_NN_CACHE_MOVE_LIMIT = 15
RAW_NN_CACHE_SPECIAL_PREFIXES = ("fair-root-candidate::",)
FAIR_ROOT_CACHE_VERSION = 1
FAIR_REFERENCE_STONE_FRACTION = 0.75
FAIR_STONE_FRACTION_MIN = 0.36
FAIR_STONE_FRACTION_MAX = 0.64


PositionState = tuple[int, set[tuple[int, int]], set[tuple[int, int]], str]


def _ply_decay(*, board_size: int) -> float:
    return float(PLY_DECAY)


def _importance_min(*, board_size: int) -> float:
    size = int(board_size)
    if size > 14:
        return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_LARGE_BOARD_IMPORTANCE_MIN))
    return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_SMALL_BOARD_IMPORTANCE_MIN))


def _extra_candidate_prior_min() -> float:
    return float(EXTRA_CANDIDATE_PRIOR_MIN)


@dataclass
class OpeningNode:
    position: str
    moves: tuple[str, ...]
    importance: float = 1.0


def _default_output_path(*, board_size: int) -> Path:
    return Path("artifacts") / "openings" / f"openings-s{int(board_size)}.json"


def _raw_nn_cache_path(*, board_size: int) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "openings" / f"openings_raw_nn_cache_s{int(board_size)}.json"


def _is_special_raw_nn_cache_key(key: str) -> bool:
    key_s = str(key)
    return any(key_s.startswith(prefix) for prefix in RAW_NN_CACHE_SPECIAL_PREFIXES)


def _run_multi_position_analyze_cached(
    *,
    hexata_main: Path,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    return lps._run_multi_position_analyze_cached(
        hexata_main=hexata_main,
        position_inputs=position_inputs,
        raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path,
        chunk_size=RAW_NN_CACHE_CHUNK_SIZE,
        move_limit=RAW_NN_CACHE_MOVE_LIMIT,
    )


def _log(message: str, *, board_size: int | None = None) -> None:
    prefix = ""
    if isinstance(board_size, int):
        prefix = f"[{board_size}] "
    lps._log(f"{prefix}{message}")


def _fmt_s(sec: float) -> str:
    return lps._fmt_s(sec)


def _empty_position(*, board_size: int) -> str:
    return lpr._serialize_position(
        board_size=int(board_size),
        red_cells=(),
        blue_cells=(),
        to_play="red",
    )


def _top_k_for_ply(ply: int) -> int:
    if int(ply) <= 0:
        top_k = DEFAULT_TOP_K
    else:
        top_k = TOP_K_BY_PLY.get(int(ply), DEFAULT_TOP_K)
    if int(top_k) > RAW_NN_CACHE_MOVE_LIMIT:
        raise ValueError(
            f"top-k policy exceeds raw-NN cache move limit: {top_k} > {RAW_NN_CACHE_MOVE_LIMIT}"
        )
    return int(top_k)


def _mover_winrate_from_child_payload(*, child_payload: dict[str, Any], parent_to_play: str) -> float:
    root_eval = child_payload.get("root_eval")
    if not isinstance(root_eval, dict) or not isinstance(root_eval.get("red_winrate"), (int, float)):
        raise ValueError("child payload missing root_eval.red_winrate")
    red_wr = float(root_eval["red_winrate"])
    if str(parent_to_play).strip().lower() == "red":
        return red_wr
    return 1.0 - red_wr


def _winrate_to_elo(winrate: float) -> float:
    p = max(WINRATE_CLAMP_EPS, min(1.0 - WINRATE_CLAMP_EPS, float(winrate)))
    return 400.0 * math.log10(p / (1.0 - p))


def _rounded_float(value: Any, *, digits: int = 6) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _full_stone_elo_from_root_study(root_study: dict[str, Any]) -> float:
    reference_elo = root_study.get("reference_elo")
    if not isinstance(reference_elo, (int, float)):
        raise ValueError("root_study missing numeric reference_elo")
    full_stone_elo = 4.0 * float(reference_elo)
    if full_stone_elo <= 0.0:
        raise ValueError(f"bad full-stone Elo calibration: {full_stone_elo!r}")
    return full_stone_elo


def _root_stone_fraction_from_study(*, move: str, root_study: dict[str, Any]) -> float:
    rows = root_study.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("root_study missing rows")
    by_move: dict[str, float] = {}
    best_distance: float | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_move = str(row.get("move") or "").strip().lower()
        stone_fraction = row.get("stone_fraction")
        if not row_move or not isinstance(stone_fraction, (int, float)):
            continue
        sf = float(stone_fraction)
        by_move[row_move] = sf
        distance = abs(sf - 0.5)
        best_distance = distance if best_distance is None else min(best_distance, distance)
    if best_distance is None:
        raise ValueError("root_study rows missing stone-fraction calibration")
    move_s = str(move).strip().lower()
    if move_s not in by_move:
        raise ValueError(f"root_study missing root move calibration: {move_s!r}")
    distance = abs(float(by_move[move_s]) - 0.5)
    raw_stone_fraction = max(0.0, min(1.0, 1.0 - (distance - float(best_distance))))
    return math.sqrt(raw_stone_fraction)


def _stone_fraction_from_elo_loss(*, elo_loss: float, full_stone_elo: float) -> float:
    return max(0.0, 1.0 - (float(elo_loss) / float(full_stone_elo)))


def _col_row_to_cell(col: int, row: int) -> str:
    if int(col) <= 0 or int(row) <= 0:
        raise ValueError(f"bad col/row for cell formatting: {(col, row)!r}")
    letters: list[str] = []
    n = int(col)
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("a") + rem))
    return "".join(reversed(letters)) + str(int(row))


def _reference_root_move(*, board_size: int) -> str:
    size = int(board_size)
    if size < 3:
        raise ValueError(f"board size too small for fair-root reference: {size}")
    return _col_row_to_cell(2, size - 1)


def _rotate_180_move(move: str, *, board_size: int) -> str:
    col, row = lpr.CELL_TO_COL_ROW(str(move).strip().lower())
    size = int(board_size)
    return _col_row_to_cell(size + 1 - int(col), size + 1 - int(row))


def _canonical_rotation_root_move(move: str, *, board_size: int) -> str:
    a = str(move).strip().lower()
    b = _rotate_180_move(a, board_size=board_size)
    a_col, a_row = lpr.CELL_TO_COL_ROW(a)
    b_col, b_row = lpr.CELL_TO_COL_ROW(b)
    size = int(board_size)

    def rep_key(col: int, row: int, cell: str) -> tuple[int, int, int, int, str]:
        on_preferred_side = int(row) + int(col) <= size + 1
        diagonal_tiebreak = 0 if int(row) >= int(col) else 1
        return (0 if on_preferred_side else 1, diagonal_tiebreak, int(row), int(col), cell)

    a_key = rep_key(int(a_col), int(a_row), a)
    b_key = rep_key(int(b_col), int(b_row), b)
    return a if a_key <= b_key else b


def _coarse_bucket_root_move(move: str, *, board_size: int) -> str:
    col, row = lpr.CELL_TO_COL_ROW(str(move).strip().lower())
    size = int(board_size)
    if int(row) == 2:
        if int(col) <= 3:
            return _col_row_to_cell(3, 2)
        if int(col) >= size - 2:
            return _col_row_to_cell(size - 2, 2)
    return str(move).strip().lower()


def _canonical_fair_root_move(move: str, *, board_size: int) -> str:
    return _coarse_bucket_root_move(
        _canonical_rotation_root_move(move, board_size=board_size),
        board_size=board_size,
    )


def _canonical_fair_root_representatives(*, board_size: int) -> tuple[str, ...]:
    size = int(board_size)
    reps = {
        _canonical_fair_root_move(_col_row_to_cell(col, row), board_size=size)
        for col in range(1, size + 1)
        for row in range(1, size + 1)
    }
    return tuple(sorted(reps, key=lambda cell: tuple(int(x) for x in reversed(lpr.CELL_TO_COL_ROW(cell)))))


def _fair_root_cache_key(*, board_size: int) -> str:
    return f"{RAW_NN_CACHE_SPECIAL_PREFIXES[0]}v{FAIR_ROOT_CACHE_VERSION}::s{int(board_size)}"


def _minimal_candidate_sweep_payload(payload: dict[str, Any], *, requested_moves: tuple[str, ...]) -> dict[str, Any]:
    rows_by_move: dict[str, float] = {}
    for row in payload.get("moves", []):
        if not isinstance(row, dict):
            continue
        move = str(row.get("move") or "").strip().lower()
        red_wr = row.get("red_winrate")
        if not move or not isinstance(red_wr, (int, float)):
            continue
        rows_by_move[move] = float(red_wr)
    kept_rows = [{"move": move, "red_winrate": rows_by_move[move]} for move in requested_moves if move in rows_by_move]
    if len(kept_rows) != len(requested_moves):
        missing = [move for move in requested_moves if move not in rows_by_move]
        raise ValueError(f"Candidate sweep missing rows for moves: {missing}")
    return {"moves": kept_rows}


def _run_fair_root_candidate_sweep_cached(
    *,
    board_size: int,
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    cache_key = _fair_root_cache_key(board_size=board_size)
    canonical_moves = list(_canonical_fair_root_representatives(board_size=board_size))
    reference_move = _reference_root_move(board_size=board_size)
    requested_moves: tuple[str, ...] = tuple(
        canonical_moves if reference_move in canonical_moves else (canonical_moves + [reference_move])
    )
    cached = raw_nn_cache.get(cache_key)
    if isinstance(cached, dict):
        cached_moves = cached.get("moves")
        if isinstance(cached_moves, list):
            returned = {
                str(row.get("move") or "").strip().lower()
                for row in cached_moves
                if isinstance(row, dict) and str(row.get("move") or "").strip()
            }
            if all(move in returned for move in requested_moves):
                return cached, 1
    ok, payload, err = lps._run_once(
        lps._build_cmd(
            {"position": _empty_position(board_size=board_size), "candidates": list(requested_moves)},
            {},
            hexata_main=lps._hexata_main_path(),
        )
    )
    if not ok:
        raise RuntimeError(f"fair-root candidate sweep failed: {err}")
    reduced = _minimal_candidate_sweep_payload(payload, requested_moves=requested_moves)
    raw_nn_cache[cache_key] = reduced
    if isinstance(raw_nn_cache_path, Path):
        lps._save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
    return reduced, 0


def _derive_fair_root_study(
    *,
    board_size: int,
    sweep_payload: dict[str, Any],
) -> dict[str, Any]:
    moves = sweep_payload.get("moves")
    if not isinstance(moves, list):
        raise ValueError("fair-root sweep payload missing moves list")
    rows_by_move = {
        str(row.get("move") or "").strip().lower(): float(row["red_winrate"])
        for row in moves
        if isinstance(row, dict)
        and str(row.get("move") or "").strip()
        and isinstance(row.get("red_winrate"), (int, float))
    }
    reference_move = _reference_root_move(board_size=board_size)
    reference_wr = rows_by_move.get(reference_move)
    if not isinstance(reference_wr, float):
        raise ValueError(f"fair-root reference move missing from sweep: {reference_move!r}")
    reference_elo = _winrate_to_elo(reference_wr)
    if abs(reference_elo) < 1e-9:
        raise ValueError(f"fair-root reference Elo is too small: {reference_move!r} -> {reference_wr}")
    rows: list[dict[str, Any]] = []
    root_openings: list[str] = []
    for move in _canonical_fair_root_representatives(board_size=board_size):
        wr = rows_by_move.get(move)
        if not isinstance(wr, float):
            raise ValueError(f"fair-root canonical move missing from sweep: {move!r}")
        elo = _winrate_to_elo(wr)
        stone_fraction = 0.5 + ((float(elo) / float(reference_elo)) * (FAIR_REFERENCE_STONE_FRACTION - 0.5))
        fair = bool(FAIR_STONE_FRACTION_MIN <= stone_fraction <= FAIR_STONE_FRACTION_MAX)
        row = {
            "move": move,
            "red_winrate": _rounded_float(wr),
            "elo": _rounded_float(elo),
            "stone_fraction": _rounded_float(stone_fraction),
            "fair": fair,
        }
        rows.append(row)
        if fair:
            root_openings.append(move)
    return {
        "reference_move": reference_move,
        "reference_red_winrate": _rounded_float(reference_wr),
        "reference_elo": _rounded_float(reference_elo),
        "reference_stone_fraction": _rounded_float(FAIR_REFERENCE_STONE_FRACTION),
        "fair_band": [_rounded_float(FAIR_STONE_FRACTION_MIN), _rounded_float(FAIR_STONE_FRACTION_MAX)],
        "rows": rows,
        "root_openings": root_openings,
    }


def _select_root_candidates(
    *,
    node: OpeningNode,
    board_size: int,
    root_openings: tuple[str, ...],
    parent_state: PositionState | None = None,
) -> list[dict[str, Any]]:
    if parent_state is None:
        size, red, blue, to_play = lps._position_state(node.position)
    else:
        size, red, blue, to_play = parent_state
    if size != int(board_size) or to_play != "red":
        raise ValueError(f"unexpected root node state for board size {board_size}: {node.position!r}")
    return [
        {
            "move": move,
            "rank": idx,
            "prior": None,
            "child_position": lps._position_after_move_from_state(
                size=size,
                red=red,
                blue=blue,
                to_play=to_play,
                move=move,
            ),
            "parent_to_play": "red",
            "board_size": int(board_size),
        }
        for idx, move in enumerate(root_openings, start=1)
    ]


def _policy_candidates_from_payload(
    *,
    payload: dict[str, Any],
    board_size: int,
    red: set[tuple[int, int]],
    blue: set[tuple[int, int]],
    to_play: str,
) -> list[dict[str, Any]]:
    occupied = red | blue
    moves = payload.get("moves")
    if not isinstance(moves, list):
        return []
    candidates: list[dict[str, Any]] = []
    seen_canonical_moves: set[str] = set()
    for seq_idx, row in enumerate(moves):
        if not isinstance(row, dict):
            continue
        move = str(row.get("move") or "").strip().lower()
        if not move or move == "pass":
            continue
        try:
            col, row_num = lpr.CELL_TO_COL_ROW(move)
        except Exception:
            continue
        if (int(col), int(row_num)) in occupied:
            continue
        if should_exclude_acute_dead_region_move(
            move=(int(col), int(row_num)),
            red=red,
            blue=blue,
            board_size=board_size,
        ):
            continue
        canonical_col, canonical_row = canonicalize_acute_equivalent_move(
            move=(int(col), int(row_num)),
            red=red,
            blue=blue,
            board_size=board_size,
        )
        move = _col_row_to_cell(canonical_col, canonical_row)
        if move in seen_canonical_moves:
            continue
        rank_val = row.get("rank")
        rank = int(rank_val) if isinstance(rank_val, (int, float)) else (seq_idx + 1)
        prior_val = row.get("prior")
        prior = float(prior_val) if isinstance(prior_val, (int, float)) else 0.0
        seen_canonical_moves.add(move)
        candidates.append(
            {
                "move": move,
                "rank": rank,
                "prior": prior,
                "child_position": lps._position_after_move_from_state(
                    size=board_size,
                    red=red,
                    blue=blue,
                    to_play=to_play,
                    move=move,
                ),
                "parent_to_play": to_play,
                "board_size": board_size,
            }
        )
    return candidates


def _select_top_prior_candidates(
    *,
    node: OpeningNode,
    payload: dict[str, Any],
    parent_state: PositionState | None = None,
) -> list[dict[str, Any]]:
    if parent_state is None:
        board_size, red, blue, to_play = lps._position_state(node.position)
    else:
        board_size, red, blue, to_play = parent_state
    cleaned = _policy_candidates_from_payload(
        payload=payload,
        board_size=board_size,
        red=red,
        blue=blue,
        to_play=to_play,
    )
    candidates: list[dict[str, Any]] = []
    top_k = _top_k_for_ply(len(node.moves))
    valid_seen = 0
    for row in cleaned:
        prior = float(row["prior"])
        valid_seen += 1
        if valid_seen > top_k and prior < _extra_candidate_prior_min():
            continue
        candidates.append(dict(row))
    return candidates


def _can_skip_child_expansion(*, node: OpeningNode, board_size: int) -> bool:
    return float(node.importance) * _ply_decay(board_size=int(board_size)) < _importance_min(board_size=int(board_size))


def _finalize_node(
    node: OpeningNode,
    *,
    candidates: list[dict[str, Any]],
    child_payloads: dict[str, dict[str, Any]],
    root_study: dict[str, Any],
) -> tuple[dict[str, Any], list[OpeningNode]]:
    evaluated: list[dict[str, Any]] = []
    best_elo: float | None = None
    full_stone_elo = _full_stone_elo_from_root_study(root_study)
    for cand in candidates:
        child_payload = child_payloads.get(str(cand["child_position"]))
        if not isinstance(child_payload, dict):
            raise ValueError(f"missing child payload for {cand['child_position']!r}")
        mover_wr = _mover_winrate_from_child_payload(
            child_payload=child_payload,
            parent_to_play=str(cand["parent_to_play"]),
        )
        elo = _winrate_to_elo(mover_wr)
        best_elo = elo if best_elo is None else max(best_elo, elo)
        evaluated.append(
            {
                **cand,
                "mover_winrate": mover_wr,
                "_elo": elo,
            }
        )
    children: list[OpeningNode] = []
    candidate_rows: list[dict[str, Any]] = []
    retained_moves: list[str] = []
    for cand in evaluated:
        elo_loss = float(best_elo - cand["_elo"]) if best_elo is not None else None
        if not node.moves:
            stone_fraction = _root_stone_fraction_from_study(move=str(cand["move"]), root_study=root_study)
        elif elo_loss is not None:
            stone_fraction = _stone_fraction_from_elo_loss(elo_loss=elo_loss, full_stone_elo=full_stone_elo)
        else:
            stone_fraction = None
        child_importance = (
            float(node.importance) * float(stone_fraction) * _ply_decay(board_size=int(cand["board_size"]))
            if isinstance(stone_fraction, (int, float))
            else None
        )
        retained = bool(
            isinstance(child_importance, (int, float))
            and float(child_importance) >= _importance_min(board_size=int(cand["board_size"]))
        )
        candidate_rows.append(
            {
                "move": str(cand["move"]),
                "rank": int(cand["rank"]),
                "prior": _rounded_float(cand["prior"]),
                "mover_winrate": _rounded_float(cand["mover_winrate"]),
                "elo_loss": _rounded_float(elo_loss) if elo_loss is not None else None,
                "stone_fraction": _rounded_float(stone_fraction) if stone_fraction is not None else None,
                "importance": _rounded_float(child_importance) if child_importance is not None else None,
                "retained": retained,
            }
        )
        if not retained:
            continue
        retained_moves.append(str(cand["move"]))
        children.append(
            OpeningNode(
                position=str(cand["child_position"]),
                moves=node.moves + (str(cand["move"]),),
                importance=float(child_importance),
            )
        )
    record = {
        "ply": len(node.moves),
        "moves": list(node.moves),
        "canonicalized_position": node.position,
        "importance": _rounded_float(node.importance),
        "candidates": candidate_rows,
        "retained_moves": retained_moves,
    }
    return record, children


def _build_output_payload(
    *,
    board_size: int,
    root_openings: tuple[str, ...],
    root_study: dict[str, Any] | None,
    nodes: list[dict[str, Any]],
    completed: bool,
    completed_ply: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "board_size": int(board_size),
        "root_openings": list(root_openings),
        "root_study": (root_study if isinstance(root_study, dict) else None),
        "completed": bool(completed),
        "completed_ply": int(completed_ply),
        "nodes": nodes,
    }


def _candidate_child_positions_for_record(record: dict[str, Any]) -> list[str]:
    position = str(record.get("canonicalized_position") or "").strip()
    if not position:
        return []
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        return []
    size, red, blue, to_play = lps._position_state(position)
    child_positions: list[str] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        move = str(row.get("move") or "").strip().lower()
        if not move:
            continue
        child_positions.append(
            lps._position_after_move_from_state(
                size=size,
                red=red,
                blue=blue,
                to_play=to_play,
                move=move,
            )
        )
    return child_positions


def _prune_raw_nn_cache(*, board_size: int, output_path: Path) -> tuple[Path, Path, int, int, int, int]:
    if not output_path.exists():
        raise FileNotFoundError(f"opening output not found for cache pruning: {output_path}")
    raw_nn_cache_path = _raw_nn_cache_path(board_size=int(board_size))
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError(f"invalid nodes payload in {output_path}")

    keep_positions: set[str] = set()
    missing_node_payloads = 0
    missing_child_payloads = 0
    for record in nodes:
        if not isinstance(record, dict):
            continue
        position = str(record.get("canonicalized_position") or "").strip()
        if position:
            keep_positions.add(position)
            if position != _empty_position(board_size=int(board_size)):
                if not isinstance(raw_nn_cache.get(lps._cache_key(position)), dict):
                    missing_node_payloads += 1
        for child_position in _candidate_child_positions_for_record(record):
            keep_positions.add(child_position)
            if not isinstance(raw_nn_cache.get(lps._cache_key(child_position)), dict):
                missing_child_payloads += 1

    before = len(raw_nn_cache)
    keep_keys = {lps._cache_key(position) for position in keep_positions}
    pruned = {
        key: raw_nn_cache[key]
        for key in raw_nn_cache
        if key in keep_keys or _is_special_raw_nn_cache_key(key)
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = raw_nn_cache_path.with_name(f"{raw_nn_cache_path.stem}.backup-{timestamp}{raw_nn_cache_path.suffix}")
    lps._save_raw_nn_cache(backup_path, raw_nn_cache)
    lps._save_raw_nn_cache(raw_nn_cache_path, pruned)
    return backup_path, raw_nn_cache_path, before, len(pruned), missing_node_payloads, missing_child_payloads


def build_opening_database(
    *,
    board_size: int = 11,
    output_path: Path | None = None,
    stop_after_ply: int | None = None,
) -> dict[str, Any]:
    board_size_i = int(board_size)
    started_at = time.time()
    raw_nn_cache_path = _raw_nn_cache_path(board_size=board_size_i)
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)

    frontier = [OpeningNode(position=_empty_position(board_size=board_size_i), moves=())]
    nodes: list[dict[str, Any]] = []
    completed_ply = 0
    root_study: dict[str, Any] | None = None
    root_openings: tuple[str, ...] = ()

    while frontier:
        depth_started_at = time.time()
        frontier_count = len(frontier)
        root_sweep_cache_hits = 0
        if any(not node.moves for node in frontier):
            sweep_payload, root_sweep_cache_hits = _run_fair_root_candidate_sweep_cached(
                board_size=board_size_i,
                raw_nn_cache=raw_nn_cache,
                raw_nn_cache_path=raw_nn_cache_path,
            )
            root_study = _derive_fair_root_study(board_size=board_size_i, sweep_payload=sweep_payload)
            root_openings = tuple(str(x) for x in list(root_study.get("root_openings") or []))
            if not root_openings:
                raise ValueError(f"no fair root openings derived for board size {board_size_i}")
        positions = [node.position for node in frontier if node.moves]
        position_cache_hits = lps._cached_request_count(raw_nn_cache, positions)
        root_payloads = _run_multi_position_analyze_cached(
            hexata_main=lps._hexata_main_path(),
            position_inputs=positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )
        prepared: list[tuple[OpeningNode, list[dict[str, Any]]]] = []
        child_positions: list[str] = []
        seen_child_positions: set[str] = set()
        for node in frontier:
            parent_state = lps._position_state(node.position)
            if not node.moves:
                candidates = _select_root_candidates(
                    node=node,
                    board_size=board_size_i,
                    root_openings=root_openings,
                    parent_state=parent_state,
                )
            else:
                payload = root_payloads.get(node.position)
                if not isinstance(payload, dict):
                    raise ValueError(f"missing root payload for {node.position!r}")
                candidates = _select_top_prior_candidates(
                    node=node,
                    payload=payload,
                    parent_state=parent_state,
                )
            if _can_skip_child_expansion(node=node, board_size=board_size_i):
                prepared.append((node, []))
                continue
            prepared.append((node, candidates))
            for cand in candidates:
                child_position = str(cand["child_position"])
                if child_position in seen_child_positions:
                    continue
                seen_child_positions.add(child_position)
                child_positions.append(child_position)
        child_cache_hits = lps._cached_request_count(raw_nn_cache, child_positions)
        cache_hits = position_cache_hits + child_cache_hits + root_sweep_cache_hits
        requested_positions = positions + child_positions
        cache_total = len(requested_positions) + (1 if any(not node.moves for node in frontier) else 0)
        if cache_hits < cache_total:
            _log(
                f"starting ply={max(len(node.moves) for node in frontier)} "
                f"nodes={frontier_count} cache={cache_hits}/{cache_total}",
                board_size=board_size_i,
            )
        child_payloads = _run_multi_position_analyze_cached(
            hexata_main=lps._hexata_main_path(),
            position_inputs=child_positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )
        next_frontier: list[OpeningNode] = []
        for node, candidates in prepared:
            record, children = _finalize_node(
                node,
                candidates=candidates,
                child_payloads=child_payloads,
                root_study=root_study,
            )
            nodes.append(record)
            for child in children:
                next_frontier.append(child)
        completed_ply = max(len(node.moves) for node in frontier)
        frontier = next_frontier
        if isinstance(output_path, Path):
            lps._write_json(
                output_path,
                _build_output_payload(
                    board_size=board_size_i,
                    root_openings=root_openings,
                    root_study=root_study,
                    nodes=nodes,
                    completed=(len(frontier) == 0),
                    completed_ply=completed_ply,
                ),
            )
        _log(
            f"ply={completed_ply} nodes={frontier_count} next={len(frontier)} "
            f"cache={cache_hits}/{cache_total} "
            f"elapsed={_fmt_s(max(0.0, time.time() - depth_started_at))}",
            board_size=board_size_i,
        )
        if isinstance(stop_after_ply, int) and completed_ply >= int(stop_after_ply):
            break

    payload = _build_output_payload(
        board_size=board_size_i,
        root_openings=root_openings,
        root_study=root_study,
        nodes=nodes,
        completed=(len(frontier) == 0),
        completed_ply=completed_ply,
    )
    status_verb = "Finished" if len(frontier) == 0 else "Stopped"
    _log(
        f"{status_verb} opening build size={board_size_i} nodes={len(nodes)} "
        f"elapsed={_fmt_s(max(0.0, time.time() - started_at))}",
        board_size=board_size_i,
    )
    return payload


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a raw-NN opening database with a derived fair-opening root phase")
    ap.add_argument("--board-size", type=int, default=11)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stop-after-ply", type=int, default=None)
    ap.add_argument("--prune-cache", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    board_size = int(args.board_size)
    out_path = Path(str(args.out)) if args.out else _default_output_path(board_size=board_size)
    if bool(args.prune_cache):
        backup_path, cache_path, before, after, missing_nodes, missing_children = _prune_raw_nn_cache(
            board_size=board_size,
            output_path=out_path,
        )
        print(f"{cache_path} {before}->{after} backup={backup_path}")
        if missing_nodes or missing_children:
            print(
                f"missing_node_payloads={missing_nodes} "
                f"missing_child_payloads={missing_children}"
            )
        return 0
    payload = build_opening_database(
        board_size=board_size,
        output_path=out_path,
        stop_after_ply=args.stop_after_ply,
    )
    lps._write_json(out_path, payload)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
