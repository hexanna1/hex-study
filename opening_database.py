#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import local_pattern_representative as lpr
import study_common as lps


TOP_K_BY_PLY: dict[int, int] = {
    1: 6,
    2: 5,
    3: 4,
    4: 3,
}
DEFAULT_TOP_K = 2
DEFAULT_SMALL_BOARD_IMPORTANCE_MIN = 0.83
DEFAULT_LARGE_BOARD_IMPORTANCE_MIN = 0.87
IMPORTANCE_MIN_BY_BOARD_SIZE: dict[int, float] = {
    13: 0.84,
    14: 0.857,
}
PLY_DECAY = 0.994
EXTRA_CANDIDATE_PRIOR_MIN = 0.15
OUTSIDE_TOP_K_PRIOR_LOG_STEP = 0.065
OUTSIDE_TOP_K_EXPONENT_RANK_STEP = 0.018
OUTSIDE_TOP_K_EXPONENT_PLY_STEP = 0.024
OPENING_ROOT_IMPORTANCE_OVERRIDES: list[tuple[int, str, float]] = [
    (11, "a6", 0.94),
    (11, "a8", 0.93),
    (11, "a9", 0.95),
    (11, "a11", 0.96),
    (11, "c2", 0.99),
    (11, "i2", 0.96),
    (12, "b4", 0.94),
    (12, "c2", 0.98),
    (12, "j2", 1.00),
    (13, "a10", 0.94),
    (13, "b4", 0.96),
    (13, "c2", 0.98),
    (13, "f3", 0.95),
    (13, "g3", 1.00),
    (13, "h3", 0.96),
    (14, "a6", 0.96),
    (14, "a9", 0.96),
    (14, "a14", 0.97),
    (14, "b4", 0.98),
    (14, "c2", 0.98),
    (14, "f3", 0.98),
    (14, "g3", 0.98),
    (14, "h3", 0.99),
    (14, "i3", 0.98),
    (17, "a10", 0.96),
    (17, "a13", 0.96),
    (17, "a14", 0.96),
    (17, "a17", 0.96),
    (17, "b4", 0.97),
    (17, "b15", 0.94),
    (17, "c2", 0.96),
    (17, "e3", 0.97),
    (17, "k3", 0.97),
    (17, "l3", 0.97),
]
WINRATE_EPS = 1e-6
RAW_NN_CACHE_CHUNK_SIZE = 30000
RAW_NN_CACHE_MOVE_LIMIT = 24
RAW_NN_CACHE_SPECIAL_PREFIXES = ("fair-root-candidate::",)
FAIR_ROOT_CACHE_VERSION = 1
FAIR_REFERENCE_STONE_FRACTION = 0.75
FAIR_STONE_FRACTION_MIN = 0.36
FAIR_STONE_FRACTION_MAX = 0.64
OPENING_RETAINED_CANDIDATE_FIELDS = [
    "move",
    "rank",
    "prior",
    "stone_fraction",
    "candidate_weight",
    "importance",
    "child",
    "tree_mover_winrate",
]
OPENING_NONRETAINED_CANDIDATE_FIELDS = ["move", "importance"]


PositionState = tuple[int, set[tuple[int, int]], set[tuple[int, int]], str]


def _ply_decay() -> float:
    return float(PLY_DECAY)


def _importance_min(*, board_size: int) -> float:
    size = int(board_size)
    if size > 14:
        return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_LARGE_BOARD_IMPORTANCE_MIN))
    return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_SMALL_BOARD_IMPORTANCE_MIN))


def _extra_candidate_prior_min() -> float:
    return float(EXTRA_CANDIDATE_PRIOR_MIN)


def _outside_top_k_prior_log_step() -> float:
    return float(OUTSIDE_TOP_K_PRIOR_LOG_STEP)


def _outside_top_k_exponent_rank_step() -> float:
    return float(OUTSIDE_TOP_K_EXPONENT_RANK_STEP)


def _outside_top_k_exponent_ply_step() -> float:
    return float(OUTSIDE_TOP_K_EXPONENT_PLY_STEP)


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


def _run_multi_position_raw_nn_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    include_moves: bool = True,
) -> tuple[dict[str, dict[str, Any]], int]:
    encoded_payloads, cache_hits = _ensure_raw_nn_cached(
        position_inputs=position_inputs,
        raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path,
        require_moves=include_moves,
    )
    return (
        {
            position: lps._decode_compact_raw_nn_payload(payload, include_moves=include_moves)
            for position, payload in encoded_payloads.items()
        },
        cache_hits,
    )


def _ensure_raw_nn_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    require_moves: bool = True,
) -> tuple[dict[str, dict[str, Any]], int]:
    board_sizes = {
        lps._extract_board_size_from_input(position)
        for position in position_inputs
        if str(position).strip()
    }
    if not board_sizes:
        return {}, 0
    if len(board_sizes) != 1 or None in board_sizes:
        raise ValueError(f"Native raw-NN batch requires one known board size, got {sorted(board_sizes, key=str)!r}")
    return lps._ensure_raw_nn_cache_entries(
        position_inputs=position_inputs,
        raw_nn_cache=raw_nn_cache,
        board_size=int(next(iter(board_sizes))),
        raw_nn_cache_path=raw_nn_cache_path,
        chunk_size=RAW_NN_CACHE_CHUNK_SIZE,
        move_limit=RAW_NN_CACHE_MOVE_LIMIT,
        require_moves=require_moves,
        precanonicalized_position_inputs=True,
    )


def _log(message: str, *, board_size: int | None = None) -> None:
    prefix = ""
    if isinstance(board_size, int):
        prefix = f"[{board_size}] "
    lps._log(f"{prefix}{message}")


def _empty_position(*, board_size: int) -> str:
    return lpr.serialize_position(
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
    red_wr = lps._cached_payload_red_winrate(child_payload)
    if not isinstance(red_wr, float):
        raise ValueError("child payload missing cached red winrate")
    if str(parent_to_play).strip().lower() == "red":
        return red_wr
    return 1.0 - red_wr


def _red_winrate_from_mover_winrate(*, mover_winrate: float, parent_ply: int) -> float:
    mover_wr = float(mover_winrate)
    if int(parent_ply) % 2 == 0:
        return mover_wr
    return 1.0 - mover_wr


def _mover_winrate_from_red_winrate(*, red_winrate: float, parent_ply: int) -> float:
    red_wr = float(red_winrate)
    if int(parent_ply) % 2 == 0:
        return red_wr
    return 1.0 - red_wr


def _winrate_to_elo(winrate: float) -> float:
    p = max(WINRATE_EPS, min(1.0 - WINRATE_EPS, float(winrate)))
    return 400.0 * math.log10(p / (1.0 - p))


def _rounded_float(value: Any, *, digits: int = 6) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _normalize_tree_move(raw: Any) -> str | None:
    if raw is None:
        return None
    move = str(raw or "").strip().lower()
    return move or None


def _build_tree_node(
    *,
    record: dict[str, Any],
    parent: int | None,
    move: str | None,
    child_by_move: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"bad move-tree record: {record!r}")
    node: dict[str, Any] = {
        "parent": (int(parent) if isinstance(parent, int) else None),
        "move": _normalize_tree_move(move),
    }
    for key, value in record.items():
        if key in {"moves", "candidates", "canonicalized_position", "retained_moves"}:
            continue
        node[str(key)] = value
    candidates_raw = record.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError(f"node missing candidates list: {record!r}")
    child_lookup = child_by_move or {}
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates_raw:
        if not isinstance(row, dict):
            raise ValueError(f"bad candidate row: {row!r}")
        row_move = _normalize_tree_move(row.get("move"))
        retained = bool(row.get("retained"))
        next_row = {
            str(key): value
            for key, value in row.items()
            if str(key) != "child"
        }
        next_row["move"] = row_move
        next_row["retained"] = retained
        next_row["child"] = (
            int(child_lookup[row_move])
            if retained and row_move in child_lookup and isinstance(child_lookup[row_move], int)
            else None
        )
        candidate_rows.append(next_row)
    node["candidates"] = candidate_rows
    return node


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
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_move = str(row.get("move") or "").strip().lower()
        stone_fraction = row.get("stone_fraction")
        if not row_move or not isinstance(stone_fraction, (int, float)):
            continue
        by_move[row_move] = float(stone_fraction)
    if not by_move:
        raise ValueError("root_study rows missing stone-fraction calibration")
    move_s = str(move).strip().lower()
    if move_s not in by_move:
        raise ValueError(f"root_study missing root move calibration: {move_s!r}")
    distance = abs(float(by_move[move_s]) - 0.5)
    raw_stone_fraction = max(0.0, min(1.0, 1.0 - distance))
    return math.sqrt(raw_stone_fraction)


def _root_importance_override(*, board_size: int, move: str) -> float | None:
    size = int(board_size)
    move_s = str(move).strip().lower()
    for rule_board_size, rule_move, importance in OPENING_ROOT_IMPORTANCE_OVERRIDES:
        if int(rule_board_size) != size:
            continue
        if str(rule_move).strip().lower() != move_s:
            continue
        return float(importance)
    return None


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


def _fair_root_sweep_payload_from_child_raw_nn(
    *,
    root_position: str,
    requested_moves: tuple[str, ...],
    child_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    red_winrate_rows: list[list[Any]] = []
    for move in requested_moves:
        child_position = lps._position_after_move(root_position, move)
        payload = child_payloads.get(child_position)
        red_wr = lps._cached_payload_red_winrate(payload) if isinstance(payload, dict) else None
        if not isinstance(red_wr, (int, float)):
            raise ValueError(f"fair-root raw-NN payload missing root winrate for move {move!r}")
        red_winrate_rows.append([move, float(red_wr)])
    return {"m": red_winrate_rows}


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
    if lps._is_valid_encoded_compact_raw_nn_payload(cached):
        cached = lps._decode_compact_raw_nn_payload(cached)
        cached_red_winrate_rows = cached.get("m")
        if isinstance(cached_red_winrate_rows, list):
            returned = {
                str(row[0] or "").strip().lower()
                for row in cached_red_winrate_rows
                if isinstance(row, list)
                and len(row) >= 2
                and str(row[0] or "").strip()
                and isinstance(row[1], (int, float))
            }
            if all(move in returned for move in requested_moves):
                return cached, 1
    root_position = _empty_position(board_size=board_size)
    child_positions = [lps._position_after_move(root_position, move) for move in requested_moves]
    child_payloads, _ = _run_multi_position_raw_nn_cached(
        position_inputs=child_positions,
        raw_nn_cache=raw_nn_cache,
        include_moves=False,
    )
    reduced = _fair_root_sweep_payload_from_child_raw_nn(
        root_position=root_position,
        requested_moves=requested_moves,
        child_payloads=child_payloads,
    )
    raw_nn_cache[cache_key] = lps._encode_compact_raw_nn_payload(reduced)
    if isinstance(raw_nn_cache_path, Path):
        lps._save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
    return reduced, 0


def _derive_fair_root_study(
    *,
    board_size: int,
    sweep_payload: dict[str, Any],
) -> dict[str, Any]:
    red_winrate_rows = sweep_payload.get("m")
    if not isinstance(red_winrate_rows, list):
        raise ValueError("fair-root sweep payload missing moves list")
    rows_by_move = {
        str(row[0] or "").strip().lower(): float(row[1])
        for row in red_winrate_rows
        if isinstance(row, list)
        and len(row) >= 2
        and str(row[0] or "").strip()
        and isinstance(row[1], (int, float))
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


def _run_opening_expansion(
    *,
    nodes: list[OpeningNode],
    payloads: dict[str, dict[str, Any]],
    board_size: int,
) -> list[list[dict[str, Any]]]:
    if not nodes:
        return []
    plies = {len(node.moves) for node in nodes}
    if len(plies) != 1:
        raise ValueError(f"Opening expansion requires one ply, got {sorted(plies)!r}")
    ply = next(iter(plies))
    config = [
        "opening-v1",
        str(ply),
        str(int(board_size)),
        str(_top_k_for_ply(ply)),
        repr(_importance_min(board_size=int(board_size))),
        repr(_ply_decay()),
        repr(_extra_candidate_prior_min()),
        repr(_outside_top_k_prior_log_step()),
        repr(_outside_top_k_exponent_rank_step()),
        repr(_outside_top_k_exponent_ply_step()),
    ]
    input_lines = ["\t".join(config)]
    for node in nodes:
        payload = payloads.get(node.position)
        if not isinstance(payload, dict):
            raise ValueError(f"missing root payload for {node.position!r}")
        policy: list[str] = []
        for row in lps._cached_payload_moves(payload):
            if not isinstance(row, list) or len(row) < 2:
                continue
            move = str(row[0] or "").strip().lower()
            prior = row[1]
            if not move:
                continue
            if not isinstance(prior, int) or isinstance(prior, bool):
                continue
            policy.append(f"{move},{prior}")
        input_lines.append(f"{node.position}\t{node.importance!r}\t{';'.join(policy)}")
    output_lines = lps._run_position_expansion(input_lines, expected_rows=len(nodes))
    expanded: list[list[dict[str, Any]]] = []
    for line in output_lines:
        candidates: list[dict[str, Any]] = []
        for encoded in line.split("\t") if line else []:
            fields = encoded.split("|", 5)
            if len(fields) != 6:
                raise ValueError(f"Bad opening expansion candidate: {encoded!r}")
            move, rank, prior, cleaned_rank, parent_to_play, child_position = fields
            candidates.append(
                {
                    "move": move,
                    "rank": int(rank),
                    "prior": lps._decode_millionths(int(prior)),
                    "cleaned_rank": int(cleaned_rank),
                    "child_position": child_position,
                    "parent_to_play": parent_to_play,
                    "board_size": int(board_size),
                }
            )
        expanded.append(candidates)
    return expanded


def _candidate_importance_weight(*, node: OpeningNode, cand: dict[str, Any]) -> float:
    if not node.moves:
        return 1.0
    cleaned_rank = cand.get("cleaned_rank")
    if not isinstance(cleaned_rank, int) or int(cleaned_rank) <= 0:
        raise ValueError(f"candidate missing cleaned_rank: {cand!r}")
    top_k = _top_k_for_ply(len(node.moves))
    if int(cleaned_rank) <= top_k:
        return 1.0
    prior = cand.get("prior")
    if isinstance(prior, (int, float)) and float(prior) >= _extra_candidate_prior_min():
        return 1.0
    prior_log10 = -math.log10(max(1e-6, float(prior or 0.0)))
    board_size = int(cand["board_size"])
    rank_delta = int(cleaned_rank) - int(top_k)
    exponent = (
        (_outside_top_k_prior_log_step() * prior_log10)
        + (_outside_top_k_exponent_rank_step() * max(0, rank_delta - 1))
        + (_outside_top_k_exponent_ply_step() * max(0, len(node.moves) - 1))
    )
    return _importance_min(board_size=board_size) ** float(exponent)


def _candidate_sets_elo_baseline(*, node: OpeningNode, cand: dict[str, Any]) -> bool:
    return _candidate_importance_weight(node=node, cand=cand) >= 1.0


def _can_skip_child_expansion(*, node: OpeningNode, board_size: int) -> bool:
    return float(node.importance) * _ply_decay() < _importance_min(board_size=int(board_size))


def _finalize_node(
    node: OpeningNode,
    *,
    candidates: list[dict[str, Any]],
    child_payloads: dict[str, dict[str, Any]],
    root_study: dict[str, Any],
) -> tuple[dict[str, Any], list[OpeningNode]]:
    evaluated: list[dict[str, Any]] = []
    best_anchor_elo: float | None = None
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
        if _candidate_sets_elo_baseline(node=node, cand=cand):
            best_anchor_elo = elo if best_anchor_elo is None else max(best_anchor_elo, elo)
        evaluated.append(
            {
                **cand,
                "mover_winrate": mover_wr,
                "_elo": elo,
            }
        )
    if best_anchor_elo is None and evaluated:
        best_anchor_elo = max(float(cand["_elo"]) for cand in evaluated)
    children: list[OpeningNode] = []
    candidate_rows: list[dict[str, Any]] = []
    retained_moves: list[str] = []
    for cand in evaluated:
        elo_loss = (
            max(0.0, float(best_anchor_elo - cand["_elo"]))
            if best_anchor_elo is not None
            else None
        )
        if not node.moves:
            stone_fraction = _root_importance_override(
                board_size=int(cand["board_size"]),
                move=str(cand["move"]),
            )
            if stone_fraction is None:
                stone_fraction = _root_stone_fraction_from_study(move=str(cand["move"]), root_study=root_study)
        elif elo_loss is not None:
            stone_fraction = _stone_fraction_from_elo_loss(elo_loss=elo_loss, full_stone_elo=full_stone_elo)
        else:
            stone_fraction = None
        candidate_weight = _candidate_importance_weight(node=node, cand=cand)
        child_importance = (
            float(node.importance)
            * float(stone_fraction)
            * _ply_decay()
            * float(candidate_weight)
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
                "raw_mover_winrate": _rounded_float(cand["mover_winrate"]),
                "stone_fraction": _rounded_float(stone_fraction) if stone_fraction is not None else None,
                "candidate_weight": _rounded_float(candidate_weight),
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


def _apply_prior_weighted_tree_values(*, nodes: list[dict[str, Any]]) -> None:
    node_raw_red_winrates: list[float | None] = [None] * len(nodes)
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        ply = int(node.get("ply") or 0)
        candidates = node.get("candidates")
        if not isinstance(candidates, list):
            continue
        for row in candidates:
            if not isinstance(row, dict):
                continue
            raw_mover_winrate = row.get("raw_mover_winrate")
            if not isinstance(raw_mover_winrate, (int, float)):
                continue
            row["tree_mover_winrate"] = _rounded_float(raw_mover_winrate)
            child = row.get("child")
            if not isinstance(child, int) or child < 0 or child >= len(nodes):
                continue
            node_raw_red_winrates[child] = _red_winrate_from_mover_winrate(
                mover_winrate=float(raw_mover_winrate),
                parent_ply=ply,
            )

    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        if not isinstance(node, dict):
            continue
        candidates = node.get("candidates")
        if not isinstance(candidates, list):
            node["tree_red_winrate"] = _rounded_float(node_raw_red_winrates[idx])
            continue
        weighted_rows: list[tuple[dict[str, Any], float, float]] = []
        for row in candidates:
            if not isinstance(row, dict) or not bool(row.get("retained")):
                continue
            child_tree_red: float | None = None
            child = row.get("child")
            if isinstance(child, int) and 0 <= child < len(nodes):
                child_red = nodes[child].get("tree_red_winrate")
                if isinstance(child_red, (int, float)):
                    child_tree_red = float(child_red)
            if child_tree_red is None:
                raw_mover_winrate = row.get("raw_mover_winrate")
                if isinstance(raw_mover_winrate, (int, float)):
                    child_tree_red = _red_winrate_from_mover_winrate(
                        mover_winrate=float(raw_mover_winrate),
                        parent_ply=int(node.get("ply") or 0),
                    )
            if child_tree_red is None:
                continue
            row["tree_mover_winrate"] = _rounded_float(
                _mover_winrate_from_red_winrate(
                    red_winrate=child_tree_red,
                    parent_ply=int(node.get("ply") or 0),
                )
            )
            prior = row.get("prior")
            if not isinstance(prior, (int, float)):
                continue
            weighted_rows.append((row, float(prior), child_tree_red))
        if weighted_rows:
            total_prior = sum(prior for _, prior, _ in weighted_rows)
            if total_prior > 0.0:
                node["tree_red_winrate"] = _rounded_float(
                    sum(prior * red_winrate for _, prior, red_winrate in weighted_rows) / total_prior
                )
                continue
        node["tree_red_winrate"] = _rounded_float(node_raw_red_winrates[idx])


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
        "format": "move_tree",
        "mode": "openings",
        "board_size": int(board_size),
        "root": (0 if nodes else None),
        "root_openings": list(root_openings),
        "root_study": (root_study if isinstance(root_study, dict) else None),
        "completed": bool(completed),
        "completed_ply": int(completed_ply),
        "nodes": nodes,
    }


def _write_opening_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        key: value
        for key, value in payload.items()
        if key != "nodes"
    }
    artifact_nodes: list[dict[str, Any]] = []
    for node in list(payload.get("nodes") or []):
        if not isinstance(node, dict):
            raise ValueError(f"bad opening artifact node: {node!r}")
        artifact_node = dict(node)
        candidates = node.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"node missing candidates list: {node!r}")
        retained_candidates: list[dict[str, Any]] = []
        nonretained_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"bad opening artifact candidate: {candidate!r}")
            if bool(candidate.get("retained")):
                retained_candidates.append(
                    {
                        field: candidate.get(field)
                        for field in OPENING_RETAINED_CANDIDATE_FIELDS
                    }
                )
            else:
                nonretained_candidates.append(
                    {
                        field: candidate.get(field)
                        for field in OPENING_NONRETAINED_CANDIDATE_FIELDS
                    }
                )
        artifact_node["candidates"] = retained_candidates
        if nonretained_candidates:
            artifact_node["nonretained_candidates"] = nonretained_candidates
        else:
            artifact_node.pop("nonretained_candidates", None)
        artifact_nodes.append(artifact_node)
    artifact_payload["nodes"] = artifact_nodes
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(artifact_payload, ensure_ascii=True, indent=0, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _tree_node_positions(*, board_size: int, nodes: list[dict[str, Any]]) -> list[str | None]:
    positions: list[str | None] = [None] * len(nodes)
    root_position = _empty_position(board_size=int(board_size))
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        parent = node.get("parent")
        move = str(node.get("move") or "").strip().lower()
        if parent is None:
            positions[idx] = root_position
            continue
        if not isinstance(parent, int) or parent < 0 or parent >= len(nodes):
            raise ValueError(f"bad parent index in opening tree: {node!r}")
        parent_position = positions[parent]
        if not isinstance(parent_position, str):
            raise ValueError(f"missing parent position for opening tree node {idx}")
        size, red, blue, to_play = lps._position_state(parent_position)
        positions[idx] = lps._position_after_move_from_state(
            size=size,
            red=red,
            blue=blue,
            to_play=to_play,
            move=move,
        )
    return positions


def _artifact_candidate_move(
    raw: Any,
    *,
    output_path: Path,
    node_idx: int,
) -> str:
    if not isinstance(raw, dict):
        raise ValueError(f"bad opening candidate row at node {node_idx} in {output_path}: {raw!r}")
    move = str(raw.get("move") or "").strip().lower()
    if not move:
        raise ValueError(f"opening candidate missing move at node {node_idx} in {output_path}: {raw!r}")
    return move


def _prune_raw_nn_cache(*, board_size: int, output_path: Path) -> tuple[Path, Path, int, int, int, int]:
    if not output_path.exists():
        raise FileNotFoundError(f"opening output not found for cache pruning: {output_path}")
    raw_nn_cache_path = _raw_nn_cache_path(board_size=int(board_size))
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError(f"invalid nodes payload in {output_path}")
    positions = _tree_node_positions(board_size=int(board_size), nodes=nodes)

    keep_positions: set[str] = set()
    missing_node_payloads = 0
    missing_child_payloads = 0
    for idx, record in enumerate(nodes):
        if not isinstance(record, dict):
            continue
        position = positions[idx]
        if not isinstance(position, str):
            continue
        keep_positions.add(position)
        if position != _empty_position(board_size=int(board_size)):
            if not isinstance(raw_nn_cache.get(lps._cache_key(position)), dict):
                missing_node_payloads += 1
        retained_candidates = record.get("candidates", [])
        nonretained_candidates = record.get("nonretained_candidates", [])
        if retained_candidates is None:
            retained_candidates = []
        if nonretained_candidates is None:
            nonretained_candidates = []
        if not isinstance(retained_candidates, list) or not isinstance(nonretained_candidates, list):
            raise ValueError(f"invalid candidates payload in {output_path} at node {idx}")
        candidates = retained_candidates + nonretained_candidates
        size, red, blue, to_play = lps._position_state(position)
        for row in candidates:
            move = _artifact_candidate_move(
                row,
                output_path=output_path,
                node_idx=idx,
            )
            child_position = lps._position_after_move_from_state(
                size=size,
                red=red,
                blue=blue,
                to_play=to_play,
                move=move,
            )
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
    line_to_idx: dict[str, int] = {}
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
        expansion_nodes = [
            node
            for node in frontier
            if node.moves and not _can_skip_child_expansion(node=node, board_size=board_size_i)
        ]
        positions = [node.position for node in expansion_nodes]
        root_payloads, position_cache_hits = _ensure_raw_nn_cached(
            position_inputs=positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )
        expansion_candidates = iter(
            _run_opening_expansion(
                nodes=expansion_nodes,
                payloads=root_payloads,
                board_size=board_size_i,
            )
        )
        prepared: list[tuple[OpeningNode, list[dict[str, Any]]]] = []
        child_positions: list[str] = []
        seen_child_positions: set[str] = set()
        for node in frontier:
            if _can_skip_child_expansion(node=node, board_size=board_size_i):
                prepared.append((node, []))
                continue
            if not node.moves:
                parent_state = lps._position_state(node.position)
                candidates = _select_root_candidates(
                    node=node,
                    board_size=board_size_i,
                    root_openings=root_openings,
                    parent_state=parent_state,
                )
            else:
                candidates = next(expansion_candidates)
            prepared.append((node, candidates))
            for cand in candidates:
                child_position = str(cand["child_position"])
                if child_position in seen_child_positions:
                    continue
                seen_child_positions.add(child_position)
                child_positions.append(child_position)
        child_cache_hits = lps._cached_raw_nn_winrate_count(
            raw_nn_cache,
            child_positions,
        )
        cache_hits = position_cache_hits + child_cache_hits + root_sweep_cache_hits
        requested_positions = positions + child_positions
        cache_total = len(requested_positions) + (1 if any(not node.moves for node in frontier) else 0)
        if cache_hits < cache_total:
            _log(
                f"starting ply={max(len(node.moves) for node in frontier)} "
                f"nodes={frontier_count} cache={cache_hits}/{cache_total}",
                board_size=board_size_i,
            )
        child_payloads, _ = _run_multi_position_raw_nn_cached(
            position_inputs=child_positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
            include_moves=False,
        )
        next_frontier: list[OpeningNode] = []
        current_ply_start = len(nodes)
        for node_index, (node, candidates) in enumerate(prepared):
            record, children = _finalize_node(
                node,
                candidates=candidates,
                child_payloads=child_payloads,
                root_study=root_study,
            )
            child_id_base = current_ply_start + frontier_count + len(next_frontier)
            child_by_move = {
                str(child.moves[-1]): (child_id_base + idx)
                for idx, child in enumerate(children)
                if child.moves
            }
            line = "".join(node.moves)
            parent_idx = None if not node.moves else line_to_idx.get("".join(node.moves[:-1]))
            if node.moves and not isinstance(parent_idx, int):
                raise ValueError(f"missing parent index for opening node {line!r}")
            tree_node = _build_tree_node(
                record=record,
                parent=parent_idx,
                move=(str(node.moves[-1]) if node.moves else None),
                child_by_move=child_by_move,
            )
            current_idx = current_ply_start + node_index
            nodes.append(tree_node)
            line_to_idx[line] = current_idx
            for child in children:
                next_frontier.append(child)
        completed_ply = max(len(node.moves) for node in frontier)
        frontier = next_frontier
        _apply_prior_weighted_tree_values(nodes=nodes)
        if isinstance(output_path, Path):
            _write_opening_artifact(
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
            f"ply={completed_ply} nodes={frontier_count}->{len(frontier)} "
            f"cache={cache_hits}/{cache_total} "
            f"elapsed={lps._fmt_s(max(0.0, time.time() - depth_started_at))}",
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
        f"elapsed={lps._fmt_s(max(0.0, time.time() - started_at))}",
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
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        # The build creates many short-lived acyclic objects alongside a large cache;
        # reference counting handles them without cyclic-GC scans of the cache.
        gc.disable()
    try:
        payload = build_opening_database(
            board_size=board_size,
            output_path=out_path,
            stop_after_ply=args.stop_after_ply,
        )
    finally:
        if gc_was_enabled:
            gc.enable()
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
