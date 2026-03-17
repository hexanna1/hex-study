#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import artifact_json as aj
import local_pattern_representative as lpr
import study_common as lps


TOP_K_BY_PLY: dict[int, int] = {
    1: 4,
    2: 3,
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
OUTSIDE_TOP_K_PRIOR_LOG_STEP = 0.06
OUTSIDE_TOP_K_EXPONENT_RANK_STEP = 0.012
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
    (12, "j2", 0.99),
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
    (14, "f3", 0.99),
    (14, "g3", 0.98),
    (14, "h3", 0.99),
    (14, "i3", 0.98),
    (17, "a10", 0.96),
    (17, "a12", 0.96),
    (17, "a13", 0.96),
    (17, "a14", 0.96),
    (17, "a17", 0.97),
    (17, "b4", 0.97),
    (17, "b15", 0.94),
    (17, "c2", 0.96),
    (17, "e3", 0.97),
    (17, "k3", 0.97),
    (17, "l3", 0.97),
]
WINRATE_EPS = 1e-6
RAW_NN_CACHE_CHUNK_SIZE = 30000
RAW_NN_CACHE_MOVE_LIMIT = 36
POLICY_IMPORTANCE_HEADROOM = 1.04
POLICY_TOP_K_HEADROOM = 1
FAIR_ROOT_CACHE_KEY_PREFIX = "fair-root-candidate::s"
FAIR_REFERENCE_STONE_FRACTION = 0.75
FAIR_STONE_FRACTION_MIN = 0.36
FAIR_STONE_FRACTION_MAX = 0.64
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


@dataclass(slots=True)
class OpeningNode:
    position: str
    ply: int
    importance: float = 1.0
    parent: int | None = None
    move: str | None = None


@dataclass(frozen=True, slots=True)
class OpeningPolicyProof:
    raw_rows: int
    cleaned_rank: int


def _default_output_path(*, board_size: int) -> Path:
    return Path("artifacts") / "openings" / f"openings-s{int(board_size)}.json"


def _raw_nn_cache_path(*, board_size: int) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "openings" / f"openings_raw_nn_cache_s{int(board_size)}.json"


def _is_special_raw_nn_cache_key(key: str) -> bool:
    key_s = str(key)
    return key_s.startswith(FAIR_ROOT_CACHE_KEY_PREFIX) and key_s.removeprefix(
        FAIR_ROOT_CACHE_KEY_PREFIX
    ).isdigit()


def _run_multi_position_raw_nn_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    include_moves: bool = True,
    store_moves: bool = True,
    policy_payloads_out: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    encoded_payloads, cache_hits = _ensure_raw_nn_cached(
        position_inputs=position_inputs,
        raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path,
        require_moves=include_moves,
        store_moves=store_moves,
    )
    if isinstance(policy_payloads_out, dict):
        policy_payloads_out.update(
            (position, payload)
            for position, payload in encoded_payloads.items()
            if lps._is_valid_encoded_raw_nn_policy(payload)
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
    store_moves: bool = True,
    policy_validator: Callable[[str, dict[str, Any]], bool] | None = None,
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
        store_moves=store_moves,
        precanonicalized_position_inputs=True,
        policy_validator=policy_validator,
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


def _normalize_opening_move(raw: Any) -> str | None:
    if raw is None:
        return None
    move = str(raw or "").strip().lower()
    return move or None


def _build_opening_node(
    *,
    record: dict[str, Any],
    parent: int | None,
    move: str | None,
    child_by_move: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"bad opening record: {record!r}")
    node: dict[str, Any] = {
        "parent": (int(parent) if isinstance(parent, int) else None),
        "move": _normalize_opening_move(move),
    }
    for key, value in record.items():
        if key in {"candidates", "canonicalized_position", "retained_moves"}:
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
        row_move = _normalize_opening_move(row.get("move"))
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
    return f"{FAIR_ROOT_CACHE_KEY_PREFIX}{int(board_size)}"


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
) -> tuple[list[list[dict[str, Any]]], list[OpeningPolicyProof | None]]:
    if not nodes:
        return [], []
    plies = {node.ply for node in nodes}
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
        repr(POLICY_IMPORTANCE_HEADROOM),
        str(POLICY_TOP_K_HEADROOM),
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
    proofs: list[OpeningPolicyProof | None] = []
    for line in output_lines:
        encoded_rows = line.split("\t") if line else []
        if not encoded_rows or not encoded_rows[-1].startswith("@|"):
            raise ValueError(f"Opening expansion missing policy proof: {line!r}")
        proof_fields = encoded_rows.pop().split("|")
        if len(proof_fields) != 3:
            raise ValueError(f"Bad opening policy proof: {proof_fields!r}")
        proof_raw_rows = int(proof_fields[1])
        proof_cleaned_rank = int(proof_fields[2])
        if (proof_raw_rows == 0) != (proof_cleaned_rank == 0):
            raise ValueError(f"Inconsistent opening policy proof: {proof_fields!r}")
        proofs.append(
            None
            if proof_raw_rows == 0
            else OpeningPolicyProof(
                raw_rows=proof_raw_rows,
                cleaned_rank=proof_cleaned_rank,
            )
        )
        candidates: list[dict[str, Any]] = []
        for encoded in encoded_rows:
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
    return expanded, proofs


def _candidate_importance_weight_for_rank(
    *,
    node: OpeningNode,
    board_size: int,
    cleaned_rank: int,
    prior: float,
    top_k: int,
) -> float:
    if int(cleaned_rank) <= int(top_k):
        return 1.0
    if float(prior) >= _extra_candidate_prior_min():
        return 1.0
    prior_log10 = -math.log10(max(1e-6, float(prior)))
    rank_delta = int(cleaned_rank) - int(top_k)
    exponent = (
        (_outside_top_k_prior_log_step() * prior_log10)
        + (_outside_top_k_exponent_rank_step() * max(0, rank_delta - 1))
        + (_outside_top_k_exponent_ply_step() * max(0, node.ply - 1))
    )
    return _importance_min(board_size=int(board_size)) ** float(exponent)


def _candidate_importance_weight(*, node: OpeningNode, cand: dict[str, Any]) -> float:
    if node.ply == 0:
        return 1.0
    cleaned_rank = cand.get("cleaned_rank")
    if not isinstance(cleaned_rank, int) or int(cleaned_rank) <= 0:
        raise ValueError(f"candidate missing cleaned_rank: {cand!r}")
    prior = cand.get("prior")
    if not isinstance(prior, (int, float)):
        raise ValueError(f"candidate missing prior: {cand!r}")
    return _candidate_importance_weight_for_rank(
        node=node,
        board_size=int(cand["board_size"]),
        cleaned_rank=int(cleaned_rank),
        prior=float(prior),
        top_k=_top_k_for_ply(node.ply),
    )


def _opening_policy_required_moves(*, node: OpeningNode, board_size: int) -> int:
    occupied_count = node.ply
    return min(
        RAW_NN_CACHE_MOVE_LIMIT,
        max(0, (int(board_size) * int(board_size)) - occupied_count),
    )


def _opening_policy_coverage_rows(
    *,
    node: OpeningNode,
    board_size: int,
    payload: dict[str, Any],
) -> int | None:
    required_moves = _opening_policy_required_moves(
        node=node,
        board_size=int(board_size),
    )
    if required_moves == 0:
        return 0
    covered_moves = 0
    for raw_rows, row in enumerate(lps._cached_payload_moves(payload), start=1):
        if str(row[0] or "").strip().lower() != "pass":
            covered_moves += 1
        if covered_moves >= required_moves:
            return raw_rows
    return None


def _opening_policy_certificate_is_sufficient(
    *,
    node: OpeningNode,
    board_size: int,
    payload: dict[str, Any],
) -> bool:
    moves = lps._cached_payload_moves(payload)
    if _opening_policy_coverage_rows(
        node=node,
        board_size=int(board_size),
        payload=payload,
    ) is not None:
        return True
    cleaned_rank = payload.get("c")
    if (
        isinstance(cleaned_rank, bool)
        or not isinstance(cleaned_rank, int)
        or cleaned_rank <= 0
        or cleaned_rank > len(moves)
    ):
        return False
    if not moves or not isinstance(moves[-1], list) or len(moves[-1]) < 2:
        return False
    prior = moves[-1][1]
    if isinstance(prior, bool) or not isinstance(prior, int):
        return False
    candidate_weight = _candidate_importance_weight_for_rank(
        node=node,
        board_size=int(board_size),
        cleaned_rank=cleaned_rank,
        prior=lps._decode_millionths(prior),
        top_k=_top_k_for_ply(node.ply),
    )
    upper_bound = float(node.importance) * _ply_decay() * candidate_weight
    return upper_bound < _importance_min(board_size=int(board_size))


def _ensure_opening_policy_cached(
    *,
    nodes: list[OpeningNode],
    board_size: int,
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    nodes_by_position: dict[str, list[OpeningNode]] = {}
    for node in nodes:
        nodes_by_position.setdefault(node.position, []).append(node)
    validated: dict[str, bool] = {}

    def validator(position: str, payload: dict[str, Any]) -> bool:
        if position not in validated:
            validated[position] = all(
                _opening_policy_certificate_is_sufficient(
                    node=node,
                    board_size=int(board_size),
                    payload=payload,
                )
                for node in nodes_by_position[position]
            )
        return validated[position]

    payloads, cache_hits = _ensure_raw_nn_cached(
        position_inputs=[node.position for node in nodes],
        raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path,
        policy_validator=validator,
    )
    for position, payload in list(payloads.items()):
        coverage_rows = _opening_policy_coverage_rows(
            node=nodes_by_position[position][0],
            board_size=int(board_size),
            payload=payload,
        )
        moves = lps._cached_payload_moves(payload)
        if coverage_rows is not None and len(moves) > coverage_rows:
            payloads[position] = {
                "r": payload["r"],
                "m": moves[:coverage_rows],
            }
    return payloads, cache_hits


def _store_opening_policy_proofs(
    *,
    nodes: list[OpeningNode],
    payloads: dict[str, dict[str, Any]],
    proofs: list[OpeningPolicyProof | None],
    board_size: int,
    raw_nn_cache: dict[str, dict[str, Any]],
) -> bool:
    if len(nodes) != len(proofs):
        raise ValueError("opening policy proof count does not match node count")
    rows_by_position: dict[str, list[tuple[OpeningNode, OpeningPolicyProof | None]]] = {}
    for node, proof in zip(nodes, proofs):
        rows_by_position.setdefault(node.position, []).append((node, proof))
    changed = False
    for position, rows in rows_by_position.items():
        payload = payloads.get(position)
        if not lps._is_valid_encoded_raw_nn_policy(payload):
            raise ValueError(f"opening policy payload missing for {position!r}")
        moves = lps._cached_payload_moves(payload)
        row_proofs = [proof for _node, proof in rows]
        if any(proof is None for proof in row_proofs):
            coverage_rows = _opening_policy_coverage_rows(
                node=rows[0][0],
                board_size=int(board_size),
                payload=payload,
            )
            if coverage_rows is not None:
                compact: dict[str, Any] = {
                    "r": payload["r"],
                    "m": moves[:coverage_rows],
                }
            elif all(
                _opening_policy_certificate_is_sufficient(
                    node=node,
                    board_size=int(board_size),
                    payload=payload,
                )
                for node, _proof in rows
            ):
                compact = payload
            else:
                raise ValueError(f"opening policy payload has insufficient coverage for {position!r}")
        else:
            proof = max(
                (value for value in row_proofs if value is not None),
                key=lambda value: value.raw_rows,
            )
            if proof.raw_rows <= 0 or proof.raw_rows > len(moves):
                raise ValueError(f"bad opening policy proof boundary for {position!r}")
            compact = {
                "r": payload["r"],
                "m": moves[:proof.raw_rows],
                "c": proof.cleaned_rank,
            }
        key = lps._precanonicalized_position_cache_key(position)
        if raw_nn_cache.get(key) != compact:
            raw_nn_cache[key] = compact
            changed = True
    return changed


def _run_opening_expansion_and_store_policy(
    *,
    nodes: list[OpeningNode],
    payloads: dict[str, dict[str, Any]],
    board_size: int,
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path,
) -> list[list[dict[str, Any]]]:
    candidates, proofs = _run_opening_expansion(
        nodes=nodes,
        payloads=payloads,
        board_size=int(board_size),
    )
    if _store_opening_policy_proofs(
        nodes=nodes,
        payloads=payloads,
        proofs=proofs,
        board_size=int(board_size),
        raw_nn_cache=raw_nn_cache,
    ):
        lps._save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
    return candidates


def _candidate_sets_elo_baseline(*, node: OpeningNode, cand: dict[str, Any]) -> bool:
    return _candidate_importance_weight(node=node, cand=cand) >= 1.0


def _can_skip_child_expansion(*, node: OpeningNode, board_size: int) -> bool:
    return float(node.importance) * _ply_decay() < _importance_min(board_size=int(board_size))


def _merge_opening_child(
    children_by_position: dict[str, OpeningNode],
    *,
    child: OpeningNode,
    parent: int,
) -> None:
    existing = children_by_position.get(child.position)
    if existing is None:
        child.parent = int(parent)
        children_by_position[child.position] = child
    elif child.importance > existing.importance:
        existing.importance = child.importance


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
        if node.ply == 0:
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
            * lps._stone_fraction_for_importance(
                stone_fraction=float(stone_fraction),
                child_ply=node.ply + 1,
            )
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
                ply=node.ply + 1,
                importance=float(child_importance),
                move=str(cand["move"]),
            )
        )
    record = {
        "ply": node.ply,
        "canonicalized_position": node.position,
        "importance": _rounded_float(node.importance),
        "candidates": candidate_rows,
        "retained_moves": retained_moves,
    }
    return record, children


def _apply_prior_weighted_graph_values(*, nodes: list[dict[str, Any]]) -> None:
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
        "board_size": int(board_size),
        "root": (0 if nodes else None),
        "root_openings": list(root_openings),
        "root_study": (root_study if isinstance(root_study, dict) else None),
        "completed": bool(completed),
        "completed_ply": int(completed_ply),
        "nodes": nodes,
    }


def _write_opening_artifact(path: Path, payload: dict[str, Any]) -> None:
    artifact = {key: value for key, value in payload.items() if key != "nodes"}
    artifact["nodes"] = []
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("opening artifact requires a nodes list")
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError(f"bad opening artifact node: {node!r}")
        retained = []
        nonretained = []
        candidates = node.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"opening artifact node requires candidates: {node!r}")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"bad opening artifact candidate: {candidate!r}")
            candidate_retained = candidate.get("retained")
            if not isinstance(candidate_retained, bool):
                raise ValueError(f"opening candidate requires retained flag: {candidate!r}")
            if candidate_retained:
                retained.append(
                    {
                        key: candidate.get(key)
                        for key in (
                            "move",
                            "rank",
                            "prior",
                            "stone_fraction",
                            "candidate_weight",
                            "importance",
                            "child",
                            "tree_mover_winrate",
                        )
                    }
                )
            else:
                nonretained.append(
                    {key: candidate.get(key) for key in ("move", "importance")}
                )
        artifact_node = {
            key: node.get(key)
            for key in (
                "parent",
                "move",
                "ply",
                "importance",
                "tree_red_winrate",
            )
        }
        artifact_node["candidates"] = retained
        if nonretained:
            artifact_node["nonretained_candidates"] = nonretained
        artifact["nodes"].append(artifact_node)
    aj.dump_tree(path, artifact)


def _opening_node_positions(*, board_size: int, nodes: list[dict[str, Any]]) -> list[str | None]:
    if not nodes:
        raise ValueError("opening graph requires a root node")
    node_keys = aj.OPENING_NODE_KEYS
    positions: list[str | None] = [None] * len(nodes)
    root_position = _empty_position(board_size=int(board_size))
    for idx, node in enumerate(nodes):
        parent = node[node_keys["parent"]]
        move = str(node.get(node_keys["move"]) or "").strip().lower()
        if parent is None:
            if idx != 0 or move:
                raise ValueError(f"bad opening root at node {idx}: {node!r}")
            positions[idx] = root_position
            continue
        if not move:
            raise ValueError(f"opening node {idx} is missing its move")
        if isinstance(parent, bool) or not isinstance(parent, int) or parent < 0 or parent >= idx:
            raise ValueError(f"bad opening parent index: {node!r}")
        parent_position = positions[parent]
        if not isinstance(parent_position, str):
            raise ValueError(f"missing parent position for opening node {idx}")
        size, red, blue, to_play = lps._position_state(parent_position)
        positions[idx] = lps._position_after_move_from_state(
            size=size,
            red=red,
            blue=blue,
            to_play=to_play,
            move=move,
        )
    return positions


def _artifact_candidate_move(raw: Any, *, node_idx: int) -> str:
    if not isinstance(raw, dict):
        raise ValueError(f"bad opening candidate at node {node_idx}: {raw!r}")
    move = str(raw.get(aj.OPENING_CANDIDATE_KEYS["move"]) or "").strip().lower()
    if not move:
        raise ValueError(f"opening candidate missing move at node {node_idx}: {raw!r}")
    return move


def _prune_raw_nn_cache(*, board_size: int, output_path: Path) -> tuple[Path, Path, int, int, int, int]:
    if not output_path.exists():
        raise FileNotFoundError(f"opening output not found for cache pruning: {output_path}")
    raw_nn_cache_path = _raw_nn_cache_path(board_size=int(board_size))
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)
    payload = aj.load(output_path)
    if not isinstance(payload, dict):
        raise ValueError(f"opening artifact must be an object: {output_path}")
    root_keys = aj.OPENING_ROOT_KEYS
    node_keys = aj.OPENING_NODE_KEYS
    nodes = payload.get(root_keys["nodes"])
    if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
        raise ValueError(f"opening artifact requires object nodes: {output_path}")
    positions = _opening_node_positions(board_size=int(board_size), nodes=nodes)

    keep_positions: set[str] = set()
    policy_positions: set[str] = set()
    missing_node_payloads = 0
    missing_child_payloads = 0
    for idx, record in enumerate(nodes):
        position = positions[idx]
        if not isinstance(position, str):
            continue
        keep_positions.add(position)
        if position != _empty_position(board_size=int(board_size)):
            policy_positions.add(position)
            if not isinstance(raw_nn_cache.get(lps._cache_key(position)), dict):
                missing_node_payloads += 1
        retained_candidates = record[node_keys["candidates"]]
        nonretained_candidates = record.get(node_keys["nonretained_candidates"], [])
        if not isinstance(retained_candidates, list) or not isinstance(nonretained_candidates, list):
            raise ValueError(f"opening node {idx} requires candidate lists")
        size, red, blue, to_play = lps._position_state(position)
        for candidates, retained in ((retained_candidates, True), (nonretained_candidates, False)):
            for row in candidates:
                move = _artifact_candidate_move(row, node_idx=idx)
                child_position = lps._position_after_move_from_state(
                    size=size,
                    red=red,
                    blue=blue,
                    to_play=to_play,
                    move=move,
                )
                keep_positions.add(child_position)
                if retained:
                    policy_positions.add(child_position)
                if not isinstance(raw_nn_cache.get(lps._cache_key(child_position)), dict):
                    missing_child_payloads += 1

    before = len(raw_nn_cache)
    keep_keys = {lps._cache_key(position) for position in keep_positions}
    policy_keys = {lps._cache_key(position) for position in policy_positions}
    pruned: dict[str, dict[str, Any]] = {}
    for key, cache_payload in raw_nn_cache.items():
        if _is_special_raw_nn_cache_key(key):
            pruned[key] = cache_payload
        elif key in keep_keys:
            pruned[key] = (
                cache_payload
                if key in policy_keys or not lps._is_valid_encoded_raw_nn_winrate(cache_payload)
                else {"r": cache_payload["r"]}
            )

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

    frontier = [OpeningNode(position=_empty_position(board_size=board_size_i), ply=0)]
    nodes: list[dict[str, Any]] = []
    completed_ply = 0
    root_study: dict[str, Any] | None = None
    root_openings: tuple[str, ...] = ()

    while frontier:
        depth_started_at = time.time()
        frontier_count = len(frontier)
        current_ply = frontier[0].ply
        root_sweep_cache_hits = 0
        if current_ply == 0:
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
            if node.ply > 0 and not _can_skip_child_expansion(node=node, board_size=board_size_i)
        ]
        expansion_positions = [node.position for node in expansion_nodes]
        root_payloads, position_cache_hits = _ensure_opening_policy_cached(
            nodes=expansion_nodes,
            board_size=board_size_i,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )
        expanded_candidates = _run_opening_expansion_and_store_policy(
            nodes=expansion_nodes,
            payloads=root_payloads,
            board_size=board_size_i,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )
        candidates_by_position = {
            node.position: candidates
            for node, candidates in zip(expansion_nodes, expanded_candidates)
        }
        prepared: list[tuple[OpeningNode, list[dict[str, Any]]]] = []
        child_positions: list[str] = []
        seen_child_positions: set[str] = set()
        for node in frontier:
            if _can_skip_child_expansion(node=node, board_size=board_size_i):
                prepared.append((node, []))
                continue
            if node.ply == 0:
                parent_state = lps._position_state(node.position)
                candidates = _select_root_candidates(
                    node=node,
                    board_size=board_size_i,
                    root_openings=root_openings,
                    parent_state=parent_state,
                )
            else:
                candidates = candidates_by_position[node.position]
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
        requested_positions = expansion_positions + child_positions
        cache_total = len(requested_positions) + (1 if current_ply == 0 else 0)
        if cache_hits < cache_total:
            _log(
                f"starting ply={current_ply} "
                f"nodes={frontier_count} cache={cache_hits}/{cache_total}",
                board_size=board_size_i,
            )
        child_policy_payloads: dict[str, dict[str, Any]] = {}
        child_payloads, _ = _run_multi_position_raw_nn_cached(
            position_inputs=child_positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
            include_moves=False,
            store_moves=False,
            policy_payloads_out=child_policy_payloads,
        )
        current_ply_start = len(nodes)
        finalized: list[tuple[OpeningNode, dict[str, Any], list[OpeningNode]]] = []
        next_by_position: dict[str, OpeningNode] = {}
        for node_index, (node, candidates) in enumerate(prepared):
            record, children = _finalize_node(
                node,
                candidates=candidates,
                child_payloads=child_payloads,
                root_study=root_study,
            )
            current_idx = current_ply_start + node_index
            for child in children:
                _merge_opening_child(
                    next_by_position,
                    child=child,
                    parent=current_idx,
                )
            finalized.append((node, record, children))
        next_frontier = list(next_by_position.values())
        next_ply_start = current_ply_start + frontier_count
        child_idx_by_position = {
            child.position: next_ply_start + idx
            for idx, child in enumerate(next_frontier)
        }
        for node, record, children in finalized:
            child_by_move = {
                str(child.move): child_idx_by_position[child.position]
                for child in children
                if child.move
            }
            opening_node = _build_opening_node(
                record=record,
                parent=node.parent,
                move=node.move,
                child_by_move=child_by_move,
            )
            nodes.append(opening_node)
        current_completed_ply = current_ply
        stopping = (
            isinstance(stop_after_ply, int)
            and current_completed_ply >= int(stop_after_ply)
        )
        expandable_frontier = [
            child
            for child in next_frontier
            if not _can_skip_child_expansion(node=child, board_size=board_size_i)
        ]
        expandable_positions = {child.position for child in expandable_frontier}
        cache_changed = False
        for position in expandable_positions:
            payload = child_policy_payloads.get(position)
            if not lps._is_valid_encoded_raw_nn_policy(payload):
                continue
            key = lps._precanonicalized_position_cache_key(position)
            if raw_nn_cache.get(key) != payload:
                raw_nn_cache[key] = payload
                cache_changed = True
        for child in next_frontier:
            if child.position in expandable_positions:
                continue
            key = lps._precanonicalized_position_cache_key(child.position)
            payload = raw_nn_cache.get(key)
            if lps._is_valid_encoded_raw_nn_winrate(payload) and payload != {"r": payload["r"]}:
                raw_nn_cache[key] = {"r": payload["r"]}
                cache_changed = True
        if cache_changed:
            lps._save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
        if stopping and expandable_frontier:
            frontier_payloads, _ = _ensure_opening_policy_cached(
                nodes=expandable_frontier,
                board_size=board_size_i,
                raw_nn_cache=raw_nn_cache,
                raw_nn_cache_path=raw_nn_cache_path,
            )
            _run_opening_expansion_and_store_policy(
                nodes=expandable_frontier,
                payloads=frontier_payloads,
                board_size=board_size_i,
                raw_nn_cache=raw_nn_cache,
                raw_nn_cache_path=raw_nn_cache_path,
            )
        completed_ply = current_completed_ply
        frontier = next_frontier
        _apply_prior_weighted_graph_values(nodes=nodes)
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
