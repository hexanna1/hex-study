#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import artifact_json as aj
import hex_board as hb
import native_tools as native
import pattern_scoring as scoring
import raw_nn as nn
import study_runtime as runtime


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


def _importance_min(*, board_size: int) -> float:
    size = int(board_size)
    if size > 14:
        return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_LARGE_BOARD_IMPORTANCE_MIN))
    return float(IMPORTANCE_MIN_BY_BOARD_SIZE.get(size, DEFAULT_SMALL_BOARD_IMPORTANCE_MIN))


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


def _evaluate_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    include_moves: bool = True,
    store_moves: bool = True,
    policy_payloads_out: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    return nn.evaluate_cached(
        position_inputs=position_inputs, raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path, chunk_size=RAW_NN_CACHE_CHUNK_SIZE,
        move_limit=RAW_NN_CACHE_MOVE_LIMIT,
        include_moves=include_moves, store_moves=store_moves,
        policy_payloads_out=policy_payloads_out,
    )


def _log(message: str, *, board_size: int | None = None) -> None:
    prefix = ""
    if isinstance(board_size, int):
        prefix = f"[{board_size}] "
    runtime.log(f"{prefix}{message}")


def _empty_position(*, board_size: int) -> str:
    return hb.serialize_position(
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
    red_wr = nn.red_winrate(child_payload)
    if not isinstance(red_wr, float):
        raise ValueError("child payload missing cached red winrate")
    if str(parent_to_play).strip().lower() == "red":
        return red_wr
    return 1.0 - red_wr


def _red_winrate_from_mover_winrate(*, mover_winrate: float, parent_ply: int) -> float:
    return mover_winrate if parent_ply % 2 == 0 else 1.0 - mover_winrate


def _mover_winrate_from_red_winrate(*, red_winrate: float, parent_ply: int) -> float:
    return red_winrate if parent_ply % 2 == 0 else 1.0 - red_winrate


def _winrate_to_elo(winrate: float) -> float:
    p = max(WINRATE_EPS, min(1.0 - WINRATE_EPS, float(winrate)))
    return 400.0 * math.log10(p / (1.0 - p))


def _rounded_float(value: float | None, *, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _build_opening_node(
    *,
    record: dict[str, Any],
    parent: int | None,
    move: str | None,
    child_by_move: dict[str, int] | None = None,
) -> dict[str, Any]:
    child_lookup = child_by_move or {}
    candidates = [
        {
            **{key: value for key, value in row.items() if key != "child"},
            "child": child_lookup.get(row["move"]) if row["retained"] else None,
        }
        for row in record["candidates"]
    ]
    return {
        "parent": parent,
        "move": move,
        **record,
        "candidates": candidates,
    }


def _full_stone_elo_from_root_study(root_study: dict[str, Any]) -> float:
    full_stone_elo = 4.0 * root_study["reference_elo"]
    if full_stone_elo <= 0.0:
        raise ValueError(f"bad full-stone Elo calibration: {full_stone_elo!r}")
    return full_stone_elo


def _root_stone_fraction_from_study(*, move: str, root_study: dict[str, Any]) -> float:
    by_move = {row["move"]: row["stone_fraction"] for row in root_study["rows"]}
    distance = abs(by_move[move] - 0.5)
    return math.sqrt(max(0.0, min(1.0, 1.0 - distance)))


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


def _reference_root_move(*, board_size: int) -> str:
    size = int(board_size)
    if size < 3:
        raise ValueError(f"board size too small for fair-root reference: {size}")
    return hb.point_to_cell(2, size - 1)


def _rotate_180_move(move: str, *, board_size: int) -> str:
    col, row = hb.parse_cell(str(move).strip().lower())
    size = int(board_size)
    return hb.point_to_cell(size + 1 - int(col), size + 1 - int(row))


def _canonical_rotation_root_move(move: str, *, board_size: int) -> str:
    a = str(move).strip().lower()
    b = _rotate_180_move(a, board_size=board_size)
    a_col, a_row = hb.parse_cell(a)
    b_col, b_row = hb.parse_cell(b)
    size = int(board_size)

    def rep_key(col: int, row: int, cell: str) -> tuple[int, int, int, int, str]:
        on_preferred_side = int(row) + int(col) <= size + 1
        diagonal_tiebreak = 0 if int(row) >= int(col) else 1
        return (0 if on_preferred_side else 1, diagonal_tiebreak, int(row), int(col), cell)

    a_key = rep_key(int(a_col), int(a_row), a)
    b_key = rep_key(int(b_col), int(b_row), b)
    return a if a_key <= b_key else b


def _coarse_bucket_root_move(move: str, *, board_size: int) -> str:
    col, row = hb.parse_cell(str(move).strip().lower())
    size = int(board_size)
    if int(row) == 2:
        if int(col) <= 3:
            return hb.point_to_cell(3, 2)
        if int(col) >= size - 2:
            return hb.point_to_cell(size - 2, 2)
    return str(move).strip().lower()


def _canonical_fair_root_move(move: str, *, board_size: int) -> str:
    return _coarse_bucket_root_move(
        _canonical_rotation_root_move(move, board_size=board_size),
        board_size=board_size,
    )


def _canonical_fair_root_representatives(*, board_size: int) -> tuple[str, ...]:
    size = int(board_size)
    reps = {
        _canonical_fair_root_move(hb.point_to_cell(col, row), board_size=size)
        for col in range(1, size + 1)
        for row in range(1, size + 1)
    }
    return tuple(sorted(reps, key=lambda cell: tuple(int(x) for x in reversed(hb.parse_cell(cell)))))


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
        child_position = hb.position_after_move(root_position, move)
        payload = child_payloads.get(child_position)
        red_wr = nn.red_winrate(payload) if isinstance(payload, dict) else None
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
    if nn.has_payload(cached):
        cached = nn.decode_payload(cached)
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
    child_positions = [hb.position_after_move(root_position, move) for move in requested_moves]
    child_payloads, _ = _evaluate_cached(
        position_inputs=child_positions,
        raw_nn_cache=raw_nn_cache,
        include_moves=False,
    )
    reduced = _fair_root_sweep_payload_from_child_raw_nn(
        root_position=root_position,
        requested_moves=requested_moves,
        child_payloads=child_payloads,
    )
    raw_nn_cache[cache_key] = nn.encode_payload(reduced)
    if isinstance(raw_nn_cache_path, Path):
        nn.save_cache(raw_nn_cache_path, raw_nn_cache)
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
        size, red, blue, to_play = hb.position_state(node.position)
    else:
        size, red, blue, to_play = parent_state
    if size != int(board_size) or to_play != "red":
        raise ValueError(f"unexpected root node state for board size {board_size}: {node.position!r}")
    return [
        {
            "move": move,
            "rank": idx,
            "prior": None,
            "child_position": hb.position_after_move_from_state(
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
        repr(PLY_DECAY),
        repr(EXTRA_CANDIDATE_PRIOR_MIN),
        repr(OUTSIDE_TOP_K_PRIOR_LOG_STEP),
        repr(OUTSIDE_TOP_K_EXPONENT_RANK_STEP),
        repr(OUTSIDE_TOP_K_EXPONENT_PLY_STEP),
        repr(POLICY_IMPORTANCE_HEADROOM),
        str(POLICY_TOP_K_HEADROOM),
    ]
    input_lines = ["\t".join(config)]
    for node in nodes:
        payload = payloads.get(node.position)
        if not isinstance(payload, dict):
            raise ValueError(f"missing root payload for {node.position!r}")
        policy: list[str] = []
        for row in nn.policy_moves(payload):
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
    output_lines = native.expand_positions(input_lines, expected_rows=len(nodes))
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
                    "prior": nn.decode_millionths(int(prior)),
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
    if float(prior) >= EXTRA_CANDIDATE_PRIOR_MIN:
        return 1.0
    prior_log10 = -math.log10(max(1e-6, float(prior)))
    rank_delta = int(cleaned_rank) - int(top_k)
    exponent = (
        (OUTSIDE_TOP_K_PRIOR_LOG_STEP * prior_log10)
        + (OUTSIDE_TOP_K_EXPONENT_RANK_STEP * max(0, rank_delta - 1))
        + (OUTSIDE_TOP_K_EXPONENT_PLY_STEP * max(0, node.ply - 1))
    )
    return _importance_min(board_size=int(board_size)) ** float(exponent)


def _candidate_importance_weight(*, node: OpeningNode, cand: dict[str, Any]) -> float:
    if node.ply == 0:
        return 1.0
    return _candidate_importance_weight_for_rank(
        node=node,
        board_size=cand["board_size"],
        cleaned_rank=cand["cleaned_rank"],
        prior=cand["prior"],
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
    for raw_rows, row in enumerate(nn.policy_moves(payload), start=1):
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
    moves = nn.policy_moves(payload)
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
        prior=nn.decode_millionths(prior),
        top_k=_top_k_for_ply(node.ply),
    )
    upper_bound = float(node.importance) * PLY_DECAY * candidate_weight
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

    payloads, cache_hits = nn.ensure_cached(
        board_size=board_size, chunk_size=RAW_NN_CACHE_CHUNK_SIZE,
        move_limit=RAW_NN_CACHE_MOVE_LIMIT, precanonicalized_position_inputs=True,
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
        moves = nn.policy_moves(payload)
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
        if not nn.has_policy(payload):
            raise ValueError(f"opening policy payload missing for {position!r}")
        moves = nn.policy_moves(payload)
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
        key = hb.serialized_position_key(position)
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
        nn.save_cache(raw_nn_cache_path, raw_nn_cache)
    return candidates


def _candidate_sets_elo_baseline(*, node: OpeningNode, cand: dict[str, Any]) -> bool:
    return _candidate_importance_weight(node=node, cand=cand) >= 1.0


def _can_skip_child_expansion(*, node: OpeningNode, board_size: int) -> bool:
    return float(node.importance) * PLY_DECAY < _importance_min(board_size=int(board_size))


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
    full_stone_elo = _full_stone_elo_from_root_study(root_study)
    evaluated: list[tuple[dict[str, Any], float, float]] = []
    anchor_elos: list[float] = []
    for candidate in candidates:
        winrate = _mover_winrate_from_child_payload(
            child_payload=child_payloads[candidate["child_position"]],
            parent_to_play=candidate["parent_to_play"],
        )
        elo = _winrate_to_elo(winrate)
        if _candidate_sets_elo_baseline(node=node, cand=candidate):
            anchor_elos.append(elo)
        evaluated.append((candidate, winrate, elo))
    best_anchor_elo = max(anchor_elos or [elo for _, _, elo in evaluated], default=None)

    children: list[OpeningNode] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate, winrate, elo in evaluated:
        move = candidate["move"]
        board_size = candidate["board_size"]
        if node.ply == 0:
            fraction = _root_importance_override(board_size=board_size, move=move)
            if fraction is None:
                fraction = _root_stone_fraction_from_study(move=move, root_study=root_study)
        else:
            fraction = _stone_fraction_from_elo_loss(
                elo_loss=max(0.0, best_anchor_elo - elo), full_stone_elo=full_stone_elo,
            )
        weight = _candidate_importance_weight(node=node, cand=candidate)
        importance = (
            node.importance
            * scoring.stone_fraction_for_importance(stone_fraction=fraction, child_ply=node.ply + 1)
            * PLY_DECAY
            * weight
        )
        retained = importance >= _importance_min(board_size=board_size)
        candidate_rows.append({
            "move": move,
            "rank": candidate["rank"],
            "prior": _rounded_float(candidate["prior"]),
            "raw_mover_winrate": _rounded_float(winrate),
            "stone_fraction": _rounded_float(fraction),
            "candidate_weight": _rounded_float(weight),
            "importance": _rounded_float(importance),
            "retained": retained,
        })
        if retained:
            children.append(OpeningNode(
                position=candidate["child_position"],
                ply=node.ply + 1,
                importance=importance,
                move=move,
            ))
    return {
        "ply": node.ply,
        "importance": _rounded_float(node.importance),
        "candidates": candidate_rows,
    }, children


def _apply_prior_weighted_graph_values(*, nodes: list[dict[str, Any]]) -> None:
    raw_red_winrates: list[float | None] = [None] * len(nodes)
    for node in nodes:
        for row in node["candidates"]:
            raw = row["raw_mover_winrate"]
            row["tree_mover_winrate"] = _rounded_float(raw)
            child = row["child"]
            if child is not None and child < len(nodes):
                raw_red_winrates[child] = _red_winrate_from_mover_winrate(
                    mover_winrate=raw, parent_ply=node["ply"],
                )

    # Breadth-first construction places every materialized child after its parents.
    # Frontier edges still carry raw values when their child is not materialized.
    for index in range(len(nodes) - 1, -1, -1):
        node = nodes[index]
        weighted: list[tuple[float, float]] = []
        for row in node["candidates"]:
            if not row["retained"]:
                continue
            child = row["child"]
            red = nodes[child]["tree_red_winrate"] if child is not None and child < len(nodes) else None
            if red is None:
                red = _red_winrate_from_mover_winrate(
                    mover_winrate=row["raw_mover_winrate"], parent_ply=node["ply"],
                )
            row["tree_mover_winrate"] = _rounded_float(
                _mover_winrate_from_red_winrate(red_winrate=red, parent_ply=node["ply"])
            )
            if row["prior"] is not None:
                weighted.append((row["prior"], red))
        total_prior = sum(prior for prior, _ in weighted)
        red = (
            sum(prior * value for prior, value in weighted) / total_prior
            if total_prior > 0.0
            else raw_red_winrates[index]
        )
        node["tree_red_winrate"] = _rounded_float(red)


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
    for node in payload["nodes"]:
        retained = []
        nonretained = []
        for candidate in node["candidates"]:
            if candidate["retained"]:
                retained.append({key: candidate.get(key) for key in aj.OPENING_CANDIDATE_KEYS})
            else:
                nonretained.append({key: candidate.get(key) for key in ("move", "importance")})
        artifact_node = {
            key: node.get(key)
            for key in ("parent", "move", "ply", "importance", "tree_red_winrate")
        }
        artifact_node["candidates"] = retained
        if nonretained:
            artifact_node["nonretained_candidates"] = nonretained
        artifact["nodes"].append(artifact_node)
    aj.dump_tree(path, artifact)


def _opening_node_positions(*, board_size: int, nodes: list[dict[str, Any]]) -> list[str]:
    node_keys = aj.OPENING_NODE_KEYS
    positions = [_empty_position(board_size=board_size)]
    for node in nodes[1:]:
        parent = node[node_keys["parent"]]
        move = node[node_keys["move"]]
        positions.append(hb.position_after_move(positions[parent], move))
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
    raw_nn_cache = nn.load_cache(raw_nn_cache_path)
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
            if not isinstance(raw_nn_cache.get(hb.position_cache_key(position)), dict):
                missing_node_payloads += 1
        retained_candidates = record[node_keys["candidates"]]
        nonretained_candidates = record.get(node_keys["nonretained_candidates"], [])
        if not isinstance(retained_candidates, list) or not isinstance(nonretained_candidates, list):
            raise ValueError(f"opening node {idx} requires candidate lists")
        size, red, blue, to_play = hb.position_state(position)
        for candidates, retained in ((retained_candidates, True), (nonretained_candidates, False)):
            for row in candidates:
                move = _artifact_candidate_move(row, node_idx=idx)
                child_position = hb.position_after_move_from_state(
                    size=size,
                    red=red,
                    blue=blue,
                    to_play=to_play,
                    move=move,
                )
                keep_positions.add(child_position)
                if retained:
                    policy_positions.add(child_position)
                if not isinstance(raw_nn_cache.get(hb.position_cache_key(child_position)), dict):
                    missing_child_payloads += 1

    before = len(raw_nn_cache)
    keep_keys = {hb.position_cache_key(position) for position in keep_positions}
    policy_keys = {hb.position_cache_key(position) for position in policy_positions}
    pruned: dict[str, dict[str, Any]] = {}
    for key, cache_payload in raw_nn_cache.items():
        if _is_special_raw_nn_cache_key(key):
            pruned[key] = cache_payload
        elif key in keep_keys:
            pruned[key] = (
                cache_payload
                if key in policy_keys or not nn.has_winrate(cache_payload)
                else {"r": cache_payload["r"]}
            )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = raw_nn_cache_path.with_name(f"{raw_nn_cache_path.stem}.backup-{timestamp}{raw_nn_cache_path.suffix}")
    nn.save_cache(backup_path, raw_nn_cache)
    nn.save_cache(raw_nn_cache_path, pruned)
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
    raw_nn_cache = nn.load_cache(raw_nn_cache_path)

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
                parent_state = hb.position_state(node.position)
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
        child_cache_hits = nn.cached_winrate_count(
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
        child_payloads, _ = _evaluate_cached(
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
            if not nn.has_policy(payload):
                continue
            key = hb.serialized_position_key(position)
            if raw_nn_cache.get(key) != payload:
                raw_nn_cache[key] = payload
                cache_changed = True
        for child in next_frontier:
            if child.position in expandable_positions:
                continue
            key = hb.serialized_position_key(child.position)
            payload = raw_nn_cache.get(key)
            if nn.has_winrate(payload) and payload != {"r": payload["r"]}:
                raw_nn_cache[key] = {"r": payload["r"]}
                cache_changed = True
        if cache_changed:
            nn.save_cache(raw_nn_cache_path, raw_nn_cache)
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
            f"elapsed={runtime.format_duration(max(0.0, time.time() - depth_started_at))}",
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
        f"elapsed={runtime.format_duration(max(0.0, time.time() - started_at))}",
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
        build_opening_database(
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
