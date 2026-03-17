#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import artifact_json as aj
import joseki_notation as jn
import hex_board as hb
import native_tools as native
import pattern_scoring as scoring
import raw_nn as nn
import study_runtime as runtime
from hex_board import (
    parse_cell,
    point_to_cell,
    select_available_pass_proxy_cell,
    serialize_position,
)


Move = tuple[int, int]
Entry = Move | None
RealizedPosition = tuple[str, set[Move], set[Move], str]


@dataclass(frozen=True, slots=True)
class PreparedExpansion:
    parent_position: str
    tenuki_move: str | None
    local_meta_by_cell: dict[str, dict[str, Any]]
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodeExpansion:
    parent_position: str
    tenuki_move: str | None
    local_meta_by_cell: dict[str, dict[str, Any]]
    child_positions_by_move: dict[str, str]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    index: int
    move: str
    kind: str
    local: Move | None
    meta: dict[str, Any] | None
    stone_fraction: float | None
    forced: bool
    override_importance: float | None


FAMILY_ACUTE = "A"
FAMILY_OBTUSE = "O"
ACUTE_BALANCE_MOVES = ("d1", "c2")
OPPOSITE_OBTUSE_BALANCE_LOCAL_MOVES: tuple[Move, ...] = ((4, 1), (2, 2))
ROOT_LOCAL_MOVES: dict[str, tuple[Move, ...]] = {
    FAMILY_ACUTE: ((4, 3), (5, 4), (6, 5), (7, 6)),
    FAMILY_OBTUSE: ((4, 4), (5, 5), (6, 6), (4, 2)),
}
TOP_K_BY_PLY: dict[int, int] = {
    1: 5,
    2: 4,
    3: 3,
}
DEFAULT_TOP_K = 2
JOSEKI_CHILD_OVERRIDE_RULES: list[tuple[str, tuple[Entry, ...], Move, float]] = [
    (FAMILY_ACUTE, (), (4, 3), 1.00),
    (FAMILY_ACUTE, (), (5, 4), 1.00),
    (FAMILY_ACUTE, ((5, 4),), (3, 4), 0.95),
    (FAMILY_ACUTE, (), (6, 5), 0.98),
    (FAMILY_ACUTE, (), (7, 6), 1.00),
    (FAMILY_OBTUSE, (), (4, 4), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (2, 3), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (2, 4), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (5, 4), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (5, 5), 0.95),
    (FAMILY_OBTUSE, (), (5, 5), 0.90),
    (FAMILY_OBTUSE, (), (6, 6), 0.87),
    (FAMILY_OBTUSE, (), (4, 2), 0.93),
]
LOCAL_DELTA_MAX_BY_FAMILY: dict[str, int] = {
    FAMILY_ACUTE: 91,
    FAMILY_OBTUSE: 49,
}
LOCAL_ANCHOR_DELTA_MAX = 7
LOCAL_ANCHOR_PRIOR_MIN = 0.0002
STONE_FRACTION_MIN = 0.85
TENUKI_STONE_FRACTION_MIN = 0.925
TENUKI_IMPORTANCE_MULT = 0.925
LINE_IMPORTANCE_MIN_BY_FAMILY: dict[str, float] = {
    FAMILY_ACUTE: 0.74,
    FAMILY_OBTUSE: 0.74,
}
PLY_DECAY = 0.994
OUTSIDE_TOP_K_PRIOR_LOG_STEP = 0.05
OUTSIDE_TOP_K_EXPONENT_RANK_STEP = 0.02
OUTSIDE_TOP_K_EXPONENT_PLY_STEP = 0.03
PRIOR_EPS = 1e-6
RAW_NN_CACHE_CHUNK_SIZE = 30000


@dataclass(frozen=True, slots=True)
class JosekiNode:
    family: str
    entries: tuple[Entry, ...]
    position: str
    importance: float = 1.0

    @property
    def line(self) -> str:
        return jn.format_single_track_line(family=self.family, entries=self.entries)

    @property
    def previous_was_tenuki(self) -> bool:
        return bool(self.entries and self.entries[-1] is None)


def _line_importance_min(*, family: str) -> float:
    return LINE_IMPORTANCE_MIN_BY_FAMILY[family]


@cache
def _child_override_rules_by_position(
    board_size: int,
    rules: tuple[tuple[str, tuple[Entry, ...], Move, float], ...],
) -> dict[tuple[str, str, Move], float]:
    compiled: dict[tuple[str, str, Move], float] = {}
    for family, entries, child, importance in rules:
        fam = str(family).strip().upper()
        if any(entry is None for entry in entries):
            raise ValueError("joseki child override parents cannot contain tenuki")
        realized_moves = tuple(
            _family_move_to_cell(family=fam, move=entry, board_size=int(board_size))
            for entry in entries
            if entry is not None
        )
        parent_position = _realize_position(
            family=fam,
            board_size=int(board_size),
            realized_moves=realized_moves,
        )[0]
        key = (fam, parent_position, (int(child[0]), int(child[1])))
        if key in compiled:
            raise ValueError(f"duplicate joseki child override: {key!r}")
        compiled[key] = float(importance)
    return compiled


def _child_override_rule(*, family: str, parent_position: str, child: Move) -> float | None:
    fam = str(family).strip().upper()
    child_move = (int(child[0]), int(child[1]))
    board_size = hb.board_size_from_position(parent_position)
    return _child_override_rules_by_position(
        int(board_size),
        tuple(JOSEKI_CHILD_OVERRIDE_RULES),
    ).get((fam, str(parent_position), child_move))


def _forced_override_children(*, family: str, parent_position: str) -> tuple[Move, ...]:
    fam = str(family).strip().upper()
    board_size = hb.board_size_from_position(parent_position)
    return tuple(
        child
        for (rule_family, rule_position, child), _importance in _child_override_rules_by_position(
            int(board_size),
            tuple(JOSEKI_CHILD_OVERRIDE_RULES),
        ).items()
        if rule_family == fam and rule_position == parent_position
    )


def _config_payload(*, family: str, board_size: int) -> dict[str, Any]:
    return {
        "family": str(family).strip().upper(),
        "board_size": int(board_size),
        "balance_moves": list(_balance_move_tokens(family=str(family).strip().upper(), board_size=int(board_size))),
    }


def _default_output_path(*, family: str, board_size: int) -> Path:
    family_s = str(family).strip().lower()
    return Path("artifacts") / "joseki" / f"joseki-{family_s}-s{int(board_size)}.json"


def _raw_nn_cache_path(*, family: str) -> Path:
    family_s = str(family).strip().lower()
    return Path(__file__).resolve().parent / "artifacts" / "joseki" / f"joseki_raw_nn_cache_{family_s}.json"


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
        include_moves=include_moves, store_moves=store_moves,
        policy_payloads_out=policy_payloads_out,
    )


def _resolve_output_path(*, out_arg: str | None, family: str, board_size: int) -> Path:
    if not out_arg:
        return _default_output_path(family=family, board_size=board_size)
    out_path = Path(str(out_arg))
    if out_path.suffix.lower() == ".json":
        return out_path
    return out_path / _default_output_path(family=family, board_size=board_size).name


def _log(message: str, *, family: str | None = None) -> None:
    prefix = ""
    if isinstance(family, str) and family.strip():
        prefix = f"[{family.strip().upper()}] "
    runtime.log(f"{prefix}{message}")


def _rounded_stone_fraction(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _rounded_importance(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _build_output_payload(
    *,
    family: str,
    board_size: int,
    nodes: list[dict[str, Any]],
    completed: bool,
    completed_depth: int,
) -> dict[str, Any]:
    return {
        **_config_payload(family=family, board_size=board_size),
        "completed": bool(completed),
        "completed_depth": int(completed_depth),
        "nodes": nodes,
    }


def _write_joseki_artifact(path: Path, payload: dict[str, Any]) -> None:
    aj.dump_tree(path, payload)


def _family_delta_metric(*, family: str, dq: int, dr: int) -> int:
    if family == FAMILY_ACUTE:
        return dq * dq + dq * dr + dr * dr
    return dq * dq - dq * dr + dr * dr


def _local_delta_max(*, family: str) -> int:
    return LOCAL_DELTA_MAX_BY_FAMILY[family]


def _corner_distance(*, family: str, local: Move) -> int:
    return _family_delta_metric(family=family, dq=local[0] - 1, dr=local[1] - 1)


def _delta_between(*, family: str, a: Move, b: Move) -> int:
    return _family_delta_metric(family=family, dq=a[0] - b[0], dr=a[1] - b[1])


def _top_k_for_ply(ply: int) -> int | None:
    return TOP_K_BY_PLY.get(ply, DEFAULT_TOP_K) if ply > 0 else None


def _family_move_to_cell(*, family: str, move: Move, board_size: int) -> str:
    x, y = int(move[0]), int(move[1])
    fam = str(family).strip().upper()
    if fam == FAMILY_ACUTE:
        col, row = (int(board_size) - y + 1), (int(board_size) - x + 1)
    elif fam == FAMILY_OBTUSE:
        col, row = y, (int(board_size) - x + 1)
    else:
        raise ValueError(f"unsupported family: {family!r}")
    if not (1 <= col <= int(board_size) and 1 <= row <= int(board_size)):
        raise ValueError(f"joseki move out of bounds for board size {board_size}: {move!r}")
    return point_to_cell(int(col), int(row))


def _cell_to_family_move(*, family: str, cell: str, board_size: int) -> Move:
    col, row = hb.parse_cell(cell)
    fam = str(family).strip().upper()
    if fam == FAMILY_ACUTE:
        return int(board_size) - int(row) + 1, int(board_size) - int(col) + 1
    if fam == FAMILY_OBTUSE:
        return int(board_size) - int(row) + 1, int(col)
    raise ValueError(f"unsupported family: {family!r}")


def _opposite_obtuse_balance_move_to_cell(*, move: Move, board_size: int) -> str:
    x, y = int(move[0]), int(move[1])
    col = int(board_size) - y + 1
    row = x
    if not (1 <= col <= int(board_size) and 1 <= row <= int(board_size)):
        raise ValueError(f"obtuse balance move out of bounds for board size {board_size}: {move!r}")
    return point_to_cell(col, row)


def _balance_move_tokens(*, family: str, board_size: int) -> tuple[str, ...]:
    fam = str(family).strip().upper()
    if fam == FAMILY_ACUTE:
        return ACUTE_BALANCE_MOVES
    if fam == FAMILY_OBTUSE:
        return tuple(
            _opposite_obtuse_balance_move_to_cell(move=move, board_size=board_size)
            for move in OPPOSITE_OBTUSE_BALANCE_LOCAL_MOVES
        )
    raise ValueError(f"unsupported family: {family!r}")


def _balance_cells(*, family: str, board_size: int) -> tuple[tuple[Move, ...], tuple[Move, ...]]:
    cells = tuple(parse_cell(token) for token in _balance_move_tokens(family=family, board_size=board_size))
    return cells[::2], cells[1::2]


def _parse_line_entries(line: str) -> tuple[Entry, ...]:
    raw = str(line or "").strip()
    if not raw:
        return ()
    parsed = jn.parse_joseki_line(raw)
    if len(parsed.blocks) != 1:
        raise ValueError(f"expected single-track joseki line: {line!r}")
    return parsed.blocks[0].entries


def _candidate_children_for_position(
    *,
    family: str,
    entries: tuple[Entry, ...],
    position: str,
    root_payload: dict[str, Any] | None,
    normalized_moves: list[str | None] | None = None,
) -> tuple[set[str], dict[str, str]]:
    board_size, red, blue, to_play = hb.position_state(position)
    prepared = _prepare_node(
        JosekiNode(family=family, entries=entries, position=position),
        board_size=board_size,
        root_payload=root_payload,
        realized=(position, red, blue, to_play),
        normalized_moves=normalized_moves,
    )
    positions = {move: hb.position_after_move(position, move) for move in prepared.moves}
    lines = {
        jn.format_single_track_line(family=family, entries=entries + (meta["local"],)): positions[move]
        for move, meta in prepared.local_meta_by_cell.items()
    }
    if prepared.tenuki_move is not None:
        line = jn.format_single_track_line(family=family, entries=entries + (None,))
        lines[line] = positions[prepared.tenuki_move]
    return set(positions.values()), lines


def _prune_raw_nn_cache(*, family: str, output_path: Path) -> tuple[Path, Path, int, int]:
    family_s = str(family).strip().upper()
    if not output_path.exists():
        raise FileNotFoundError(f"debug output not found for cache pruning: {output_path}")
    raw_nn_cache_path = _raw_nn_cache_path(family=family_s)
    raw_nn_cache = nn.load_cache(raw_nn_cache_path)
    payload = aj.load(output_path)
    if not isinstance(payload, dict):
        raise ValueError(f"joseki artifact must be an object: {output_path}")
    root_keys = aj.JOSEKI_ROOT_KEYS
    node_keys = aj.JOSEKI_NODE_KEYS
    candidate_keys = aj.JOSEKI_CANDIDATE_KEYS
    nodes = payload.get(root_keys["nodes"])
    if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
        raise ValueError(f"joseki artifact requires object nodes: {output_path}")
    board_size = int(payload[root_keys["board_size"]])

    for idx, node in enumerate(nodes):
        line = node.get(node_keys["line"])
        if not isinstance(line, str):
            raise ValueError(f"joseki node {idx} requires a line")
    if not nodes or nodes[0].get(node_keys["line"]) != "":
        raise ValueError("joseki artifact requires a root node")

    keep_positions: set[str] = set()
    policy_positions: set[str] = set()
    frontier_positions: dict[str, str] = {}
    missing_materialized_payloads = 0

    normalization_items: list[tuple[int, str, int, dict[str, Any]]] = []
    for idx, node in enumerate(nodes):
        position = str(node.get(node_keys["canonicalized_position"]) or "").strip()
        if not position:
            continue
        entries = _parse_line_entries(str(node.get(node_keys["line"]) or ""))
        cache_payload = raw_nn_cache.get(hb.position_cache_key(position))
        decoded = (
            nn.decode_payload(cache_payload)
            if nn.has_policy(cache_payload)
            else {}
        )
        normalization_items.append((idx, position, len(entries), decoded))
    normalized_by_node = dict(
        zip(
            (item[0] for item in normalization_items),
            _normalized_candidate_rows(
                family=family_s,
                board_size=board_size,
                rows=[item[1:] for item in normalization_items],
            ),
        )
    )

    for node_idx, node in enumerate(nodes):
        line = str(node.get(node_keys["line"]) or "")
        position = str(node.get(node_keys["canonicalized_position"]) or "").strip()
        if position:
            keep_positions.add(position)
        if not position:
            continue
        entries = _parse_line_entries(line)
        if entries:
            policy_positions.add(position)
        cache_payload = raw_nn_cache.get(hb.position_cache_key(position))
        has_policy = nn.has_policy(cache_payload)
        if entries and not has_policy:
            missing_materialized_payloads += 1
        decoded_payload = nn.decode_payload(cache_payload) if has_policy else None
        child_positions, line_to_position = _candidate_children_for_position(
            family=family_s,
            entries=entries,
            position=position,
            root_payload=decoded_payload,
            normalized_moves=normalized_by_node.get(node_idx),
        )
        keep_positions.update(child_positions)
        for candidate in node.get(node_keys["candidates"], []) or []:
            if not isinstance(candidate, dict):
                raise ValueError("joseki candidates must be objects")
            child = candidate.get(candidate_keys["child"])
            if child is None:
                continue
            if isinstance(child, bool) or not isinstance(child, int) or child < 0:
                raise ValueError(f"bad joseki child reference: {child!r}")
            if child < len(nodes):
                continue
            kind = str(candidate.get(candidate_keys["kind"]) or "")
            if kind == "local":
                local_raw = candidate.get(candidate_keys["local"])
                if not isinstance(local_raw, list) or len(local_raw) != 2:
                    raise ValueError(f"bad joseki local move: {local_raw!r}")
                child_entries = entries + ((int(local_raw[0]), int(local_raw[1])),)
            elif kind == "tenuki":
                child_entries = entries + (None,)
            else:
                raise ValueError(f"retained joseki candidate has bad kind: {kind!r}")
            child_line = jn.format_single_track_line(
                family=family_s,
                entries=child_entries,
            )
            child_position = line_to_position.get(child_line)
            if child_position is not None:
                frontier_positions[child_line] = child_position

    normalization_items = []
    for line, position in frontier_positions.items():
        cache_payload = raw_nn_cache.get(hb.position_cache_key(position))
        if nn.has_policy(cache_payload):
            normalization_items.append(
                (
                    line,
                    position,
                    len(_parse_line_entries(line)),
                    nn.decode_payload(cache_payload),
                )
            )
    frontier_normalized = dict(
        zip(
            (item[0] for item in normalization_items),
            _normalized_candidate_rows(
                family=family_s,
                board_size=board_size,
                rows=[item[1:] for item in normalization_items],
            ),
        )
    )

    missing_frontier_payloads = 0
    for line, position in frontier_positions.items():
        keep_positions.add(position)
        policy_positions.add(position)
        cache_payload = raw_nn_cache.get(hb.position_cache_key(position))
        has_policy = nn.has_policy(cache_payload)
        if not has_policy:
            missing_frontier_payloads += 1
            continue
        decoded_payload = nn.decode_payload(cache_payload)
        entries = _parse_line_entries(line)
        child_positions, _line_to_position = _candidate_children_for_position(
            family=family_s,
            entries=entries,
            position=position,
            root_payload=decoded_payload,
            normalized_moves=frontier_normalized.get(line),
        )
        keep_positions.update(child_positions)

    if missing_materialized_payloads or missing_frontier_payloads:
        raise ValueError(
            "cannot prune joseki cache with missing policy payloads: "
            f"materialized={missing_materialized_payloads} frontier={missing_frontier_payloads}"
        )

    before = len(raw_nn_cache)
    keep_keys = {hb.position_cache_key(position) for position in keep_positions}
    policy_keys = {hb.position_cache_key(position) for position in policy_positions}
    pruned: dict[str, dict[str, Any]] = {}
    for key, cache_payload in raw_nn_cache.items():
        if key not in keep_keys:
            continue
        pruned[key] = (
            cache_payload
            if key in policy_keys or not nn.has_winrate(cache_payload)
            else {"r": cache_payload["r"]}
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = raw_nn_cache_path.with_name(f"{raw_nn_cache_path.stem}.backup-{timestamp}{raw_nn_cache_path.suffix}")
    nn.save_cache(backup_path, raw_nn_cache)
    nn.save_cache(raw_nn_cache_path, pruned)
    return backup_path, raw_nn_cache_path, before, len(pruned)


def _realize_position(*, family: str, board_size: int, realized_moves: tuple[str, ...]) -> RealizedPosition:
    red_cells, blue_cells = _balance_cells(family=family, board_size=board_size)
    red = set(red_cells)
    blue = set(blue_cells)
    side = "red"
    for token in realized_moves:
        col, row = hb.parse_cell(token)
        point = (int(col), int(row))
        if point in red or point in blue:
            raise ValueError(f"duplicate occupied move in joseki realization: {token}")
        if side == "red":
            red.add(point)
            side = "blue"
        else:
            blue.add(point)
            side = "red"
    position = serialize_position(
        board_size=int(board_size),
        red_cells=tuple(sorted(red)),
        blue_cells=tuple(sorted(blue)),
        to_play=side,
    )
    return position, red, blue, side


def _select_available_pass_proxy_move(*, board_size: int, to_play: str, occupied: set[Move]) -> str:
    return point_to_cell(
        *select_available_pass_proxy_cell(
            board_size=board_size,
            to_play=to_play,
            occupied=occupied,
        )
    )


def _candidate_move_tokens(
    *,
    family: str,
    board_size: int,
    ply: int,
    payload: dict[str, Any],
) -> list[str | None]:
    if int(ply) == 0:
        return [
            _family_move_to_cell(family=family, move=local, board_size=board_size)
            for local in ROOT_LOCAL_MOVES.get(str(family).strip().upper(), ())
        ]
    move_priors = [
        nn.move_prior(row)
        for row in nn.policy_moves(payload)
    ]
    return [row[0] if row is not None and row[0] != "pass" else None for row in move_priors]


def _normalized_candidate_rows(
    *,
    family: str,
    board_size: int,
    rows: list[tuple[str, int, dict[str, Any]]],
) -> list[list[str | None]]:
    return native.normalize_moves(
        requests=[
            (
                position,
                _candidate_move_tokens(
                    family=family,
                    board_size=board_size,
                    ply=ply,
                    payload=payload,
                ),
            )
            for position, ply, payload in rows
        ],
        board_size=int(board_size),
    )


def _select_candidates_from_root_payload(
    *,
    family: str,
    board_size: int,
    ply: int,
    entries: tuple[Entry, ...],
    payload: dict[str, Any],
    position: str,
    normalized_moves: list[str | None] | None = None,
) -> tuple[list[tuple[str, Move]], str | None, dict[str, dict[str, Any]]]:
    family_s = str(family).strip().upper()
    candidate_tokens = _candidate_move_tokens(
        family=family_s,
        board_size=board_size,
        ply=ply,
        payload=payload,
    )
    if normalized_moves is None:
        normalized_moves = native.normalize_moves(
            requests=[(position, candidate_tokens)],
            board_size=board_size,
        )[0]
    if len(normalized_moves) != len(candidate_tokens):
        raise ValueError("Move normalization count does not match policy rows")
    if int(ply) == 0:
        root_moves = ROOT_LOCAL_MOVES.get(str(family).strip().upper(), ())
        filtered_root_moves: list[tuple[str, Move]] = []
        meta_by_move: dict[str, dict[str, Any]] = {}
        for idx, local in enumerate(root_moves):
            move = normalized_moves[idx]
            if move is None:
                continue
            filtered_root_moves.append((move, local))
            meta_by_move[move] = {
                "cleaned_rank": int(len(filtered_root_moves)),
                "prior": None,
                "is_forced": False,
            }
        return filtered_root_moves, None, meta_by_move
    moves = nn.policy_moves(payload)
    if not moves:
        return [], None, {}
    local_out: list[tuple[float, str, Move]] = []
    tenuki_move: str | None = None
    tenuki_best: tuple[int, int, float] | None = None
    for seq_idx, row in enumerate(moves):
        move_prior = nn.move_prior(row)
        if move_prior is None:
            continue
        prior = move_prior[1]
        move = normalized_moves[seq_idx]
        if not move:
            continue
        try:
            local = _cell_to_family_move(family=family, cell=move, board_size=board_size)
        except Exception:
            continue
        prior = float(prior) if isinstance(prior, float) else float("-inf")
        if _corner_distance(family=family, local=local) <= _local_delta_max(family=family):
            local_out.append((prior, move, local))
        else:
            rank = seq_idx + 1
            tenuki_key = (rank, seq_idx, -prior)
            if tenuki_best is None or tenuki_key < tenuki_best:
                tenuki_best = tenuki_key
                tenuki_move = move

    if not local_out:
        return [], tenuki_move, {}

    anchors = [entry for entry in entries if isinstance(entry, tuple)]
    if anchors:
        local_out = [
            item
            for item in local_out
            if any(_delta_between(family=family, a=item[2], b=anchor) <= LOCAL_ANCHOR_DELTA_MAX for anchor in anchors)
        ]
    if not local_out:
        meta_by_move = {}
        if tenuki_move is not None and tenuki_best is not None:
            meta_by_move[tenuki_move] = {
                "cleaned_rank": None,
                "prior": None,
                "is_forced": False,
            }
        return [], tenuki_move, meta_by_move

    best_local_move = local_out[0][1]
    filtered: list[tuple[str, Move]] = []
    meta_by_move: dict[str, dict[str, Any]] = {}
    seen_cells: set[str] = set()
    for prior, move, local in local_out:
        if move != best_local_move and prior < float(LOCAL_ANCHOR_PRIOR_MIN):
            continue
        if move in seen_cells:
            continue
        seen_cells.add(move)
        filtered.append((move, local))
        meta_by_move[move] = {
            "cleaned_rank": int(len(filtered)),
            "prior": float(prior),
            "is_forced": False,
        }
    if tenuki_move is not None and tenuki_best is not None:
        meta_by_move[tenuki_move] = {
            "cleaned_rank": None,
            "prior": None,
            "is_forced": False,
        }
    return filtered, tenuki_move, meta_by_move


def _outside_top_k_candidate_weight(
    node: JosekiNode,
    *,
    meta: dict[str, Any] | None,
    head_local_count: int,
) -> float:
    cleaned_rank = meta.get("cleaned_rank") if isinstance(meta, dict) else None
    if isinstance(cleaned_rank, int) and cleaned_rank <= int(head_local_count):
        return 1.0
    prior = meta.get("prior") if isinstance(meta, dict) else None
    rank_delta = max(0, int(cleaned_rank) - int(head_local_count)) if isinstance(cleaned_rank, int) else 0
    prior_log10 = -math.log10(max(PRIOR_EPS, float(prior))) if isinstance(prior, (int, float)) else 0.0
    exponent = (
        OUTSIDE_TOP_K_PRIOR_LOG_STEP * prior_log10
        + OUTSIDE_TOP_K_EXPONENT_RANK_STEP * max(0, rank_delta - 1)
        + OUTSIDE_TOP_K_EXPONENT_PLY_STEP * max(0, len(node.entries) - 1)
    )
    return _line_importance_min(family=node.family) ** exponent


def _candidate_stone_fractions_from_rows(
    rows: list[dict[str, Any]],
    *,
    proxy_move: str | None,
    baseline_move_allowed: Callable[[str], bool],
) -> dict[str, float]:
    valid = [row for row in rows if isinstance(row.get("mean_winrate"), (int, float))]
    if not valid or not isinstance(proxy_move, str) or not proxy_move.strip():
        return {}
    proxy_row = next((row for row in valid if str(row.get("move") or "").lower() == proxy_move), None)
    if proxy_row is None:
        return {}
    baseline_rows = [
        row
        for row in valid
        if str(row.get("move") or "").lower() != proxy_move
        and bool(baseline_move_allowed(str(row.get("move") or "").lower()))
    ]
    if not baseline_rows:
        return {}
    best_row = max(baseline_rows, key=lambda row: float(row["mean_winrate"]))
    l_proxy = scoring.logit_clamped(float(proxy_row["mean_winrate"]))
    l_best = scoring.logit_clamped(float(best_row["mean_winrate"]))
    denom = l_best - l_proxy
    if abs(denom) < 1e-12:
        return {}
    stone_fraction_by_move: dict[str, float] = {}
    for row in valid:
        move = str(row.get("move") or "").strip().lower()
        if not move or move == proxy_move:
            continue
        l_row = scoring.logit_clamped(float(row["mean_winrate"]))
        stone_fraction_by_move[move] = min(1.0, (l_row - l_proxy) / denom)
    return stone_fraction_by_move


def _tenuki_child_retention_allowed(*, node: JosekiNode) -> bool:
    return bool(node.entries) and not node.previous_was_tenuki


def _move_sets_scoring_baseline(
    *,
    node: JosekiNode,
    move: str,
    head_move_set: set[str],
    tenuki_move: str | None,
) -> bool:
    move_s = str(move)
    if move_s in head_move_set:
        return True
    return (
        isinstance(tenuki_move, str)
        and tenuki_move.strip() == move_s
        and not _tenuki_child_retention_allowed(node=node)
    )


def _prepare_node(
    node: JosekiNode,
    *,
    board_size: int,
    root_payload: dict[str, Any] | None,
    realized: RealizedPosition,
    normalized_moves: list[str | None] | None = None,
) -> PreparedExpansion:
    position, red, blue, to_play = realized
    local_moves, tenuki_move, meta_by_move = _select_candidates_from_root_payload(
        family=node.family,
        board_size=board_size,
        ply=len(node.entries),
        entries=node.entries,
        payload=(root_payload or {}),
        position=position,
        normalized_moves=normalized_moves,
    )
    occupied = set(red) | set(blue)
    local_meta_by_cell = {
        cell: {
            "local": local,
            **dict(meta_by_move.get(cell) or {}),
        }
        for cell, local in local_moves
    }
    for forced_local in _forced_override_children(family=node.family, parent_position=position):
        move = _family_move_to_cell(family=node.family, move=forced_local, board_size=board_size)
        col, row_num = hb.parse_cell(move)
        if (int(col), int(row_num)) in occupied or move in local_meta_by_cell:
            continue
        local_moves.append((move, forced_local))
        local_meta_by_cell[move] = {
            "local": forced_local,
            "cleaned_rank": None,
            "prior": None,
            "is_forced": True,
        }
    pass_proxy = _select_available_pass_proxy_move(board_size=board_size, to_play=to_play, occupied=occupied)
    candidates = [pass_proxy] + [cell for cell, _local in local_moves]
    if tenuki_move is not None and tenuki_move not in candidates:
        candidates.append(tenuki_move)
    return PreparedExpansion(
        parent_position=position,
        tenuki_move=tenuki_move,
        local_meta_by_cell=local_meta_by_cell,
        moves=tuple(candidates),
    )


def _can_skip_non_tenuki_child_expansion(*, node: JosekiNode) -> bool:
    return float(node.importance) * PLY_DECAY < _line_importance_min(family=node.family)


def _prune_prepared_to_non_local_children(prepared: PreparedExpansion) -> PreparedExpansion:
    return replace(
        prepared,
        local_meta_by_cell={},
        moves=tuple(move for move in prepared.moves if move not in prepared.local_meta_by_cell),
    )


def _child_evaluation_rows(
    *,
    parent_position: str,
    child_positions_by_move: dict[str, str],
    raw_nn_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    _board_size, _red, _blue, to_play = hb.position_state(parent_position)
    rows: list[dict[str, Any]] = []
    for idx, (move, child_position) in enumerate(child_positions_by_move.items()):
        payload = raw_nn_payloads.get(child_position)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing raw-NN payload for child position {child_position!r}")
        red_winrate = nn.red_winrate(payload)
        if not isinstance(red_winrate, float):
            raise ValueError(f"Raw-NN payload missing root winrate for child position {child_position!r}")
        winrate = float(red_winrate) if to_play == "red" else 1.0 - float(red_winrate)
        rows.append(
            {
                "move": move,
                "n": 1,
                "mean_winrate": winrate,
                "stdev_winrate": 0.0,
                "min_winrate": winrate,
                "max_winrate": winrate,
                "_idx": idx,
            }
        )
    rows.sort(key=lambda row: (-float(row["mean_winrate"]), int(row["_idx"]), str(row["move"])))
    scoring.attach_stone_fractions(
        rows,
        position_input=parent_position,
        allow_first_row_proxy_fallback=True,
    )
    return rows


def _node_key(node: JosekiNode) -> tuple[str, bool]:
    return (node.position, node.previous_was_tenuki)


def _merge_joseki_child(
    children_by_key: dict[tuple[str, bool], JosekiNode],
    *,
    child: JosekiNode,
) -> None:
    key = _node_key(child)
    existing = children_by_key.get(key)
    if existing is None:
        children_by_key[key] = child
    elif child.importance > existing.importance:
        children_by_key[key] = replace(existing, importance=child.importance)


def _finalize_node_expansion(
    node: JosekiNode,
    prepared: NodeExpansion,
    *,
    raw_nn_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[JosekiNode]]:
    rows = _child_evaluation_rows(
        parent_position=prepared.parent_position,
        child_positions_by_move=prepared.child_positions_by_move,
        raw_nn_payloads=raw_nn_payloads,
    )
    evaluations: list[CandidateEvaluation] = []
    for index, row in enumerate(rows):
        move = row["move"]
        meta = prepared.local_meta_by_cell.get(move)
        local = meta.get("local") if meta is not None else None
        override = (
            _child_override_rule(family=node.family, parent_position=prepared.parent_position, child=local)
            if local is not None else None
        )
        if local is not None:
            kind = "local"
        elif move == prepared.tenuki_move:
            kind = "tenuki"
        else:
            continue
        evaluations.append(CandidateEvaluation(
            index=index,
            move=move,
            kind=kind,
            local=local,
            meta=meta,
            stone_fraction=row.get("stone_fraction"),
            forced=override is not None or bool(meta and meta.get("is_forced")),
            override_importance=override,
        ))

    head: list[tuple[float, CandidateEvaluation]] = []
    for candidate in evaluations:
        fraction = candidate.stone_fraction
        if fraction is None:
            continue
        if candidate.kind == "local" and candidate.override_importance is not None:
            effective_fraction = candidate.override_importance
        else:
            minimum = TENUKI_STONE_FRACTION_MIN if candidate.kind == "tenuki" else STONE_FRACTION_MIN
            if fraction < minimum:
                continue
            effective_fraction = scoring.stone_fraction_for_importance(
                stone_fraction=fraction, child_ply=len(node.entries) + 1,
            )
        if candidate.kind == "tenuki":
            if not _tenuki_child_retention_allowed(node=node):
                continue
            effective_fraction *= TENUKI_IMPORTANCE_MULT
        head.append((effective_fraction, candidate))

    head.sort(key=lambda item: (-item[0], item[1].index))
    top_k = _top_k_for_ply(len(node.entries))
    if top_k is not None and len(head) > top_k:
        forced = [item for item in head if item[1].forced]
        ordinary = [item for item in head if not item[1].forced]
        head = sorted(forced + ordinary[:top_k], key=lambda item: (-item[0], item[1].index))
    head_moves = {candidate.move for _, candidate in head}
    head_local_count = sum(candidate.kind == "local" for _, candidate in head)
    tail_fractions = _candidate_stone_fractions_from_rows(
        rows,
        proxy_move=next(iter(prepared.child_positions_by_move), None),
        baseline_move_allowed=lambda move: _move_sets_scoring_baseline(
            node=node, move=move, head_move_set=head_moves, tenuki_move=prepared.tenuki_move,
        ),
    )

    def child_for(candidate: CandidateEvaluation, importance: float) -> JosekiNode:
        return JosekiNode(
            family=node.family,
            entries=node.entries + (candidate.local,),
            position=prepared.child_positions_by_move[candidate.move],
            importance=importance,
        )

    minimum_importance = _line_importance_min(family=node.family)
    retained: list[tuple[JosekiNode, CandidateEvaluation]] = []
    for effective_fraction, candidate in head:
        importance = node.importance * effective_fraction * PLY_DECAY
        if importance >= minimum_importance:
            retained.append((child_for(candidate, importance), candidate))

    tail: list[tuple[JosekiNode, CandidateEvaluation]] = []
    for candidate in evaluations:
        if candidate.move in head_moves or candidate.kind != "local" or candidate.forced:
            continue
        fraction = tail_fractions.get(candidate.move, candidate.stone_fraction)
        if fraction is None or fraction < STONE_FRACTION_MIN:
            continue
        importance = (
            node.importance
            * scoring.stone_fraction_for_importance(stone_fraction=fraction, child_ply=len(node.entries) + 1)
            * PLY_DECAY
            * _outside_top_k_candidate_weight(node, meta=candidate.meta, head_local_count=head_local_count)
        )
        if importance >= minimum_importance:
            tail.append((child_for(candidate, importance), candidate))
    retained.extend(sorted(tail, key=lambda item: (-item[0].importance, item[1].index)))

    candidate_rows: list[dict[str, Any]] = []
    row_by_move: dict[str, dict[str, Any]] = {}
    for candidate in evaluations:
        fraction = candidate.stone_fraction
        if candidate.kind == "local" and candidate.move not in head_moves:
            fraction = tail_fractions.get(candidate.move, fraction)
        row: dict[str, Any] = {
            "kind": candidate.kind,
            "stone_fraction": _rounded_stone_fraction(fraction),
        }
        if candidate.local is not None:
            row["local"] = list(candidate.local)
        candidate_rows.append(row)
        row_by_move[candidate.move] = row
    for child, candidate in retained:
        row_by_move[candidate.move]["child"] = child
    return {
        "line": node.line,
        "candidates": candidate_rows,
        "canonicalized_position": prepared.parent_position,
        "importance": _rounded_importance(node.importance),
    }, [child for child, _ in retained]


def build_joseki_database(
    *,
    family: str,
    board_size: int,
    output_path: Path | None = None,
    stop_after_depth: int | None = None,
) -> dict[str, Any]:
    family_s = str(family).strip().upper()
    if family_s not in {FAMILY_ACUTE, FAMILY_OBTUSE}:
        raise ValueError("family must be 'A' or 'O'")
    started_at = time.time()
    raw_nn_cache_path = _raw_nn_cache_path(family=family_s)
    raw_nn_cache = nn.load_cache(raw_nn_cache_path)
    root_position = _realize_position(
        family=family_s,
        board_size=board_size,
        realized_moves=(),
    )[0]
    root = JosekiNode(
        family=family_s,
        entries=(),
        position=root_position,
        importance=1.0,
    )
    frontier = [root]
    nodes: list[dict[str, Any]] = []
    completed_depth = -1
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        # This build allocates many short-lived acyclic containers while a very large
        # raw-NN cache dict is live. Disabling cyclic GC avoids repeated full-container
        # scans in the hot loop; refcount cleanup still handles ordinary temporaries.
        gc.disable()
    try:
        if frontier and completed_depth < 0:
            completed_depth = max(len(node.entries) for node in frontier) - 1
        while frontier:
            depth_started_at = time.time()
            realized_positions: list[RealizedPosition] = []
            for node in frontier:
                _size, red, blue, to_play = hb.position_state(node.position)
                realized_positions.append((node.position, red, blue, to_play))
            root_positions = list(
                dict.fromkeys(node.position for node in frontier if node.entries)
            )
            if root_positions:
                fetched_root_payloads, root_cache_hits = _evaluate_cached(
                    position_inputs=root_positions,
                    raw_nn_cache=raw_nn_cache,
                    raw_nn_cache_path=raw_nn_cache_path,
                )
            else:
                fetched_root_payloads = {}
                root_cache_hits = 0

            normalized_by_index = _normalized_candidate_rows(
                family=family_s,
                board_size=board_size,
                rows=[
                    (
                        realized[0],
                        len(node.entries),
                        fetched_root_payloads.get(node.position) or {},
                    )
                    for node, realized in zip(frontier, realized_positions)
                ],
            )

            prepared_selections: list[tuple[JosekiNode, PreparedExpansion]] = []
            for node, realized, normalized_moves in zip(
                frontier,
                realized_positions,
                normalized_by_index,
            ):
                prepared = _prepare_node(
                    node,
                    board_size=board_size,
                    root_payload=fetched_root_payloads.get(node.position),
                    realized=realized,
                    normalized_moves=normalized_moves,
                )
                if _can_skip_non_tenuki_child_expansion(node=node):
                    prepared = _prune_prepared_to_non_local_children(prepared)
                prepared_selections.append((node, prepared))
            expansion_requests: list[tuple[str, list[str]]] = []
            for _node, prepared in prepared_selections:
                expansion_requests.append((prepared.parent_position, list(prepared.moves)))
            expanded_positions = iter(
                native.child_positions(
                    requests=expansion_requests,
                    board_size=board_size,
                )
            )
            prepared_items: list[tuple[JosekiNode, NodeExpansion]] = []
            child_positions: list[str] = []
            seen_child_positions: set[str] = set()
            for node, prepared in prepared_selections:
                positions_by_move = dict(zip(prepared.moves, next(expanded_positions)))
                expanded = NodeExpansion(
                    parent_position=prepared.parent_position,
                    tenuki_move=prepared.tenuki_move,
                    local_meta_by_cell=prepared.local_meta_by_cell,
                    child_positions_by_move=positions_by_move,
                )
                prepared_items.append((node, expanded))
                for child_position in positions_by_move.values():
                    if child_position in seen_child_positions:
                        continue
                    seen_child_positions.add(child_position)
                    child_positions.append(child_position)
            child_cache_hits = nn.cached_winrate_count(
                raw_nn_cache,
                child_positions,
            )
            cache_hits = root_cache_hits + child_cache_hits
            cache_total = len(root_positions) + len(child_positions)
            if cache_hits < cache_total:
                _log(
                    f"starting depth={max(len(node.entries) for node in frontier)} "
                    f"nodes={len(frontier)} cache={cache_hits}/{cache_total}",
                    family=family_s,
                )
            child_policy_payloads: dict[str, dict[str, Any]] = {}
            child_raw_nn_payloads, _ = _evaluate_cached(
                position_inputs=child_positions,
                raw_nn_cache=raw_nn_cache,
                raw_nn_cache_path=raw_nn_cache_path,
                include_moves=False,
                store_moves=False,
                policy_payloads_out=child_policy_payloads,
            )
            layer_records: list[dict[str, Any]] = []
            layer_children: list[list[JosekiNode]] = []
            for node, prepared in prepared_items:
                record, children = _finalize_node_expansion(
                    node,
                    prepared,
                    raw_nn_payloads=child_raw_nn_payloads,
                )
                layer_records.append(record)
                layer_children.append(children)
            if len(layer_records) != len(frontier) or len(layer_children) != len(frontier):
                raise ValueError("joseki layer expansion did not produce one record per node")

            next_by_key: dict[tuple[str, bool], JosekiNode] = {}
            for children in layer_children:
                for child in children:
                    _merge_joseki_child(next_by_key, child=child)
            next_frontier = list(next_by_key.values())
            first_child_index = len(nodes) + len(layer_records)
            child_index_by_key = {
                key: first_child_index + idx
                for idx, key in enumerate(next_by_key)
            }
            for record in layer_records:
                for candidate in record["candidates"]:
                    child = candidate.get("child")
                    if not isinstance(child, JosekiNode):
                        continue
                    candidate["child"] = child_index_by_key[_node_key(child)]
            nodes.extend(layer_records)
            completed_depth = max(len(node.entries) for node in frontier)
            frontier = next_frontier
            frontier_positions = list(dict.fromkeys(node.position for node in frontier))
            cache_changed = False
            for position in frontier_positions:
                payload = child_policy_payloads.get(position)
                key = hb.serialized_position_key(position)
                if isinstance(payload, dict) and raw_nn_cache.get(key) != payload:
                    raw_nn_cache[key] = payload
                    cache_changed = True
            frontier_positions_without_moves = [
                position
                for position in frontier_positions
                if not nn.has_policy(
                    raw_nn_cache.get(hb.serialized_position_key(position))
                )
            ]
            if frontier_positions_without_moves:
                _evaluate_cached(
                    position_inputs=frontier_positions_without_moves,
                    raw_nn_cache=raw_nn_cache,
                    raw_nn_cache_path=raw_nn_cache_path,
                )
            elif cache_changed:
                nn.save_cache(raw_nn_cache_path, raw_nn_cache)
            if isinstance(output_path, Path):
                _write_joseki_artifact(
                    output_path,
                    _build_output_payload(
                        family=family_s,
                        board_size=board_size,
                        nodes=nodes,
                        completed=(len(frontier) == 0),
                        completed_depth=int(completed_depth),
                    ),
                )
            _log(
                f"depth={int(completed_depth)} "
                f"nodes={len(layer_records)}->{len(frontier)} cache={cache_hits}/{cache_total} "
                f"elapsed={runtime.format_duration(max(0.0, time.time() - depth_started_at))}",
                family=family_s,
            )
            if isinstance(stop_after_depth, int) and completed_depth >= int(stop_after_depth):
                break
    finally:
        if gc_was_enabled:
            gc.enable()

    payload = _build_output_payload(
        family=family_s,
        board_size=board_size,
        nodes=nodes,
        completed=(len(frontier) == 0),
        completed_depth=int(completed_depth),
    )
    status_verb = "Finished" if len(frontier) == 0 else "Stopped"
    _log(
        f"{status_verb} joseki build size={int(board_size)} "
        f"nodes={len(nodes)} elapsed={runtime.format_duration(max(0.0, time.time() - started_at))}",
        family=family_s,
    )
    return payload


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a raw-NN joseki database from fixed family-specific balance contexts")
    ap.add_argument("--family", choices=["A", "O", "a", "o"], required=True)
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stop-after-depth", type=int, default=None)
    ap.add_argument("--prune-cache", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    path = _resolve_output_path(
        out_arg=args.out,
        family=str(args.family),
        board_size=int(args.board_size),
    )
    if bool(args.prune_cache):
        backup_path, cache_path, before, after = _prune_raw_nn_cache(
            family=str(args.family),
            output_path=path,
        )
        print(f"{cache_path} {before}->{after} backup={backup_path}")
        return 0
    build_joseki_database(
        family=str(args.family),
        board_size=int(args.board_size),
        output_path=path,
        stop_after_depth=args.stop_after_depth,
    )
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
