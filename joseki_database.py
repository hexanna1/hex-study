#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dead_region_rules import canonicalize_acute_equivalent_move, should_exclude_acute_dead_region_move
import joseki_notation as jn
import local_pattern_representative as lpr
import study_common as lps
from local_pattern_representative import _serialize_position, parse_cell, point_to_cell


Move = tuple[int, int]
Entry = Move | None

FAMILY_ACUTE = "A"
FAMILY_OBTUSE = "O"
ACUTE_BALANCE_MOVES = ("d1", "c2")
OPPOSITE_OBTUSE_BALANCE_LOCAL_MOVES: tuple[Move, ...] = ((4, 1), (2, 2))
ROOT_LOCAL_MOVES: dict[str, tuple[Move, ...]] = {
    FAMILY_ACUTE: ((4, 3), (5, 4), (6, 5), (7, 6)),
    FAMILY_OBTUSE: ((4, 4), (5, 5), (6, 6), (4, 2)),
}
JOSEKI_CHILD_OVERRIDE_RULES: list[tuple[str, tuple[Entry, ...], Move, float]] = [
    (FAMILY_ACUTE, (), (4, 3), 1.00),
    (FAMILY_ACUTE, (), (5, 4), 1.00),
    (FAMILY_ACUTE, ((5, 4),), (3, 4), 0.95),
    (FAMILY_ACUTE, (), (6, 5), 1.00),
    (FAMILY_ACUTE, (), (7, 6), 1.00),
    (FAMILY_OBTUSE, (), (4, 4), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (2, 4), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (2, 3), 0.95),
    (FAMILY_OBTUSE, ((4, 4),), (5, 4), 0.95),
    (FAMILY_OBTUSE, (), (5, 5), 0.90),
    (FAMILY_OBTUSE, (), (6, 6), 0.85),
    (FAMILY_OBTUSE, (), (4, 2), 0.90),
]
LOCAL_DELTA_MAX_BY_FAMILY: dict[str, int] = {
    FAMILY_ACUTE: 91,
    FAMILY_OBTUSE: 49,
}
LOCAL_ANCHOR_DELTA_MAX = 7
LOCAL_ANCHOR_PRIOR_MIN = 0.001
LOCAL_LAST_ANCHOR_PRIOR_MIN = 0.0002
STONE_FRACTION_MIN = 0.85
TENUKI_STONE_FRACTION_MIN = 0.95
TENUKI_IMPORTANCE_MULT = 0.9
LINE_IMPORTANCE_MIN_BY_FAMILY: dict[str, float] = {
    FAMILY_ACUTE: 0.72,
    FAMILY_OBTUSE: 0.73,
}
PLY_DECAY = 0.994
RAW_NN_CACHE_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class JosekiNode:
    family: str
    entries: tuple[Entry, ...]
    realized_moves: tuple[str, ...]
    importance: float = 1.0

    @property
    def line(self) -> str:
        return jn.format_single_track_line(family=self.family, entries=self.entries)

    @property
    def consecutive_tenukis(self) -> int:
        count = 0
        for entry in reversed(self.entries):
            if entry is not None:
                break
            count += 1
        return count


def _ply_decay(*, family: str) -> float:
    return float(PLY_DECAY)


def _line_importance_min(*, family: str) -> float:
    fam = str(family).strip().upper()
    return float(LINE_IMPORTANCE_MIN_BY_FAMILY[fam])


def _child_override_rule(*, family: str, entries: tuple[Entry, ...], child: Move) -> float | None:
    fam = str(family).strip().upper()
    parent_entries = tuple(entries)
    child_move = (int(child[0]), int(child[1]))
    for rule_family, rule_entries, rule_child, importance in JOSEKI_CHILD_OVERRIDE_RULES:
        if str(rule_family).strip().upper() != fam:
            continue
        if tuple(rule_entries) != parent_entries:
            continue
        if (int(rule_child[0]), int(rule_child[1])) != child_move:
            continue
        return float(importance)
    return None


def _forced_override_children(*, family: str, entries: tuple[Entry, ...]) -> tuple[Move, ...]:
    fam = str(family).strip().upper()
    parent_entries = tuple(entries)
    forced: list[Move] = []
    for rule_family, rule_entries, rule_child, _importance in JOSEKI_CHILD_OVERRIDE_RULES:
        if str(rule_family).strip().upper() != fam:
            continue
        if tuple(rule_entries) != parent_entries:
            continue
        forced.append((int(rule_child[0]), int(rule_child[1])))
    return tuple(forced)


def _config_payload(*, family: str, board_size: int) -> dict[str, Any]:
    return {
        "version": 1,
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
    lps._log(f"{prefix}{message}")


def _fmt_s(sec: float) -> str:
    return lps._fmt_s(sec)


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


def _family_delta_metric(*, family: str, dq: int, dr: int) -> int:
    fam = str(family).strip().upper()
    if fam == FAMILY_ACUTE:
        return int(dq * dq + dq * dr + dr * dr)
    if fam == FAMILY_OBTUSE:
        return int(dq * dq - dq * dr + dr * dr)
    raise ValueError(f"unsupported family: {family!r}")


def _local_delta_max(*, family: str) -> int:
    fam = str(family).strip().upper()
    if fam not in LOCAL_DELTA_MAX_BY_FAMILY:
        raise ValueError(f"unsupported family: {family!r}")
    return int(LOCAL_DELTA_MAX_BY_FAMILY[fam])


def _corner_distance(*, family: str, local: Move) -> int:
    dq = int(local[0]) - 1
    dr = int(local[1]) - 1
    return _family_delta_metric(family=family, dq=dq, dr=dr)


def _delta_between(*, family: str, a: Move, b: Move) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return _family_delta_metric(family=family, dq=dq, dr=dr)


def _max_children_for_ply(ply: int) -> int | None:
    d = int(ply)
    if d <= 0:
        return None
    if d == 1:
        return 6
    if d == 2:
        return 5
    if d == 3:
        return 4
    return 3


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
    col, row = lps._cell_to_col_row(cell)
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
    red: list[Move] = []
    blue: list[Move] = []
    for i, token in enumerate(_balance_move_tokens(family=family, board_size=board_size)):
        cell = parse_cell(token)
        if i % 2 == 0:
            red.append(cell)
        else:
            blue.append(cell)
    return tuple(red), tuple(blue)


def _parse_line_entries(line: str) -> tuple[Entry, ...]:
    raw = str(line or "").strip()
    if not raw:
        return ()
    if not raw.endswith("]") or "[" not in raw:
        raise ValueError(f"invalid joseki line: {line!r}")
    inside = raw[raw.index("[") + 1 : -1]
    if inside == "":
        return ()
    entries: list[Entry] = []
    for tok in inside.split(":"):
        if tok == "":
            entries.append(None)
            continue
        a, b = tok.split(",")
        entries.append((int(a), int(b)))
    return tuple(entries)


def _candidate_children_for_position(
    *,
    family: str,
    entries: tuple[Entry, ...],
    position: str,
    root_payload: dict[str, Any] | None,
) -> tuple[set[str], dict[str, str]]:
    board_size, red, blue, to_play = lps._position_state(position)
    local_moves, tenuki_move, _rank_by_move = _select_candidates_from_root_payload(
        family=family,
        board_size=board_size,
        ply=len(entries),
        entries=entries,
        payload=(root_payload or {}),
        red=red,
        blue=blue,
    )
    occupied = set(red) | set(blue)
    pass_proxy = _pass_proxy_move(board_size=board_size, to_play=to_play, occupied=occupied)
    moves = [pass_proxy] + [cell for cell, _local in local_moves]
    if tenuki_move is not None and tenuki_move not in moves:
        moves.append(tenuki_move)
    move_to_position = {move: lps._position_after_move(position, move) for move in moves}
    child_positions = set(move_to_position.values())
    line_to_position: dict[str, str] = {}
    for cell, local in local_moves:
        child_line = jn.format_single_track_line(family=family, entries=entries + (local,))
        line_to_position[child_line] = move_to_position[cell]
    if tenuki_move is not None:
        child_line = jn.format_single_track_line(family=family, entries=entries + (None,))
        line_to_position[child_line] = move_to_position[tenuki_move]
    return child_positions, line_to_position


def _prune_raw_nn_cache(*, family: str, output_path: Path) -> tuple[Path, Path, int, int, int, int]:
    family_s = str(family).strip().upper()
    if not output_path.exists():
        raise FileNotFoundError(f"debug output not found for cache pruning: {output_path}")
    raw_nn_cache_path = _raw_nn_cache_path(family=family_s)
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError(f"invalid nodes payload in {output_path}")

    by_line = {str(node.get("line") or ""): node for node in nodes if isinstance(node, dict)}
    keep_positions: set[str] = set()
    frontier_positions: dict[str, str] = {}
    missing_materialized_payloads = 0

    for node in nodes:
        if not isinstance(node, dict):
            continue
        line = str(node.get("line") or "")
        position = str(node.get("canonicalized_position") or "").strip()
        if position:
            keep_positions.add(position)
        if not position:
            continue
        entries = _parse_line_entries(line)
        cache_payload = raw_nn_cache.get(lps._cache_key(position))
        if entries and not isinstance(cache_payload, dict):
            missing_materialized_payloads += 1
        child_positions, line_to_position = _candidate_children_for_position(
            family=family_s,
            entries=entries,
            position=position,
            root_payload=(cache_payload if isinstance(cache_payload, dict) else None),
        )
        keep_positions.update(child_positions)
        for retained_line_value in node.get("retained_lines", []) or []:
            retained_line = str(retained_line_value or "")
            if retained_line not in by_line and retained_line in line_to_position:
                frontier_positions[retained_line] = line_to_position[retained_line]

    missing_frontier_payloads = 0
    for line, position in frontier_positions.items():
        keep_positions.add(position)
        cache_payload = raw_nn_cache.get(lps._cache_key(position))
        if not isinstance(cache_payload, dict):
            missing_frontier_payloads += 1
            continue
        entries = _parse_line_entries(line)
        child_positions, _line_to_position = _candidate_children_for_position(
            family=family_s,
            entries=entries,
            position=position,
            root_payload=cache_payload,
        )
        keep_positions.update(child_positions)

    before = len(raw_nn_cache)
    keep_keys = {lps._cache_key(position) for position in keep_positions}
    pruned = {key: raw_nn_cache[key] for key in raw_nn_cache if key in keep_keys}

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = raw_nn_cache_path.with_name(f"{raw_nn_cache_path.stem}.backup-{timestamp}{raw_nn_cache_path.suffix}")
    lps._save_raw_nn_cache(backup_path, raw_nn_cache)
    lps._save_raw_nn_cache(raw_nn_cache_path, pruned)
    return backup_path, raw_nn_cache_path, before, len(pruned), missing_materialized_payloads, missing_frontier_payloads


def _realize_position(*, family: str, board_size: int, realized_moves: tuple[str, ...]) -> tuple[str, set[Move], set[Move], str]:
    red_cells, blue_cells = _balance_cells(family=family, board_size=board_size)
    red = set(red_cells)
    blue = set(blue_cells)
    side = "red"
    for token in realized_moves:
        col, row = lps._cell_to_col_row(token)
        point = (int(col), int(row))
        if point in red or point in blue:
            raise ValueError(f"duplicate occupied move in joseki realization: {token}")
        if side == "red":
            red.add(point)
            side = "blue"
        else:
            blue.add(point)
            side = "red"
    position = _serialize_position(
        board_size=int(board_size),
        red_cells=tuple(sorted(red)),
        blue_cells=tuple(sorted(blue)),
        to_play=side,
    )
    return position, red, blue, side


def _pass_proxy_move(*, board_size: int, to_play: str, occupied: set[Move]) -> str:
    m = (int(board_size) // 2) + 1
    side = str(to_play).strip().lower()
    if side not in {"red", "blue"}:
        raise ValueError(f"unsupported to_play value: {to_play!r}")
    vals = [m]
    for d in range(1, int(board_size)):
        if m - d >= 1:
            vals.append(m - d)
        if m + d <= int(board_size):
            vals.append(m + d)
    for v in vals:
        point = (int(v), 1) if side == "red" else (1, int(v))
        if point not in occupied:
            return point_to_cell(*point)
    raise ValueError(f"Could not place {side} pass proxy")


def _select_candidates_from_root_payload(
    *,
    family: str,
    board_size: int,
    ply: int,
    entries: tuple[Entry, ...],
    payload: dict[str, Any],
    red: set[Move],
    blue: set[Move],
) -> tuple[list[tuple[str, Move]], str | None, dict[str, int]]:
    occupied = set(red) | set(blue)
    if int(ply) == 0:
        root_moves = ROOT_LOCAL_MOVES.get(str(family).strip().upper(), ())
        filtered_root_moves: list[tuple[str, Move]] = []
        rank_by_move: dict[str, int] = {}
        for idx, local in enumerate(root_moves, start=1):
            move = _family_move_to_cell(family=family, move=local, board_size=board_size)
            col, row_num = lps._cell_to_col_row(move)
            if (int(col), int(row_num)) in occupied:
                continue
            if (
                str(family).strip().upper() == FAMILY_ACUTE
                and should_exclude_acute_dead_region_move(
                    move=(int(col), int(row_num)),
                    red=red,
                    blue=blue,
                    board_size=board_size,
                )
            ):
                continue
            if str(family).strip().upper() == FAMILY_ACUTE:
                col, row_num = canonicalize_acute_equivalent_move(
                    move=(int(col), int(row_num)),
                    red=red,
                    blue=blue,
                    board_size=board_size,
                )
                move = point_to_cell(int(col), int(row_num))
            filtered_root_moves.append((move, local))
            rank_by_move[move] = idx
        return filtered_root_moves, None, rank_by_move
    moves = lps._cached_payload_moves(payload)
    if not moves:
        return [], None, {}
    local_out: list[tuple[int, int, float, str, Move]] = []
    tenuki_move: str | None = None
    tenuki_best: tuple[int, int, float] | None = None
    for seq_idx, row in enumerate(moves):
        move_prior = lps._cached_payload_move_prior(row)
        if move_prior is None:
            continue
        move, prior = move_prior
        if not move:
            continue
        try:
            col, row_num = lps._cell_to_col_row(move)
            if (int(col), int(row_num)) in occupied:
                continue
            if (
                str(family).strip().upper() == FAMILY_ACUTE
                and should_exclude_acute_dead_region_move(
                    move=(int(col), int(row_num)),
                    red=red,
                    blue=blue,
                    board_size=board_size,
                )
            ):
                continue
            if str(family).strip().upper() == FAMILY_ACUTE:
                col, row_num = canonicalize_acute_equivalent_move(
                    move=(int(col), int(row_num)),
                    red=red,
                    blue=blue,
                    board_size=board_size,
                )
                move = point_to_cell(int(col), int(row_num))
            local = _cell_to_family_move(family=family, cell=move, board_size=board_size)
        except Exception:
            continue
        rank = seq_idx + 1
        prior = float(prior) if isinstance(prior, float) else float("-inf")
        if _corner_distance(family=family, local=local) <= _local_delta_max(family=family):
            local_out.append((rank, len(local_out), prior, move, local))
        else:
            tenuki_key = (rank, seq_idx, -prior)
            if tenuki_best is None or tenuki_key < tenuki_best:
                tenuki_best = tenuki_key
                tenuki_move = move

    if not local_out:
        return [], tenuki_move, {}

    anchors = [entry for entry in entries if isinstance(entry, tuple)]
    last_anchor = anchors[-1] if anchors else None
    if anchors:
        local_out = [
            item
            for item in local_out
            if any(_delta_between(family=family, a=item[4], b=anchor) <= LOCAL_ANCHOR_DELTA_MAX for anchor in anchors)
        ]
    if not local_out:
        return [], tenuki_move, ({tenuki_move: int(tenuki_best[0])} if tenuki_move is not None and tenuki_best is not None else {})

    local_out.sort(key=lambda item: (item[0], item[1], item[2]))
    best_local_move = local_out[0][3]
    filtered: list[tuple[str, Move]] = []
    rank_by_move: dict[str, int] = {}
    seen_cells: set[str] = set()
    for rank, idx, prior, move, local in local_out:
        near_last_anchor = (
            last_anchor is not None
            and _delta_between(family=family, a=local, b=last_anchor) <= LOCAL_ANCHOR_DELTA_MAX
        )
        candidate_prior_min = float(LOCAL_LAST_ANCHOR_PRIOR_MIN) if near_last_anchor else float(LOCAL_ANCHOR_PRIOR_MIN)
        if move != best_local_move and prior < candidate_prior_min:
            continue
        if move in seen_cells:
            continue
        seen_cells.add(move)
        filtered.append((move, local))
        rank_by_move[move] = int(rank)
    if tenuki_move is not None and tenuki_best is not None:
        rank_by_move[tenuki_move] = int(tenuki_best[0])
    return filtered, tenuki_move, rank_by_move


def _prepare_node(
    node: JosekiNode,
    *,
    board_size: int,
    root_payload: dict[str, Any] | None,
) -> tuple[str, str, str | None, dict[str, Move], dict[str, str]]:
    position, red, blue, to_play = _realize_position(family=node.family, board_size=board_size, realized_moves=node.realized_moves)
    local_moves, tenuki_move, _rank_by_move = _select_candidates_from_root_payload(
        family=node.family,
        board_size=board_size,
        ply=len(node.entries),
        entries=node.entries,
        payload=(root_payload or {}),
        red=red,
        blue=blue,
    )
    occupied = set(red) | set(blue)
    local_by_cell = {cell: local for cell, local in local_moves}
    for forced_local in _forced_override_children(family=node.family, entries=node.entries):
        move = _family_move_to_cell(family=node.family, move=forced_local, board_size=board_size)
        col, row_num = lps._cell_to_col_row(move)
        if (int(col), int(row_num)) in occupied or move in local_by_cell:
            continue
        local_moves.append((move, forced_local))
        local_by_cell[move] = forced_local
    pass_proxy = _pass_proxy_move(board_size=board_size, to_play=to_play, occupied=occupied)
    candidates = [pass_proxy] + [cell for cell, _local in local_moves]
    if tenuki_move is not None and tenuki_move not in candidates:
        candidates.append(tenuki_move)
    child_positions = {
        move: _realize_position(family=node.family, board_size=board_size, realized_moves=node.realized_moves + (move,))[0]
        for move in candidates
    }
    return position, to_play, tenuki_move, local_by_cell, child_positions


def _can_skip_non_tenuki_child_expansion(*, node: JosekiNode) -> bool:
    return float(node.importance) * _ply_decay(family=node.family) < _line_importance_min(family=node.family)


def _prune_prepared_to_non_local_children(
    prepared: tuple[str, str, str | None, dict[str, Move], dict[str, str]],
) -> tuple[str, str, str | None, dict[str, Move], dict[str, str]]:
    parent_position, parent_to_play, tenuki_move, local_by_cell, child_positions_by_move = prepared
    filtered_positions = {
        move: child_position
        for move, child_position in child_positions_by_move.items()
        if move not in local_by_cell
    }
    return parent_position, parent_to_play, tenuki_move, {}, filtered_positions


def _candidate_payload_from_analyze_payloads(
    *,
    parent_position: str,
    parent_to_play: str,
    child_positions_by_move: dict[str, str],
    analyze_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    moves_in_order = list(child_positions_by_move.keys())
    move_rows: list[dict[str, Any]] = []
    for move in moves_in_order:
        child_position = child_positions_by_move[move]
        payload = analyze_payloads.get(child_position)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing analyze payload for child position {child_position!r}")
        red_winrate = lps._cached_payload_red_winrate(payload)
        if not isinstance(red_winrate, float):
            raise ValueError(f"Analyze payload missing root eval for child position {child_position!r}")
        move_rows.append(
            {
                "move": move,
                "red_winrate": float(red_winrate),
                "visits": None,
            }
        )
    return {
        "ok": True,
        "mode": "candidate",
        "method": "raw_nn_via_child_analyze",
        "position": {"input": parent_position, "to_play": parent_to_play},
        "moves": move_rows,
    }


def _node_key(node: JosekiNode) -> tuple[str, tuple[Entry, ...], tuple[str, ...]]:
    return (node.family, tuple(node.entries), tuple(node.realized_moves))


def _position_for_node(*, node: JosekiNode, board_size: int) -> str:
    return _realize_position(family=node.family, board_size=board_size, realized_moves=node.realized_moves)[0]


def _prepare_record_or_state(
    node: JosekiNode,
    *,
    board_size: int,
    root_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, str, str | None, dict[str, Move], dict[str, str], dict[str, int]] | None]:
    position, _red, _blue, to_play = _realize_position(family=node.family, board_size=board_size, realized_moves=node.realized_moves)
    if node.consecutive_tenukis >= 2:
        record: dict[str, Any] = {
            "line": ("" if not node.entries else node.line),
            "candidates": [],
            "retained_lines": [],
            "canonicalized_position": position,
            "importance": _rounded_importance(node.importance),
        }
        return record, None
    return {}, _prepare_node(node, board_size=board_size, root_payload=root_payload)


def _finalize_node_expansion(
    node: JosekiNode,
    prepared: tuple[str, str, str | None, dict[str, Move], dict[str, str]],
    *,
    analyze_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[JosekiNode]]:
    parent_position, parent_to_play, tenuki_move, local_by_cell, child_positions_by_move = prepared
    candidate_payload = _candidate_payload_from_analyze_payloads(
        parent_position=parent_position,
        parent_to_play=parent_to_play,
        child_positions_by_move=child_positions_by_move,
        analyze_payloads=analyze_payloads,
    )
    rows = lps._aggregate_moves(candidate_payload, position_input=parent_position)
    children: list[JosekiNode] = []
    candidate_rows: list[dict[str, Any]] = []
    retained_lines: list[str] = []
    retained_candidates: list[tuple[bool, float, int, JosekiNode]] = []

    for idx, row in enumerate(rows):
        move = str(row.get("move") or "").strip().lower()
        stone_fraction = row.get("stone_fraction")
        kind = "pass_proxy"
        local = local_by_cell.get(move)
        override_rule = (
            _child_override_rule(family=node.family, entries=node.entries, child=local)
            if local is not None
            else None
        )
        override_importance = override_rule if override_rule is not None else None
        is_forced = override_rule is not None
        if local is not None:
            kind = "local"
        elif tenuki_move is not None and move == tenuki_move:
            kind = "tenuki"
        if kind == "pass_proxy":
            continue
        candidate_row: dict[str, Any] = {
            "kind": kind,
            "stone_fraction": _rounded_stone_fraction(stone_fraction),
        }
        if local is not None:
            candidate_row["local"] = [int(local[0]), int(local[1])]
        candidate_rows.append(candidate_row)
        if not isinstance(stone_fraction, (int, float)):
            continue
        if kind in {"local", "tenuki"}:
            min_fraction = TENUKI_STONE_FRACTION_MIN if kind == "tenuki" else STONE_FRACTION_MIN
            if kind == "local" and isinstance(override_importance, (int, float)):
                pass
            elif float(stone_fraction) < min_fraction:
                continue
        if kind == "local" and isinstance(override_importance, (int, float)):
            effective_fraction = float(override_importance)
        else:
            effective_fraction = float(stone_fraction)
        child_importance = float(node.importance) * effective_fraction * _ply_decay(family=node.family)
        if kind == "tenuki":
            child_importance *= TENUKI_IMPORTANCE_MULT
        if child_importance < _line_importance_min(family=node.family):
            continue
        if kind == "local":
            child = JosekiNode(
                family=node.family,
                entries=node.entries + (local,),
                realized_moves=node.realized_moves + (move,),
                importance=child_importance,
            )
        elif kind == "tenuki":
            if not node.entries:
                continue
            child = JosekiNode(
                family=node.family,
                entries=node.entries + (None,),
                realized_moves=node.realized_moves + (move,),
                importance=child_importance,
            )
        else:
            continue
        retained_candidates.append((bool(is_forced), float(child_importance), idx, child))

    child_cap = _max_children_for_ply(len(node.entries))
    retained_candidates.sort(key=lambda item: (-item[1], item[2]))
    if child_cap is not None and len(retained_candidates) > child_cap:
        forced_candidates = [item for item in retained_candidates if item[0]]
        ordinary_candidates = [item for item in retained_candidates if not item[0]]
        retained_candidates = forced_candidates + ordinary_candidates[:child_cap]
        retained_candidates.sort(key=lambda item: (-item[1], item[2]))
    for _rank, (_is_forced, _child_importance, _idx, child) in enumerate(retained_candidates, start=1):
        children.append(child)
        retained_lines.append(child.line)

    terminated = len(children) == 0
    termination_reason = "no_retained_continuations" if terminated else None
    record: dict[str, Any] = {
        "line": ("" if not node.entries else node.line),
        "candidates": candidate_rows,
        "retained_lines": retained_lines,
        "canonicalized_position": parent_position,
        "importance": _rounded_importance(node.importance),
    }
    return record, children


def _expand_node(
    node: JosekiNode,
    *,
    board_size: int,
    root_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[JosekiNode]]:
    record, prepared = _prepare_record_or_state(
        node,
        board_size=board_size,
        root_payload=root_payload,
    )
    if prepared is None:
        return record, []
    return _finalize_node_expansion(
        node,
        prepared,
        analyze_payloads=lps._run_multi_position_analyze(
            hexata_main=lps._hexata_main_path(),
            position_inputs=list(prepared[4].values()),
        ),
    )


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
    raw_nn_cache = lps._load_raw_nn_cache(raw_nn_cache_path)
    root = JosekiNode(family=family_s, entries=(), realized_moves=(), importance=1.0)
    frontier = [root]
    nodes = []
    completed_depth = -1
    seen: set[tuple[str, tuple[Entry, ...], tuple[str, ...]]] = set()
    for node in frontier:
        seen.add(_node_key(node))
    if frontier and completed_depth < 0:
        completed_depth = max(len(node.entries) for node in frontier) - 1
    while frontier:
        depth_started_at = time.time()
        positions = [_position_for_node(node=node, board_size=board_size) for node in frontier]
        root_positions = [position for node, position in zip(frontier, positions) if node.entries]
        if root_positions:
            fetched_root_payloads = _run_multi_position_analyze_cached(
                hexata_main=lps._hexata_main_path(),
                position_inputs=root_positions,
                raw_nn_cache=raw_nn_cache,
                raw_nn_cache_path=raw_nn_cache_path,
            )
        else:
            fetched_root_payloads = {}

        prepared_items: list[tuple[JosekiNode, tuple[str, str, str | None, dict[str, Move], dict[str, str]]] | None] = []
        immediate_records: list[dict[str, Any]] = []
        child_positions: list[str] = []
        seen_child_positions: set[str] = set()
        for node, position in zip(frontier, positions):
            record, prepared = _prepare_record_or_state(
                node,
                board_size=board_size,
                root_payload=fetched_root_payloads.get(position),
            )
            if prepared is None:
                immediate_records.append(record)
                prepared_items.append(None)
                continue
            if _can_skip_non_tenuki_child_expansion(node=node):
                prepared = _prune_prepared_to_non_local_children(prepared)
            prepared_items.append((node, prepared))
            for child_position in prepared[4].values():
                if child_position in seen_child_positions:
                    continue
                seen_child_positions.add(child_position)
                child_positions.append(child_position)
        requested_positions = root_positions + child_positions
        cache_hits = lps._cached_request_count(raw_nn_cache, requested_positions)
        cache_total = len(requested_positions)
        if cache_hits < cache_total:
            _log(
                f"starting depth={max(len(node.entries) for node in frontier)} "
                f"nodes={len(positions)} cache={cache_hits}/{cache_total}",
                family=family_s,
            )
        child_analyze_payloads = _run_multi_position_analyze_cached(
            hexata_main=lps._hexata_main_path(),
            position_inputs=child_positions,
            raw_nn_cache=raw_nn_cache,
            raw_nn_cache_path=raw_nn_cache_path,
        )

        next_frontier: list[JosekiNode] = []
        nodes.extend(immediate_records)
        for item in prepared_items:
            if item is None:
                continue
            node, prepared = item
            record, children = _finalize_node_expansion(
                node,
                prepared,
                analyze_payloads=child_analyze_payloads,
            )
            nodes.append(record)
            for child in children:
                key = _node_key(child)
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append(child)
        completed_depth = max(len(node.entries) for node in frontier)
        frontier = next_frontier
        if isinstance(output_path, Path):
            lps._write_json(
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
            f"nodes={len(positions)}->{len(frontier)} cache={cache_hits}/{cache_total} "
            f"elapsed={_fmt_s(max(0.0, time.time() - depth_started_at))}",
            family=family_s,
        )
        if isinstance(stop_after_depth, int) and completed_depth >= int(stop_after_depth):
            break

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
        f"nodes={len(nodes)} elapsed={_fmt_s(max(0.0, time.time() - started_at))}",
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
        backup_path, cache_path, before, after, missing_materialized, missing_frontier = _prune_raw_nn_cache(
            family=str(args.family),
            output_path=path,
        )
        print(f"{cache_path} {before}->{after} backup={backup_path}")
        if missing_materialized or missing_frontier:
            print(
                f"missing_materialized_root_payloads={missing_materialized} "
                f"missing_frontier_root_payloads={missing_frontier}"
            )
        return 0
    payload = build_joseki_database(
        family=str(args.family),
        board_size=int(args.board_size),
        output_path=path,
        stop_after_depth=args.stop_after_depth,
    )
    lps._write_json(path, payload)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
