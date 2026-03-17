from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from hex_symmetry import ROTATIONS, D6, apply_transform_ax


Point = tuple[int, int]  # (col, row) using board-style axial coordinates
EDGE_BILATERAL_TRANSFORM_IDS = (0, 6)


def _center_index(board_size: int) -> int:
    # Ceiling center for even boards, exact center for odd boards.
    return (int(board_size) // 2) + 1


def _hexata_dir() -> Path:
    return (Path(__file__).resolve().parent / ".." / "hexata").resolve()


def _ensure_hexata_imports() -> None:
    hexata = str(_hexata_dir())
    if hexata not in sys.path:
        sys.path.insert(0, hexata)


@lru_cache(maxsize=1)
def _imports() -> tuple[Any, Any, Any, Any, Any, Any, int, int]:
    _ensure_hexata_imports()
    from board import MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard, MoveKind, Side, coord_to_human  # type: ignore
    from hexworld import cell_to_col_row, parse_hexworld_position  # type: ignore

    return (
        HexBoard,
        MoveKind,
        Side,
        coord_to_human,
        cell_to_col_row,
        parse_hexworld_position,
        int(MIN_BOARD_SIZE),
        int(MAX_BOARD_SIZE),
    )


(
    HEXBOARD_CLS,
    MOVE_KIND,
    SIDE_ENUM,
    COORD_TO_HUMAN,
    CELL_TO_COL_ROW,
    PARSE_HEXWORLD_POSITION,
    MIN_BOARD_SIZE,
    MAX_BOARD_SIZE,
) = _imports()


@dataclass(frozen=True)
class ExtractedPattern:
    source_input: str
    board_size_source: int
    to_play_at_cursor: str  # red | blue
    plus_cells: tuple[str, ...]
    minus_cells: tuple[str, ...]
    source_cells_all: tuple[str, ...]
    plus_rel: tuple[Point, ...]
    minus_rel: tuple[Point, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "source_input": self.source_input,
            "board_size_source": self.board_size_source,
            "to_play_at_cursor": self.to_play_at_cursor,
            "plus_cells": list(self.plus_cells),
            "minus_cells": list(self.minus_cells),
            "source_cells_all": list(self.source_cells_all),
            "plus_rel": [list(p) for p in self.plus_rel],
            "minus_rel": [list(p) for p in self.minus_rel],
        }


@dataclass(frozen=True)
class Orientation:
    index: int
    transform_id: int
    norm_shift: Point
    plus_rel: tuple[Point, ...]
    minus_rel: tuple[Point, ...]


@dataclass(frozen=True)
class BalanceProfile:
    index: int
    moves: tuple[str, ...]
    red_cells: tuple[Point, ...]
    blue_cells: tuple[Point, ...]


@dataclass(frozen=True)
class Representative:
    balance_index: int
    orientation_index: int
    orientation_transform_id: int
    orientation_norm_shift: Point
    placement_offset: Point
    balance_moves: tuple[str, ...]
    to_play_at_cursor: str
    plus_abs: tuple[Point, ...]
    minus_abs: tuple[Point, ...]
    red_cells: tuple[Point, ...]
    blue_cells: tuple[Point, ...]
    position: str


def _cell_sort_key(cell: Point) -> tuple[int, int]:
    return (cell[0], cell[1])


def parse_cell(token: str) -> Point:
    col, row = CELL_TO_COL_ROW(token.strip().lower())
    return int(col), int(row)


def point_to_cell(col: int, row: int) -> str:
    return COORD_TO_HUMAN(col, row).lower()


def extract_pattern(source: str) -> ExtractedPattern:
    size, past_moves, _future_moves, to_play = PARSE_HEXWORLD_POSITION(source)
    board = HEXBOARD_CLS(size)

    for idx, mv in enumerate(past_moves, start=1):
        ok = False
        if mv.kind == MOVE_KIND.PLACE:
            ok = board.place(mv.side, mv.col, mv.row)
        elif mv.kind == MOVE_KIND.PASS:
            ok = board.pass_move(mv.side)
        elif mv.kind == MOVE_KIND.SWAP:
            ok = board.swap_move(mv.side)
        if not ok:
            raise ValueError(f"Illegal past move at index {idx}")

    occupied: list[tuple[Point, int]] = []
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            v = board.get(col, row)
            if v >= 0:
                occupied.append(((col, row), int(v)))
    if not occupied:
        raise ValueError("Source has zero occupied cells at cursor")

    plus_side = SIDE_ENUM.RED if to_play == SIDE_ENUM.RED else SIDE_ENUM.BLUE
    plus_num = int(plus_side)

    plus_pts = tuple(sorted([pt for pt, side in occupied if side == plus_num], key=_cell_sort_key))
    minus_pts = tuple(sorted([pt for pt, side in occupied if side != plus_num], key=_cell_sort_key))
    all_pts = tuple(sorted([pt for pt, _ in occupied], key=_cell_sort_key))

    anchor_q, anchor_r = min(all_pts)
    plus_rel = tuple(sorted(((q - anchor_q, r - anchor_r) for q, r in plus_pts), key=_cell_sort_key))
    minus_rel = tuple(sorted(((q - anchor_q, r - anchor_r) for q, r in minus_pts), key=_cell_sort_key))

    return ExtractedPattern(
        source_input=source,
        board_size_source=size,
        to_play_at_cursor=("red" if to_play == SIDE_ENUM.RED else "blue"),
        plus_cells=tuple(point_to_cell(q, r) for q, r in plus_pts),
        minus_cells=tuple(point_to_cell(q, r) for q, r in minus_pts),
        source_cells_all=tuple(point_to_cell(q, r) for q, r in all_pts),
        plus_rel=plus_rel,
        minus_rel=minus_rel,
    )


def _normalize_rel(
    plus: Iterable[Point],
    minus: Iterable[Point],
) -> tuple[tuple[Point, ...], tuple[Point, ...], Point]:
    plus_t = tuple(plus)
    minus_t = tuple(minus)
    all_pts = plus_t + minus_t
    if not all_pts:
        raise ValueError("Pattern has no stones")
    min_q, min_r = min(all_pts)
    plus_n = tuple(sorted(((q - min_q, r - min_r) for q, r in plus_t), key=_cell_sort_key))
    minus_n = tuple(sorted(((q - min_q, r - min_r) for q, r in minus_t), key=_cell_sort_key))
    return plus_n, minus_n, (min_q, min_r)


def _apply_transform(pts: Iterable[Point], transform_id: int) -> tuple[Point, ...]:
    return tuple(apply_transform_ax(p, transform_id) for p in pts)


def _transform_ids_for_policy(symmetry: str) -> list[int]:
    policy = symmetry.strip().lower()
    if policy == "identity":
        return [0]
    if policy == "edge-bilateral":
        return list(EDGE_BILATERAL_TRANSFORM_IDS)
    if policy == "rotations":
        return list(range(len(ROTATIONS)))
    if policy == "d6":
        return list(range(len(D6)))
    raise ValueError("symmetry must be one of: identity, edge-bilateral, rotations, d6")


def expand_orientations(plus_rel: tuple[Point, ...], minus_rel: tuple[Point, ...], symmetry: str) -> list[Orientation]:
    transform_ids = _transform_ids_for_policy(symmetry)

    seen: set[tuple[tuple[Point, ...], tuple[Point, ...]]] = set()
    out: list[Orientation] = []
    for ti in transform_ids:
        plus_t = _apply_transform(plus_rel, ti)
        minus_t = _apply_transform(minus_rel, ti)
        plus_n, minus_n, shift = _normalize_rel(plus_t, minus_t)
        key = (plus_n, minus_n)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Orientation(
                index=len(out) + 1,
                transform_id=ti,
                norm_shift=shift,
                plus_rel=plus_n,
                minus_rel=minus_n,
            )
        )
    return out


def _local_key_orbit_for_pattern(
    *,
    plus_rel: tuple[Point, ...],
    minus_rel: tuple[Point, ...],
    symmetry: str,
) -> list[dict[str, Any]]:
    base_plus, base_minus, _base_shift = _normalize_rel(plus_rel, minus_rel)
    out: list[dict[str, Any]] = []
    for ti in _transform_ids_for_policy(symmetry):
        plus_t = _apply_transform(base_plus, ti)
        minus_t = _apply_transform(base_minus, ti)
        plus_n, minus_n, shift = _normalize_rel(plus_t, minus_t)
        if plus_n != base_plus or minus_n != base_minus:
            continue
        out.append(
            {
                "transform_id": int(ti),
                "norm_shift": [int(shift[0]), int(shift[1])],
            }
        )
    if not out:
        return [{"transform_id": 0, "norm_shift": [0, 0]}]
    return out


def _round_half_away(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def snap_hex(qf: float, rf: float) -> Point:
    x = qf
    z = rf
    y = -x - z

    rx = _round_half_away(x)
    ry = _round_half_away(y)
    rz = _round_half_away(z)

    dx = abs(rx - x)
    dy = abs(ry - y)
    dz = abs(rz - z)

    # Deterministic tie-break: x, then y, then z.
    if dx >= dy and dx >= dz:
        rx = -ry - rz
    elif dy >= dz:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(rz)


def _centroid(points: tuple[Point, ...]) -> tuple[float, float]:
    if not points:
        raise ValueError("centroid requires at least one point")
    n = float(len(points))
    return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n


def _translate(points: tuple[Point, ...], dq: int, dr: int) -> tuple[Point, ...]:
    return tuple((q + dq, r + dr) for q, r in points)


def parse_balance_profiles(raw: list[str] | None) -> list[BalanceProfile]:
    if not raw:
        return []
    out: list[BalanceProfile] = []
    for i, spec in enumerate(raw, start=1):
        s = spec.strip().lower()
        if s in {"", "none"}:
            out.append(BalanceProfile(index=i, moves=(), red_cells=(), blue_cells=()))
            continue
        toks = [t.strip().lower() for t in s.split(",") if t.strip()]
        if not toks:
            out.append(BalanceProfile(index=i, moves=(), red_cells=(), blue_cells=()))
            continue
        pts = [parse_cell(t) for t in toks]
        if len(set(pts)) != len(pts):
            raise ValueError(f"duplicate balance move in profile: {spec!r}")
        red: list[Point] = []
        blue: list[Point] = []
        for j, p in enumerate(pts):
            if j % 2 == 0:
                red.append(p)
            else:
                blue.append(p)
        out.append(
            BalanceProfile(
                index=i,
                moves=tuple(toks),
                red_cells=tuple(sorted(red, key=_cell_sort_key)),
                blue_cells=tuple(sorted(blue, key=_cell_sort_key)),
            )
        )
    return out


def _in_bounds(p: Point, size: int) -> bool:
    return 1 <= p[0] <= size and 1 <= p[1] <= size


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _scan_center_out(n: int, mid: int) -> list[int]:
    out = [mid]
    for d in range(1, n):
        lo = mid - d
        hi = mid + d
        if lo >= 1:
            out.append(lo)
        if hi <= n:
            out.append(hi)
    return out


def _select_pass_proxy_cell(*, board_size: int, to_play: str, occupied: set[Point]) -> Point:
    m = _center_index(board_size)
    side = to_play.strip().lower()
    if side == "red":
        for col in _scan_center_out(board_size, m):
            p = (col, 1)
            if p not in occupied:
                return p
        raise ValueError("Could not place red pass-proxy on first row")
    if side == "blue":
        for row in _scan_center_out(board_size, m):
            p = (1, row)
            if p not in occupied:
                return p
        raise ValueError("Could not place blue pass-proxy on first column")
    raise ValueError(f"Unsupported to_play value: {to_play!r}")


def _prepend_proxy(candidates: list[str], proxy_cell: str) -> list[str]:
    if proxy_cell in candidates:
        return [proxy_cell] + [x for x in candidates if x != proxy_cell]
    return [proxy_cell] + candidates


def _validate_board_size(board_size: int) -> None:
    if not (MIN_BOARD_SIZE <= board_size <= MAX_BOARD_SIZE):
        raise ValueError(
            f"Target board size {board_size} out of supported range [{MIN_BOARD_SIZE}, {MAX_BOARD_SIZE}]"
        )


def _validate_balance_profiles_for_board(balance_profiles: list[BalanceProfile], board_size: int) -> None:
    for bal in balance_profiles:
        for tok in bal.moves:
            col, row = parse_cell(tok)
            if not (1 <= col <= board_size and 1 <= row <= board_size):
                raise ValueError(
                    f"Balance profile #{bal.index} has out-of-bounds move {tok!r} for board size {board_size}"
                )


def _placement_offset(
    *,
    plus_rel: tuple[Point, ...],
    minus_rel: tuple[Point, ...],
    board_size: int,
    placement: str,
    edge_anchor_col_from_right: int = 1,
) -> Point:
    all_rel = tuple(sorted(plus_rel + minus_rel, key=_cell_sort_key))
    if not all_rel:
        raise ValueError("Pattern has no stones")

    placement_s = placement.strip().lower()
    c_q, c_r = _centroid(all_rel)
    s_q, s_r = snap_hex(c_q, c_r)
    if placement_s == "centered":
        center = _center_index(board_size)
        return center - s_q, center - s_r

    if placement_s == "edge":
        target_anchor_col = board_size - int(edge_anchor_col_from_right) + 1
        if not (1 <= target_anchor_col <= board_size):
            raise ValueError("edge_anchor_col_from_right must be within [1, board_size]")
        anchor_rel = _select_edge_anchor(plus_rel=plus_rel, minus_rel=minus_rel)
        max_r = max(r for _q, r in all_rel)
        center = _center_index(board_size)
        return target_anchor_col - anchor_rel[0], _clamp(center - s_r, 1, board_size - max_r)

    raise ValueError("placement must be 'centered' or 'edge'")


def _select_edge_anchor(*, plus_rel: tuple[Point, ...], minus_rel: tuple[Point, ...]) -> Point:
    points = plus_rel + minus_rel
    if not points:
        raise ValueError("Pattern has no stones")
    return min(points, key=lambda p: (-p[0], p[1]))


def _serialize_position_stream(
    *,
    red_cells: tuple[Point, ...],
    blue_cells: tuple[Point, ...],
    to_play: str,
) -> str:
    red = list(sorted(red_cells, key=_cell_sort_key))
    blue = list(sorted(blue_cells, key=_cell_sort_key))
    if set(red) & set(blue):
        raise ValueError("red/blue overlap in serialization input")

    ri = 0
    bi = 0
    side = "red"
    stream: list[str] = []

    while ri < len(red) or bi < len(blue):
        if side == "red":
            if ri < len(red):
                col, row = red[ri]
                stream.append(point_to_cell(col, row))
                ri += 1
            else:
                stream.append(":p")
            side = "blue"
        else:
            if bi < len(blue):
                col, row = blue[bi]
                stream.append(point_to_cell(col, row))
                bi += 1
            else:
                stream.append(":p")
            side = "red"

    if side != to_play:
        stream.append(":p")

    return "".join(stream)


def _serialize_position(
    *,
    board_size: int,
    red_cells: tuple[Point, ...],
    blue_cells: tuple[Point, ...],
    to_play: str,
) -> str:
    past_stream = _serialize_position_stream(
        red_cells=red_cells,
        blue_cells=blue_cells,
        to_play=to_play,
    )
    return f"https://hexworld.org/board/#{board_size}c1,{past_stream}"


def _validate_position(
    *,
    position: str,
    expected_red: set[Point],
    expected_blue: set[Point],
    expected_to_play: str,
) -> None:
    size, past_moves, _future_moves, to_play = PARSE_HEXWORLD_POSITION(position)
    board = HEXBOARD_CLS(size)
    for idx, mv in enumerate(past_moves, start=1):
        ok = False
        if mv.kind == MOVE_KIND.PLACE:
            ok = board.place(mv.side, mv.col, mv.row)
        elif mv.kind == MOVE_KIND.PASS:
            ok = board.pass_move(mv.side)
        elif mv.kind == MOVE_KIND.SWAP:
            ok = board.swap_move(mv.side)
        if not ok:
            raise ValueError(f"Serialized position illegal at move {idx}")

    got_red: set[Point] = set()
    got_blue: set[Point] = set()
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            v = board.get(col, row)
            if v == 0:
                got_red.add((col, row))
            elif v == 1:
                got_blue.add((col, row))

    got_to_play = "red" if to_play == SIDE_ENUM.RED else "blue"
    if got_red != expected_red or got_blue != expected_blue or got_to_play != expected_to_play:
        raise ValueError("Serialized position does not reproduce expected occupancy/to-play")


def generate_representatives(
    *,
    extracted: ExtractedPattern,
    board_size: int,
    symmetry: str,
    balance_profiles: list[BalanceProfile],
    placement: str = "centered",
    edge_anchor_col_from_right: int = 1,
) -> list[Representative]:
    _validate_board_size(board_size)
    placement_s = placement.strip().lower()
    if placement_s != "edge" and int(edge_anchor_col_from_right) != 1:
        raise ValueError("edge_anchor_col_from_right requires placement='edge'")
    symmetry_s = symmetry.strip().lower()
    if placement_s == "edge" and symmetry_s not in {"identity", "edge-bilateral"}:
        raise ValueError("placement='edge' requires symmetry in {'identity', 'edge-bilateral'}")

    orientations = expand_orientations(extracted.plus_rel, extracted.minus_rel, symmetry)
    if not orientations:
        raise ValueError("No orientations produced")

    if not balance_profiles:
        balance_profiles = [BalanceProfile(index=1, moves=(), red_cells=(), blue_cells=())]
    _validate_balance_profiles_for_board(balance_profiles, board_size)

    out: list[Representative] = []
    seen_full: set[str] = set()

    for bal in balance_profiles:
        for ori in orientations:
            dq, dr = _placement_offset(
                plus_rel=ori.plus_rel,
                minus_rel=ori.minus_rel,
                board_size=board_size,
                placement=placement,
                edge_anchor_col_from_right=edge_anchor_col_from_right,
            )

            plus_abs = _translate(ori.plus_rel, dq, dr)
            minus_abs = _translate(ori.minus_rel, dq, dr)

            if any(not _in_bounds(p, board_size) for p in plus_abs + minus_abs):
                continue

            plus_set = set(plus_abs)
            minus_set = set(minus_abs)
            if plus_set & minus_set:
                continue

            if extracted.to_play_at_cursor == "red":
                pattern_red = plus_set
                pattern_blue = minus_set
            else:
                pattern_red = minus_set
                pattern_blue = plus_set

            red_all = set(bal.red_cells) | pattern_red
            blue_all = set(bal.blue_cells) | pattern_blue
            if red_all & blue_all:
                continue
            if any(not _in_bounds(p, board_size) for p in red_all | blue_all):
                continue

            position = _serialize_position(
                board_size=board_size,
                red_cells=tuple(sorted(red_all, key=_cell_sort_key)),
                blue_cells=tuple(sorted(blue_all, key=_cell_sort_key)),
                to_play=extracted.to_play_at_cursor,
            )
            if position in seen_full:
                continue
            _validate_position(
                position=position,
                expected_red=red_all,
                expected_blue=blue_all,
                expected_to_play=extracted.to_play_at_cursor,
            )
            seen_full.add(position)

            out.append(
                Representative(
                    balance_index=bal.index,
                    orientation_index=ori.index,
                    orientation_transform_id=ori.transform_id,
                    orientation_norm_shift=ori.norm_shift,
                    placement_offset=(dq, dr),
                    balance_moves=bal.moves,
                    to_play_at_cursor=extracted.to_play_at_cursor,
                    plus_abs=tuple(sorted(plus_abs, key=_cell_sort_key)),
                    minus_abs=tuple(sorted(minus_abs, key=_cell_sort_key)),
                    red_cells=tuple(sorted(red_all, key=_cell_sort_key)),
                    blue_cells=tuple(sorted(blue_all, key=_cell_sort_key)),
                    position=position,
                )
            )

    if not out:
        raise ValueError("All representatives filtered out")
    return out


def _Δ(a: Point, b: Point) -> int:
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return int(dq * dq + dq * dr + dr * dr)


def build_candidates_for_representative(
    *,
    rep: Representative,
    board_size: int,
    mode: str,
    explicit: list[str],
    candidate_Δ_max: int,
) -> list[str]:
    mode_s = mode.strip().lower()
    occupied = set(rep.red_cells) | set(rep.blue_cells)
    proxy_cell = point_to_cell(*_select_pass_proxy_cell(board_size=board_size, to_play=rep.to_play_at_cursor, occupied=occupied))

    if mode_s == "explicit":
        if not explicit:
            raise ValueError("explicit candidate mode requires at least one candidate")
        out: list[str] = []
        seen: set[Point] = set()
        for tok in explicit:
            p = parse_cell(tok)
            if p in seen:
                continue
            seen.add(p)
            if not _in_bounds(p, board_size):
                raise ValueError(f"Explicit candidate out of bounds: {tok}")
            if p in occupied:
                raise ValueError(f"Explicit candidate occupied: {tok}")
            out.append(point_to_cell(*p))
        if not out:
            raise ValueError("explicit candidate mode produced empty set")
        return _prepend_proxy(out, proxy_cell)

    if mode_s != "auto-near-pattern":
        raise ValueError("candidate mode must be 'explicit' or 'auto-near-pattern'")

    if candidate_Δ_max < 0:
        raise ValueError("candidate_Δ_max must be >= 0")

    pattern_cells = set(rep.plus_abs) | set(rep.minus_abs)
    if not pattern_cells:
        raise ValueError("pattern cells empty")

    cand: list[Point] = []
    for row in range(1, board_size + 1):
        for col in range(1, board_size + 1):
            p = (col, row)
            if p in occupied:
                continue
            if min(_Δ(p, q) for q in pattern_cells) <= candidate_Δ_max:
                cand.append(p)

    cand = sorted(cand, key=_cell_sort_key)
    out = [point_to_cell(col, row) for col, row in cand]
    if not out:
        raise ValueError("Candidate set empty after auto-near-pattern generation")
    return _prepend_proxy(out, proxy_cell)


def build_study_spec(
    *,
    extracted: ExtractedPattern,
    representatives: list[Representative],
    board_size: int,
    symmetry: str,
    candidate_mode: str,
    explicit_candidates: list[str],
    candidate_Δ_max: int,
    search_seconds: float | None = None,
    awrn: float | None = None,
) -> dict[str, Any]:
    if not representatives:
        raise ValueError("No representatives provided")

    exps: list[dict[str, Any]] = []
    exp_meta: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    local_key_orbit = _local_key_orbit_for_pattern(
        plus_rel=extracted.plus_rel,
        minus_rel=extracted.minus_rel,
        symmetry=symmetry,
    )

    for i, rep in enumerate(representatives, start=1):
        candidates = build_candidates_for_representative(
            rep=rep,
            board_size=board_size,
            mode=candidate_mode,
            explicit=explicit_candidates,
            candidate_Δ_max=candidate_Δ_max,
        )
        name = f"b{rep.balance_index:02d}-o{rep.orientation_index:02d}"
        if name in seen_names:
            name = f"{name}-{i:03d}"
        seen_names.add(name)
        exp_meta[name] = {
            "orientation_transform_id": int(rep.orientation_transform_id),
            "orientation_norm_shift": [int(rep.orientation_norm_shift[0]), int(rep.orientation_norm_shift[1])],
            "placement_offset": [int(rep.placement_offset[0]), int(rep.placement_offset[1])],
            "local_key_orbit": [dict(x) for x in local_key_orbit],
        }
        balance_label = "none" if not rep.balance_moves else ",".join(rep.balance_moves)
        exps.append(
            {
                "name": name,
                "label": f"balance={balance_label} ori={rep.orientation_index}",
                "position": rep.position,
                "candidates": candidates,
            }
        )

    defaults: dict[str, Any] = {}
    if search_seconds is not None:
        defaults["search_seconds"] = float(search_seconds)
    if awrn is not None:
        defaults["awrn"] = float(awrn)

    return {
        "defaults": defaults,
        "experiments": exps,
        "generator_meta": {
            "source": extracted.source_input,
            "source_board_size": extracted.board_size_source,
            "source_to_play": extracted.to_play_at_cursor,
            "target_board_size": board_size,
            "symmetry_policy": symmetry.strip().lower(),
            "candidate_mode": candidate_mode,
            "candidate_Δ_max": candidate_Δ_max,
            "representative_count": len(representatives),
            "experiment_meta": exp_meta,
        },
    }
