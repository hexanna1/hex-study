"""HexWorld move replay, board-state serialization, and board coordinates."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_hexata = str(Path(__file__).resolve().parent.parent / "hexata")
if _hexata not in sys.path:
    sys.path.insert(0, _hexata)

from board import Board, Side, MIN_BOARD_SIZE, MAX_BOARD_SIZE, coord_to_human
from formats.hexworld import parse_hexworld_position, cell_to_col_row

Point = tuple[int, int]
CELL_PARSE_RE = re.compile(r"^([A-Za-z]+)([1-9][0-9]*)$")
SIZE_RE = re.compile(r"^\s*([0-9]+)")


def center_index(board_size: int) -> int:
    # Ceiling center for even boards, exact center for odd boards.
    return board_size // 2 + 1


def board_size_from_position(position_input: str) -> int | None:
    s = str(position_input or "").strip()
    if not s:
        return None
    frag = s.split("#", 1)[1] if "#" in s else s
    m = SIZE_RE.match(frag)
    if not m:
        return None
    return int(m.group(1))


def position_cache_key(position_input: str, move: str | None = None) -> str:
    """Replay ordered moves before deriving an occupancy-based NN cache key."""
    position = str(position_input or "").strip()
    if move is not None:
        token = str(move).strip().lower()
        if not token:
            raise ValueError("Missing move token for child cache key")
        position = position_after_move(position, token)
    size, red, blue, to_play = position_state(position)
    past_stream = serialize_position_stream(
        red_cells=tuple(sorted(red)),
        blue_cells=tuple(sorted(blue)),
        to_play=to_play,
    )
    return f"{int(size)},{past_stream}" if past_stream else str(int(size))


def serialized_position_key(position_input: str) -> str:
    """Extract the key from a URL already produced by serialize_position."""
    position = str(position_input or "").strip()
    fragment = position.split("#", 1)[1] if "#" in position else position
    marker = "c1,"
    if marker not in fragment:
        raise ValueError(f"Expected a serialized HexWorld position, got {position_input!r}")
    size_text, stream = fragment.split(marker, 1)
    if not size_text.isdigit():
        raise ValueError(f"Expected a serialized HexWorld position, got {position_input!r}")
    size = str(int(size_text))
    return f"{size},{stream}" if stream else size


def canonical_pass_proxy_move(board_size: int, to_play: str) -> str | None:
    m = center_index(board_size)
    side = str(to_play or "").strip().lower()
    if side == "red":
        return point_to_cell(m, 1)
    if side == "blue":
        return point_to_cell(1, m)
    return None


def parse_cell(cell: str) -> tuple[int, int]:
    m = CELL_PARSE_RE.fullmatch(cell.strip())
    if not m:
        raise ValueError(f"Bad cell token: {cell!r}")
    return cell_to_col_row(cell.strip().lower())


def position_state(position: str) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], str]:
    size, past_moves, _future_moves, to_play = parse_hexworld_position(position)
    board = Board(size)
    for idx, mv in enumerate(past_moves, start=1):
        if not board.apply_move(mv):
            raise ValueError(f"Illegal past move at index {idx} for {position!r}")
    red: set[tuple[int, int]] = set()
    blue: set[tuple[int, int]] = set()
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            value = board.get(col, row)
            if value == Side.RED:
                red.add((col, row))
            elif value == Side.BLUE:
                blue.add((col, row))
    to_play_s = "red" if to_play == Side.RED else "blue"
    return size, red, blue, to_play_s


def position_after_move(position: str, move: str) -> str:
    size, red, blue, to_play = position_state(position)
    return position_after_move_from_state(
        size=size,
        red=red,
        blue=blue,
        to_play=to_play,
        move=move,
    )


def position_after_move_from_state(
    *,
    size: int,
    red: set[tuple[int, int]],
    blue: set[tuple[int, int]],
    to_play: str,
    move: str,
) -> str:
    col, row = cell_to_col_row(str(move).strip().lower())
    point = (col, row)
    if point in red or point in blue:
        raise ValueError(f"child move already occupied: {move!r}")
    red_next = set(red)
    blue_next = set(blue)
    if to_play == "red":
        red_next.add(point)
        next_to_play = "blue"
    else:
        blue_next.add(point)
        next_to_play = "red"
    return serialize_position(
        board_size=size,
        red_cells=tuple(sorted(red_next)),
        blue_cells=tuple(sorted(blue_next)),
        to_play=next_to_play,
    )


def point_to_cell(col: int, row: int) -> str:
    return coord_to_human(col, row).lower()


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


def select_available_pass_proxy_cell(*, board_size: int, to_play: str, occupied: set[Point]) -> Point:
    m = center_index(board_size)
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


def serialize_position_stream(
    *,
    red_cells: tuple[Point, ...],
    blue_cells: tuple[Point, ...],
    to_play: str,
) -> str:
    red = sorted(red_cells)
    blue = sorted(blue_cells)
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


def serialize_position(
    *,
    board_size: int,
    red_cells: tuple[Point, ...],
    blue_cells: tuple[Point, ...],
    to_play: str,
) -> str:
    past_stream = serialize_position_stream(
        red_cells=red_cells,
        blue_cells=blue_cells,
        to_play=to_play,
    )
    return f"https://hexworld.org/board/#{board_size}c1,{past_stream}"
