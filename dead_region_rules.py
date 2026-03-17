from __future__ import annotations


Point = tuple[int, int]

_CANONICAL_RED_ANCHORS = frozenset({(4, 2)})
_CANONICAL_BLUE_ANCHORS = frozenset({(2, 3)})
_CANONICAL_DEAD_REGION = frozenset({
    (1, 1), (2, 1), (3, 1), (4, 1),
    (1, 2), (2, 2), (3, 2),
})
_CANONICAL_EQUIV_RED_ANCHORS = frozenset({(4, 2), (4, 3)})
_CANONICAL_EQUIV_BLUE_ANCHORS = frozenset({(3, 3)})
_CANONICAL_EQUIV_REQUIRED_EMPTY = frozenset({
    (1, 1), (2, 1), (3, 1), (4, 1),
    (1, 2), (2, 2), (3, 2),
    (1, 3), (2, 3),
})
_CANONICAL_EQUIV_LOSER = (2, 3)
_CANONICAL_EQUIV_WINNER = (3, 2)


def _transform_local_point(point: Point, *, swap_axes: bool) -> Point:
    x, y = int(point[0]), int(point[1])
    return (y, x) if swap_axes else (x, y)


def _place_acute_local_point(*, point: Point, board_size: int, top_right: bool) -> Point:
    x, y = int(point[0]), int(point[1])
    size = int(board_size)
    if top_right:
        return size + 1 - y, size + 1 - x
    return y, x


def _acute_rule_applies(
    *,
    red_anchors: frozenset[Point],
    blue_anchors: frozenset[Point],
    required_empty: frozenset[Point],
    red: set[Point],
    blue: set[Point],
) -> bool:
    occupied = set(red) | set(blue)
    return red_anchors.issubset(red) and blue_anchors.issubset(blue) and required_empty.isdisjoint(occupied)


def _iter_acute_variants(
    *,
    red_anchors: frozenset[Point],
    blue_anchors: frozenset[Point],
    cells: frozenset[Point],
    board_size: int,
):
    for swap_axes in (False, True):
        transformed_red = frozenset(_transform_local_point(point, swap_axes=swap_axes) for point in red_anchors)
        transformed_blue = frozenset(_transform_local_point(point, swap_axes=swap_axes) for point in blue_anchors)
        transformed_cells = frozenset(_transform_local_point(point, swap_axes=swap_axes) for point in cells)
        if swap_axes:
            transformed_red, transformed_blue = transformed_blue, transformed_red
        for top_right in (False, True):
            yield (
                frozenset(
                    _place_acute_local_point(point=point, board_size=board_size, top_right=top_right)
                    for point in transformed_red
                ),
                frozenset(
                    _place_acute_local_point(point=point, board_size=board_size, top_right=top_right)
                    for point in transformed_blue
                ),
                frozenset(
                    _place_acute_local_point(point=point, board_size=board_size, top_right=top_right)
                    for point in transformed_cells
                ),
                swap_axes,
                top_right,
            )


def should_exclude_acute_dead_region_move(
    *,
    move: Point,
    red: set[Point],
    blue: set[Point],
    board_size: int,
) -> bool:
    move_point = (int(move[0]), int(move[1]))
    for rule_red, rule_blue, rule_dead, _swap_axes, _top_right in _iter_acute_variants(
        red_anchors=_CANONICAL_RED_ANCHORS,
        blue_anchors=_CANONICAL_BLUE_ANCHORS,
        cells=_CANONICAL_DEAD_REGION,
        board_size=board_size,
    ):
        if move_point not in rule_dead:
            continue
        if _acute_rule_applies(
            red_anchors=rule_red,
            blue_anchors=rule_blue,
            required_empty=rule_dead,
            red=red,
            blue=blue,
        ):
            return True
    return False


def canonicalize_acute_equivalent_move(
    *,
    move: Point,
    red: set[Point],
    blue: set[Point],
    board_size: int,
) -> Point:
    move_point = (int(move[0]), int(move[1]))
    for rule_red, rule_blue, rule_empty, swap_axes, top_right in _iter_acute_variants(
        red_anchors=_CANONICAL_EQUIV_RED_ANCHORS,
        blue_anchors=_CANONICAL_EQUIV_BLUE_ANCHORS,
        cells=_CANONICAL_EQUIV_REQUIRED_EMPTY,
        board_size=board_size,
    ):
        transformed_loser = _transform_local_point(_CANONICAL_EQUIV_LOSER, swap_axes=swap_axes)
        transformed_winner = _transform_local_point(_CANONICAL_EQUIV_WINNER, swap_axes=swap_axes)
        loser = _place_acute_local_point(point=transformed_loser, board_size=board_size, top_right=top_right)
        if move_point != loser:
            continue
        if _acute_rule_applies(
            red_anchors=rule_red,
            blue_anchors=rule_blue,
            required_empty=rule_empty,
            red=red,
            blue=blue,
        ):
            return _place_acute_local_point(
                point=transformed_winner,
                board_size=board_size,
                top_right=top_right,
            )
    return move_point
