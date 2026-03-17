"""Cell classifications determined by local Hex pattern configurations."""

from collections.abc import Iterable

from hex_symmetry import apply_transform_ax

Point = tuple[int, int]
Shape = tuple[tuple[Point, ...], tuple[Point, ...], tuple[Point, ...]]


def _rules(shapes: tuple[Shape, ...], *, swap_colors: bool) -> tuple[Shape, ...]:
    rules = set()
    for a, b, empty in shapes:
        colors = ((a, b), (b, a)) if swap_colors else ((a, b),)
        for own, other in colors:
            for transform in range(12):
                rules.add(tuple(
                    tuple(sorted(apply_transform_ax(p, transform) for p in group))
                    for group in (own, other, empty)
                ))
    return tuple(sorted(rules))


_DEAD_RULES = _rules((
    (((-1, 0),), ((0, 1), (1, -1), (1, 0)), ((0, 0),)),
    (((-1, 0), (-1, 1)), ((1, -1), (1, 0)), ((0, 0),)),
    (((-1, 0), (-1, 1), (0, 1)), ((1, -2), (2, -2), (2, -1)), ((0, 0),)),
), swap_colors=True)
_CAPTURED_RULES = _rules((
    (((1, -1), (0, 1)), ((-1, 0), (2, 0)), ((0, 0), (1, 0))),
    (((0, -1), (1, 0), (0, 1)), ((2, -2),), ((0, 0), (1, -1))),
    (((-2, 1), (-1, 1), (0, 0)), ((-3, 0), (-2, -1)), ((-2, 0), (-1, 0))),
    (((-2, 0), (-1, 0), (0, 0)), ((-1, -2), (1, -2)), ((-1, -1), (0, -1))),
), swap_colors=False)


def _matching_cells(plus: Iterable[Point], minus: Iterable[Point], rules: tuple[Shape, ...]) -> set[Point]:
    own, other = set(plus), set(minus)
    occupied = own | other
    found = set()
    for required_own, required_other, required_empty in rules:
        aq, ar = required_own[0]
        for q, r in own:
            x, y = q - aq, r - ar
            empty = {(x + dq, y + dr) for dq, dr in required_empty}
            if empty & occupied:
                continue
            if (
                all((x + dq, y + dr) in own for dq, dr in required_own)
                and all((x + dq, y + dr) in other for dq, dr in required_other)
            ):
                found.update(empty)
    return found


def dead_cells(plus: Iterable[Point], minus: Iterable[Point]) -> set[Point]:
    """Return empty cells proven irrelevant for both colors on the local grid."""
    return _matching_cells(plus, minus, _DEAD_RULES)


def captured_cells(plus: Iterable[Point], minus: Iterable[Point]) -> set[Point]:
    """Return cells in proven captured pairs belonging to the plus player."""
    return _matching_cells(plus, minus, _CAPTURED_RULES)
