"""Cell classifications determined by local Hex pattern configurations."""

from collections.abc import Iterable

from hex_symmetry import apply_transform_ax

Point = tuple[int, int]
Shape = tuple[tuple[Point, ...], tuple[Point, ...], tuple[Point, ...]]
RuleIndex = dict[Point, tuple[Shape, ...]]


def _rules(shapes: tuple[Shape, ...], *, swap_colors: bool) -> RuleIndex:
    # Index each placement by the displacement between one stone of each color.
    index: dict[Point, set[Shape]] = {}
    for a, b, empty in shapes:
        colors = ((a, b), (b, a)) if swap_colors else ((a, b),)
        for own, other in colors:
            for transform in range(12):
                own_t, other_t, empty_t = (
                    tuple(sorted(apply_transform_ax(p, transform) for p in group))
                    for group in (own, other, empty)
                )
                own_q, own_r = own_t[0]
                other_q, other_r = other_t[0]
                offset = other_q - own_q, other_r - own_r
                index.setdefault(offset, set()).add((
                    tuple((q - own_q, r - own_r) for q, r in own_t[1:]),
                    tuple((q - own_q, r - own_r) for q, r in other_t[1:]),
                    tuple((q - own_q, r - own_r) for q, r in empty_t),
                ))
    return {pair: tuple(sorted(placements)) for pair, placements in index.items()}


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


def _matching_cells(
    plus: Iterable[Point], minus: Iterable[Point], rules: RuleIndex
) -> set[Point]:
    own, other = set(plus), set(minus)
    occupied = own | other
    found = set()
    for x, y in own:
        for a, b in other:
            for required_own, required_other, required_empty in rules.get((a - x, b - y), ()):
                if (
                    all((x + dq, y + dr) in own for dq, dr in required_own)
                    and all((x + dq, y + dr) in other for dq, dr in required_other)
                ):
                    empty = {(x + dq, y + dr) for dq, dr in required_empty}
                    if not (empty & occupied):
                        found.update(empty)
    return found


def dead_cells(plus: Iterable[Point], minus: Iterable[Point]) -> set[Point]:
    """Return empty cells proven irrelevant for both colors on the local grid."""
    return _matching_cells(plus, minus, _DEAD_RULES)


def captured_cells(plus: Iterable[Point], minus: Iterable[Point]) -> set[Point]:
    """Return cells in proven captured pairs belonging to the plus player."""
    return _matching_cells(plus, minus, _CAPTURED_RULES)
