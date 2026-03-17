from __future__ import annotations

Point = tuple[int, int]


def hex_distance_squared(a: Point, b: Point) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return int(dq * dq + dq * dr + dr * dr)


# Row-major axial matrices: (q, r) -> (a*q + b*r, c*q + d*r).
ROTATIONS = (
    (1, 0, 0, 1),
    (0, -1, 1, 1),
    (-1, -1, 1, 0),
    (-1, 0, 0, -1),
    (0, 1, -1, -1),
    (1, 1, -1, 0),
)
D6 = ROTATIONS + tuple((a, b, -a - c, -b - d) for a, b, c, d in ROTATIONS)


def inverse_transform_id(transform_id: int) -> int:
    if 0 <= transform_id < 6:
        return (6 - transform_id) % 6
    if 6 <= transform_id < 12:
        return transform_id
    raise ValueError(f"Bad transform id: {transform_id}")


def apply_transform_ax(p: Point, transform_id: int) -> Point:
    if not (0 <= transform_id < len(D6)):
        raise ValueError(f"Bad transform id: {transform_id}")
    a, b, c, d = D6[transform_id]
    q, r = p
    return a * q + b * r, c * q + d * r
