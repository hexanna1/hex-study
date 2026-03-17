from __future__ import annotations

from typing import Callable


Point = tuple[int, int]
Cube = tuple[int, int, int]


def ax_to_cube(p: Point) -> Cube:
    q, r = p
    return (q, -q - r, r)


def cube_to_ax(c: Cube) -> Point:
    x, _y, z = c
    return (x, z)


def _refl(c: Cube) -> Cube:
    x, y, z = c
    return (x, z, y)


def _rot_k(k: int) -> Callable[[Cube], Cube]:
    def f(v: Cube) -> Cube:
        x, y, z = v
        for _ in range(k):
            x, y, z = (-z, -x, -y)
        return (x, y, z)

    return f


def _compose(f: Callable[[Cube], Cube], g: Callable[[Cube], Cube]) -> Callable[[Cube], Cube]:
    return lambda v: f(g(v))


ROTATIONS: tuple[Callable[[Cube], Cube], ...] = tuple(_rot_k(k) for k in range(6))
D6: tuple[Callable[[Cube], Cube], ...] = ROTATIONS + tuple(_compose(_refl, r) for r in ROTATIONS)


def inverse_transform_id(transform_id: int) -> int:
    if 0 <= transform_id < 6:
        return (6 - transform_id) % 6
    if 6 <= transform_id < 12:
        return transform_id
    raise ValueError(f"Bad transform id: {transform_id}")


def apply_transform_ax(p: Point, transform_id: int) -> Point:
    if not (0 <= transform_id < len(D6)):
        raise ValueError(f"Bad transform id: {transform_id}")
    t = D6[transform_id]
    return cube_to_ax(t(ax_to_cube(p)))
