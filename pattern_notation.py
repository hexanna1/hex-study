from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Tuple, Union

from hex_symmetry import D6, apply_transform_ax

Point = Tuple[int, int]


@dataclass(frozen=True)
class LabeledPattern:
    plus: Tuple[Point, ...]
    minus: Tuple[Point, ...]


@dataclass(frozen=True)
class UnlabeledPattern:
    a: Tuple[Point, ...]
    b: Tuple[Point, ...]


Pattern = Union[LabeledPattern, UnlabeledPattern]

_INT_RE = re.compile(r"-?[0-9]+$")
_WS_RE = re.compile(r"\s+")


def _parse_int(s: str) -> int:
    if not _INT_RE.fullmatch(s):
        raise ValueError(f"bad integer '{s}'")
    return int(s)


def _parse_points(body: str) -> Tuple[Point, ...]:
    if body == "":
        return ()
    pts: list[Point] = []
    for tok in body.split(":"):
        if tok == "":
            raise ValueError("empty point")
        if "," not in tok:
            raise ValueError(f"bad point '{tok}'")
        q_s, r_s = tok.split(",", 1)
        pts.append((_parse_int(q_s), _parse_int(r_s)))
    if len(set(pts)) != len(pts):
        raise ValueError("duplicate point in block")
    return tuple(pts)


def _read_block(s: str, i: int) -> Tuple[Tuple[Point, ...], int]:
    if i >= len(s) or s[i] != "[":
        raise ValueError(f"expected '[' at pos {i}")
    j = s.find("]", i + 1)
    if j == -1:
        raise ValueError("missing ']'")
    return _parse_points(s[i + 1 : j]), j + 1


def _assert_disjoint(a: Tuple[Point, ...], b: Tuple[Point, ...]) -> None:
    if set(a) & set(b):
        raise ValueError("cross-block overlap is not allowed")


def parse_pattern(raw: str) -> Pattern:
    s = _WS_RE.sub("", raw)
    if not s:
        raise ValueError("empty pattern")

    if s[0] == "+":
        plus, i = _read_block(s, 1)
        if i >= len(s) or s[i] != "-":
            raise ValueError("expected '-' after + block")
        minus, j = _read_block(s, i + 1)
        if j != len(s):
            raise ValueError("trailing characters")
        _assert_disjoint(plus, minus)
        return LabeledPattern(plus=plus, minus=minus)

    if s[0] == "[":
        a, i = _read_block(s, 0)
        b, j = _read_block(s, i)
        if j != len(s):
            raise ValueError("trailing characters")
        _assert_disjoint(a, b)
        return UnlabeledPattern(a=a, b=b)

    raise ValueError("pattern must start with '+' or '['")


def canonicalize(parsed: Pattern, *, collapse_d6: bool = True) -> Pattern:
    def apply_t(pts: Tuple[Point, ...], transform_id: int) -> list[Point]:
        return [apply_transform_ax(p, transform_id) for p in pts]

    def normalize(a: list[Point], b: list[Point]) -> Tuple[Tuple[Point, ...], Tuple[Point, ...]]:
        all_pts = a + b
        if not all_pts:
            return (), ()
        aq, ar = min(all_pts)
        return (
            tuple(sorted((q - aq, r - ar) for q, r in a)),
            tuple(sorted((q - aq, r - ar) for q, r in b)),
        )

    transform_ids = range(len(D6)) if collapse_d6 else [0]

    if isinstance(parsed, LabeledPattern):
        best: LabeledPattern | None = None
        best_key = None
        for transform_id in transform_ids:
            plus_n, minus_n = normalize(
                apply_t(parsed.plus, transform_id),
                apply_t(parsed.minus, transform_id),
            )
            key = (plus_n, minus_n)
            if best_key is None or key < best_key:
                best_key = key
                best = LabeledPattern(plus=plus_n, minus=minus_n)
        if best is None:
            raise AssertionError("internal error: no transform candidates")
        return best

    best_u: UnlabeledPattern | None = None
    best_key = None
    for transform_id in transform_ids:
        a_n, b_n = normalize(apply_t(parsed.a, transform_id), apply_t(parsed.b, transform_id))
        if a_n > b_n:
            a_n, b_n = b_n, a_n
        key = (a_n, b_n)
        if best_key is None or key < best_key:
            best_key = key
            best_u = UnlabeledPattern(a=a_n, b=b_n)
    if best_u is None:
        raise AssertionError("internal error: no transform candidates")
    return best_u


def _fmt_points(pts: Tuple[Point, ...]) -> str:
    return ":".join(f"{q},{r}" for q, r in pts)


def format_pattern(pattern: Pattern) -> str:
    if isinstance(pattern, LabeledPattern):
        return f"+[{_fmt_points(pattern.plus)}]-[{_fmt_points(pattern.minus)}]"
    return f"[{_fmt_points(pattern.a)}][{_fmt_points(pattern.b)}]"
