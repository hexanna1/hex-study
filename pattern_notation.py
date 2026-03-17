from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import re

from hex_symmetry import D6, apply_transform_ax

Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class LabeledPattern:
    plus: tuple[Point, ...]
    minus: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class UnlabeledPattern:
    a: tuple[Point, ...]
    b: tuple[Point, ...]


Pattern = LabeledPattern | UnlabeledPattern

_INT_RE = re.compile(r"-?[0-9]+$")
_WS_RE = re.compile(r"\s+")


def _parse_int(s: str) -> int:
    if not _INT_RE.fullmatch(s):
        raise ValueError(f"bad integer '{s}'")
    return int(s)


def _parse_points(body: str) -> tuple[Point, ...]:
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


def _parse_block(s: str, i: int) -> tuple[tuple[Point, ...], int]:
    if i >= len(s) or s[i] != "[":
        raise ValueError(f"expected '[' at pos {i}")
    j = s.find("]", i + 1)
    if j == -1:
        raise ValueError("missing ']'")
    return _parse_points(s[i + 1 : j]), j + 1


def _assert_disjoint(a: tuple[Point, ...], b: tuple[Point, ...]) -> None:
    if set(a) & set(b):
        raise ValueError("cross-block overlap is not allowed")


def parse_pattern(raw: str) -> Pattern:
    s = _WS_RE.sub("", raw)
    if not s:
        raise ValueError("empty pattern")

    if s[0] == "+":
        plus, i = _parse_block(s, 1)
        if i >= len(s) or s[i] != "-":
            raise ValueError("expected '-' after + block")
        minus, j = _parse_block(s, i + 1)
        if j != len(s):
            raise ValueError("trailing characters")
        _assert_disjoint(plus, minus)
        return LabeledPattern(plus=plus, minus=minus)

    if s[0] == "[":
        a, i = _parse_block(s, 0)
        b, j = _parse_block(s, i)
        if j != len(s):
            raise ValueError("trailing characters")
        _assert_disjoint(a, b)
        return UnlabeledPattern(a=a, b=b)

    raise ValueError("pattern must start with '+' or '['")


def normalize_points(
    a: Iterable[Point], b: Iterable[Point],
) -> tuple[tuple[Point, ...], tuple[Point, ...], Point]:
    a, b = tuple(a), tuple(b)
    aq, ar = min(a + b, default=(0, 0))
    return (
        tuple(sorted((q - aq, r - ar) for q, r in a)),
        tuple(sorted((q - aq, r - ar) for q, r in b)),
        (aq, ar),
    )


def canonical_frame(
    parsed: Pattern, *, collapse_d6: bool = True,
) -> tuple[Pattern, Point, int]:
    """Return the canonical pattern together with its translation and D6 frame."""
    labeled = isinstance(parsed, LabeledPattern)
    a, b = (parsed.plus, parsed.minus) if labeled else (parsed.a, parsed.b)
    frames = []
    for transform_id in range(len(D6) if collapse_d6 else 1):
        an, bn, anchor = normalize_points(
            (apply_transform_ax(p, transform_id) for p in a),
            (apply_transform_ax(p, transform_id) for p in b),
        )
        if not labeled and an > bn:
            an, bn = bn, an
        frames.append(((an, bn), anchor, transform_id))
    (an, bn), anchor, transform_id = min(frames, key=lambda frame: frame[0])
    pattern = LabeledPattern(an, bn) if labeled else UnlabeledPattern(an, bn)
    return pattern, anchor, transform_id


def canonicalize(parsed: Pattern, *, collapse_d6: bool = True) -> Pattern:
    return canonical_frame(parsed, collapse_d6=collapse_d6)[0]


def _format_points(pts: tuple[Point, ...]) -> str:
    return ":".join(f"{q},{r}" for q, r in pts)


def format_pattern(pattern: Pattern) -> str:
    if isinstance(pattern, LabeledPattern):
        return f"+[{_format_points(pattern.plus)}]-[{_format_points(pattern.minus)}]"
    return f"[{_format_points(pattern.a)}][{_format_points(pattern.b)}]"
