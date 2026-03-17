from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Tuple


Move = Tuple[int, int]
Entry = Move | None

_WS_RE = re.compile(r"\s+")
_INT_RE = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class JosekiBlock:
    family: str  # A | O
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class JosekiLine:
    blocks: tuple[JosekiBlock, ...]


def _parse_positive_int(text: str) -> int:
    if not _INT_RE.fullmatch(text):
        raise ValueError(f"bad integer '{text}'")
    value = int(text)
    if value <= 0:
        raise ValueError(f"coordinate must be > 0, got {text!r}")
    return value


def _parse_entry(text: str) -> Entry:
    if text == "":
        return None
    if "," not in text:
        raise ValueError(f"bad move '{text}'")
    x_s, y_s = text.split(",", 1)
    return (_parse_positive_int(x_s), _parse_positive_int(y_s))


def _parse_block(s: str, i: int) -> tuple[JosekiBlock, int]:
    if i >= len(s) or s[i] not in {"A", "O"}:
        raise ValueError(f"expected family at pos {i}")
    family = s[i]
    if i + 1 >= len(s) or s[i + 1] != "[":
        raise ValueError(f"expected '[' after family at pos {i}")
    j = s.find("]", i + 2)
    if j == -1:
        raise ValueError("missing ']'")
    body = s[i + 2 : j]
    entries = tuple(_parse_entry(part) for part in body.split(":"))
    return JosekiBlock(family=family, entries=entries), j + 1


def parse_joseki_line(raw: str) -> JosekiLine:
    s = _WS_RE.sub("", raw)
    if not s:
        raise ValueError("empty joseki line")
    blocks: list[JosekiBlock] = []
    i = 0
    while i < len(s):
        block, i = _parse_block(s, i)
        blocks.append(block)
    if not blocks:
        raise ValueError("joseki line has no blocks")
    return JosekiLine(blocks=tuple(blocks))


def _format_entry(entry: Entry) -> str:
    if entry is None:
        return ""
    return f"{entry[0]},{entry[1]}"


def format_joseki_line(line: JosekiLine) -> str:
    if not line.blocks:
        raise ValueError("joseki line has no blocks")
    return "".join(
        f"{block.family}[{':'.join(_format_entry(entry) for entry in block.entries)}]"
        for block in line.blocks
    )


def merge_same_family_blocks(line: JosekiLine) -> JosekiLine:
    if not line.blocks:
        raise ValueError("joseki line has no blocks")
    merged: list[JosekiBlock] = []
    for block in line.blocks:
        family = str(block.family)
        if family not in {"A", "O"}:
            raise ValueError(f"unsupported family: {family!r}")
        if merged and merged[-1].family == family:
            merged[-1] = JosekiBlock(family=family, entries=merged[-1].entries + tuple(block.entries))
        else:
            merged.append(JosekiBlock(family=family, entries=tuple(block.entries)))
    return JosekiLine(blocks=tuple(merged))


def format_single_track_line(*, family: str, entries: tuple[Entry, ...]) -> str:
    family_s = str(family).strip().upper()
    if family_s not in {"A", "O"}:
        raise ValueError(f"unsupported family: {family!r}")
    return format_joseki_line(JosekiLine(blocks=(JosekiBlock(family=family_s, entries=tuple(entries)),)))
