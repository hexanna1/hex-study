from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from website_bundle_utils import encode_optional_thousandths

OPENING_ROOT_KEYS = {
    "board_size": "s",
    "root": "r",
    "root_openings": "o",
    "root_study": "u",
    "completed": "c",
    "completed_ply": "p",
    "nodes": "n",
}
OPENING_NODE_KEYS = {
    "parent": "p",
    "move": "m",
    "ply": "y",
    "importance": "i",
    "candidates": "c",
    "nonretained_candidates": "x",
    "tree_red_winrate": "t",
}
OPENING_CANDIDATE_KEYS = {
    "move": "m",
    "rank": "r",
    "prior": "p",
    "stone_fraction": "s",
    "candidate_weight": "w",
    "importance": "i",
    "child": "c",
    "tree_mover_winrate": "t",
}
OPENING_ROOT_STUDY_KEYS = {
    "reference_move": "m",
    "reference_red_winrate": "w",
    "reference_elo": "e",
    "reference_stone_fraction": "s",
    "fair_band": "b",
    "rows": "r",
    "root_openings": "o",
    "move": "m",
    "red_winrate": "w",
    "elo": "e",
    "stone_fraction": "s",
    "fair": "f",
}
JOSEKI_ROOT_KEYS = {
    "family": "f",
    "board_size": "s",
    "balance_moves": "b",
    "completed": "c",
    "completed_depth": "d",
    "nodes": "n",
}
JOSEKI_NODE_KEYS = {
    "line": "l",
    "candidates": "c",
    "canonicalized_position": "p",
    "importance": "i",
}
JOSEKI_CANDIDATE_KEYS = {
    "kind": "k",
    "stone_fraction": "s",
    "local": "l",
    "child": "c",
}
PATTERN_TILE_KEYS = {
    "pattern": "p",
    "cells": "c",
}
PATTERN_CELL_KEYS = {
    "stone_fraction": "s",
    "local_rel": "l",
}

_TREE_KEY_GROUPS = (
    OPENING_ROOT_KEYS,
    OPENING_NODE_KEYS,
    OPENING_CANDIDATE_KEYS,
    OPENING_ROOT_STUDY_KEYS,
    JOSEKI_ROOT_KEYS,
    JOSEKI_NODE_KEYS,
    JOSEKI_CANDIDATE_KEYS,
)
_TREE_SHORT_KEY: dict[str, str] = {}
for group in _TREE_KEY_GROUPS:
    for name, short in group.items():
        existing = _TREE_SHORT_KEY.setdefault(name, short)
        if existing != short:
            raise ValueError(f"conflicting artifact key for {name!r}")


_THOUSANDTH_FIELDS = {
    "importance",
    "prior",
    "stone_fraction",
    "candidate_weight",
    "tree_mover_winrate",
    "tree_red_winrate",
    "reference_red_winrate",
    "reference_elo",
    "reference_stone_fraction",
    "red_winrate",
    "elo",
}


def _encode_tree(value: Any) -> Any:
    if isinstance(value, list):
        return [_encode_tree(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        short = _TREE_SHORT_KEY.get(key) if isinstance(key, str) else None
        if short is None:
            raise ValueError(f"unsupported artifact field: {key!r}")
        if key in _THOUSANDTH_FIELDS:
            out[short] = encode_optional_thousandths(item)
        elif key == "fair_band":
            out[short] = [encode_optional_thousandths(entry) for entry in item]
        else:
            out[short] = _encode_tree(item)
    return out


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def load(path: Path) -> Any:
    return json.loads(path.read_bytes(), parse_constant=_reject_json_constant)


def load_pattern_tile(path: Path) -> dict[str, Any]:
    payload = load(path)
    tile_keys = PATTERN_TILE_KEYS
    if not isinstance(payload, dict):
        raise ValueError("pattern tile must be an object")
    pattern = payload.get(tile_keys["pattern"])
    cells = payload.get(tile_keys["cells"])
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern tile requires a pattern")
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError("pattern tile requires object cells")
    return payload


def optional_thousandths(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"artifact value is not integer thousandths: {value!r}")
    return value


def thousandths(value: Any) -> int:
    decoded = optional_thousandths(value)
    if decoded is None:
        raise ValueError("artifact value is null")
    return decoded


def _dump_encoded(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def dump_tree(path: Path, payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("tree artifact must be an object")
    _dump_encoded(path, _encode_tree(payload))


def dump_pattern_tile(path: Path, spec: dict[str, Any]) -> None:
    cells: list[dict[str, Any]] = []
    pattern = spec["pattern"]
    source_cells = spec["cells"]
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern tile requires a pattern")
    if not isinstance(source_cells, list) or not all(
        isinstance(cell, dict) for cell in source_cells
    ):
        raise ValueError("pattern tile requires object cells")

    def sort_key(cell: dict[str, Any]) -> tuple[Any, ...]:
        local = cell.get("local_rel")
        q, r = local if isinstance(local, list) and len(local) == 2 else (10**9, 10**9)
        return (
            int(cell["rank"]),
            str(cell.get("kind") or ""),
            int(q),
            int(r),
            float(cell["stone_fraction"]),
        )

    for cell in sorted(source_cells, key=sort_key):
        kind = cell["kind"]
        if kind not in {"local", "tenuki"}:
            raise ValueError(f"invalid pattern cell kind: {kind!r}")
        stone_fraction = float(cell["stone_fraction"])
        row = {PATTERN_CELL_KEYS["stone_fraction"]: stone_fraction}
        if kind == "local":
            local = cell["local_rel"]
            if not isinstance(local, list) or len(local) != 2:
                raise ValueError(f"local pattern cell missing coordinates: {cell!r}")
            row[PATTERN_CELL_KEYS["local_rel"]] = [int(local[0]), int(local[1])]
        elif cell.get("local_rel") is not None:
            raise ValueError(f"tenuki pattern cell has local coordinates: {cell!r}")
        cells.append(row)
    payload = {
        PATTERN_TILE_KEYS["pattern"]: pattern,
        PATTERN_TILE_KEYS["cells"]: cells,
    }
    _dump_encoded(path, payload)
