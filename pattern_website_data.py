from __future__ import annotations

import argparse
import hashlib
import json
import struct
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import pattern_output_utils as sout
from pattern_notation import LabeledPattern, canonicalize, format_pattern, parse_pattern

BUNDLE_MAGIC = b"HPB1"
BUNDLE_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHIII")
PAIR_COUNT_STRUCT = struct.Struct("<H")
PAIR_ROW_STRUCT = struct.Struct("<bb")
PACKED_STONE_FRACTION_BITS = 10
PACKED_OPTIONAL_NULL = 1023
FRACTION_MODE_COORD_MAJOR_BITPLANE = 2
PRESENCE_MODE_ALL = 0
PRESENCE_MODE_BITMAP = 1
PRESENCE_MODE_PRESENT_GAPS = 2
PRESENCE_MODE_ABSENT_GAPS = 3
PACKED_KEY_COORD_MIN = -8
PACKED_KEY_COORD_MAX = 7
THOUSANDTH = Decimal("0.001")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "interior_patterns_m5_d4_span13"


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / "patterns_current.json"


def _catalog_selected_tile_paths(artifacts_root: Path) -> list[Path]:
    root = Path(artifacts_root)
    if root.name == "tiles":
        raise ValueError("artifacts_root must be the artifact base dir, not its tiles/ subdir")
    catalog_path = root / "catalog.json"
    if not catalog_path.exists():
        raise ValueError(f"Missing required catalog.json: {catalog_path}")
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Catalog must be a JSON object: {catalog_path}")
    patterns = raw.get("patterns")
    if not isinstance(patterns, list):
        raise ValueError(f"Catalog missing patterns list: {catalog_path}")
    tiles_dir = root / "tiles"
    selected: list[Path] = []
    for row in patterns:
        if not isinstance(row, dict):
            continue
        hexworld = str(row.get("hexworld_21") or "").strip()
        candidate_Δ_max = row.get("candidate_Δ_max")
        if not hexworld:
            raise ValueError(f"Catalog row missing hexworld_21: {catalog_path}")
        if not isinstance(candidate_Δ_max, int):
            raise ValueError(f"Catalog row missing integer candidate_Δ_max: {catalog_path}")
        slug = sout._movelist_slug_from_hexworld(hexworld)
        path = tiles_dir / f"d{int(candidate_Δ_max):02d}-{slug}.json"
        if path.exists():
            selected.append(path)
    return selected


def _canonical_labeled_pattern(pattern_text: str) -> str:
    parsed = parse_pattern(pattern_text)
    if not isinstance(parsed, LabeledPattern):
        raise ValueError("Pattern website data requires labeled pattern notation")
    return format_pattern(canonicalize(parsed))


def _normalize_local_rel(raw: Any) -> list[int]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"Bad local_rel value: {raw!r}")
    q, r = raw
    if not isinstance(q, int) or not isinstance(r, int):
        raise ValueError(f"Bad local_rel coordinates: {raw!r}")
    return [int(q), int(r)]


def _normalize_cell(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Bad cell payload: {raw!r}")
    kind = str(raw.get("kind") or "").strip()
    key = str(raw.get("key") or "").strip()
    stone_fraction = raw.get("stone_fraction")
    rank = raw.get("rank")
    if not kind or not key:
        raise ValueError(f"Cell missing kind/key: {raw!r}")
    if not isinstance(stone_fraction, (int, float)):
        raise ValueError(f"Cell missing stone_fraction: {raw!r}")
    if not isinstance(rank, int):
        raise ValueError(f"Cell missing rank: {raw!r}")
    cell = {"kind": kind, "stone_fraction": float(stone_fraction), "rank": int(rank)}
    if kind == "local":
        cell["local_rel"] = _normalize_local_rel(raw.get("local_rel"))
    return cell


def _encode_thousandths(value: Any) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Bad numeric payload: {value!r}")
    quantized = Decimal(str(value)).quantize(THOUSANDTH, rounding=ROUND_HALF_EVEN)
    return int(quantized * 1000)


def _cell_sort_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    local_rel = cell.get("local_rel")
    q = int(local_rel[0]) if isinstance(local_rel, list) and len(local_rel) == 2 else 10**9
    r = int(local_rel[1]) if isinstance(local_rel, list) and len(local_rel) == 2 else 10**9
    return (
        int(cell["rank"]),
        str(cell["kind"]),
        q,
        r,
        float(cell["stone_fraction"]),
    )


def _normalize_spec(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Tile spec must be a JSON object")
    pattern = str(raw.get("pattern") or "").strip()
    to_play = str(raw.get("to_play") or "").strip().lower()
    cells = raw.get("cells")
    if not pattern:
        raise ValueError("Tile spec missing pattern")
    if to_play not in {"red", "blue"}:
        raise ValueError(f"Tile spec has unsupported to_play: {to_play!r}")
    if not isinstance(cells, list):
        raise ValueError("Tile spec missing cells list")
    canonical_pattern = _canonical_labeled_pattern(pattern)
    if canonical_pattern != pattern:
        raise ValueError(f"Tile spec pattern is not canonical labeled notation: {pattern!r}")
    return {
        "pattern": canonical_pattern,
        "to_play": to_play,
        "cells": sorted((_normalize_cell(cell) for cell in cells), key=_cell_sort_key),
    }


def _website_entry_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    local_cells: list[list[int]] = []
    tenuki_stone_fraction: float | None = None
    for cell in list(spec.get("cells") or []):
        kind = str(cell.get("kind") or "")
        if kind == "local":
            local_rel = cell.get("local_rel")
            if not isinstance(local_rel, list) or len(local_rel) != 2:
                raise ValueError(f"Local cell missing local_rel: {cell!r}")
            local_cells.append(
                [int(local_rel[0]), int(local_rel[1]), _encode_thousandths(cell["stone_fraction"])]
            )
        elif kind == "tenuki":
            tenuki_stone_fraction = _encode_thousandths(cell["stone_fraction"])
    out: dict[str, Any] = {
        "p": str(spec["to_play"]),
        "c": local_cells,
    }
    if tenuki_stone_fraction is not None:
        out["t"] = tenuki_stone_fraction
    return out


def build_pattern_index(*, artifacts_root: Path, repo_root: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    tile_paths = _catalog_selected_tile_paths(artifacts_root)
    for path in tile_paths:
        spec = _normalize_spec(json.loads(path.read_text(encoding="utf-8")))
        pattern = str(spec["pattern"])
        website_entry = _website_entry_from_spec(spec)
        existing = entries.get(pattern)
        if existing is None:
            entries[pattern] = website_entry
            continue
        if existing != website_entry:
            rel_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
            raise ValueError(f"Conflicting tile specs for pattern {pattern!r}: {rel_path}")
    ordered_patterns = {pattern: entries[pattern] for pattern in sorted(entries)}
    return {
        "version": 1,
        "pattern_count": int(len(ordered_patterns)),
        "patterns": ordered_patterns,
    }


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def _bundle_filename(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"pattern_index.{digest}.bin"


def _pack_bits(rows: list[tuple[int, int]]) -> bytes:
    bit_len = sum(int(bits) for _word, bits in rows)
    out = bytearray((bit_len + 7) // 8)
    bit_offset = 0
    for word, bits in rows:
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        chunk = int(word) << shift
        for byte_idx in range(4):
            out_idx = byte_offset + byte_idx
            if out_idx >= len(out):
                break
            out[out_idx] |= (chunk >> (8 * byte_idx)) & 0xFF
        bit_offset += int(bits)
    return bytes(out)


def _encode_uvarint(value: int) -> bytes:
    out = bytearray()
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"Bad varint payload: {value!r}")
    while value_i >= 0x80:
        out.append((value_i & 0x7F) | 0x80)
        value_i >>= 7
    out.append(value_i)
    return bytes(out)


def _pack_bitplanes(values: list[int], bits: int) -> bytes:
    rows: list[tuple[int, int]] = []
    for bit in range(int(bits) - 1, -1, -1):
        rows.extend(((int(value) >> bit) & 1, 1) for value in values)
    return _pack_bits(rows)


def _encode_presence_row(present_indices: list[int], pattern_count: int) -> bytes:
    present_set = set(present_indices)
    if len(present_indices) == pattern_count:
        return bytes([PRESENCE_MODE_ALL])

    bitmap = bytes([PRESENCE_MODE_BITMAP]) + _pack_bits(
        [(1 if idx in present_set else 0, 1) for idx in range(pattern_count)]
    )

    present_gaps = bytearray([PRESENCE_MODE_PRESENT_GAPS])
    present_gaps.extend(_encode_uvarint(len(present_indices)))
    prev = -1
    for idx in present_indices:
        present_gaps.extend(_encode_uvarint(int(idx) - prev - 1))
        prev = int(idx)

    absent_indices = [idx for idx in range(pattern_count) if idx not in present_set]
    absent_gaps = bytearray([PRESENCE_MODE_ABSENT_GAPS])
    absent_gaps.extend(_encode_uvarint(len(absent_indices)))
    prev = -1
    for idx in absent_indices:
        absent_gaps.extend(_encode_uvarint(int(idx) - prev - 1))
        prev = int(idx)

    return min((bitmap, bytes(present_gaps), bytes(absent_gaps)), key=len)


def _pack_key_coord(q: int, r: int) -> int:
    if (
        q < PACKED_KEY_COORD_MIN
        or q > PACKED_KEY_COORD_MAX
        or r < PACKED_KEY_COORD_MIN
        or r > PACKED_KEY_COORD_MAX
    ):
        raise ValueError(f"Pattern key coordinate out of packed range: {q!r},{r!r}")
    return (int(q) - PACKED_KEY_COORD_MIN) | ((int(r) - PACKED_KEY_COORD_MIN) << 4)


def _pack_pattern_key(pattern: str) -> bytes:
    parsed = parse_pattern(pattern)
    if not isinstance(parsed, LabeledPattern):
        raise ValueError("Pattern website data requires labeled pattern notation")
    plus_count = len(parsed.plus)
    minus_count = len(parsed.minus)
    if plus_count > 15 or minus_count > 15:
        raise ValueError(f"Pattern key has too many stones for binary bundle: {pattern!r}")
    out = bytearray([plus_count | (minus_count << 4)])
    for point in [*parsed.plus, *parsed.minus]:
        out.append(_pack_key_coord(int(point[0]), int(point[1])))
    return bytes(out)


def _to_play_from_packed_pattern_key(pattern_b: bytes) -> str:
    if not pattern_b:
        raise ValueError("Empty packed pattern key")
    plus_count = pattern_b[0] & 0x0F
    minus_count = (pattern_b[0] >> 4) & 0x0F
    return "red" if minus_count - plus_count <= 0 else "blue"


def _build_pattern_bundle_from_index(index: dict[str, Any]) -> bytes:
    patterns = index["patterns"]
    out = bytearray()
    key_stream = bytearray()
    tenuki_rows: list[int] = []
    entry_cell_maps: list[dict[tuple[int, int], int]] = []
    prev_key = b""
    for pattern, entry in patterns.items():
        pattern_b = _pack_pattern_key(str(pattern))
        if str(entry.get("p") or "") != _to_play_from_packed_pattern_key(pattern_b):
            raise ValueError(f"Pattern side-to-play is not derivable from key: {pattern!r}")
        prefix_len = 0
        for left, right in zip(prev_key, pattern_b):
            if left != right:
                break
            prefix_len += 1
        suffix = pattern_b[prefix_len:]
        key_stream.extend(_encode_uvarint(prefix_len))
        key_stream.extend(_encode_uvarint(len(suffix)))
        key_stream.extend(suffix)
        prev_key = pattern_b
        tenuki_sf = PACKED_OPTIONAL_NULL
        if "t" in entry:
            tenuki_sf = int(entry["t"])
        if tenuki_sf < 0 or tenuki_sf >= (1 << PACKED_STONE_FRACTION_BITS):
            raise ValueError(f"Pattern tenuki fraction out of u10 range: {tenuki_sf!r}")
        tenuki_rows.append(tenuki_sf)
        cells = list(entry["c"])
        cell_map: dict[tuple[int, int], int] = {}
        for cell in cells:
            if not isinstance(cell, list) or len(cell) != 3:
                raise ValueError(f"Bad compact pattern cell: {cell!r}")
            q = int(cell[0])
            r = int(cell[1])
            stone_fraction = int(cell[2])
            if q < -128 or q > 127 or r < -128 or r > 127:
                raise ValueError(f"Pattern local coordinate out of i8 range: {cell!r}")
            if stone_fraction < 0 or stone_fraction >= (1 << PACKED_STONE_FRACTION_BITS):
                raise ValueError(f"Pattern stone fraction out of u10 range: {cell!r}")
            pair = (q, r)
            if pair in cell_map:
                raise ValueError(f"Duplicate pattern local coordinate in binary bundle: {cell!r}")
            cell_map[pair] = stone_fraction
        entry_cell_maps.append(cell_map)

    pair_rows = sorted({pair for cell_map in entry_cell_maps for pair in cell_map})
    if len(pair_rows) > 0xFFFF:
        raise ValueError(f"Too many local coordinate pairs for binary bundle: {len(pair_rows)!r}")

    presence_stream = bytearray()
    fraction_values: list[int] = []
    for pair in pair_rows:
        present_indices: list[int] = []
        for entry_idx, cell_map in enumerate(entry_cell_maps):
            stone_fraction = cell_map.get(pair)
            if stone_fraction is None:
                continue
            present_indices.append(entry_idx)
            fraction_values.append(stone_fraction)
        presence_stream.extend(_encode_presence_row(present_indices, len(entry_cell_maps)))

    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            FRACTION_MODE_COORD_MAJOR_BITPLANE,
            len(entry_cell_maps),
            len(fraction_values),
            len(key_stream),
        )
    )
    out.extend(_pack_bitplanes(tenuki_rows, PACKED_STONE_FRACTION_BITS))
    out.extend(PAIR_COUNT_STRUCT.pack(len(pair_rows)))
    for q, r in pair_rows:
        out.extend(PAIR_ROW_STRUCT.pack(int(q), int(r)))
    out.extend(struct.pack("<I", len(presence_stream)))
    out.extend(presence_stream)
    out.extend(_pack_bitplanes(fraction_values, PACKED_STONE_FRACTION_BITS))
    out.extend(key_stream)
    return bytes(out)


def build_pattern_bundle(*, artifacts_root: Path, repo_root: Path) -> bytes:
    return _build_pattern_bundle_from_index(
        build_pattern_index(
            artifacts_root=artifacts_root,
            repo_root=repo_root,
        )
    )


def write_pattern_index(*, artifacts_root: Path, out_path: Path) -> Path:
    repo_root = _repo_root()
    payload = build_pattern_index(
        artifacts_root=artifacts_root,
        repo_root=repo_root,
    )
    bundle = _build_pattern_bundle_from_index(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_name = _bundle_filename(bundle)
    bundle_path = out_path.parent / bundle_name
    bundle_path.write_bytes(bundle)
    for path in out_path.parent.glob("pattern_index.*.bin"):
        if path.name != bundle_name:
            path.unlink()
    manifest = {
        "version": BUNDLE_VERSION,
        "bundle": bundle_name,
        "pattern_count": int(payload["pattern_count"]),
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a unified binary bundle for the pattern website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    write_pattern_index(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
