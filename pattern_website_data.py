from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pattern_output_utils import movelist_slug_from_hexworld
from pattern_notation import LabeledPattern, canonicalize, format_pattern, parse_pattern
from website_bundle_utils import (
    BundlePayload,
    PACKED_OPTIONAL_U10_NULL,
    encode_uvarint,
    encode_thousandths,
    pack_little_endian_bits,
    write_hashed_bundle_manifest,
)

BUNDLE_MAGIC = b"HPB1"
BUNDLE_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHIII")
PAIR_COUNT_STRUCT = struct.Struct("<H")
PAIR_ROW_STRUCT = struct.Struct("<bb")
PACKED_STONE_FRACTION_BITS = 10
PACKED_OPTIONAL_NULL = PACKED_OPTIONAL_U10_NULL
FRACTION_MODE_AFFINE_ROW_COORD_BITPLANE = 3
ROW_MEAN_PREDICTOR_STRUCT = struct.Struct("<BBhB")
ROW_MEAN_PREDICTOR_NUMERATOR = 3
ROW_MEAN_PREDICTOR_DENOMINATOR = 4
ROW_MEAN_PREDICTOR_INTERCEPT = 84
PACKED_KEY_COORD_MIN = -8
PACKED_KEY_COORD_MAX = 7
def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / "patterns_current.json"


@dataclass(frozen=True)
class CatalogTile:
    path: Path
    source_span: int


@dataclass(frozen=True)
class WebsitePatternLayer:
    id: str
    min_moves: int
    max_moves: int


WEBSITE_PATTERN_LAYERS = (
    WebsitePatternLayer("base", 1, 5),
    WebsitePatternLayer("m6", 6, 6),
)


def _default_artifacts_roots() -> tuple[Path, ...]:
    root = _repo_root() / "artifacts"
    return (
        root / "interior_patterns_m5_d4_span16",
        root / "interior_patterns_m6_d4_span13",
    )


def _as_artifacts_roots(artifacts_roots: Iterable[Path]) -> tuple[Path, ...]:
    roots = tuple(Path(root) for root in artifacts_roots)
    if not roots:
        raise ValueError("At least one artifacts root is required")
    return roots


def _catalog_selected_tile_paths(artifacts_root: Path) -> list[CatalogTile]:
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
    max_pair_delta = raw.get("max_pair_delta")
    source_span = int(max_pair_delta) if isinstance(max_pair_delta, int) else -1
    tiles_dir = root / "tiles"
    selected: list[CatalogTile] = []
    for row in patterns:
        if not isinstance(row, dict):
            continue
        hexworld = str(row.get("hexworld_21") or "").strip()
        candidate_Δ_max = row.get("candidate_Δ_max")
        if not hexworld:
            raise ValueError(f"Catalog row missing hexworld_21: {catalog_path}")
        if not isinstance(candidate_Δ_max, int):
            raise ValueError(f"Catalog row missing integer candidate_Δ_max: {catalog_path}")
        slug = movelist_slug_from_hexworld(hexworld)
        path = tiles_dir / f"d{int(candidate_Δ_max):02d}-{slug}.json"
        if path.exists():
            selected.append(CatalogTile(path=path, source_span=source_span))
    return selected


def _canonical_labeled_pattern(pattern_text: str) -> str:
    parsed = parse_pattern(pattern_text)
    if not isinstance(parsed, LabeledPattern):
        raise ValueError("Pattern website data requires labeled pattern notation")
    return format_pattern(canonicalize(parsed))


def _move_count_for_labeled_pattern(pattern_text: str) -> int:
    parsed = parse_pattern(pattern_text)
    if not isinstance(parsed, LabeledPattern):
        raise ValueError("Pattern website data requires labeled pattern notation")
    diff = len(parsed.minus) - len(parsed.plus)
    if diff < -1 or diff > 2:
        raise ValueError(
            f"Pattern is not a supported labeled family under red-first play with at most one tenuki: {pattern_text!r}"
        )
    return len(parsed.plus) + len(parsed.minus) + (1 if diff in {-1, 2} else 0)


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
                [
                    int(local_rel[0]),
                    int(local_rel[1]),
                    encode_thousandths(cell["stone_fraction"], clamp=True),
                ]
            )
        elif kind == "tenuki":
            tenuki_stone_fraction = encode_thousandths(cell["stone_fraction"], clamp=True)
    out: dict[str, Any] = {
        "p": str(spec["to_play"]),
        "c": local_cells,
    }
    if tenuki_stone_fraction is not None:
        out["t"] = tenuki_stone_fraction
    return out


@dataclass(frozen=True)
class PatternEntrySource:
    entry: dict[str, Any]
    source_span: int


def build_pattern_index(*, artifacts_roots: Iterable[Path], repo_root: Path) -> dict[str, Any]:
    entries: dict[str, PatternEntrySource] = {}
    tile_paths: list[CatalogTile] = []
    for artifacts_root in _as_artifacts_roots(artifacts_roots):
        tile_paths.extend(_catalog_selected_tile_paths(artifacts_root))
    for tile in tile_paths:
        path = tile.path
        spec = _normalize_spec(json.loads(path.read_text(encoding="utf-8")))
        pattern = str(spec["pattern"])
        website_entry = _website_entry_from_spec(spec)
        existing = entries.get(pattern)
        if existing is None:
            entries[pattern] = PatternEntrySource(
                entry=website_entry,
                source_span=int(tile.source_span),
            )
            continue
        if existing.entry == website_entry:
            continue
        if int(tile.source_span) > int(existing.source_span):
            entries[pattern] = PatternEntrySource(
                entry=website_entry,
                source_span=int(tile.source_span),
            )
            continue
        if int(tile.source_span) < int(existing.source_span):
            continue
        rel_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
        raise ValueError(f"Conflicting tile specs for pattern {pattern!r}: {rel_path}")
    ordered_patterns = {pattern: entries[pattern].entry for pattern in sorted(entries)}
    return {
        "version": 1,
        "pattern_count": int(len(ordered_patterns)),
        "patterns": ordered_patterns,
    }


def _pack_bitplanes(values: list[int], bits: int) -> bytes:
    rows: list[tuple[int, int]] = []
    for bit in range(int(bits) - 1, -1, -1):
        rows.extend(((int(value) >> bit) & 1, 1) for value in values)
    return pack_little_endian_bits(rows, chunk_bytes=4)


def _zigzag(value: int) -> int:
    return (int(value) << 1) if value >= 0 else ((-int(value) << 1) - 1)


def _pack_signed_bitplanes(values: list[int]) -> tuple[bytes, int]:
    encoded = [_zigzag(value) for value in values]
    bits = max(1, max(encoded, default=0).bit_length())
    return _pack_bitplanes(encoded, bits), bits


def _encode_presence_stream(
    entry_cell_maps: list[dict[tuple[int, int], int]],
    pair_rows: list[tuple[int, int]],
) -> bytes:
    pair_index = {pair: idx for idx, pair in enumerate(pair_rows)}
    previous = [0] * len(pair_rows)
    mask_byte_length = (len(pair_rows) + 7) // 8
    flag_byte_length = (mask_byte_length + 7) // 8
    out = bytearray()
    for cell_map in entry_cell_maps:
        current = [0] * len(pair_rows)
        for pair in cell_map:
            current[pair_index[pair]] = 1
        changed = pack_little_endian_bits(
            [(left ^ right, 1) for left, right in zip(previous, current)],
            chunk_bytes=4,
        )
        flags = bytearray(flag_byte_length)
        values = bytearray()
        for byte_idx, value in enumerate(changed):
            if value:
                flags[byte_idx // 8] |= 1 << (byte_idx % 8)
                values.append(value)
        out.extend(flags)
        out.extend(values)
        previous = current
    return bytes(out)


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


def _build_row_coord_fraction_stream(
    entry_cell_maps: list[dict[tuple[int, int], int]],
    pair_rows: list[tuple[int, int]],
    tenuki_rows: list[int],
) -> bytes:
    all_values = [value for cell_map in entry_cell_maps for value in cell_map.values()]
    global_mean = int(round(sum(all_values) / len(all_values))) if all_values else 0
    row_means = [
        int(round(sum(cell_map.values()) / len(cell_map))) if cell_map else global_mean
        for cell_map in entry_cell_maps
    ]
    row_mean_residuals: list[int] = []
    for row_mean, tenuki in zip(row_means, tenuki_rows):
        prediction = _round_ratio_half_even(
            ROW_MEAN_PREDICTOR_NUMERATOR * int(tenuki),
            ROW_MEAN_PREDICTOR_DENOMINATOR,
        ) + ROW_MEAN_PREDICTOR_INTERCEPT
        row_mean_residuals.append(int(row_mean) - prediction)
    row_mean_residual_magnitude_bits = max(
        1,
        max((abs(residual) for residual in row_mean_residuals), default=0).bit_length(),
    )
    row_mean_residual_codes = [
        abs(residual)
        | ((1 if residual < 0 else 0) << row_mean_residual_magnitude_bits)
        for residual in row_mean_residuals
    ]
    pair_means: dict[tuple[int, int], int] = {}
    for pair in pair_rows:
        values = [cell_map[pair] for cell_map in entry_cell_maps if pair in cell_map]
        pair_means[pair] = int(round(sum(values) / len(values)))
    residuals: list[int] = []
    for pair in pair_rows:
        for entry_idx, cell_map in enumerate(entry_cell_maps):
            value = cell_map.get(pair)
            if value is None:
                continue
            prediction = row_means[entry_idx] + pair_means[pair] - global_mean
            residuals.append(int(value) - prediction)
    residual_stream, residual_bits = _pack_signed_bitplanes(residuals)
    out = bytearray()
    out.extend(
        ROW_MEAN_PREDICTOR_STRUCT.pack(
            ROW_MEAN_PREDICTOR_NUMERATOR,
            ROW_MEAN_PREDICTOR_DENOMINATOR,
            ROW_MEAN_PREDICTOR_INTERCEPT,
            row_mean_residual_magnitude_bits,
        )
    )
    out.extend(
        _pack_bitplanes(
            row_mean_residual_codes,
            row_mean_residual_magnitude_bits + 1,
        )
    )
    out.extend(_pack_bitplanes([pair_means[pair] for pair in pair_rows], PACKED_STONE_FRACTION_BITS))
    out.extend(struct.pack("<H", global_mean))
    out.append(residual_bits)
    out.extend(residual_stream)
    return bytes(out)


def _round_ratio_half_even(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"Pattern predictor denominator must be positive: {denominator!r}")
    quotient, remainder = divmod(int(numerator), int(denominator))
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


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
        key_stream.extend(encode_uvarint(prefix_len))
        key_stream.extend(encode_uvarint(len(suffix)))
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

    cell_count = sum(len(cell_map) for cell_map in entry_cell_maps)
    presence_stream = _encode_presence_stream(entry_cell_maps, pair_rows)

    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            FRACTION_MODE_AFFINE_ROW_COORD_BITPLANE,
            len(entry_cell_maps),
            cell_count,
            len(key_stream),
        )
    )
    out.extend(_pack_bitplanes(tenuki_rows, PACKED_STONE_FRACTION_BITS))
    out.extend(PAIR_COUNT_STRUCT.pack(len(pair_rows)))
    for q, r in pair_rows:
        out.extend(PAIR_ROW_STRUCT.pack(int(q), int(r)))
    out.extend(struct.pack("<I", len(presence_stream)))
    out.extend(presence_stream)
    out.extend(_build_row_coord_fraction_stream(entry_cell_maps, pair_rows, tenuki_rows))
    out.extend(key_stream)
    return bytes(out)


def _layer_for_pattern(pattern: str) -> WebsitePatternLayer:
    move_count = _move_count_for_labeled_pattern(pattern)
    for layer in WEBSITE_PATTERN_LAYERS:
        if layer.min_moves <= move_count <= layer.max_moves:
            return layer
    raise ValueError(f"Pattern {pattern!r} has no website layer for {move_count} moves")


def _layer_manifest_row(layer: WebsitePatternLayer, *, bundle_name: str, pattern_count: int) -> dict[str, Any]:
    return {
        "id": str(layer.id),
        "bundle": str(bundle_name),
        "min_moves": int(layer.min_moves),
        "max_moves": int(layer.max_moves),
        "pattern_count": int(pattern_count),
    }


def write_layered_pattern_index(
    *,
    artifacts_roots: Iterable[Path],
    out_path: Path,
) -> Path:
    repo_root = _repo_root()
    payload = build_pattern_index(artifacts_roots=artifacts_roots, repo_root=repo_root)
    total_count = 0
    layer_bundles: dict[str, BundlePayload] = {}
    layer_counts: dict[str, int] = {}
    patterns_by_layer_id = {layer.id: {} for layer in WEBSITE_PATTERN_LAYERS}
    for pattern, entry in dict(payload["patterns"]).items():
        layer = _layer_for_pattern(str(pattern))
        patterns_by_layer_id[layer.id][pattern] = entry

    for layer in WEBSITE_PATTERN_LAYERS:
        layer_patterns = patterns_by_layer_id[layer.id]
        layer_index = {
            "version": BUNDLE_VERSION,
            "pattern_count": int(len(layer_patterns)),
            "patterns": layer_patterns,
        }
        bundle = _build_pattern_bundle_from_index(layer_index)
        total_count += int(layer_index["pattern_count"])
        layer_bundles[layer.id] = BundlePayload(prefix="pattern_index", payload=bundle)
        layer_counts[layer.id] = int(layer_index["pattern_count"])

    def manifest_from_bundle_names(bundle_names: dict[str, str]) -> dict[str, Any]:
        return {
            "version": BUNDLE_VERSION,
            "pattern_count": int(total_count),
            "layers": [
                _layer_manifest_row(
                    layer,
                    bundle_name=bundle_names[layer.id],
                    pattern_count=layer_counts[layer.id],
                )
                for layer in WEBSITE_PATTERN_LAYERS
            ],
        }

    return write_hashed_bundle_manifest(
        out_path=out_path,
        bundles=layer_bundles,
        stale_globs=["pattern_index.*.bin"],
        manifest_from_bundle_names=manifest_from_bundle_names,
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build layered binary bundles for the pattern website")
    ap.add_argument("--out", default=str(_default_out_path()))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    write_layered_pattern_index(
        artifacts_roots=_default_artifacts_roots(),
        out_path=Path(str(args.out)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
