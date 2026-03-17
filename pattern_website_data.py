from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pattern_output_utils import movelist_slug_from_hexworld
from pattern_notation import LabeledPattern, canonicalize, format_pattern, parse_pattern
from website_bundle_utils import (
    BundlePayload,
    PACKED_OPTIONAL_U10_NULL,
    encode_thousandths,
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
PACKED_KEY_COORD_MIN = -8
PACKED_KEY_COORD_MAX = 7
ENCODER_INPUT_HEADER_STRUCT = struct.Struct("<4sI")
ENCODER_INPUT_CELL_STRUCT = struct.Struct("<bbH")
ENCODER_INPUT_MAGIC = b"HPI1"


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


def _pattern_bundle_encoder() -> Path:
    source = _repo_root() / "pattern_bundle_encode.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path(tempfile.gettempdir()) / f"hexwiki-pattern-bundle-{digest}"
    if binary.exists():
        return binary
    pending = binary.with_name(f"{binary.name}.{os.getpid()}.tmp")
    proc = subprocess.run(
        ["c++", "-O3", "-std=c++17", str(source), "-o", str(pending)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Pattern bundle encoder build failed{suffix}")
    pending.replace(binary)
    return binary


def _encode_pattern_bundle_input(index: dict[str, Any]) -> bytes:
    patterns = index.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError("Pattern index missing patterns object")
    if int(index.get("pattern_count", -1)) != len(patterns):
        raise ValueError("Pattern index count does not match patterns")
    out = bytearray(ENCODER_INPUT_HEADER_STRUCT.pack(ENCODER_INPUT_MAGIC, len(patterns)))
    for pattern, entry in patterns.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Bad compact pattern entry: {entry!r}")
        pattern_b = _pack_pattern_key(str(pattern))
        if len(pattern_b) > 0xFF:
            raise ValueError(f"Pattern key too long for encoder input: {pattern!r}")
        if str(entry.get("p") or "") != _to_play_from_packed_pattern_key(pattern_b):
            raise ValueError(f"Pattern side-to-play is not derivable from key: {pattern!r}")
        tenuki = int(entry.get("t", PACKED_OPTIONAL_NULL))
        if tenuki < 0 or tenuki >= (1 << PACKED_STONE_FRACTION_BITS):
            raise ValueError(f"Pattern tenuki fraction out of u10 range: {tenuki!r}")
        cells = entry.get("c")
        if not isinstance(cells, list) or len(cells) > 0xFFFF:
            raise ValueError(f"Bad compact pattern cells: {cells!r}")
        out.append(len(pattern_b))
        out.extend(pattern_b)
        out.extend(struct.pack("<HH", tenuki, len(cells)))
        seen: set[tuple[int, int]] = set()
        for cell in cells:
            if not isinstance(cell, list) or len(cell) != 3:
                raise ValueError(f"Bad compact pattern cell: {cell!r}")
            q, r, stone_fraction = (int(value) for value in cell)
            if q < -128 or q > 127 or r < -128 or r > 127:
                raise ValueError(f"Pattern local coordinate out of i8 range: {cell!r}")
            if stone_fraction < 0 or stone_fraction >= (1 << PACKED_STONE_FRACTION_BITS):
                raise ValueError(f"Pattern stone fraction out of u10 range: {cell!r}")
            pair = (q, r)
            if pair in seen:
                raise ValueError(f"Duplicate pattern local coordinate in binary bundle: {cell!r}")
            seen.add(pair)
            out.extend(ENCODER_INPUT_CELL_STRUCT.pack(q, r, stone_fraction))
    return bytes(out)


def _build_pattern_bundle_from_index(index: dict[str, Any]) -> bytes:
    proc = subprocess.run(
        [str(_pattern_bundle_encoder())],
        input=_encode_pattern_bundle_input(index),
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Pattern bundle encoding failed{suffix}")
    return proc.stdout


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
