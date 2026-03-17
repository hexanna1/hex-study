from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pattern_output_utils as sout
from pattern_notation import LabeledPattern, canonicalize, format_pattern, parse_pattern


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "interior_patterns_m5_d7_span12"


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
        study_delta = row.get("study_delta")
        if not hexworld:
            raise ValueError(f"Catalog row missing hexworld_21: {catalog_path}")
        if not isinstance(study_delta, int):
            raise ValueError(f"Catalog row missing integer study_delta: {catalog_path}")
        slug = sout._movelist_slug_from_hexworld(hexworld)
        path = tiles_dir / f"d{int(study_delta):02d}-{slug}.json"
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
    cell = {"kind": kind, "stone_fraction": round(float(stone_fraction), 3), "rank": int(rank)}
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
    local_cells: list[dict[str, Any]] = []
    tenuki_stone_fraction: float | None = None
    for cell in list(spec.get("cells") or []):
        kind = str(cell.get("kind") or "")
        if kind == "local":
            local_rel = cell.get("local_rel")
            if not isinstance(local_rel, list) or len(local_rel) != 2:
                raise ValueError(f"Local cell missing local_rel: {cell!r}")
            local_cells.append(
                {
                    "l": [int(local_rel[0]), int(local_rel[1])],
                    "s": round(float(cell["stone_fraction"]), 3),
                }
            )
        elif kind == "tenuki":
            tenuki_stone_fraction = round(float(cell["stone_fraction"]), 3)
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


def _bundle_filename(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_serialize_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"pattern_index.{digest}.json"


def write_pattern_index(*, artifacts_root: Path, out_path: Path) -> Path:
    repo_root = _repo_root()
    payload = build_pattern_index(
        artifacts_root=artifacts_root,
        repo_root=repo_root,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_name = _bundle_filename(payload)
    bundle_path = out_path.parent / bundle_name
    bundle_path.write_text(_serialize_json(payload), encoding="utf-8")
    for path in out_path.parent.glob("pattern_index.*.json"):
        if path.name != bundle_name:
            path.unlink()
    manifest = {
        "version": 1,
        "bundle": bundle_name,
        "pattern_count": int(payload["pattern_count"]),
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a unified JSON bundle for the pattern website")
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
