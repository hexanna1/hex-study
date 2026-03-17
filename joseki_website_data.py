from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joseki_notation as jn


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "joseki"


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / "joseki_current.json"


def _artifact_path(*, artifacts_root: Path, family: str, board_size: int) -> Path:
    family_s = str(family).strip().lower()
    return artifacts_root / f"joseki-{family_s}-s{int(board_size)}.json"


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def _bundle_filename(*, family: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_serialize_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"joseki_{str(family).strip().lower()}.{digest}.json"


def _normalize_local(raw: Any) -> list[int]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"bad local move payload: {raw!r}")
    x, y = raw
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"bad local move coordinates: {raw!r}")
    return [int(x), int(y)]


def _parse_entries(line: str) -> tuple[tuple[int, int] | None, ...]:
    raw = str(line or "").strip()
    if not raw:
        return ()
    return jn.parse_joseki_line(raw).blocks[0].entries


def _compact_node(*, family: str, node: dict[str, Any]) -> dict[str, Any]:
    line = str(node.get("line") or "")
    entries = _parse_entries(line)
    retained = {str(line_value or "") for line_value in list(node.get("retained_lines") or [])}
    local_rows: list[list[Any]] = []
    tenuki_row: list[Any] | None = None

    for row in list(node.get("candidates") or []):
        kind = str(row.get("kind") or "").strip()
        stone_fraction = row.get("stone_fraction")
        if not isinstance(stone_fraction, (int, float)):
            continue
        if kind == "local":
            local = _normalize_local(row.get("local"))
            child_line = jn.format_single_track_line(
                family=family,
                entries=entries + ((int(local[0]), int(local[1])),),
            )
            if child_line not in retained:
                continue
            local_rows.append([int(local[0]), int(local[1]), round(float(stone_fraction), 3)])
        elif kind == "tenuki":
            child_line = jn.format_single_track_line(
                family=family,
                entries=entries + (None,),
            ) if line else ""
            tenuki_row = [round(float(stone_fraction), 3), 1 if child_line in retained and line else 0]

    compact: dict[str, Any] = {
        "l": line,
        "c": local_rows,
        "i": round(float(node.get("importance", 0.0)), 3),
    }
    if tenuki_row is not None:
        compact["t"] = tenuki_row
    return compact


def build_family_bundle(*, artifacts_root: Path, family: str, board_size: int) -> dict[str, Any]:
    family_s = str(family).strip().upper()
    if family_s not in {"A", "O"}:
        raise ValueError(f"unsupported family: {family!r}")
    data = json.loads(
        _artifact_path(artifacts_root=artifacts_root, family=family_s, board_size=board_size).read_text(encoding="utf-8")
    )
    nodes = list(data.get("nodes") or [])
    return {
        "version": 1,
        "family": family_s,
        "board_size": int(data["board_size"]),
        "nodes": [_compact_node(family=family_s, node=node) for node in nodes],
    }


def write_joseki_bundles(*, artifacts_root: Path, out_path: Path, board_size: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bundles: dict[str, str] = {}
    written_bundles: set[str] = set()
    for family in ("A", "O"):
        payload = build_family_bundle(artifacts_root=artifacts_root, family=family, board_size=board_size)
        bundle_name = _bundle_filename(family=family, payload=payload)
        bundle_path = out_path.parent / bundle_name
        bundle_path.write_text(_serialize_json(payload), encoding="utf-8")
        manifest_bundles[family] = bundle_name
        written_bundles.add(bundle_name)
    for path in out_path.parent.glob("joseki_[ao].*.json"):
        if path.name not in written_bundles:
            path.unlink()
    manifest = {
        "version": 1,
        "bundles": manifest_bundles,
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact JSON bundles for the joseki website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    ap.add_argument("--board-size", default="19")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    write_joseki_bundles(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
        board_size=int(args.board_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
