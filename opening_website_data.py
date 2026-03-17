from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "openings"


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / "openings_current.json"


def _artifact_path(*, artifacts_root: Path, board_size: int) -> Path:
    return artifacts_root / f"openings-s{int(board_size)}.json"


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def _bundle_filename(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_serialize_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"opening_index.{digest}.json"


def _parse_board_sizes(raw: Any) -> list[int]:
    values = str(raw or "").strip()
    if not values:
        raise ValueError("board sizes must not be empty")
    out: list[int] = []
    for item in values.split(","):
        size = int(str(item).strip())
        if size not in {11, 12, 13, 14, 17}:
            raise ValueError(f"unsupported board size: {size!r}")
        if size not in out:
            out.append(size)
    return out


def _normalize_moves(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Bad moves payload: {raw!r}")
    out: list[str] = []
    for item in raw:
        move = str(item or "").strip().lower()
        if not move:
            raise ValueError(f"Bad move in moves payload: {raw!r}")
        out.append(move)
    return out


def _normalize_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 3)
    raise ValueError(f"Bad numeric payload: {raw!r}")


def _compact_candidate(raw: Any) -> list[Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Bad candidate payload: {raw!r}")
    move = str(raw.get("move") or "").strip().lower()
    retained = raw.get("retained")
    if not move:
        raise ValueError(f"Candidate missing move: {raw!r}")
    if not isinstance(retained, bool):
        raise ValueError(f"Candidate missing retained flag: {raw!r}")
    return [
        move,
        _normalize_optional_float(raw.get("prior")),
        _normalize_optional_float(raw.get("mover_winrate")),
        1 if retained else 0,
    ]


def _compact_node(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Bad opening node payload: {raw!r}")
    moves = _normalize_moves(raw.get("moves") or [])
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Node missing candidates list: {raw!r}")
    compact = {
        "m": "".join(moves),
        "c": [_compact_candidate(row) for row in candidates],
    }
    importance = raw.get("importance")
    if isinstance(importance, (int, float)):
        compact["i"] = round(float(importance), 3)
    return compact


def build_opening_bundle(*, artifacts_root: Path, board_size: int) -> dict[str, Any]:
    data = json.loads(_artifact_path(artifacts_root=artifacts_root, board_size=board_size).read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Opening artifact missing nodes list")
    return {
        "version": 1,
        "board_size": int(data["board_size"]),
        "node_count": int(len(nodes)),
        "nodes": [_compact_node(node) for node in nodes],
    }


def write_opening_bundles(*, artifacts_root: Path, out_path: Path, board_sizes: list[int]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_names: dict[str, str] = {}
    written_bundles: set[str] = set()
    for board_size in board_sizes:
        payload = build_opening_bundle(artifacts_root=artifacts_root, board_size=board_size)
        bundle_name = _bundle_filename(payload)
        bundle_path = out_path.parent / bundle_name
        bundle_path.write_text(_serialize_json(payload), encoding="utf-8")
        bundle_names[str(int(board_size))] = bundle_name
        written_bundles.add(bundle_name)
    for path in out_path.parent.glob("opening_index.*.json"):
        if path.name not in written_bundles:
            path.unlink()
    manifest = {
        "version": 1,
        "bundles": bundle_names,
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a compact JSON bundle for the opening website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    ap.add_argument("--board-sizes", default="11,12,13,14,17")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    write_opening_bundles(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
        board_sizes=_parse_board_sizes(args.board_sizes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
