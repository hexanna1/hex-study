from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


OPENINGS_MODE = "openings"
OPENINGS_ARTIFACT_DIR = "openings"
OPENINGS_OUT_NAME = "openings_current.json"
OPENING_BUNDLE_PREFIX = "opening_index"
BUNDLE_MAGIC = b"HOB1"
BUNDLE_VERSION = 1
ROOT_MOVE_ID = -1
CORE_IMPORTANCE_MIN_THOUSANDTHS = 910
PACKED_MOVE_ID_MAX = 1023
PACKED_OPTIONAL_NULL = 1023
HEADER_STRUCT = struct.Struct("<4sHHII")
NODE_TAIL_STRUCT = struct.Struct("<hB")


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def _bundle_filename(*, prefix: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{prefix}.{digest}.bin"


def _require_move_tree_version(raw: Any) -> None:
    if not isinstance(raw, int):
        raise ValueError(f"bad move-tree version payload: {raw!r}")
    if int(raw) != 1:
        raise ValueError(f"unsupported move-tree version: {raw!r}")


def _require_move_tree_format(raw: Any) -> None:
    value = str(raw or "").strip()
    if value != "move_tree":
        raise ValueError(f"unsupported move-tree format: {raw!r}")


def _normalize_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 3)
    raise ValueError(f"bad numeric payload: {raw!r}")


def _encode_optional_thousandths(raw: Any) -> int | None:
    value = _normalize_optional_float(raw)
    if value is None:
        return None
    return int(round(float(value) * 1000.0))


def _pack_optional_u10(value: int | None) -> int:
    if value is None:
        return PACKED_OPTIONAL_NULL
    if value < 0 or value >= PACKED_OPTIONAL_NULL:
        raise ValueError(f"bad u10 payload: {value!r}")
    return int(value)


def _pack_parent_u24(parent: int) -> bytes:
    value = int(parent) + 1
    if value < 0 or value > 0xFFFFFF:
        raise ValueError(f"bad u24 parent payload: {parent!r}")
    return bytes((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF))


def _pack_packed_candidate(*, move_id: int, prior: int | None, mover_winrate: int | None) -> bytes:
    move_id_i = int(move_id)
    if move_id_i < 0 or move_id_i > PACKED_MOVE_ID_MAX:
        raise ValueError(f"bad packed move id payload: {move_id!r}")
    prior_u10 = _pack_optional_u10(prior)
    mover_winrate_u10 = _pack_optional_u10(mover_winrate)
    word = move_id_i | (prior_u10 << 10) | (mover_winrate_u10 << 20)
    return struct.pack("<I", word)


def _pack_node_flags(*, candidate_count: int, is_core: bool) -> int:
    count = int(candidate_count)
    if count < 0 or count > 127:
        raise ValueError(f"bad 7-bit candidate count payload: {candidate_count!r}")
    return count | (0x80 if bool(is_core) else 0x00)


def _normalize_parent(raw: Any) -> int:
    if raw is None:
        return -1
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"bad parent index: {raw!r}")
        return int(raw)
    raise ValueError(f"bad parent payload: {raw!r}")


def _move_to_cell_id(move: str, *, board_size: int) -> int:
    token = str(move or "").strip().lower()
    if not token:
        raise ValueError("move must not be empty")
    idx = 0
    col = 0
    while idx < len(token) and "a" <= token[idx] <= "z":
        col = (26 * col) + (ord(token[idx]) - 96)
        idx += 1
    if idx == 0 or idx >= len(token):
        raise ValueError(f"bad move payload: {move!r}")
    row_text = token[idx:]
    if not row_text.isdigit() or row_text.startswith("0"):
        raise ValueError(f"bad move payload: {move!r}")
    row = int(row_text)
    if col < 1 or col > int(board_size) or row < 1 or row > int(board_size):
        raise ValueError(f"move out of bounds for board size {board_size}: {move!r}")
    return ((row - 1) * int(board_size)) + (col - 1)


def _normalize_optional_move_id(raw: Any, *, board_size: int) -> int:
    if raw is None:
        return ROOT_MOVE_ID
    move = str(raw or "").strip().lower()
    if not move:
        return ROOT_MOVE_ID
    return _move_to_cell_id(move, board_size=board_size)


def _normalize_retained(raw: Any) -> int:
    if isinstance(raw, bool):
        return 1 if raw else 0
    raise ValueError(f"bad retained payload: {raw!r}")


def build_opening_bundle(*, artifacts_root: Path, board_size: int) -> bytes:
    artifact_path = Path(artifacts_root) / f"openings-s{int(board_size)}.json"
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"move-tree artifact must be a JSON object: {artifact_path}")
    _require_move_tree_version(data.get("version"))
    _require_move_tree_format(data.get("format"))
    mode = str(data.get("mode") or "").strip().lower()
    if mode != OPENINGS_MODE:
        raise ValueError(
            f"move-tree artifact mode mismatch for {artifact_path}: expected {OPENINGS_MODE!r}, got {mode!r}"
        )
    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, list):
        raise ValueError(f"move-tree artifact missing nodes list: {artifact_path}")
    artifact_board_size = int(data["board_size"])

    nodes: list[list[Any]] = []
    candidates: list[list[Any]] = []
    for node_raw in nodes_raw:
        if not isinstance(node_raw, dict):
            raise ValueError(f"bad node payload: {node_raw!r}")
        node_candidates = node_raw.get("candidates")
        if not isinstance(node_candidates, list):
            raise ValueError(f"node missing candidates list: {node_raw!r}")
        retained_count = 0
        for candidate_raw in node_candidates:
            if not isinstance(candidate_raw, dict):
                raise ValueError(f"bad candidate payload: {candidate_raw!r}")
            retained = _normalize_retained(candidate_raw.get("retained"))
            if retained != 1:
                continue
            move = str(candidate_raw.get("move") or "").strip().lower()
            if not move:
                raise ValueError(f"candidate missing move: {candidate_raw!r}")
            candidates.append(
                [
                    _move_to_cell_id(move, board_size=artifact_board_size),
                    _encode_optional_thousandths(candidate_raw.get("prior")),
                    _encode_optional_thousandths(candidate_raw.get("tree_mover_winrate")),
                ]
            )
            retained_count += 1
        nodes.append(
            [
                _normalize_parent(node_raw.get("parent")),
                _normalize_optional_move_id(node_raw.get("move"), board_size=artifact_board_size),
                _encode_optional_thousandths(node_raw.get("importance")),
                int(retained_count),
            ]
        )
    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            artifact_board_size,
            len(nodes),
            len(candidates),
        )
    )
    for parent, move_id, importance, candidate_count in nodes:
        if not isinstance(importance, int):
            raise ValueError(f"bad importance payload: {importance!r}")
        out.extend(_pack_parent_u24(parent))
        out.extend(
            NODE_TAIL_STRUCT.pack(
                move_id,
                _pack_node_flags(
                    candidate_count=candidate_count,
                    is_core=(int(importance) >= CORE_IMPORTANCE_MIN_THOUSANDTHS),
                ),
            )
        )
    for move_id, prior, mover_winrate in candidates:
        out.extend(
            _pack_packed_candidate(
                move_id=move_id,
                prior=prior,
                mover_winrate=mover_winrate,
            )
        )
    return bytes(out)


def write_opening_bundles(
    *,
    artifacts_root: Path,
    out_path: Path,
    board_sizes: list[int],
) -> Path:
    bundle_entries = {
        str(int(board_size)): build_opening_bundle(
            artifacts_root=artifacts_root,
            board_size=board_size,
        )
        for board_size in board_sizes
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_names: dict[str, str] = {}
    written_bundles: set[str] = set()
    for key, payload in bundle_entries.items():
        bundle_name = _bundle_filename(prefix=OPENING_BUNDLE_PREFIX, payload=payload)
        bundle_path = out_path.parent / bundle_name
        bundle_path.write_bytes(payload)
        bundle_names[str(key)] = bundle_name
        written_bundles.add(bundle_name)
    for path in out_path.parent.glob(f"{OPENING_BUNDLE_PREFIX}.*.bin"):
        if path.name not in written_bundles:
            path.unlink()
    manifest = {
        "version": 1,
        "bundles": bundle_names,
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / OPENINGS_ARTIFACT_DIR


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / OPENINGS_OUT_NAME


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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact binary bundles for opening move-tree websites")
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
