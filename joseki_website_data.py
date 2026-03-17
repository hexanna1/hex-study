from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import joseki_notation as jn

BUNDLE_MAGIC = b"HJB1"
BUNDLE_VERSION = 1
FAMILY_CODE_BY_NAME = {
    "A": 1,
    "O": 2,
}
TENUKI_MOVE_CODE = 255
ROOT_MOVE_CODE = 0
HEADER_STRUCT = struct.Struct("<4sHBBII")
LOCAL_ROW_STRUCT = struct.Struct("<BH")
NODE_ROW_SIZE = 7


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


def _bundle_filename(*, family: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"joseki_{str(family).strip().lower()}.{digest}.bin"


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


def _format_line(*, family: str, entries: tuple[tuple[int, int] | None, ...]) -> str:
    if not entries:
        return ""
    return jn.format_single_track_line(family=family, entries=entries)


def _encode_thousandths(raw: Any) -> int:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"bad numeric payload: {raw!r}")
    return int(round(round(float(raw), 3) * 1000.0))


def _encode_family_code(family: str) -> int:
    family_s = str(family).strip().upper()
    code = FAMILY_CODE_BY_NAME.get(family_s)
    if not isinstance(code, int):
        raise ValueError(f"unsupported family: {family!r}")
    return int(code)


def _encode_local_move_code(local: list[int]) -> int:
    x = int(local[0])
    y = int(local[1])
    if x < 1 or x > 10 or y < 1 or y > 10:
        raise ValueError(f"bad joseki local move payload: {local!r}")
    return ((x - 1) * 10) + (y - 1)


def _pack_node_flags(*, local_count: int, is_core: bool, tenuki_retained: bool, tenuki_present: bool) -> int:
    local_count_i = int(local_count)
    if local_count_i < 0 or local_count_i > 31:
        raise ValueError(f"bad joseki local child count: {local_count!r}")
    return (
        local_count_i
        | (0x20 if bool(is_core) else 0x00)
        | (0x40 if bool(tenuki_retained) else 0x00)
        | (0x80 if bool(tenuki_present) else 0x00)
    )


def _pack_parent_plus_1(parent_index: int) -> bytes:
    parent_plus_1 = int(parent_index) + 1
    if parent_plus_1 < 0 or parent_plus_1 > 0xFFFFFF:
        raise ValueError(f"bad joseki parent index: {parent_index!r}")
    return bytes((
        parent_plus_1 & 0xFF,
        (parent_plus_1 >> 8) & 0xFF,
        (parent_plus_1 >> 16) & 0xFF,
    ))


def _compact_node(
    *,
    family: str,
    node: dict[str, Any],
    line_to_index: dict[str, int],
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    line = str(node.get("line") or "")
    entries = _parse_entries(line)
    retained = {str(line_value or "") for line_value in list(node.get("retained_lines") or [])}
    local_rows: list[tuple[int, int]] = []
    tenuki_present = False
    tenuki_sf = 0
    tenuki_retained = False

    for row in list(node.get("candidates") or []):
        kind = str(row.get("kind") or "").strip()
        stone_fraction = row.get("stone_fraction")
        if not isinstance(stone_fraction, (int, float)):
            continue
        stone_fraction_i = _encode_thousandths(stone_fraction)
        if kind == "local":
            local = _normalize_local(row.get("local"))
            child_line = _format_line(family=family, entries=entries + ((int(local[0]), int(local[1])),))
            if child_line not in retained:
                continue
            local_rows.append((_encode_local_move_code(local), stone_fraction_i))
        elif kind == "tenuki":
            child_line = _format_line(family=family, entries=entries + (None,)) if line else ""
            tenuki_present = True
            tenuki_sf = stone_fraction_i
            tenuki_retained = bool(child_line in retained and line)

    parent_index = -1
    move_code = ROOT_MOVE_CODE
    if entries:
        parent_line = _format_line(family=family, entries=entries[:-1])
        parent_index = line_to_index.get(parent_line)
        if not isinstance(parent_index, int):
            raise ValueError(f"missing joseki parent node for line {line!r}")
        if parent_index >= int(line_to_index.get(line, -1)):
            raise ValueError(f"joseki nodes out of parent-first order at line {line!r}")
        move_code = TENUKI_MOVE_CODE if entries[-1] is None else _encode_local_move_code([entries[-1][0], entries[-1][1]])

    is_core = float(node.get("importance", 0.0)) >= 0.875
    node_row = (
        int(parent_index),
        int(move_code),
        _pack_node_flags(
            local_count=len(local_rows),
            is_core=is_core,
            tenuki_retained=tenuki_retained,
            tenuki_present=tenuki_present,
        ),
        int(tenuki_sf),
    )
    return node_row, local_rows


def build_family_bundle(*, artifacts_root: Path, family: str, board_size: int) -> bytes:
    family_s = str(family).strip().upper()
    data = json.loads(
        _artifact_path(artifacts_root=artifacts_root, family=family_s, board_size=board_size).read_text(encoding="utf-8")
    )
    nodes = list(data.get("nodes") or [])
    line_to_index = {str(node.get("line") or ""): idx for idx, node in enumerate(nodes)}
    compact_nodes: list[tuple[int, int, int, int]] = []
    compact_local_rows: list[tuple[int, int]] = []
    for node in nodes:
        node_row, local_rows = _compact_node(family=family_s, node=node, line_to_index=line_to_index)
        compact_nodes.append(node_row)
        compact_local_rows.extend(local_rows)
    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            _encode_family_code(family_s),
            int(data["board_size"]),
            len(compact_nodes),
            len(compact_local_rows),
        )
    )
    for parent_index, move_code, flags, tenuki_sf in compact_nodes:
        out.extend(_pack_parent_plus_1(parent_index))
        out.extend(
            bytes((
                int(move_code),
                int(flags),
            ))
        )
        out.extend(struct.pack("<H", int(tenuki_sf)))
    for move_code, stone_fraction in compact_local_rows:
        out.extend(LOCAL_ROW_STRUCT.pack(int(move_code), int(stone_fraction)))
    return bytes(out)


def write_joseki_bundles(*, artifacts_root: Path, out_path: Path, board_size: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bundles: dict[str, str] = {}
    written_bundles: set[str] = set()
    for family in ("A", "O"):
        payload = build_family_bundle(artifacts_root=artifacts_root, family=family, board_size=board_size)
        bundle_name = _bundle_filename(family=family, payload=payload)
        bundle_path = out_path.parent / bundle_name
        bundle_path.write_bytes(payload)
        manifest_bundles[family] = bundle_name
        written_bundles.add(bundle_name)
    for path in out_path.parent.glob("joseki_[ao].*.bin"):
        if path.name not in written_bundles:
            path.unlink()
    manifest = {
        "version": 1,
        "bundles": manifest_bundles,
    }
    out_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact binary bundles for the joseki website")
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
