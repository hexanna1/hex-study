from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import joseki_notation as jn
from website_bundle_utils import (
    BundlePayload,
    encode_thousandths,
    pack_little_endian_bits,
    write_hashed_bundle_manifest,
)

BUNDLE_MAGIC = b"HJB1"
BUNDLE_VERSION = 1
FAMILY_CODE_BY_NAME = {
    "A": 1,
    "O": 2,
}
CORE_IMPORTANCE_MIN_THOUSANDTHS = 825
HEADER_STRUCT = struct.Struct("<4sHBBII")
PACKED_NODE_LOCAL_COUNT_BITS = 4
PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_LOCAL_COUNT_BITS
PACKED_NODE_TENUKI_RETAINED_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
PACKED_NODE_TENUKI_CHILD_SHIFT = PACKED_NODE_TENUKI_RETAINED_SHIFT + 1
PACKED_NODE_LOCAL_CHILDREN_SHIFT = PACKED_NODE_TENUKI_CHILD_SHIFT + 1
PACKED_TENUKI_DROP_HIGH_BITS = 2
PACKED_LOCAL_DROP_MAX = 0xFF


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "joseki"


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / "joseki_current.json"


def _artifact_path(*, artifacts_root: Path, family: str, board_size: int) -> Path:
    family_s = str(family).strip().lower()
    return artifacts_root / f"joseki-{family_s}-s{int(board_size)}.json"


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


def _pack_bitplanes(values: list[int], bits: int) -> bytes:
    return pack_little_endian_bits(
        [
            ((int(value) >> bit) & 1, 1)
            for bit in range(int(bits) - 1, -1, -1)
            for value in values
        ],
        chunk_bytes=4,
    )


def _pack_node_control(
    *,
    local_count: int,
    is_core: bool,
    tenuki_retained: bool,
    tenuki_child_present: bool,
    local_children: bool,
) -> int:
    local_count_i = int(local_count)
    if local_count_i < 0 or local_count_i >= (1 << PACKED_NODE_LOCAL_COUNT_BITS):
        raise ValueError(f"bad joseki local child count: {local_count!r}")
    return (
        local_count_i
        | ((1 if bool(is_core) else 0) << PACKED_NODE_IS_CORE_SHIFT)
        | ((1 if bool(tenuki_retained) else 0) << PACKED_NODE_TENUKI_RETAINED_SHIFT)
        | ((1 if bool(tenuki_child_present) else 0) << PACKED_NODE_TENUKI_CHILD_SHIFT)
        | ((1 if bool(local_children) else 0) << PACKED_NODE_LOCAL_CHILDREN_SHIFT)
    )


def _compact_node(
    *,
    family: str,
    node: dict[str, Any],
    line_to_node: dict[str, dict[str, Any]],
) -> tuple[int, bool, int, list[tuple[int, int]], list[dict[str, Any]]]:
    line = str(node.get("line") or "")
    entries = _parse_entries(line)
    retained = {str(line_value or "") for line_value in list(node.get("retained_lines") or [])}
    local_rows: list[tuple[int, int]] = []
    local_child_bits: list[bool] = []
    local_children: list[dict[str, Any]] = []
    tenuki_child: dict[str, Any] | None = None
    tenuki_present = False
    tenuki_sf = 0
    tenuki_retained = False
    tenuki_child_present = False

    for row in list(node.get("candidates") or []):
        kind = str(row.get("kind") or "").strip()
        stone_fraction = row.get("stone_fraction")
        if not isinstance(stone_fraction, (int, float)):
            continue
        stone_fraction_i = encode_thousandths(stone_fraction)
        if kind == "local":
            local = _normalize_local(row.get("local"))
            child_line = _format_line(family=family, entries=entries + ((int(local[0]), int(local[1])),))
            if child_line not in retained:
                continue
            local_rows.append((_encode_local_move_code(local), stone_fraction_i))
            child_node = line_to_node.get(child_line)
            local_child_bits.append(child_node is not None)
            if child_node is not None:
                local_children.append(child_node)
        elif kind == "tenuki":
            child_line = _format_line(family=family, entries=entries + (None,)) if line else ""
            tenuki_present = True
            tenuki_sf = stone_fraction_i
            tenuki_retained = bool(child_line in retained and line)
            child_node = line_to_node.get(child_line) if tenuki_retained else None
            tenuki_child_present = child_node is not None
            if child_node is not None:
                tenuki_child = child_node

    importance = encode_thousandths(node.get("importance", 0.0))
    is_core = int(importance) >= CORE_IMPORTANCE_MIN_THOUSANDTHS
    local_child_count = sum(1 for value in local_child_bits if value)
    if local_child_count not in {0, len(local_child_bits)}:
        raise ValueError("joseki node has mixed local child presence")
    node_control = _pack_node_control(
        local_count=len(local_rows),
        is_core=is_core,
        tenuki_retained=tenuki_retained,
        tenuki_child_present=tenuki_child_present,
        local_children=bool(local_child_count),
    )
    children = local_children + ([tenuki_child] if tenuki_child is not None else [])
    return node_control, tenuki_present, int(tenuki_sf), local_rows, children


def build_family_bundle(*, artifacts_root: Path, family: str, board_size: int) -> bytes:
    family_s = str(family).strip().upper()
    data = json.loads(
        _artifact_path(artifacts_root=artifacts_root, family=family_s, board_size=board_size).read_text(encoding="utf-8")
    )
    if not isinstance(data, dict):
        raise ValueError(f"joseki artifact must be a JSON object: {data!r}")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or int(version) != BUNDLE_VERSION:
        raise ValueError(f"unsupported joseki artifact version: {version!r}")
    artifact_family = str(data.get("family") or "").strip().upper()
    if artifact_family != family_s:
        raise ValueError(f"joseki artifact family mismatch: expected {family_s!r}, got {artifact_family!r}")
    artifact_board_size = data.get("board_size")
    if not isinstance(artifact_board_size, int) or isinstance(artifact_board_size, bool):
        raise ValueError(f"bad joseki artifact board size: {artifact_board_size!r}")
    if int(artifact_board_size) != int(board_size):
        raise ValueError(
            f"joseki artifact board size mismatch: expected {int(board_size)}, got {int(artifact_board_size)}"
        )
    nodes = list(data.get("nodes") or [])
    line_to_node = {str(node.get("line") or ""): node for node in nodes}
    node_controls: list[int] = []
    tenuki_drops: list[int] = []
    compact_local_rows: list[tuple[int, int]] = []
    visited: set[str] = set()

    def append_preorder(node: dict[str, Any]) -> None:
        line = str(node.get("line") or "")
        if line in visited:
            raise ValueError(f"duplicate joseki node reference: {line!r}")
        visited.add(line)
        node_control, tenuki_present, tenuki_sf, local_rows, children = _compact_node(
            family=family_s,
            node=node,
            line_to_node=line_to_node,
        )
        is_root = not node_controls
        if bool(tenuki_present) != (not is_root):
            raise ValueError("joseki tenuki presence must be absent only at the root")
        tenuki_drop = 0 if is_root else 1000 - int(tenuki_sf)
        if tenuki_drop < 0 or tenuki_drop >= (1 << 10):
            raise ValueError(f"bad joseki tenuki drop: {tenuki_drop!r}")
        node_controls.append(node_control)
        tenuki_drops.append(tenuki_drop)
        compact_local_rows.extend(local_rows)
        for child in children:
            append_preorder(child)

    root = line_to_node.get("")
    if root is None:
        raise ValueError("missing joseki root node")
    append_preorder(root)
    if len(visited) != len(nodes):
        raise ValueError("joseki artifact contains unreachable nodes")

    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            BUNDLE_VERSION,
            _encode_family_code(family_s),
            int(artifact_board_size),
            len(node_controls),
            len(compact_local_rows),
        )
    )
    out.extend(bytes(node_controls))
    out.extend(bytes(drop & 0xFF for drop in tenuki_drops))
    out.extend(
        _pack_bitplanes(
            [drop >> 8 for drop in tenuki_drops],
            PACKED_TENUKI_DROP_HIGH_BITS,
        )
    )
    local_moves: list[int] = []
    local_drops: list[int] = []
    for move_code, stone_fraction in compact_local_rows:
        drop = 1000 - int(stone_fraction)
        if drop < 0 or drop > PACKED_LOCAL_DROP_MAX:
            raise ValueError(f"joseki local stone-fraction drop does not fit one byte: {drop!r}")
        local_moves.append(int(move_code))
        local_drops.append(drop)
    first_local_drops: list[int] = []
    sibling_local_drop_deltas: list[int] = []
    local_offset = 0
    for node_control in node_controls:
        local_count = int(node_control) & ((1 << PACKED_NODE_LOCAL_COUNT_BITS) - 1)
        node_drops = local_drops[local_offset:local_offset + local_count]
        if node_drops:
            first_local_drops.append(node_drops[0])
            sibling_local_drop_deltas.extend(
                (current - previous) & 0xFF
                for previous, current in zip(node_drops, node_drops[1:])
            )
        local_offset += local_count
    if local_offset != len(local_drops):
        raise ValueError("joseki node counts do not match local rows")
    out.extend(bytes(local_moves))
    out.extend(bytes(first_local_drops))
    out.extend(bytes(sibling_local_drop_deltas))
    return bytes(out)


def write_joseki_bundles(*, artifacts_root: Path, out_path: Path, board_size: int) -> Path:
    return write_hashed_bundle_manifest(
        out_path=out_path,
        bundles={
            family: BundlePayload(
                prefix=f"joseki_{str(family).strip().lower()}",
                payload=build_family_bundle(
                    artifacts_root=artifacts_root,
                    family=family,
                    board_size=board_size,
                ),
            )
            for family in ("A", "O")
        },
        stale_globs=["joseki_[ao].*.bin"],
        manifest_from_bundle_names=lambda bundle_names: {
            "version": 1,
            "bundles": bundle_names,
        },
    )


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
