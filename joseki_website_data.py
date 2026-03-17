from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Any

import artifact_json as aj
from website_bundle_utils import (
    BundlePayload,
    linearize_preorder_graph,
    pack_little_endian_bits,
    process_map,
    write_hashed_bundle_manifest,
)

BUNDLE_MAGIC = b"HJB"
FAMILY_CODE_BY_NAME = {
    "A": 1,
    "O": 2,
}
CORE_IMPORTANCE_MIN_THOUSANDTHS = 825
HEADER_STRUCT = struct.Struct("<3sBBIII")
PACKED_NODE_LOCAL_COUNT_BITS = 4
PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_LOCAL_COUNT_BITS
PACKED_NODE_TENUKI_RETAINED_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
PACKED_NODE_TENUKI_CHILD_SHIFT = PACKED_NODE_TENUKI_RETAINED_SHIFT + 1
PACKED_NODE_LOCAL_CHILDREN_SHIFT = PACKED_NODE_TENUKI_CHILD_SHIFT + 1
PACKED_TENUKI_DROP_HIGH_BITS = 2
PARALLEL_INPUT_BYTES_MIN = 1_000_000
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
    node_idx: int,
    node_count: int,
    node: dict[str, Any],
) -> tuple[int, bool, int, list[tuple[int, int]], list[int | None]]:
    node_keys = aj.JOSEKI_NODE_KEYS
    candidate_keys = aj.JOSEKI_CANDIDATE_KEYS
    local_rows: list[tuple[int, int]] = []
    local_child_bits: list[bool] = []
    local_children: list[int | None] = []
    tenuki_child: int | None = None
    tenuki_present = False
    tenuki_sf = 0
    tenuki_retained = False
    tenuki_child_present = False

    for row in list(node.get(node_keys["candidates"]) or []):
        kind = str(row.get(candidate_keys["kind"]) or "").strip()
        stone_fraction = aj.optional_thousandths(
            row.get(candidate_keys["stone_fraction"])
        )
        if stone_fraction is None:
            continue
        stone_fraction_i = stone_fraction
        child_raw = row.get(candidate_keys["child"])
        if child_raw is None:
            child = None
        elif isinstance(child_raw, bool) or not isinstance(child_raw, int) or child_raw < 0:
            raise ValueError(f"bad joseki child at node {node_idx}: {child_raw!r}")
        else:
            child = int(child_raw)
            if child < node_count and child <= node_idx:
                raise ValueError(f"joseki child does not follow its parent at node {node_idx}")
        if kind == "local":
            if child_raw is None:
                continue
            local = _normalize_local(row.get(candidate_keys["local"]))
            local_rows.append((_encode_local_move_code(local), stone_fraction_i))
            local_child_bits.append(child < node_count)
            local_children.append(child if child < node_count else None)
        elif kind == "tenuki":
            tenuki_present = True
            tenuki_sf = stone_fraction_i
            tenuki_retained = child_raw is not None
            tenuki_child_present = bool(tenuki_retained and child < node_count)
            if tenuki_retained:
                tenuki_child = child if child < node_count else None

    importance = aj.thousandths(node.get(node_keys["importance"]))
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
    children = local_children + ([tenuki_child] if tenuki_retained else [])
    return node_control, tenuki_present, int(tenuki_sf), local_rows, children


def build_family_bundle(*, artifacts_root: Path, family: str, board_size: int) -> bytes:
    family_s = str(family).strip().upper()
    data = aj.load(
        _artifact_path(artifacts_root=artifacts_root, family=family_s, board_size=board_size)
    )
    if not isinstance(data, dict):
        raise ValueError("joseki artifact must be an object")
    root_keys = aj.JOSEKI_ROOT_KEYS
    node_keys = aj.JOSEKI_NODE_KEYS
    artifact_family = str(data.get(root_keys["family"]) or "").strip().upper()
    if artifact_family != family_s:
        raise ValueError(f"joseki artifact family mismatch: expected {family_s!r}, got {artifact_family!r}")
    artifact_board_size = data.get(root_keys["board_size"])
    if not isinstance(artifact_board_size, int) or isinstance(artifact_board_size, bool):
        raise ValueError(f"bad joseki artifact board size: {artifact_board_size!r}")
    if int(artifact_board_size) != int(board_size):
        raise ValueError(
            f"joseki artifact board size mismatch: expected {int(board_size)}, got {int(artifact_board_size)}"
        )
    nodes_raw = data.get(root_keys["nodes"])
    if not isinstance(nodes_raw, list) or not all(isinstance(node, dict) for node in nodes_raw):
        raise ValueError("joseki artifact requires object nodes")
    nodes = list(nodes_raw)
    if not nodes or nodes[0].get(node_keys["line"]) != "":
        raise ValueError("missing joseki root node")
    compact_by_node: dict[int, tuple[int, bool, int, list[tuple[int, int]], list[int | None]]] = {
        node_idx: _compact_node(
            node_idx=node_idx,
            node_count=len(nodes),
            node=node,
        )
        for node_idx, node in enumerate(nodes)
    }
    layout = linearize_preorder_graph(
        root_node=0,
        child_targets_by_node={
            node_idx: compact[4]
            for node_idx, compact in compact_by_node.items()
        },
    )
    node_controls: list[int] = []
    tenuki_drops: list[int] = []
    compact_local_rows: list[tuple[int, int]] = []
    for node_idx in layout.node_ids:
        node_control, tenuki_present, tenuki_sf, local_rows, _children = compact_by_node[node_idx]
        is_root = not node_controls
        if bool(tenuki_present) != (not is_root):
            raise ValueError("joseki tenuki presence must be absent only at the root")
        tenuki_drop = 0 if is_root else 1000 - int(tenuki_sf)
        if tenuki_drop < 0 or tenuki_drop >= (1 << 10):
            raise ValueError(f"bad joseki tenuki drop: {tenuki_drop!r}")
        node_controls.append(node_control)
        tenuki_drops.append(tenuki_drop)
        compact_local_rows.extend(local_rows)

    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            _encode_family_code(family_s),
            int(artifact_board_size),
            len(node_controls),
            len(compact_local_rows),
            len(layout.redirect_targets),
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
    out.extend(
        pack_little_endian_bits(
            [(1 if flag else 0, 1) for flag in layout.redirect_flags],
        )
    )
    node_id_bits = max(1, (len(node_controls) - 1).bit_length())
    out.extend(
        pack_little_endian_bits(
            [(target, node_id_bits) for target in layout.redirect_targets],
        )
    )
    return bytes(out)


def _build_family_bundle_task(task: tuple[Path, str, int]) -> tuple[str, bytes]:
    artifacts_root, family, board_size = task
    return family, build_family_bundle(
        artifacts_root=artifacts_root,
        family=family,
        board_size=board_size,
    )


def write_joseki_bundles(
    *,
    artifacts_root: Path,
    out_path: Path,
    board_size: int,
    workers: int = os.cpu_count() or 1,
) -> Path:
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    tasks = [
        (Path(artifacts_root), family, int(board_size))
        for family in ("A", "O")
    ]
    input_bytes = sum(
        _artifact_path(
            artifacts_root=task_artifacts_root,
            family=family,
            board_size=task_board_size,
        ).stat().st_size
        for task_artifacts_root, family, task_board_size in tasks
    )
    if len(tasks) > 1 and input_bytes >= PARALLEL_INPUT_BYTES_MIN:
        built = list(process_map(_build_family_bundle_task, tasks, workers=workers))
    else:
        built = [_build_family_bundle_task(task) for task in tasks]
    return write_hashed_bundle_manifest(
        out_path=out_path,
        bundles={
            family: BundlePayload(
                prefix=f"joseki_{str(family).strip().lower()}",
                payload=payload,
            )
            for family, payload in built
        },
        stale_globs=["joseki_[ao].*.bin"],
        manifest_from_bundle_names=lambda bundle_names: {"bundles": bundle_names},
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact binary bundles for the joseki website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    ap.add_argument("--board-size", default="19")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    if int(args.workers) < 1:
        ap.error("--workers must be positive")
    return args


def main() -> int:
    args = _parse_args()
    write_joseki_bundles(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
        board_size=int(args.board_size),
        workers=int(args.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
