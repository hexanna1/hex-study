from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Any

import artifact_json as aj
from website_bundle_utils import (
    pack_little_endian_bits,
    process_map,
    write_gzip_bundle,
    write_uvarint,
)

BUNDLE_MAGIC = b"HJB"
FAMILY_CODE_BY_NAME = {
    "A": 1,
    "O": 2,
}
CORE_IMPORTANCE_MIN_THOUSANDTHS = 825
HEADER_STRUCT = struct.Struct("<3sBBIIIIII")
PACKED_NODE_LOCAL_COUNT_BITS = 4
PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_LOCAL_COUNT_BITS
PACKED_NODE_TENUKI_RETAINED_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
PACKED_NODE_TENUKI_CHILD_SHIFT = PACKED_NODE_TENUKI_RETAINED_SHIFT + 1
PACKED_NODE_LOCAL_CHILDREN_SHIFT = PACKED_NODE_TENUKI_CHILD_SHIFT + 1
PACKED_TENUKI_DROP_HIGH_BITS = 2
PARALLEL_INPUT_BYTES_MIN = 1_000_000
PACKED_LOCAL_DROP_MAX = 0xFF
CompactNode = tuple[int, bool, int, list[tuple[int, int]], list[int | None]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / "joseki"


def _default_out_dir() -> Path:
    return _repo_root() / "docs" / "data"


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
        (
            ((int(value) >> bit) & 1, 1)
            for bit in range(int(bits) - 1, -1, -1)
            for value in values
        ),
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
) -> CompactNode:
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
    compact_by_node = _intern_compact_nodes({
        node_idx: _compact_node(
            node_idx=node_idx,
            node_count=len(nodes),
            node=node,
        )
        for node_idx, node in enumerate(nodes)
    })
    order, node_indices = _joseki_postorder(compact_by_node)
    node_controls = bytearray()
    tenuki_drops: tuple[list[int], list[int]] = ([], [])
    local_moves = bytearray()
    first_local_presence: list[bool] = []
    first_local_drops = bytearray()
    sibling_local_drop_deltas = bytearray()
    redirect_flags: list[bool] = []
    redirects = bytearray()
    continuations: list[int] = []
    for node_idx, tree_edges in order:
        node_control, tenuki_present, tenuki_sf, local_rows, children = compact_by_node[node_idx]
        is_root = node_idx == 0
        if bool(tenuki_present) != (not is_root):
            raise ValueError("joseki tenuki presence must be absent only at the root")
        tenuki_drop = 0 if is_root else 1000 - tenuki_sf
        if tenuki_drop < 0 or tenuki_drop >= (1 << 10):
            raise ValueError(f"bad joseki tenuki drop: {tenuki_drop!r}")
        node_controls.append(node_control)
        first_local_drop = 1000 - local_rows[0][1] if local_rows else 0
        has_first_local_drop = first_local_drop != 0
        first_local_presence.append(has_first_local_drop)
        # Both kinds of move share the same score baseline. Separating nodes
        # by their leading local score puts tenuki drops with similar ranges together.
        tenuki_drops[has_first_local_drop].append(tenuki_drop)
        previous_drop = 0
        for offset, (move_code, stone_fraction) in enumerate(local_rows):
            drop = 1000 - stone_fraction
            if drop < 0 or drop > PACKED_LOCAL_DROP_MAX:
                raise ValueError(f"joseki local stone-fraction drop does not fit one byte: {drop!r}")
            local_moves.append(move_code)
            if offset == 0:
                if has_first_local_drop:
                    first_local_drops.append(drop)
            else:
                sibling_local_drop_deltas.append((drop - previous_drop) & 0xFF)
            previous_drop = drop
        continuation_count = len(children)
        for offset, child in enumerate(children):
            if child is None:
                continue
            target = node_indices[child]
            redirect = not (tree_edges & (1 << offset))
            redirect_flags.append(redirect)
            if redirect:
                write_uvarint(redirects, target)
            continuation_count += continuations[target]
        continuations.append(continuation_count)

    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            _encode_family_code(family_s),
            int(artifact_board_size),
            len(node_controls),
            len(local_moves),
            len(redirect_flags),
            continuations[-1],
            len(first_local_drops),
            len(sibling_local_drop_deltas),
        )
    )
    out.extend(node_controls)
    out.extend(local_moves)
    out.extend(pack_little_endian_bits((flag, 1) for flag in first_local_presence))
    out.extend(first_local_drops)
    out.extend(sibling_local_drop_deltas)
    for drops in tenuki_drops:
        out.extend(bytes(drop & 0xFF for drop in drops))
        out.extend(
            _pack_bitplanes(
                [drop >> 8 for drop in drops],
                PACKED_TENUKI_DROP_HIGH_BITS,
            )
        )
    out.extend(pack_little_endian_bits((flag, 1) for flag in redirect_flags))
    out.extend(redirects)
    return bytes(out)


def _intern_compact_nodes(nodes: dict[int, CompactNode]) -> dict[int, CompactNode]:
    representatives: dict[tuple, int] = {}
    canonical: dict[int, int] = {}
    interned = {}
    # Children follow their parents in the input, so intern them first.
    for node in reversed(nodes):
        control, present, fraction, local_rows, children = nodes[node]
        targets = tuple(None if child is None else canonical[child] for child in children)
        signature = (control, present, fraction, tuple(local_rows), targets)
        representative = representatives.setdefault(signature, node)
        canonical[node] = representative
        if representative == node:
            interned[node] = (control, present, fraction, local_rows, list(targets))
    return interned


def _joseki_postorder(nodes: dict[int, CompactNode]) -> tuple[list[tuple[int, int]], dict[int, int]]:
    order: list[tuple[int, int]] = []
    indices: dict[int, int] = {}

    def visit(node_id: int) -> None:
        children = nodes[node_id][4]
        tree_edges = 0
        # Reverse candidate order so the first tree child sits just before the parent.
        for offset in range(len(children) - 1, -1, -1):
            child = children[offset]
            if child is not None and child not in indices:
                visit(child)
                tree_edges |= 1 << offset
        indices[node_id] = len(order)
        order.append((node_id, tree_edges))

    visit(0)
    if len(order) != len(nodes):
        raise ValueError("joseki graph contains unreachable nodes")
    return order, indices


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
    out_dir: Path,
    board_size: int,
    workers: int = os.cpu_count() or 1,
) -> None:
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
    for family, payload in built:
        write_gzip_bundle(out_dir / f"joseki_{family.lower()}.bin.gz", payload)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact binary bundles for the joseki website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
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
        out_dir=args.out_dir,
        board_size=int(args.board_size),
        workers=int(args.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
