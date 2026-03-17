from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Any

import artifact_json as aj
from website_bundle_utils import (
    BundlePayload,
    cell_id_from_move,
    linearize_preorder_graph,
    pack_little_endian_bits,
    pack_optional_u10,
    process_map,
    write_hashed_bundle_manifest,
)


OPENINGS_ARTIFACT_DIR = "openings"
OPENINGS_OUT_NAME = "openings_current.json"
OPENING_BUNDLE_PREFIX = "opening_index"
BUNDLE_MAGIC = b"HOD"
CORE_IMPORTANCE_MIN_THOUSANDTHS = 910
PACKED_MOVE_ID_MAX = 1023
HEADER_STRUCT = struct.Struct("<3sHIII")
PACKED_NODE_COUNT_BITS = 6
PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
PACKED_NODE_HAS_CHILDREN_SHIFT = PACKED_NODE_IS_CORE_SHIFT + 1
PACKED_CANDIDATE_METRIC_BITS = 10
PACKED_CANDIDATE_DELTA_BITS = 8
PACKED_CANDIDATE_DELTA_ESCAPE = (1 << PACKED_CANDIDATE_DELTA_BITS) - 1
PACKED_CANDIDATE_DELTA_MAX_ABS = (PACKED_CANDIDATE_DELTA_ESCAPE - 1) // 2
PARALLEL_INPUT_BYTES_MIN = 1_000_000


def _packed_move_id_bits(board_size: int) -> int:
    size = int(board_size)
    if size <= 0:
        raise ValueError(f"bad board size: {board_size!r}")
    return ((size * size) - 1).bit_length()


def _red_winrate_from_mover_winrate(*, mover_winrate: int | None, parent_ply: int) -> int | None:
    if mover_winrate is None:
        return None
    value = int(mover_winrate)
    return value if int(parent_ply) % 2 == 0 else 1000 - value


def _delta_code(*, red_winrate: int | None, parent_edge_red_winrate: int | None) -> int | None:
    if red_winrate is None or parent_edge_red_winrate is None:
        return None
    delta = int(red_winrate) - int(parent_edge_red_winrate)
    if -PACKED_CANDIDATE_DELTA_MAX_ABS <= delta <= PACKED_CANDIDATE_DELTA_MAX_ABS:
        return delta + PACKED_CANDIDATE_DELTA_MAX_ABS
    return None


def _pack_candidate_bitstream(
    *,
    board_size: int,
    candidates: list[dict[str, Any]],
    node_candidate_counts: list[int],
) -> bytes:
    move_values: list[int] = []
    prior_values: list[int] = []
    delta_rows: list[tuple[int, int]] = []
    exceptions: list[int] = []
    move_id_bits = _packed_move_id_bits(board_size)
    for candidate in candidates:
        move_id = candidate["move_id"]
        prior = candidate["prior"]
        red_winrate = candidate["red_winrate"]
        parent_edge_red_winrate = candidate["parent_red_winrate"]
        node_candidate_count = candidate["node_candidate_count"]
        move_id_i = int(move_id)
        if move_id_i < 0 or move_id_i > PACKED_MOVE_ID_MAX:
            raise ValueError(f"bad packed move id payload: {move_id!r}")
        if move_id_i >= (1 << move_id_bits):
            raise ValueError(f"packed move id exceeds board-size capacity: {move_id!r}")
        move_values.append(move_id_i)
        prior_values.append(pack_optional_u10(prior))
        if int(node_candidate_count) == 1:
            if parent_edge_red_winrate is None and red_winrate is not None:
                raise ValueError("single-candidate root/opening node cannot encode a non-null winrate")
            if (
                parent_edge_red_winrate is not None
                and red_winrate is not None
                and int(red_winrate) != int(parent_edge_red_winrate)
            ):
                raise ValueError(
                    "single-candidate opening node winrate must match parent-edge winrate"
                )
            continue
        delta_code = _delta_code(
            red_winrate=red_winrate,
            parent_edge_red_winrate=parent_edge_red_winrate,
        )
        if delta_code is None:
            exceptions.append(pack_optional_u10(red_winrate))
            delta_code = PACKED_CANDIDATE_DELTA_ESCAPE
        delta_rows.append((int(delta_code), PACKED_CANDIDATE_DELTA_BITS))

    first_priors: list[int] = []
    prior_drops: list[int] = []
    candidate_offset = 0
    for candidate_count in node_candidate_counts:
        count = int(candidate_count)
        node_priors = prior_values[candidate_offset:candidate_offset + count]
        if node_priors:
            first_priors.append(node_priors[0])
            for previous, current in zip(node_priors, node_priors[1:]):
                drop = int(previous) - int(current)
                if drop < 0 or drop >= (1 << PACKED_CANDIDATE_METRIC_BITS):
                    raise ValueError("opening priors must be non-increasing within each node")
                prior_drops.append(drop)
        candidate_offset += count
    if candidate_offset != len(candidates):
        raise ValueError("opening node candidate counts do not match candidate rows")

    logical_priors = first_priors + prior_drops
    move_high_bits = max(0, move_id_bits - 8)
    move_high_stream = b""
    if move_high_bits:
        move_high_stream = pack_little_endian_bits(
            [(value >> 8, move_high_bits) for value in move_values],
        )
    return (
        bytes(value & 0xFF for value in move_values)
        + move_high_stream
        + pack_little_endian_bits(
            [(value & 0b11, 2) for value in logical_priors],
        )
        + bytes(value >> 2 for value in logical_priors)
        + pack_little_endian_bits(delta_rows)
        + pack_little_endian_bits(
            [(value, PACKED_CANDIDATE_METRIC_BITS) for value in exceptions],
        )
    )


def _pack_node(*, candidate_count: int, is_core: bool, has_children: bool) -> bytes:
    count = int(candidate_count)
    if count < 0 or count > PACKED_NODE_COUNT_MASK:
        raise ValueError(f"bad packed node candidate count payload: {candidate_count!r}")
    word = (
        count
        | ((1 if bool(is_core) else 0) << PACKED_NODE_IS_CORE_SHIFT)
        | ((1 if bool(has_children) else 0) << PACKED_NODE_HAS_CHILDREN_SHIFT)
    )
    return bytes([word])


def _opening_node_ply(node_raw: dict[str, Any], *, node_idx: int) -> int:
    node_keys = aj.OPENING_NODE_KEYS
    raw = node_raw.get(node_keys["ply"])
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"bad opening ply at node {node_idx}: {raw!r}")
    return raw


def _opening_candidate_move_id(candidate: dict[str, Any], *, board_size: int) -> int:
    candidate_keys = aj.OPENING_CANDIDATE_KEYS
    raw = candidate.get(candidate_keys["move"])
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"bad opening candidate move: {raw!r}")
    return cell_id_from_move(raw.strip().lower(), board_size=board_size)


def _opening_graph(
    *,
    nodes_raw: list[Any],
    board_size: int,
) -> tuple[int, dict[int, dict[str, Any]]]:
    node_keys = aj.OPENING_NODE_KEYS
    candidate_keys = aj.OPENING_CANDIDATE_KEYS
    graph: dict[int, dict[str, Any]] = {}
    for node_idx, node_raw in enumerate(nodes_raw):
        if not isinstance(node_raw, dict):
            raise ValueError(f"bad opening node payload: {node_raw!r}")
        node_candidates = node_raw.get(node_keys["candidates"])
        if not isinstance(node_candidates, list):
            raise ValueError(f"node missing candidates list: {node_raw!r}")
        ply = _opening_node_ply(node_raw, node_idx=node_idx)
        candidates: list[dict[str, Any]] = []
        for candidate_raw in node_candidates:
            if not isinstance(candidate_raw, dict):
                raise ValueError(f"bad candidate payload: {candidate_raw!r}")
            move_id = _opening_candidate_move_id(candidate_raw, board_size=board_size)
            child = candidate_raw.get(candidate_keys["child"])
            if isinstance(child, bool) or not isinstance(child, int) or child < 0:
                raise ValueError(f"bad opening child at node {node_idx}: {child!r}")
            target = child if child < len(nodes_raw) else None
            if target is not None and target <= node_idx:
                raise ValueError(f"opening child does not follow its parent at node {node_idx}")
            candidates.append(
                {
                    "move_id": move_id,
                    "prior": aj.optional_thousandths(candidate_raw.get(candidate_keys["prior"])),
                    "red_winrate": _red_winrate_from_mover_winrate(
                        mover_winrate=aj.optional_thousandths(
                            candidate_raw.get(candidate_keys["tree_mover_winrate"])
                        ),
                        parent_ply=ply,
                    ),
                    "target": target,
                }
            )
        target_count = sum(candidate["target"] is not None for candidate in candidates)
        if target_count not in {0, len(candidates)}:
            raise ValueError("opening node has mixed child presence")
        graph[node_idx] = {
            "ply": ply,
            "importance": aj.optional_thousandths(node_raw.get(node_keys["importance"])),
            "tree_red_winrate": aj.optional_thousandths(
                node_raw.get(node_keys["tree_red_winrate"])
            ),
            "candidates": candidates,
        }
    return 0, graph


def _linearize_opening_graph(
    *,
    root_node: int,
    graph: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[bool], list[int]]:
    nodes: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    layout = linearize_preorder_graph(
        root_node=root_node,
        child_targets_by_node={
            node_id: [candidate["target"] for candidate in node["candidates"]]
            for node_id, node in graph.items()
        },
    )
    for node_id in layout.node_ids:
        node = graph[node_id]
        importance = node["importance"]
        if not isinstance(importance, int):
            raise ValueError(f"bad importance payload: {importance!r}")
        node_candidates = node["candidates"]
        has_children = bool(node_candidates and node_candidates[0]["target"] is not None)
        nodes.append(
            {
                "candidate_count": len(node_candidates),
                "is_core": importance >= CORE_IMPORTANCE_MIN_THOUSANDTHS,
                "has_children": has_children,
            }
        )
        for candidate in node_candidates:
            candidates.append(
                {
                    "move_id": candidate["move_id"],
                    "prior": candidate["prior"],
                    "red_winrate": candidate["red_winrate"],
                    "parent_red_winrate": (
                        None if node_id == root_node else node["tree_red_winrate"]
                    ),
                    "node_candidate_count": len(node_candidates),
                }
            )
    return nodes, candidates, layout.redirect_flags, layout.redirect_targets


def build_opening_bundle(*, artifacts_root: Path, board_size: int) -> bytes:
    artifact_path = Path(artifacts_root) / f"openings-s{int(board_size)}.json"
    data = aj.load(artifact_path)
    if not isinstance(data, dict):
        raise ValueError(f"opening artifact must be an object: {artifact_path}")
    root_keys = aj.OPENING_ROOT_KEYS
    nodes_raw = data.get(root_keys["nodes"])
    if not isinstance(nodes_raw, list):
        raise ValueError(f"opening artifact missing nodes list: {artifact_path}")
    if not nodes_raw:
        raise ValueError(f"opening artifact missing root node: {artifact_path}")
    artifact_board_size_raw = data.get(root_keys["board_size"])
    if not isinstance(artifact_board_size_raw, int) or isinstance(artifact_board_size_raw, bool):
        raise ValueError(f"bad opening board size for {artifact_path}: {artifact_board_size_raw!r}")
    artifact_board_size = int(artifact_board_size_raw)
    if artifact_board_size != int(board_size):
        raise ValueError(
            f"opening artifact board size mismatch for {artifact_path}: "
            f"expected {int(board_size)}, got {artifact_board_size}"
        )
    if data.get(root_keys["root"]) != 0:
        raise ValueError(f"opening artifact root must be 0: {artifact_path}")
    root_node, graph = _opening_graph(
        nodes_raw=nodes_raw,
        board_size=artifact_board_size,
    )
    nodes, candidates, redirect_flags, redirect_targets = _linearize_opening_graph(
        root_node=root_node,
        graph=graph,
    )
    node_id_bits = max(1, (len(nodes) - 1).bit_length())

    out = bytearray()
    out.extend(
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC,
            artifact_board_size,
            len(nodes),
            len(candidates),
            len(redirect_targets),
        )
    )
    for node in nodes:
        out.extend(_pack_node(**node))
    out.extend(
        _pack_candidate_bitstream(
            board_size=artifact_board_size,
            candidates=candidates,
            node_candidate_counts=[node["candidate_count"] for node in nodes],
        )
    )
    out.extend(
        pack_little_endian_bits(
            [(1 if flag else 0, 1) for flag in redirect_flags],
        )
    )
    out.extend(
        pack_little_endian_bits(
            [(target, node_id_bits) for target in redirect_targets],
        )
    )
    return bytes(out)


def _build_opening_bundle_task(task: tuple[Path, int]) -> tuple[int, bytes]:
    artifacts_root, board_size = task
    return board_size, build_opening_bundle(
        artifacts_root=artifacts_root,
        board_size=board_size,
    )


def write_opening_bundles(
    *,
    artifacts_root: Path,
    out_path: Path,
    board_sizes: list[int],
    workers: int = os.cpu_count() or 1,
) -> Path:
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    tasks = [(Path(artifacts_root), int(board_size)) for board_size in board_sizes]
    input_bytes = sum(
        (Path(artifacts_root) / f"openings-s{board_size}.json").stat().st_size
        for _artifacts_root, board_size in tasks
    )
    if len(tasks) > 1 and input_bytes >= PARALLEL_INPUT_BYTES_MIN:
        built = list(process_map(_build_opening_bundle_task, tasks, workers=workers))
    else:
        built = [_build_opening_bundle_task(task) for task in tasks]
    bundle_entries = {str(board_size): payload for board_size, payload in built}
    return write_hashed_bundle_manifest(
        out_path=out_path,
        bundles={
            key: BundlePayload(prefix=OPENING_BUNDLE_PREFIX, payload=payload)
            for key, payload in bundle_entries.items()
        },
        stale_globs=[f"{OPENING_BUNDLE_PREFIX}.*.bin"],
        manifest_from_bundle_names=lambda bundle_names: {"bundles": bundle_names},
    )


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
    ap = argparse.ArgumentParser(description="Build compact binary bundles for opening websites")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    ap.add_argument("--board-sizes", default="11,12,13,14,17")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    if int(args.workers) < 1:
        ap.error("--workers must be positive")
    return args


def main() -> int:
    args = _parse_args()
    write_opening_bundles(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
        board_sizes=_parse_board_sizes(args.board_sizes),
        workers=int(args.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
