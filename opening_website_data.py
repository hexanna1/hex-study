from __future__ import annotations

import argparse
import os
import struct
from array import array
from pathlib import Path
from typing import Any

import artifact_json as aj
from website_bundle_utils import (
    PACKED_OPTIONAL_U10_NULL,
    cell_id_from_move,
    pack_little_endian_bits,
    pack_optional_u10,
    process_map,
    write_gzip_bundle,
    write_uvarint,
)


OPENINGS_ARTIFACT_DIR = "openings"
BUNDLE_MAGIC = b"HOD"
HEADER_STRUCT = struct.Struct("<3sHIIIIIIII")
CORE_IMPORTANCE_MIN_THOUSANDTHS = 910
PACKED_NODE_COUNT_BITS = 2
PACKED_NODE_COUNT_ESCAPE = (1 << PACKED_NODE_COUNT_BITS) - 1
PACKED_NODE_COUNT_MAX = 63
PACKED_MOVE_ID_MAX = 1023
PACKED_METRIC_BITS = 10
LEAD_DELTA_BITS = 4
LEAD_DELTA_BIAS = 7
OTHER_DELTA_BIAS = 127
MOVE_MODEL_BITS = 10
MOVE_MODEL_TOTAL = 1 << MOVE_MODEL_BITS
MOVE_STATE_FLOOR = 1 << 23
PARALLEL_INPUT_BYTES_MIN = 1_000_000


def _red_winrate(*, mover_winrate: int | None, parent_ply: int) -> int | None:
    if mover_winrate is None:
        return None
    return 1000 - mover_winrate if parent_ply & 1 else mover_winrate


def _lead_prediction(
    parent: int | None, priors: list[int], red_winrates: list[int | None]
) -> int | None:
    lead_prior = priors[0]
    if parent is None or lead_prior in (0, PACKED_OPTIONAL_U10_NULL):
        return parent
    # Solve the weighted mean for the leading move, then encode the exact
    # residual. Integer-thousandth inputs often predict it within a point.
    correction = 0
    for prior, winrate in zip(priors[1:], red_winrates[1:]):
        if prior == PACKED_OPTIONAL_U10_NULL:
            continue
        if winrate is None:
            return parent
        correction += prior * (parent - winrate)
    return parent + (correction + lead_prior // 2) // lead_prior


def _append_residual(
    codes: bytearray,
    exceptions: list[int],
    *,
    value: int | None,
    prediction: int | None,
    bias: int,
) -> None:
    if value is not None and prediction is not None:
        delta = value - prediction
        if -bias <= delta <= bias:
            codes.append(delta + bias)
            return
    codes.append(2 * bias + 1)
    exceptions.append(pack_optional_u10(value))


def _candidate_move_id(
    raw: Any, *, board_size: int, move_ids: dict[str, int], move_id_bits: int
) -> int:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"bad opening candidate move: {raw!r}")
    move_id = move_ids.get(raw)
    if move_id is None:
        move_id = cell_id_from_move(raw, board_size=board_size)
        if move_id > PACKED_MOVE_ID_MAX or move_id >= 1 << move_id_bits:
            raise ValueError(f"bad packed opening move id: {raw!r}")
        move_ids[raw] = move_id
    return move_id


def _encode_moves(codes: array, cell_count: int) -> bytes:
    # Each code pairs a move with the stone that first reached its position.
    # The root uses its own context, one past the last cell.
    table_size = (cell_count + 1) * cell_count
    frequencies = [0] * table_size
    for code in codes:
        frequencies[code] += 1

    starts = [0] * table_size
    for base in range(0, table_size, cell_count):
        counts = frequencies[base:base + cell_count]
        total = sum(counts)
        if not total:
            continue
        # Reserve one slot per observed move and apportion the rest by largest remainder.
        budget = MOVE_MODEL_TOTAL - sum(count > 0 for count in counts)
        remainders = []
        assigned = 0
        for cell, count in enumerate(counts):
            frequency, remainder = divmod(count * budget, total)
            frequency += count > 0
            frequencies[base + cell] = frequency
            if remainder:
                remainders.append((remainder, cell))
            assigned += frequency
        remainders.sort(key=lambda pair: -pair[0])
        for _, cell in remainders[:MOVE_MODEL_TOTAL - assigned]:
            frequencies[base + cell] += 1
        start = 0
        for code in range(base, base + cell_count):
            starts[code] = start
            start += frequencies[code]

    # rANS emits bytes in reverse order. The browser reconstructs the moves as
    # it walks the breadth-first graph and discovers each context.
    state = MOVE_STATE_FLOOR
    renormalize = (MOVE_STATE_FLOOR >> MOVE_MODEL_BITS) << 8
    stream = bytearray()
    for code in reversed(codes):
        frequency = frequencies[code]
        while state >= renormalize * frequency:
            stream.append(state & 255)
            state >>= 8
        quotient, remainder = divmod(state, frequency)
        state = (quotient << MOVE_MODEL_BITS) + remainder + starts[code]
    stream.reverse()
    present = bytearray((table_size + 7) // 8)
    common = bytearray()
    rare = bytearray()
    for code, frequency in enumerate(frequencies):
        if not frequency:
            continue
        present[code >> 3] |= 1 << (code & 7)
        value = frequency - 1
        common.append(min(255, value))
        if value >= 255:
            rare.extend(struct.pack("<H", value))
    section_size = 8 + len(present) + len(common) + len(rare) + len(stream)
    return b"".join((
        struct.pack("<II", section_size, len(common)),
        present,
        common,
        rare,
        struct.pack("<I", state),
        stream,
    ))


def _build_opening_payload(nodes_raw: list[Any], *, board_size: int) -> bytes:
    node_keys = aj.OPENING_NODE_KEYS
    candidate_keys = aj.OPENING_CANDIDATE_KEYS
    cell_count = board_size * board_size
    move_id_bits = (cell_count - 1).bit_length()
    transposed_moves = [
        (cell % board_size) * board_size + cell // board_size for cell in range(cell_count)
    ]
    move_ids: dict[str, int] = {}
    node_order = [0]
    node_index_by_id = [-1] * len(nodes_raw)
    node_index_by_id[0] = 0
    node_counts = bytearray((len(nodes_raw) + 3) // 4)
    large_counts = bytearray()
    flagged_nodes: list[tuple[int, int]] = []
    last_parent = -1
    incoming_moves = array("H", [cell_count])
    move_codes = array("I")
    first_priors: list[int] = []
    other_priors: list[int] = []
    lead_deltas = bytearray()
    other_deltas = bytearray()
    lead_exceptions: list[int] = []
    other_exceptions: list[int] = []
    redirect_positions = bytearray()
    redirect_distances = bytearray()
    redirect_count = 0
    edge_index = 0
    previous_redirect = -1

    # Assign IDs at discovery time. Each first visit becomes the next child ID,
    # so only edges to a child that already has an ID need an explicit target.
    for node_index, node_id in enumerate(node_order):
        node = nodes_raw[node_id]
        if not isinstance(node, dict):
            raise ValueError(f"bad opening node payload: {node!r}")
        candidates = node.get(node_keys["candidates"])
        if not isinstance(candidates, list):
            raise ValueError(f"node missing candidates list: {node!r}")
        count = len(candidates)
        if count > PACKED_NODE_COUNT_MAX:
            raise ValueError(f"bad packed node candidate count payload: {count}")
        node_counts[node_index // 4] |= min(count, PACKED_NODE_COUNT_ESCAPE) << ((node_index & 3) * 2)
        if count >= PACKED_NODE_COUNT_ESCAPE:
            large_counts.append(count)
        ply = node.get(node_keys["ply"])
        if isinstance(ply, bool) or not isinstance(ply, int) or ply < 0:
            raise ValueError(f"bad opening ply at node {node_id}: {ply!r}")
        importance = aj.thousandths(node.get(node_keys["importance"]))
        flags = int(importance >= CORE_IMPORTANCE_MIN_THOUSANDTHS)
        parent_winrate = None if node_id == 0 else aj.optional_thousandths(node.get(node_keys["tree_red_winrate"]))
        move_context = incoming_moves[node_index] * cell_count
        materialized_count = 0
        priors: list[int] = []
        winrates: list[int | None] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"bad candidate payload: {candidate!r}")
            move = _candidate_move_id(
                candidate.get(candidate_keys["move"]), board_size=board_size, move_ids=move_ids, move_id_bits=move_id_bits,
            )
            oriented_move = transposed_moves[move] if ply & 1 else move
            move_codes.append(move_context + oriented_move)
            prior = pack_optional_u10(aj.optional_thousandths(candidate.get(candidate_keys["prior"])))
            if priors and prior > priors[-1]:
                raise ValueError("opening priors must be non-increasing within each node")
            priors.append(prior)
            winrates.append(_red_winrate(
                mover_winrate=aj.optional_thousandths(candidate.get(candidate_keys["tree_mover_winrate"])),
                parent_ply=ply,
            ))
            child = candidate.get(candidate_keys["child"])
            if isinstance(child, bool) or not isinstance(child, int) or child < 0:
                raise ValueError(f"bad opening child at node {node_id}: {child!r}")
            if child >= len(nodes_raw):
                continue
            if child <= node_id:
                raise ValueError(f"opening child does not follow its parent at node {node_id}")
            materialized_count += 1
            child_index = node_index_by_id[child]
            if child_index < 0:
                node_index_by_id[child] = len(node_order)
                node_order.append(child)
                incoming_moves.append(transposed_moves[oriented_move])
            else:
                write_uvarint(redirect_positions, edge_index - previous_redirect - 1)
                write_uvarint(redirect_distances, len(node_order) - 1 - child_index)
                previous_redirect = edge_index
                redirect_count += 1
            edge_index += 1
        if materialized_count:
            if materialized_count != count:
                raise ValueError("opening node has mixed child presence")
            last_parent = node_index
        elif count:
            flags |= 2
        if flags:
            flagged_nodes.append((node_index, flags))
        if not count:
            continue
        first_priors.append(priors[0])
        other_priors.extend(priors[1:])
        if count == 1:
            if winrates[0] is not None and winrates[0] != parent_winrate:
                raise ValueError("single-candidate opening node winrate must match parent-edge winrate")
            continue
        _append_residual(
            lead_deltas, lead_exceptions, value=winrates[0],
            prediction=_lead_prediction(parent_winrate, priors, winrates), bias=LEAD_DELTA_BIAS,
        )
        for winrate in winrates[1:]:
            _append_residual(other_deltas, other_exceptions, value=winrate, prediction=parent_winrate, bias=OTHER_DELTA_BIAS)
    if len(node_order) != len(nodes_raw):
        raise ValueError("graph contains unreachable nodes")

    child_limit = last_parent + 1
    shape_flags = bytearray()
    shape_flag_count = 0
    previous_flag = -1
    for node_index, flags in flagged_nodes:
        if node_index >= child_limit:
            flags &= 1
        if flags:
            write_uvarint(shape_flags, ((node_index - previous_flag - 1) << 2) | flags)
            previous_flag = node_index
            shape_flag_count += 1

    prior_values = first_priors + other_priors
    return b"".join((
        HEADER_STRUCT.pack(
            BUNDLE_MAGIC, board_size, len(node_order), len(move_codes), len(first_priors), len(lead_deltas),
            redirect_count, len(large_counts), child_limit, shape_flag_count,
        ),
        node_counts,
        large_counts,
        shape_flags,
        _encode_moves(move_codes, cell_count),
        pack_little_endian_bits((prior & 3, 2) for prior in prior_values),
        bytes(prior >> 2 for prior in prior_values),
        pack_little_endian_bits((delta, LEAD_DELTA_BITS) for delta in lead_deltas),
        pack_little_endian_bits((value, PACKED_METRIC_BITS) for value in lead_exceptions),
        other_deltas,
        pack_little_endian_bits((value, PACKED_METRIC_BITS) for value in other_exceptions),
        redirect_positions,
        redirect_distances,
    ))


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
    artifact_board_size = data.get(root_keys["board_size"])
    if not isinstance(artifact_board_size, int) or isinstance(artifact_board_size, bool) or artifact_board_size <= 0:
        raise ValueError(f"bad opening board size for {artifact_path}: {artifact_board_size!r}")
    if artifact_board_size != int(board_size):
        raise ValueError(
            f"opening artifact board size mismatch for {artifact_path}: "
            f"expected {int(board_size)}, got {artifact_board_size}"
        )
    if data.get(root_keys["root"]) != 0:
        raise ValueError(f"opening artifact root must be 0: {artifact_path}")
    return _build_opening_payload(nodes_raw, board_size=artifact_board_size)


def _build_opening_bundle_task(task: tuple[Path, int]) -> tuple[int, bytes]:
    artifacts_root, board_size = task
    return board_size, build_opening_bundle(
        artifacts_root=artifacts_root,
        board_size=board_size,
    )


def write_opening_bundles(
    *,
    artifacts_root: Path,
    out_dir: Path,
    board_sizes: list[int],
    workers: int = os.cpu_count() or 1,
) -> None:
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
    for board_size, payload in built:
        write_gzip_bundle(out_dir / f"opening_index_{board_size}.bin.gz", payload)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / OPENINGS_ARTIFACT_DIR


def _default_out_dir() -> Path:
    return _repo_root() / "docs" / "data"


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
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
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
        out_dir=args.out_dir,
        board_sizes=_parse_board_sizes(args.board_sizes),
        workers=int(args.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
