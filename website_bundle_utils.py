from __future__ import annotations

import concurrent.futures
import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar


THOUSANDTH = Decimal("0.001")
PACKED_OPTIONAL_U10_NULL = 1023
_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class BundlePayload:
    prefix: str
    payload: bytes


@dataclass(frozen=True)
class PreorderGraphLayout:
    node_ids: list[int]
    redirect_flags: list[bool]
    redirect_targets: list[int]


def linearize_preorder_graph(
    *,
    root_node: int,
    child_targets_by_node: Mapping[int, Iterable[int | None]],
) -> PreorderGraphLayout:
    node_ids: list[int] = []
    redirect_flags: list[bool] = []
    redirect_node_ids: list[int] = []
    emitted_index_by_node: dict[int, int] = {}
    claimed_nodes = {int(root_node)}

    def append_preorder(node_id: int) -> None:
        if node_id in emitted_index_by_node:
            raise ValueError(f"duplicate graph node: {node_id!r}")
        targets = child_targets_by_node.get(node_id)
        if targets is None:
            raise ValueError(f"missing graph node: {node_id!r}")
        emitted_index_by_node[node_id] = len(node_ids)
        node_ids.append(node_id)
        tree_children: list[int] = []
        for target in targets:
            if target is None:
                redirect_flags.append(False)
                continue
            target_i = int(target)
            if target_i not in child_targets_by_node:
                raise ValueError(f"missing graph target: {target_i!r}")
            if target_i in claimed_nodes:
                redirect_flags.append(True)
                redirect_node_ids.append(target_i)
            else:
                redirect_flags.append(False)
                claimed_nodes.add(target_i)
                tree_children.append(target_i)
        for child in tree_children:
            append_preorder(child)

    append_preorder(int(root_node))
    if len(emitted_index_by_node) != len(child_targets_by_node):
        raise ValueError("graph contains unreachable nodes")
    return PreorderGraphLayout(
        node_ids=node_ids,
        redirect_flags=redirect_flags,
        redirect_targets=[emitted_index_by_node[node_id] for node_id in redirect_node_ids],
    )


def process_map(
    function: Callable[[_Task], _Result],
    tasks: Iterable[_Task],
    *,
    workers: int,
) -> Iterator[_Result]:
    task_list = list(tasks)
    requested_workers = int(workers)
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(len(task_list), requested_workers)
    if worker_count <= 1:
        yield from (function(task) for task in task_list)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        yield from pool.map(function, task_list)


def encode_thousandths(raw: Any, *, clamp: bool = False) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"bad numeric payload: {raw!r}")
    value = Decimal(str(raw)).quantize(THOUSANDTH, rounding=ROUND_HALF_EVEN)
    encoded = int(value * 1000)
    if clamp:
        return max(0, min(1000, encoded))
    return encoded


def encode_optional_thousandths(raw: Any) -> int | None:
    if raw is None:
        return None
    return encode_thousandths(raw)


def pack_optional_u10(value: int | None, *, null_value: int = PACKED_OPTIONAL_U10_NULL) -> int:
    if value is None:
        return int(null_value)
    if int(value) < 0 or int(value) >= int(null_value):
        raise ValueError(f"bad u10 payload: {value!r}")
    return int(value)


def serialize_manifest_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def hashed_bundle_filename(*, prefix: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{prefix}.{digest}.bin"


def write_hashed_bundle_manifest(
    *,
    out_path: Path,
    bundles: Mapping[str, BundlePayload],
    stale_globs: Iterable[str],
    manifest_from_bundle_names: Callable[[dict[str, str]], dict[str, Any]],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_names: dict[str, str] = {}
    written_bundles: set[str] = set()
    for key, bundle in bundles.items():
        bundle_name = hashed_bundle_filename(prefix=bundle.prefix, payload=bundle.payload)
        (out_path.parent / bundle_name).write_bytes(bundle.payload)
        bundle_names[str(key)] = bundle_name
        written_bundles.add(bundle_name)
    for stale_glob in stale_globs:
        for path in out_path.parent.glob(str(stale_glob)):
            if path.name not in written_bundles:
                path.unlink()
    out_path.write_text(
        serialize_manifest_json(manifest_from_bundle_names(bundle_names)),
        encoding="utf-8",
    )
    return out_path


def pack_little_endian_bits(rows: Iterable[tuple[int, int]]) -> bytes:
    out = bytearray()
    accumulator = 0
    pending_bits = 0
    for word, bits in rows:
        accumulator |= word << pending_bits
        pending_bits += bits
        while pending_bits >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            pending_bits -= 8
    if pending_bits:
        out.append(accumulator)
    return bytes(out)


def encode_uvarint(value: int) -> bytes:
    out = bytearray()
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"negative varuint payload: {value!r}")
    while value_i >= 0x80:
        out.append((value_i & 0x7F) | 0x80)
        value_i >>= 7
    out.append(value_i)
    return bytes(out)


def write_uvarint(out: bytearray, value: int) -> None:
    out.extend(encode_uvarint(value))


def cell_id_from_move(move: str, *, board_size: int) -> int:
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
