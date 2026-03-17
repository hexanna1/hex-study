from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, TypeVar


PACKED_OPTIONAL_U10_NULL = 1023
_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class BundlePayload:
    prefix: str
    payload: bytes


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
    return f"{prefix}.{digest}.bin.gz"


def write_gzip_bundle(path: Path, payload: bytes) -> None:
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    libdeflate = shutil.which("libdeflate-gzip")
    if libdeflate:
        candidate = subprocess.run(
            [libdeflate, "-12", "-c"], input=payload, stdout=subprocess.PIPE, check=True,
        ).stdout
        compressed = min(compressed, candidate, key=len)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    pending = Path(file.name)
    try:
        with file:
            file.write(compressed)
        pending.chmod(0o644)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


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
        bundle_path = out_path.parent / bundle_name
        tmp_bundle = bundle_path.with_name(bundle_path.name + ".tmp")
        tmp_bundle.write_bytes(gzip.compress(bundle.payload, compresslevel=9, mtime=0))
        tmp_bundle.replace(bundle_path)
        bundle_names[str(key)] = bundle_name
        written_bundles.add(bundle_name)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        serialize_manifest_json(manifest_from_bundle_names(bundle_names)),
        encoding="utf-8",
    )
    tmp.replace(out_path)
    for stale_glob in stale_globs:
        for path in out_path.parent.glob(str(stale_glob)):
            if path.name not in written_bundles:
                path.unlink()
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


def write_uvarint(out: bytearray, value: int) -> None:
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"negative varuint payload: {value!r}")
    while value_i >= 0x80:
        out.append((value_i & 0x7F) | 0x80)
        value_i >>= 7
    out.append(value_i)


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
