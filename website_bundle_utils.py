from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


THOUSANDTH = Decimal("0.001")
PACKED_OPTIONAL_U10_NULL = 1023


@dataclass(frozen=True)
class BundlePayload:
    prefix: str
    payload: bytes


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


def pack_little_endian_bits(rows: list[tuple[int, int]], *, chunk_bytes: int) -> bytes:
    bit_len = sum(int(bits) for _word, bits in rows)
    out = bytearray((bit_len + 7) // 8)
    bit_offset = 0
    for word, bits in rows:
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        chunk = int(word) << shift
        for byte_idx in range(int(chunk_bytes)):
            out_idx = byte_offset + byte_idx
            if out_idx >= len(out):
                break
            out[out_idx] |= (chunk >> (8 * byte_idx)) & 0xFF
        bit_offset += int(bits)
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
