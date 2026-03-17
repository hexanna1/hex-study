from __future__ import annotations

import argparse
import hashlib
import json
import struct
from decimal import Decimal, ROUND_HALF_EVEN
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
NODE_ROW_SIZE = 1
PACKED_NODE_COUNT_BITS = 5
PACKED_NODE_COUNT_MASK = (1 << PACKED_NODE_COUNT_BITS) - 1
PACKED_NODE_IS_CORE_SHIFT = PACKED_NODE_COUNT_BITS
PACKED_CANDIDATE_METRIC_BITS = 10
PACKED_CANDIDATE_DELTA_BITS = 7
PACKED_CANDIDATE_DELTA_ESCAPE = (1 << PACKED_CANDIDATE_DELTA_BITS) - 1
PACKED_CANDIDATE_DELTA_MAX_ABS = (PACKED_CANDIDATE_DELTA_ESCAPE - 1) // 2
THOUSANDTH = Decimal("0.001")


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


def _encode_optional_thousandths(raw: Any) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)):
        raise ValueError(f"bad numeric payload: {raw!r}")
    value = Decimal(str(raw)).quantize(THOUSANDTH, rounding=ROUND_HALF_EVEN)
    return int(value * 1000)


def _pack_optional_u10(value: int | None) -> int:
    if value is None:
        return PACKED_OPTIONAL_NULL
    if value < 0 or value >= PACKED_OPTIONAL_NULL:
        raise ValueError(f"bad u10 payload: {value!r}")
    return int(value)


def _pack_bits(rows: list[tuple[int, int]]) -> bytes:
    bit_len = sum(int(bits) for _word, bits in rows)
    out = bytearray((bit_len + 7) // 8)
    bit_offset = 0
    for word, bits in rows:
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        chunk = int(word) << shift
        for byte_idx in range(6):
            out_idx = byte_offset + byte_idx
            if out_idx >= len(out):
                break
            out[out_idx] |= (chunk >> (8 * byte_idx)) & 0xFF
        bit_offset += int(bits)
    return bytes(out)


def _pack_candidate_word(
    *,
    board_size: int,
    move_id: int,
    prior: int | None,
    delta_code: int | None,
    has_child: bool,
) -> tuple[int, int]:
    move_id_bits = _packed_move_id_bits(board_size)
    row_bits = _candidate_row_bits(board_size)
    has_child_shift = move_id_bits + PACKED_CANDIDATE_METRIC_BITS + PACKED_CANDIDATE_DELTA_BITS
    move_id_i = int(move_id)
    if move_id_i < 0 or move_id_i > PACKED_MOVE_ID_MAX:
        raise ValueError(f"bad packed move id payload: {move_id!r}")
    if move_id_i >= (1 << move_id_bits):
        raise ValueError(f"packed move id exceeds board-size capacity: {move_id!r}")
    prior_u10 = _pack_optional_u10(prior)
    delta_i = int(delta_code) if delta_code is not None else PACKED_CANDIDATE_DELTA_ESCAPE
    if delta_i < 0 or delta_i > PACKED_CANDIDATE_DELTA_ESCAPE:
        raise ValueError(f"bad packed delta payload: {delta_code!r}")
    return (
        move_id_i
        | (prior_u10 << move_id_bits)
        | (delta_i << (move_id_bits + PACKED_CANDIDATE_METRIC_BITS))
        | ((1 if bool(has_child) else 0) << has_child_shift)
    ), row_bits


def _pack_single_candidate_word(
    *,
    board_size: int,
    move_id: int,
    prior: int | None,
    has_child: bool,
) -> tuple[int, int]:
    move_id_bits = _packed_move_id_bits(board_size)
    row_bits = _single_candidate_row_bits(board_size)
    move_id_i = int(move_id)
    if move_id_i < 0 or move_id_i > PACKED_MOVE_ID_MAX:
        raise ValueError(f"bad packed move id payload: {move_id!r}")
    if move_id_i >= (1 << move_id_bits):
        raise ValueError(f"packed move id exceeds board-size capacity: {move_id!r}")
    prior_u10 = _pack_optional_u10(prior)
    return (
        move_id_i
        | (prior_u10 << move_id_bits)
        | ((1 if bool(has_child) else 0) << (move_id_bits + PACKED_CANDIDATE_METRIC_BITS))
    ), row_bits


def _candidate_row_bits(board_size: int) -> int:
    return _packed_move_id_bits(board_size) + PACKED_CANDIDATE_METRIC_BITS + PACKED_CANDIDATE_DELTA_BITS + 1


def _single_candidate_row_bits(board_size: int) -> int:
    return _packed_move_id_bits(board_size) + PACKED_CANDIDATE_METRIC_BITS + 1


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


def _pack_candidate_bitstream(*, board_size: int, candidates: list[list[Any]]) -> bytes:
    rows: list[tuple[int, int]] = []
    exceptions: list[int] = []
    for move_id, prior, red_winrate, parent_edge_red_winrate, has_child, node_candidate_count in candidates:
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
            rows.append(
                _pack_single_candidate_word(
                    board_size=board_size,
                    move_id=move_id,
                    prior=prior,
                    has_child=has_child,
                )
            )
            continue
        delta_code = _delta_code(
            red_winrate=red_winrate,
            parent_edge_red_winrate=parent_edge_red_winrate,
        )
        if delta_code is None:
            exceptions.append(_pack_optional_u10(red_winrate))
        rows.append(
            _pack_candidate_word(
                board_size=board_size,
                move_id=move_id,
                prior=prior,
                delta_code=delta_code,
                has_child=has_child,
            )
        )
    return _pack_bits(rows) + _pack_bits([(value, PACKED_CANDIDATE_METRIC_BITS) for value in exceptions])


def _pack_node(*, candidate_count: int, is_core: bool) -> bytes:
    count = int(candidate_count)
    if count < 0 or count > PACKED_NODE_COUNT_MASK:
        raise ValueError(f"bad packed node candidate count payload: {candidate_count!r}")
    word = count | ((1 if bool(is_core) else 0) << PACKED_NODE_IS_CORE_SHIFT)
    return bytes([word])


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


def _require_root_shape(*, nodes_raw: list[Any], artifact_path: Path, board_size: int) -> None:
    root_raw = nodes_raw[0]
    if not isinstance(root_raw, dict):
        raise ValueError(f"bad root node payload: {root_raw!r}")
    root_parent = root_raw.get("parent")
    if root_parent is not None:
        raise ValueError(f"move-tree artifact root parent must be null: {artifact_path}")
    root_move = root_raw.get("move")
    if _normalize_optional_move_id(root_move, board_size=board_size) != ROOT_MOVE_ID:
        raise ValueError(f"move-tree artifact root move must be empty: {artifact_path}")
    for node_idx, node_raw in enumerate(nodes_raw[1:], start=1):
        if not isinstance(node_raw, dict):
            raise ValueError(f"bad node payload: {node_raw!r}")
        if node_raw.get("parent") is None:
            raise ValueError(f"move-tree artifact has non-root node with null parent at index {node_idx}: {artifact_path}")


def _normalize_child(raw: Any, *, node_count: int) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"bad child index: {raw!r}")
        if raw >= node_count:
            return None
        return int(raw)
    raise ValueError(f"bad child payload: {raw!r}")


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
    if not nodes_raw:
        raise ValueError(f"move-tree artifact missing root node: {artifact_path}")
    artifact_board_size = int(data["board_size"])
    if data.get("root") != 0:
        raise ValueError(f"move-tree artifact root must be 0: {artifact_path}")
    _require_root_shape(nodes_raw=nodes_raw, artifact_path=artifact_path, board_size=artifact_board_size)

    normalized_nodes: list[dict[str, Any]] = []
    for node_idx, node_raw in enumerate(nodes_raw):
        if not isinstance(node_raw, dict):
            raise ValueError(f"bad node payload: {node_raw!r}")
        node_candidates = node_raw.get("candidates")
        if not isinstance(node_candidates, list):
            raise ValueError(f"node missing candidates list: {node_raw!r}")
        retained_candidates: list[dict[str, Any]] = []
        for candidate_raw in node_candidates:
            if not isinstance(candidate_raw, dict):
                raise ValueError(f"bad candidate payload: {candidate_raw!r}")
            move = str(candidate_raw.get("move") or "").strip().lower()
            if not move:
                raise ValueError(f"candidate missing move: {candidate_raw!r}")
            move_id = _move_to_cell_id(move, board_size=artifact_board_size)
            child = _normalize_child(candidate_raw.get("child"), node_count=len(nodes_raw))
            if child is not None:
                child_raw = nodes_raw[child]
                if not isinstance(child_raw, dict):
                    raise ValueError(f"bad child node payload: {child_raw!r}")
                child_parent = _normalize_parent(child_raw.get("parent"))
                child_move = _normalize_optional_move_id(child_raw.get("move"), board_size=artifact_board_size)
                if child_parent != node_idx or child_move != move_id:
                    raise ValueError(f"child node does not match retained candidate: {candidate_raw!r}")
            mover_winrate = _encode_optional_thousandths(candidate_raw.get("tree_mover_winrate"))
            retained_candidates.append(
                {
                    "move_id": move_id,
                    "prior": _encode_optional_thousandths(candidate_raw.get("prior")),
                    "mover_winrate": mover_winrate,
                    "red_winrate": _red_winrate_from_mover_winrate(
                        mover_winrate=mover_winrate,
                        parent_ply=int(node_raw.get("ply") or 0),
                    ),
                    "child": child,
                }
            )
        normalized_nodes.append(
            {
                "importance": _encode_optional_thousandths(node_raw.get("importance")),
                "candidates": retained_candidates,
            }
        )

    nodes: list[list[Any]] = []
    candidates: list[list[Any]] = []
    visited: set[int] = set()

    def append_preorder(node_idx: int, parent_edge_red_winrate: int | None = None) -> None:
        if node_idx in visited:
            raise ValueError(f"duplicate move-tree node reference: {node_idx!r}")
        visited.add(node_idx)
        node = normalized_nodes[node_idx]
        importance = node["importance"]
        if not isinstance(importance, int):
            raise ValueError(f"bad importance payload: {importance!r}")
        node_candidates = node["candidates"]
        nodes.append(
            [
                int(len(node_candidates)),
                int(importance) >= CORE_IMPORTANCE_MIN_THOUSANDTHS,
            ]
        )
        child_indices: list[int] = []
        node_candidate_count = int(len(node_candidates))
        for candidate in node_candidates:
            child = candidate["child"]
            candidates.append(
                [
                    candidate["move_id"],
                    candidate["prior"],
                    candidate["red_winrate"],
                    parent_edge_red_winrate,
                    child is not None,
                    node_candidate_count,
                ]
            )
            if child is not None:
                child_indices.append(int(child))
        for child in child_indices:
            parent_red = None
            for candidate in node_candidates:
                if candidate["child"] == child:
                    parent_red = candidate["red_winrate"]
                    break
            append_preorder(child, parent_red)

    append_preorder(0)
    if len(visited) != len(nodes_raw):
        raise ValueError("move-tree artifact contains unreachable nodes")

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
    for candidate_count, is_core in nodes:
        out.extend(
            _pack_node(
                candidate_count=candidate_count,
                is_core=is_core,
            )
        )
    out.extend(_pack_candidate_bitstream(board_size=artifact_board_size, candidates=candidates))
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
