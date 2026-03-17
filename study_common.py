from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from hex_symmetry import apply_transform_ax, inverse_transform_id
import local_pattern_representative as lpr

_apply_transform_ax = apply_transform_ax
_inverse_transform_id = inverse_transform_id


def _center_index(board_size: int) -> int:
    # Ceiling center for even boards, exact center for odd boards.
    return (int(board_size) // 2) + 1


CELL_PARSE_RE = re.compile(r"^([A-Za-z]+)([1-9][0-9]*)$")
SIZE_RE = re.compile(r"^\s*([0-9]+)")


def _stone_fraction_for_importance(*, stone_fraction: float, child_ply: int) -> float:
    ply = int(child_ply)
    if ply <= 1:
        return float(stone_fraction)
    return float(stone_fraction) ** (ply / (ply + 1.0))


def _safe_name(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    cleaned = "".join(out).strip("-")
    return cleaned or "exp"


def _duration_parts(sec: float) -> tuple[int, int, int]:
    total = int(round(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return h, m, s


def _fmt_duration_compact(sec: float, *, subsecond_under_minute: bool) -> str:
    h, m, s = _duration_parts(sec)
    if h == 0 and m == 0:
        if subsecond_under_minute:
            return f"{float(sec):.1f}s"
        return f"{s}s"

    parts: list[str] = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    if s > 0:
        parts.append(f"{s:02d}s")
    if not parts:
        return "0s"
    return "".join(parts)


def _fmt_s(sec: float) -> str:
    return _fmt_duration_compact(sec, subsecond_under_minute=True)


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}")


def _load_raw_nn_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Raw-NN cache file is not a JSON object: {path}")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError(f"Bad raw-NN cache entry in {path}: {key!r}")
        out[key] = value
    return out


def _save_raw_nn_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {k: cache[k] for k in sorted(cache)},
            ensure_ascii=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _encode_millionths(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"value is not numeric: {value!r}")
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1_000_000))
    raise ValueError(f"value is not numeric: {value!r}")


def _decode_millionths(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"value is not integer millionths: {value!r}")
    return float(value) / 1_000_000.0


def _encode_compact_raw_nn_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "r" in payload:
        out["r"] = _encode_millionths(payload.get("r"))
    moves = payload.get("m")
    if isinstance(moves, list):
        rows: list[Any] = []
        for row in moves:
            if isinstance(row, list) and row:
                next_row = list(row)
                if len(next_row) >= 2:
                    next_row[1] = _encode_millionths(next_row[1])
                rows.append(next_row)
        if rows:
            out["m"] = rows
    return out


def _decode_compact_raw_nn_payload(
    payload: dict[str, Any], *, include_moves: bool = True
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "r" in payload:
        out["r"] = _decode_millionths(payload.get("r"))
    moves = payload.get("m") if include_moves else None
    if isinstance(moves, list):
        rows: list[Any] = []
        for row in moves:
            if isinstance(row, list) and row:
                next_row = list(row)
                if len(next_row) >= 2:
                    next_row[1] = _decode_millionths(next_row[1])
                rows.append(next_row)
        if rows:
            out["m"] = rows
    elif moves is not None:
        raise ValueError(f"raw-NN cache moves payload is not a list: {moves!r}")
    return out


def _cached_payload_red_winrate(payload: dict[str, Any]) -> float | None:
    red_winrate = payload.get("r")
    if isinstance(red_winrate, float):
        return red_winrate
    return None


def _cached_payload_moves(payload: dict[str, Any]) -> list[Any]:
    moves = payload.get("m")
    if isinstance(moves, list):
        return moves
    return []


def _cached_payload_move_prior(row: Any) -> tuple[str, float | None] | None:
    if isinstance(row, list) and row:
        move = str(row[0] or "").strip().lower()
        if not move:
            return None
        prior_value = row[1] if len(row) >= 2 else None
        prior = prior_value if isinstance(prior_value, float) else None
        return move, prior
    return None


def _is_valid_compact_raw_nn_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if "r" not in payload and "m" not in payload:
        return False
    red_winrate = payload.get("r")
    if red_winrate is not None and not isinstance(red_winrate, (int, float)):
        return False
    moves = payload.get("m")
    if moves is None:
        return True
    if not isinstance(moves, list):
        return False
    for row in moves:
        if not isinstance(row, list) or not row:
            return False
        move = str(row[0] or "").strip().lower()
        if not move:
            return False
        if len(row) >= 2 and not isinstance(row[1], (int, float)):
            return False
    return True


def _is_valid_encoded_compact_raw_nn_payload(payload: Any) -> bool:
    if not _is_valid_compact_raw_nn_payload(payload):
        return False
    red_winrate = payload.get("r")
    if red_winrate is not None and (isinstance(red_winrate, bool) or not isinstance(red_winrate, int)):
        return False
    for row in _cached_payload_moves(payload):
        if len(row) >= 2 and (isinstance(row[1], bool) or not isinstance(row[1], int)):
            return False
    return True


def _is_valid_encoded_raw_nn_winrate(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    red_winrate = payload.get("r")
    return isinstance(red_winrate, int) and not isinstance(red_winrate, bool)


def _is_valid_encoded_raw_nn_policy(payload: Any) -> bool:
    return (
        _is_valid_encoded_compact_raw_nn_payload(payload)
        and isinstance(payload.get("m"), list)
        and all(isinstance(row, list) and len(row) >= 2 for row in payload["m"])
    )


def _native_batch_raw_nn_command(*, board_size: int, move_limit: int) -> list[str]:
    home = Path.home()
    katago = Path(os.environ.get("HEXWIKI_KATAGO", home / "KataGo-hex" / "build-opencl" / "katago")).expanduser()
    config = Path(os.environ.get("HEXWIKI_KATAGO_CONFIG", home / "lizzieyzy" / "engine.cfg")).expanduser()
    model = Path(os.environ.get("HEXWIKI_KATAGO_MODEL", home / "lizzieyzy" / "weights" / "hex27x3.bin.gz")).expanduser()
    workers = int(os.environ.get("HEXWIKI_RAW_NN_WORKERS", "128"))
    return [
        str(katago),
        "batchrawnn",
        "-config",
        str(config),
        "-model",
        str(model),
        "-board-size",
        str(int(board_size)),
        "-top-n",
        str(int(move_limit)),
        "-workers",
        str(workers),
    ]


def _position_expand_binary() -> Path:
    source = Path(__file__).resolve().with_name("position_expand.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path(tempfile.gettempdir()) / f"hexwiki-position-expand-{digest}"
    if binary.exists():
        return binary
    pending = binary.with_name(f"{binary.name}.{os.getpid()}.tmp")
    proc = subprocess.run(
        ["c++", "-O3", "-std=c++17", str(source), "-o", str(pending)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Position expansion build failed{suffix}")
    pending.replace(binary)
    return binary


def _run_position_expansion(input_lines: list[str], *, expected_rows: int) -> list[str]:
    proc = subprocess.run(
        [str(_position_expand_binary())],
        input="\n".join(input_lines) + "\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Position expansion failed{suffix}")
    output_lines = proc.stdout.splitlines()
    if len(output_lines) != expected_rows:
        raise ValueError(
            f"Position expansion returned {len(output_lines)} rows for {expected_rows} requests"
        )
    return output_lines


def _run_native_child_positions(
    *,
    requests: list[tuple[str, list[str]]],
    board_size: int,
) -> list[list[str]]:
    if not requests:
        return []
    input_lines = [f"children-v1\t{int(board_size)}"]
    for position, moves in requests:
        input_lines.append(f"{position}\t{';'.join(moves)}")
    output_lines = _run_position_expansion(input_lines, expected_rows=len(requests))
    expanded = [line.split("\t") if line else [] for line in output_lines]
    for (_position, moves), children in zip(requests, expanded):
        if len(children) != len(moves):
            raise ValueError(
                f"Position expansion returned {len(children)} children for {len(moves)} moves"
            )
    return expanded


def _run_multi_position_raw_nn_native(
    *,
    position_inputs: list[str],
    board_size: int,
    move_limit: int,
    precanonicalized_position_inputs: bool = False,
) -> dict[str, dict[str, Any]]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    if not positions:
        return {}
    cache_key = _precanonicalized_position_cache_key if precanonicalized_position_inputs else _cache_key
    canonical_positions = [cache_key(position) for position in positions]
    proc = subprocess.run(
        _native_batch_raw_nn_command(board_size=board_size, move_limit=move_limit),
        input="".join(f"{position}\n" for position in canonical_positions),
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(stdout_lines) != len(positions):
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            f"Native raw-NN batch returned {len(stdout_lines)} records for "
            f"{len(positions)} positions{suffix}"
        )
    records: dict[str, dict[str, Any]] = {}
    for position, canonical_position, line in zip(positions, canonical_positions, stdout_lines):
        payload = json.loads(line)
        if not isinstance(payload, dict) or str(payload.get("position") or "").strip() != canonical_position:
            raise ValueError("Native raw-NN batch returned a mismatched record")
        error = str(payload.get("error") or "").strip()
        if error:
            raise ValueError(f"Native raw-NN evaluation failed: {error}")
        reduced = _decode_compact_raw_nn_payload(payload)
        if not _is_valid_compact_raw_nn_payload(reduced):
            raise ValueError("Native raw-NN batch returned an invalid compact payload")
        records[position] = reduced
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Native raw-NN batch exited with status {proc.returncode}{suffix}")
    return records


def _ensure_raw_nn_cache_entries(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    board_size: int,
    raw_nn_cache_path: Path | None = None,
    chunk_size: int = 10000,
    move_limit: int = 24,
    require_moves: bool = True,
    store_moves: bool = True,
    precanonicalized_position_inputs: bool = False,
    policy_validator: Callable[[str, dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    encoded_payloads: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    missing_positions: list[str] = []
    seen_missing: set[str] = set()
    cache_key_for = _precanonicalized_position_cache_key if precanonicalized_position_inputs else _cache_key
    for position in positions:
        cache_key = cache_key_for(position)
        cached = raw_nn_cache.get(cache_key)
        cached_is_valid = (
            _is_valid_encoded_raw_nn_policy(cached)
            and (policy_validator is None or policy_validator(position, cached))
            if require_moves
            else _is_valid_encoded_raw_nn_winrate(cached)
        )
        if cached_is_valid:
            encoded_payloads[position] = cached
            cache_hits += 1
            continue
        if position not in seen_missing:
            seen_missing.add(position)
            missing_positions.append(position)
    for i in range(0, len(missing_positions), int(chunk_size)):
        batch = missing_positions[i : i + int(chunk_size)]
        fetched = _run_multi_position_raw_nn_native(
            position_inputs=batch,
            board_size=board_size,
            move_limit=move_limit,
            precanonicalized_position_inputs=precanonicalized_position_inputs,
        )
        for position, reduced in fetched.items():
            if not require_moves:
                red_winrate = _cached_payload_red_winrate(reduced)
                if red_winrate is None:
                    raise ValueError(f"Raw-NN payload missing root winrate for {position!r}")
            encoded = _encode_compact_raw_nn_payload(reduced)
            encoded_payloads[position] = encoded
            raw_nn_cache[cache_key_for(position)] = (
                encoded
                if store_moves
                else {"r": encoded["r"]}
            )
        if isinstance(raw_nn_cache_path, Path):
            _save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
    return encoded_payloads, cache_hits


def _cached_raw_nn_winrate_count(
    raw_nn_cache: dict[str, dict[str, Any]],
    positions: list[str],
) -> int:
    return sum(
        _is_valid_encoded_raw_nn_winrate(
            raw_nn_cache.get(_precanonicalized_position_cache_key(position))
        )
        for position in positions
    )


def _extract_board_size_from_input(position_input: str) -> int | None:
    s = str(position_input or "").strip()
    if not s:
        return None
    frag = s.split("#", 1)[1] if "#" in s else s
    m = SIZE_RE.match(frag)
    if not m:
        return None
    return int(m.group(1))


def _cache_key(position_input: str, move: str | None = None) -> str:
    position = str(position_input or "").strip()
    if move is not None:
        token = str(move).strip().lower()
        if not token:
            raise ValueError("Missing move token for child cache key")
        position = _position_after_move(position, token)
    size, red, blue, to_play = _position_state(position)
    past_stream = lpr.serialize_position_stream(
        red_cells=tuple(sorted(red)),
        blue_cells=tuple(sorted(blue)),
        to_play=to_play,
    )
    return f"{int(size)},{past_stream}" if past_stream else str(int(size))


def _precanonicalized_position_cache_key(position_input: str) -> str:
    position = str(position_input or "").strip()
    fragment = position.split("#", 1)[1] if "#" in position else position
    marker = "c1,"
    if marker not in fragment:
        raise ValueError(f"Expected a serialized HexWorld position, got {position_input!r}")
    size_text, stream = fragment.split(marker, 1)
    if not size_text.isdigit():
        raise ValueError(f"Expected a serialized HexWorld position, got {position_input!r}")
    size = str(int(size_text))
    return f"{size},{stream}" if stream else size


def _letters_for_col(col: int) -> str:
    out: list[str] = []
    v = col
    while v > 0:
        v -= 1
        out.append(chr(ord("a") + (v % 26)))
        v //= 26
    return "".join(reversed(out))


def _canonical_pass_proxy_move(board_size: int, to_play: str) -> str | None:
    m = _center_index(board_size)
    side = str(to_play or "").strip().lower()
    if side == "red":
        return f"{_letters_for_col(m)}1"
    if side == "blue":
        return f"a{m}"
    return None


def _letters_to_col(letters: str) -> int:
    n = 0
    for ch in letters.lower():
        if not ("a" <= ch <= "z"):
            raise ValueError(f"Bad column letters: {letters!r}")
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n


def _cell_to_col_row(cell: str) -> tuple[int, int]:
    m = CELL_PARSE_RE.fullmatch(cell.strip())
    if not m:
        raise ValueError(f"Bad cell token: {cell!r}")
    return _letters_to_col(m.group(1)), int(m.group(2))


def _position_state(position: str) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], str]:
    size, past_moves, _future_moves, to_play = lpr.PARSE_HEXWORLD_POSITION(position)
    board = lpr.BOARD_CLS(size)
    for idx, mv in enumerate(past_moves, start=1):
        if not board.apply_move(mv):
            raise ValueError(f"Illegal past move at index {idx} for {position!r}")
    red: set[tuple[int, int]] = set()
    blue: set[tuple[int, int]] = set()
    for row in range(1, int(size) + 1):
        for col in range(1, int(size) + 1):
            value = board.get(col, row)
            if value == int(lpr.SIDE_ENUM.RED):
                red.add((int(col), int(row)))
            elif value == int(lpr.SIDE_ENUM.BLUE):
                blue.add((int(col), int(row)))
    to_play_s = "red" if to_play == lpr.SIDE_ENUM.RED else "blue"
    return int(size), red, blue, to_play_s


def _position_after_move(position: str, move: str) -> str:
    size, red, blue, to_play = _position_state(position)
    return _position_after_move_from_state(
        size=size,
        red=red,
        blue=blue,
        to_play=to_play,
        move=move,
    )


def _position_after_move_from_state(
    *,
    size: int,
    red: set[tuple[int, int]],
    blue: set[tuple[int, int]],
    to_play: str,
    move: str,
) -> str:
    col, row = lpr.CELL_TO_COL_ROW(str(move).strip().lower())
    point = (int(col), int(row))
    if point in red or point in blue:
        raise ValueError(f"child move already occupied: {move!r}")
    red_next = set(red)
    blue_next = set(blue)
    if to_play == "red":
        red_next.add(point)
        next_to_play = "blue"
    else:
        blue_next.add(point)
        next_to_play = "red"
    return lpr.serialize_position(
        board_size=size,
        red_cells=tuple(sorted(red_next)),
        blue_cells=tuple(sorted(blue_next)),
        to_play=next_to_play,
    )


def _canonicalize_base_rel_under_orbit(base_rel: tuple[int, int], exp_meta: dict[str, Any] | None) -> tuple[int, int]:
    if not exp_meta:
        return base_rel
    orbit = exp_meta.get("local_key_orbit")
    if not isinstance(orbit, list) or not orbit:
        return base_rel
    best = base_rel
    for entry in orbit:
        if not isinstance(entry, dict):
            continue
        try:
            ti = int(entry["transform_id"])
            shift = entry["norm_shift"]
            sq, sr = int(shift[0]), int(shift[1])
            p_t = _apply_transform_ax(base_rel, ti)
            cand = (p_t[0] - sq, p_t[1] - sr)
        except Exception:
            continue
        if cand < best:
            best = cand
    return best


def _candidate_key_local_for_move(move: str, exp_meta: dict[str, Any] | None) -> str | None:
    if not exp_meta:
        return None
    try:
        col, row = _cell_to_col_row(move)
    except ValueError:
        return None

    try:
        transform_id = int(exp_meta["orientation_transform_id"])
        shift = exp_meta["orientation_norm_shift"]
        offset = exp_meta["placement_offset"]
        shift_q, shift_r = int(shift[0]), int(shift[1])
        dq, dr = int(offset[0]), int(offset[1])
    except Exception:
        return None

    ori_rel = (col - dq, row - dr)
    unnorm = (ori_rel[0] + shift_q, ori_rel[1] + shift_r)
    inv_id = _inverse_transform_id(transform_id)
    base_rel = _apply_transform_ax(unnorm, inv_id)
    canonical_base_rel = _canonicalize_base_rel_under_orbit(base_rel, exp_meta)
    return f"{canonical_base_rel[0]},{canonical_base_rel[1]}"


def _attach_candidate_keys(rows: list[dict[str, Any]], exp_meta: dict[str, Any] | None) -> None:
    for row in rows:
        move = str(row.get("move") or "")
        row["candidate_abs"] = move
        row["candidate_key_local"] = _candidate_key_local_for_move(move, exp_meta)


def _logit_clamped(winrate: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, float(winrate)))
    return math.log(p / (1.0 - p))


def _position_to_play(position_input: str) -> str:
    position = str(position_input or "").strip()
    if not position:
        return ""
    try:
        _size, _red, _blue, to_play = _position_state(position)
    except Exception:
        return ""
    return to_play


def _attach_stone_fractions(
    rows: list[dict[str, Any]],
    *,
    position_input: str,
    allow_first_row_proxy_fallback: bool = False,
) -> None:
    for row in rows:
        row["stone_fraction"] = None

    valid = [r for r in rows if isinstance(r.get("mean_winrate"), (int, float))]
    if not valid:
        return

    board_size = _extract_board_size_from_input(position_input)
    to_play = _position_to_play(position_input)
    canonical_proxy = (
        _canonical_pass_proxy_move(board_size, to_play) if isinstance(board_size, int) else None
    )

    proxy_row: dict[str, Any] | None = None
    if canonical_proxy is not None:
        for row in valid:
            if str(row.get("move") or "").lower() == canonical_proxy:
                proxy_row = row
                break
    if proxy_row is None and allow_first_row_proxy_fallback:
        proxy_row = min(valid, key=lambda r: int(r.get("_idx", 0)))
    if proxy_row is None:
        return

    best_row = max(valid, key=lambda r: float(r["mean_winrate"]))
    l_proxy = _logit_clamped(float(proxy_row["mean_winrate"]))
    l_best = _logit_clamped(float(best_row["mean_winrate"]))
    denom = l_best - l_proxy
    if abs(denom) < 1e-12:
        return

    for row in valid:
        l_row = _logit_clamped(float(row["mean_winrate"]))
        row["stone_fraction"] = (l_row - l_proxy) / denom


def _build_pooled_candidates(
    summary_rows: list[dict[str, Any]], *, total_representatives: int, value_field: str = "corrected_value"
) -> list[dict[str, Any]]:
    if value_field not in {"stone_fraction", "corrected_value"}:
        raise ValueError(f"Unsupported pooled candidate value field: {value_field!r}")
    corrected = value_field == "corrected_value"
    mean_key = "mean_corrected_value" if corrected else "mean_stone_fraction"
    stdev_key = "stdev_corrected_value" if corrected else "stdev_stone_fraction"
    min_key = "min_corrected_value" if corrected else "min_stone_fraction"
    max_key = "max_corrected_value" if corrected else "max_stone_fraction"

    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = row.get("candidate_key_local")
        value = row.get(value_field)
        exp_name = str(row.get("experiment") or "")
        # Intentional: pool all retained row-level samples for a local key.
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(value, (int, float)):
            continue
        by_key.setdefault(key, []).append(
            {
                "experiment": exp_name,
                "candidate_abs": str(row.get("candidate_abs") or ""),
                value_field: float(value),
            }
        )

    out: list[dict[str, Any]] = []
    for key, rows in by_key.items():
        values = [float(r[value_field]) for r in rows]
        exp_set = {r["experiment"] for r in rows if r["experiment"]}
        sample_abs = next((r["candidate_abs"] for r in rows if r["candidate_abs"]), "")
        n = len(exp_set) if exp_set else len(rows)
        coverage = (n / total_representatives) if total_representatives > 0 else 0.0
        out.append(
            {
                "candidate_key_local": key,
                "sample_candidate_abs": sample_abs,
                "n": n,
                "coverage": coverage,
                mean_key: mean(values),
                stdev_key: pstdev(values) if len(values) >= 2 else 0.0,
                min_key: min(values),
                max_key: max(values),
            }
        )

    if not out:
        return out

    if corrected:
        denom = max((float(row[mean_key]) for row in out), default=0.0)
        if denom > 1e-12:
            for row in out:
                row["mean_stone_fraction"] = float(row[mean_key]) / denom
                row["stdev_stone_fraction"] = float(row[stdev_key]) / denom
                row["min_stone_fraction"] = float(row[min_key]) / denom
                row["max_stone_fraction"] = float(row[max_key]) / denom
        else:
            # Intentional flat fallback when no positive corrected anchor exists.
            for row in out:
                row["mean_stone_fraction"] = 0.0
                row["stdev_stone_fraction"] = 0.0
                row["min_stone_fraction"] = 0.0
                row["max_stone_fraction"] = 0.0

    out.sort(
        key=lambda r: (
            -float(r["mean_stone_fraction"]),
            -(float(r.get("mean_corrected_value")) if isinstance(r.get("mean_corrected_value"), (int, float)) else float("-inf")),
            -int(r["n"]),
            str(r["candidate_key_local"]),
        )
    )
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out


def _write_pooled_candidates_json(path: Path, pooled_rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(pooled_rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
