from __future__ import annotations

from datetime import datetime
import json
import math
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from hex_symmetry import apply_transform_ax, inverse_transform_id
import local_pattern_representative as lpr

_apply_transform_ax = apply_transform_ax
_inverse_transform_id = inverse_transform_id


def _center_index(board_size: int) -> int:
    # Ceiling center for even boards, exact center for odd boards.
    return (int(board_size) // 2) + 1


CELL_RE = re.compile(r"^[A-Za-z]+[1-9][0-9]*$")
CELL_PARSE_RE = re.compile(r"^([A-Za-z]+)([1-9][0-9]*)$")
SIZE_RE = re.compile(r"^\s*([0-9]+)")


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
    return raw


def _save_raw_nn_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({k: cache[k] for k in sorted(cache)}, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _minimal_analyze_payload(payload: dict[str, Any], *, move_limit: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    root_eval = payload.get("root_eval")
    if isinstance(root_eval, dict):
        red_winrate = root_eval.get("red_winrate")
        if isinstance(red_winrate, (int, float)):
            out["r"] = float(red_winrate)
    kept_moves: list[list[Any]] = []
    for row in payload.get("moves", []):
        if not isinstance(row, dict):
            continue
        move = str(row.get("move") or "").strip().lower()
        if not move:
            continue
        kept: list[Any] = [move]
        if isinstance(row.get("prior"), (int, float)):
            kept.append(float(row["prior"]))
        kept_moves.append(kept)
        if isinstance(move_limit, int) and len(kept_moves) >= int(move_limit):
            break
    if kept_moves:
        out["m"] = kept_moves
    return out


def _cached_payload_red_winrate(payload: dict[str, Any]) -> float | None:
    red_winrate = payload.get("r")
    if isinstance(red_winrate, (int, float)):
        return float(red_winrate)
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
        prior = float(row[1]) if len(row) >= 2 and isinstance(row[1], (int, float)) else None
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


def _run_multi_position_analyze_cached(
    *,
    hexata_main: Path,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    chunk_size: int = 250,
    move_limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    payloads: dict[str, dict[str, Any]] = {}
    missing_positions: list[str] = []
    seen_missing: set[str] = set()
    for position in positions:
        cache_key = _cache_key(position)
        cached = raw_nn_cache.get(cache_key)
        if _is_valid_compact_raw_nn_payload(cached):
            payloads[position] = cached
            continue
        if position not in seen_missing:
            seen_missing.add(position)
            missing_positions.append(position)
    for i in range(0, len(missing_positions), int(chunk_size)):
        batch = missing_positions[i : i + int(chunk_size)]
        fetched = _run_multi_position_analyze(
            hexata_main=hexata_main,
            position_inputs=batch,
        )
        for position, payload in fetched.items():
            reduced = _minimal_analyze_payload(payload, move_limit=move_limit)
            payloads[position] = reduced
            raw_nn_cache[_cache_key(position)] = reduced
        if isinstance(raw_nn_cache_path, Path):
            _save_raw_nn_cache(raw_nn_cache_path, raw_nn_cache)
    return payloads


def _cached_request_count(raw_nn_cache: dict[str, dict[str, Any]], positions: list[str]) -> int:
    return sum(1 for position in positions if _is_valid_compact_raw_nn_payload(raw_nn_cache.get(_cache_key(position))))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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
    past_stream = lpr._serialize_position_stream(
        red_cells=tuple(sorted(red)),
        blue_cells=tuple(sorted(blue)),
        to_play=to_play,
    )
    return f"{int(size)},{past_stream}" if past_stream else str(int(size))


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
    board = lpr.HEXBOARD_CLS(size)
    for idx, mv in enumerate(past_moves, start=1):
        ok = False
        if mv.kind == lpr.MOVE_KIND.PLACE:
            ok = board.place(mv.side, mv.col, mv.row)
        elif mv.kind == lpr.MOVE_KIND.PASS:
            ok = board.pass_move(mv.side)
        elif mv.kind == lpr.MOVE_KIND.SWAP:
            ok = board.swap_move(mv.side)
        if not ok:
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
    return lpr._serialize_position(
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


def _row_side_to_play_winrate(row: dict[str, Any], *, to_play: str) -> float | None:
    red_wr = row.get("red_winrate")
    if not isinstance(red_wr, (int, float)):
        return None
    side = str(to_play or "").strip().lower()
    p_red = float(red_wr)
    if side == "red":
        return p_red
    if side == "blue":
        return 1.0 - p_red
    return None


def _root_eval_side_to_play_winrate(payload: dict[str, Any]) -> float | None:
    root_eval = payload.get("root_eval")
    if not isinstance(root_eval, dict):
        return None
    position = payload.get("position") if isinstance(payload.get("position"), dict) else {}
    to_play = str(position.get("to_play") or "")
    return _row_side_to_play_winrate(root_eval, to_play=to_play)


def _attach_stone_fractions(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    position_input: str | None = None,
) -> None:
    for row in rows:
        row["stone_fraction"] = None

    valid = [r for r in rows if isinstance(r.get("mean_winrate"), (int, float))]
    if not valid:
        return

    position = payload.get("position") if isinstance(payload.get("position"), dict) else {}
    board_size_input = position_input if isinstance(position_input, str) and position_input.strip() else str(position.get("input") or "")
    board_size = _extract_board_size_from_input(board_size_input)
    to_play = str(position.get("to_play") or "")
    canonical_proxy = (
        _canonical_pass_proxy_move(board_size, to_play) if isinstance(board_size, int) else None
    )

    proxy_row: dict[str, Any] | None = None
    if canonical_proxy is not None:
        for row in valid:
            if str(row.get("move") or "").lower() == canonical_proxy:
                proxy_row = row
                break
    mode = str(payload.get("mode") or "").strip().lower()
    if proxy_row is None and mode == "candidate":
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


def _hexata_main_path() -> Path:
    here = Path(__file__).resolve().parent
    return (here / ".." / "hexata" / "main.py").resolve()


def _build_cmd(exp: dict[str, Any], defaults: dict[str, Any], *, hexata_main: Path) -> list[str]:
    position = str(exp.get("position", "")).strip()
    if not position:
        raise ValueError("Experiment missing 'position'")
    candidates = exp.get("candidates", defaults.get("candidates"))
    candidate_tokens: list[str] | None = None
    if candidates is not None:
        if not isinstance(candidates, list):
            raise ValueError("'candidates' must be an array")
        toks = []
        for tok in candidates:
            s = str(tok).strip()
            if not s:
                continue
            if not CELL_RE.match(s):
                raise ValueError(f"Bad candidate token: {tok!r}")
            toks.append(s.lower())
        if not toks:
            raise ValueError("'candidates' provided but empty after trimming")
        candidate_tokens = toks

    if candidate_tokens is None:
        cmd = ["python3", str(hexata_main), "cli", "analyze", position]
    else:
        cmd = ["python3", str(hexata_main), "cli", "candidate", position, "--moves", ",".join(candidate_tokens)]
    search_seconds = exp.get("search_seconds", defaults.get("search_seconds"))
    has_search_budget = isinstance(search_seconds, (int, float))
    if candidate_tokens is None:
        if has_search_budget:
            cmd += ["--search-seconds", str(search_seconds)]
        awrn = exp.get("awrn", defaults.get("awrn"))
        if has_search_budget and awrn is not None:
            cmd += ["--awrn", str(awrn)]
    else:
        if has_search_budget:
            cmd += ["--total-search-seconds", str(search_seconds)]
    return cmd


def _run_once(cmd: list[str]) -> tuple[bool, dict[str, Any], str]:
    def fail(
        message: str,
        *,
        returncode: int,
        stderr: str,
        stdout: str | None = None,
    ) -> tuple[bool, dict[str, Any], str]:
        payload: dict[str, Any] = {"ok": False, "error": message, "returncode": returncode}
        if stdout is not None:
            payload["stdout"] = stdout
        if stderr:
            payload["stderr"] = stderr
            message = f"{message}; stderr: {stderr}"
        return False, payload, message

    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        msg = f"Launch failed: {exc}"
        return False, {"ok": False, "error": msg}, msg
    out = p.stdout.strip()
    err_out = p.stderr.strip()
    if not out:
        return fail(f"Empty stdout (returncode {p.returncode})", returncode=p.returncode, stderr=err_out)
    try:
        payload = json.loads(out)
    except json.JSONDecodeError as exc:
        return fail(f"Bad JSON: {exc}", returncode=p.returncode, stderr=err_out, stdout=out)
    if not isinstance(payload, dict):
        return fail("JSON payload is not an object", returncode=p.returncode, stderr=err_out)
    if err_out:
        payload.setdefault("stderr", err_out)
    payload.setdefault("ok", False)
    if p.returncode != 0:
        payload.setdefault("error", f"Return code {p.returncode}")
        msg = str(payload.get("error") or f"Return code {p.returncode}")
        return False, payload, msg
    if not payload.get("ok"):
        msg = str(payload.get("error") or "Run failed")
        return False, payload, msg
    return True, payload, ""


def _run_multi_position_analyze(
    *,
    hexata_main: Path,
    position_inputs: list[str],
) -> dict[str, dict[str, Any]]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    if not positions:
        return {}
    cmd = ["python3", str(hexata_main), "cli", "analyze", "-"]
    proc = subprocess.run(
        cmd,
        input="".join(f"{position}\n" for position in positions),
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or len(stdout_lines) != len(positions):
        raise ValueError(f"Multi-position analyze failed for {len(positions)} positions")
    records: dict[str, dict[str, Any]] = {}
    for position, line in zip(positions, stdout_lines):
        payload = json.loads(line)
        if not isinstance(payload, dict) or not bool(payload.get("ok")) or str(payload.get("input") or "").strip() != position:
            raise ValueError("Multi-position analyze returned a failed record")
        records[position] = payload
    return records


def _aggregate_moves(payload: dict[str, Any], *, position_input: str | None = None) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    first_idx = 0
    moves = payload.get("moves")
    position = payload.get("position") if isinstance(payload.get("position"), dict) else {}
    to_play = str(position.get("to_play") or "")
    if not isinstance(moves, list):
        return []
    for row in moves:
        if not isinstance(row, dict):
            continue
        move = str(row.get("move", "")).strip().lower()
        if not move:
            continue
        if move not in stats:
            stats[move] = {"idx": first_idx, "winrates": []}
            first_idx += 1
        wr = _row_side_to_play_winrate(row, to_play=to_play)
        if isinstance(wr, (int, float)):
            stats[move]["winrates"].append(float(wr))

    out = []
    for move, s in stats.items():
        wrs = s["winrates"]
        out.append(
            {
                "move": move,
                "n": len(wrs),
                "mean_winrate": mean(wrs) if wrs else None,
                "stdev_winrate": pstdev(wrs) if len(wrs) >= 2 else 0.0 if len(wrs) == 1 else None,
                "min_winrate": min(wrs) if wrs else None,
                "max_winrate": max(wrs) if wrs else None,
                "_idx": s["idx"],
            }
        )
    out.sort(
        key=lambda r: (
            -(r["mean_winrate"] if r["mean_winrate"] is not None else float("-inf")),
            r["_idx"],
            r["move"],
        )
    )
    _attach_stone_fractions(payload, out, position_input=position_input)
    return out


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
