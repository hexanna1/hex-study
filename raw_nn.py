"""KataHex batch evaluation and persistent NN payloads in integer millionths."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path
from collections.abc import Callable
from typing import Any

import hex_board as hb


def load_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes()) if path.exists() else {}


def save_cache(path: Path, cache: dict[str, Any], *, merge: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge:
        with path.with_suffix(path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = load_cache(path)
            for key, value in cache.items():
                # Child winrates must never erase a persisted root policy.
                if not isinstance(merged.get(key), dict) or isinstance(value, dict):
                    merged[key] = value
            _write_cache(path, merged)
    else:
        _write_cache(path, cache)


def _write_cache(path: Path, cache: dict[str, Any]) -> None:
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


def encode_millionths(value: float) -> int:
    return round(value * 1_000_000)


def decode_millionths(value: int) -> float:
    return value / 1_000_000


def _map_payload(payload: dict[str, Any], convert: Callable, *, include_moves: bool = True) -> dict[str, Any]:
    out = {}
    if "r" in payload:
        out["r"] = convert(payload["r"])
    if include_moves and payload.get("m"):
        out["m"] = [[move, convert(prior), *extra] for move, prior, *extra in payload["m"]]
    return out


def encode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _map_payload(payload, encode_millionths)


def decode_payload(payload: dict[str, Any], *, include_moves: bool = True) -> dict[str, Any]:
    return _map_payload(payload, decode_millionths, include_moves=include_moves)


def red_winrate(payload: dict[str, Any]) -> float | None:
    return payload.get("r")


def policy_moves(payload: dict[str, Any]) -> list[Any]:
    return payload.get("m", [])


def move_prior(row: list[Any]) -> tuple[str, float]:
    return row[0].strip().lower(), row[1]


def has_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and ("r" in payload or "m" in payload)


def has_winrate(payload: Any) -> bool:
    return isinstance(payload, dict) and "r" in payload


def has_policy(payload: Any) -> bool:
    return isinstance(payload, dict) and "m" in payload


def batch_command(*, board_size: int, move_limit: int) -> list[str]:
    home = Path.home()
    katago = Path(os.environ.get("HEXWIKI_KATAGO", home / "KataGomo" / "build-engine-42" / "katago")).expanduser()
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


def evaluate(
    *,
    position_inputs: list[str],
    board_size: int,
    move_limit: int,
    precanonicalized_position_inputs: bool = False,
) -> dict[str, dict[str, Any]]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    if not positions:
        return {}
    cache_key = hb.serialized_position_key if precanonicalized_position_inputs else hb.position_cache_key
    canonical_positions = [cache_key(position) for position in positions]
    proc = subprocess.run(
        batch_command(board_size=board_size, move_limit=move_limit),
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
        reduced = decode_payload(payload)
        if not has_payload(reduced):
            raise ValueError("Native raw-NN batch returned an invalid compact payload")
        records[position] = reduced
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"Native raw-NN batch exited with status {proc.returncode}{suffix}")
    return records


def ensure_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    board_size: int | None = None,
    raw_nn_cache_path: Path | None = None,
    chunk_size: int = 10000,
    move_limit: int | None = None,
    require_moves: bool = True,
    store_moves: bool = True,
    precanonicalized_position_inputs: bool = False,
    policy_validator: Callable[[str, dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    positions = [str(p).strip() for p in position_inputs if str(p).strip()]
    if not positions:
        return {}, 0
    board_size = board_size or hb.board_size_from_position(positions[0])
    move_limit = move_limit or board_size**2 + 1
    encoded_payloads: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    missing_positions: list[str] = []
    seen_missing: set[str] = set()
    cache_key_for = hb.serialized_position_key if precanonicalized_position_inputs else hb.position_cache_key
    for position in positions:
        cache_key = cache_key_for(position)
        cached = raw_nn_cache.get(cache_key)
        cache_covers_request = (
            has_policy(cached)
            and (policy_validator is None or policy_validator(position, cached))
            if require_moves
            else has_winrate(cached)
        )
        if cache_covers_request:
            encoded_payloads[position] = cached
            cache_hits += 1
            continue
        if position not in seen_missing:
            seen_missing.add(position)
            missing_positions.append(position)
    for i in range(0, len(missing_positions), int(chunk_size)):
        batch = missing_positions[i : i + int(chunk_size)]
        fetched = evaluate(
            position_inputs=batch,
            board_size=board_size,
            move_limit=move_limit,
            precanonicalized_position_inputs=precanonicalized_position_inputs,
        )
        for position, reduced in fetched.items():
            if not require_moves:
                winrate = red_winrate(reduced)
                if winrate is None:
                    raise ValueError(f"Raw-NN payload missing root winrate for {position!r}")
            encoded = encode_payload(reduced)
            encoded_payloads[position] = encoded
            raw_nn_cache[cache_key_for(position)] = (
                encoded
                if store_moves
                else {"r": encoded["r"]}
            )
        if isinstance(raw_nn_cache_path, Path):
            save_cache(raw_nn_cache_path, raw_nn_cache)
    return encoded_payloads, cache_hits


def evaluate_cached(
    *,
    position_inputs: list[str],
    raw_nn_cache: dict[str, dict[str, Any]],
    raw_nn_cache_path: Path | None = None,
    chunk_size: int = 10000,
    move_limit: int | None = None,
    include_moves: bool = True,
    store_moves: bool = True,
    policy_payloads_out: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    payloads, hits = ensure_cached(
        position_inputs=position_inputs, raw_nn_cache=raw_nn_cache,
        raw_nn_cache_path=raw_nn_cache_path, chunk_size=chunk_size, move_limit=move_limit,
        require_moves=include_moves, store_moves=store_moves,
        precanonicalized_position_inputs=True,
    )
    if policy_payloads_out is not None:
        policy_payloads_out.update(
            (position, payload) for position, payload in payloads.items() if has_policy(payload)
        )
    return {
        position: decode_payload(payload, include_moves=include_moves)
        for position, payload in payloads.items()
    }, hits


def cached_winrate_count(
    raw_nn_cache: dict[str, dict[str, Any]],
    positions: list[str],
) -> int:
    return sum(
        has_winrate(
            raw_nn_cache.get(hb.serialized_position_key(position))
        )
        for position in positions
    )
