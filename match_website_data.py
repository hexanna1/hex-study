from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from website_bundle_utils import (
    BundlePayload,
    cell_id_from_move,
    encode_thousandths,
    pack_optional_u10,
    write_hashed_bundle_manifest,
    write_uvarint,
)


MATCH_ARTIFACT_DIR = "match"
MATCH_OUT_NAME = "matches_current.json"
MATCH_BUNDLE_PREFIX = "matches"
BUNDLE_MAGIC = b"HMB1"
BUNDLE_VERSION = 1
BUNDLE_HEADER_STRUCT = struct.Struct("<4sHHI")
BUNDLE_GAME_HEADER_STRUCT = struct.Struct("<HBB")
BUNDLE_ANALYSIS_BEST_STRUCT = struct.Struct("<HH")
BUNDLE_CANDIDATE_STRUCT = struct.Struct("<HHH")
MOVE_NONE = 0
MOVE_PASS = 65534
MOVE_SWAP = 65535
RESULT_CODES = {
    "": 0,
    "red_resigned": 1,
    "blue_resigned": 2,
}
KIND_CODES = {
    "batch": 1,
    "match": 2,
}
MATCH_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "path": "20200722_kspttw_gzero.json",
        "games": (
            {
                "red": "kspttw",
                "blue": "gzero_bot",
                "url": "https://www.littlegolem.net/jsp/game/game.jsp?gid=2175191",
            },
        ),
    },
    {
        "path": "20170111_maciej_lazyplayer.json",
        "games": (
            {
                "red": "Maciej Celuch",
                "blue": "lazyplayer",
                "url": "https://www.littlegolem.net/jsp/game/game.jsp?gid=1806774",
            },
        ),
    },
    {
        "path": "20211129_hexanna_jaro04.json",
        "games": (
            {
                "red": "hexanna",
                "blue": "jaro04",
                "url": "https://www.littlegolem.net/jsp/game/game.jsp?gid=2273657",
            },
        ),
    },
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_artifacts_root() -> Path:
    return _repo_root() / "artifacts" / MATCH_ARTIFACT_DIR


def _default_out_path() -> Path:
    return _repo_root() / "docs" / "data" / MATCH_OUT_NAME


def _write_string(out: bytearray, value: Any) -> None:
    data = _clean_text(value).encode("utf-8")
    write_uvarint(out, len(data))
    out.extend(data)


def _pack_metric(value: Any) -> int:
    number = _finite_number(value)
    if number is None:
        return pack_optional_u10(None)
    if float(number) < 0.0 or float(number) > 1.0:
        raise ValueError(f"metric outside [0, 1] for binary bundle: {value!r}")
    return pack_optional_u10(encode_thousandths(number))


def _pack_move(move: Any, board_size: int) -> int:
    token = _clean_token(move)
    if not token:
        return MOVE_NONE
    if token == "pass":
        return MOVE_PASS
    if token == "swap":
        return MOVE_SWAP
    return 1 + cell_id_from_move(token, board_size=board_size)


def _pack_visit_count(value: Any) -> int:
    if value is None:
        return 0
    visits = _finite_number(value)
    if not isinstance(visits, int) or visits < 0:
        raise ValueError(f"bad visits payload for binary bundle: {value!r}")
    return visits + 1


def _artifact_path(*, artifacts_root: Path, source: dict[str, Any]) -> Path:
    raw_path = Path(str(source.get("path") or ""))
    if not raw_path.parts:
        raise ValueError(f"match source missing path: {source!r}")
    return raw_path if raw_path.is_absolute() else Path(artifacts_root) / raw_path


def _infer_board_size_from_hexworld(raw_url: Any) -> int:
    text = str(raw_url or "").strip()
    hash_text = text.split("#", 1)[1] if "#" in text else text
    match = re.match(r"([1-9][0-9]*)", hash_text)
    if match is None:
        raise ValueError(f"could not infer board size from hexworld URL: {raw_url!r}")
    return int(match.group(1))


def _clean_text(raw: Any) -> str:
    return str(raw or "").strip()


def _clean_token(raw: Any) -> str:
    return _clean_text(raw).lower()


def _finite_number(raw: Any) -> int | float | None:
    if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        return None
    return int(raw) if isinstance(raw, int) else float(raw)


def _optional_number(raw: dict[str, Any], key: str) -> int | float | None:
    if key not in raw:
        return None
    value = _finite_number(raw.get(key))
    if value is None:
        raise ValueError(f"bad numeric payload for {key}: {raw.get(key)!r}")
    return value


def _normalize_candidate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"bad candidate payload: {raw!r}")
    move = _clean_token(raw.get("move"))
    if not move:
        raise ValueError(f"candidate missing move: {raw!r}")
    out: dict[str, Any] = {"move": move}
    red_winrate = _optional_number(raw, "red_winrate")
    visits = _optional_number(raw, "visits")
    prior = _optional_number(raw, "prior")
    if red_winrate is not None:
        out["red_winrate"] = red_winrate
    if visits is not None:
        out["visits"] = visits
    if prior is not None:
        out["prior"] = prior
    return out


def _normalize_best(raw: Any, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = raw if isinstance(raw, dict) else {}
    move = _clean_token(best.get("move"))
    red_winrate = _finite_number(best.get("red_winrate"))
    if not move and candidates:
        move = str(candidates[0].get("move") or "")
    if red_winrate is None and candidates:
        red_winrate = _finite_number(candidates[0].get("red_winrate"))
    if not move or red_winrate is None:
        return None
    return {
        "move": move,
        "red_winrate": red_winrate,
    }


def _normalize_analyze(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"bad analysis payload: {raw!r}")
    moves_raw = raw.get("moves")
    if not isinstance(moves_raw, list):
        raise ValueError(f"analysis missing moves list: {raw!r}")
    candidates = [_normalize_candidate(row) for row in moves_raw]
    out: dict[str, Any] = {"moves": candidates}
    best = _normalize_best(raw.get("best"), candidates)
    if best is not None:
        out["best"] = best
    return out


def _normalize_ply(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"bad ply payload: {raw!r}")
    side = _clean_token(raw.get("side"))
    played = _clean_token(raw.get("played"))
    if not side:
        raise ValueError(f"ply missing side: {raw!r}")
    if not played:
        raise ValueError(f"ply missing played move: {raw!r}")
    out: dict[str, Any] = {
        "side": side,
        "played": played,
    }
    ply = _finite_number(raw.get("ply"))
    if isinstance(ply, int):
        out["ply"] = ply
    analyze = _normalize_analyze(raw.get("analyze"))
    if analyze is not None:
        out["analyze"] = analyze
    return out


def _normalize_final(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"bad final analysis payload: {raw!r}")
    side = _clean_token(raw.get("side"))
    if not side:
        raise ValueError(f"final analysis missing side: {raw!r}")
    out: dict[str, Any] = {
        "side": side,
    }
    analyze = _normalize_analyze(raw.get("analyze"))
    if analyze is not None:
        out["analyze"] = analyze
    return out


def _normalize_plies(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"plies must be a list: {raw!r}")
    return [_normalize_ply(row) for row in raw]


def _source_game_metadata(source: dict[str, Any], ordinal: int) -> dict[str, Any]:
    games = source.get("games")
    if not isinstance(games, (list, tuple)) or ordinal >= len(games):
        return {}
    row = games[ordinal]
    return dict(row) if isinstance(row, dict) else {}


def _add_optional_url(game: dict[str, Any], raw: Any) -> dict[str, Any]:
    url = _clean_text(raw)
    if url:
        game["url"] = url
    return game


def _normalize_match_game(payload: dict[str, Any], source_meta: dict[str, Any]) -> dict[str, Any]:
    raw_match = payload.get("match")
    if not isinstance(raw_match, dict):
        raise ValueError("match payload missing match object")
    meta = dict(source_meta)
    meta.update(raw_match)
    return _add_optional_url({
        "kind": "match",
        "board_size": _infer_board_size_from_hexworld(payload.get("hexworld")),
        "red": _clean_text(meta.get("red")),
        "blue": _clean_text(meta.get("blue")),
        "opening": _clean_text(meta.get("opening")),
        "result": _clean_token(meta.get("result")),
        "plies": _normalize_plies(raw_match.get("plies")),
        "final": _normalize_final(raw_match.get("final")),
    }, meta.get("url"))


def _normalize_batch_game(payload: dict[str, Any], source_meta: dict[str, Any]) -> dict[str, Any]:
    raw_batch = payload.get("batch")
    if not isinstance(raw_batch, dict):
        raise ValueError("batch payload missing batch object")
    result = raw_batch.get("result") or source_meta.get("result")
    return _add_optional_url({
        "kind": "batch",
        "board_size": _infer_board_size_from_hexworld(payload.get("hexworld")),
        "red": _clean_text(source_meta.get("red") or "Red"),
        "blue": _clean_text(source_meta.get("blue") or "Blue"),
        "opening": _clean_text(source_meta.get("opening")),
        "result": _clean_token(result),
        "plies": _normalize_plies(raw_batch.get("plies")),
        "final": _normalize_final(raw_batch.get("final")),
    }, source_meta.get("url"))


def build_match_index(*, artifacts_root: Path, sources: tuple[dict[str, Any], ...] = MATCH_SOURCES) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    for source in sources:
        artifact_path = _artifact_path(artifacts_root=artifacts_root, source=source)
        source_game_ordinal = 0
        for line_number, raw_line in enumerate(artifact_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                continue
            source_meta = _source_game_metadata(source, source_game_ordinal)
            if isinstance(payload.get("match"), dict):
                game = _normalize_match_game(payload, source_meta)
            elif isinstance(payload.get("batch"), dict):
                game = _normalize_batch_game(payload, source_meta)
            else:
                raise ValueError(f"unsupported match artifact row at {artifact_path}:{line_number}")
            source_game_ordinal += 1
            game["game_index"] = len(games) + 1
            games.append(game)
    return {
        "version": 1,
        "games": games,
    }


def _write_analysis(out: bytearray, raw: Any, board_size: int) -> None:
    analysis = raw if isinstance(raw, dict) else {}
    candidates = analysis.get("moves")
    if not isinstance(candidates, list):
        candidates = []
    best = analysis.get("best") if isinstance(analysis.get("best"), dict) else {}
    out.extend(BUNDLE_ANALYSIS_BEST_STRUCT.pack(
        _pack_move(best.get("move"), board_size),
        _pack_metric(best.get("red_winrate")),
    ))
    write_uvarint(out, len(candidates))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"bad candidate payload for binary bundle: {candidate!r}")
        out.extend(BUNDLE_CANDIDATE_STRUCT.pack(
            _pack_move(candidate.get("move"), board_size),
            _pack_metric(candidate.get("red_winrate")),
            _pack_metric(candidate.get("prior")),
        ))
        write_uvarint(out, _pack_visit_count(candidate.get("visits") if "visits" in candidate else None))


def _write_game(out: bytearray, game: dict[str, Any]) -> None:
    board_size = int(game.get("board_size") or 0)
    if board_size <= 0:
        raise ValueError(f"bad board size for binary bundle: {game.get('board_size')!r}")
    kind = _clean_token(game.get("kind"))
    if kind not in KIND_CODES:
        raise ValueError(f"bad game kind for binary bundle: {kind!r}")
    result = _clean_token(game.get("result"))
    if result not in RESULT_CODES:
        raise ValueError(f"bad game result for binary bundle: {result!r}")
    out.extend(BUNDLE_GAME_HEADER_STRUCT.pack(
        board_size,
        KIND_CODES[kind],
        RESULT_CODES[result],
    ))
    for key in ("red", "blue", "opening", "url"):
        _write_string(out, game.get(key))
    plies = game.get("plies")
    if not isinstance(plies, list):
        raise ValueError(f"bad game plies for binary bundle: {plies!r}")
    write_uvarint(out, len(plies))
    for ply in plies:
        if not isinstance(ply, dict):
            raise ValueError(f"bad ply payload for binary bundle: {ply!r}")
        out.extend(struct.pack("<H", _pack_move(ply.get("played"), board_size)))
        analysis = ply.get("analyze")
        has_analysis = isinstance(analysis, dict)
        out.append(1 if has_analysis else 0)
        if has_analysis:
            _write_analysis(out, analysis, board_size)
    final = game.get("final") if isinstance(game.get("final"), dict) else None
    final_analysis = final.get("analyze") if isinstance(final, dict) else None
    has_final_analysis = isinstance(final_analysis, dict)
    out.append(1 if has_final_analysis else 0)
    if has_final_analysis:
        _write_analysis(out, final_analysis, board_size)


def build_match_bundle(data: dict[str, Any]) -> bytes:
    games = data.get("games")
    if not isinstance(games, list):
        raise ValueError("match bundle data missing games list")
    out = bytearray(BUNDLE_HEADER_STRUCT.pack(
        BUNDLE_MAGIC,
        BUNDLE_VERSION,
        0,
        len(games),
    ))
    for game in games:
        if not isinstance(game, dict):
            raise ValueError(f"bad game payload for binary bundle: {game!r}")
        _write_game(out, game)
    return bytes(out)


def write_match_index(
    *,
    artifacts_root: Path,
    out_path: Path,
    sources: tuple[dict[str, Any], ...] = MATCH_SOURCES,
) -> Path:
    data = build_match_index(artifacts_root=artifacts_root, sources=sources)
    if not data["games"]:
        raise ValueError("no games found in match artifacts")
    payload = build_match_bundle(data)
    return write_hashed_bundle_manifest(
        out_path=out_path,
        bundles={"main": BundlePayload(prefix=MATCH_BUNDLE_PREFIX, payload=payload)},
        stale_globs=[f"{MATCH_BUNDLE_PREFIX}.*.bin"],
        manifest_from_bundle_names=lambda bundle_names: {
            "version": 1,
            "bundle": bundle_names["main"],
            "game_count": len(data["games"]),
        },
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build compact binary data for the match website")
    ap.add_argument("--artifacts-root", default=str(_default_artifacts_root()))
    ap.add_argument("--out", default=str(_default_out_path()))
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    write_match_index(
        artifacts_root=Path(str(args.artifacts_root)),
        out_path=Path(str(args.out)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
