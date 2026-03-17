#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pattern_output_utils as sout
from local_pattern_representative import _placement_offset, _serialize_position
from pattern_notation import LabeledPattern, canonicalize, format_pattern


Point = tuple[int, int]
DEFAULT_PNG_PATTERN_LIMIT = 2000
DEFAULT_MAX_MOVES = 5
DEFAULT_MAX_PAIR_DELTA = 13
DEFAULT_LOCAL_HEXHEX_SIDE = 7
DEFAULT_MULTI_STONE_CANDIDATE_Δ_CAP = 7
DEFAULT_PAGE_COLUMNS = 10
DEFAULT_PAGE_ROWS = 10
DEFAULT_CANDIDATE_Δ_FLOOR = 4
DEFAULT_STUDY_WORKERS = 2


def _move_cap_label(*, max_moves: int) -> str:
    return f"m{int(max_moves)}"


@dataclass(frozen=True)
class PatternRecord:
    pattern: str
    plus_count: int
    minus_count: int
    candidate_Δ_max: int
    to_play: str
    points_plus: tuple[Point, ...]
    points_minus: tuple[Point, ...]
    hexworld_21: str

    @property
    def visible_stone_count(self) -> int:
        return int(self.plus_count + self.minus_count)

    def to_json(self) -> dict[str, object]:
        return {
            "pattern": self.pattern,
            "plus": int(self.plus_count),
            "minus": int(self.minus_count),
            "candidate_Δ_max": int(self.candidate_Δ_max),
            "to_play": str(self.to_play),
            "hexworld_21": self.hexworld_21,
        }


def _delta(a: Point, b: Point) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return int(dq * dq + dq * dr + dr * dr)


def _is_connected(points: Iterable[Point], *, Δ_max: int) -> bool:
    pts = tuple(points)
    if not pts:
        return False
    if len(pts) == 1:
        return True
    todo = [pts[0]]
    seen = {pts[0]}
    all_pts = set(pts)
    while todo:
        cur = todo.pop()
        for other in all_pts:
            if other in seen:
                continue
            if _delta(cur, other) <= Δ_max:
                seen.add(other)
                todo.append(other)
    return len(seen) == len(all_pts)


def _max_pair_delta(points: Iterable[Point]) -> int:
    pts = tuple(points)
    if len(pts) <= 1:
        return 0
    best = 0
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            best = max(best, _delta(a, b))
    return int(best)


def _max_pair_hex_span(points: Iterable[Point]) -> int:
    pts = tuple(points)
    if len(pts) <= 1:
        return 0
    qs = [int(q) for q, _r in pts]
    rs = [int(r) for _q, r in pts]
    ss = [int(q) + int(r) for q, r in pts]
    return int(max(max(qs) - min(qs), max(rs) - min(rs), max(ss) - min(ss)))


def _hexhex_radius(side: int) -> int:
    side_i = int(side)
    if side_i <= 0 or side_i % 2 == 0:
        raise ValueError("local_hexhex_side must be a positive odd integer")
    return side_i - 1


def _first_outside_delta_for_slack(rho: int) -> int:
    if rho < 0:
        raise ValueError("rho must be >= 0")
    k = int(rho) + 1
    return int((3 * k * k + 3) // 4)


@lru_cache(maxsize=None)
def _largest_loschian_below(limit: int) -> int:
    limit_i = int(limit)
    if limit_i <= 0:
        raise ValueError("limit must be positive")
    best = 0
    coord_limit = math.isqrt(limit_i) + 2
    for q in range(-coord_limit, coord_limit + 1):
        for r in range(-coord_limit, coord_limit + 1):
            d = _delta((0, 0), (q, r))
            if d < limit_i:
                best = max(best, d)
    return int(best)


def _min_delta_within(points: Iterable[Point]) -> int | None:
    pts = tuple(points)
    if len(pts) < 2:
        return None
    best: int | None = None
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            d = _delta(a, b)
            best = d if best is None else min(best, d)
    return best


def _min_delta_between(a_points: Iterable[Point], b_points: Iterable[Point]) -> int | None:
    a_pts = tuple(a_points)
    b_pts = tuple(b_points)
    if not a_pts or not b_pts:
        return None
    best: int | None = None
    for a in a_pts:
        for b in b_pts:
            d = _delta(a, b)
            best = d if best is None else min(best, d)
    return best


def _passes_cross_color_closest_filter(red: Iterable[Point], blue: Iterable[Point]) -> bool:
    red_t = tuple(red)
    blue_t = tuple(blue)
    min_rb = _min_delta_between(red_t, blue_t)
    if min_rb is None:
        return True
    same_deltas = [d for d in (_min_delta_within(red_t), _min_delta_within(blue_t)) if d is not None]
    if same_deltas and min(same_deltas) == 1 and min_rb >= 4:
        return False
    return True


def _canonicalize_labeled(plus: Iterable[Point], minus: Iterable[Point]) -> LabeledPattern:
    return _canonicalize_labeled_cached(
        tuple(sorted(tuple(plus))),
        tuple(sorted(tuple(minus))),
    )


@lru_cache(maxsize=None)
def _canonicalize_labeled_cached(plus: tuple[Point, ...], minus: tuple[Point, ...]) -> LabeledPattern:
    parsed = LabeledPattern(
        plus=plus,
        minus=minus,
    )
    canon = canonicalize(parsed)
    if not isinstance(canon, LabeledPattern):
        raise AssertionError("labeled canonicalization unexpectedly produced unlabeled pattern")
    return canon


def _canonicalize_point_set(points: Iterable[Point]) -> tuple[Point, ...]:
    canon = _canonicalize_labeled_cached(tuple(sorted(tuple(points))), ())
    return tuple(canon.plus)


def _to_play_for_labeled_family(plus_count: int, minus_count: int) -> str:
    return "red" if int(plus_count) >= int(minus_count) else "blue"


def _moves_for_labeled_family(plus_count: int, minus_count: int) -> int:
    diff = int(minus_count) - int(plus_count)
    tenuki_used = diff in {-1, 2}
    return int(plus_count) + int(minus_count) + (1 if tenuki_used else 0)


def _candidate_Δ_max_for_pattern(
    *,
    plus: Iterable[Point],
    minus: Iterable[Point],
    candidate_Δ_floor: int,
    multi_stone_candidate_Δ_cap: int,
    max_pair_delta: int,
    local_hexhex_side: int,
) -> int:
    low = int(candidate_Δ_floor)
    multi_cap = int(multi_stone_candidate_Δ_cap)
    high = int(max_pair_delta)
    if low < 0:
        raise ValueError("candidate_Δ_floor must be >= 0")
    if multi_cap < 0:
        raise ValueError("multi_stone_candidate_Δ_cap must be >= 0")
    if high < 0:
        raise ValueError("max_pair_delta must be >= 0")
    if low > high:
        raise ValueError("candidate_Δ_floor must be <= max_pair_delta")

    points = tuple(plus) + tuple(minus)
    if not points:
        raise ValueError("pattern has no stones")

    radius = _hexhex_radius(local_hexhex_side)
    span_h = _max_pair_hex_span(points)
    if span_h > radius:
        raise ValueError(
            f"Pattern hex-ring span h={span_h} exceeds local hexhex radius {radius}"
        )

    rho = radius - span_h
    first_outside = _first_outside_delta_for_slack(rho)
    safe = _largest_loschian_below(first_outside)
    cap = high if len(points) == 1 else min(multi_cap, high)
    candidate = min(int(safe), int(cap))
    if candidate < low:
        raise ValueError(
            f"Pattern hex-ring span h={span_h} leaves candidate_Δ_max={candidate}, "
            f"below requested minimum {low}"
        )
    return int(candidate)


def _canonical_families(max_moves: int) -> list[tuple[int, int]]:
    family_order: list[tuple[int, int]] = []
    for total in range(1, int(max_moves) + 1):
        if total % 2 == 0:
            families = [(total // 2, total // 2)]
            if total >= 2:
                families.append(((total // 2) - 1, (total // 2) + 1))
        else:
            families = [
                ((total - 1) // 2, (total + 1) // 2),
                ((total + 1) // 2, (total - 1) // 2),
            ]
        for plus_count, minus_count in families:
            if _moves_for_labeled_family(plus_count, minus_count) <= int(max_moves):
                family_order.append((plus_count, minus_count))
    return family_order


def _centered_hexworld_url(
    plus_rel: Iterable[Point],
    minus_rel: Iterable[Point],
    *,
    to_play: str,
    board_size: int = 21,
) -> str:
    plus = tuple(sorted(tuple(plus_rel)))
    minus = tuple(sorted(tuple(minus_rel)))
    dq, dr = _placement_offset(
        plus_rel=plus,
        minus_rel=minus,
        board_size=int(board_size),
        placement="centered",
    )
    plus_abs = tuple(sorted(((q + dq, r + dr) for q, r in plus)))
    minus_abs = tuple(sorted(((q + dq, r + dr) for q, r in minus)))
    side = str(to_play).strip().lower()
    if side == "red":
        red_abs, blue_abs = plus_abs, minus_abs
    elif side == "blue":
        red_abs, blue_abs = minus_abs, plus_abs
    else:
        raise ValueError(f"Unsupported to_play value: {to_play!r}")
    return _serialize_position(
        board_size=int(board_size),
        red_cells=red_abs,
        blue_cells=blue_abs,
        to_play=str(to_play),
    )


def _all_pair_deltas_within(points: tuple[Point, ...], *, Δ_max: int) -> bool:
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            if _delta(a, b) > int(Δ_max):
                return False
    return True


def _anchored_span_domain(Δ_max: int) -> tuple[Point, ...]:
    if Δ_max < 0:
        raise ValueError("Δ_max must be >= 0")
    origin = (0, 0)
    limit = int(Δ_max)
    out: list[Point] = []
    for q in range(-limit, limit + 1):
        for r in range(-limit, limit + 1):
            point = (int(q), int(r))
            if point <= origin:
                continue
            if _delta(origin, point) <= int(Δ_max):
                out.append(point)
    out.sort()
    return tuple(out)


def _generate_max_span_geometries(*, total_stones: int, Δ_max: int) -> list[tuple[Point, ...]]:
    if total_stones <= 0:
        return []
    if total_stones == 1:
        return [((0, 0),)]

    seen: set[tuple[Point, ...]] = set()
    origin = (0, 0)
    for rest in combinations(_anchored_span_domain(int(Δ_max)), int(total_stones) - 1):
        geom = (origin,) + tuple(rest)
        if not _all_pair_deltas_within(geom, Δ_max=int(Δ_max)):
            continue
        seen.add(_canonicalize_point_set(geom))

    out = sorted(seen)
    for geom in out:
        if _max_pair_delta(geom) > int(Δ_max):
            raise AssertionError("generated geometry exceeds max pair delta")
        if not _is_connected(geom, Δ_max=int(Δ_max)):
            raise AssertionError("generated geometry is unexpectedly disconnected")
    return out


def enumerate_patterns(
    *,
    max_moves: int = DEFAULT_MAX_MOVES,
    max_pair_delta: int = DEFAULT_MAX_PAIR_DELTA,
    candidate_Δ_floor: int = DEFAULT_CANDIDATE_Δ_FLOOR,
    multi_stone_candidate_Δ_cap: int = DEFAULT_MULTI_STONE_CANDIDATE_Δ_CAP,
    local_hexhex_side: int = DEFAULT_LOCAL_HEXHEX_SIDE,
) -> dict[str, object]:
    max_moves = int(max_moves)
    if max_moves <= 0:
        raise ValueError("max_moves must be >= 1")
    if int(max_pair_delta) < 0:
        raise ValueError("max_pair_delta must be >= 0")
    if int(candidate_Δ_floor) < 0:
        raise ValueError("candidate_Δ_floor must be >= 0")
    if int(candidate_Δ_floor) > int(max_pair_delta):
        raise ValueError("candidate_Δ_floor must be <= max_pair_delta")
    if int(multi_stone_candidate_Δ_cap) < 0:
        raise ValueError("multi_stone_candidate_Δ_cap must be >= 0")
    _hexhex_radius(local_hexhex_side)

    family_order = _canonical_families(max_moves)

    geometries_by_total: dict[int, list[tuple[Point, ...]]] = {}
    for total in sorted({plus + minus for plus, minus in family_order}):
        geometries_by_total[total] = _generate_max_span_geometries(
            total_stones=total,
            Δ_max=int(max_pair_delta),
        )

    records: list[PatternRecord] = []
    counts_by_family: list[dict[str, object]] = []
    for plus_count, minus_count in family_order:
        to_play = _to_play_for_labeled_family(plus_count, minus_count)
        total = int(plus_count + minus_count)
        seen: dict[str, PatternRecord] = {}
        for geom in geometries_by_total[total]:
            for plus_idx in combinations(range(total), plus_count):
                plus_pos = set(plus_idx)
                plus = tuple(geom[i] for i in range(total) if i in plus_pos)
                minus = tuple(geom[i] for i in range(total) if i not in plus_pos)
                if not _passes_cross_color_closest_filter(plus, minus):
                    continue
                canon = _canonicalize_labeled(plus, minus)
                pattern = format_pattern(canon)
                if pattern in seen:
                    continue
                seen[pattern] = PatternRecord(
                    pattern=pattern,
                    plus_count=int(plus_count),
                    minus_count=int(minus_count),
                    candidate_Δ_max=_candidate_Δ_max_for_pattern(
                        plus=canon.plus,
                        minus=canon.minus,
                        candidate_Δ_floor=candidate_Δ_floor,
                        multi_stone_candidate_Δ_cap=multi_stone_candidate_Δ_cap,
                        max_pair_delta=int(max_pair_delta),
                        local_hexhex_side=int(local_hexhex_side),
                    ),
                    to_play=str(to_play),
                    points_plus=tuple(canon.plus),
                    points_minus=tuple(canon.minus),
                    hexworld_21=_centered_hexworld_url(canon.plus, canon.minus, to_play=to_play, board_size=21),
                )
        family_records = sorted(seen.values(), key=lambda rec: rec.pattern)
        counts_by_family.append(
            {
                "plus": int(plus_count),
                "minus": int(minus_count),
                "to_play": str(to_play),
                "count": int(len(family_records)),
            }
        )
        records.extend(family_records)

    return {
        "max_moves": int(max_moves),
        "max_pair_delta": int(max_pair_delta),
        "candidate_Δ_floor": int(candidate_Δ_floor),
        "multi_stone_candidate_Δ_cap": int(multi_stone_candidate_Δ_cap),
        "local_hexhex_side": int(local_hexhex_side),
        "total_patterns": int(len(records)),
        "counts_by_family": counts_by_family,
        "patterns": [rec.to_json() for rec in records],
    }


def _default_artifact_dir(
    *,
    max_moves: int,
    candidate_Δ_floor: int,
    max_pair_delta: int,
) -> Path:
    stem = (
        f"interior_patterns_{_move_cap_label(max_moves=max_moves)}"
        f"_d{int(candidate_Δ_floor)}_span{int(max_pair_delta)}"
    )
    return Path("artifacts") / stem


def _catalog_move_cap_label(catalog: dict[str, object]) -> str:
    total = catalog.get("max_moves")
    if isinstance(total, int):
        return _move_cap_label(max_moves=total)
    return "?"


def _minimal_square_board_layout(points: Iterable[Point]) -> tuple[int, dict[Point, Point]]:
    return sout._minimal_square_board_layout(list(points))


def write_catalog_json(catalog: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_catalog_png_pages(
    catalog: dict[str, object],
    out_dir: Path,
    *,
    columns: int = DEFAULT_PAGE_COLUMNS,
    rows_per_page: int = DEFAULT_PAGE_ROWS,
) -> list[str]:
    patterns = catalog.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return []
    items = []
    for record in patterns:
        if not isinstance(record, dict):
            continue
        pattern = str(record.get("pattern") or "").strip()
        to_play = str(record.get("to_play") or "").strip().lower()
        if not pattern:
            continue
        if to_play not in {"red", "blue"}:
            raise ValueError("Catalog PNG rendering requires explicit to_play per pattern")
        items.append(
            {
                "spec": sout.build_local_map_spec_from_pattern(pattern, to_play=to_play),
                "title": pattern,
            }
        )
    return sout.write_local_map_contact_sheet_pages(
        items,
        out_dir,
        columns=int(columns),
        rows_per_page=int(rows_per_page),
        suptitle_prefix=(
            "Interior Patterns"
            f"  cap={_catalog_move_cap_label(catalog)}"
            f"  span={catalog.get('max_pair_delta')}"
            f"  hex={catalog.get('local_hexhex_side')}"
            f"  total={int(catalog.get('total_patterns', 0))}"
        ),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_study_batch_from_catalog(
    *,
    catalog_path: Path,
    out_dir: Path,
    workers: int,
    no_png: bool,
) -> dict[str, object]:
    cmd = [
        "python3",
        str(_repo_root() / "pattern_study_batch.py"),
        "--catalog",
        str(catalog_path),
        "--workers",
        str(int(workers)),
        "--out-dir",
        str(out_dir),
    ]
    if bool(no_png):
        cmd.append("--no-png")
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        start_new_session=True,
    )
    try:
        returncode = int(proc.wait())
    except KeyboardInterrupt:
        try:
            os.killpg(int(proc.pid), signal.SIGINT)
        except Exception:
            pass
        try:
            returncode = int(proc.wait())
        except KeyboardInterrupt:
            try:
                if proc.poll() is None:
                    os.killpg(int(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            raise SystemExit(130)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                os.killpg(int(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        raise SystemExit(130)
    manifest_path = out_dir / "manifest.json"
    manifest_summary: dict[str, object] | None = None
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            manifest_summary = {
                "targets_total": raw.get("targets_total"),
                "targets_ok": raw.get("targets_ok"),
                "png_page_count": raw.get("png_page_count"),
                "png_pages": raw.get("png_pages"),
            }
    return {
        "ok": returncode == 0,
        "returncode": int(returncode),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest_summary,
        "command": cmd,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Enumerate canonical interior local patterns")
    ap.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    ap.add_argument("--max-pair-delta", type=int, default=DEFAULT_MAX_PAIR_DELTA)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--force-png", action="store_true")
    ap.add_argument("--run-study", action="store_true")
    ap.add_argument("--study-workers", type=int, default=DEFAULT_STUDY_WORKERS)
    ap.add_argument("--study-out-dir", default=None)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = (
        Path(str(args.out_dir))
        if args.out_dir
        else _default_artifact_dir(
            max_moves=int(args.max_moves),
            candidate_Δ_floor=DEFAULT_CANDIDATE_Δ_FLOOR,
            max_pair_delta=int(args.max_pair_delta),
        )
    )
    out_json = out_dir / "catalog.json"
    catalog = enumerate_patterns(
        max_moves=int(args.max_moves),
        max_pair_delta=int(args.max_pair_delta),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_catalog_json(catalog, out_json)
    total_patterns = int(catalog["total_patterns"])

    png_pages: list[str] = []
    png_skipped_reason: str | None = None
    if args.no_png:
        png_skipped_reason = "disabled by --no-png"
    elif total_patterns > DEFAULT_PNG_PATTERN_LIMIT and not args.force_png:
        png_skipped_reason = (
            f"skipped by default because total_patterns={total_patterns} exceeds "
            f"png_limit={DEFAULT_PNG_PATTERN_LIMIT}; rerun with --force-png"
        )
    else:
        png_pages = write_catalog_png_pages(catalog, out_dir)
        if not png_pages:
            png_skipped_reason = "matplotlib unavailable or render failed"

    study_result: dict[str, object] | None = None
    if args.run_study:
        study_out_dir = (
            Path(str(args.study_out_dir))
            if args.study_out_dir
            else out_dir
        )
        study_result = _run_study_batch_from_catalog(
            catalog_path=out_json,
            out_dir=study_out_dir,
            workers=int(args.study_workers),
            no_png=bool(args.no_png),
        )

    print(
        json.dumps(
            {
                "ok": (study_result is None or bool(study_result.get("ok"))),
                "max_moves": int(args.max_moves),
                "max_pair_delta": catalog["max_pair_delta"],
                "local_hexhex_side": catalog["local_hexhex_side"],
                "total_patterns": total_patterns,
                "out_dir": str(out_dir),
                "out_json": str(out_json),
                "png_page_count": int(len(png_pages)),
                "png_pages": png_pages,
                "png_skipped_reason": png_skipped_reason,
                "study": study_result,
            },
            ensure_ascii=True,
        )
    )
    return 0 if study_result is None or bool(study_result.get("ok")) else int(study_result.get("returncode") or 1)


if __name__ == "__main__":
    raise SystemExit(main())
