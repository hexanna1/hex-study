#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pattern_output_utils as sout
import study_common as sc


DEFAULT_WORKERS = 2
DEFAULT_CATALOG_PAGE_SIDE = 10
_SPAWN_LOCK = threading.Lock()
_ACTIVE_PROCS_LOCK = threading.Lock()
_ACTIVE_PROCS: dict[int, subprocess.Popen[str]] = {}
_SHUTDOWN_REQUESTED = threading.Event()
_SHUTDOWN_SIGNAL: int | None = None
_CLEANUP_IN_PROGRESS = threading.Event()


class BatchInterrupted(Exception):
    pass


@dataclass(frozen=True)
class StudyTarget:
    candidate_Δ_max: int
    hexworld: str


STUDY_TARGETS: tuple[StudyTarget, ...] = (
    StudyTarget(candidate_Δ_max=12, hexworld="https://hexworld.org/board/#21c1,k11"),
    StudyTarget(candidate_Δ_max=12, hexworld="https://hexworld.org/board/#21c1,k11:p"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k10k11"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k12l10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k12k11l10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j13j11l10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k11k10k12"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,k11k10l12"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j12j13k10m10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j11j12k12l10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j12k13k10l11"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j11k11j12l10"),
    StudyTarget(candidate_Δ_max=3, hexworld="https://hexworld.org/board/#21c1,j11k10j12k11"),
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the curated pattern study batch and merge the tile outputs")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--catalog", default=None, help="Optional catalog.json from pattern_enumeration.py")
    ap.add_argument("--delta", type=int, default=None, help="Override candidate delta for all catalog-driven runs")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-png", action="store_true", help="Skip merged eval_maps contact-sheet PNG generation")
    return ap.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_out_dir(*, catalog_path: Path | None = None) -> Path:
    if isinstance(catalog_path, Path):
        return catalog_path.parent
    return Path("artifacts") / "study_batch"


def _load_catalog_payload(catalog_path: Path) -> dict[str, Any]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Catalog must be a JSON object: {catalog_path}")
    patterns = raw.get("patterns")
    if not isinstance(patterns, list):
        raise ValueError(f"Catalog missing patterns list: {catalog_path}")
    return raw


def _load_catalog_targets(catalog: dict[str, Any], *, catalog_path: Path) -> tuple[StudyTarget, ...]:
    patterns = catalog["patterns"]
    targets: list[StudyTarget] = []
    for row in patterns:
        if not isinstance(row, dict):
            continue
        hexworld = str(row.get("hexworld_21") or "").strip()
        if not hexworld:
            raise ValueError(f"Catalog row missing hexworld_21: {catalog_path}")
        row_candidate_Δ_max = row.get("candidate_Δ_max")
        if not isinstance(row_candidate_Δ_max, int):
            raise ValueError(f"Catalog row missing integer candidate_Δ_max: {catalog_path}")
        targets.append(StudyTarget(candidate_Δ_max=int(row_candidate_Δ_max), hexworld=hexworld))
    return tuple(targets)


def _target_slug(target: StudyTarget) -> str:
    movelist_slug = sout._movelist_slug_from_hexworld(target.hexworld)
    return f"d{int(target.candidate_Δ_max):02d}-{movelist_slug}"


def _target_json_path(*, out_dir: Path, target: StudyTarget) -> Path:
    return out_dir / "tiles" / f"{_target_slug(target)}.json"


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _eval_maps_dir(out_dir: Path) -> Path:
    return out_dir / "eval_maps"


def _fmt_s(sec: float) -> str:
    return sc._fmt_duration_compact(sec, subsecond_under_minute=True)


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_page_side(*, count: int, catalog_mode: bool) -> int:
    if catalog_mode:
        return DEFAULT_CATALOG_PAGE_SIDE
    return max(1, int(math.ceil(math.sqrt(max(0, int(count))))))


def _resolve_page_grid(*, count: int, catalog_mode: bool) -> tuple[int, int]:
    side = _default_page_side(count=count, catalog_mode=catalog_mode)
    return (side, side)


def _load_existing_tile_spec(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        spec = sout._load_local_map_spec_json(path)
    except Exception:
        return None
    pattern = str(spec.get("pattern") or "").strip()
    cells = spec.get("cells")
    if not pattern or not isinstance(cells, list):
        return None
    return spec


def _reset_shutdown_state() -> None:
    global _SHUTDOWN_SIGNAL
    _SHUTDOWN_SIGNAL = None
    _SHUTDOWN_REQUESTED.clear()
    _CLEANUP_IN_PROGRESS.clear()


def _register_active_proc(proc: subprocess.Popen[str]) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS[int(proc.pid)] = proc


def _unregister_active_proc(proc: subprocess.Popen[str]) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.pop(int(proc.pid), None)


def _terminate_active_procs() -> None:
    with _SPAWN_LOCK:
        with _ACTIVE_PROCS_LOCK:
            procs = list(_ACTIVE_PROCS.values())
    if not procs:
        return
    for proc in procs:
        try:
            if proc.poll() is None:
                os.killpg(int(proc.pid), signal.SIGTERM)
        except Exception:
            continue
    for proc in procs:
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    for proc in procs:
        try:
            if proc.poll() is None:
                os.killpg(int(proc.pid), signal.SIGKILL)
        except Exception:
            continue


def _interrupt_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN_SIGNAL
    if _CLEANUP_IN_PROGRESS.is_set():
        return
    _SHUTDOWN_SIGNAL = int(signum)
    _SHUTDOWN_REQUESTED.set()
    raise BatchInterrupted()


def _spawn_tracked_proc(*, cmd: list[str], cwd: str) -> subprocess.Popen[str]:
    with _SPAWN_LOCK:
        if _SHUTDOWN_REQUESTED.is_set():
            raise BatchInterrupted()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        _register_active_proc(proc)
        return proc


def _stream_proc_output(proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(line, end="")


def _run_one(*, index: int, target: StudyTarget, out_dir: Path, repo_root: Path) -> dict[str, Any]:
    output_json = _target_json_path(out_dir=out_dir, target=target)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing_spec = _load_existing_tile_spec(output_json)
    if isinstance(existing_spec, dict):
        completed_at = _timestamp_now()
        return {
            "index": int(index),
            "candidate_Δ_max": int(target.candidate_Δ_max),
            "hexworld": str(target.hexworld),
            "output_json": str(output_json),
            "returncode": 0,
            "ok": True,
            "pattern": str(existing_spec.get("pattern") or ""),
            "reused_existing": True,
            "runtime_seconds": 0.0,
            "completed_at": completed_at,
        }
    if output_json.exists():
        output_json.unlink()
    cmd = [
        "python3",
        "-u",
        str(repo_root / "pattern_study.py"),
        "--delta",
        str(int(target.candidate_Δ_max)),
        "--hexworld",
        str(target.hexworld),
        "--out-dir",
        str(output_json),
    ]
    started_at = time.time()
    proc = _spawn_tracked_proc(cmd=cmd, cwd=str(repo_root))
    try:
        _stream_proc_output(proc)
        proc.wait()
    finally:
        _unregister_active_proc(proc)
    runtime_seconds = max(0.0, time.time() - started_at)
    completed_at = _timestamp_now()
    result: dict[str, Any] = {
        "index": int(index),
        "candidate_Δ_max": int(target.candidate_Δ_max),
        "hexworld": str(target.hexworld),
        "output_json": str(output_json),
        "returncode": int(proc.returncode),
        "reused_existing": False,
        "runtime_seconds": float(runtime_seconds),
        "completed_at": completed_at,
    }
    if proc.returncode == 0 and output_json.exists():
        spec = sout._load_local_map_spec_json(output_json)
        result["ok"] = True
        result["pattern"] = str(spec.get("pattern") or "")
    else:
        result["ok"] = False
    return result


def _build_contact_sheet_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in sorted(results, key=lambda r: int(r.get("index") or 0)):
        if not bool(row.get("ok")):
            continue
        spec = sout._load_local_map_spec_json(Path(str(row["output_json"])))
        items.append(
            {
                "spec": spec,
                "title": str(spec.get("pattern") or ""),
            }
        )
    return items


def _write_manifest(
    *,
    out_dir: Path,
    targets: tuple[StudyTarget, ...],
    target_source: dict[str, Any],
    targets_total: int,
    targets_ok: int,
    png_pages: list[str],
    columns: int,
    rows: int,
) -> None:
    payload = {
        "target_source": target_source,
        "targets": [{"candidate_Δ_max": int(t.candidate_Δ_max), "hexworld": str(t.hexworld)} for t in targets],
        "targets_total": int(targets_total),
        "targets_ok": int(targets_ok),
        "columns": int(columns),
        "rows": int(rows),
        "png_page_count": int(len(png_pages)),
        "png_pages": png_pages,
    }
    path = _manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    workers = max(1, int(args.workers))
    repo_root = _repo_root()
    _reset_shutdown_state()
    catalog_path = Path(str(args.catalog)) if args.catalog else None
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(catalog_path=catalog_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if catalog_path is not None:
        delta = None if args.delta is None else int(args.delta)
        catalog = _load_catalog_payload(catalog_path)
        if delta is not None:
            for row in catalog["patterns"]:
                if isinstance(row, dict):
                    row["candidate_Δ_max"] = int(delta)
        runtime_catalog_path = out_dir / "catalog.json"
        runtime_catalog_path.write_text(json.dumps(catalog, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        targets = _load_catalog_targets(catalog, catalog_path=runtime_catalog_path)
        target_source = {"kind": "catalog", "catalog": str(runtime_catalog_path)}
    else:
        targets = STUDY_TARGETS
        target_source = {"kind": "manual", "count": len(targets)}
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    print(f"Running {len(targets)} study targets with {workers} workers")
    results: list[dict[str, Any]] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_map: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, StudyTarget]] = {}
    try:
        future_map = {
            pool.submit(_run_one, index=i, target=target, out_dir=out_dir, repo_root=repo_root): (i, target)
            for i, target in enumerate(targets, start=1)
        }
        for future in concurrent.futures.as_completed(future_map):
            if _SHUTDOWN_REQUESTED.is_set():
                raise BatchInterrupted()
            index, target = future_map[future]
            row = future.result()
            results.append(row)
            if bool(row.get("reused_existing")):
                status = "reuse"
            else:
                status = "ok" if bool(row.get("ok")) else "fail"
            pattern = str(row.get("pattern") or "")
            runtime_seconds = row.get("runtime_seconds")
            runtime_label = (
                f" runtime={_fmt_s(float(runtime_seconds))}"
                if (not bool(row.get("reused_existing")) and isinstance(runtime_seconds, (int, float)))
                else ""
            )
            completed_at = str(row.get("completed_at") or _timestamp_now())
            print(
                f"[{completed_at}] [{index}/{len(targets)}] {status}"
                f" {_target_slug(target)}"
                f" {pattern}"
                f"{runtime_label}".rstrip()
            )
    except BatchInterrupted:
        signum = _SHUTDOWN_SIGNAL
        if signum is None:
            print("Interrupted; terminating active study subprocesses...")
        else:
            print(f"Interrupted by signal {signum}; terminating active study subprocesses...")
        _SHUTDOWN_REQUESTED.set()
        _CLEANUP_IN_PROGRESS.set()
        try:
            _terminate_active_procs()
            for future in future_map:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            _CLEANUP_IN_PROGRESS.clear()
        raise SystemExit(130)
    finally:
        if not _SHUTDOWN_REQUESTED.is_set():
            pool.shutdown(wait=True, cancel_futures=False)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    results.sort(key=lambda r: int(r.get("index") or 0))
    items = _build_contact_sheet_items(results)
    columns, rows = _resolve_page_grid(
        count=len(items),
        catalog_mode=(catalog_path is not None),
    )
    png_pages: list[str] = []
    if items and not bool(args.no_png):
        png_names = sout.write_local_map_contact_sheet_pages(
            items,
            _eval_maps_dir(out_dir),
            columns=columns,
            rows_per_page=rows,
            suptitle_prefix=f"pattern study batch  total={len(items)}",
        )
        png_pages = [f"eval_maps/{name}" for name in png_names]
    _write_manifest(
        out_dir=out_dir,
        targets=targets,
        target_source=target_source,
        targets_total=len(results),
        targets_ok=sum(1 for row in results if bool(row.get("ok"))),
        png_pages=png_pages,
        columns=columns,
        rows=rows,
    )
    failed = [row for row in results if not bool(row.get("ok"))]

    if png_pages:
        print(str((out_dir / png_pages[0]).resolve()))
        return 1 if failed else 0
    print(f"Batch completed without merged image; failures={len(failed)} manifest={_manifest_path(out_dir).resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
