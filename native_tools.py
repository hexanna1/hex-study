"""Native position expansion and move normalization."""

from __future__ import annotations

import fcntl
import os
import subprocess
import tempfile
from pathlib import Path


def compile_cpp(source_name: str) -> Path:
    """Build a standalone helper once per source edit, sharing it across workers."""
    source = Path(__file__).resolve().with_name(source_name)
    build_dir = Path(tempfile.gettempdir()) / "hexwiki-native" / source.parent.relative_to(source.anchor)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / source.stem
    with binary.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.exists() and binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return binary
        pending = binary.with_suffix(f".{os.getpid()}.tmp")
        try:
            subprocess.run(
                ["c++", "-O3", "-std=c++17", str(source), "-o", str(pending)],
                capture_output=True, text=True, check=True,
            )
            pending.replace(binary)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compile {source.name}:\n{exc.stderr.strip()}") from None
        finally:
            pending.unlink(missing_ok=True)
    return binary


def expand_positions(input_lines: list[str], *, expected_rows: int) -> list[str]:
    proc = subprocess.run(
        [str(compile_cpp("position_expand.cpp"))],
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


def child_positions(
    *,
    requests: list[tuple[str, list[str]]],
    board_size: int,
) -> list[list[str]]:
    if not requests:
        return []
    input_lines = [f"children-v1\t{int(board_size)}"]
    for position, moves in requests:
        input_lines.append(f"{position}\t{';'.join(moves)}")
    output_lines = expand_positions(input_lines, expected_rows=len(requests))
    expanded = [line.split("\t") if line else [] for line in output_lines]
    for (_position, moves), children in zip(requests, expanded):
        if len(children) != len(moves):
            raise ValueError(
                f"Position expansion returned {len(children)} children for {len(moves)} moves"
            )
    return expanded


def normalize_moves(
    *,
    requests: list[tuple[str, list[str | None]]],
    board_size: int,
) -> list[list[str | None]]:
    if not requests:
        return []
    input_lines = [f"normalize-rules-v1\t{int(board_size)}"]
    for position, moves in requests:
        encoded = ";".join(move if isinstance(move, str) and move else "-" for move in moves)
        input_lines.append(f"{position}\t{encoded}")
    output_lines = expand_positions(input_lines, expected_rows=len(requests))
    normalized: list[list[str | None]] = []
    for (_position, moves), line in zip(requests, output_lines):
        raw = line.split(";") if line else []
        if len(raw) != len(moves):
            raise ValueError(
                f"Move normalization returned {len(raw)} moves for {len(moves)} candidates"
            )
        normalized.append([None if move == "-" else move for move in raw])
    return normalized
