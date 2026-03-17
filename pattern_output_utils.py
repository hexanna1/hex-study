from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from hex_symmetry import apply_transform_ax, inverse_transform_id
import study_common as lps
from pattern_notation import LabeledPattern, canonicalize, format_pattern, parse_pattern


def _load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _redact_personal_text(text: str) -> str:
    s = str(text)
    s = re.sub(r"/Users/[^/\s]+", "/Users/$USER", s)
    s = re.sub(r"/home/[^/\s]+", "/home/$USER", s)
    home = str(Path.home())
    if home and home != "/" and s:
        s = s.replace(home, "~")
    return s


def _redact_personal_obj(v: Any) -> Any:
    if isinstance(v, str):
        return _redact_personal_text(v)
    if isinstance(v, list):
        return [_redact_personal_obj(x) for x in v]
    if isinstance(v, tuple):
        return [_redact_personal_obj(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _redact_personal_obj(val) for k, val in v.items()}
    return v


def _movelist_slug_from_hexworld(hexworld: str) -> str:
    s = str(hexworld or "").strip().lower()
    frag = s.split("#", 1)[1] if "#" in s else s
    movelist = frag.split(",", 1)[1] if "," in frag else frag
    movelist = movelist.replace(":", "_")
    movelist = re.sub(r"[^a-z0-9_]", "", movelist)
    return movelist or "nomoves"


def _axial_to_xy(col: int, row: int) -> tuple[float, float]:
    x = (col - 1) + 0.5 * (row - 1)
    y = (3.0**0.5 / 2.0) * (row - 1)
    return x, y


def _parse_geometric_local_key(key: str) -> tuple[int, int] | None:
    try:
        q_s, r_s = str(key).split(",", 1)
        return int(q_s), int(r_s)
    except Exception:
        return None


def _abs_point_to_base_rel(col: int, row: int, exp_meta: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(exp_meta, dict):
        return None
    try:
        transform_id = int(exp_meta["orientation_transform_id"])
        shift = exp_meta["orientation_norm_shift"]
        offset = exp_meta["placement_offset"]
        shift_q, shift_r = int(shift[0]), int(shift[1])
        dq, dr = int(offset[0]), int(offset[1])
    except Exception:
        return None
    ori_rel = (int(col) - dq, int(row) - dr)
    unnorm = (ori_rel[0] + shift_q, ori_rel[1] + shift_r)
    inv_id = inverse_transform_id(transform_id)
    base_rel = apply_transform_ax(unnorm, inv_id)
    return int(base_rel[0]), int(base_rel[1])


def _minimal_square_board_layout(points: list[tuple[int, int]]) -> tuple[int, dict[tuple[int, int], tuple[int, int]]]:
    if not points:
        raise ValueError("board layout requires at least one point")
    min_q = min(q for q, _r in points)
    max_q = max(q for q, _r in points)
    min_r = min(r for _q, r in points)
    max_r = max(r for _q, r in points)
    width = int(max_q - min_q + 1)
    height = int(max_r - min_r + 1)
    board_size = max(width, height)
    shift_q = 1 - min_q + ((board_size - width) // 2)
    shift_r = 1 - min_r + ((board_size - height) // 2)
    mapped = {
        (int(q), int(r)): (int(q + shift_q), int(r + shift_r))
        for q, r in points
    }
    return int(board_size), mapped


def _pattern_points_from_rep(
    *,
    first_rep: Any,
    first_exp_meta: dict[str, Any] | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pattern_plus_rel: list[tuple[int, int]] = []
    pattern_minus_rel: list[tuple[int, int]] = []
    for col, row in sorted(set(getattr(first_rep, "plus_abs", ()))):
        rel = _abs_point_to_base_rel(int(col), int(row), first_exp_meta)
        if rel is not None:
            pattern_plus_rel.append(rel)
    for col, row in sorted(set(getattr(first_rep, "minus_abs", ()))):
        rel = _abs_point_to_base_rel(int(col), int(row), first_exp_meta)
        if rel is not None:
            pattern_minus_rel.append(rel)
    return pattern_plus_rel, pattern_minus_rel


def _pattern_points_from_spec(pattern: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    parsed = parse_pattern(pattern)
    if not isinstance(parsed, LabeledPattern):
        raise ValueError("Local map pattern must use labeled notation")
    return list(parsed.plus), list(parsed.minus)


def build_local_map_spec_from_pattern(
    pattern: str,
    *,
    to_play: str = "red",
) -> dict[str, Any]:
    pattern_text = str(pattern or "").strip()
    if not pattern_text:
        raise ValueError("pattern must be non-empty")
    _plus, _minus = _pattern_points_from_spec(pattern_text)
    return {
        "pattern": pattern_text,
        "to_play": str(to_play or "red").strip().lower() or "red",
        "cells": [],
    }


def _normalize_labeled_points(
    plus: list[tuple[int, int]],
    minus: list[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, int]]:
    all_pts = plus + minus
    anchor = min(all_pts) if all_pts else (0, 0)
    plus_n = tuple(sorted((int(q - anchor[0]), int(r - anchor[1])) for q, r in plus))
    minus_n = tuple(sorted((int(q - anchor[0]), int(r - anchor[1])) for q, r in minus))
    return plus_n, minus_n, (int(anchor[0]), int(anchor[1]))


def _canonicalize_labeled_points(
    plus: list[tuple[int, int]],
    minus: list[tuple[int, int]],
) -> tuple[str, tuple[int, int], int]:
    parsed = LabeledPattern(plus=tuple(plus), minus=tuple(minus))
    canonical_pattern = canonicalize(parsed)
    if not isinstance(canonical_pattern, LabeledPattern):
        raise ValueError("Local map pattern must canonicalize to labeled notation")
    canonical_key = (tuple(canonical_pattern.plus), tuple(canonical_pattern.minus))
    for transform_id in range(12):
        plus_t = [apply_transform_ax(p, transform_id) for p in plus]
        minus_t = [apply_transform_ax(p, transform_id) for p in minus]
        plus_n, minus_n, anchor = _normalize_labeled_points(plus_t, minus_t)
        if (plus_n, minus_n) == canonical_key:
            return format_pattern(canonical_pattern), anchor, int(transform_id)
    raise ValueError("Failed to recover canonical local frame")


def _canonicalize_local_point(
    point: tuple[int, int],
    *,
    transform_id: int,
    anchor: tuple[int, int],
) -> tuple[int, int]:
    q_t, r_t = apply_transform_ax((int(point[0]), int(point[1])), int(transform_id))
    return int(q_t - anchor[0]), int(r_t - anchor[1])


def _build_local_map_spec(
    *,
    first_rep: Any,
    first_exp_meta: dict[str, Any] | None,
    pooled_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(first_exp_meta, dict):
        return None

    pattern_plus_rel, pattern_minus_rel = _pattern_points_from_rep(
        first_rep=first_rep,
        first_exp_meta=first_exp_meta,
    )
    pattern_text, canonical_anchor, canonical_transform_id = _canonicalize_labeled_points(
        pattern_plus_rel,
        pattern_minus_rel,
    )

    geometric_cells: list[dict[str, Any]] = []
    tenuki_cells: list[dict[str, Any]] = []
    for prow in pooled_rows:
        key = str(prow.get("candidate_key_local") or "")
        if not key or key == "pass_proxy":
            continue
        base = {
            "kind": ("tenuki" if key == "tenuki" else "local"),
            "key": key,
            "stone_fraction": float(prow.get("mean_stone_fraction") or 0.0),
            "rank": int(prow.get("rank") or 0),
        }
        if key == "tenuki":
            tenuki_cells.append(base)
            continue
        rel = _parse_geometric_local_key(key)
        if rel is None:
            continue
        rel_can = _canonicalize_local_point(
            rel,
            transform_id=canonical_transform_id,
            anchor=canonical_anchor,
        )
        geometric_cells.append({**base, "local_rel": [int(rel_can[0]), int(rel_can[1])]})

    basis_points = list(pattern_plus_rel) + list(pattern_minus_rel) + [
        (int(cell["local_rel"][0]), int(cell["local_rel"][1])) for cell in geometric_cells
    ]
    if not basis_points:
        return None
    cells = sorted(
        geometric_cells + tenuki_cells,
        key=lambda row: (int(row.get("rank") or 0), str(row.get("key") or "")),
    )
    return {
        "pattern": pattern_text,
        "to_play": str(getattr(first_rep, "to_play_at_cursor", "") or ""),
        "cells": cells,
    }


def _write_local_map_spec_json(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_local_map_spec_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Local map spec must be a JSON object: {path}")
    return raw


def _local_map_title_color(spec: dict[str, Any]) -> Any:
    to_play = str(spec.get("to_play") or "").strip().lower()
    if to_play == "red":
        return (220 / 255.0, 60 / 255.0, 60 / 255.0)
    if to_play == "blue":
        return (40 / 255.0, 100 / 255.0, 220 / 255.0)
    return "#111111"


def _draw_local_map_spec(
    ax: Any,
    spec: dict[str, Any],
    *,
    title: str | None = None,
    footer: str | None = None,
    show_cell_text: bool = True,
    cell_text_fontsize: float = 6.5,
    tenuki_text_fontsize: float = 6.0,
) -> bool:
    try:
        from matplotlib.patches import Polygon
    except Exception:
        return False
    pattern_text = str(spec.get("pattern") or "").strip()
    if not pattern_text:
        return False
    pattern_plus_rel, pattern_minus_rel = _pattern_points_from_spec(pattern_text)
    to_play = str(spec.get("to_play") or "").strip().lower()
    if to_play == "blue":
        pattern_red_rel = list(pattern_minus_rel)
        pattern_blue_rel = list(pattern_plus_rel)
    else:
        pattern_red_rel = list(pattern_plus_rel)
        pattern_blue_rel = list(pattern_minus_rel)
    geometric_candidates = [
        {**cell, "local_rel": tuple(int(x) for x in cell["local_rel"])}
        for cell in list(spec.get("cells") or [])
        if isinstance(cell, dict) and str(cell.get("kind") or "") == "local" and isinstance(cell.get("local_rel"), list)
    ]
    tenuki_row = next(
        (cell for cell in list(spec.get("cells") or []) if isinstance(cell, dict) and str(cell.get("kind") or "") == "tenuki"),
        None,
    )
    basis_points = list(pattern_red_rel) + list(pattern_blue_rel) + [tuple(r["local_rel"]) for r in geometric_candidates]
    if not basis_points:
        return False
    board_size, mapped = _minimal_square_board_layout(basis_points)

    # Keep labels readable on small boards and shrink them on larger ones.
    size_scale = (3.0 / float(board_size)) ** 0.5 if board_size > 0 else 1.0
    cell_text_fontsize_eff = max(4.2, min(float(cell_text_fontsize), float(cell_text_fontsize) * size_scale))
    tenuki_text_fontsize_eff = max(4.0, min(float(tenuki_text_fontsize), float(tenuki_text_fontsize) * size_scale))

    RED = (220 / 255.0, 60 / 255.0, 60 / 255.0)
    BLUE = (40 / 255.0, 100 / 255.0, 220 / 255.0)
    OFF_WHITE = (246 / 255.0, 241 / 255.0, 232 / 255.0)
    GRID_EDGE = (182 / 255.0, 182 / 255.0, 182 / 255.0)
    CANDIDATE_LOW = (244 / 255.0, 232 / 255.0, 250 / 255.0)
    CANDIDATE_HIGH = (170 / 255.0, 125 / 255.0, 210 / 255.0)

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def lerp_rgb(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
        tt = clamp01(t)
        return (
            a[0] + (b[0] - a[0]) * tt,
            a[1] + (b[1] - a[1]) * tt,
            a[2] + (b[2] - a[2]) * tt,
        )

    xs: list[float] = []
    ys: list[float] = []
    cell_verts: dict[tuple[int, int], tuple[float, float, list[tuple[float, float]]]] = {}
    hex_r = 1.0 / math.sqrt(3.0)
    corner_deg = [90, 30, -30, -90, -150, 150]
    for row in range(1, board_size + 1):
        for col in range(1, board_size + 1):
            cx, cy = _axial_to_xy(col, row)
            xs.append(cx)
            ys.append(cy)
            verts = []
            for deg in corner_deg:
                ang = math.radians(deg)
                verts.append((cx + hex_r * math.cos(ang), cy + hex_r * math.sin(ang)))
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=OFF_WHITE,
                    edgecolor=GRID_EDGE,
                    linewidth=0.55,
                    zorder=1,
                )
            )
            cell_verts[(col, row)] = (cx, cy, verts)

    for row in geometric_candidates:
        rel = tuple(row["local_rel"])
        mapped_cell = mapped.get(rel)
        if mapped_cell is None:
            continue
        geom = cell_verts.get(mapped_cell)
        if not geom:
            continue
        _cx, _cy, verts = geom
        frac_raw = float(row["stone_fraction"])
        frac = max(0.0, min(1.0, frac_raw))
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=lerp_rgb(CANDIDATE_LOW, CANDIDATE_HIGH, frac**0.9),
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=2,
            )
        )

    for rel in pattern_red_rel:
        mapped_cell = mapped.get(rel)
        geom = cell_verts.get(mapped_cell) if mapped_cell is not None else None
        if geom:
            _cx, _cy, verts = geom
            ax.add_patch(Polygon(verts, closed=True, facecolor=RED, edgecolor="none", linewidth=0.0, zorder=3))
    for rel in pattern_blue_rel:
        mapped_cell = mapped.get(rel)
        geom = cell_verts.get(mapped_cell) if mapped_cell is not None else None
        if geom:
            _cx, _cy, verts = geom
            ax.add_patch(Polygon(verts, closed=True, facecolor=BLUE, edgecolor="none", linewidth=0.0, zorder=3))

    if show_cell_text:
        for row in geometric_candidates:
            rel = tuple(row["local_rel"])
            mapped_cell = mapped.get(rel)
            geom = cell_verts.get(mapped_cell) if mapped_cell is not None else None
            if not geom:
                continue
            cx, cy, _verts = geom
            percent = 100.0 * float(row["stone_fraction"])
            ax.text(
                cx,
                cy,
                f"{percent:.1f}",
                ha="center",
                va="center",
                fontsize=cell_text_fontsize_eff,
                color="#111111",
                zorder=4,
            )

    if isinstance(tenuki_row, dict):
        mid_row = max(1, (board_size + 1) // 2)
        tx, ty = _axial_to_xy(-1, mid_row)
        tenuki_verts = []
        for deg in corner_deg:
            ang = math.radians(deg)
            tenuki_verts.append((tx + hex_r * math.cos(ang), ty + hex_r * math.sin(ang)))
        frac_raw = float(tenuki_row["stone_fraction"])
        frac = max(0.0, min(1.0, frac_raw))
        ax.add_patch(
            Polygon(
                tenuki_verts,
                closed=True,
                facecolor=lerp_rgb(CANDIDATE_LOW, CANDIDATE_HIGH, frac**0.9),
                edgecolor=GRID_EDGE,
                linewidth=0.7,
                zorder=2,
            )
        )
        if show_cell_text:
            ax.text(
                tx,
                ty,
                f"{100.0 * frac_raw:.1f}",
                ha="center",
                va="center",
                fontsize=tenuki_text_fontsize_eff,
                color="#111111",
                zorder=4,
            )
        xs.append(tx)
        ys.append(ty)

    if title:
        ax.text(
            0.5,
            1.02,
            str(title),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=_local_map_title_color(spec),
            transform=ax.transAxes,
        )
    if footer:
        ax.text(0.5, -0.10, str(footer), ha="center", va="top", fontsize=6.5, transform=ax.transAxes)

    min_x, max_x = min(xs) - 0.95, max(xs) + 0.95
    min_y, max_y = min(ys) - 0.95, max(ys) + 0.95
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect("equal")
    ax.axis("off")
    return True


def write_local_map_contact_sheet(
    items: list[dict[str, Any]],
    out_path: Path,
    *,
    columns: int = 4,
    suptitle: str | None = None,
) -> bool:
    if not items:
        return False
    plt = _load_matplotlib_pyplot()
    if plt is None:
        return False

    cols = max(1, int(columns))
    rows = max(1, int(math.ceil(len(items) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2))
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]
    for ax in axes_list:
        ax.set_axis_off()
        ax.set_aspect("equal")

    for ax, item in zip(axes_list, items):
        spec = item.get("spec")
        if not isinstance(spec, dict):
            ax.set_visible(False)
            continue
        _draw_local_map_spec(
            ax,
            spec,
            title=str(item.get("title") or ""),
            footer=str(item.get("footer") or ""),
        )

    for ax in axes_list[len(items) :]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(str(suptitle), fontsize=14, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def write_local_map_contact_sheet_pages(
    items: list[dict[str, Any]],
    out_dir: Path,
    *,
    columns: int,
    rows_per_page: int,
    suptitle_prefix: str | None = None,
) -> list[str]:
    if not items:
        return []
    cols = max(1, int(columns))
    page_rows = max(1, int(rows_per_page))
    per_page = cols * page_rows
    page_count = int(math.ceil(len(items) / per_page))
    out_dir.mkdir(parents=True, exist_ok=True)
    page_names: list[str] = []
    for page_index in range(page_count):
        page_items = items[page_index * per_page : (page_index + 1) * per_page]
        page_name = f"{page_index + 1:03d}.png"
        suptitle = None
        if suptitle_prefix:
            suptitle = f"{suptitle_prefix}  page={page_index + 1}/{page_count}"
        ok = write_local_map_contact_sheet(
            page_items,
            out_dir / page_name,
            columns=cols,
            suptitle=suptitle,
        )
        if not ok:
            return []
        page_names.append(page_name)
    return page_names


def _write_local_pooled_map_artifacts(
    *,
    out_dir: Path,
    first_rep: Any,
    first_exp_meta: dict[str, Any] | None,
    pooled_rows: list[dict[str, Any]],
    file_suffix: str = "",
    output_png_name: str | None = None,
) -> dict[str, str]:
    spec = _build_local_map_spec(
        first_rep=first_rep,
        first_exp_meta=first_exp_meta,
        pooled_rows=pooled_rows,
    )
    if not isinstance(first_exp_meta, dict) or not isinstance(spec, dict):
        return {}
    plt = _load_matplotlib_pyplot()
    if plt is None:
        return {}

    suffix = str(file_suffix or "")
    png_path = out_dir / (str(output_png_name) if output_png_name else f"pooled_map{suffix}.png")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern_red_rel, pattern_blue_rel = _pattern_points_from_spec(str(spec["pattern"]))
    geometric_candidates = [
        cell
        for cell in list(spec.get("cells") or [])
        if isinstance(cell, dict) and str(cell.get("kind") or "") == "local" and isinstance(cell.get("local_rel"), list)
    ]
    board_size, _mapped = _minimal_square_board_layout(
        list(pattern_red_rel) + list(pattern_blue_rel) + [tuple(int(x) for x in cell["local_rel"]) for cell in geometric_candidates]
    )
    fig_w = max(5.0, 0.9 * board_size + 1.8)
    fig_h = max(4.0, 0.85 * board_size + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor((246 / 255.0, 241 / 255.0, 232 / 255.0))
    ax.set_facecolor((246 / 255.0, 241 / 255.0, 232 / 255.0))
    if not _draw_local_map_spec(ax, spec):
        plt.close(fig)
        return {}
    fig.tight_layout(pad=0.2)
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path.name}


def _write_scored_outputs(
    *,
    out_dir: Path,
    first_rep: Any,
    first_exp_meta: dict[str, Any] | None,
    summary_rows: list[dict[str, Any]],
    total_representatives: int,
    file_suffix: str = "",
    value_field: str = "stone_fraction",
    save_json: bool = True,
    output_png_name: str | None = None,
) -> dict[str, Any]:
    suffix = str(file_suffix or "")
    pooled_rows = lps._build_pooled_candidates(
        summary_rows,
        total_representatives=total_representatives,
        value_field=value_field,
    )
    pooled_map_artifacts = _write_local_pooled_map_artifacts(
        out_dir=out_dir,
        first_rep=first_rep,
        first_exp_meta=first_exp_meta,
        pooled_rows=pooled_rows,
        file_suffix=suffix,
        output_png_name=output_png_name,
    )
    out: dict[str, Any] = {
        "file_suffix": suffix,
        "pooled_candidates_count": len(pooled_rows),
        "pooled_map_artifacts": pooled_map_artifacts,
    }
    if save_json:
        pooled_json = out_dir / f"pooled_candidates{suffix}.json"
        lps._write_pooled_candidates_json(pooled_json, pooled_rows)
        out["pooled_candidates_json"] = pooled_json.name
    return out
