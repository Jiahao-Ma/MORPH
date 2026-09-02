#!/usr/bin/env python3
"""Build an immutable >=3 s, terrain-balanced Stage 2/Stage 3 subset.

The input is the existing forward-only Stage 2 and G1 Stage 3 pair. Motions
shorter than ``min_duration`` are rejected in full. Remaining motions are
classified from the G1 root trajectory:

* horizontal excursion from the first frame;
* p95 horizontal speed measured over a fixed window;
* root quaternion excursion from the first frame.

A motion is stationary only when all three values are below their configured
thresholds. Within each exact ``terrain_ref`` (and within each terrain-less
category), all moving motions are retained and stationary motions are capped
at the number of moving motions. If downsampling is necessary, a deterministic
round-robin over source recording, duration bin, and initial XY cell preserves
more diversity than taking the first N rows.

No input artifact is modified. Accepted Stage 2 JSONL/meta and G1 qpos/meta
pairs are copied into new output roots; retargeting is not repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np


FILTER_VERSION = 1


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp, path)


def atomic_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(temp, path)


def copy_atomic(source: pathlib.Path, destination: pathlib.Path) -> str:
    """Copy one immutable artifact, safely resuming an interrupted run."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if source_stat.st_size == destination_stat.st_size:
            return "existing"
        raise FileExistsError(
            f"existing output has a different size: {destination}"
        )
    temp = destination.with_name(f".{destination.name}.tmp")
    if temp.exists():
        temp.unlink()
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return "copied"


def stable_hash(seed: int, *values: Any) -> str:
    text = "\x1f".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(text.encode()).hexdigest()


def duration_bin(duration_s: float) -> str:
    if duration_s < 5.0:
        return "3-5"
    if duration_s < 10.0:
        return "5-10"
    if duration_s < 20.0:
        return "10-20"
    if duration_s < 60.0:
        return "20-60"
    return "60+"


def terrain_group(category: str, terrain_ref: str | None) -> str:
    if terrain_ref:
        return terrain_ref
    return f"__no_terrain__/{category}"


def root_motion_descriptor(
    qpos: np.ndarray,
    *,
    fps: float,
    speed_window_s: float,
) -> dict[str, Any]:
    """Measure translation/rotation without frame-to-frame jitter sensitivity."""
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"qpos shape must be (T, 36), got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError("qpos must contain at least two frames")
    if fps <= 0.0 or speed_window_s <= 0.0:
        raise ValueError("fps and speed_window_s must be positive")
    if not bool(np.isfinite(qpos).all()):
        raise ValueError("qpos contains non-finite values")

    xy = np.asarray(qpos[:, :2], dtype=np.float64)
    relative_xy = xy - xy[0]
    excursion = np.linalg.norm(relative_xy, axis=1)
    max_horizontal_excursion_m = float(excursion.max())

    window_frames = max(1, int(round(speed_window_s * fps)))
    window_frames = min(window_frames, qpos.shape[0] - 1)
    window_dt = window_frames / fps
    window_speed = (
        np.linalg.norm(xy[window_frames:] - xy[:-window_frames], axis=1)
        / window_dt
    )
    p95_window_speed_mps = float(np.quantile(window_speed, 0.95))

    quaternions = np.array(qpos[:, 3:7], dtype=np.float64, copy=True)
    norms = np.linalg.norm(quaternions, axis=1)
    if bool(np.any(norms < 1e-8)):
        raise ValueError("qpos contains a near-zero root quaternion")
    quaternions /= norms[:, None]
    # abs(dot) makes q and -q equivalent.
    dots = np.clip(np.abs(quaternions @ quaternions[0]), 0.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    max_root_rotation_deg = float(np.degrees(angles_rad.max()))

    return {
        "start_xy_m": [float(xy[0, 0]), float(xy[0, 1])],
        "end_xy_m": [float(xy[-1, 0]), float(xy[-1, 1])],
        "max_horizontal_excursion_m": max_horizontal_excursion_m,
        "p95_window_speed_mps": p95_window_speed_mps,
        "max_root_rotation_deg": max_root_rotation_deg,
        "speed_window_s": speed_window_s,
        "speed_window_frames": window_frames,
    }


def is_stationary(
    descriptor: dict[str, Any],
    *,
    max_excursion_m: float,
    max_p95_speed_mps: float,
    max_rotation_deg: float,
) -> bool:
    """Use strict inequalities so a value on a threshold is moving."""
    return (
        float(descriptor["max_horizontal_excursion_m"]) < max_excursion_m
        and float(descriptor["p95_window_speed_mps"]) < max_p95_speed_mps
        and float(descriptor["max_root_rotation_deg"]) < max_rotation_deg
    )


def select_diverse_stationary(
    rows: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    xy_cell_m: float,
) -> set[str]:
    """Deterministically round-robin stationary candidates across strata."""
    if quota < 0:
        raise ValueError("quota must be non-negative")
    if xy_cell_m <= 0.0:
        raise ValueError("xy_cell_m must be positive")
    if quota >= len(rows):
        return {str(row["seg_id"]) for row in rows}
    if quota == 0:
        return set()

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        start_xy = row["motion_metrics"]["start_xy_m"]
        grid = (
            math.floor(float(start_xy[0]) / xy_cell_m),
            math.floor(float(start_xy[1]) / xy_cell_m),
        )
        key = (
            row["source_recording"],
            duration_bin(float(row["duration_s"])),
            grid[0],
            grid[1],
        )
        buckets[key].append(row)

    keys = sorted(
        buckets,
        key=lambda key: stable_hash(seed, "stratum", *key),
    )
    for key, candidates in buckets.items():
        candidates.sort(
            key=lambda row: stable_hash(seed, "motion", row["seg_id"])
        )

    selected: set[str] = set()
    while len(selected) < quota:
        made_progress = False
        for key in keys:
            if buckets[key]:
                selected.add(str(buckets[key].pop(0)["seg_id"]))
                made_progress = True
                if len(selected) == quota:
                    break
        if not made_progress:
            raise RuntimeError("stationary selection exhausted before quota")
    return selected


def validate_manifest_pair(
    stage2_rows: list[dict[str, Any]],
    g1_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stage2_by_id: dict[str, dict[str, Any]] = {}
    for row in stage2_rows:
        seg_id = str(row["seg_id"])
        if seg_id in stage2_by_id:
            raise ValueError(f"duplicate Stage 2 seg_id: {seg_id}")
        stage2_by_id[seg_id] = row

    g1_by_id: dict[str, dict[str, Any]] = {}
    for row in g1_rows:
        seg_id = str(row["seg_id"])
        if seg_id in g1_by_id:
            raise ValueError(f"duplicate G1 seg_id: {seg_id}")
        g1_by_id[seg_id] = row

    missing_g1 = sorted(stage2_by_id.keys() - g1_by_id.keys())
    extra_g1 = sorted(g1_by_id.keys() - stage2_by_id.keys())
    if missing_g1 or extra_g1:
        raise ValueError(
            "Stage 2/G1 seg_id sets differ: "
            f"missing_g1={len(missing_g1)}, extra_g1={len(extra_g1)}"
        )
    return g1_by_id


def analyze(
    *,
    stage2_rows: list[dict[str, Any]],
    g1_rows: list[dict[str, Any]],
    min_duration_s: float,
    speed_window_s: float,
    max_excursion_m: float,
    max_p95_speed_mps: float,
    max_rotation_deg: float,
    seed: int,
    xy_cell_m: float,
) -> list[dict[str, Any]]:
    g1_by_id = validate_manifest_pair(stage2_rows, g1_rows)
    decisions: list[dict[str, Any]] = []

    for index, stage2_row in enumerate(stage2_rows, 1):
        seg_id = str(stage2_row["seg_id"])
        category = str(stage2_row["category"])
        g1_row = g1_by_id[seg_id]
        stage2_jsonl = pathlib.Path(stage2_row["washed_jsonl"])
        stage2_meta = pathlib.Path(stage2_row["washed_meta"])
        qpos_path = pathlib.Path(g1_row["qpos_npy"])
        g1_meta = pathlib.Path(g1_row["seg_meta"])
        for label, path in (
            ("Stage 2 JSONL", stage2_jsonl),
            ("Stage 2 meta", stage2_meta),
            ("G1 qpos", qpos_path),
            ("G1 meta", g1_meta),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{seg_id}: missing {label}: {path}")

        n_frames = int(stage2_row["n_frames"])
        if int(g1_row["n_frames"]) != n_frames:
            raise ValueError(f"{seg_id}: Stage 2/G1 n_frames differ")
        duration_s = float(stage2_row["duration_s"])
        if not math.isclose(
            float(g1_row["duration_s"]),
            duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{seg_id}: Stage 2/G1 duration_s differ")

        meta = read_json(stage2_meta)
        terrain_ref_raw = meta.get("terrain_ref")
        terrain_ref = str(terrain_ref_raw) if terrain_ref_raw else None
        decision: dict[str, Any] = {
            "seg_id": seg_id,
            "category": category,
            "terrain_ref": terrain_ref,
            "terrain_group": terrain_group(category, terrain_ref),
            "source_recording": str(
                stage2_row.get("source_clean_jsonl", stage2_jsonl)
            ),
            "n_frames": n_frames,
            "duration_s": duration_s,
            "stage2_jsonl": str(stage2_jsonl.resolve()),
            "stage2_meta": str(stage2_meta.resolve()),
            "qpos_npy": str(qpos_path.resolve()),
            "g1_meta": str(g1_meta.resolve()),
            "motion_metrics": None,
            "stationary": None,
            "decision": None,
            "reject_reason": None,
        }

        qpos = np.load(qpos_path, mmap_mode="r", allow_pickle=False)
        if qpos.shape != (n_frames, 36):
            raise ValueError(
                f"{seg_id}: qpos shape {qpos.shape}, expected ({n_frames}, 36)"
            )

        if duration_s < min_duration_s:
            # Still validate shape/finiteness for a trustworthy input audit.
            if not bool(np.isfinite(qpos).all()):
                raise ValueError(f"{seg_id}: qpos contains non-finite values")
            decision["decision"] = "reject"
            decision["reject_reason"] = "duration_lt_min"
        else:
            descriptor = root_motion_descriptor(
                qpos,
                fps=float(g1_row.get("fps_src") or meta.get("sample_rate_hz") or 60),
                speed_window_s=speed_window_s,
            )
            stationary = is_stationary(
                descriptor,
                max_excursion_m=max_excursion_m,
                max_p95_speed_mps=max_p95_speed_mps,
                max_rotation_deg=max_rotation_deg,
            )
            decision["motion_metrics"] = descriptor
            decision["stationary"] = stationary
        decisions.append(decision)
        if index == 1 or index % 250 == 0 or index == len(stage2_rows):
            print(f"  [Analyze {index:>5}/{len(stage2_rows)}] {seg_id}")

    eligible_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if decision["reject_reason"] is None:
            eligible_by_group[str(decision["terrain_group"])].append(decision)

    for group, candidates in eligible_by_group.items():
        moving = [row for row in candidates if not row["stationary"]]
        stationary = [row for row in candidates if row["stationary"]]
        quota = len(moving)
        selected_stationary = select_diverse_stationary(
            stationary,
            quota=quota,
            seed=seed,
            xy_cell_m=xy_cell_m,
        )
        for row in moving:
            row["decision"] = "keep"
        for row in stationary:
            if str(row["seg_id"]) in selected_stationary:
                row["decision"] = "keep"
            else:
                row["decision"] = "reject"
                row["reject_reason"] = "stationary_balance_excess"
        if len(stationary) > quota:
            print(
                f"  [Balance] {group}: moving={len(moving)}, "
                f"stationary={len(stationary)}, kept_stationary={quota}"
            )

    if any(row["decision"] is None for row in decisions):
        raise RuntimeError("internal error: an analyzed motion has no decision")
    return decisions


def fresh_summary() -> dict[str, int | float]:
    return {
        "input_motions": 0,
        "input_frames": 0,
        "input_duration_s": 0.0,
        "duration_rejected_motions": 0,
        "duration_rejected_frames": 0,
        "duration_rejected_duration_s": 0.0,
        "eligible_motions": 0,
        "moving_motions": 0,
        "stationary_motions": 0,
        "balance_rejected_motions": 0,
        "balance_rejected_frames": 0,
        "balance_rejected_duration_s": 0.0,
        "output_motions": 0,
        "output_frames": 0,
        "output_duration_s": 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    result = fresh_summary()
    for row in rows:
        frames = int(row["n_frames"])
        duration_s = float(row["duration_s"])
        result["input_motions"] += 1
        result["input_frames"] += frames
        result["input_duration_s"] += duration_s
        reason = row["reject_reason"]
        if reason == "duration_lt_min":
            result["duration_rejected_motions"] += 1
            result["duration_rejected_frames"] += frames
            result["duration_rejected_duration_s"] += duration_s
            continue
        result["eligible_motions"] += 1
        if row["stationary"]:
            result["stationary_motions"] += 1
        else:
            result["moving_motions"] += 1
        if reason == "stationary_balance_excess":
            result["balance_rejected_motions"] += 1
            result["balance_rejected_frames"] += frames
            result["balance_rejected_duration_s"] += duration_s
        if row["decision"] == "keep":
            result["output_motions"] += 1
            result["output_frames"] += frames
            result["output_duration_s"] += duration_s
    return result


def build_stats(
    decisions: list[dict[str, Any]],
    *,
    parameters: dict[str, Any],
    inputs: dict[str, str],
    outputs: dict[str, str],
    elapsed_s: float,
) -> dict[str, Any]:
    per_category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_terrain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        per_category_rows[str(row["category"])].append(row)
        per_terrain_rows[str(row["terrain_group"])].append(row)

    kept_durations = [
        float(row["duration_s"]) for row in decisions if row["decision"] == "keep"
    ]
    duration_histogram: list[dict[str, Any]] = []
    duration_bins = [
        (0.0, 3.0, "<3"),
        (3.0, 5.0, "3-5"),
        (5.0, 10.0, "5-10"),
        (10.0, 20.0, "10-20"),
        (20.0, 30.0, "20-30"),
        (30.0, 60.0, "30-60"),
        (60.0, 120.0, "60-120"),
        (120.0, 300.0, "120-300"),
        (300.0, math.inf, "300+"),
    ]
    for lo, hi, label in duration_bins:
        count = sum(1 for value in kept_durations if lo <= value < hi)
        duration_histogram.append(
            {
                "label": label,
                "count": count,
                "percent": (
                    100.0 * count / len(kept_durations)
                    if kept_durations
                    else 0.0
                ),
            }
        )
    duration_percentiles = {
        str(q): (
            float(np.quantile(np.asarray(kept_durations), q / 100.0))
            if kept_durations
            else 0.0
        )
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    return {
        "schema_version": 1,
        "filter_version": FILTER_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": parameters,
        "inputs": inputs,
        "outputs": outputs,
        "global": summarize_rows(decisions),
        "per_category": {
            key: summarize_rows(rows)
            for key, rows in sorted(per_category_rows.items())
        },
        "per_terrain": {
            key: summarize_rows(rows)
            for key, rows in sorted(per_terrain_rows.items())
        },
        "output_duration_distribution": {
            "n": len(kept_durations),
            "total_h": sum(kept_durations) / 3600.0,
            "min_s": min(kept_durations) if kept_durations else 0.0,
            "median_s": (
                statistics.median(kept_durations) if kept_durations else 0.0
            ),
            "mean_s": (
                statistics.fmean(kept_durations) if kept_durations else 0.0
            ),
            "max_s": max(kept_durations) if kept_durations else 0.0,
            "percentiles_s": duration_percentiles,
            "histogram": duration_histogram,
        },
        "elapsed_s": elapsed_s,
    }


def fmt_pct(value: int | float, total: int | float) -> str:
    if not total:
        return "0.0%"
    return f"{100.0 * float(value) / float(total):.1f}%"


def build_report(stats: dict[str, Any]) -> str:
    p = stats["parameters"]
    g = stats["global"]
    lines = [
        "# Minimum-duration and stationary-balance filter",
        "",
        f"Generated: {stats['generated_at']}",
        "",
        "Rules:",
        "",
        f"- reject an entire motion when `duration_s < {p['min_duration_s']}`",
        (
            "- stationary iff root horizontal excursion "
            f"`< {p['max_excursion_m']} m`, p95 {p['speed_window_s']} s-window "
            f"speed `< {p['max_p95_speed_mps']} m/s`, and root rotation "
            f"`< {p['max_rotation_deg']} deg`"
        ),
        "- within each terrain, keep all moving motions and cap stationary "
        "motions at the moving-motion count",
        "- accepted Stage 2/G1 pairs are copied; inputs are untouched",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input motions | {g['input_motions']:,} |",
        f"| input frames | {g['input_frames']:,} |",
        f"| input duration | {g['input_duration_s']/3600.0:.2f} h |",
        f"| rejected `<3s` motions | {g['duration_rejected_motions']:,} |",
        f"| eligible moving motions | {g['moving_motions']:,} |",
        f"| eligible stationary motions | {g['stationary_motions']:,} |",
        f"| rejected excess stationary motions | {g['balance_rejected_motions']:,} |",
        (
            f"| **output motions** | **{g['output_motions']:,}** "
            f"({fmt_pct(g['output_motions'], g['input_motions'])}) |"
        ),
        (
            f"| **output frames** | **{g['output_frames']:,}** "
            f"({fmt_pct(g['output_frames'], g['input_frames'])}) |"
        ),
        f"| **output duration** | **{g['output_duration_s']/3600.0:.2f} h** |",
        "",
        "## Before vs after",
        "",
        "| metric | before | after | change |",
        "|---|---:|---:|---:|",
        (
            f"| motions | {g['input_motions']:,} | **{g['output_motions']:,}** | "
            f"−{g['input_motions'] - g['output_motions']:,} "
            f"(−{100.0 * (g['input_motions'] - g['output_motions']) / g['input_motions']:.1f}%) |"
        ),
        (
            f"| frames | {g['input_frames']:,} | **{g['output_frames']:,}** | "
            f"−{g['input_frames'] - g['output_frames']:,} "
            f"(−{100.0 * (g['input_frames'] - g['output_frames']) / g['input_frames']:.1f}%) |"
        ),
        (
            f"| duration | {g['input_duration_s']/3600.0:.2f} h | "
            f"**{g['output_duration_s']/3600.0:.2f} h** | "
            f"−{(g['input_duration_s'] - g['output_duration_s'])/3600.0:.2f} h |"
        ),
        (
            f"| motions `< {p['min_duration_s']}s` | "
            f"{g['duration_rejected_motions']:,} | **0** | eliminated ✓ |"
        ),
        "",
        "## Per category",
        "",
        "| category | input | `<3s` reject | moving | stationary | balance reject | output | frames | hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in stats["per_category"].items():
        lines.append(
            f"| {category} | {row['input_motions']:,} | "
            f"{row['duration_rejected_motions']:,} | {row['moving_motions']:,} | "
            f"{row['stationary_motions']:,} | "
            f"{row['balance_rejected_motions']:,} | "
            f"{row['output_motions']:,} | {row['output_frames']:,} | "
            f"{row['output_duration_s']/3600.0:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per terrain",
            "",
            "| terrain group | input | `<3s` reject | moving | stationary | balance reject | output |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for terrain, row in stats["per_terrain"].items():
        lines.append(
            f"| `{terrain}` | {row['input_motions']:,} | "
            f"{row['duration_rejected_motions']:,} | {row['moving_motions']:,} | "
            f"{row['stationary_motions']:,} | "
            f"{row['balance_rejected_motions']:,} | "
            f"{row['output_motions']:,} |"
        )
    distribution = stats["output_duration_distribution"]
    lines.extend(
        [
            "",
            "## Output duration distribution",
            "",
            "| bin (s) | motions | % |",
            "|---|---:|---:|",
        ]
    )
    for row in distribution["histogram"]:
        lines.append(
            f"| {row['label']} | {row['count']:,} | {row['percent']:.1f}% |"
        )
    percentiles = distribution["percentiles_s"]
    lines.extend(
        [
            "",
            (
                f"Percentiles: p1={percentiles['1']:.2f}s, "
                f"p5={percentiles['5']:.2f}s, "
                f"p10={percentiles['10']:.2f}s, "
                f"p25={percentiles['25']:.2f}s, "
                f"**p50={percentiles['50']:.2f}s**, "
                f"p75={percentiles['75']:.2f}s, "
                f"p90={percentiles['90']:.2f}s, "
                f"p95={percentiles['95']:.2f}s, "
                f"p99={percentiles['99']:.2f}s."
            ),
            (
                f"Min={distribution['min_s']:.2f}s, "
                f"max={distribution['max_s']:.2f}s, "
                f"mean={distribution['mean_s']:.2f}s, "
                f"total={distribution['total_h']:.2f} h."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_stage2_meta(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    filter_record: dict[str, Any],
    output_jsonl: pathlib.Path,
) -> None:
    meta = read_json(source)
    record = dict(filter_record)
    record["source_meta"] = str(source.resolve())
    record["output_jsonl"] = str(output_jsonl.resolve())
    meta["min3_static_balanced"] = True
    meta["min_duration_static_balance"] = record
    atomic_json(destination, meta)


def copy_outputs(
    *,
    decisions: list[dict[str, Any]],
    stage2_by_id: dict[str, dict[str, Any]],
    g1_by_id: dict[str, dict[str, Any]],
    stage2_output_root: pathlib.Path,
    g1_output_root: pathlib.Path,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    stage2_output_rows: list[dict[str, Any]] = []
    g1_output_rows: list[dict[str, Any]] = []
    copied_stage2 = reused_stage2 = copied_g1 = reused_g1 = 0
    kept = [row for row in decisions if row["decision"] == "keep"]

    for index, decision in enumerate(kept, 1):
        seg_id = str(decision["seg_id"])
        category = str(decision["category"])
        stage2_source = stage2_by_id[seg_id]
        g1_source = g1_by_id[seg_id]

        source_jsonl = pathlib.Path(stage2_source["washed_jsonl"])
        source_stage2_meta = pathlib.Path(stage2_source["washed_meta"])
        output_jsonl = stage2_output_root / category / source_jsonl.name
        output_stage2_meta = (
            stage2_output_root / category / source_stage2_meta.name
        )
        status = copy_atomic(source_jsonl, output_jsonl)
        copied_stage2 += int(status == "copied")
        reused_stage2 += int(status == "existing")

        filter_record = {
            "version": FILTER_VERSION,
            "parameters": parameters,
            "decision": "keep",
            "stationary": bool(decision["stationary"]),
            "terrain_group": decision["terrain_group"],
            "motion_metrics": decision["motion_metrics"],
            "source_jsonl": str(source_jsonl.resolve()),
        }
        write_stage2_meta(
            source_stage2_meta,
            output_stage2_meta,
            filter_record=filter_record,
            output_jsonl=output_jsonl,
        )
        stage2_result = dict(stage2_source)
        stage2_result.update(
            {
                "washed_jsonl": str(output_jsonl.resolve()),
                "washed_meta": str(output_stage2_meta.resolve()),
                "status": "min3_static_balanced",
                "min_duration_static_balance": filter_record,
            }
        )
        stage2_output_rows.append(stage2_result)

        source_qpos = pathlib.Path(g1_source["qpos_npy"])
        source_g1_meta = pathlib.Path(g1_source["seg_meta"])
        output_qpos = g1_output_root / category / source_qpos.name
        output_g1_meta = g1_output_root / category / source_g1_meta.name
        status = copy_atomic(source_qpos, output_qpos)
        copied_g1 += int(status == "copied")
        reused_g1 += int(status == "existing")

        g1_result = dict(g1_source)
        g1_result.update(
            {
                "qpos_npy": str(output_qpos.resolve()),
                "seg_meta": str(output_g1_meta.resolve()),
                "washed_jsonl": str(output_jsonl.resolve()),
                "washed_meta": str(output_stage2_meta.resolve()),
                "status": "qa_ok_min3_static_balanced",
                "min_duration_static_balance": filter_record,
            }
        )
        atomic_json(output_g1_meta, g1_result)
        g1_output_rows.append(g1_result)

        if index == 1 or index % 100 == 0 or index == len(kept):
            print(
                f"  [Copy {index:>5}/{len(kept)}] {seg_id} "
                f"(Stage2 copied/reused={copied_stage2}/{reused_stage2}; "
                f"G1={copied_g1}/{reused_g1})"
            )
    return (
        stage2_output_rows,
        g1_output_rows,
        {
            "stage2_copied": copied_stage2,
            "stage2_reused": reused_stage2,
            "g1_copied": copied_g1,
            "g1_reused": reused_g1,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--g1-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--stage2-output-root", required=True, type=pathlib.Path)
    parser.add_argument("--g1-output-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--audit-root",
        type=pathlib.Path,
        default=None,
        help="default: Stage 2 output root; useful with --scan-only and /tmp",
    )
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--speed-window", type=float, default=0.5)
    parser.add_argument("--max-excursion", type=float, default=0.30)
    parser.add_argument("--max-p95-speed", type=float, default=0.20)
    parser.add_argument("--max-rotation", type=float, default=20.0)
    parser.add_argument("--xy-cell", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="write audit manifests/stats without copying accepted artifacts",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for label in (
        "min_duration",
        "speed_window",
        "max_excursion",
        "max_p95_speed",
        "max_rotation",
        "xy_cell",
    ):
        if float(getattr(args, label)) <= 0.0:
            parser.error(f"--{label.replace('_', '-')} must be positive")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    stage2_manifest = args.stage2_manifest.resolve()
    g1_manifest = args.g1_manifest.resolve()
    if not stage2_manifest.is_file():
        parser.error(f"missing --stage2-manifest: {stage2_manifest}")
    if not g1_manifest.is_file():
        parser.error(f"missing --g1-manifest: {g1_manifest}")
    stage2_output_root = args.stage2_output_root.resolve()
    g1_output_root = args.g1_output_root.resolve()
    audit_root = (
        args.audit_root.resolve()
        if args.audit_root
        else stage2_output_root
    )

    parameters = {
        "min_duration_s": args.min_duration,
        "speed_window_s": args.speed_window,
        "max_excursion_m": args.max_excursion,
        "max_p95_speed_mps": args.max_p95_speed,
        "max_rotation_deg": args.max_rotation,
        "stationary_to_moving_max_ratio": 1.0,
        "xy_cell_m": args.xy_cell,
        "selection_seed": args.seed,
        "strict_threshold_comparison": True,
    }
    stage2_rows = read_jsonl(stage2_manifest)
    g1_rows = read_jsonl(g1_manifest)
    print(
        f"[Input] Stage2={len(stage2_rows):,}, G1={len(g1_rows):,}; "
        f"min_duration={args.min_duration:g}s"
    )
    started = time.monotonic()
    decisions = analyze(
        stage2_rows=stage2_rows,
        g1_rows=g1_rows,
        min_duration_s=args.min_duration,
        speed_window_s=args.speed_window,
        max_excursion_m=args.max_excursion,
        max_p95_speed_mps=args.max_p95_speed,
        max_rotation_deg=args.max_rotation,
        seed=args.seed,
        xy_cell_m=args.xy_cell,
    )

    stage2_by_id = {str(row["seg_id"]): row for row in stage2_rows}
    g1_by_id = {str(row["seg_id"]): row for row in g1_rows}
    kept = [row for row in decisions if row["decision"] == "keep"]
    duration_rejected = [
        row for row in decisions if row["reject_reason"] == "duration_lt_min"
    ]
    balance_rejected = [
        row
        for row in decisions
        if row["reject_reason"] == "stationary_balance_excess"
    ]

    copy_stats: dict[str, int] | None = None
    if not args.scan_only:
        stage2_output_root.mkdir(parents=True, exist_ok=True)
        g1_output_root.mkdir(parents=True, exist_ok=True)
        (
            stage2_output_rows,
            g1_output_rows,
            copy_stats,
        ) = copy_outputs(
            decisions=decisions,
            stage2_by_id=stage2_by_id,
            g1_by_id=g1_by_id,
            stage2_output_root=stage2_output_root,
            g1_output_root=g1_output_root,
            parameters=parameters,
        )
        atomic_jsonl(
            stage2_output_root / "segments_manifest.jsonl",
            stage2_output_rows,
        )
        atomic_jsonl(
            g1_output_root / "segments_manifest.jsonl",
            g1_output_rows,
        )
        for root in (stage2_output_root, g1_output_root):
            atomic_jsonl(root / "duration_reject_manifest.jsonl", duration_rejected)
            atomic_jsonl(
                root / "static_balance_reject_manifest.jsonl",
                balance_rejected,
            )
            atomic_jsonl(
                root / "motion_classification_manifest.jsonl",
                decisions,
            )

    outputs = {
        "stage2_output_root": str(stage2_output_root),
        "g1_output_root": str(g1_output_root),
        "audit_root": str(audit_root),
        "mode": "scan_only" if args.scan_only else "copy",
    }
    stats = build_stats(
        decisions,
        parameters=parameters,
        inputs={
            "stage2_manifest": str(stage2_manifest),
            "g1_manifest": str(g1_manifest),
        },
        outputs=outputs,
        elapsed_s=time.monotonic() - started,
    )
    if copy_stats is not None:
        stats["copies"] = copy_stats
    audit_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(audit_root / "motion_classification_manifest.jsonl", decisions)
    atomic_jsonl(audit_root / "duration_reject_manifest.jsonl", duration_rejected)
    atomic_jsonl(
        audit_root / "static_balance_reject_manifest.jsonl",
        balance_rejected,
    )
    atomic_json(audit_root / "min3_static_balance_stats.json", stats)
    report = build_report(stats)
    report_path = audit_root / "min3_static_balance_report.md"
    temp_report = report_path.with_name(f".{report_path.name}.tmp")
    temp_report.write_text(report, encoding="utf-8")
    os.replace(temp_report, report_path)

    g = stats["global"]
    print(
        "[Result] "
        f"input={g['input_motions']:,}, short_reject="
        f"{g['duration_rejected_motions']:,}, moving={g['moving_motions']:,}, "
        f"stationary={g['stationary_motions']:,}, balance_reject="
        f"{g['balance_rejected_motions']:,}, output={g['output_motions']:,}"
    )
    print(f"[Report] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
