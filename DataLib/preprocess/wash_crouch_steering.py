#!/usr/bin/env python3
"""Split already-preprocessed MORPH JSONL on crouch and lateral steering.

This is deliberately a *second* cleaning stage. It assumes clean_morph.py has
already written a separate preprocessed dataset containing the stuck-frame,
capture-gap, interpolation, glitch, teleport/reset, and frame/time fixes. This
script does not repeat those rules and never mutates its input dataset.

Bad-frame rules:
  crouch  = bool(cmd.crouch)
  steering = abs(cmd.move[0]) > eps_side and hypot(cmd.move) > eps_mag

cmd.move follows the source convention [right, forward]. Turning commands
(desYR/dLookY/look yaw) and cmd.jump are intentionally ignored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import time
from typing import BinaryIO


DEFAULT_CATEGORIES = [
    "ground",
    "stairs",
    "traversal_mantle",
    "traversal_mantle_vault",
    "traversal_vault",
]

CMD_RE = re.compile(rb'"cmd"\s*:\s*\{(.*?)\}\s*,\s*"root"', re.DOTALL)
MOVE_RE = re.compile(rb'"move"\s*:\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\]')
CROUCH_RE = re.compile(rb'"crouch"\s*:\s*(true|false)')
DT_RE = re.compile(rb'"dt"\s*:\s*([-+0-9.eE]+)')
T_RE = re.compile(rb'"t"\s*:\s*([-+0-9.eE]+)')
MESH_SCALE_RE = re.compile(
    rb'"mesh"\s*:\s*\{.*?"s"\s*:\s*\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*,\s*([^\]]+)\]',
    re.DOTALL,
)

DURATION_BINS = [
    (0.0, 1.0, "<1"),
    (1.0, 2.0, "1-2"),
    (2.0, 3.0, "2-3"),
    (3.0, 5.0, "3-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 20.0, "10-20"),
    (20.0, 30.0, "20-30"),
    (30.0, 60.0, "30-60"),
    (60.0, 120.0, "60-120"),
    (120.0, 300.0, "120-300"),
    (300.0, 600.0, "300-600"),
    (600.0, 1800.0, "600-1800"),
    (1800.0, math.inf, ">1800"),
]


def _float(raw: bytes) -> float:
    return float(raw.strip())


def parse_frame_header(line: bytes) -> tuple[bool, list[float], float, float | None, list[float] | None]:
    """Parse only the small header before joints; avoid decoding 88 joints."""
    prefix = line[:8192]
    cmd_match = CMD_RE.search(prefix)
    if cmd_match is None:
        # Schema/order fallback. It is slower but keeps the parser correct for
        # a valid JSON object whose fields were reordered.
        row = json.loads(line)
        cmd = row.get("cmd") or {}
        move = cmd.get("move", [0.0, 0.0])
        root = row.get("root") or {}
        mesh = root.get("mesh") or {}
        return (
            bool(cmd.get("crouch", False)),
            [float(move[0]), float(move[1])],
            float(row.get("dt", 0.0)),
            float(row["t"]) if row.get("t") is not None else None,
            [float(v) for v in mesh["s"]] if mesh.get("s") is not None else None,
        )

    cmd_block = cmd_match.group(1)
    crouch_match = CROUCH_RE.search(cmd_block)
    move_match = MOVE_RE.search(cmd_block)
    crouch = crouch_match is not None and crouch_match.group(1) == b"true"
    move = [0.0, 0.0]
    if move_match is not None:
        move = [_float(move_match.group(1)), _float(move_match.group(2))]

    dt_match = DT_RE.search(prefix)
    t_match = T_RE.search(prefix)
    scale_match = MESH_SCALE_RE.search(prefix)
    scale = None
    if scale_match is not None:
        scale = [_float(scale_match.group(i)) for i in (1, 2, 3)]
    return (
        crouch,
        move,
        _float(dt_match.group(1)) if dt_match is not None else 0.0,
        _float(t_match.group(1)) if t_match is not None else None,
        scale,
    )


def classify_line(line: bytes, eps_side: float, eps_mag: float) -> tuple[bool, bool]:
    crouch, move, _, _, _ = parse_frame_header(line)
    steering = abs(move[0]) > eps_side and math.hypot(move[0], move[1]) > eps_mag
    return crouch, steering


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(tmp, path)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    a = sorted(values)
    pos = (len(a) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(a[lo])
    frac = pos - lo
    return float(a[lo] * (1.0 - frac) + a[hi] * frac)


def distribution(values: list[float]) -> dict:
    hist = []
    for lo, hi, label in DURATION_BINS:
        count = sum(1 for v in values if lo <= v < hi)
        hist.append({"label": label, "count": count})
    return {
        "n": len(values),
        "total_h": sum(values) / 3600.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean": statistics.fmean(values) if values else 0.0,
        "percentiles": {
            str(q): percentile(values, q)
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "histogram": hist,
    }


def fresh_counter() -> dict:
    return {
        "input_files": 0,
        "input_frames": 0,
        "input_duration_s": 0.0,
        "crouch_frames": 0,
        "steering_frames": 0,
        "overlap_frames": 0,
        "bad_union_frames": 0,
        "good_frames": 0,
        "candidate_runs": 0,
        "kept_segments": 0,
        "rejected_segments": 0,
        "output_frames": 0,
        "output_duration_s": 0.0,
        "rejected_good_frames": 0,
        "rejected_good_duration_s": 0.0,
        "files_without_bad": 0,
        "files_all_rejected": 0,
        "parse_failures": 0,
    }


def add_counter(dst: dict, src: dict) -> None:
    for key, value in src.items():
        dst[key] += value


def source_meta_for(frames_path: pathlib.Path) -> pathlib.Path:
    suffix = "_frames.jsonl"
    if not frames_path.name.endswith(suffix):
        raise ValueError(f"unexpected frames filename: {frames_path}")
    return frames_path.with_name(frames_path.name[: -len(suffix)] + "_meta.json")


def write_segment_meta(
    source_meta: pathlib.Path,
    output_meta: pathlib.Path,
    *,
    seg_id: str,
    source_jsonl: pathlib.Path,
    i0: int,
    i1: int,
    n_frames: int,
    duration_s: float,
    eps_side: float,
    eps_mag: float,
) -> float:
    meta = {}
    if source_meta.exists():
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
    fps = float(meta.get("sample_rate_hz", 60.0))
    meta["num_frames"] = n_frames
    meta["washed_crouch_steering"] = True
    meta["wash"] = {
        "seg_id": seg_id,
        "source_clean_jsonl": str(source_jsonl.resolve()),
        "source_clean_meta": str(source_meta.resolve()),
        "i0": i0,
        "i1": i1,
        "duration_s": duration_s,
        "eps_side": eps_side,
        "eps_mag": eps_mag,
    }
    atomic_json(output_meta, meta)
    return fps


def process_file(
    frames_path: pathlib.Path,
    category: str,
    out_root: pathlib.Path,
    *,
    eps_side: float,
    eps_mag: float,
    min_frames: int,
    min_duration: float,
) -> tuple[list[dict], list[dict], dict, float]:
    """Stream one cleaned recording and return accepted/rejected rows/stats."""
    source_meta = source_meta_for(frames_path)
    source_stem = frames_path.name.removesuffix("_frames.jsonl")
    out_cat = out_root / category
    out_cat.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    rejected: list[dict] = []
    stats = fresh_counter()
    stats["input_files"] = 1

    run_file: BinaryIO | None = None
    run_tmp: pathlib.Path | None = None
    run_start = 0
    run_frames = 0
    run_dt_sum = 0.0
    run_first_t: float | None = None
    run_last_t: float | None = None
    candidate_idx = 0
    n_bad = 0
    input_dt_sum = 0.0
    input_first_t: float | None = None
    input_last_t: float | None = None
    scale_min = [math.inf, math.inf, math.inf]
    scale_max = [-math.inf, -math.inf, -math.inf]
    scale_missing = 0

    def finish_run(i1: int) -> None:
        nonlocal run_file, run_tmp, run_start, run_frames, run_dt_sum
        nonlocal run_first_t, run_last_t, candidate_idx
        if run_file is None or run_tmp is None:
            return
        run_file.flush()
        os.fsync(run_file.fileno())
        run_file.close()
        seg_id = f"{source_stem}_seg{candidate_idx:03d}"
        candidate_idx += 1
        stats["candidate_runs"] += 1
        if run_first_t is not None and run_last_t is not None:
            duration_s = max(0.0, run_last_t - run_first_t)
        else:
            duration_s = max(0.0, run_dt_sum)
        keep = run_frames >= min_frames and duration_s >= min_duration
        if keep:
            final_frames = out_cat / f"{seg_id}_frames.jsonl"
            final_meta = out_cat / f"{seg_id}_meta.json"
            os.replace(run_tmp, final_frames)
            fps = write_segment_meta(
                source_meta,
                final_meta,
                seg_id=seg_id,
                source_jsonl=frames_path,
                i0=run_start,
                i1=i1,
                n_frames=run_frames,
                duration_s=duration_s,
                eps_side=eps_side,
                eps_mag=eps_mag,
            )
            row = {
                "seg_id": seg_id,
                "category": category,
                "source_clean_jsonl": str(frames_path.resolve()),
                "source_clean_meta": str(source_meta.resolve()),
                "washed_jsonl": str(final_frames.resolve()),
                "washed_meta": str(final_meta.resolve()),
                "i0": run_start,
                "i1": i1,
                "n_frames": run_frames,
                "duration_s": duration_s,
                "fps_src": fps,
                "status": "washed",
                "wash": {
                    "eps_side": eps_side,
                    "eps_mag": eps_mag,
                    "rule": "cmd.crouch OR lateral cmd.move[0]",
                },
            }
            accepted.append(row)
            stats["kept_segments"] += 1
            stats["output_frames"] += run_frames
            stats["output_duration_s"] += duration_s
        else:
            run_tmp.unlink(missing_ok=True)
            rejected.append({
                "seg_id": seg_id,
                "category": category,
                "source_clean_jsonl": str(frames_path.resolve()),
                "source_clean_meta": str(source_meta.resolve()),
                "i0": run_start,
                "i1": i1,
                "n_frames": run_frames,
                "duration_s": duration_s,
                "reject_stage": "crouch_steering_wash",
                "reject_reason": "too_short_after_crouch_steering",
                "detail": (
                    f"frames={run_frames} (min={min_frames}), "
                    f"duration={duration_s:.6f}s (min={min_duration:.6f}s)"
                ),
            })
            stats["rejected_segments"] += 1
            stats["rejected_good_frames"] += run_frames
            stats["rejected_good_duration_s"] += duration_s
        run_file = None
        run_tmp = None
        run_frames = 0
        run_dt_sum = 0.0
        run_first_t = None
        run_last_t = None

    try:
        with frames_path.open("rb") as src:
            for index, raw_line in enumerate(src):
                line = raw_line.strip()
                if not line:
                    continue
                crouch, move, dt, timestamp, scale = parse_frame_header(line)
                steering = (
                    abs(move[0]) > eps_side
                    and math.hypot(move[0], move[1]) > eps_mag
                )
                bad = crouch or steering
                stats["input_frames"] += 1
                input_dt_sum += max(0.0, dt)
                if timestamp is not None:
                    if input_first_t is None:
                        input_first_t = timestamp
                    input_last_t = timestamp
                stats["crouch_frames"] += int(crouch)
                stats["steering_frames"] += int(steering)
                stats["overlap_frames"] += int(crouch and steering)
                stats["bad_union_frames"] += int(bad)
                stats["good_frames"] += int(not bad)
                n_bad += int(bad)

                if scale is None:
                    scale_missing += 1
                else:
                    for axis in range(3):
                        scale_min[axis] = min(scale_min[axis], scale[axis])
                        scale_max[axis] = max(scale_max[axis], scale[axis])

                if bad:
                    finish_run(stats["input_frames"] - 1)
                    continue

                if run_file is None:
                    run_start = stats["input_frames"] - 1
                    next_id = f"{source_stem}_seg{candidate_idx:03d}"
                    run_tmp = out_cat / f".{next_id}_frames.jsonl.tmp"
                    run_file = run_tmp.open("wb")
                run_file.write(line)
                run_file.write(b"\n")
                run_frames += 1
                run_dt_sum += max(0.0, dt)
                if timestamp is not None:
                    if run_first_t is None:
                        run_first_t = timestamp
                    run_last_t = timestamp
        finish_run(stats["input_frames"])
    except Exception:
        if run_file is not None:
            run_file.close()
        if run_tmp is not None:
            run_tmp.unlink(missing_ok=True)
        raise

    if input_first_t is not None and input_last_t is not None:
        stats["input_duration_s"] = max(0.0, input_last_t - input_first_t)
    else:
        stats["input_duration_s"] = input_dt_sum
    if n_bad == 0:
        stats["files_without_bad"] = 1
    if not accepted:
        stats["files_all_rejected"] = 1

    scale_record = {
        "min": None if scale_missing == stats["input_frames"] else scale_min,
        "max": None if scale_missing == stats["input_frames"] else scale_max,
        "missing_frames": scale_missing,
    }
    for row in accepted:
        row["mesh_scale"] = scale_record
    return accepted, rejected, stats, stats["input_duration_s"]


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "0.0%"


def duration_markdown(title: str, dist: dict) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"n={dist['n']:,}, total={dist['total_h']:.2f} h",
        "",
        "| bin (s) | segments | % |",
        "|---|---:|---:|",
    ]
    for item in dist["histogram"]:
        pct = 100.0 * item["count"] / dist["n"] if dist["n"] else 0.0
        lines.append(f"| {item['label']} | {item['count']:,} | {pct:.1f}% |")
    p = dist["percentiles"]
    lines.extend([
        "",
        (
            f"Percentiles: p1={p['1']:.2f}s, p5={p['5']:.2f}s, "
            f"p10={p['10']:.2f}s, p25={p['25']:.2f}s, "
            f"**p50={p['50']:.2f}s**, p75={p['75']:.2f}s, "
            f"p90={p['90']:.2f}s, p95={p['95']:.2f}s, p99={p['99']:.2f}s; "
            f"min={dist['min']:.2f}s, max={dist['max']:.2f}s, "
            f"mean={dist['mean']:.2f}s."
        ),
        "",
    ])
    return lines


def build_report(stats: dict) -> str:
    g = stats["global"]
    lines = [
        "# Before vs after crouch/steering wash on MorphData_v1",
        "",
        f"Generated: {stats['generated_at']}",
        "",
        "Rules:",
        "",
        f"- crouch: `cmd.crouch == true`",
        (
            f"- steering: `abs(cmd.move[0]) > {stats['parameters']['eps_side']}` "
            f"and `hypot(cmd.move) > {stats['parameters']['eps_mag']}`"
        ),
        "- `cmd.jump`, `desYR`, `dLookY`, and look yaw are not filtered.",
        "",
        "## Cleaning stats",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input files (preprocessed segments) | {fmt_int(g['input_files'])} |",
        f"| input frames | {fmt_int(g['input_frames'])} |",
        f"| candidate good runs | {fmt_int(g['candidate_runs'])} |",
        f"| **output segments** | **{fmt_int(g['kept_segments'])}** |",
        (
            f"| **output frames** | **{fmt_int(g['output_frames'])}** "
            f"({fmt_pct(g['output_frames'], g['input_frames'])} retained) |"
        ),
        f"| crouch frames | {fmt_int(g['crouch_frames'])} |",
        f"| steering frames | {fmt_int(g['steering_frames'])} |",
        f"| crouch ∩ steering frames | {fmt_int(g['overlap_frames'])} |",
        f"| bad union frames removed | {fmt_int(g['bad_union_frames'])} |",
        (
            f"| short good runs rejected | {fmt_int(g['rejected_segments'])} "
            f"({fmt_int(g['rejected_good_frames'])} frames) |"
        ),
        f"| files without crouch/steering | {fmt_int(g['files_without_bad'])} |",
        f"| files with no retained output | {fmt_int(g['files_all_rejected'])} |",
        "",
        "## Before vs after",
        "",
        "| metric | before | after | change |",
        "|---|---:|---:|---:|",
        (
            f"| segments | {fmt_int(g['input_files'])} | "
            f"**{fmt_int(g['kept_segments'])}** | "
            f"{g['kept_segments'] - g['input_files']:+,} |"
        ),
        (
            f"| total frames | {fmt_int(g['input_frames'])} | "
            f"**{fmt_int(g['output_frames'])}** | "
            f"{g['output_frames'] - g['input_frames']:+,} "
            f"({fmt_pct(g['output_frames'] - g['input_frames'], g['input_frames'])}) |"
        ),
        (
            f"| total duration | {g['input_duration_s']/3600:.2f} h | "
            f"**{g['output_duration_s']/3600:.2f} h** | "
            f"{(g['output_duration_s']-g['input_duration_s'])/3600:+.2f} h |"
        ),
        "",
        "## Bad-frame composition",
        "",
        "| type | frames | % of input |",
        "|---|---:|---:|",
        (
            f"| crouch only | {fmt_int(g['crouch_frames']-g['overlap_frames'])} | "
            f"{fmt_pct(g['crouch_frames']-g['overlap_frames'], g['input_frames'])} |"
        ),
        (
            f"| steering only | {fmt_int(g['steering_frames']-g['overlap_frames'])} | "
            f"{fmt_pct(g['steering_frames']-g['overlap_frames'], g['input_frames'])} |"
        ),
        (
            f"| crouch and steering | {fmt_int(g['overlap_frames'])} | "
            f"{fmt_pct(g['overlap_frames'], g['input_frames'])} |"
        ),
        (
            f"| **union removed by mask** | **{fmt_int(g['bad_union_frames'])}** | "
            f"**{fmt_pct(g['bad_union_frames'], g['input_frames'])}** |"
        ),
        "",
        "## Per-category distribution after wash",
        "",
        "| category | input segs | output segs | input frames | output frames | retained | output hours |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in stats["per_category"].items():
        lines.append(
            f"| {category} | {row['input_files']:,} | {row['kept_segments']:,} | "
            f"{row['input_frames']:,} | {row['output_frames']:,} | "
            f"{fmt_pct(row['output_frames'], row['input_frames'])} | "
            f"{row['output_duration_s']/3600:.2f} |"
        )
    lines.append(
        f"| **total** | **{g['input_files']:,}** | **{g['kept_segments']:,}** | "
        f"**{g['input_frames']:,}** | **{g['output_frames']:,}** | "
        f"**{fmt_pct(g['output_frames'], g['input_frames'])}** | "
        f"**{g['output_duration_s']/3600:.2f}** |"
    )
    lines.append("")
    lines.extend(duration_markdown("Input segment duration distribution", stats["input_duration_distribution"]))
    lines.extend(duration_markdown("Output segment duration distribution", stats["output_duration_distribution"]))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="already-preprocessed MorphData_v1 root")
    ap.add_argument("--out-root", required=True,
                    help="staging root for accepted washed JSONL/meta")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--eps-side", type=float, default=0.25)
    ap.add_argument("--eps-mag", type=float, default=0.2)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reject-manifest", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N files total (smoke tests)")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).resolve()
    out_root = pathlib.Path(args.out_root).resolve()
    manifest_path = pathlib.Path(args.manifest).resolve()
    reject_path = pathlib.Path(args.reject_manifest).resolve()
    stats_path = pathlib.Path(args.stats).resolve()
    report_path = pathlib.Path(args.report).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    global_stats = fresh_counter()
    per_category: dict[str, dict] = {}
    input_durations: list[float] = []
    output_durations: list[float] = []
    processed = 0
    started = time.time()

    for category in args.categories:
        cat_stats = fresh_counter()
        files = sorted((data_root / category).glob("*_frames.jsonl"))
        for frames_path in files:
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            try:
                accepted, rejected, file_stats, input_duration = process_file(
                    frames_path,
                    category,
                    out_root,
                    eps_side=args.eps_side,
                    eps_mag=args.eps_mag,
                    min_frames=args.min_frames,
                    min_duration=args.min_duration,
                )
                accepted_rows.extend(accepted)
                rejected_rows.extend(rejected)
                add_counter(cat_stats, file_stats)
                input_durations.append(input_duration)
                output_durations.extend(row["duration_s"] for row in accepted)
            except Exception as exc:
                cat_stats["parse_failures"] += 1
                rejected_rows.append({
                    "seg_id": frames_path.name.removesuffix("_frames.jsonl"),
                    "category": category,
                    "source_clean_jsonl": str(frames_path.resolve()),
                    "reject_stage": "crouch_steering_wash",
                    "reject_reason": "source_parse_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
            if processed % 25 == 0:
                print(
                    f"[{processed}] accepted={len(accepted_rows)} "
                    f"rejected={len(rejected_rows)} frames={global_stats['input_frames'] + cat_stats['input_frames']:,}",
                    flush=True,
                )
        per_category[category] = cat_stats
        add_counter(global_stats, cat_stats)
        if args.limit is not None and processed >= args.limit:
            break

    accepted_rows.sort(key=lambda r: (r["category"], r["source_clean_jsonl"], r["i0"]))
    rejected_rows.sort(key=lambda r: (
        r.get("category", ""),
        r.get("source_clean_jsonl", ""),
        r.get("i0", -1),
    ))
    atomic_jsonl(manifest_path, accepted_rows)
    atomic_jsonl(reject_path, rejected_rows)

    stats = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.time() - started,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "parameters": {
            "eps_side": args.eps_side,
            "eps_mag": args.eps_mag,
            "min_frames": args.min_frames,
            "min_duration": args.min_duration,
            "categories": args.categories,
        },
        "global": global_stats,
        "per_category": per_category,
        "input_duration_distribution": distribution(input_durations),
        "output_duration_distribution": distribution(output_durations),
        "manifest": str(manifest_path),
        "reject_manifest": str(reject_path),
    }
    atomic_json(stats_path, stats)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_name(f".{report_path.name}.tmp")
    report_tmp.write_text(build_report(stats), encoding="utf-8")
    os.replace(report_tmp, report_path)

    print("\n=== crouch/steering wash ===")
    for key, value in global_stats.items():
        print(f"  {key}: {value}")
    print(f"manifest: {manifest_path}")
    print(f"rejects:  {reject_path}")
    print(f"stats:    {stats_path}")
    print(f"report:   {report_path}")


if __name__ == "__main__":
    main()
