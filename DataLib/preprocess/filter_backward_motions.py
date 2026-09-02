#!/usr/bin/env python3
"""Build an immutable forward-only subset from the washed Stage 2 dataset.

The capture schema stores keyboard movement as::

    cmd.move = [right, forward]

Therefore the keyboard ``S`` command is represented by a negative forward
component, not by a literal ``"cmd": "s"`` string. This stage rejects an
entire washed motion when *any* frame satisfies::

    cmd.move[1] < -eps_back

Rejecting the whole motion (instead of cutting only the backward frames)
ensures that every accepted training sequence is forward-perception
compatible. The input Stage 2 and Stage 3 directories are never modified.
Accepted Stage 2 JSONL/meta pairs are copied to a new directory. When a Stage
3 manifest/output directory is supplied, the already-retargeted qpos for the
same accepted segment IDs is copied as well; retargeting is not repeated
because an unchanged accepted motion has unchanged qpos.
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
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np


MOVE_RE = re.compile(
    rb'"move"\s*:\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\]'
)
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
    """Copy without replacing a completed immutable output."""
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


def scan_motion(
    path: pathlib.Path, *, eps_back: float
) -> dict[str, int | float | None]:
    frame_count = 0
    backward_frames = 0
    first_backward_frame: int | None = None
    last_backward_frame: int | None = None
    min_forward = math.inf
    max_forward = -math.inf

    with path.open("rb") as f:
        for raw_line in f:
            if raw_line in (b"", b"\n", b"\r\n"):
                continue
            # cmd.move is near the start of every frame. Searching the original
            # bytes avoids allocating an 8 KiB prefix copy for every frame.
            match = MOVE_RE.search(raw_line)
            if match is None:
                try:
                    row = json.loads(raw_line)
                    move = (row.get("cmd") or {}).get("move", [0.0, 0.0])
                    forward = float(move[1])
                except Exception as exc:
                    raise ValueError(
                        f"{path}: frame {frame_count} has no valid cmd.move"
                    ) from exc
            else:
                forward = float(match.group(2).strip())

            min_forward = min(min_forward, forward)
            max_forward = max(max_forward, forward)
            if forward < -eps_back:
                backward_frames += 1
                if first_backward_frame is None:
                    first_backward_frame = frame_count
                last_backward_frame = frame_count
            frame_count += 1

    if frame_count == 0:
        raise ValueError(f"empty motion: {path}")
    return {
        "frame_count": frame_count,
        "backward_frames": backward_frames,
        "first_backward_frame": first_backward_frame,
        "last_backward_frame": last_backward_frame,
        "min_forward_cmd": min_forward,
        "max_forward_cmd": max_forward,
    }


def fresh_counter() -> dict[str, int | float]:
    return {
        "input_motions": 0,
        "input_frames": 0,
        "input_duration_s": 0.0,
        "kept_motions": 0,
        "kept_frames": 0,
        "kept_duration_s": 0.0,
        "backward_motions": 0,
        "backward_motion_frames": 0,
        "backward_motion_duration_s": 0.0,
        "backward_frames": 0,
        "copied_jsonl": 0,
        "reused_jsonl": 0,
    }


def add_counter(
    destination: dict[str, int | float],
    source: dict[str, int | float],
) -> None:
    for key, value in source.items():
        destination[key] += value


def duration_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def copy_stage2_meta(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    source_jsonl: pathlib.Path,
    output_jsonl: pathlib.Path,
    eps_back: float,
) -> None:
    meta = read_json(source)
    meta["forward_only"] = True
    meta["backward_motion_filter"] = {
        "version": FILTER_VERSION,
        "rule": "reject entire motion if any cmd.move[1] < -eps_back",
        "eps_back": eps_back,
        "source_jsonl": str(source_jsonl.resolve()),
        "source_meta": str(source.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
    }
    atomic_json(destination, meta)


def subset_g1(
    *,
    input_manifest: pathlib.Path,
    output_root: pathlib.Path,
    accepted_stage2: list[dict[str, Any]],
    rejected_ids: set[str],
) -> dict[str, Any]:
    input_rows = read_jsonl(input_manifest)
    by_id: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        seg_id = str(row["seg_id"])
        if seg_id in by_id:
            raise ValueError(f"duplicate G1 seg_id: {seg_id}")
        by_id[seg_id] = row

    accepted_ids = {str(row["seg_id"]) for row in accepted_stage2}
    missing = sorted(accepted_ids - by_id.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} forward Stage 2 motions have no G1 qpos; "
            f"first: {missing[:5]}"
        )

    output_rows: list[dict[str, Any]] = []
    copied = reused = frames = 0
    for index, stage2_row in enumerate(accepted_stage2, 1):
        seg_id = str(stage2_row["seg_id"])
        source_row = by_id[seg_id]
        source_qpos = pathlib.Path(source_row["qpos_npy"])
        if not source_qpos.is_file():
            raise FileNotFoundError(f"missing G1 qpos: {source_qpos}")
        qpos = np.load(source_qpos, mmap_mode="r", allow_pickle=False)
        expected = int(stage2_row["n_frames"])
        if qpos.shape != (expected, 36):
            raise ValueError(
                f"{seg_id}: G1 qpos shape {qpos.shape}, expected ({expected}, 36)"
            )

        category = str(stage2_row["category"])
        output_qpos = output_root / category / source_qpos.name
        copy_status = copy_atomic(source_qpos, output_qpos)
        copied += int(copy_status == "copied")
        reused += int(copy_status == "existing")

        output_meta = output_qpos.with_name(
            output_qpos.name.removesuffix("_frames.npy") + "_meta.json"
        )
        result = dict(source_row)
        result.update(
            {
                "qpos_npy": str(output_qpos.resolve()),
                "seg_meta": str(output_meta.resolve()),
                "washed_jsonl": stage2_row["washed_jsonl"],
                "washed_meta": stage2_row["washed_meta"],
                "status": "qa_ok_forward_only",
                "forward_only": {
                    "version": FILTER_VERSION,
                    "source_g1_manifest": str(input_manifest.resolve()),
                    "source_qpos_npy": str(source_qpos.resolve()),
                    "rule": "Stage 2 motion contains no cmd.move[1] < -eps_back",
                },
            }
        )
        atomic_json(output_meta, result)
        output_rows.append(result)
        frames += expected
        if index == 1 or index % 250 == 0 or index == len(accepted_stage2):
            print(
                f"  [G1 {index:>5}/{len(accepted_stage2)}] "
                f"{seg_id} ({frames:,} frames)"
            )

    unexpected = sorted(
        seg_id
        for seg_id in by_id
        if seg_id not in accepted_ids and seg_id not in rejected_ids
    )
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} G1 rows are absent from the Stage 2 decision set"
        )

    manifest_path = output_root / "segments_manifest.jsonl"
    reject_path = output_root / "backward_reject_manifest.jsonl"
    rejected_g1 = [
        {
            "seg_id": seg_id,
            "category": by_id[seg_id].get("category"),
            "qpos_npy": by_id[seg_id].get("qpos_npy"),
            "reject_stage": "backward_motion_filter",
            "reject_reason": "contains_backward_cmd",
        }
        for seg_id in sorted(rejected_ids)
        if seg_id in by_id
    ]
    atomic_jsonl(manifest_path, output_rows)
    atomic_jsonl(reject_path, rejected_g1)
    return {
        "input_manifest": str(input_manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "reject_manifest": str(reject_path.resolve()),
        "input_motions": len(input_rows),
        "output_motions": len(output_rows),
        "rejected_motions": len(rejected_g1),
        "output_frames": frames,
        "copied_qpos": copied,
        "reused_qpos": reused,
    }


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0.0%"
    return f"{100.0 * float(numerator) / float(denominator):.1f}%"


def build_report(stats: dict[str, Any]) -> str:
    global_stats = stats["global"]
    lines = [
        "# Forward-only motion filter on MorphData_v1",
        "",
        f"Generated: {stats['generated_at']}",
        "",
        "Rule:",
        "",
        (
            f"- reject the **entire motion** when any frame has "
            f"`cmd.move[1] < -{stats['parameters']['eps_back']}`"
        ),
        "- `cmd.move` follows `[right, forward]`; negative forward is keyboard S.",
        "- accepted motions are copied to a new directory; the input is untouched.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input motions | {fmt_int(global_stats['input_motions'])} |",
        f"| input frames | {fmt_int(global_stats['input_frames'])} |",
        (
            f"| **kept forward-only motions** | "
            f"**{fmt_int(global_stats['kept_motions'])}** "
            f"({fmt_pct(global_stats['kept_motions'], global_stats['input_motions'])}) |"
        ),
        (
            f"| **kept frames** | **{fmt_int(global_stats['kept_frames'])}** "
            f"({fmt_pct(global_stats['kept_frames'], global_stats['input_frames'])}) |"
        ),
        (
            f"| **kept duration** | "
            f"**{global_stats['kept_duration_s']/3600.0:.2f} h** |"
        ),
        f"| rejected motions containing S | {fmt_int(global_stats['backward_motions'])} |",
        (
            f"| all frames removed with those motions | "
            f"{fmt_int(global_stats['backward_motion_frames'])} |"
        ),
        f"| actual backward-command frames | {fmt_int(global_stats['backward_frames'])} |",
        "",
        "## Per-category",
        "",
        "| category | input motions | kept | rejected | input frames | kept frames | retained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in stats["per_category"].items():
        lines.append(
            f"| {category} | {row['input_motions']:,} | "
            f"{row['kept_motions']:,} | {row['backward_motions']:,} | "
            f"{row['input_frames']:,} | {row['kept_frames']:,} | "
            f"{fmt_pct(row['kept_frames'], row['input_frames'])} |"
        )
    if stats.get("g1"):
        g1 = stats["g1"]
        lines.extend(
            [
                "",
                "## Stage 3 G1 subset",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| input qpos motions | {g1['input_motions']:,} |",
                f"| output qpos motions | **{g1['output_motions']:,}** |",
                f"| rejected qpos motions | {g1['rejected_motions']:,} |",
                f"| output qpos frames | **{g1['output_frames']:,}** |",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help="default: <output-root>/forward_manifest.jsonl",
    )
    parser.add_argument(
        "--reject-manifest",
        type=pathlib.Path,
        default=None,
        help="default: <output-root>/backward_reject_manifest.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=pathlib.Path,
        default=None,
        help="default: <output-root>/forward_only_stats.json",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=None,
        help="default: <output-root>/forward_only_report.md",
    )
    parser.add_argument(
        "--eps-back",
        type=float,
        default=0.2,
        help="reject when cmd.move[1] is below -eps-back (default: 0.2)",
    )
    parser.add_argument(
        "--g1-manifest",
        type=pathlib.Path,
        default=None,
        help="optional existing Stage 3 manifest to subset without re-retargeting",
    )
    parser.add_argument(
        "--g1-output-root",
        type=pathlib.Path,
        default=None,
        help="required with --g1-manifest",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="classify and report without copying Stage 2/G1 artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process the first N manifest rows for a smoke test",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.eps_back < 0:
        parser.error("--eps-back must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if (args.g1_manifest is None) != (args.g1_output_root is None):
        parser.error("--g1-manifest and --g1-output-root must be used together")
    if args.scan_only and args.g1_manifest is not None:
        parser.error("--scan-only cannot be combined with --g1-manifest")
    if args.limit is not None and args.g1_manifest is not None:
        parser.error("--limit cannot be combined with --g1-manifest")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    input_manifest = args.input_manifest.resolve()
    if not input_manifest.is_file():
        parser.error(f"--input-manifest does not exist: {input_manifest}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output_root / "forward_manifest.jsonl"
    )
    reject_path = (
        args.reject_manifest.resolve()
        if args.reject_manifest
        else output_root / "backward_reject_manifest.jsonl"
    )
    stats_path = (
        args.stats.resolve()
        if args.stats
        else output_root / "forward_only_stats.json"
    )
    report_path = (
        args.report.resolve()
        if args.report
        else output_root / "forward_only_report.md"
    )

    rows = read_jsonl(input_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(
        f"[Input] {len(rows):,} Stage 2 motion(s); "
        f"reject whole motion if cmd.move[1] < -{args.eps_back:g}"
    )

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    stage2_by_id: dict[str, dict[str, Any]] = {}
    global_stats = fresh_counter()
    per_category: dict[str, dict[str, int | float]] = defaultdict(fresh_counter)
    kept_durations: list[float] = []
    rejected_durations: list[float] = []
    started = time.monotonic()

    for index, row in enumerate(rows, 1):
        seg_id = str(row["seg_id"])
        if seg_id in stage2_by_id:
            raise ValueError(f"duplicate Stage 2 seg_id: {seg_id}")
        stage2_by_id[seg_id] = row
        category = str(row["category"])
        source_jsonl = pathlib.Path(row["washed_jsonl"])
        source_meta = pathlib.Path(row["washed_meta"])
        if not source_jsonl.is_file() or not source_meta.is_file():
            raise FileNotFoundError(
                f"{seg_id}: missing Stage 2 JSONL/meta pair"
            )

        scan = scan_motion(source_jsonl, eps_back=args.eps_back)
        expected_frames = int(row["n_frames"])
        actual_frames = int(scan["frame_count"])
        if actual_frames != expected_frames:
            raise ValueError(
                f"{seg_id}: JSONL has {actual_frames} frames, "
                f"manifest says {expected_frames}"
            )
        duration_s = float(row["duration_s"])
        rejected = int(scan["backward_frames"]) > 0
        file_stats = fresh_counter()
        file_stats["input_motions"] = 1
        file_stats["input_frames"] = actual_frames
        file_stats["input_duration_s"] = duration_s
        file_stats["backward_frames"] = int(scan["backward_frames"])

        if rejected:
            file_stats["backward_motions"] = 1
            file_stats["backward_motion_frames"] = actual_frames
            file_stats["backward_motion_duration_s"] = duration_s
            rejected_durations.append(duration_s)
            rejected_rows.append(
                {
                    "seg_id": seg_id,
                    "category": category,
                    "source_jsonl": str(source_jsonl.resolve()),
                    "source_meta": str(source_meta.resolve()),
                    "n_frames": actual_frames,
                    "duration_s": duration_s,
                    "reject_stage": "backward_motion_filter",
                    "reject_reason": "contains_backward_cmd",
                    "detail": (
                        f"backward_frames={scan['backward_frames']}, "
                        f"first={scan['first_backward_frame']}, "
                        f"last={scan['last_backward_frame']}, "
                        f"min_cmd.move[1]={scan['min_forward_cmd']}"
                    ),
                    "filter": {
                        "version": FILTER_VERSION,
                        "eps_back": args.eps_back,
                        "rule": (
                            "reject entire motion if any "
                            "cmd.move[1] < -eps_back"
                        ),
                    },
                }
            )
        else:
            file_stats["kept_motions"] = 1
            file_stats["kept_frames"] = actual_frames
            file_stats["kept_duration_s"] = duration_s
            kept_durations.append(duration_s)
            result = dict(row)
            output_jsonl = output_root / category / source_jsonl.name
            output_meta = output_root / category / source_meta.name
            if not args.scan_only:
                copy_status = copy_atomic(source_jsonl, output_jsonl)
                file_stats["copied_jsonl"] = int(copy_status == "copied")
                file_stats["reused_jsonl"] = int(copy_status == "existing")
                copy_stage2_meta(
                    source_meta,
                    output_meta,
                    source_jsonl=source_jsonl,
                    output_jsonl=output_jsonl,
                    eps_back=args.eps_back,
                )
            result.update(
                {
                    "washed_jsonl": str(output_jsonl.resolve()),
                    "washed_meta": str(output_meta.resolve()),
                    "status": "forward_only",
                    "forward_only": {
                        "version": FILTER_VERSION,
                        "eps_back": args.eps_back,
                        "source_jsonl": str(source_jsonl.resolve()),
                        "source_meta": str(source_meta.resolve()),
                        "rule": (
                            "motion contains no frame with "
                            "cmd.move[1] < -eps_back"
                        ),
                    },
                }
            )
            accepted_rows.append(result)

        add_counter(global_stats, file_stats)
        add_counter(per_category[category], file_stats)
        if index == 1 or index % 50 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"  [{index:>5}/{len(rows)}] kept="
                f"{int(global_stats['kept_motions']):,} rejected="
                f"{int(global_stats['backward_motions']):,} "
                f"({elapsed:.1f}s)"
            )

    stats: dict[str, Any] = {
        "schema_version": 1,
        "filter_version": FILTER_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_manifest": str(input_manifest),
        "output_root": str(output_root),
        "parameters": {
            "eps_back": args.eps_back,
            "rule": "reject entire motion if any cmd.move[1] < -eps_back",
            "scan_only": args.scan_only,
            "limit": args.limit,
        },
        "global": global_stats,
        "per_category": dict(per_category),
        "kept_duration_distribution": duration_summary(kept_durations),
        "rejected_duration_distribution": duration_summary(rejected_durations),
    }

    rejected_ids = {str(row["seg_id"]) for row in rejected_rows}
    if args.g1_manifest is not None:
        print("[G1] copying the accepted qpos subset")
        stats["g1"] = subset_g1(
            input_manifest=args.g1_manifest.resolve(),
            output_root=args.g1_output_root.resolve(),
            accepted_stage2=accepted_rows,
            rejected_ids=rejected_ids,
        )

    atomic_jsonl(manifest_path, accepted_rows)
    atomic_jsonl(reject_path, rejected_rows)
    atomic_json(stats_path, stats)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_temp = report_path.with_name(f".{report_path.name}.tmp")
    report_temp.write_text(build_report(stats), encoding="utf-8")
    os.replace(report_temp, report_path)

    print(
        f"[Done] kept {int(global_stats['kept_motions']):,}/"
        f"{int(global_stats['input_motions']):,} motions and "
        f"{int(global_stats['kept_frames']):,}/"
        f"{int(global_stats['input_frames']):,} frames"
    )
    print(f"[Manifest] {manifest_path}")
    print(f"[Report] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
