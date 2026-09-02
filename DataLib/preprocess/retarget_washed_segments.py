#!/usr/bin/env python3
"""Retarget accepted crouch/steering-washed segments to Unitree G1 qpos."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
RETARGET = REPO / "DataLib" / "retarget" / "ue_world_skeleton_retarget.py"
CONFIG = REPO / "DataLib" / "retarget" / "gasp_bvh_alignment_g1_height.json"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def infer_data_scale(mesh_scale: dict, tolerance: float = 0.05) -> tuple[float, float]:
    mins = mesh_scale.get("min")
    maxs = mesh_scale.get("max")
    missing = int(mesh_scale.get("missing_frames", 0))
    if not mins or not maxs or missing:
        raise ValueError(f"mesh scale missing in {missing} frame(s)")
    values = [float(v) for v in mins + maxs]
    if max(values) - min(values) > 0.02:
        raise ValueError(f"mesh scale drifts: min={mins}, max={maxs}")
    observed = sum(values) / len(values)
    if abs(observed - 0.77) <= tolerance:
        return observed, 1.0
    if abs(observed - 1.0) <= tolerance:
        return observed, 0.77
    raise ValueError(f"unsupported mesh scale {observed:.6f}")


def qa_qpos(path: pathlib.Path, expected_frames: int) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "qpos output is missing or empty"
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return False, f"cannot load qpos: {type(exc).__name__}: {exc}"
    if arr.ndim != 2 or arr.shape[1] != 36:
        return False, f"qpos shape {arr.shape}, expected (T, 36)"
    if arr.shape[0] != expected_frames:
        return False, f"qpos T={arr.shape[0]}, source T={expected_frames}"
    # np.isfinite on an mmap still streams and avoids a second resident copy.
    if not bool(np.isfinite(arr).all()):
        return False, "qpos contains NaN or Inf"
    return True, "ok"


@dataclass
class Task:
    index: int
    row: dict
    out_qpos: pathlib.Path
    out_meta: pathlib.Path
    observed_scale: float
    data_scale: float


def make_reject(row: dict, reason: str, detail: str) -> dict:
    return {
        "seg_id": row.get("seg_id"),
        "category": row.get("category"),
        "source_clean_jsonl": row.get("source_clean_jsonl"),
        "washed_jsonl": row.get("washed_jsonl"),
        "i0": row.get("i0"),
        "i1": row.get("i1"),
        "n_frames": row.get("n_frames"),
        "reject_stage": "retarget",
        "reject_reason": reason,
        "detail": detail,
    }


def run_task(
    task: Task,
    *,
    timeout: int,
    height_from_data: bool,
    per_foot_ground: bool,
    strict_grounding: bool,
    skip_existing: bool,
) -> tuple[int, dict | None, dict | None, float]:
    row = task.row
    started = time.time()
    if skip_existing:
        ok, detail = qa_qpos(task.out_qpos, int(row["n_frames"]))
        if ok:
            success = dict(row)
            success.update({
                "qpos_npy": str(task.out_qpos.resolve()),
                "seg_meta": str(task.out_meta.resolve()),
                "mesh_scale_observed": task.observed_scale,
                "data_scale": task.data_scale,
                "status": "qa_ok",
                "retarget_skipped_existing": True,
            })
            if not task.out_meta.exists():
                atomic_json(task.out_meta, success)
            return task.index, success, None, time.time() - started

    task.out_qpos.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RETARGET),
        "--jsonl", str(row["washed_jsonl"]),
        "--meta", str(row["washed_meta"]),
        "--config", str(CONFIG),
        "--src-human", "bvh_ue5_g1scale",
        "--data-scale", str(task.data_scale),
        "--output-qpos", str(task.out_qpos),
        "--no-visualize",
        "--no-terrain",
        "--no-output-cmd",
        "--no-output-scaled",
    ]
    if height_from_data:
        cmd.append("--height-from-data")
    if per_foot_ground:
        cmd.append("--per-foot-ground")

    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        task.out_qpos.unlink(missing_ok=True)
        return (
            task.index,
            None,
            make_reject(row, "retarget_timeout", f"timeout after {timeout}s: {exc}"),
            time.time() - started,
        )
    except Exception as exc:
        task.out_qpos.unlink(missing_ok=True)
        return (
            task.index,
            None,
            make_reject(row, "retarget_failed", f"{type(exc).__name__}: {exc}"),
            time.time() - started,
        )

    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        task.out_qpos.unlink(missing_ok=True)
        tail = "\n".join(combined.strip().splitlines()[-20:])
        return (
            task.index,
            None,
            make_reject(row, "retarget_failed", f"exit={proc.returncode}\n{tail}"),
            time.time() - started,
        )
    critical_failures = [
        marker
        for marker in ("[S5 pre-IK grounding] FAILED", "[S5 grounding] FAILED")
        if marker in combined
    ]
    if strict_grounding and critical_failures:
        task.out_qpos.unlink(missing_ok=True)
        return (
            task.index,
            None,
            make_reject(
                row,
                "retarget_degraded",
                f"critical internal stage failed: {', '.join(critical_failures)}",
            ),
            time.time() - started,
        )

    ok, detail = qa_qpos(task.out_qpos, int(row["n_frames"]))
    if not ok:
        task.out_qpos.unlink(missing_ok=True)
        reason = "qpos_shape_mismatch" if "shape" in detail else (
            "length_mismatch" if "source T" in detail else "non_finite_qpos"
        )
        return (
            task.index,
            None,
            make_reject(row, reason, detail),
            time.time() - started,
        )

    success = dict(row)
    success.update({
        "qpos_npy": str(task.out_qpos.resolve()),
        "seg_meta": str(task.out_meta.resolve()),
        "mesh_scale_observed": task.observed_scale,
        "data_scale": task.data_scale,
        "status": "qa_ok",
        "retarget_elapsed_s": time.time() - started,
        "retarget_warnings": critical_failures,
    })
    atomic_json(task.out_meta, success)
    return task.index, success, None, time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wash-manifest", required=True)
    ap.add_argument("--wash-reject-manifest", default=None)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-height-from-data", action="store_true")
    ap.add_argument("--no-per-foot-ground", action="store_true")
    ap.add_argument("--allow-grounding-fallback", action="store_true",
                    help="accept qpos even when a critical grounding stage logs FAILED")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    wash_manifest = pathlib.Path(args.wash_manifest).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_rows = read_jsonl(wash_manifest)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    pre_rejects = []
    tasks: list[Task] = []
    for index, row in enumerate(input_rows):
        try:
            observed, data_scale = infer_data_scale(row.get("mesh_scale") or {})
        except Exception as exc:
            pre_rejects.append((
                index,
                make_reject(row, "invalid_or_unknown_data_scale", str(exc)),
            ))
            continue
        category = row["category"]
        seg_id = row["seg_id"]
        out_cat = output_root / category
        tasks.append(Task(
            index=index,
            row=row,
            out_qpos=out_cat / f"{seg_id}_frames.npy",
            out_meta=out_cat / f"{seg_id}_meta.json",
            observed_scale=observed,
            data_scale=data_scale,
        ))

    successes: list[tuple[int, dict]] = []
    rejects: list[tuple[int, dict]] = list(pre_rejects)
    timings: list[float] = []
    started = time.time()
    print(
        f"retarget tasks={len(tasks)}, pre-rejected={len(pre_rejects)}, "
        f"workers={args.workers}",
        flush=True,
    )
    kwargs = {
        "timeout": args.timeout,
        "height_from_data": not args.no_height_from_data,
        "per_foot_ground": not args.no_per_foot_ground,
        "strict_grounding": not args.allow_grounding_fallback,
        "skip_existing": args.skip_existing,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(run_task, task, **kwargs): task
            for task in tasks
        }
        done = 0
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            done += 1
            try:
                index, success, reject, elapsed = future.result()
            except Exception as exc:
                index = task.index
                success = None
                reject = make_reject(
                    task.row, "retarget_worker_failed", f"{type(exc).__name__}: {exc}"
                )
                elapsed = 0.0
            timings.append(elapsed)
            if success is not None:
                successes.append((index, success))
            if reject is not None:
                rejects.append((index, reject))
            if done % 10 == 0 or done == len(tasks):
                print(
                    f"[{done}/{len(tasks)}] ok={len(successes)} "
                    f"reject={len(rejects)} last={elapsed:.1f}s",
                    flush=True,
                )

    # Include wash-stage rejects in the final audit manifest.
    wash_reject_rows = []
    if args.wash_reject_manifest:
        p = pathlib.Path(args.wash_reject_manifest)
        if p.exists():
            wash_reject_rows = read_jsonl(p)

    successes.sort(key=lambda item: item[0])
    rejects.sort(key=lambda item: item[0])
    success_rows = [row for _, row in successes]
    reject_rows = wash_reject_rows + [row for _, row in rejects]
    segments_manifest = output_root / "segments_manifest.jsonl"
    reject_manifest = output_root / "reject_manifest.jsonl"
    atomic_jsonl(segments_manifest, success_rows)
    atomic_jsonl(reject_manifest, reject_rows)

    per_category: dict[str, dict] = {}
    for row in success_rows:
        cat = row["category"]
        item = per_category.setdefault(cat, {"segments": 0, "frames": 0, "duration_s": 0.0})
        item["segments"] += 1
        item["frames"] += int(row["n_frames"])
        item["duration_s"] += float(row.get("duration_s", 0.0))
    stats = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wash_manifest": str(wash_manifest),
        "output_root": str(output_root),
        "parameters": {
            "workers": args.workers,
            "timeout": args.timeout,
            "height_from_data": not args.no_height_from_data,
            "per_foot_ground": not args.no_per_foot_ground,
            "strict_grounding": not args.allow_grounding_fallback,
        },
        "input_segments": len(input_rows),
        "success_segments": len(success_rows),
        "retarget_rejects": len(rejects),
        "wash_rejects": len(wash_reject_rows),
        "success_frames": sum(int(row["n_frames"]) for row in success_rows),
        "success_duration_h": sum(float(row.get("duration_s", 0.0)) for row in success_rows) / 3600.0,
        "elapsed_s": time.time() - started,
        "task_elapsed_s_sum": sum(timings),
        "task_elapsed_s_mean": (sum(timings) / len(timings)) if timings else 0.0,
        "per_category": per_category,
        "segments_manifest": str(segments_manifest),
        "reject_manifest": str(reject_manifest),
    }
    stats_path = output_root / "retarget_stats.json"
    atomic_json(stats_path, stats)
    print("\n=== retarget summary ===")
    print(f"success: {len(success_rows)}/{len(input_rows)}")
    print(f"frames:  {stats['success_frames']:,}")
    print(f"elapsed: {stats['elapsed_s']/3600:.2f} h")
    print(f"stats:   {stats_path}")


if __name__ == "__main__":
    main()
