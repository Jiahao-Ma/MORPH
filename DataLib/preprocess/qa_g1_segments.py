#!/usr/bin/env python3
"""Validate final G1 manifests/qpos and optionally rescan washed source cmd."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from wash_crouch_steering import atomic_json, atomic_jsonl, classify_line


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_has_bad(path: pathlib.Path, eps_side: float, eps_mag: float) -> tuple[int, int]:
    n = 0
    bad = 0
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            crouch, steering = classify_line(line, eps_side, eps_mag)
            bad += int(crouch or steering)
    return n, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments-manifest", required=True)
    ap.add_argument("--out-stats", required=True)
    ap.add_argument("--out-failures", required=True)
    ap.add_argument("--check-source", action="store_true")
    ap.add_argument("--eps-side", type=float, default=0.25)
    ap.add_argument("--eps-mag", type=float, default=0.2)
    args = ap.parse_args()

    manifest = pathlib.Path(args.segments_manifest).resolve()
    rows = read_jsonl(manifest)
    failures: list[dict] = []
    seen: set[str] = set()
    checked_frames = 0
    checked_source_frames = 0
    started = time.time()

    for index, row in enumerate(rows):
        seg_id = row.get("seg_id")
        reasons = []
        if not seg_id or seg_id in seen:
            reasons.append("missing_or_duplicate_seg_id")
        if seg_id:
            seen.add(seg_id)
        qpos_path = pathlib.Path(row.get("qpos_npy", ""))
        meta_path = pathlib.Path(row.get("seg_meta", ""))
        expected = int(row.get("n_frames", -1))
        if not qpos_path.is_file():
            reasons.append("missing_qpos")
        else:
            try:
                qpos = np.load(qpos_path, mmap_mode="r", allow_pickle=False)
                if qpos.ndim != 2 or qpos.shape[1] != 36:
                    reasons.append(f"qpos_shape={qpos.shape}")
                if qpos.shape[0] != expected:
                    reasons.append(f"length={qpos.shape[0]} expected={expected}")
                if not bool(np.isfinite(qpos).all()):
                    reasons.append("non_finite_qpos")
                checked_frames += int(qpos.shape[0]) if qpos.ndim else 0
            except Exception as exc:
                reasons.append(f"qpos_load_failed:{type(exc).__name__}:{exc}")
        if not meta_path.is_file():
            reasons.append("missing_segment_meta")
        if args.check_source:
            source = pathlib.Path(row.get("washed_jsonl", ""))
            if not source.is_file():
                reasons.append("missing_washed_source")
            else:
                try:
                    source_n, bad_n = source_has_bad(
                        source, args.eps_side, args.eps_mag
                    )
                    checked_source_frames += source_n
                    if source_n != expected:
                        reasons.append(f"source_length={source_n} expected={expected}")
                    if bad_n:
                        reasons.append(f"source_bad_frames={bad_n}")
                except Exception as exc:
                    reasons.append(
                        f"source_scan_failed:{type(exc).__name__}:{exc}"
                    )
        if reasons:
            failures.append({
                "index": index,
                "seg_id": seg_id,
                "category": row.get("category"),
                "reasons": reasons,
            })
        if (index + 1) % 100 == 0:
            print(
                f"[{index+1}/{len(rows)}] failures={len(failures)} "
                f"qpos_frames={checked_frames:,}",
                flush=True,
            )

    stats = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "segments_manifest": str(manifest),
        "segments": len(rows),
        "unique_seg_ids": len(seen),
        "qpos_frames_checked": checked_frames,
        "source_frames_checked": checked_source_frames,
        "check_source": args.check_source,
        "failures": len(failures),
        "elapsed_s": time.time() - started,
        "status": "ok" if not failures else "failed",
    }
    atomic_json(pathlib.Path(args.out_stats).resolve(), stats)
    atomic_jsonl(pathlib.Path(args.out_failures).resolve(), failures)
    print(json.dumps(stats, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
