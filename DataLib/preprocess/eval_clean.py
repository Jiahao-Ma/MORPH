#!/usr/bin/env python3
"""Evaluate a (cleaned) MorphData_v1 dataset: disp/dt speed thresholds, dt
distribution, and per-category duration. Scans every frame of every segment.

Outputs a JSON summary and prints a human-readable table. Use this before
and after cleaning to verify the artifacts are gone.
"""
import os
import re
import glob
import math
import json
import argparse

import numpy as np

DEFAULT_CATEGORIES = [
    "ground", "stairs",
    "traversal_mantle", "traversal_mantle_vault", "traversal_vault",
]
DEFAULT_THRESHOLDS = [6.0, 8.0, 10.0, 15.0, 20.0, 30.0]


def parse_t_p(line: bytes):
    t = None
    m = re.search(rb'"t":\s*([-\d.]+)', line)
    if m:
        t = float(m.group(1))
    p = None
    m = re.search(rb'"p":\s*\[([^\]]*)\]', line)
    if m:
        try:
            p = [float(x) for x in m.group(1).split(b',')]
        except ValueError:
            p = None
    return t, p


def scan_file(path, thresholds):
    counts = {th: 0 for th in thresholds}
    max_sp = 0.0
    n = 0
    dur = 0.0
    n_stuck = 0
    n_gap_small = 0
    n_gap_big = 0
    dts = []
    prev_t = None
    prev_p = None
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            t, p = parse_t_p(line)
            if (prev_t is not None and prev_p is not None
                    and p is not None and t is not None):
                dt = t - prev_t
                dts.append(dt)
                dur += dt
                if dt < 0.005:
                    n_stuck += 1
                elif dt > 0.1:
                    n_gap_big += 1
                elif dt > 0.05:
                    n_gap_small += 1
                if dt > 1e-4:
                    disp = math.sqrt((p[0] - prev_p[0]) ** 2 +
                                     (p[1] - prev_p[1]) ** 2 +
                                     (p[2] - prev_p[2]) ** 2)
                    sp = disp / dt / 100.0
                    if sp > max_sp:
                        max_sp = sp
                    for th in thresholds:
                        if sp > th:
                            counts[th] += 1
            prev_t = t
            prev_p = p
    med_dt = float(np.median(dts)) if dts else 0.0
    max_dt = float(max(dts)) if dts else 0.0
    return dict(n=n, dur=dur, max_sp=max_sp, counts=counts,
                med_dt=med_dt, max_dt=max_dt,
                n_stuck=n_stuck, n_gap_small=n_gap_small, n_gap_big=n_gap_big)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/MorphData_v1")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--out-json", default="data/MorphData_v1_eval.json")
    args = ap.parse_args()
    thresholds = args.thresholds

    res = {}
    tot_files = 0
    tot_frames = 0
    tot_dur = 0.0
    all_max, all_med_dt, all_max_dt = [], [], []
    g_stuck = g_gap_s = g_gap_b = 0
    g_counts = {th: 0 for th in thresholds}
    for cat in args.categories:
        files = sorted(glob.glob(os.path.join(args.data_root, cat, "*_frames.jsonl")))
        res[cat] = {"files": 0, "frames": 0, "dur": 0.0, "max_sp": 0.0,
                    "counts": {th: 0 for th in thresholds}}
        for fp in files:
            r = scan_file(fp, thresholds)
            tot_files += 1
            res[cat]["files"] += 1
            tot_frames += r["n"]
            res[cat]["frames"] += r["n"]
            tot_dur += r["dur"]
            res[cat]["dur"] += r["dur"]
            all_max.append(r["max_sp"])
            all_med_dt.append(r["med_dt"])
            all_max_dt.append(r["max_dt"])
            g_stuck += r["n_stuck"]
            g_gap_s += r["n_gap_small"]
            g_gap_b += r["n_gap_big"]
            if r["max_sp"] > res[cat]["max_sp"]:
                res[cat]["max_sp"] = r["max_sp"]
            for th in thresholds:
                res[cat]["counts"][th] += r["counts"][th]
                g_counts[th] += r["counts"][th]
        print(f"  {cat}: {res[cat]['files']} files, {res[cat]['frames']} frames, "
              f"{res[cat]['dur']/3600:.2f}h, max_sp={res[cat]['max_sp']:.1f} m/s",
              flush=True)
    summary = {
        "total_files": tot_files, "total_frames": tot_frames,
        "total_dur_h": tot_dur / 3600.0,
        "global_max_sp": max(all_max) if all_max else 0.0,
        "median_of_med_dt": float(np.median(all_med_dt)) if all_med_dt else 0.0,
        "median_of_max_dt": float(np.median(all_max_dt)) if all_max_dt else 0.0,
        "max_dt_global": max(all_max_dt) if all_max_dt else 0.0,
        "n_stuck": g_stuck, "n_gap_small": g_gap_s, "n_gap_big": g_gap_b,
        "threshold_counts": g_counts,
        "per_cat": res,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print("\n=== eval summary ===")
    print(f"total files(segments): {tot_files}")
    print(f"total frames: {tot_frames}")
    print(f"total duration: {tot_dur/3600:.2f} h")
    print(f"global max disp/dt speed: {summary['global_max_sp']:.1f} m/s")
    print(f"median per-file median_dt: {summary['median_of_med_dt']:.5f} s")
    print(f"global max_dt: {summary['max_dt_global']:.4f} s")
    print(f"stuck frames (dt<0.005): {g_stuck}")
    print(f"small gaps (0.05<dt<=0.1): {g_gap_s}")
    print(f"big gaps (dt>0.1): {g_gap_b}")
    print("threshold table (disp/dt frames above):")
    for th in thresholds:
        print(f"  >{th:>4.0f} m/s: {g_counts[th]} frames")
    print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
