#!/usr/bin/env python3
"""Clean MorphData_v1 motion clips (dt-gap + disp/dt-speed rules).

This is the offline preprocessing step that should be run on raw MorphData_v1
captures *before* retargeting/training. The raw captures contain two pervasive
logging/capture artifacts that poison downstream motion:

  1. "Stuck frames"  - identical-position duplicate frames written within a
     single engine tick (dt ~ 1e-5 s). ~47% of all raw frames are these.
  2. "Big gaps"      - multi-threaded capture instability drops frames for
     0.1 s+ at a time, which inflates per-frame displacement speed.

It also removes single-frame position "glitches" and splits true "teleports".

Per frame k (with predecessor):

  dt < 0.005s            -> stuck frame: drop (dedup)
  dt > 0.1s              -> big gap: SPLIT point (keep k as new segment start)
  0.05 < dt <= 0.1s      -> small gap: keep k, interpolate missing frames
  0.005<dt<0.05 and disp/dt>10 m/s -> mutation:
       return-speed |p[k+1]-p[k-1]|/(t[k+1]-t[k-1]) < 10 m/s -> glitch: drop k
       else -> teleport: SPLIT point (keep k as new segment start)
  else                  -> normal: keep

Segment filter: keep if n_frames >= MIN_FRAMES AND sum(dt) >= MIN_DUR.

Output: cleaned segments are *staged* to <out-root>/<cat>/<base>_c<j>_frames.jsonl
(+ copied/updated _meta.json). Originals are NOT touched by this script; use
apply_clean.py to swap staged segments into the dataset (with backup + rollback).

Speed is always computed from the real dt = t[k]-t[k-1] (never the assumed
1/60). disp/dt = |root.p[k]-root.p[k-1]| / dt / 100  (m/s; root.p is UE cm).
"""
import os
import re
import math
import json
import glob
import copy
import argparse

import numpy as np

DEFAULT_CATEGORIES = [
    "ground", "stairs",
    "traversal_mantle", "traversal_mantle_vault", "traversal_vault",
]

STUCK_DT = 0.005
BIG_GAP_DT = 0.1
SMALL_GAP_DT = 0.05
MUT_MPS = 10.0
GLITCH_RET_MPS = 10.0
MIN_FRAMES = 60
MIN_DUR = 1.0
TARGET_DT = 1.0 / 60.0
QUAT_KEYS = {"q", "wq", "lq", "cq"}


def parse_t_p(line: bytes):
    """Fast byte-level parse of `t` and `root.p` from a single JSONL line.

    Robust to the whitespace variations produced by json.dumps (spaces after
    ':' and inside arrays). Returns (t, p) where p is a list of floats or None.
    """
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


def slerp(q0, q1, u):
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        r = a + u * (b - a)
        n = np.linalg.norm(r)
        return (r / n).tolist() if n > 0 else a.tolist()
    theta = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(theta)
    r = (math.sin((1 - u) * theta) / s) * a + (math.sin(u * theta) / s) * b
    return r.tolist()


def interp(v0, v1, u, key=None):
    """Recursively interpolate two frame field values.

    Lists of length 4 under a quaternion key (q/wq/lq/cq) are slerp'd; other
    equal-length numeric lists are linearly interpolated; nested dicts are
    interpolated key-by-key; bools snap; ints snap; everything else is copied.
    """
    if isinstance(v0, dict) and isinstance(v1, dict):
        out = {}
        for k in v0:
            if k in v1:
                out[k] = interp(v0[k], v1[k], u, k)
            else:
                out[k] = copy.deepcopy(v0[k])
        for k in v1:
            if k not in v0:
                out[k] = copy.deepcopy(v1[k])
        return out
    if isinstance(v0, list) and isinstance(v1, list) and len(v0) == len(v1):
        if key in QUAT_KEYS and len(v0) == 4:
            return slerp(v0, v1, u)
        try:
            a = np.asarray(v0, dtype=np.float64)
            b = np.asarray(v1, dtype=np.float64)
            return (a + u * (b - a)).tolist()
        except Exception:
            return copy.deepcopy(v0)
    if isinstance(v0, bool) or isinstance(v1, bool):
        return v0 if u < 0.5 else v1
    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
        if isinstance(v0, int) and isinstance(v1, int):
            return v0 if u < 0.5 else v1
        return v0 + u * (v1 - v0)
    return copy.deepcopy(v0)


def make_interp_frame(a, b, u, t_new):
    """Interpolate full frame dicts a->b at fraction u, with timestamp t_new."""
    f = interp(a, b, u)
    f["t"] = t_new
    f["f"] = None  # renumbered later
    return f


def classify_file(path, stuck_dt, big_gap_dt, small_gap_dt, mut_mps, glitch_ret_mps):
    """Pass 1: fast stream parse. Returns (info, decisions, n).

    `info` is a list of (t, p) tuples; `decisions` is one of
    'keep' | 'drop_stuck' | 'drop_glitch' | 'split' | 'gap' per frame.
    'gap' means keep k but interpolate between k-1 and k.
    """
    info = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, p = parse_t_p(line)
            info.append((t, p))
    n = len(info)
    decisions = ["keep"] * n
    for k in range(n):
        if k == 0:
            continue
        t_prev, p_prev = info[k - 1]
        t_cur, p_cur = info[k]
        if t_prev is None or t_cur is None:
            continue
        dt = t_cur - t_prev
        if dt < stuck_dt:
            decisions[k] = "drop_stuck"
            continue
        if dt > big_gap_dt:
            decisions[k] = "split"
            continue
        if dt > small_gap_dt:
            decisions[k] = "gap"
            continue
        if p_prev is None or p_cur is None or dt <= 1e-4:
            continue
        disp = math.sqrt((p_cur[0] - p_prev[0]) ** 2 +
                        (p_cur[1] - p_prev[1]) ** 2 +
                        (p_cur[2] - p_prev[2]) ** 2)
        spd = disp / dt / 100.0
        if spd > mut_mps:
            # glitch vs teleport: look at k-1 -> k+1 return speed
            if k + 1 < n:
                t_next, p_next = info[k + 1]
                if t_next is not None and p_next is not None:
                    dtn = t_next - t_prev
                    if dtn > 1e-4:
                        disp_ret = math.sqrt((p_next[0] - p_prev[0]) ** 2 +
                                             (p_next[1] - p_prev[1]) ** 2 +
                                             (p_next[2] - p_prev[2]) ** 2)
                        if disp_ret / dtn / 100.0 < glitch_ret_mps:
                            decisions[k] = "drop_glitch"
                            continue
            decisions[k] = "split"
    return info, decisions, n


def build_segments(path, decisions, target_dt):
    """Pass 2: stream full JSON lines, build kept segments with interpolation.

    Kept frames are json.loads'd so dt/f can be renumbered; interpolated frames
    are constructed from the two boundary frames. Only the 2 boundary frames
    per gap are held in memory. Returns a list of segments (lists of frame dicts).
    """
    segments = []
    cur = []
    cur_prev_t = None
    cur_idx = 0
    boundary = None
    j = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = decisions[j]
            j += 1
            if d in ("drop_stuck", "drop_glitch"):
                continue
            fr = json.loads(line)
            if d == "split":
                if cur:
                    segments.append(cur)
                cur = [fr]
                cur_prev_t = fr["t"]
                cur_idx = 1
                boundary = fr
                continue
            if d == "gap" and cur:
                a, b = boundary, fr
                t_a, t_b = a["t"], b["t"]
                gap = t_b - t_a
                intervals = max(1, int(round(gap / target_dt)))
                for i in range(1, intervals):
                    u = i / intervals
                    t_new = t_a + u * gap
                    nf = make_interp_frame(a, b, u, t_new)
                    nf["dt"] = t_new - cur_prev_t
                    nf["f"] = cur_idx
                    cur.append(nf)
                    cur_prev_t = t_new
                    cur_idx += 1
                fr["dt"] = t_b - cur_prev_t
                fr["f"] = cur_idx
                cur.append(fr)
                cur_prev_t = t_b
                cur_idx += 1
                boundary = fr
            else:
                fr["dt"] = (fr["t"] - cur_prev_t) if cur_prev_t is not None else 0.0
                fr["f"] = cur_idx
                cur.append(fr)
                cur_prev_t = fr["t"]
                cur_idx += 1
                boundary = fr
    if cur:
        segments.append(cur)
    return segments


def filter_segment(seg, min_frames, min_dur):
    n = len(seg)
    dur = sum(fr["dt"] for fr in seg)
    return n >= min_frames and dur >= min_dur


def process_file(path, cat, fname, out_root, params, stats, manifest):
    info, decisions, n = classify_file(
        path, params.stuck_dt, params.big_gap_dt, params.small_gap_dt,
        params.mut_mps, params.glitch_ret_mps)
    n_stuck = decisions.count("drop_stuck")
    n_glitch = decisions.count("drop_glitch")
    n_split = decisions.count("split")
    n_gap = decisions.count("gap")
    stats["files"] += 1
    stats["frames_in"] += n
    stats["stuck"] += n_stuck
    stats["glitch"] += n_glitch
    stats["splits"] += n_split
    stats["gaps"] += n_gap
    # skip pass 2 entirely if nothing to fix
    if not (n_stuck or n_glitch or n_split or n_gap):
        stats["files_unchanged"] += 1
        stats["frames_out"] += n
        stats["segs_total"] += 1
        return
    segs = build_segments(path, decisions, params.target_dt)
    kept = [s for s in segs if filter_segment(s, params.min_frames, params.min_dur)]
    dropped_segs = len(segs) - len(kept)
    stats["files_changed"] += 1
    stats["segs_total"] += len(segs)
    stats["segs_dropped"] += dropped_segs
    out_frames = sum(len(s) for s in kept)
    stats["frames_out"] += out_frames
    base = fname.replace("_frames.jsonl", "")
    out_cat = os.path.join(out_root, cat)
    os.makedirs(out_cat, exist_ok=True)
    new_paths = []
    for j, s in enumerate(kept):
        on = f"{base}_c{j}_frames.jsonl"
        op = os.path.join(out_cat, on)
        with open(op, "w") as wf:
            for fr in s:
                wf.write(json.dumps(fr))
                wf.write("\n")
        meta_src = path.replace("_frames.jsonl", "_meta.json")
        if os.path.exists(meta_src):
            with open(meta_src) as mf:
                meta = json.load(mf)
            if "num_frames" in meta:
                meta["num_frames"] = len(s)
            meta["cleaned"] = True
            with open(os.path.join(out_cat, f"{base}_c{j}_meta.json"), "w") as mf:
                json.dump(meta, mf)
        new_paths.append(op)
    manifest[os.path.join(cat, fname)] = {
        "segments": [os.path.join(cat, os.path.basename(p)) for p in new_paths],
        "n_in": n, "n_out_total": out_frames, "n_stuck": n_stuck,
        "n_glitch": n_glitch, "n_split": n_split, "n_gap": n_gap,
        "segs_total": len(segs), "segs_kept": len(kept),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/MorphData_v1",
                    help="root of raw MorphData_v1 (default: data/MorphData_v1)")
    ap.add_argument("--out-root", default="data/MorphData_v1_cleaned_staging",
                    help="where cleaned segments are staged (default: "
                         "data/MorphData_v1_cleaned_staging)")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                    help="category subdirs to process (default: all 5)")
    ap.add_argument("--stuck-dt", type=float, default=STUCK_DT)
    ap.add_argument("--big-gap-dt", type=float, default=BIG_GAP_DT)
    ap.add_argument("--small-gap-dt", type=float, default=SMALL_GAP_DT)
    ap.add_argument("--mut-mps", type=float, default=MUT_MPS,
                    help="disp/dt speed (m/s) above which a normal-rate frame is "
                         "a mutation (default 10)")
    ap.add_argument("--glitch-ret-mps", type=float, default=GLITCH_RET_MPS)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument("--min-dur", type=float, default=MIN_DUR)
    ap.add_argument("--target-dt", type=float, default=TARGET_DT)
    ap.add_argument("--manifest", default="data/MorphData_v1_clean_manifest.json")
    ap.add_argument("--stats", default="data/MorphData_v1_clean_stats.json")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    stats = {k: 0 for k in ["files", "files_changed", "files_unchanged",
                            "frames_in", "frames_out", "stuck", "glitch",
                            "splits", "gaps", "segs_total", "segs_dropped"]}
    manifest = {}
    for cat in args.categories:
        files = sorted(glob.glob(os.path.join(args.data_root, cat, "*_frames.jsonl")))
        for fp in files:
            process_file(fp, cat, os.path.basename(fp), args.out_root,
                         args, stats, manifest)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=1)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=1)
    print("=== clean stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"files changed: {stats['files_changed']}  -> manifest {args.manifest}")


if __name__ == "__main__":
    main()
