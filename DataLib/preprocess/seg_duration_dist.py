#!/usr/bin/env python3
"""Duration distribution of the (cleaned) segments in MorphData_v1.

Per segment: dur = t[last] - t[first] (s). Reports per-category summary,
global percentiles, a duration histogram (binned), and writes a PNG plot
(overall histogram on log-x + per-category box plot).
"""
import os
import re
import glob
import json
import argparse
import mmap

import numpy as np

DEFAULT_CATEGORIES = [
    "ground", "stairs",
    "traversal_mantle", "traversal_mantle_vault", "traversal_vault",
]
SPEED_CAP = 15.0


def parse_t(line: bytes):
    m = re.search(rb'"t":\s*([-\d.]+)', line)
    return float(m.group(1)) if m else None


def seg_dur(path):
    """Read only the first/last JSONL records; duration needs no middle rows."""
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return 0.0
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            first_start = 0
            while first_start < len(mm) and mm[first_start] in b" \t\r\n":
                first_start += 1
            first_end = mm.find(b"\n", first_start)
            first_line = (
                mm[first_start:]
                if first_end < 0
                else mm[first_start:first_end]
            )

            end = len(mm)
            while end > 0 and mm[end - 1] in b" \t\r\n":
                end -= 1
            last_start = mm.rfind(b"\n", 0, end) + 1
            last_line = mm[last_start:end]

    first = parse_t(first_line)
    last = parse_t(last_line)
    if first is None or last is None:
        return 0.0
    return last - first


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/MorphData_v1")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--out-json", default="data/MorphData_v1_seg_dur_dist.json")
    ap.add_argument("--out-png", default="data/MorphData_v1_seg_dur_dist.png")
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="write/print statistics without importing matplotlib or writing PNG",
    )
    args = ap.parse_args()

    all_dur = []
    per_cat = {}
    for cat in args.categories:
        files = sorted(glob.glob(os.path.join(args.data_root, cat, "*_frames.jsonl")))
        durs = []
        for fp in files:
            durs.append(seg_dur(fp))
        per_cat[cat] = durs
        all_dur.extend(durs)
        dn = np.asarray(durs)
        print(f"{cat}: segs={len(durs)} total={sum(durs)/3600:.2f}h "
              f"median={np.median(dn):.2f}s mean={dn.mean():.2f}s "
              f"min={dn.min():.2f}s max={dn.max():.2f}s")
    a = np.asarray(all_dur)
    print(f"\nALL: segs={len(a)} total={a.sum()/3600:.2f}h")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:>2} = {np.percentile(a, q):.2f} s")
    print(f"  min = {a.min():.2f} s   max = {a.max():.2f} s   mean = {a.mean():.2f} s")
    bins = [0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 1e9]
    labels = ["<1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-30", "30-60",
              "60-120", "120-300", "300-600", "600-1800", ">1800"]
    counts, _ = np.histogram(a, bins=bins)
    print("\nhistogram (duration s):")
    for lbl, c in zip(labels, counts):
        print(f"  {lbl:>10} s: {c:5d}  ({100*c/len(a):5.1f}%)")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"per_cat": {c: d for c, d in per_cat.items()},
                   "percentiles": {q: float(np.percentile(a, q))
                                   for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
                   "min": float(a.min()), "max": float(a.max()),
                   "mean": float(a.mean()), "total_h": float(a.sum()/3600),
                   "n": int(len(a))}, f, indent=1)

    if args.no_plot:
        print(f"\n-> {args.out_json}")
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        b = [0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600]
        ax.hist(a, bins=b, color="#3366cc", edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("segment duration (s, log scale)")
        ax.set_ylabel("# segments")
        ax.set_title(f"Segment duration distribution (n={len(a)}, "
                     f"total={a.sum()/3600:.1f}h)")
        ax.grid(alpha=0.3, which="both")
        ax = axes[1]
        data = [per_cat[c] for c in args.categories]
        ax.boxplot(data, tick_labels=args.categories, showfliers=True,
                   sym=".", whis=[5, 95])
        ax.set_yscale("log")
        ax.set_ylabel("duration (s, log scale)")
        ax.set_title("Per-category duration spread (box 5-95%)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=110)
        plt.close(fig)
        print(f"\n-> {args.out_json}\n-> {args.out_png}")


if __name__ == "__main__":
    main()
