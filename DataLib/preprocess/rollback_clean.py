#!/usr/bin/env python3
"""Rollback a previous apply_clean.py swap.

For every entry in the rollback manifest:
  1. remove the placed cleaned segments from the dataset,
  2. move the original <base>_frames.jsonl + <base>_meta.json back from the
     backup dir into the dataset.
"""
import os
import json
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/MorphData_v1")
    ap.add_argument("--backup-dir", default="data/MorphData_v1_originals_backup")
    ap.add_argument("--rollback-manifest",
                    default="data/MorphData_v1_rollback_manifest.json")
    args = ap.parse_args()

    with open(args.rollback_manifest) as f:
        repl = json.load(f)
    n_done = 0
    for rel, info in repl.items():
        cat, fname = rel.split("/", 1)
        base = fname.replace("_frames.jsonl", "")
        # remove placed cleaned segments
        for sname in info["placed"]:
            p = os.path.join(args.data_root, cat, sname)
            if os.path.exists(p):
                os.remove(p)
            m = p.replace("_frames.jsonl", "_meta.json")
            if os.path.exists(m):
                os.remove(m)
        # restore originals from backup
        for suffix in ["_frames.jsonl", "_meta.json"]:
            src = os.path.join(info["backup_dir"], base + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(args.data_root, cat, base + suffix))
        n_done += 1
    print(f"done: rolled back {n_done} files to their originals.")


if __name__ == "__main__":
    main()
