#!/usr/bin/env python3
"""Apply staged cleaned segments to the dataset (with backup + rollback).

Reads the manifest written by clean_morph.py and, for every changed original:
  1. moves the original <base>_frames.jsonl + <base>_meta.json into a backup dir,
  2. copies the staged cleaned segments (<base>_c<j>_frames.jsonl + _meta.json)
     from the staging dir into the dataset,
  3. records a rollback manifest so the swap can be undone.

Originals are MOVED (not copied) to the backup dir so the dataset ends up
containing only the cleaned segments. Use rollback_clean.py to undo.
"""
import os
import json
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/MorphData_v1")
    ap.add_argument("--stage-root", default="data/MorphData_v1_cleaned_staging")
    ap.add_argument("--backup-dir", default="data/MorphData_v1_originals_backup")
    ap.add_argument("--manifest", default="data/MorphData_v1_clean_manifest.json")
    ap.add_argument("--rollback-manifest",
                    default="data/MorphData_v1_rollback_manifest.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = json.load(f)
    os.makedirs(args.backup_dir, exist_ok=True)
    repl = {}
    n_done = 0
    for rel, info in man.items():
        cat, fname = rel.split("/", 1)
        base = fname.replace("_frames.jsonl", "")
        bdir = os.path.join(args.backup_dir, cat)
        os.makedirs(bdir, exist_ok=True)
        # backup + remove original frames & meta
        for suffix in ["_frames.jsonl", "_meta.json"]:
            src = os.path.join(args.data_root, cat, base + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(bdir, base + suffix))
        # copy staged segments into dataset
        placed = []
        for segrel in info["segments"]:
            sname = os.path.basename(segrel)
            ssrc = os.path.join(args.stage_root, cat, sname)
            sdst = os.path.join(args.data_root, cat, sname)
            shutil.copy(ssrc, sdst)
            placed.append(sname)
        repl[rel] = {"backup_dir": bdir,
                    "removed": [base + "_frames.jsonl", base + "_meta.json"],
                    "placed": placed}
        n_done += 1
        if n_done % 100 == 0:
            print(f"  replaced {n_done}/{len(man)} files...", flush=True)
    os.makedirs(os.path.dirname(args.rollback_manifest) or ".", exist_ok=True)
    with open(args.rollback_manifest, "w") as f:
        json.dump(repl, f, indent=1)
    print(f"done: replaced {n_done} originals with staged segments. "
          f"rollback manifest -> {args.rollback_manifest}")


if __name__ == "__main__":
    main()
