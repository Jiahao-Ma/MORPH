"""Batch retarget all GASP recordings in a directory to G1 qpos .npy files.

Scans the input directory for *_meta.json files, pairs each with its
corresponding *_frames.jsonl, and runs ue_world_skeleton_retarget.py on
each pair. Output .npy files are saved to an output directory preserving
the original naming.

Usage:
  # Process all recordings in Scripts/data/Stairs, output to Scripts/data/UEGMR/Stairs
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs

  # Dry-run: print commands without executing
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs --dry-run

  # Custom retarget options (passed through to ue_world_skeleton_retarget.py)
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs --extra-args "--no-contact-fix --no-penetration-fix --ball-align-extra-drop 0.03"

  # Parallel processing (4 workers)
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs --workers 4

  # Skip files that already have output .npy
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs --skip-existing

  # Custom timeout for large files (default: 3600s = 1 hour)
  python batch_retarget.py --input-dir ../data/Stairs --output-dir ../data/UEGMR/Stairs --timeout 7200
"""

import argparse
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
# Release layout: HERE = <repo>/DataLib/batch ; retarget scripts live next to it.
RETARGET_SCRIPT = HERE.parent / "retarget" / "ue_world_skeleton_retarget.py"
DEFAULT_CONFIG = HERE.parent / "retarget" / "gasp_bvh_alignment_g1_height.json"

# Force UTF-8 stdout/stderr so non-ASCII prints don't crash on Windows GBK consoles.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_recording_pairs(input_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path, str]]:
    """Find all (meta.json, frames.jsonl) pairs in input_dir.

    Returns list of (meta_path, frames_path, stem) tuples.
    The stem is the common prefix (everything before _meta.json / _frames.jsonl).
    """
    pairs = []
    for meta_file in sorted(input_dir.glob("*_meta.json")):
        stem = meta_file.name.removesuffix("_meta.json")
        frames_file = meta_file.parent / f"{stem}_frames.jsonl"
        if frames_file.exists():
            pairs.append((meta_file, frames_file, stem))
        else:
            print(f"  [WARN] No matching frames file for: {meta_file.name}")
    return pairs


def run_retarget(
    meta: pathlib.Path,
    frames: pathlib.Path,
    output_npy: pathlib.Path,
    config: pathlib.Path,
    extra_args: list[str],
    timeout: int = 3600,
    data_scale: float = 1.0,
    terrain_dir: str | None = None,
) -> tuple[str, bool, str, float]:
    """Run the retarget script on a single recording. Returns (stem, success, message, elapsed_sec)."""
    stem = output_npy.stem

    cmd = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--jsonl", str(frames),
        "--meta", str(meta),
        "--config", str(config),
        "--output-qpos", str(output_npy),
        "--no-visualize",
    ] + extra_args
    if abs(float(data_scale) - 1.0) > 1e-9:
        cmd += ["--data-scale", str(data_scale)]
    if terrain_dir:
        cmd += ["--terrain-dir", str(terrain_dir)]

    t_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.time() - t_start
        if result.returncode == 0:
            return (stem, True, "OK", elapsed)
        else:
            err_lines = result.stderr.strip().split("\n") if result.stderr else []
            err_msg = err_lines[-1] if err_lines else "Unknown error"
            # Also show stdout tail for context
            out_lines = result.stdout.strip().split("\n") if result.stdout else []
            last_progress = out_lines[-1] if out_lines else ""
            return (stem, False, f"{err_msg} | last_output: {last_progress}", elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        return (stem, False, f"TIMEOUT ({timeout}s)", elapsed)
    except Exception as e:
        elapsed = time.time() - t_start
        return (stem, False, f"{type(e).__name__}: {e}", elapsed)


def main():
    ap = argparse.ArgumentParser(
        description="Batch retarget GASP recordings to G1 qpos .npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input-dir", required=True,
                    help="Directory containing *_meta.json + *_frames.jsonl pairs")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write output .npy files")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="Alignment config JSON (default: gasp_bvh_alignment.json)")
    ap.add_argument("--extra-args", default="",
                    help="Extra arguments passed to ue_world_skeleton_retarget.py (quoted string)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (default: 1 = sequential)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-file timeout in seconds (default: 3600 = 1 hour)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip recordings whose output .npy already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without executing")
    ap.add_argument("--data-scale", type=float, default=1.0,
                    help="Uniform scale applied to the source UE motion (and viz "
                         "terrain) about the UE world origin before retarget. Use "
                         "0.77 for original-UE-height recordings (ground/traversal) "
                         "and 1.0 for already-G1-scaled recordings (stairs). "
                         "Forwarded to ue_world_skeleton_retarget.py --data-scale. "
                         "Default 1.0.")
    ap.add_argument("--terrain-dir", default=None,
                    help="Directory holding terrain_<hash>.json files. Forwarded "
                         "to ue_world_skeleton_retarget.py --terrain-dir so the "
                         "retarget/viz terrain is resolved from a single shared "
                         "folder instead of next to each recording. Default: none "
                         "(the retarget script falls back to a `terrain` folder "
                         "next to each --meta).")

    args = ap.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    config = pathlib.Path(args.config).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not config.is_file():
        print(f"ERROR: Config file not found: {config}")
        sys.exit(1)

    extra_args = args.extra_args.split() if args.extra_args.strip() else []

    # Discover recording pairs
    print(f"[Batch Retarget]")
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Config:  {config}")
    print(f"  Extra:   {extra_args if extra_args else '(none)'}")
    print(f"  Workers: {args.workers}")
    print(f"  Data scale: {args.data_scale}")
    if args.terrain_dir:
        print(f"  Terrain dir: {args.terrain_dir}")
    print()

    pairs = find_recording_pairs(input_dir)
    if not pairs:
        print("No recording pairs found. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(pairs)} recording(s):")
    for meta, frames, stem in pairs:
        size_mb = frames.stat().st_size / (1024 * 1024)
        print(f"  - {stem}  ({size_mb:.0f} MB)")
    print()

    # Prepare output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for meta, frames, stem in pairs:
        output_npy = output_dir / f"{stem}.npy"

        if args.skip_existing and output_npy.exists():
            print(f"  [SKIP] {stem} (output exists)")
            continue

        tasks.append((meta, frames, output_npy, config, extra_args, args.timeout, args.data_scale, args.terrain_dir))

    if not tasks:
        print("\nAll recordings already processed. Nothing to do.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(tasks)} recording(s):")
        for meta, frames, output_npy, _, _, _, _, _ in tasks:
            print(f"  {frames.name} -> {output_npy.name}")
        return

    # Execute
    print(f"\nProcessing {len(tasks)} recording(s)  (timeout={args.timeout}s per file)...\n")
    t0 = time.time()
    success_count = 0
    fail_count = 0

    if args.workers <= 1:
        # Sequential
        for i, (meta, frames, output_npy, cfg, extra, timeout, dscale, tdir) in enumerate(tasks, 1):
            stem = output_npy.stem
            size_mb = frames.stat().st_size / (1024 * 1024)
            print(f"[{i}/{len(tasks)}] {stem} ({size_mb:.0f} MB) ...", flush=True)
            _, ok, msg, elapsed_file = run_retarget(meta, frames, output_npy, cfg, extra, timeout, dscale, tdir)
            if ok:
                print(f"         [OK] done in {elapsed_file:.1f}s")
                success_count += 1
            else:
                print(f"         [FAIL] ({msg}) after {elapsed_file:.1f}s")
                fail_count += 1
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for meta, frames, output_npy, cfg, extra, timeout, dscale, tdir in tasks:
                fut = executor.submit(run_retarget, meta, frames, output_npy, cfg, extra, timeout, dscale, tdir)
                futures[fut] = output_npy.stem

            for fut in as_completed(futures):
                stem = futures[fut]
                _, ok, msg, elapsed_file = fut.result()
                if ok:
                    print(f"  [OK]   {stem}  ({elapsed_file:.1f}s)")
                    success_count += 1
                else:
                    print(f"  [FAIL] {stem}: {msg}  ({elapsed_file:.1f}s)")
                    fail_count += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min) — {success_count} succeeded, {fail_count} failed")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
