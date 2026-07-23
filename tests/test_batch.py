"""End-to-end test: batch retarget via DataLib/batch/batch_retarget.py.

batch_retarget.py runs one --input-dir with a single --config / --data-scale,
so a batch is per-category (ground/stairs/traversal each need their own
src-human config + scale). This test runs the batcher on each sample
category (which now contains 2 recordings each) with the category-correct
flags and asserts every .npy is produced.

Run:
    python tests/test_batch.py
"""
import pathlib

from _common import (
    CATEGORIES, OUTPUTS, PY, BATCH_DIR, run, sample_paths, check_files,
    ensure_gmr_on_env, recordings,
)


def batch_category(cat: str) -> None:
    cfg = CATEGORIES[cat]
    p = sample_paths(cat, cfg["recordings"][0])
    out_dir = OUTPUTS / f"batch_{cat}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, str(BATCH_DIR / "batch_retarget.py"),
        "--input-dir", str(p["jsonl"].parent),
        "--output-dir", str(out_dir),
        "--config", str(p["config"]),
        "--data-scale", str(cfg["data_scale"]),
        "--terrain-dir", str(p["terrain_dir"]),
        "--workers", "1",
        # thread src-human + height mode through --extra-args
        "--extra-args", f"--src-human {cfg['src_human']} --height-from-data",
    ]
    run(cmd, env=ensure_gmr_on_env(None))

    expected = [out_dir / f"{stem}.npy" for stem in recordings(cat)]
    check_files(f"batch {cat}", expected)


def main():
    print("=" * 70)
    print("TEST: batch retarget (per category, --data-scale threaded)")
    print("=" * 70)
    for cat in CATEGORIES:
        batch_category(cat)
    print("\nALL BATCH TESTS PASSED.")


if __name__ == "__main__":
    main()
