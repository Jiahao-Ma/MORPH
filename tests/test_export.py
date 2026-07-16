"""End-to-end test: retarget + export G1 qpos for all sample recordings.

For every recording in every category (ground / stairs / traversal) this runs
the real ue_world_skeleton_retarget.py with the category-correct --src-human /
--data-scale / --config, in --no-visualize batch mode, and asserts the
expected .npy / .npz export files are produced under data/RetargetOutputs/.

Run:
    python tests/test_export.py
"""
import pathlib

from _common import (
    CATEGORIES, OUTPUTS, PY, RETARGET_DIR, run, sample_paths, check_files,
    ensure_gmr_on_env, all_sample_paths,
)


def export_one(p: dict) -> None:
    cfg = p["cfg"]
    out_dir = OUTPUTS / p["cat"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_qpos = out_dir / f"{p['stem']}_frames.npy"

    cmd = [
        PY, str(RETARGET_DIR / "ue_world_skeleton_retarget.py"),
        "--jsonl", str(p["jsonl"]),
        "--meta", str(p["meta"]),
        "--config", str(p["config"]),
        "--src-human", cfg["src_human"],
        "--data-scale", str(cfg["data_scale"]),
        "--output-qpos", str(out_qpos),
        "--no-visualize",
    ]
    # ground has no terrain; stairs/traversal load the sample terrain for viz
    # consistency (the export itself is terrain-independent, but loading it
    # exercises the terrain path too).
    if p["terrain"] is not None:
        cmd += ["--terrain", str(p["terrain"])]
    else:
        cmd += ["--no-terrain"]

    run(cmd, env=ensure_gmr_on_env(None))

    expected = [
        out_qpos,
        out_dir / f"{p['stem']}_frames_cmd.npy",
        out_dir / f"{p['stem']}_frames_scaled.npz",
    ]
    check_files(f"export {p['cat']}/{p['stem']}", expected)


def main():
    print("=" * 70)
    print("TEST: retarget + export G1 qpos (ground / stairs / traversal)")
    print("=" * 70)
    for p in all_sample_paths():
        export_one(p)
    print("\nALL EXPORT TESTS PASSED.")


if __name__ == "__main__":
    main()
