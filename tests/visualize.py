"""Visualize a sample recording's retargeted G1 motion (+ terrain).

Two viewers are available:

  --mode retarget (default)
      Runs ue_world_skeleton_retarget.py WITHOUT --no-visualize, opening the
      MuJoCo native viewer on the retargeted G1 qpos with the (data-scale
      matched) terrain loaded for context. This is the "remapped data +
      terrain" view the release is built around.

  --mode verify
      Runs the richer diagnostic viewer:
        stairs      -> verify_pipeline.py          (already G1-scaled data)
        ground/trav -> verify_pipeline_scaled.py   (category --data-scale)
      These show the raw UE skeleton, the Kabsch-aligned skeleton and the
      retargeted G1 side by side through the pipeline stages. Backend defaults
      to --viewer mujoco (no extra deps); use --viewer viser after
      `pip install viser` for the web 3D viewer.

Usage:
    python tests/visualize.py --cat ground
    python tests/visualize.py --cat stairs
    python tests/visualize.py --cat traversal
    python tests/visualize.py --cat stairs --mode verify
    python tests/visualize.py --cat ground --rec 1      # 2nd recording

NOTE: opens an interactive MuJoCo window — not part of the automated test
suite. Requires a display.
"""
import argparse

from _common import (
    CATEGORIES, PY, RETARGET_DIR, run, sample_paths, ensure_gmr_on_env,
    recordings,
)


def visualize(cat: str, rec_idx: int, mode: str, viewer: str) -> None:
    stems = recordings(cat)
    stem = stems[min(rec_idx, len(stems) - 1)]
    p = sample_paths(cat, stem)
    cfg = p["cfg"]
    env = ensure_gmr_on_env(None)
    print(f"[visualize] cat={cat} rec={rec_idx} stem={stem} mode={mode} viewer={viewer}")

    if mode == "retarget":
        cmd = [
            PY, str(RETARGET_DIR / "ue_world_skeleton_retarget.py"),
            "--jsonl", str(p["jsonl"]),
            "--meta", str(p["meta"]),
            "--config", str(p["config"]),
            "--src-human", cfg["src_human"],
            "--data-scale", str(cfg["data_scale"]),
            "--height-from-data",
            "--no-save",
        ]
        if p["terrain"] is not None:
            cmd += ["--terrain-dir", str(p["terrain_dir"])]
        else:
            cmd += ["--no-terrain"]
        run(cmd, env=env)
    elif mode == "verify":
        if cat == "stairs":
            script = RETARGET_DIR / "verify_pipeline.py"
            cmd = [
                PY, str(script),
                "--jsonl", str(p["jsonl"]),
                "--meta", str(p["meta"]),
                "--src-human", cfg["src_human"],
                "--viewer", viewer,
            ]
            if p["config"]:
                cmd += ["--config", str(p["config"])]
            if p["terrain"] is not None:
                cmd += ["--terrain-dir", str(p["terrain_dir"])]
        else:
            script = RETARGET_DIR / "verify_pipeline_scaled.py"
            cmd = [
                PY, str(script),
                "--jsonl", str(p["jsonl"]),
                "--meta", str(p["meta"]),
                "--src-human", cfg["src_human"],
                "--data-scale", str(cfg["data_scale"]),
                "--viewer", viewer,
            ]
            if p["config"]:
                cmd += ["--config", str(p["config"])]
            if p["terrain"] is not None:
                cmd += ["--terrain-dir", str(p["terrain_dir"])]
        run(cmd, env=env)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat", required=True, choices=list(CATEGORIES),
                    help="Sample category to visualize.")
    ap.add_argument("--rec", type=int, default=0,
                    help="Index of the recording within the category (0-based, default 0).")
    ap.add_argument("--mode", default="retarget", choices=["retarget", "verify"],
                    help="Which viewer to launch (default: retarget).")
    ap.add_argument("--viewer", default="mujoco", choices=["mujoco", "viser"],
                    help="Backend for --mode verify (default: mujoco, no extra "
                         "deps. Use 'viser' for the web 3D viewer after "
                         "`pip install viser`.)")
    args = ap.parse_args()
    visualize(args.cat, args.rec, args.mode, args.viewer)


if __name__ == "__main__":
    main()
