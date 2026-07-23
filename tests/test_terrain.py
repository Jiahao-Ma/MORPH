"""End-to-end test: export terrain (MuJoCo MJCF + IsaacLab OBJ/NPZ) for the
stairs and traversal samples, including the appended flat ground plane.

Output layout matches Scripts/data/RetargetOutputs/terrain:
    data/RetargetOutputs/terrain/<name>_mujoco/<hash>.xml  + <hash>.obj
    data/RetargetOutputs/terrain/<name>_isaaclab/<hash>.obj + <hash>.npz

convert_terrain.py exports every terrain_*.json in --terrain-dir, so a single
invocation per category covers all of that category's recordings (they share
one terrain hash).

Run:
    python tests/test_terrain.py
"""
import pathlib

from _common import (
    CATEGORIES, OUTPUTS, PY, TERRAIN_DIR, SAMPLE_TERRAIN_DIR,
    run, sample_paths, check_files,
)


def export_terrain(cat: str) -> None:
    cfg = CATEGORIES[cat]
    if cfg["terrain_hash"] is None:
        print(f"[terrain {cat}] no terrain for this category, skipping.")
        return
    # any recording's config works — terrain files all live in the shared
    # data/sample/terrain/ folder now.
    p = sample_paths(cat, cfg["recordings"][0])
    out_dir = OUTPUTS / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cat  # -> <cat>_mujoco / <cat>_isaaclab

    cmd = [
        PY, str(TERRAIN_DIR / "convert_terrain.py"),
        "--terrain-dir", str(SAMPLE_TERRAIN_DIR),
        "--config", str(p["config"]),
        "--output-dir", str(out_dir),
        "--name", name,
    ]
    run(cmd)

    h = cfg["terrain_hash"]
    expected = [
        out_dir / f"{name}_mujoco" / f"{h}.xml",
        out_dir / f"{name}_mujoco" / f"{h}.obj",
        out_dir / f"{name}_isaaclab" / f"{h}.obj",
        out_dir / f"{name}_isaaclab" / f"{h}.npz",
    ]
    check_files(f"terrain {cat}", expected)


def main():
    print("=" * 70)
    print("TEST: terrain export (MuJoCo + IsaacLab, with ground plane)")
    print("=" * 70)
    for cat in CATEGORIES:
        export_terrain(cat)
    print("\nALL TERRAIN TESTS PASSED.")


if __name__ == "__main__":
    main()
