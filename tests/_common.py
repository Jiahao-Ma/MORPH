"""Shared helpers for the Morph end-to-end tests.

Every test invokes the real release scripts (DataLib/retarget,
DataLib/terrain, DataLib/batch) as subprocesses, so a green run means the
repo is reproducible end-to-end from the vendored GMR + sample data.
"""
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent                      # Morph/
DATALIB = REPO / "DataLib"
RETARGET_DIR = DATALIB / "retarget"
TERRAIN_DIR = DATALIB / "terrain"
BATCH_DIR = DATALIB / "batch"
GMR_DIR = DATALIB / "gmr"

DATA = REPO / "data"
SAMPLE = DATA / "sample"
# All terrain_<hash>.json files live in one shared folder (one per unique
# terrain hash; recordings reference theirs via meta.terrain_ref). Each
# terrain-consuming script accepts --terrain-dir pointing here.
SAMPLE_TERRAIN_DIR = SAMPLE / "terrain"
OUTPUTS = DATA / "RetargetOutputs"

PY = sys.executable

# Per-category retarget configuration.
#
# The retarget pipeline reads world-space joint positions (wp), which already
# include the in-engine character mesh scale (root.mesh.s in the JSONL):
#   ground/stairs : recorded with mesh.s=0.77 (already G1 height) -> data_scale=1.0
#   traversal     : recorded with mesh.s=1.0  (original height)   -> data_scale=0.77
# After data_scale, everything is at G1 height, so every category uses the
# G1-scale pairing: src=bvh_ue5_g1scale + gasp_bvh_alignment_g1_height.json.
# (The old ground/traversal pairing of data_scale=0.77 + bvh_ue5_native +
# gasp_bvh_alignment.json double-shrank ground and mis-scaled traversal,
# forcing the IK against joint limits -> violent pelvis jitter.)
#
# Each category lists >=1 recording; all stairs recordings share terrain
# E4B166AB and all traversal recordings share terrain 43D5EBED. The terrain
# JSONs themselves live together in data/sample/terrain/ (one per hash), not
# inside each category folder.
CATEGORIES = {
    "ground": {
        "src_human": "bvh_ue5_g1scale",
        "data_scale": 1.0,
        "config": RETARGET_DIR / "gasp_bvh_alignment_g1_height.json",
        "terrain_hash": None,
        "recordings": [
            "WalkTurnCrouch_C_10_Ovu0mkr5",
            "WalkTurn_C_11_rTsKE0_Q",
        ],
    },
    "stairs": {
        "src_human": "bvh_ue5_g1scale",
        "data_scale": 1.0,
        "config": RETARGET_DIR / "gasp_bvh_alignment_g1_height.json",
        "terrain_hash": "E4B166AB",
        "recordings": [
            "WalkTurnCrouch_C_11_RvWsfUQi",
            "WalkTurnCrouch_C_3_9qeNakh1",
        ],
    },
    "traversal": {
        "src_human": "bvh_ue5_g1scale",
        "data_scale": 0.77,
        "config": RETARGET_DIR / "gasp_bvh_alignment_g1_height.json",
        "terrain_hash": "43D5EBED",
        "recordings": [
            "Traversal_C_0_9TSt7kc8",
        ],
    },
}


def recordings(cat: str) -> list[str]:
    return CATEGORIES[cat]["recordings"]


def sample_paths(cat: str, stem: str) -> dict:
    cfg = CATEGORIES[cat]
    d = SAMPLE / cat
    has_terrain = cfg["terrain_hash"] is not None
    return {
        "jsonl": d / f"{stem}_frames.jsonl",
        "meta": d / f"{stem}_meta.json",
        "config": cfg["config"],
        # Terrain JSONs now live in the shared data/sample/terrain/ folder.
        "terrain": (SAMPLE_TERRAIN_DIR / f"terrain_{cfg['terrain_hash']}.json") if has_terrain else None,
        "terrain_dir": SAMPLE_TERRAIN_DIR,
        "cfg": cfg,
        "stem": stem,
        "cat": cat,
    }


def all_sample_paths() -> list[dict]:
    out = []
    for cat in CATEGORIES:
        for stem in recordings(cat):
            out.append(sample_paths(cat, stem))
    return out


def run(cmd: list, env: dict | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a command, stream its stdout, raise on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, timeout=timeout, text=True,
                          encoding="utf-8", errors="replace")
    print(f"  (exit={proc.returncode}, {time.time()-t0:.1f}s)")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {cmd}")
    return proc


def ensure_gmr_on_env(env: dict | None) -> dict:
    """Put DataLib/gmr on PYTHONPATH so `general_motion_retargeting` imports."""
    env = dict(env or os.environ)
    gmr = str(GMR_DIR)
    if gmr not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = gmr + os.pathsep + env.get("PYTHONPATH", "")
    return env


def check_files(label: str, files: list[pathlib.Path]) -> None:
    print(f"[{label}] checking outputs:")
    ok = True
    for f in files:
        exists = f.exists()
        size = f.stat().st_size if exists else 0
        print(f"  {'OK ' if exists else 'MISS'} {f.relative_to(REPO)}  ({size} bytes)")
        ok = ok and exists and size > 0
    if not ok:
        raise RuntimeError(f"[{label}] some output files are missing/empty")
