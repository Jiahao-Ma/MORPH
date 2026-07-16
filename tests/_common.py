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
OUTPUTS = DATA / "RetargetOutputs"

PY = sys.executable

# Per-category retarget configuration.
#   stairs      : recorded with a G1-scaled UE character  -> src=bvh_ue5_g1scale, scale=1.0
#   ground/trav : recorded with a 1.75 m (original) UE char -> src=bvh_ue5_native, scale=0.77
#
# Each category lists >=1 recording; all stairs recordings share terrain
# E4B166AB and all traversal recordings share terrain 581C18B5, so only one
# terrain_*.json per category is needed in data/sample/<cat>/.
CATEGORIES = {
    "ground": {
        "src_human": "bvh_ue5_native",
        "data_scale": 0.77,
        "config": RETARGET_DIR / "gasp_bvh_alignment.json",
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
        "src_human": "bvh_ue5_native",
        "data_scale": 0.77,
        "config": RETARGET_DIR / "gasp_bvh_alignment.json",
        "terrain_hash": "581C18B5",
        "recordings": [
            "Traversal_C_163_5X5oWEQK",
            "Traversal_C_164_HbggcUi1",
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
        "terrain": (d / f"terrain_{cfg['terrain_hash']}.json") if has_terrain else None,
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
