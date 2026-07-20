# Morph — UE World-Skeleton → Unitree G1 Retarget + Terrain Export

Morph turns motion captures recorded from an Unreal Engine 5 AI character
(GASP traversal project) into Unitree G1 robot joint trajectories (`qpos`) and
exports the matching UE terrain into MuJoCo (MJCF + OBJ) and IsaacLab
(OBJ + NPZ height-field) formats. It also visualizes the retargeted motion
together with the (scale-matched) terrain.

The repository is self-contained: the retargeting engine is a **trimmed,
vendored copy** of the [GMR](https://github.com/YanjieZe/GMR) library under
`DataLib/gmr/`, so no external `general_motion_retargeting` install is needed.
You only supply your own recordings (drop them under `data/`) and run the
scripts for post-processing, export, and visualization.

---

## Dataset (v1)

First-release GASP capture stats (`RetargetInputs`, 60 Hz). Traversal counts
are **continuous `cmd.jump=true` runs** (one pulse streak = one attempt, not
per-frame).

| category | clips | frames | hours | traversal attempts |
|----------|------:|-------:|------:|-------------------:|
| `ground` | 8 | 864,000 | 4.00 | — |
| `stairs` | 24 | 1,303,032 | 6.03 | — |
| `traversal_mantle` | 364 | 2,025,284 | 9.65 | **5,483** |
| `traversal_mantle_vault` | 250 | 2,158,286 | 10.77 | **9,795** |
| `traversal_vault` | 245 | 2,222,712 | 10.92 | **5,930** |
| **total** | **891** | **8,573,314** | **41.38** | **21,208** |

This repo ships trimmed samples under `data/sample/` only; the full v1 set is
not vendored here.

---

## 1. Repository layout

```
Morph/
├── DataLib/
│   ├── retarget/                      # motion retarget + export + viewers
│   │   ├── ue_world_skeleton_retarget.py   # main: retarget + export qpos/cmd/scaled
│   │   ├── batch_retarget.py               # batch driver over a recording dir
│   │   ├── verify_pipeline.py              # rich viewer (already G1-scaled data)
│   │   ├── verify_pipeline_scaled.py       # rich viewer (original-height data, --data-scale)
│   │   ├── vis_skeleton_compare.py          # GASP data loader + skeleton compare
│   │   ├── gasp_bvh_calibrate.py           # Kabsch alignment calibration
│   │   ├── gasp_bvh_alignment.json         # alignment config  (1.75 m human src)
│   │   ├── gasp_bvh_alignment_g1_height.json  # alignment config (G1-scaled src)
│   │   └── joint_mapping.json              # robot↔human joint pairing for errors
│   ├── terrain/
│   │   └── convert_terrain.py             # UE terrain JSON -> MuJoCo / IsaacLab
│   └── gmr/                              # vendored trimmed GMR (no torch/etc.)
│       └── general_motion_retargeting/   # params, motion_retarget, neck_retarget, data_loader
│           ├── ik_configs/               # bvh_ue5_native_to_g1.json, bvh_ue5_g1scale_to_g1.json
│           └── assets/unitree_g1/        # g1_mocap_29dof.xml + meshes
├── data/
│   ├── sample/                           # 2 trimmed recordings per category (~1200 frames, ~28 MB each)
│   │   ├── ground/                       # flat-ground walk (no terrain)
│   │   ├── stairs/                       # stair climb + terrain_E4B166AB.json
│   │   └── traversal/                    # mantle traversal + terrain_581C18B5.json
│   └── RetargetOutputs/                  # generated exports (gitignored-ish; tests write here)
├── tests/                                # end-to-end reproducibility tests
│   ├── _common.py
│   ├── test_export.py                    # retarget+export all 3 categories
│   ├── test_terrain.py                   # terrain export (stairs + traversal)
│   ├── test_batch.py                     # batch_retarget per category
│   ├── visualize.py                      # interactive MuJoCo viewer
│   └── run_all.py                        # run export+terrain+batch
├── requirements.txt
└── README.md
```

---

## 2. Install

```bash
conda create -n morph python=3.10 -y
conda activate morph
pip install -r requirements.txt
```

That's it — `mujoco`, `mink`, `numpy`, `scipy`, `rich` are the only runtime
deps. The vendored GMR is imported via `DataLib/gmr/` (the scripts put it on
`sys.path` automatically, and the tests also set `PYTHONPATH`).

---

## 3. The scaling gap (important)

The pipeline reads world-space joint positions (`joints[*].wp`), which
already include the in-engine character mesh scale (`root.mesh.s` in the
JSONL). Check that field to know what a recording actually contains:

| category   | `root.mesh.s` | effective data height | src-human         | `--data-scale` | alignment config                    |
|------------|---------------|-----------------------|-------------------|----------------|-------------------------------------|
| `ground`   | `0.77`        | already G1            | `bvh_ue5_g1scale` | `1.0`          | `gasp_bvh_alignment_g1_height.json` |
| `stairs`   | `0.77`        | already G1            | `bvh_ue5_g1scale` | `1.0`          | `gasp_bvh_alignment_g1_height.json` |
| `traversal`| `1.0`         | original UE height    | `bvh_ue5_g1scale` | `0.77`         | `gasp_bvh_alignment_g1_height.json` |

G1 is ≈ 0.77× the height of the original UE character. `--data-scale`
uniformly scales the source UE motion positions (and the visualization
terrain) about the UE world origin **before** the Kabsch/ball-align/retarget
pipeline runs. The rule is simple: after `mesh.s × data-scale` everything
must be at G1 height, and then **every** category uses the same G1-scale
pairing (`bvh_ue5_g1scale` + `gasp_bvh_alignment_g1_height.json`).

> ⚠ Do NOT stack `--data-scale 0.77` on a recording whose `mesh.s` is already
> `0.77`, and do not pair scaled data with the `bvh_ue5_native` /
> `gasp_bvh_alignment.json` combo (that combo is only for unscaled 1.75 m
> data with `--data-scale 1.0`). A wrong pairing shrinks the IK targets far
> below G1's reachable pose, drives the solver against joint limits and shows
> up as violent pelvis jitter.

`ue_world_skeleton_retarget.py` and `batch_retarget.py` both accept
`--data-scale` (the batcher forwards it to the retarget script).

Two more flags matter for output quality (both used by the tests):

* `--height-from-data` — estimate the human height from the motion itself
  (same as the verify_pipeline viewers) instead of the config constant, so
  the export scale matches the verified diagnostic view.
* `--smooth-win N` (default 9) — zero-phase temporal smoothing of the root
  XY trajectory and the grounding Δz, so raw per-frame capture noise does not
  feed 1:1 into the G1 pelvis. Set `0` to disable.

---

## 4. Quick start — run the tests

From the `Morph/` directory:

```bash
# (a) Export G1 qpos for ground + stairs + traversal (no viewer)
python tests/test_export.py

# (b) Export terrain (MuJoCo + IsaacLab, with appended ground plane)
python tests/test_terrain.py

# (c) Batch retarget, one category at a time
python tests/test_batch.py

# (d) All of the above, in order
python tests/run_all.py
```

Outputs land under `data/RetargetOutputs/`:

```
data/RetargetOutputs/<cat>/<stem>_frames.npy        # G1 qpos, [n_frames, nq]
data/RetargetOutputs/<cat>/<stem>_frames_cmd.npy    # per-frame command/root struct
data/RetargetOutputs/<cat>/<stem>_frames_scaled.npz # pre-IK scaled skeleton
data/RetargetOutputs/terrain/<cat>_mujoco/<hash>.xml   + <hash>.obj
data/RetargetOutputs/terrain/<cat>_isaaclab/<hash>.obj + <hash>.npz
```

---

## 5. Visualize

```bash
# Retargeted G1 + terrain in the MuJoCo viewer
python tests/visualize.py --cat ground
python tests/visualize.py --cat stairs
python tests/visualize.py --cat traversal

# Each category has 2 recordings — pick the 2nd with --rec 1
python tests/visualize.py --cat stairs --rec 1

# Richer diagnostic viewer (raw UE / Kabsch-aligned / retargeted G1).
# Defaults to --viewer mujoco (no extra deps). Use --viewer viser after
# `pip install viser` for the web 3D viewer.
python tests/visualize.py --cat stairs --mode verify      # uses verify_pipeline.py
python tests/visualize.py --cat ground --mode verify      # uses verify_pipeline_scaled.py
python tests/visualize.py --cat traversal --mode verify --viewer viser
```

> Windows console note: the scripts reconfigure stdout/stderr to UTF-8 so the
> non-ASCII characters in their output (arrows, check-marks) don't crash on a
> GBK code page. If you ever see a `UnicodeEncodeError`, set
> `$env:PYTHONIOENCODING="utf-8"` before running.

---

## 6. Running on your own data

1. Drop your recordings under `data/` — one folder per category, each
   containing `<stem>_frames.jsonl` + `<stem>_meta.json` pairs (and, for
   stairs/traversal, the referenced `terrain_<hash>.json`).
2. Export a single recording:

   ```bash
   # recording captured at the original UE height (root.mesh.s == 1.0)
   python DataLib/retarget/ue_world_skeleton_retarget.py \
       --jsonl data/mytraversal/MyRec_frames.jsonl \
       --meta  data/mytraversal/MyRec_meta.json \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --src-human bvh_ue5_g1scale --data-scale 0.77 --height-from-data \
       --output-qpos data/RetargetOutputs/mytraversal/MyRec_frames.npy \
       --no-visualize

   # recording captured with an already G1-scaled character (root.mesh.s == 0.77)
   python DataLib/retarget/ue_world_skeleton_retarget.py \
       --jsonl data/mystairs/MyRec_frames.jsonl \
       --meta  data/mystairs/MyRec_meta.json \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --src-human bvh_ue5_g1scale --data-scale 1.0 --height-from-data \
       --output-qpos data/RetargetOutputs/mystairs/MyRec_frames.npy \
       --no-visualize
   ```

3. Batch a whole folder (one category per invocation):

   ```bash
   python DataLib/batch/batch_retarget.py \
       --input-dir data/mytraversal --output-dir data/RetargetOutputs/mytraversal \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --data-scale 0.77 \
       --extra-args "--src-human bvh_ue5_g1scale --height-from-data" --workers 4
   ```

4. Export terrain for a folder:

   ```bash
   python DataLib/terrain/convert_terrain.py \
       --terrain-dir data/mystairs \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --output-dir data/RetargetOutputs/terrain --name mystairs
   ```

   A flat ground plane is appended by default (the UE terrain export only
   contains stair/box geometry, no floor). Use `--no-ground` to disable, or
   `--ground-margin` / `--ground-z` to tune it.

---

## 7. Output format reference

Motion (per recording):

* `<stem>.npy` — `np.ndarray`, shape `[n_frames, nq]`, dtype float64. G1
  `nq` = 7 (floating base: 3 pos + 4 quat wxyz) + 29 joints = 36.
* `<stem>_cmd.npy` — structured numpy array, one row per frame, with the
  locomotion command + root transform (see `COMMAND_DTYPE` in
  `ue_world_skeleton_retarget.py`).
* `<stem>_scaled.npz` — pre-IK scaled skeleton (the UE motion after
  `--data-scale` + UE→MuJoCo transform + Kabsch), used for inspection/debug.

Terrain (per `--name`):

* `<name>_mujoco/<hash>.xml` — standalone MJCF with inline mesh.
* `<name>_mujoco/<hash>.obj` — OBJ of the same mesh.
* `<name>_isaaclab/<hash>.obj` — OBJ, Z-up, right-handed, metres.
* `<name>_isaaclab/<hash>.npz` — `{vertices, faces, height_field, ...}`.

---

## 8. Notes

* The vendored GMR under `DataLib/gmr/` is a **trimmed subset** of upstream
  GMR — only `params`, `motion_retarget`, `neck_retarget`, `data_loader`,
  the two `bvh_ue5_*_to_g1.json` IK configs, and the `unitree_g1` asset +
  meshes. The upstream viewer / streaming / torch-based helpers are omitted
  to keep the dep surface small. If you need those, install full GMR from
  https://github.com/YanjieZe/GMR.
* `rot_utils.py` (which pulls `torch`) was intentionally excluded from the
  vendored subset.
* Sample recordings are trimmed to the first 1200 frames (~28 MB each, 20 s
  @ 60 Hz) of the full captures, two per category, so the repo stays
  cloneable; the pipeline behaves identically on full recordings.
