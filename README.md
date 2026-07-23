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
| `ground` | 8 | 864,000 | 4.00 | - |
| `stairs` | 24 | 1,303,032 | 6.03 | - |
| `traversal_mantle` | 364 | 2,025,284 | 9.65 | **5,483** |
| `traversal_mantle_vault` | 250 | 2,158,286 | 10.77 | **9,795** |
| `traversal_vault` | 245 | 2,222,712 | 10.92 | **5,930** |
| **total** | **891** | **8,573,314** | **41.38** | **21,208** |

Full v1 data: [`MorphData_v1.zip`](https://pan.baidu.com/s/1skKd-Ds431xWPmUoKSicIQ)
(Baidu Netdisk, extract code: `md7d`).

Terrain meshes (the `terrain_<hash>.json` scene exports for every recording in
the v1 set, used by the retarget / viz pipeline): [`MorphDataTerrain_v1.zip`](https://pan.baidu.com/s/1smKxBp1n72U4CifP7BHZvQ)
(Baidu Netdisk, extract code: `ajpw`). Drop the extracted `terrain_*.json`
files into `data/sample/terrain/` (or point `--terrain-dir` at the folder you
extract them to).

This repo ships trimmed samples under `data/sample/` only; the full v1 set is
not vendored here.

> ⚠ **Clean the raw captures before use.** The v1 recordings contain two
> pervasive logging/capture artifacts — ~47% of all frames are stuck
> duplicate frames, and the rest is fragmented by 200k+ big capture gaps —
> which produce phantom velocities, 0.1 s+ holes, and 70+ m/s teleport spikes.
> **Always run the preprocessing pipeline in [`DataLib/preprocess/`](DataLib/preprocess)
> first** (see [§9. Preprocess / clean the raw captures](#9-preprocess--clean-the-raw-captures-important));
> it is non-destructive and reversible. The before/after tables there show
> why this step matters.

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
│   ├── preprocess/                      # OFFLINE cleaning of raw MorphData_v1 (run before retarget)
│   │   ├── clean_morph.py                # dt-gap + disp/dt-speed cleaner (stages segments)
│   │   ├── apply_clean.py                # swap staged segments into the dataset (backup + rollback)
│   │   ├── rollback_clean.py             # undo an apply_clean.py swap
│   │   ├── eval_clean.py                # eval disp/dt speed / dt dist / duration (before & after)
│   │   ├── seg_duration_dist.py          # duration distribution of cleaned segments
│   │   ├── browse_clean.py               # interactive G1 viewer; N/P jump between trajectories
│   │   └── README.md                     # rules, usage, before/after tables
│   └── gmr/                              # vendored trimmed GMR (no torch/etc.)
│       └── general_motion_retargeting/   # params, motion_retarget, neck_retarget, data_loader
│           ├── ik_configs/               # bvh_ue5_native_to_g1.json, bvh_ue5_g1scale_to_g1.json
│           └── assets/unitree_g1/        # g1_mocap_29dof.xml + meshes
├── data/
│   ├── sample/                           # 2 trimmed recordings per category (~1200 frames, ~28 MB each)
│   │   ├── ground/                       # flat-ground walk (no terrain)
│   │   ├── stairs/                       # stair climb recordings (motion only)
│   │   ├── traversal/                    # mantle traversal recordings (motion only)
│   │   └── terrain/                      # shared terrain_<hash>.json for all categories
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
   containing `<stem>_frames.jsonl` + `<stem>_meta.json` pairs. Put the
   referenced `terrain_<hash>.json` files together in one shared folder
   (e.g. `data/sample/terrain/`); each recording's `meta.json` points at its
   terrain via `terrain_ref`.
2. Export a single recording (pass the shared terrain folder with
   `--terrain-dir` so the viz terrain is loaded for context):

   ```bash
   # recording captured at the original UE height (root.mesh.s == 1.0)
   python DataLib/retarget/ue_world_skeleton_retarget.py \
       --jsonl data/mytraversal/MyRec_frames.jsonl \
       --meta  data/mytraversal/MyRec_meta.json \
       --terrain-dir data/sample/terrain \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --src-human bvh_ue5_g1scale --data-scale 0.77 --height-from-data \
       --output-qpos data/RetargetOutputs/mytraversal/MyRec_frames.npy \
       --no-visualize

   # recording captured with an already G1-scaled character (root.mesh.s == 0.77)
   python DataLib/retarget/ue_world_skeleton_retarget.py \
       --jsonl data/mystairs/MyRec_frames.jsonl \
       --meta  data/mystairs/MyRec_meta.json \
       --terrain-dir data/sample/terrain \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --src-human bvh_ue5_g1scale --data-scale 1.0 --height-from-data \
       --output-qpos data/RetargetOutputs/mystairs/MyRec_frames.npy \
       --no-visualize
   ```

   (`--terrain <path>` also works to point at a single terrain file
   explicitly; `--no-terrain` skips terrain entirely.)

3. Batch a whole folder (one category per invocation). `--terrain-dir` is
   forwarded to each retarget run:

   ```bash
   python DataLib/batch/batch_retarget.py \
       --input-dir data/mytraversal --output-dir data/RetargetOutputs/mytraversal \
       --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
       --data-scale 0.77 --terrain-dir data/sample/terrain \
       --extra-args "--src-human bvh_ue5_g1scale --height-from-data" --workers 4
   ```

4. Export terrain for a folder:

   ```bash
   python DataLib/terrain/convert_terrain.py \
       --terrain-dir data/sample/terrain \
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
* Terrain JSONs are kept together in `data/sample/terrain/` (one file per
  unique terrain hash) rather than duplicated inside each category folder.
  Every terrain-consuming script takes a `--terrain-dir` pointing there;
  recordings resolve theirs from `meta.json`'s `terrain_ref`.

---

## 9. Preprocess / clean the raw captures (important)

> **This step is mandatory before retargeting or training on the full v1
> set.** The raw `MorphData_v1` captures are not directly usable: ~47% of
> all frames are stuck duplicate frames (a logging artifact of writing
> multiple frames per engine tick, `dt≈1e-5` s with identical positions),
> and the rest is fragmented by 213,627 capture gaps of `dt>0.1` s caused by
> multi-threaded capture instability. Left untreated, these inflate
> per-frame displacement to 70+ m/s teleport spikes and punch 0.1 s+ holes
> through every clip — which then poison the retargeting IK and any model
> trained on the data. The cleaner in [`DataLib/preprocess/`](DataLib/preprocess)
> removes both classes of artifact and is **non-destructive and reversible**
> (originals are backed up; a rollback manifest is written).

### Cleaning rules (per frame `k`, with a predecessor)

```
dt < 0.005s            -> stuck frame: drop (dedup)
dt > 0.1s              -> big gap: SPLIT point (keep k as new segment start)
0.05 < dt <= 0.1s      -> small gap: keep k, interpolate missing frames
0.005<dt<0.05 and disp/dt>10 m/s -> mutation:
     return-speed |p[k+1]-p[k-1]|/(t[k+1]-t[k-1]) < 10 m/s -> glitch: drop k
     else                                              -> teleport: SPLIT point
else                   -> normal: keep
Segment filter: keep if n_frames >= 60 AND sum(dt) >= 1.0s
```

* `dt` is the **real** `t[k]-t[k-1]` (never the assumed 1/60), so stuck frames
  are caught instead of inflating speed.
* `disp/dt = |root.p[k]-root.p[k-1]| / dt / 100` (m/s; `root.p` is UE cm)
  measures **actual root displacement** — robust to the engine's phantom
  `phy.v` spikes and also catches kinematic position corrections `phy.v`
  does not reflect. 10 m/s is the natural gap between real motion
  (envelope ≤ ~7.5 m/s) and noise spikes (≥ 10 m/s, always 1–3 frame needles).

### Usage (from the `Morph/` directory)

```bash
# 1. Stage cleaned segments (originals untouched). Writes a manifest + stats.
python DataLib/preprocess/clean_morph.py \
    --data-root data/MorphData_v1 \
    --out-root  data/MorphData_v1_cleaned_staging

# 2. (optional) Evaluate the raw dataset first as a baseline.
python DataLib/preprocess/eval_clean.py --data-root data/MorphData_v1

# 3. Swap staged segments into the dataset (originals backed up).
python DataLib/preprocess/apply_clean.py \
    --data-root data/MorphData_v1 \
    --stage-root data/MorphData_v1_cleaned_staging \
    --backup-dir data/MorphData_v1_originals_backup

# 4. Evaluate the cleaned dataset in place.
python DataLib/preprocess/eval_clean.py --data-root data/MorphData_v1

# 5. Duration distribution of the cleaned segments.
python DataLib/preprocess/seg_duration_dist.py --data-root data/MorphData_v1

# Browse the cleaned segments in one MuJoCo G1 viewer (N = next trajectory).
python DataLib/preprocess/browse_clean.py --cat traversal_mantle

# Browse with terrain + ground (terrain mode; relaunches per clip).
# stairs works with the shipped samples; traversal_* need MorphDataTerrain_v1.
python DataLib/preprocess/browse_clean.py --cat stairs --terrain-dir data/sample/terrain

# Browse in the viser web 3D viewer instead of the native MuJoCo window.
# One persistent server; N/P (or the clip slider) swap terrain + robot in place
# — no window relaunch even with terrain. Needs `pip install viser`.
python DataLib/preprocess/browse_clean.py --cat traversal_vault --viewer viser \
    --terrain-dir data/MorphData_v1/terrain --port 8080

# Rollback if needed:
python DataLib/preprocess/rollback_clean.py --data-root data/MorphData_v1
```

All paths and rule thresholds are CLI-configurable; see
[`DataLib/preprocess/README.md`](DataLib/preprocess/README.md) for the full
reference and tuning guide.

### Before vs after on MorphData_v1 (v1)

Cleaning stats:

| metric | value |
|---|---:|
| input files (segments) | 1,858 |
| input frames | 8,546,577 |
| **output segments** | **4,746** |
| **output frames** | **4,257,710** (49.8% retained) |
| dropped stuck frames (dt<0.005) | 4,058,815 |
| deleted glitch frames | 9,007 |
| big-gap split points (dt>0.1) | 213,627 |
| small-gap interpolated (0.05<dt<=0.1) | 2,361 |
| dropped segments (<60 frames or <1s) | 210,739 |
| unchanged files | 15 |

Before vs after (disp/dt real displacement speed):

| metric | before | after | change |
|---|---:|---:|---|
| segments | 1,858 | **4,746** | +2,888 (split) |
| total frames | 8,546,577 | **4,257,710** | −4,288,867 (−50%, mostly stuck dupes) |
| total duration | 40.68 h | **19.86 h** | −20.82 h (−51%, mostly <1s shards dropped) |
| stuck frames (dt<0.005) | many | **0** | eliminated ✓ |
| big gaps (dt>0.1) | 213,627 | **0** | eliminated ✓ |
| global max dt | 5.67 s | **0.066 s** | no big gaps ✓ |
| per-segment median dt | ~0.013–0.017 | **0.015 s** | back to ~1/60 ✓ |
| disp/dt speed peak | 74.3 m/s | **13.4 m/s** | large drop ✓ |
| >10 m/s frames | — | **3** | only small-gap boundary frames (kept by rule) |
| >15 m/s frames | many | **0** | ✓ |

Threshold hits after cleaning (disp/dt real speed):

| speed threshold | frames hit |
|---|---:|
| > 6 m/s | 39,560 |
| > 8 m/s | 5,633 |
| > 10 m/s | 3 |
| > 15 m/s | 0 |
| > 20 m/s | 0 |
| > 30 m/s | 0 |

Per-category distribution after cleaning:

| category | segments | frames | hours | disp/dt peak (m/s) |
|---|---:|---:|---:|---:|
| ground | 21 | 840,406 | 3.90 | 8.2 |
| stairs | 230 | 380,699 | 1.79 | 10.0 |
| traversal_mantle | 1,588 | 1,104,080 | 5.13 | 10.0 |
| traversal_mantle_vault | 1,950 | 1,310,826 | 6.14 | 10.0 |
| traversal_vault | 957 | 621,699 | 2.91 | 13.4 |
| **total** | **4,746** | **4,257,710** | **19.86** | — |

Duration distribution of the cleaned segments (n=4,746, 19.86 h):

| bin (s) | segments | % |
|---|---:|---:|
| <1 | 1 | 0.0% |
| 1–2 | 733 | 15.4% |
| 2–3 | 459 | 9.7% |
| 3–5 | 712 | 15.0% |
| **5–10** | **996** | **21.0%** |
| 10–20 | 962 | 20.3% |
| 20–30 | 439 | 9.2% |
| 30–60 | 351 | 7.4% |
| 60–120 | 77 | 1.6% |
| 120–300 | 7 | 0.1% |
| 300–600 | 1 | 0.0% |
| 600–1800 | 8 | 0.2% |

Percentiles: p1=1.11s, p5=1.36s, p10=1.51s, p25=2.97s, **p50=7.29s**,
p75=15.96s, p90=28.95s, p95=41.15s, p99=67.43s; min=1.00s, max=1750.15s,
mean=15.07s. Bimodal + long-tailed: primary peak 5–20 s (41%), secondary peak
1–3 s (25%), long tail to ~29 min (all from `ground`).

![segment duration distribution](DataLib/preprocess/assets/seg_duration_dist.png)

> **Note on the 51% duration loss.** It is dominated by sub-second shards
> being dropped (210,739 segments), not by valid motion being deleted — the
> big gaps already shredded the raw data into sub-second pieces. To keep more
> short actions, lower `--min-frames` (e.g. 30) or `--min-dur` (e.g. 0.5), or
> interpolate 0.1–0.5 s gaps instead of splitting them. The 3 residual >10 m/s
> frames are small-gap (0.05–0.1 s) boundary frames, which the rule
> interpolates rather than splits (3/4.26M ≈ 0.00007%).

