"""Calibrate GASP ↔ BVH skeleton alignment and save a config file.

Two-step alignment, computed once on a chosen calibration frame
(idle pose recommended for both BVH and GASP):

  1) GLOBAL rigid alignment (Kabsch / Umeyama) on a set of anchor bones:
        R_global, t_global   such that
        R_global @ p_gasp + t_global  ≈  p_bvh
     This removes whole-body rotation/translation differences.

  2) PER-BONE residual delta (after applying R_global to GASP rotations):
        Δ_i = (R_global · R_gasp_i)^T · R_bvh_i
     This captures each bone's local rest-frame mismatch
     (UE5 mannequin bind orientation vs BVH-FK identity rest).

The two are saved together to a JSON config consumed by
ue_world_skeleton_retarget.py:

  global.rotation_quat_wxyz, global.translation_xyz
  per_bone_delta_quat_wxyz   = { bone_name: [w, x, y, z] }

Frame selection:
  The BVH calibration frame is fixed (default 0). For GASP we *search* every
  frame of --gasp-jsonl (optionally strided) and pick the one whose anchor
  bones, after optimal rigid alignment, best match the BVH anchor positions
  (lowest Kabsch residual RMS). This way the per-bone Δ truly captures only
  the structural binding-orientation difference — not pose differences from a
  randomly-chosen GASP frame. Override with --gasp-frame N to pin a frame.

Usage:
  python gasp_bvh_calibrate.py
  python gasp_bvh_calibrate.py --bvh ... --gasp-jsonl ... --gasp-meta ... \
                               --output gasp_bvh_alignment.json
  python gasp_bvh_calibrate.py --gasp-frame 0     # skip search, fix GASP frame
  python gasp_bvh_calibrate.py --bvh-frame 5      # use BVH frame 5 as reference
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the verified loaders from vis_skeleton_compare so we use the *exact*
# same coordinate convention (BVH-native: X=skel_left, Y=skel_fwd, Z=up, m).
from vis_skeleton_compare import (  # noqa: E402
    parse_bvh,
    compute_fk,
    load_gasp_data,
    compute_per_bone_delta,
)

DATA_DIR = HERE.parent / "data"

# Default calibration sources: idle animation in BOTH formats. This is the
# cleanest case because the same physical motion is represented in both files,
# so frame-N of BVH and frame-N of GASP are in approximately the same pose.
DEFAULT_BVH = str(DATA_DIR / "UEGMR" / "step2_bvh" / "M_Neutral_Stand_Idle_Loop.bvh")
DEFAULT_JSONL = str(DATA_DIR / "stand_idle_loop_frames.jsonl")
DEFAULT_META  = str(DATA_DIR / "stand_idle_loop_meta.json")
DEFAULT_OUTPUT = str(HERE / "gasp_bvh_alignment.json")

# Anchor bones spread across the body, used to constrain the global Kabsch fit.
# All must be present in BOTH skeletons.
DEFAULT_ANCHORS = [
    "pelvis", "spine_03", "spine_05", "head",
    "clavicle_l", "clavicle_r",
    "upperarm_l", "upperarm_r",
    "hand_l", "hand_r",
    "thigh_l", "thigh_r",
    "calf_l", "calf_r",
    "foot_l", "foot_r",
]


def kabsch(P_src: np.ndarray, P_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find proper rotation R and translation t minimizing
    ||R @ P_src.T + t  -  P_dst.T||^2 .

    P_src, P_dst : (N, 3) corresponding 3D points (rows = points).
    Returns: R (3,3), t (3,)
    """
    assert P_src.shape == P_dst.shape and P_src.shape[1] == 3
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)        # 3x3 covariance
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))   # ±1 to enforce det(R)=+1
    D = np.diag([1.0, 1.0, d])
    R_mat = Vt.T @ D @ U.T
    t_vec = c_dst - R_mat @ c_src
    return R_mat, t_vec


def kabsch_residual_rms(P_src: np.ndarray, P_dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Kabsch and return (rms_m, R, t)."""
    R_mat, t_vec = kabsch(P_src, P_dst)
    aligned = (R_mat @ P_src.T).T + t_vec
    rms = float(np.sqrt(np.mean(np.sum((aligned - P_dst) ** 2, axis=1))))
    return rms, R_mat, t_vec


def find_best_gasp_frame(
    bvh_anchor_pos_m: np.ndarray,
    gasp_pos_mj: np.ndarray,
    gasp_anchor_indices: list[int],
    stride: int = 1,
    max_frames: int | None = None,
) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    """For every (strided) GASP frame, run Kabsch on the anchor bones against
    the fixed BVH anchor positions and score by residual RMS.

    Returns: list of (frame_idx, rms_m, R, t) sorted ascending by rms.
    The lowest-rms frame is the GASP pose closest to the BVH reference pose
    (after optimal rigid alignment) — and therefore the safest frame to use
    for computing the per-bone structural Δ.
    """
    n_total = gasp_pos_mj.shape[0]
    upper = min(n_total, max_frames) if max_frames is not None else n_total
    candidates = list(range(0, upper, max(1, stride)))
    scores: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for fi in candidates:
        P_src = gasp_pos_mj[fi, gasp_anchor_indices, :]   # (N, 3)
        rms, R_f, t_f = kabsch_residual_rms(P_src, bvh_anchor_pos_m)
        scores.append((fi, rms, R_f, t_f))
    scores.sort(key=lambda s: s[1])
    return scores


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bvh", default=DEFAULT_BVH)
    ap.add_argument("--gasp-jsonl", default=DEFAULT_JSONL,
                    help="GASP recording. Auto-search picks the frame whose pose "
                         "best matches the BVH reference frame.")
    ap.add_argument("--gasp-meta", default=DEFAULT_META)
    ap.add_argument("--bvh-frame", type=int, default=0,
                    help="BVH reference frame (default 0 — start of idle loop)")
    ap.add_argument("--gasp-frame", type=int, default=None,
                    help="If set, pin the GASP calibration frame instead of "
                         "auto-searching across the recording.")
    ap.add_argument("--search-stride", type=int, default=1,
                    help="Search every Nth GASP frame (default 1 = all frames)")
    ap.add_argument("--search-max-frames", type=int, default=None,
                    help="Limit the search to the first N GASP frames")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Print the K best-matching GASP frames before picking")
    ap.add_argument("--anchors", nargs="*", default=DEFAULT_ANCHORS,
                    help="Bone names used for global Kabsch alignment (≥3 required)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Where to write the JSON alignment config")
    args = ap.parse_args()

    # ── Load BVH at fixed reference frame ────────────────────────────────
    print(f"[BVH ] {args.bvh}")
    bvh = parse_bvh(args.bvh)
    bvh_names = [j.name for j in bvh.joints]
    cf_b = max(0, min(args.bvh_frame, bvh.num_frames - 1))
    bvh_pos_cm, bvh_rot_frame = compute_fk(bvh, cf_b)
    bvh_pos_m = bvh_pos_cm * 0.01
    print(f"      frames={bvh.num_frames}, bones={len(bvh_names)}, ref_frame={cf_b}")

    # ── BVH rest-pose human height ─────────────────────────────────────────
    # Same formula as step4_retarget_bvh_to_g1.py / vis_skeleton_compare.py:
    # accumulate joint offsets with identity rotations (T-pose) and add ~9cm
    # for skull-top above the head bone. This is the canonical, frame-
    # independent height used by GMR.
    rest_pos_cm = np.zeros((len(bvh.joints), 3))
    for bi, joint in enumerate(bvh.joints):
        if joint.parent_idx < 0:
            rest_pos_cm[bi] = joint.offset
        else:
            rest_pos_cm[bi] = rest_pos_cm[joint.parent_idx] + joint.offset
    bvh_name_to_idx = {j.name: i for i, j in enumerate(bvh.joints)}
    head_rest_z_m = (rest_pos_cm[bvh_name_to_idx["head"]][2] * 0.01
                     if "head" in bvh_name_to_idx else 1.7)
    bvh_human_height = float(head_rest_z_m + 0.09)
    print(f"      rest-pose human height = {bvh_human_height:.3f} m")

    # ── Load all GASP frames ─────────────────────────────────────────────
    print(f"[GASP] {args.gasp_jsonl}")
    gasp_pos_mj, gasp_rot_all, gasp_names, _ = load_gasp_data(args.gasp_jsonl, args.gasp_meta)
    n_gasp_frames = gasp_pos_mj.shape[0]
    print(f"      frames={n_gasp_frames}, bones={len(gasp_names)}")

    # ── Validate anchor bones ────────────────────────────────────────────
    bvh_idx  = {n: i for i, n in enumerate(bvh_names)}
    gasp_idx = {n: i for i, n in enumerate(gasp_names)}
    anchor_used = [n for n in args.anchors if n in bvh_idx and n in gasp_idx]
    if len(anchor_used) < 3:
        raise RuntimeError(
            f"Need ≥3 shared anchor bones for Kabsch, got: {anchor_used}"
        )
    bvh_anchor_pos_m   = np.stack([bvh_pos_m[bvh_idx[n]]   for n in anchor_used])
    gasp_anchor_idxs   = [gasp_idx[n] for n in anchor_used]

    # ── Step 1a: Pick the best-matching GASP frame ───────────────────────
    if args.gasp_frame is not None:
        cf_g = max(0, min(args.gasp_frame, n_gasp_frames - 1))
        rms_pinned, R_global, t_global = kabsch_residual_rms(
            gasp_pos_mj[cf_g, gasp_anchor_idxs, :], bvh_anchor_pos_m,
        )
        print(f"\n[Step 1a: GASP frame pinned by --gasp-frame]")
        print(f"  Using GASP frame {cf_g}, Kabsch RMS = {rms_pinned*100:.2f} cm")
        rms_m = rms_pinned
        search_summary: list[dict] = []
        searched_count = 0
    else:
        print(f"\n[Step 1a: Searching {n_gasp_frames} GASP frames for best pose match "
              f"to BVH frame {cf_b}]")
        scores = find_best_gasp_frame(
            bvh_anchor_pos_m, gasp_pos_mj, gasp_anchor_idxs,
            stride=args.search_stride, max_frames=args.search_max_frames,
        )
        searched_count = len(scores)
        print(f"  Searched {searched_count} candidate frames "
              f"(stride={args.search_stride}, max_frames={args.search_max_frames})")
        print(f"  Top {min(args.top_k, len(scores))} matches (lowest Kabsch RMS first):")
        print(f"    {'frame':>8s}  {'RMS_cm':>8s}  {'R_angle_deg':>12s}  {'t_xyz_m':>30s}")
        for fi, rms_f, R_f, t_f in scores[: args.top_k]:
            ang = float(np.linalg.norm(R.from_matrix(R_f).as_rotvec(degrees=True)))
            t_str = f"[{t_f[0]:+.3f}, {t_f[1]:+.3f}, {t_f[2]:+.3f}]"
            print(f"    {fi:>8d}  {rms_f*100:>8.2f}  {ang:>12.2f}  {t_str:>30s}")
        cf_g, rms_m, R_global, t_global = scores[0]
        search_summary = [
            {"frame": int(fi), "rms_m": float(rms_f)}
            for fi, rms_f, _, _ in scores[: args.top_k]
        ]
        print(f"  → Picked GASP frame {cf_g}  (RMS = {rms_m*100:.2f} cm)")

    gasp_pos_m = gasp_pos_mj[cf_g]
    gasp_rot_frame = gasp_rot_all[cf_g]

    # ── Step 1b: Report the chosen global alignment ──────────────────────
    R_global_rotvec = R.from_matrix(R_global).as_rotvec(degrees=True)
    R_global_angle  = float(np.linalg.norm(R_global_rotvec))

    print(f"\n[Step 1b: Global Kabsch alignment at chosen frame]")
    print(f"  BVH frame  = {cf_b}     GASP frame = {cf_g}")
    print(f"  Anchors used ({len(anchor_used)}): {anchor_used}")
    print(f"  R_global axis-angle: axis={R_global_rotvec/max(R_global_angle,1e-9)} "
          f"angle={R_global_angle:.2f} deg")
    print(f"  t_global (m): {t_global}")
    print(f"  Residual RMS after alignment: {rms_m*100:.2f} cm")

    # ── Step 2: Per-bone Δ on globally-aligned GASP rotations ──
    aligned_gasp_rot = [R_global @ Rg for Rg in gasp_rot_frame]
    delta = compute_per_bone_delta(
        bvh_rot_frame,    bvh_names,
        aligned_gasp_rot, gasp_names,
    )

    print(f"\n[Step 2: Per-bone Δ (after global align)]")
    delta_quat_dict: dict[str, list[float]] = {}
    delta_angle_log: list[tuple[str, float]] = []
    for name in gasp_names:
        if name not in delta:
            continue
        rv = R.from_matrix(delta[name]).as_rotvec(degrees=True)
        ang = float(np.linalg.norm(rv))
        delta_angle_log.append((name, ang))
        delta_quat_dict[name] = R.from_matrix(delta[name]).as_quat(scalar_first=True).tolist()

    # Print angles for the bones GMR actually uses (IK match table) so user
    # can quickly see whether the calibration looks reasonable.
    ik_relevant = {
        "pelvis", "spine_05",
        "thigh_l", "calf_l", "foot_l", "ball_l",
        "thigh_r", "calf_r", "foot_r", "ball_r",
        "upperarm_l", "lowerarm_l", "hand_l",
        "upperarm_r", "lowerarm_r", "hand_r",
    }
    for name, ang in delta_angle_log:
        if name in ik_relevant:
            print(f"  {name:18s}  Δ angle = {ang:6.2f} deg")
    print(f"  ... ({len(delta_quat_dict)} bones stored in total)")

    # ── Save config ──
    R_global_quat_wxyz = R.from_matrix(R_global).as_quat(scalar_first=True).tolist()
    config = {
        "schema_version": "1.2",
        "description": "GASP→BVH skeleton alignment for GMR retargeting (Path A: "
                       "global rigid + per-bone delta).",
        "coord_frame": "BVH-native: X=skel_left, Y=skel_fwd, Z=up, RH, m",
        "calib_meta": {
            "bvh": str(args.bvh),
            "gasp_jsonl": str(args.gasp_jsonl),
            "gasp_meta": str(args.gasp_meta),
            "bvh_calib_frame": cf_b,
            "gasp_calib_frame": cf_g,
            "gasp_frame_pinned": args.gasp_frame is not None,
            "anchors": anchor_used,
            "kabsch_residual_rms_m": rms_m,
            "global_rotation_angle_deg": R_global_angle,
            "search": {
                "searched_frames": int(searched_count),
                "search_stride": int(args.search_stride),
                "search_max_frames": (None if args.search_max_frames is None
                                      else int(args.search_max_frames)),
                "top_matches": search_summary,
            },
        },
        # Canonical, frame-independent human height computed from BVH rest pose.
        # GMR uses this to scale source motion to robot size; using a per-frame
        # estimate from a mid-action pose (e.g. flip_run frame 0) gives a much
        # smaller height and causes the retargeted G1 feet to clip through the
        # ground. Prefer this value over any heuristic in the retarget script.
        "actual_human_height_m": bvh_human_height,
        "global": {
            "rotation_quat_wxyz": R_global_quat_wxyz,
            "translation_xyz": t_global.tolist(),
        },
        "per_bone_delta_quat_wxyz": delta_quat_dict,
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2))
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
