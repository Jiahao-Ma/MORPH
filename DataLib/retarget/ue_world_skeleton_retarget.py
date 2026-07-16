r"""Retarget GASP (UE5 World) skeleton to Unitree G1 using BVH-native GMR.

The retarget pipeline matches debug/verify_pipeline.py (stages s1/s3/s5):
  s1) File read: bone world positions (wp) directly from JSONL (UE world cm).
  s3) UE → MuJoCo (X↔Y + cm→m) + Kabsch global alignment (R_global/t_global).
  s5) Full retarget:
      1) Positions come from wp (skel_mj = UE→MuJoCo of joints[*].wp) — the same
         source as the Kabsch skeleton used for grounding (no ground_z_cm
         mismatch). Rotations come from vis_skeleton_compare.load_gasp_data
         (cq + root.q → BVH coords).
      2) Build GMR input frames: apply (R_global, t_global) + per-bone Δ, with
         root-XY recentered to the origin (restored to qpos after GMR).
      3) Pre-IK grounding (method 3): re-anchor the scaled-human IK targets in
         Z at the support foot (per-foot toe pairing + soft-min handover), then
         back-project the shift onto the GMR input (δ = dz/s_root).
      4) GMR retarget → G1 qpos. --src-human selects the IK config:
         'bvh_ue5_g1scale' (default, character pre-scaled to ~1.32 m G1 height)
         or 'bvh_ue5_native' (1.75 m human character).
      5) Root-Z grounding (toe-to-toe): one per-frame root Z shift so the robot's
         lower toe-to-ground matches the human's lower toe-to-ground.
      6) Per-foot grounding (--per-foot-ground): independently bend each leg so
         each toe_link lands on its own target ground (stairs / uneven feet).

Data export (three sibling files, auto-named after the input JSONL stem):
  Given e.g. WalkTurn_C_8_DjMGcE36_frames.jsonl, the outputs (in the JSONL's
  directory, unless --output-qpos overrides the base path) are:
    <stem>.npy        part 2: IK-optimised G1 joint info (qpos, (T, nq))
    <stem>_cmd.npy    part 3: per-frame Player Command + Actor root block
                              (cmd.lookY/move/desYR, root.q/lv) as a STRUCTURED
                              array (COMMAND_DTYPE) — never on top of the motion.
    <stem>_scaled.npz part 1: scaled pre-IK skeleton (the GMR IK *input*) —
                              joint positions the IK solver tries to match, in
                              the same MuJoCo display frame as the qpos, bundled
                              with their joint names.
  A downstream policy can load each independently:
        motion = np.load("WalkTurn_..._frames.npy")          # (T, nq) qpos
        cmd    = np.load("WalkTurn_..._frames_cmd.npy")      # (T,)    structured
        scaled = np.load("WalkTurn_..._frames_scaled.npz")   # positions + names
  Pass --no-save to skip all exports (e.g. visualization-only runs).

Visualization (kept): replay the G1 with trajectory overlay + orientation /
velocity ground arrows (toggle orange/cyan with 'O', green/red with 'V').

This script intentionally contains *no* alignment computation. It only
*consumes* the config. To (re)compute the config, run gasp_bvh_calibrate.py.

Usage (Windows PowerShell):
  cd D:/tool/ue5/UnrealProjects/GASP/Scripts

  # S1: raw UE world wp (no --config needed)
  python ./UEGMR/Retargeting/ue_world_skeleton_retarget.py `
      --jsonl ./data/walk_turn_frames.jsonl --meta ./data/walk_turn_meta.json `
      --stage s1

  # S3: + MuJoCo + Kabsch
  python ./UEGMR/Retargeting/ue_world_skeleton_retarget.py `
      --jsonl ./data/walk_turn_frames.jsonl --meta ./data/walk_turn_meta.json `
      --config ./UEGMR/Retargeting/gasp_bvh_alignment.json --stage s3

  # S5: full retarget + export + viewer (default src-human = bvh_ue5_g1scale,
  #     which pairs with the G1-height calibration config).
  python ./UEGMR/Retargeting/ue_world_skeleton_retarget.py `
      --jsonl ./data/walk_turn_frames.jsonl --meta ./data/walk_turn_meta.json `
      --config ./tools/gasp_bvh_alignment_g1_height.json --stage s5 `
      --output-qpos data/walk_turn.npy --per-foot-ground

  # S5 batch export (no viewer); writes walk_turnN.npy + walk_turnN_cmd.npy
  python ./UEGMR/Retargeting/ue_world_skeleton_retarget.py `
      --jsonl ./data/walk_turn_frames2.jsonl --meta ./data/walk_turn_meta2.json `
      --config ./tools/gasp_bvh_alignment_g1_height.json `
      --output-qpos data/walk_turn2.npy --no-visualize --per-foot-ground

  # Playback speed (viewer): default = recording's native sample rate (1.0x).
  #   --speed 2.0  (2x) | --speed 0.25 (4x slow-mo) | --fps 120 (force FPS)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Force UTF-8 stdout/stderr so non-ASCII prints don't crash on Windows GBK consoles.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Reuse the verified GASP loader (puts data in BVH-native coords).
from vis_skeleton_compare import load_gasp_data  # noqa: E402

# Repo layout (release): HERE = <repo>/DataLib/retarget
#   <repo>/data/         sample + user recordings
#   <repo>/DataLib/gmr/  vendored general_motion_retargeting package
REPO_ROOT = HERE.parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_JSONL = str(DATA_DIR / "sample" / "ground" / "WalkTurnCrouch_C_10_Ovu0mkr5_frames.jsonl")
DEFAULT_META  = str(DATA_DIR / "sample" / "ground" / "WalkTurnCrouch_C_10_Ovu0mkr5_meta.json")
# Default config pairs with the default --src-human (bvh_ue5_g1scale): its
# actual_human_height_m (~1.32) matches the IK config's human_height_assumption
# (1.32) so the GMR height ratio is ~1.0 (no double-scaling). For the 1.75 m
# 'bvh_ue5_native' src, pass --config gasp_bvh_alignment.json instead.
DEFAULT_CONFIG = str(HERE / "gasp_bvh_alignment_g1_height.json")


# ── Config loader ────────────────────────────────────────────────────────────

def load_alignment_config(path: str):
    """Load the JSON produced by gasp_bvh_calibrate.py.

    Returns:
      R_global : (3,3) rotation matrix
      t_global : (3,)  translation
      delta    : dict[bone_name, (3,3) rotation matrix]
      raw_cfg  : original dict (for metadata access)
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    g = cfg["global"]
    R_global = R.from_quat(g["rotation_quat_wxyz"], scalar_first=True).as_matrix()
    t_global = np.array(g["translation_xyz"], dtype=np.float64)
    delta = {
        name: R.from_quat(q, scalar_first=True).as_matrix()
        for name, q in cfg["per_bone_delta_quat_wxyz"].items()
    }
    return R_global, t_global, delta, cfg


def load_meta(meta_path: str) -> tuple[list[str], list[int]]:
    meta = json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))
    return meta["bone_names"], meta["parent_indices"]


def load_frames(jsonl_path: str) -> list[dict]:
    out: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def skeleton_world_direct(frame: dict) -> np.ndarray:
    """S1: read bone world positions (wp) directly from JSONL frame.

    Returns (B, 3) in UE world centimetres.
    """
    return np.asarray([j["wp"] for j in frame["joints"]], dtype=np.float64)


def ue_to_mujoco(p_cm: np.ndarray) -> np.ndarray:
    """S2/S3: UE world cm -> MuJoCo world m with X<->Y swap."""
    out = np.empty_like(p_cm)
    out[..., 0] = p_cm[..., 1]
    out[..., 1] = p_cm[..., 0]
    out[..., 2] = p_cm[..., 2]
    return out * 0.01


def apply_kabsch(p: np.ndarray, R_global: np.ndarray, t_global: np.ndarray) -> np.ndarray:
    """S3: apply global Kabsch transform."""
    return (R_global @ p.T).T + t_global


# ── GASP command + root loader ───────────────────────────────────────────────

# Structured dtype for the per-frame command record saved to *_cmd.npy.
#
# All fields are in MuJoCo display frame conventions (same system as the
# retargeted qpos): right-hand, Z-up, metres / degrees, with axes aligned
# by the Kabsch R_global from gasp_bvh_alignment.json.
#
# Coordinate conventions applied on top of the raw UE5 JSONL values:
#   look_fwd_mj_xy : UE5 ControlRotation → UE→BVH axis-swap → R_global (computed
#                    by compute_world_orientation_arrows, filled by save_command_array)
#   move_input     : raw [right, forward] WASD reordered to [forward, right]
#   des_vel        : REAL measured body-frame velocity (m/s).
#                    Source: root.lv = Actor->GetVelocity() (UE5 world cm/s, LH).
#                    Pipeline: lv_world_ue → ActorQuat.Inverse() (UE5 body cm/s LH)
#                              → cm/s→m/s, lateral flip (right→left for MuJoCo RH).
#                    NOT normalised — magnitude reflects actual ground speed
#                    (typically 0–7.5 m/s for walk/run/sprint), not WASD intent.
#                    For the OLD normalised "desired" intent see cmd.move_input
#                    or recompute from cmd.desVB in the source JSONL.
#   des_yaw        : UE5 positive = turn-right (CW, LH) → negated for MuJoCo positive
#                    = turn-left (CCW, RH)
COMMAND_DTYPE = np.dtype([
    # Camera / user forward direction in MuJoCo display frame (XY, unit vector).
    # Filled in by save_command_array after compute_world_orientation_arrows.
    ("look_fwd_mj_xy",  "f4", 2),
    # Movement intent in body frame, MuJoCo convention [forward, right], [-1, 1].
    # forward: W=+1, S=−1; right: D=+1, A=−1.
    ("move_input",      "f4", 2),
    # Button states.
    ("jump",            "?"),
    ("crouch",          "?"),
    ("sprint",          "?"),
    ("walk",            "?"),
    # REAL body-frame velocity (m/s) — actual measured Actor velocity rotated
    # into the actor body frame, NOT the normalised WASD-derived intent:
    #   des_vel[0] = forward  (m/s, positive = forward)
    #   des_vel[1] = leftward (m/s, positive = left; UE5 LH right → MuJoCo RH left)
    ("des_vel",         "f4", 2),
    # Desired yaw rate (deg/s) in MuJoCo convention:
    #   positive = CCW = turn left (= −UE5 DesiredYawRate)
    ("des_yaw",         "f4"),
])


def load_gasp_command_data(jsonl_path: str) -> np.ndarray:
    """Read every JSONL frame and return a (N,) structured array (COMMAND_DTYPE).

    All fields are converted to MuJoCo conventions on the fly:
      move_input : JSONL [right, forward] → stored as [forward, right]
      des_vel    : REAL body-frame velocity in m/s (NOT the normalised cmd.desVB).
                   Source: root.lv (UE5 world cm/s, LH) rotated into the actor
                   body frame via root.q^{-1}, then cm/s→m/s with the right→left
                   axis flip for MuJoCo RH. Magnitude reflects actual ground
                   speed, not WASD intent — see cmd.move_input for the latter.
      des_yaw    : JSONL desYR (positive=turn-right in UE5) → negated (positive=turn-left)

    `look_fwd_mj_xy` is left zero here and filled by `save_command_array`
    after `compute_world_orientation_arrows` applies the Kabsch R_global.
    """
    rows: list = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cmd  = row.get("cmd",  {}) or {}
            root = row.get("root", {}) or {}
            rec = np.zeros((), dtype=COMMAND_DTYPE)
            # look_fwd_mj_xy: filled later — leave as zeros
            move_raw = cmd.get("move", [0.0, 0.0])          # JSONL: [right, forward]
            rec["move_input"]  = [move_raw[1], move_raw[0]] # reorder → [forward, right]
            rec["jump"]        = bool(cmd.get("jump",   False))
            rec["crouch"]      = bool(cmd.get("crouch", False))
            rec["sprint"]      = bool(cmd.get("sprint", False))
            rec["walk"]        = bool(cmd.get("walk",   False))

            # des_vel: REAL measured body-frame velocity (m/s).
            # 1) lv_world_ue : UE5 world cm/s LH (Actor->GetVelocity()).
            # 2) Rotate into actor body via q^{-1} (matches UE5's own desVB recipe
            #    in MotionCaptureComponent.cpp: ActorQuat.Inverse().Rotate(...)).
            # 3) Take XY in body, cm/s → m/s, flip Y (UE5 LH right → MuJoCo RH left).
            lv_world_ue = np.asarray(
                root.get("lv", [0.0, 0.0, 0.0]), dtype=np.float64,
            )
            q_xyzw = np.asarray(
                root.get("q", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64,
            )
            try:
                R_actor = R.from_quat(q_xyzw).as_matrix()    # body → world
                lv_body_ue = R_actor.T @ lv_world_ue          # world → body (cm/s, LH)
            except Exception:
                lv_body_ue = np.zeros(3, dtype=np.float64)
            rec["des_vel"]     = [
                lv_body_ue[0] * 0.01,        # forward  (m/s)
                -lv_body_ue[1] * 0.01,       # leftward (m/s) ← right→left flip
            ]

            rec["des_yaw"]     = -float(cmd.get("desYR", 0.0))  # flip: CW→CCW
            rows.append(rec)
    if not rows:
        return np.zeros((0,), dtype=COMMAND_DTYPE)
    return np.stack(rows, axis=0)


# UE5 "GASP World" → BVH coord-change matrix. Identical to the one used in
# vis_skeleton_compare.py for per-bone rotation conversion (a (X,Y,Z)→(Y,X,Z)
# swap; symmetric, so M.T == M). Lifted here so this module is self-contained.
_M_UE_TO_BVH_MAT = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 1]], dtype=np.float64)


def compute_world_orientation_arrows(
    cmd_arr: np.ndarray,
    R_global: np.ndarray,
) -> np.ndarray:
    """Project per-frame camera (Look) forward vectors into the MuJoCo display frame.

    Coordinate chain (same as the per-bone rotation chain in
    vis_skeleton_compare.load_gasp_data):

        forward_ue  = [cos(lookY), sin(lookY), 0]  in UE5 world (from cmd.lookY)
        forward_bvh = _M_UE_TO_BVH @ forward_ue    (axis swap X↔Y)
        forward_mj  = R_global @ forward_bvh        (Kabsch alignment)

    Returns:
        look_fwd_xy (N, 2) ground-projected unit vectors in MuJoCo display frame.

    Not re-normalized after the Z drop; with near-vertical R_global (typical)
    the XY components are already ≈unit length.
    """
    n = cmd_arr.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Camera forward in UE5 world = [cos(yaw), sin(yaw), 0].
    # look_yaw_deg is stored in the raw UE5 degree convention (not yet in the
    # COMMAND_DTYPE — it was removed from the simplified dtype but we still need
    # it here). Reconstruct from move_input is not possible, so we re-read from
    # the raw JSONL via the actor_quat workaround: since actor_yaw == camera_yaw
    # in GASP, we can derive look_yaw from the actor quaternion stored in the
    # structured array... but actually look_yaw is NOT in the new COMMAND_DTYPE.
    # We must therefore accept cmd_arr with a temporary "look_yaw_deg" field, or
    # pass the yaw array explicitly. For simplicity, read look_yaw from the same
    # JSONL data that was already read into cmd_arr. Since cmd_arr no longer
    # stores look_yaw_deg, we compute the camera forward from the actor_quat that
    # IS available inside the JSONL (via a separate pass), OR we pass the yaw
    # directly. The cleanest solution: keep accepting the original cmd_arr before
    # the dtype was stripped, or accept a separate yaw array.
    #
    # Implementation: this function is called with the full raw data produced by
    # load_gasp_command_data. Since look_yaw_deg is not in COMMAND_DTYPE, the
    # caller must pass the look_yaw array explicitly. See signature below.
    raise NotImplementedError("Use compute_look_fwd_xy(look_yaw_deg_arr, R_global) instead.")


def compute_look_fwd_xy(
    look_yaw_deg: np.ndarray,
    R_global: np.ndarray,
) -> np.ndarray:
    """Project per-frame camera Yaw into a ground-projected unit forward vector
    in the MuJoCo display frame.

    Args:
        look_yaw_deg : (N,) float array of UE5 ControlRotation Yaw in degrees.
                       Read from JSONL cmd.lookY before building COMMAND_DTYPE.
        R_global     : (3,3) Kabsch rotation from gasp_bvh_alignment.json.

    Returns:
        look_fwd_xy  : (N, 2) float32, ground-projected unit vectors in the
                       same coordinate system as the retargeted qpos.
    """
    n = len(look_yaw_deg)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    yaw_rad = np.deg2rad(np.asarray(look_yaw_deg, dtype=np.float64))
    look_fwd_ue = np.stack(
        [np.cos(yaw_rad), np.sin(yaw_rad), np.zeros(n)], axis=1
    )                                            # (N, 3) in UE5 world
    M = _M_UE_TO_BVH_MAT
    look_fwd_bvh = look_fwd_ue @ M.T            # axis swap X↔Y
    look_fwd_mj  = look_fwd_bvh @ R_global.T   # Kabsch alignment
    return look_fwd_mj[:, :2].astype(np.float32)


def save_command_array(
    out_path: pathlib.Path,
    cmd_arr: np.ndarray,
    look_fwd_xy: np.ndarray,
) -> None:
    """Serialize the simplified command array to .npy.

    Embeds the derived `look_fwd_mj_xy` column (computed by
    `compute_look_fwd_xy`) so the file is self-contained.

    Loading example:
        data = np.load("walk_turn_cmd.npy")
        print(data.dtype.names)
        look = data["look_fwd_mj_xy"]   # (T, 2)  camera forward, MuJoCo frame
        move = data["move_input"]        # (T, 2)  [forward, right], body frame
        vel  = data["des_vel"]           # (T, 2)  [forward, leftward] m/s,
                                         #         REAL body-frame velocity
                                         #         (root.lv rotated into actor body)
        yaw  = data["des_yaw"]           # (T,)    deg/s, positive=turn-left (CCW)
    """
    n = min(cmd_arr.shape[0], look_fwd_xy.shape[0])
    cmd_arr = cmd_arr[:n].copy()
    cmd_arr["look_fwd_mj_xy"] = look_fwd_xy[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), cmd_arr)
    print(f"  Saved cmd → {out_path}  (dtype fields: {len(COMMAND_DTYPE.names)}, frames: {n})")


def save_scaled_skeleton(
    out_path: pathlib.Path,
    positions: np.ndarray,
    names: list[str],
) -> None:
    """Serialize the scaled pre-IK skeleton (the GMR IK *input*) to .npz.

    This is export "part 1": the scaled-human joint positions that the IK solver
    tries to match, in the SAME MuJoCo display frame as the retargeted qpos
    (Kabsch-aligned, grounded). Stored as .npz (not .npy) because it bundles the
    per-joint name list with the positions so the file is self-describing.

    Archive contents:
        positions : (F, N, 3) float32  scaled IK-input joint positions
        names     : (N,)      str      joint names, index-aligned with positions

    Loading example:
        d   = np.load("walk_turn_frames_scaled.npz", allow_pickle=False)
        pos = d["positions"]            # (F, N, 3)
        nm  = [str(x) for x in d["names"]]
        idx = nm.index("LeftFootMod")   # look up a joint by name
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        positions=np.asarray(positions, dtype=np.float32),
        names=np.asarray(names, dtype="U"),
    )
    print(f"  Saved scaled pre-IK skeleton → {out_path}  "
          f"(positions: {positions.shape}, joints: {len(names)})")


# ── Foot grounding (verify_pipeline S5: pre-IK + root-Z + per-foot) ───────────

def scaled_human_skeleton(gmr_frames, human_height, root_xy,
                          src_human="bvh_ue5_g1scale"):
    """Extract the GMR "scale human data" — the IK *input* skeleton.

    GMR's pipeline first runs update_targets(), which scales the human joints
    about the root + applies the IK offsets, and stores the result in
    ``retargeter.scaled_human_data`` (exactly what the IK solver tries to
    match). We run only update_targets() per frame (no IK solve) and read those
    scaled positions back out.

    The GMR frames had their root XY recentered to the origin (see
    build_gmr_frames); we add the per-frame root XY back so the scaled skeleton
    lands in the same world frame as the Kabsch skeleton / terrain / G1.

    The scaled skeleton has NO toe (ball) joint — GMR's scale table only carries
    the ankle (LeftFootMod/RightFootMod). For toe-to-toe grounding we also need
    the scaled toe, so we scale the original ball_* about the root with the SAME
    leg/foot factor GMR uses for the ankle, reproducing GMR's per-joint scaling:
        scaled_p = (p - root) * s_joint + s_root * root
    toe is (F, 2, 3) = scaled [ball_l, ball_r] (NaN-filled if a side is absent).

    Returns (skel (F, N, 3), names list, retargeter, toe (F, 2, 3)).
    """
    _ensure_gmr_on_path()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
    retargeter = GMR(src_human=src_human, tgt_robot="unitree_g1",
                     actual_human_height=human_height)
    root_name = retargeter.human_root_name
    scale_tbl = retargeter.human_scale_table          # already × height ratio
    s_root = float(scale_tbl.get(root_name, 1.0))
    s_foot = {"ball_l": float(scale_tbl.get("LeftFootMod", s_root)),
              "ball_r": float(scale_tbl.get("RightFootMod", s_root))}

    names = None
    per_frame = []
    toe_frames = []
    for frame in gmr_frames:
        retargeter.update_targets(frame)
        shd = retargeter.scaled_human_data
        if names is None:
            names = list(shd.keys())
        per_frame.append(np.array([shd[n][0] for n in names], dtype=np.float64))

        # Reproduce GMR's scaling for the toe (ball_*), not in the scale table.
        root_pos = np.asarray(frame[root_name][0], dtype=np.float64)
        toe_lr = np.full((2, 3), np.nan, dtype=np.float64)
        for k, bn in enumerate(("ball_l", "ball_r")):
            if bn in frame:
                b = np.asarray(frame[bn][0], dtype=np.float64)
                toe_lr[k] = (b - root_pos) * s_foot[bn] + s_root * root_pos
        toe_frames.append(toe_lr)

    skel = np.stack(per_frame, axis=0)
    toe = np.stack(toe_frames, axis=0)
    for arr in (skel, toe):
        arr[:, :, 0] += root_xy[:, None, 0]
        arr[:, :, 1] += root_xy[:, None, 1]
    return skel, names, retargeter, toe


def support_soft_ground_dz(blue_l_z: np.ndarray, blue_r_z: np.ndarray,
                           red_l_z: np.ndarray, red_r_z: np.ndarray,
                           k: float) -> np.ndarray:
    """Method-3 grounding dz from per-foot pairing + soft-min support blend."""
    res_l = blue_l_z - red_l_z
    res_r = blue_r_z - red_r_z
    zmin = np.minimum(red_l_z, red_r_z)
    e_l = np.exp(-k * (red_l_z - zmin))
    e_r = np.exp(-k * (red_r_z - zmin))
    w_l = e_l / (e_l + e_r)
    w_r = 1.0 - w_l
    return w_l * res_l + w_r * res_r


def g1_fk_foot_z(model, qpos_arr, front_only=True):
    """Per-frame min foot contact sphere bottom Z via FK."""
    import mujoco as mj  # type: ignore
    foot_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
    body_ids = set()
    for bname in foot_bodies:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            body_ids.add(bid)
    geom_ids = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        if int(model.geom_type[gid]) != mj.mjtGeom.mjGEOM_SPHERE:
            continue
        if front_only and model.geom_pos[gid, 0] <= 0:
            continue
        geom_ids.append(gid)

    data = mj.MjData(model)
    n = qpos_arr.shape[0]
    foot_z = np.zeros(n, dtype=np.float64)
    for fi in range(n):
        data.qpos[:] = qpos_arr[fi]
        mj.mj_forward(model, data)
        zmin = float("inf")
        for gid in geom_ids:
            z = float(data.geom_xpos[gid, 2]) - float(model.geom_size[gid, 0])
            if z < zmin:
                zmin = z
        foot_z[fi] = zmin
    return foot_z


def g1_toe_standing_offset(model,
                           toe_bodies=("left_toe_link", "right_toe_link")):
    """offset_robot (m): height of the G1 toe_link above the foot's lowest
    contact-sphere bottom in the neutral (flat-foot) standing pose.

    Derived purely from the URDF/model geometry: with all joints at 0 the foot
    is flat, the contact spheres rest on the ground, and the toe_link sits a
    fixed distance above. This is the robot analogue of the human 'toe joint to
    ground' offset, so the two can be matched ground-to-ground.
    """
    import mujoco as mj  # type: ignore
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    if model.nq >= 7:           # free joint quat → identity
        data.qpos[3] = 1.0
    mj.mj_forward(model, data)

    toe_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in toe_bodies]
    toe_ids = [t for t in toe_ids if t >= 0]
    toe_z = min(float(data.xpos[t, 2]) for t in toe_ids)

    foot_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
    bids = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in foot_bodies}
    bids.discard(-1)
    contact_z = float("inf")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        if int(model.geom_type[gid]) != mj.mjtGeom.mjGEOM_SPHERE:
            continue
        z = float(data.geom_xpos[gid, 2]) - float(model.geom_size[gid, 0])
        contact_z = min(contact_z, z)
    return toe_z - contact_z


def g1_toe_fk_z(model, qpos_arr, toe_bodies=("left_toe_link", "right_toe_link")):
    """Per-frame min toe_link world Z over both feet (live FK)."""
    import mujoco as mj  # type: ignore
    toe_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in toe_bodies]
    toe_ids = [t for t in toe_ids if t >= 0]
    data = mj.MjData(model)
    n = qpos_arr.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for fi in range(n):
        data.qpos[:] = qpos_arr[fi]
        mj.mj_forward(model, data)
        out[fi] = min(float(data.xpos[t, 2]) for t in toe_ids)
    return out


def per_foot_ground(model, qpos_arr, ball_l_z, ball_r_z,
                    offset_human, offset_robot,
                    iters=40, damping=1e-2, tol=5e-4):
    """Per-foot grounding: bend each leg (sagittal hip_pitch/knee/ankle_pitch)
    so that each toe_link lands on its OWN target ground, independently.

    Target toe_link world Z (per side, per frame):
        toe_z* = ball_z_side - offset_human + offset_robot
    so that  toe_z* - offset_robot == ball_z_side - offset_human  (robot
    ground == human ground for THAT foot). XY is held at its current value;
    only the sagittal leg joints move, so the root and upper body are intact.

    Unlike global ball-align (one root Z shift, lower foot only), this lets the
    two feet sit on different stair levels at the same time.
    """
    import mujoco as mj  # type: ignore
    legs = {
        "left":  ("left_toe_link",
                  ["left_hip_pitch_joint", "left_knee_joint",
                   "left_ankle_pitch_joint"]),
        "right": ("right_toe_link",
                  ["right_hip_pitch_joint", "right_knee_joint",
                   "right_ankle_pitch_joint"]),
    }
    info = {}
    for side, (toe, jns) in legs.items():
        tid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, toe)
        jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j) for j in jns]
        qadr = [int(model.jnt_qposadr[j]) for j in jids]
        dadr = [int(model.jnt_dofadr[j]) for j in jids]
        rng = [tuple(model.jnt_range[j]) for j in jids]
        info[side] = (tid, qadr, dadr, rng)

    tgt_z = {"left":  ball_l_z - offset_human + offset_robot,
             "right": ball_r_z - offset_human + offset_robot}

    data = mj.MjData(model)
    n = qpos_arr.shape[0]
    eye3 = np.eye(3)
    for fi in range(n):
        data.qpos[:] = qpos_arr[fi]
        for side, (tid, qadr, dadr, rng) in info.items():
            mj.mj_forward(model, data)
            target = data.xpos[tid].copy()
            target[2] = tgt_z[side][fi]
            for _ in range(iters):
                mj.mj_forward(model, data)
                err = target - data.xpos[tid]
                if np.linalg.norm(err) < tol:
                    break
                jacp = np.zeros((3, model.nv))
                mj.mj_jac(model, data, jacp, None, data.xpos[tid], tid)
                J = jacp[:, dadr]
                dq = J.T @ np.linalg.solve(J @ J.T + damping * eye3, err)
                for k, qa in enumerate(qadr):
                    v = data.qpos[qa] + dq[k]
                    lo, hi = rng[k]
                    if lo < hi:
                        v = min(max(v, lo), hi)
                    data.qpos[qa] = v
        qpos_arr[fi] = data.qpos
    return qpos_arr


def _foot_geom_ids(model, front_only=True):
    """Return geom ids of G1 foot contact spheres."""
    import mujoco as mj  # type: ignore
    foot_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
    bids = set()
    for bn in foot_bodies:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bn)
        if bid >= 0:
            bids.add(bid)
    out = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        if int(model.geom_type[gid]) != mj.mjtGeom.mjGEOM_SPHERE:
            continue
        if front_only and model.geom_pos[gid, 0] <= 0:
            continue
        out.append(gid)
    return out


def apply_s5_grounding(
    qpos_arr: np.ndarray,
    g1_model,
    skel_kabsch: np.ndarray,
    ball_idx: dict,
    n_frames: int,
    extra_drop: float,
    no_ball_align: bool,
    per_foot: bool,
):
    """verify_pipeline S5 main + per-foot grounding on the retargeted G1.

    (1) Root-Z grounding (toe-to-toe): a single per-frame root Z shift so the
        robot's (lower) toe-to-ground matches the human's (lower) toe-to-ground.
    (2) Per-foot grounding: independently bend each leg so each toe_link lands
        on its own target (handles stairs / feet at different heights).

    Returns (qpos_arr, g1_foot_geoms).
    """
    import mujoco as mj  # type: ignore

    offset_human = extra_drop
    offset_robot = g1_toe_standing_offset(g1_model)

    have_balls = "ball_l" in ball_idx and "ball_r" in ball_idx
    ball_l_kabsch = ball_r_kabsch = ue_ball_min_z = None
    if have_balls:
        bl_s, br_s = ball_idx["ball_l"], ball_idx["ball_r"]
        ball_l_kabsch = skel_kabsch[:, bl_s, :]
        ball_r_kabsch = skel_kabsch[:, br_s, :]
        ue_ball_min_z = np.minimum(ball_l_kabsch[:, 2], ball_r_kabsch[:, 2])

    if no_ball_align:
        print("[Root-Z grounding] DISABLED (--no-ball-align).")
    elif have_balls:
        g1_toe_z_pre = g1_toe_fk_z(g1_model, qpos_arr)
        dist_human = ue_ball_min_z - offset_human          # human ground
        dist_robot = g1_toe_z_pre - offset_robot           # robot ground
        dz = dist_human - dist_robot
        qpos_arr[:, 2] += dz
        print(f"[Root-Z grounding] offset_human={offset_human:.3f} m  "
              f"offset_robot(URDF)={offset_robot:.4f} m")
        print(f"  Δz: min={dz.min():.4f}  med={np.median(dz):.4f}  "
              f"max={dz.max():.4f}  std={dz.std():.4f}")

    if per_foot and have_balls:
        qpos_arr = per_foot_ground(
            g1_model, qpos_arr, ball_l_kabsch[:, 2], ball_r_kabsch[:, 2],
            offset_human, offset_robot)
        tL = mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, "left_toe_link")
        tR = mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, "right_toe_link")
        _d = mj.MjData(g1_model)
        rL = np.zeros(n_frames); rR = np.zeros(n_frames)
        for _i in range(n_frames):
            _d.qpos[:] = qpos_arr[_i]; mj.mj_forward(g1_model, _d)
            rL[_i] = (_d.xpos[tL, 2] - offset_robot) - (ball_l_kabsch[_i, 2] - offset_human)
            rR[_i] = (_d.xpos[tR, 2] - offset_robot) - (ball_r_kabsch[_i, 2] - offset_human)
        print(f"[Per-foot grounding] residual (robot ground - human ground):")
        print(f"  left : mean|.|={np.mean(np.abs(rL)):.4f}  max|.|={np.max(np.abs(rL)):.4f} m")
        print(f"  right: mean|.|={np.mean(np.abs(rR)):.4f}  max|.|={np.max(np.abs(rR)):.4f} m")

    g1_foot_geoms = _foot_geom_ids(g1_model)
    g1_foot_z_final = g1_fk_foot_z(g1_model, qpos_arr)
    print(f"  G1 foot Z (final): min={g1_foot_z_final.min():.4f}  "
          f"med={np.median(g1_foot_z_final):.4f}  "
          f"max={g1_foot_z_final.max():.4f}")
    print(f"  G1 ngeom={g1_model.ngeom}, foot contact geoms={len(g1_foot_geoms)}")
    return qpos_arr, g1_foot_geoms


# ── Apply alignment + build GMR input frames ─────────────────────────────────

def build_gmr_frames(
    gasp_pos_mj: np.ndarray,
    gasp_rot_all: list[list[np.ndarray]],
    gasp_names: list[str],
    R_global: np.ndarray,
    t_global: np.ndarray,
    delta: dict[str, np.ndarray],
    root_name: str = "pelvis",
    recenter_root_xy: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """For each frame: apply (R_global, t_global) and per-bone Δ, return GMR input dict.

    GMR expects {bone_name: [pos_xyz_m, quat_wxyz]} per frame, with the
    synthesized LeftFootMod / RightFootMod entries (foot pos + ball quat).
    """
    identity3 = np.eye(3)
    n_frames = gasp_pos_mj.shape[0]
    root_xy = np.zeros((n_frames, 2), dtype=np.float64)
    frames: list[dict] = []
    for fi in range(n_frames):
        result: dict = {}
        for gi, name in enumerate(gasp_names):
            # Step 2a: global rigid transform
            pos_aligned = R_global @ gasp_pos_mj[fi, gi] + t_global
            rot_aligned = R_global @ gasp_rot_all[fi][gi]
            # Step 2b: per-bone delta (residual local-frame mismatch)
            D = delta.get(name, identity3)
            R_eq = rot_aligned @ D
            quat = R.from_matrix(R_eq).as_quat(scalar_first=True)  # wxyz
            result[name] = [pos_aligned, quat]

        # Match verify_pipeline: recenter root XY before GMR so scale_human_data
        # does not pull the clip toward world origin; add this XY back to qpos
        # after retargeting.
        if recenter_root_xy and root_name in result:
            off = result[root_name][0][:2].copy()
            root_xy[fi] = off
            for name in result:
                result[name][0][:2] -= off

        # Synthesized IK targets used by bvh_ue5_native_to_g1 config:
        # LeftFootMod = foot_* position (ankle) + ball_* orientation (drives
        # foot pitch). RightFootMod likewise.
        if "foot_l" in result and "ball_l" in result:
            result["LeftFootMod"]  = [result["foot_l"][0].copy(), result["ball_l"][1].copy()]
        if "foot_r" in result and "ball_r" in result:
            result["RightFootMod"] = [result["foot_r"][0].copy(), result["ball_r"][1].copy()]
        frames.append(result)
    return frames, root_xy


def estimate_human_height(gmr_frames: list[dict]) -> float:
    """Same heuristic as step4_retarget_bvh_to_g1.py: head_z - min(foot_z) + 0.09 m."""
    if not gmr_frames:
        return 1.7
    head = gmr_frames[0].get("head", [np.array([0.0, 0.0, 1.7])])[0]
    fl   = gmr_frames[0].get("foot_l", [np.array([0.0, 0.0, 0.0])])[0]
    fr   = gmr_frames[0].get("foot_r", [np.array([0.0, 0.0, 0.0])])[0]
    floor_z = min(fl[2], fr[2])
    return float(head[2] - floor_z + 0.09)


# ── GMR retarget driver ──────────────────────────────────────────────────────

def _ensure_gmr_on_path():
    """Make sure general_motion_retargeting is importable."""
    candidates = [
        HERE.parent / "gmr",                         # release: DataLib/gmr
        HERE.parent / "UEGMR" / "GMR",                # Scripts/UEGMR/GMR
        HERE.parent.parent / "UEGMR" / "GMR",         # ../UEGMR/GMR (legacy layout)
    ]
    for root in candidates:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def retarget_to_g1(
    gmr_frames: list[dict],
    actual_human_height: float,
    warmup_iters: int = 20,
    src_human: str = "bvh_ue5_g1scale",
) -> tuple[np.ndarray, object]:
    """Run GMR (src_human → unitree_g1) on every frame, return qpos array.

    A warm-up loop on frame 0 is essential — GMR's IK is iterative and a single
    call from G1's rest pose does NOT converge, so the first stored qpos would
    have a wrong pelvis Z (feet through the floor) and an unconverged pose.
    Same trick used by vis_skeleton_compare.py and step4_retarget_bvh_to_g1.py.
    """
    _ensure_gmr_on_path()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore

    print(f"  Init GMR (src={src_human}, tgt=unitree_g1, height={actual_human_height:.3f}m)")
    retargeter = GMR(
        src_human=src_human,
        tgt_robot="unitree_g1",
        actual_human_height=actual_human_height,
    )
    ik_bones = sorted(retargeter.human_scale_table.keys())
    missing = [b for b in ik_bones if b not in gmr_frames[0]]
    if missing:
        print(f"  [WARN] Missing bones in retarget input: {missing}")

    if warmup_iters > 0 and gmr_frames:
        print(f"  Warming up IK on frame 0 ({warmup_iters} iters) ...")
        for _ in range(warmup_iters):
            retargeter.retarget(gmr_frames[0])

    qpos_list: list[np.ndarray] = []
    for fi, frame in enumerate(gmr_frames):
        qpos_list.append(retargeter.retarget(frame).copy())
    # Return retargeter so downstream passes (e.g. ball-Z grounding) can reuse
    # its model for FK.
    return np.stack(qpos_list, axis=0), retargeter


# ── Visualization ────────────────────────────────────────────────────────────
    # <headlight diffuse="0.32 0.32 0.32"
    #            ambient="0.20 0.20 0.20"
    #            specular="0.06 0.06 0.06"/>
    # <light name="sun_n" directional="true" diffuse=".10 .10 .10" specular="0 0 0"
    #        pos=" 0  4 4" dir=" 0 -1 -1"/>
    # <light name="sun_s" directional="true" diffuse=".10 .10 .10" specular="0 0 0"
    #        pos=" 0 -4 4" dir=" 0  1 -1"/>
    # <light name="sun_e" directional="true" diffuse=".08 .08 .08" specular="0 0 0"
    #        pos=" 4  0 4" dir="-1  0 -1"/>
    # <light name="sun_w" directional="true" diffuse=".08 .08 .08" specular="0 0 0"
    #        pos="-4  0 4" dir=" 1  0 -1"/>
    # <light name="sun_top" directional="true" diffuse=".08 .08 .08" specular="0 0 0"
    #        pos=" 0  0 6" dir=" 0  0 -1"/>
# ── Terrain (scene) loader ───────────────────────────────────────────────────
#
# Loads a terrain_<hash>.json exported by the UE MotionCaptureComponent and
# applies the SAME transform the motion goes through, so terrain and robot land
# in one consistent world:
#
#     p_mj    = [p_ue.y, p_ue.x, p_ue.z] * 0.01     (UE cm, LH  ->  BVH/MuJoCo m)
#     p_final = R_global @ p_mj + t_global          (Kabsch global alignment)
#
# The X<->Y swap is a reflection (det = -1), so triangle winding is reversed
# per-instance whenever the net linear map is left-handed (keeps normals out).

_M_UE_TO_BVH_LOCAL = np.array([[0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)


def load_terrain_world(
    terrain_path: str,
    R_global: np.ndarray,
    t_global: np.ndarray,
    ground_z0_cm: float = 0.0,
    skip_engine: bool = True,
    data_scale: float = 1.0,
):
    """Return (vertices (M,3) float32, faces (K,3) int32) in final MuJoCo world
    coords, or None if the file is missing / empty.

    skip_engine: drop instances whose mesh comes from /Engine/ (camera models,
    editor gizmos, etc.) so they don't pollute the scene.
    """
    p = pathlib.Path(terrain_path)
    if not p.is_file():
        print(f"  [Terrain] not found: {p}")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    meshes = data.get("meshes", [])
    instances = data.get("instances", [])
    if not meshes or not instances:
        print(f"  [Terrain] empty terrain in {p.name}")
        return None

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    voff = 0
    skipped_engine = 0
    for inst in instances:
        mi = int(inst.get("mesh", -1))
        if mi < 0 or mi >= len(meshes):
            continue
        mesh = meshes[mi]
        if skip_engine and str(mesh.get("asset_path", "")).startswith("/Engine/"):
            skipped_engine += 1
            continue
        V = np.asarray(mesh.get("vertices", []), dtype=np.float64).reshape(-1, 3)
        idx = np.asarray(mesh.get("indices", []), dtype=np.int64)
        if V.size == 0 or idx.size < 3:
            continue
        F = idx.reshape(-1, 3)
        Nloc = np.asarray(mesh.get("normals", []), dtype=np.float64)
        Nloc = Nloc.reshape(-1, 3) if Nloc.size == V.size else None

        loc = np.asarray(inst["location"], dtype=np.float64)
        quat_xyzw = np.asarray(inst["rotation_quat_xyzw"], dtype=np.float64)
        scl = np.asarray(inst["scale"], dtype=np.float64)
        Rinst = R.from_quat(quat_xyzw).as_matrix()

        # local -> UE world (cm)
        Vw = (Rinst @ (V * scl).T).T + loc
        Vw[:, 2] -= ground_z0_cm
        # Apply the same uniform data scale the motion got (about the UE world
        # origin) so terrain stays consistent with the scaled motion.
        if abs(data_scale - 1.0) > 1e-9:
            Vw = Vw * data_scale
        # UE world cm -> BVH/MuJoCo m  (X<->Y swap + 0.01 scale)
        Vmj = Vw[:, [1, 0, 2]] * 0.01
        # global Kabsch alignment (same R_global/t_global as the motion)
        Vf = (R_global @ Vmj.T).T + t_global

        # Orient every triangle so its winding's right-hand normal agrees with
        # the (transformed) UE surface normal. MuJoCo derives face normals from
        # winding and back-face culls, so consistent outward winding removes the
        # "see-through" faces. Robust to the X<->Y reflection AND per-instance
        # mirrored (negative) scales.
        lin = R_global @ (0.01 * _M_UE_TO_BVH_LOCAL) @ Rinst @ np.diag(scl)
        tri = Vf[F]
        gnorm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if Nloc is not None:
            Nf = (lin @ Nloc.T).T                      # transformed surface normals
            ref = Nf[F].sum(axis=1)                    # per-triangle reference
            flip = np.einsum("ij,ij->i", gnorm, ref) < 0.0
        else:
            flip = np.full(F.shape[0], np.linalg.det(lin) < 0.0)
        if flip.any():
            F = F.copy()
            F[flip] = F[flip][:, ::-1]

        all_v.append(Vf.astype(np.float32))
        all_f.append((F + voff).astype(np.int32))
        voff += V.shape[0]

    if skipped_engine:
        print(f"  [Terrain] skipped {skipped_engine} /Engine/ helper instances "
              f"(camera models, gizmos, etc.)")
    if not all_v:
        return None
    verts = np.concatenate(all_v, axis=0)
    faces = np.concatenate(all_f, axis=0)
    print(f"  [Terrain] {p.name}: {len(meshes)} meshes, {len(instances)} instances "
          f"-> {verts.shape[0]} verts, {faces.shape[0]} tris")
    print(f"  [Terrain] final Z range: "
          f"[{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}] m")
    return verts, faces


def _resolve_terrain_path(args):
    """Resolve the terrain JSON to load: explicit --terrain wins, else the
    recording meta's `terrain_ref`, searched in --terrain-dir / a `terrain`
    folder next to --meta / next to --meta itself."""
    if args.terrain:
        return args.terrain
    ref = None
    try:
        with open(args.meta, "r") as f:
            ref = json.load(f).get("terrain_ref") or None
    except Exception:
        ref = None
    if not ref:
        print("  [Terrain] meta has no `terrain_ref`; pass --terrain to override.")
        return None
    meta_dir = pathlib.Path(args.meta).parent
    cands = []
    if args.terrain_dir:
        cands.append(pathlib.Path(args.terrain_dir) / ref)
    cands.append(meta_dir / "terrain" / ref)
    cands.append(meta_dir / ref)
    for c in cands:
        if c.is_file():
            return str(c)
    print(f"  [Terrain] could not locate '{ref}' (looked in "
          f"{[str(c) for c in cands]}).")
    return None


def _build_terrain_xml(terrain, xy_offset):
    """Bake the transformed terrain into inline MuJoCo <mesh>/<geom> XML.

    The robot viewer re-centers the clip so frame-0 pelvis XY sits at the
    origin; the terrain gets the same XY shift so it stays aligned. Vertices
    are baked in world coords (geom at the origin) — MuJoCo internally
    compensates its mesh-centering, so they render where authored.
    """
    if terrain is None:
        return "", ""
    verts, faces = terrain
    v = verts.astype(np.float64).copy()
    v[:, 0] += xy_offset[0]
    v[:, 1] += xy_offset[1]
    vtx_str = " ".join(f"{x:.5f}" for x in v.reshape(-1))
    face_str = " ".join(str(int(i)) for i in faces.reshape(-1))
    # Match the UE editor look: light "sky" blue, matte (low specular /
    # shininess, no reflectance) so it reads soft like the engine viewport
    # rather than a glossy plastic. MuJoCo's per-face shading then naturally
    # gives lit step tops a lighter blue and the risers a deeper blue, the
    # same gradient seen in the UE screenshots.
    asset = (f'    <mesh name="ue_terrain" vertex="{vtx_str}" face="{face_str}"/>\n'
             '    <material name="ue_terrain_mat" rgba="0.42 0.66 0.92 1.0" '
             'specular="0.08" shininess="0.12" reflectance="0.0"/>\n')
    geom = ('    <geom name="ue_terrain_geom" type="mesh" mesh="ue_terrain" '
            'contype="0" conaffinity="0" group="2" material="ue_terrain_mat"/>\n')
    return asset, geom


_WRAPPER_XML_TMPL = """<mujoco>
  <include file="{g1_xml_name}"/>
  <visual>
    <map znear="0.001" zfar="100"/>
    <quality shadowsize="2048"/>
    <!-- UE-like soft lighting: strong sky-tinted ambient so shadowed faces
         stay visible (mimics the engine's SkyLight), and a restrained,
         low-specular diffuse so nothing blows out / looks over-bright. -->
    <headlight ambient="0.52 0.55 0.60" diffuse="0.30 0.30 0.32"
               specular="0.06 0.06 0.06"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".55 .55 .55" rgb2=".45 .45 .45"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.04"/>
{terrain_asset}  </asset>
  <worldbody>
    <!-- Soft directional fills from N/S/E/W + a faint top light. Sun-style
         (no distance falloff, parallel rays) so lighting stays uniform as
         the robot moves. Kept low so the headlight still defines the look. -->
    
    <geom name="viz_floor" type="plane" size="6 6 0.01" material="grid_mat"/>
{terrain_geom}  </worldbody>
</mujoco>
"""


_TRAJ_CAP_WARNED = [False]


# ── Static (full-clip) trajectory cache ──────────────────────────────────────
#
# The windowed overlay (`_add_traj_overlay`) is cheap because each frame only
# touches ~window*2 geoms. The FULL-clip overlay is the opposite: the polyline
# never changes shape — only "which segment is the current frame" colors shift
# and the gold dot moves. For long clips (10k+ frames) rebuilding it every
# frame is the playback bottleneck (15-30 ms/frame) and silently overflows
# user_scn.maxgeom (~5000), so most of the polyline is dropped anyway.
#
# Optimization: build the polyline + triad markers ONCE up front and stash
# them in scn.geoms[0..static_count). Per frame just bump scn.ngeom and
# refresh slot static_count with the gold "current" marker. Cost drops to
# O(1) per frame, fps cap goes from ~30 to whatever the renderer + sleep loop
# allows.

def _build_static_full_traj(
    scn,                              # viewer.user_scn
    viz_positions: np.ndarray,        # (N, 3)
    rot_matrices: np.ndarray,         # (N, 3, 3)
    marker_step: int,
    polyline_stride: int | None = None,
    triad_scale: float = 0.18,
) -> int:
    """Fill scn.geoms with strided polyline + triads. Reserve 1 trailing slot
    for the dynamic current-frame marker. Returns count of static geoms.

    `polyline_stride` defaults to `marker_step`: drawing polyline denser than
    markers is rarely worth the user_scn budget on long clips.
    """
    import mujoco as mj  # type: ignore

    n = viz_positions.shape[0]
    if polyline_stride is None:
        polyline_stride = marker_step
    polyline_stride = max(1, polyline_stride)
    marker_step = max(1, marker_step)

    traj_z = 0.005
    line_color = np.array([0.30, 0.55, 1.00, 0.85], dtype=np.float32)
    dot_color  = np.array([1.00, 0.55, 0.10, 1.00], dtype=np.float32)
    triad_colors = (
        np.array([1.0, 0.25, 0.25, 1.0], dtype=np.float32),  # +X red
        np.array([0.25, 1.0, 0.25, 1.0], dtype=np.float32),  # +Y green
        np.array([0.30, 0.55, 1.0,  1.0], dtype=np.float32),  # +Z blue
    )

    cap = scn.maxgeom
    # Reserve 5 trailing slots for the dynamic per-frame overlays:
    #   slot N+0 : gold   current-frame sphere
    #   slot N+1 : orange pelvis-orientation arrow (fixed length)
    #   slot N+2 : cyan   look-orientation  arrow (fixed length)
    #   slot N+3 : green  desired-velocity  arrow (length ∝ magnitude)
    #   slot N+4 : red    move_input world  arrow (fixed length)
    # Even when arrows are toggled off we still reserve them so toggling
    # at runtime never has to rebuild the static cache.
    cap_static = max(0, cap - 5)
    written = 0

    def _add(init_fn) -> bool:
        nonlocal written
        if written >= cap_static:
            return False
        init_fn(scn.geoms[written])
        written += 1
        return True

    # Strided polyline
    for i in range(0, n - polyline_stride, polyline_stride):
        p0 = np.array([viz_positions[i, 0],                   viz_positions[i, 1],                   traj_z])
        p1 = np.array([viz_positions[i + polyline_stride, 0], viz_positions[i + polyline_stride, 1], traj_z])
        def _init(g, _p0=p0, _p1=p1, _c=line_color):
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), _c)
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_LINE, 4.0, _p0, _p1)
        if not _add(_init):
            break

    # Triads + dots every marker_step
    for i in range(0, n, marker_step):
        pos = np.array([viz_positions[i, 0], viz_positions[i, 1], traj_z + 0.005])
        def _init_dot(g, _p=pos, _c=dot_color):
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.025, 0, 0]), _p,
                            np.eye(3).flatten(), _c)
        if not _add(_init_dot):
            break
        Rmat = rot_matrices[i]
        for axis in range(3):
            end = pos + triad_scale * Rmat[:, axis]
            col = triad_colors[axis]
            def _init_arrow(g, _from=pos.copy(), _to=end, _c=col):
                mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), _c)
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_ARROW, 0.010, _from, _to)
            if not _add(_init_arrow):
                break

    if written >= cap_static and not _TRAJ_CAP_WARNED[0]:
        print(f"  [Trajectory] WARN user_scn cap reached ({cap} geoms) while "
              f"baking full-clip overlay. Increase --marker-step to thin it out.")
        _TRAJ_CAP_WARNED[0] = True
    return written


def _set_current_marker(scn, slot_idx: int, pos_xy: np.ndarray) -> None:
    """Re-init scn.geoms[slot_idx] as the gold 'current frame' sphere."""
    import mujoco as mj  # type: ignore
    color = np.array([1.0, 0.85, 0.10, 1.0], dtype=np.float32)
    mj.mjv_initGeom(
        scn.geoms[slot_idx], mj.mjtGeom.mjGEOM_SPHERE,
        np.array([0.05, 0, 0]),
        np.array([pos_xy[0], pos_xy[1], 0.012]),
        np.eye(3).flatten(), color,
    )


# Ground-arrow colors for orientation overlay. Kept distinct from the
# pelvis-pose triad (R/G/B) and the trajectory line so it's unambiguous.
#   orange = robot pelvis forward (derived from qpos[3:7] — G1 IK output)
#   cyan   = user / camera forward (derived from cmd.lookY — player input)
_ORIENT_PELVIS_COLOR = np.array([1.00, 0.55, 0.10, 1.0], dtype=np.float32)  # orange
_ORIENT_LOOK_COLOR   = np.array([0.10, 0.85, 0.95, 1.0], dtype=np.float32)  # cyan

# Per-arrow style. Pelvis and Look are laterally staggered so they stay
# readable even when their directions are nearly colinear. DesVel is always
# centred on the robot and scales in length with its magnitude.
_ORIENT_PERP_OFFSET_M   = 0.09     # lateral offset from current position (each side)
_ORIENT_PELVIS_LENGTH_M = 0.70
_ORIENT_LOOK_LENGTH_M   = 0.55
_ORIENT_PELVIS_Z        = 0.025
_ORIENT_LOOK_Z          = 0.045
_ORIENT_PELVIS_SHAFT    = 0.022
_ORIENT_LOOK_SHAFT      = 0.016

# DesVel arrow: bright green, length = |des_vel| * VIZ_SCALE metres.
# `des_vel` is now REAL body-frame velocity in m/s (typical 0–7.5 m/s for
# walk/run/sprint), so the viz scale is meters-of-arrow per m/s of velocity.
# Magnitudes below MIN_MAG (m/s) are rendered as hidden so tiny numerical
# noise (e.g. idle frames with lv≈0) does not flicker a stub on screen.
_ORIENT_DESVEL_COLOR    = np.array([0.20, 0.95, 0.30, 1.0], dtype=np.float32)  # bright green
_ORIENT_DESVEL_VIZ_SCALE = 0.12    # m of arrow per m/s of body-frame velocity
_ORIENT_DESVEL_Z         = 0.065
_ORIENT_DESVEL_SHAFT     = 0.020
_ORIENT_DESVEL_MIN_MAG   = 0.05    # m/s — below this the arrow is hidden

# MoveInput arrow: red, fixed length.
# Direction = move_input rotated by camera yaw (look_fwd_mj_xy) into world frame.
# Hidden when move_input magnitude < MIN_MAG (no key pressed).
_ORIENT_MOVEINPUT_COLOR   = np.array([0.95, 0.15, 0.15, 1.0], dtype=np.float32)  # red
_ORIENT_MOVEINPUT_LENGTH_M = 0.60
_ORIENT_MOVEINPUT_Z        = 0.085
_ORIENT_MOVEINPUT_SHAFT    = 0.018
_ORIENT_MOVEINPUT_MIN_MAG  = 0.02


def _orientation_perp(pelvis_dir_xy: np.ndarray | None,
                      look_dir_xy: np.ndarray | None) -> np.ndarray:
    """Return a unit perpendicular (XY) to the average forward direction.

    Used to lateral-stagger the two arrows so they stay readable when the
    pelvis and look directions are nearly colinear. Falls back to world +Y
    when both inputs are degenerate.
    """
    ref = np.zeros(2, dtype=np.float64)
    if pelvis_dir_xy is not None:
        ref += np.asarray(pelvis_dir_xy, dtype=np.float64)
    if look_dir_xy is not None:
        ref += np.asarray(look_dir_xy, dtype=np.float64)
    n = float(np.linalg.norm(ref))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float64)
    ref /= n
    # 90° CCW rotation: (x, y) -> (-y, x)
    return np.array([-ref[1], ref[0]], dtype=np.float64)


def _set_orientation_arrow(
    scn,
    slot_idx: int,
    base_xy: np.ndarray,
    dir_xy: np.ndarray,
    color: np.ndarray,
    length: float = 0.55,
    z_offset: float = 0.02,
    shaft_radius: float = 0.018,
) -> None:
    """Re-init scn.geoms[slot_idx] as a flat ground arrow.

    The arrow lies just above the floor (`z_offset` m) so it stays visible
    on top of the trajectory polyline and the floor checker. Direction is
    taken straight from `dir_xy` (assumed already in MuJoCo display frame —
    see compute_world_orientation_arrows). Zero-length input is rendered as
    a tiny degenerate arrow so the slot stays valid without disappearing
    into the floor (still effectively invisible to the eye).
    """
    import mujoco as mj  # type: ignore

    norm = float(np.linalg.norm(dir_xy))
    if norm < 1e-6:
        # Frame has no meaningful direction (e.g. zero quat) — collapse the
        # arrow to a near-zero stub at base. Keeps the geom slot consistent.
        unit = np.array([1.0, 0.0], dtype=np.float64)
        eff_len = 1e-3
    else:
        unit = dir_xy.astype(np.float64) / norm
        eff_len = length

    p0 = np.array([base_xy[0],                base_xy[1],                z_offset], dtype=np.float64)
    p1 = np.array([base_xy[0] + unit[0] * eff_len,
                   base_xy[1] + unit[1] * eff_len,
                   z_offset], dtype=np.float64)
    g = scn.geoms[slot_idx]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), np.zeros(3),
                    np.eye(3).flatten(), color)
    mj.mjv_connector(g, mj.mjtGeom.mjGEOM_ARROW, shaft_radius, p0, p1)


def _hide_geom_slot(scn, slot_idx: int) -> None:
    """Collapse a reserved slot to an invisible (alpha=0) stub. Used when
    orientation arrows are toggled off but the slot must stay valid."""
    import mujoco as mj  # type: ignore
    transparent = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mj.mjv_initGeom(
        scn.geoms[slot_idx], mj.mjtGeom.mjGEOM_SPHERE,
        np.array([1e-4, 0, 0]),
        np.array([0.0, 0.0, -1.0]),
        np.eye(3).flatten(), transparent,
    )


def _set_desvel_arrow(
    scn,
    slot_idx: int,
    base_xy: np.ndarray,
    dv_world_xy: np.ndarray,
    enabled: bool,
) -> None:
    """Draw the desired-velocity arrow at a pre-reserved slot.

    `dv_world_xy` is the body-frame velocity in m/s already rotated into
    MuJoCo world XY (body frame [fwd, left] multiplied by the pelvis rotation
    matrix each frame — see the main viewer loop). Arrow length scales
    linearly with |dv_world_xy| via _ORIENT_DESVEL_VIZ_SCALE
    (metres of arrow per m/s of velocity). When the magnitude is below
    _ORIENT_DESVEL_MIN_MAG (m/s) the slot is hidden so tiny numerical noise
    does not flicker a stub arrow on screen.
    """
    if not enabled:
        _hide_geom_slot(scn, slot_idx)
        return
    mag = float(np.linalg.norm(dv_world_xy))
    if mag < _ORIENT_DESVEL_MIN_MAG:
        _hide_geom_slot(scn, slot_idx)
        return
    length = mag * _ORIENT_DESVEL_VIZ_SCALE
    direction = np.asarray(dv_world_xy, dtype=np.float64) / mag
    _set_orientation_arrow(
        scn, slot_idx, np.asarray(base_xy),
        direction, _ORIENT_DESVEL_COLOR,
        length=length,
        z_offset=_ORIENT_DESVEL_Z,
        shaft_radius=_ORIENT_DESVEL_SHAFT,
    )


def _append_desvel_arrow(
    scn,
    base_xy: np.ndarray,
    dv_world_xy: np.ndarray,
) -> None:
    """Append the desired-velocity arrow to scn.ngeom (windowed / no-traj modes)."""
    mag = float(np.linalg.norm(dv_world_xy))
    if mag < _ORIENT_DESVEL_MIN_MAG or scn.ngeom >= scn.maxgeom:
        return
    length = mag * _ORIENT_DESVEL_VIZ_SCALE
    direction = np.asarray(dv_world_xy, dtype=np.float64) / mag
    _set_orientation_arrow(
        scn, scn.ngeom, np.asarray(base_xy),
        direction, _ORIENT_DESVEL_COLOR,
        length=length,
        z_offset=_ORIENT_DESVEL_Z,
        shaft_radius=_ORIENT_DESVEL_SHAFT,
    )
    scn.ngeom += 1


def _emit_orientation_arrows(
    scn,
    slot_pelvis: int,
    slot_look: int,
    base_xy: np.ndarray,
    pelvis_dir_xy: np.ndarray | None,
    look_dir_xy: np.ndarray | None,
    enabled: bool,
) -> None:
    """Write pelvis + look orientation arrows into pre-reserved scn slots.

    slot_pelvis : orange arrow — robot G1 pelvis forward (from qpos[3:7])
    slot_look   : cyan  arrow — user camera forward     (from cmd.lookY)

    When `enabled=False` the slots are kept but collapsed to a 0-alpha
    stub so toggling at runtime never has to renumber scn.ngeom. The two
    arrows are laterally offset (±_ORIENT_PERP_OFFSET_M) from base_xy
    along the perpendicular of the average forward direction."""
    if enabled and pelvis_dir_xy is None and look_dir_xy is None:
        enabled = False
    if not enabled:
        _hide_geom_slot(scn, slot_pelvis)
        _hide_geom_slot(scn, slot_look)
        return

    perp = _orientation_perp(pelvis_dir_xy, look_dir_xy)
    base_pelvis = np.asarray(base_xy, dtype=np.float64) + perp * _ORIENT_PERP_OFFSET_M
    base_look   = np.asarray(base_xy, dtype=np.float64) - perp * _ORIENT_PERP_OFFSET_M

    if pelvis_dir_xy is not None:
        _set_orientation_arrow(scn, slot_pelvis, base_pelvis, pelvis_dir_xy,
                               _ORIENT_PELVIS_COLOR,
                               length=_ORIENT_PELVIS_LENGTH_M,
                               z_offset=_ORIENT_PELVIS_Z,
                               shaft_radius=_ORIENT_PELVIS_SHAFT)
    else:
        _hide_geom_slot(scn, slot_pelvis)
    if look_dir_xy is not None:
        _set_orientation_arrow(scn, slot_look, base_look, look_dir_xy,
                               _ORIENT_LOOK_COLOR,
                               length=_ORIENT_LOOK_LENGTH_M,
                               z_offset=_ORIENT_LOOK_Z,
                               shaft_radius=_ORIENT_LOOK_SHAFT)
    else:
        _hide_geom_slot(scn, slot_look)


def _append_orientation_arrows(
    scn,
    base_xy: np.ndarray,
    pelvis_dir_xy: np.ndarray,
    look_dir_xy: np.ndarray | None,
) -> None:
    """Append pelvis (+ optional look) orientation arrows after scn.ngeom.
    Used by the windowed-trajectory mode where geoms are rebuilt every frame.
    Same lateral-stagger / Z-stagger / size-stagger as _emit_orientation_arrows."""
    n_needed = 2 if look_dir_xy is not None else 1
    if scn.ngeom + n_needed > scn.maxgeom:
        return
    perp = _orientation_perp(pelvis_dir_xy, look_dir_xy)
    base_pelvis = np.asarray(base_xy, dtype=np.float64) + perp * _ORIENT_PERP_OFFSET_M

    _set_orientation_arrow(scn, scn.ngeom, base_pelvis, pelvis_dir_xy,
                           _ORIENT_PELVIS_COLOR,
                           length=_ORIENT_PELVIS_LENGTH_M,
                           z_offset=_ORIENT_PELVIS_Z,
                           shaft_radius=_ORIENT_PELVIS_SHAFT)
    scn.ngeom += 1

    if look_dir_xy is not None:
        base_look = np.asarray(base_xy, dtype=np.float64) - perp * _ORIENT_PERP_OFFSET_M
        _set_orientation_arrow(scn, scn.ngeom, base_look, look_dir_xy,
                               _ORIENT_LOOK_COLOR,
                               length=_ORIENT_LOOK_LENGTH_M,
                               z_offset=_ORIENT_LOOK_Z,
                               shaft_radius=_ORIENT_LOOK_SHAFT)
        scn.ngeom += 1


def _set_moveinput_arrow(
    scn,
    slot_idx: int,
    base_xy: np.ndarray,
    mi_world_xy: np.ndarray,
    enabled: bool,
) -> None:
    """Draw the move_input direction arrow at a pre-reserved slot.

    `mi_world_xy` is the move_input vector already projected into MuJoCo world
    XY by the caller: fwd_world * mi[0] + right_world * mi[1].
    Hidden when magnitude < _ORIENT_MOVEINPUT_MIN_MAG (no key pressed).
    """
    if not enabled:
        _hide_geom_slot(scn, slot_idx)
        return
    mag = float(np.linalg.norm(mi_world_xy))
    if mag < _ORIENT_MOVEINPUT_MIN_MAG:
        _hide_geom_slot(scn, slot_idx)
        return
    direction = np.asarray(mi_world_xy, dtype=np.float64) / mag
    _set_orientation_arrow(
        scn, slot_idx, np.asarray(base_xy),
        direction, _ORIENT_MOVEINPUT_COLOR,
        length=_ORIENT_MOVEINPUT_LENGTH_M,
        z_offset=_ORIENT_MOVEINPUT_Z,
        shaft_radius=_ORIENT_MOVEINPUT_SHAFT,
    )


def _append_moveinput_arrow(
    scn,
    base_xy: np.ndarray,
    mi_world_xy: np.ndarray,
) -> None:
    """Append the move_input arrow to scn.ngeom (windowed / no-traj modes)."""
    mag = float(np.linalg.norm(mi_world_xy))
    if mag < _ORIENT_MOVEINPUT_MIN_MAG or scn.ngeom >= scn.maxgeom:
        return
    direction = np.asarray(mi_world_xy, dtype=np.float64) / mag
    _set_orientation_arrow(
        scn, scn.ngeom, np.asarray(base_xy),
        direction, _ORIENT_MOVEINPUT_COLOR,
        length=_ORIENT_MOVEINPUT_LENGTH_M,
        z_offset=_ORIENT_MOVEINPUT_Z,
        shaft_radius=_ORIENT_MOVEINPUT_SHAFT,
    )
    scn.ngeom += 1


def _add_traj_overlay(
    viewer,
    fi: int,
    viz_positions: np.ndarray,   # (N, 3) pelvis xy in scene coords (z ignored)
    rot_matrices: np.ndarray,    # (N, 3, 3) pelvis world rotation
    window: int,
    marker_step: int,
    triad_scale: float = 0.18,
) -> None:
    """Populate viewer.user_scn with:
        • a polyline of pelvis ground-projected positions in [fi-window, fi+window]
        • XYZ-triad markers (red/green/blue arrows) every `marker_step` frames
          inside that window, showing pelvis pose

    Pass `window <= 0` (or any window >= n_frames) to render the FULL motion
    trajectory.
    """
    import mujoco as mj  # type: ignore

    n = viz_positions.shape[0]
    if window <= 0 or window >= n:
        lo, hi = 0, n
    else:
        lo = max(0, fi - window)
        hi = min(n, fi + window + 1)

    scn = viewer.user_scn
    scn.ngeom = 0
    cap = scn.maxgeom

    def _add(init_fn):
        if scn.ngeom >= cap:
            return None
        g = scn.geoms[scn.ngeom]
        init_fn(g)
        scn.ngeom += 1
        return g

    # Trajectory polyline (drawn slightly above floor so it isn't z-fought away)
    traj_z = 0.005
    past_color    = np.array([0.30, 0.55, 1.00, 0.85], dtype=np.float32)  # light blue
    future_color  = np.array([0.30, 1.00, 0.55, 0.85], dtype=np.float32)  # light green
    current_color = np.array([1.00, 0.85, 0.10, 1.00], dtype=np.float32)  # gold

    for i in range(lo, hi - 1):
        p0 = np.array([viz_positions[i, 0],   viz_positions[i, 1],   traj_z])
        p1 = np.array([viz_positions[i+1, 0], viz_positions[i+1, 1], traj_z])
        col = past_color if i + 1 <= fi else future_color

        def _init(g, _p0=p0, _p1=p1, _c=col):
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), _c)
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_LINE, 4.0, _p0, _p1)
        _add(_init)

    # Frame triads + position dot every marker_step
    triad_colors = (
        np.array([1.0, 0.25, 0.25, 1.0], dtype=np.float32),  # +X red
        np.array([0.25, 1.0, 0.25, 1.0], dtype=np.float32),  # +Y green
        np.array([0.30, 0.55, 1.0,  1.0], dtype=np.float32),  # +Z blue
    )
    # Snap markers to multiples of marker_step so they stay anchored frame-to-frame.
    first_marker = ((lo + marker_step - 1) // marker_step) * marker_step
    for i in range(first_marker, hi, marker_step):
        if i >= n:
            break
        pos = np.array([viz_positions[i, 0], viz_positions[i, 1], traj_z + 0.005])

        # base dot
        dot_color = current_color if i == fi else np.array([1.0, 0.55, 0.10, 1.0], dtype=np.float32)
        def _init_dot(g, _p=pos, _c=dot_color, _r=(0.045 if i == fi else 0.025)):
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE,
                            np.array([_r, 0, 0]), _p,
                            np.eye(3).flatten(), _c)
        _add(_init_dot)

        # triad (3 arrows along pelvis local axes, in world)
        Rmat = rot_matrices[i]
        for axis in range(3):
            end = pos + triad_scale * Rmat[:, axis]
            col = triad_colors[axis]
            def _init_arrow(g, _from=pos.copy(), _to=end, _c=col):
                mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), _c)
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_ARROW, 0.010, _from, _to)
            _add(_init_arrow)

    # One-shot warning: clip too long for the viewer's user_scn capacity.
    # Tell the user to bump --marker-step (cheaper than per-frame striding here).
    if scn.ngeom >= cap and not _TRAJ_CAP_WARNED[0]:
        n_lines_needed = max(0, hi - lo - 1)
        n_marker_geoms = ((hi - first_marker + marker_step - 1) // marker_step) * 4 \
            if marker_step > 0 else 0
        print(f"  [Trajectory] WARN user_scn cap reached ({cap} geoms). "
              f"Wanted ~{n_lines_needed + max(0, n_marker_geoms)}; "
              f"some segments/markers were dropped. "
              f"Increase --marker-step or shrink --traj-window.")
        _TRAJ_CAP_WARNED[0] = True


def visualize_g1(
    qpos_arr: np.ndarray,
    fps: float,
    traj_window: int = 100,
    marker_step: int = 10,
    show_trajectory: bool = True,
    look_fwd_xy: np.ndarray | None = None,
    des_vel_arr: np.ndarray | None = None,
    move_input_arr: np.ndarray | None = None,
    show_orient_arrows: bool = True,
    terrain=None,
):
    """Replay qpos on a Unitree G1 in a passive MuJoCo viewer.

    The qpos pelvis XY can be far from the world origin (the global Kabsch
    translation places it at the BVH-calibration anchor, e.g. y≈-8 m). We
    re-center the trajectory on the origin in XY only — Z is left untouched
    so the IK-computed foot height is preserved exactly.

    Trajectory overlay: at each frame, draws a polyline of pelvis ground
    positions in the ±`traj_window`-frame window, with XYZ triads every
    `marker_step` frames showing pelvis pose. Past = blue, future = green,
    current = gold.

    Orientation arrows (toggle with `O` key):
        orange = Robot pelvis forward    — from qpos[3:7], fixed length
        cyan   = User / camera forward   — from cmd.lookY, fixed length
        green  = REAL body-frame velocity (m/s) rotated into world XY —
                 from cmd.des_vel (root.lv in actor body), length ∝ |v|
        red    = Move-input world dir    — from cmd.move_input rotated by
                 camera yaw (look_fwd_mj_xy), fixed length, hidden when no key
    Pelvis always available. Cyan/green/red require cmd data to be passed.
    """
    _ensure_gmr_on_path()
    import mujoco as mj  # type: ignore
    import mujoco.viewer  # type: ignore
    from general_motion_retargeting.params import ROBOT_XML_DICT  # type: ignore

    # XY-only re-centering offset (Z=0 → preserves foot ground contact).
    # Computed up front so the terrain baked into the wrapper XML shares it.
    xy_offset = np.array([-qpos_arr[0, 0], -qpos_arr[0, 1], 0.0], dtype=np.float64)
    print(f"  Visualization XY offset (frame 0 → origin): {xy_offset[:2]}")

    terrain_asset, terrain_geom = _build_terrain_xml(terrain, xy_offset)

    g1_xml = pathlib.Path(str(ROBOT_XML_DICT["unitree_g1"]))
    wrapper_path = g1_xml.parent / "_ue_world_retarget_wrapper.xml"

    def _load_model(asset_str, geom_str):
        wrapper_path.write_text(
            _WRAPPER_XML_TMPL.format(
                g1_xml_name=g1_xml.name,
                terrain_asset=asset_str,
                terrain_geom=geom_str,
            ),
            encoding="utf-8",
        )
        try:
            return mj.MjModel.from_xml_path(str(wrapper_path))
        finally:
            wrapper_path.unlink(missing_ok=True)

    try:
        model = _load_model(terrain_asset, terrain_geom)
        if terrain_asset:
            print(f"  [Viewer] terrain baked into scene "
                  f"({terrain[0].shape[0]} verts, {terrain[1].shape[0]} tris).")
    except Exception as e:
        if terrain_asset:
            print(f"  [Viewer] terrain failed to compile "
                  f"({type(e).__name__}: {e}); loading WITHOUT terrain.")
            model = _load_model("", "")
        else:
            raise

    assert model.nq == qpos_arr.shape[1], (
        f"G1 nq={model.nq} but qpos has {qpos_arr.shape[1]} dims"
    )
    data = mj.MjData(model)

    # Pre-compute trajectory overlay data in scene coords.
    viz_positions = qpos_arr[:, :3] + xy_offset            # (N, 3)
    quats_wxyz = qpos_arr[:, 3:7]                           # MuJoCo: [w, x, y, z]
    quats_xyzw = np.stack(
        [quats_wxyz[:, 1], quats_wxyz[:, 2], quats_wxyz[:, 3], quats_wxyz[:, 0]],
        axis=1,
    )
    rot_matrices = R.from_quat(quats_xyzw).as_matrix()      # (N, 3, 3)

    n_frames = qpos_arr.shape[0]

    # Pelvis forward in MuJoCo display frame: first column of the rotation
    # matrix = +X axis in world. Computed directly from qpos — always available.
    pelvis_fwd_xy = rot_matrices[:, :2, 0].astype(np.float32)   # (N, 2)

    # look_fwd_xy and des_vel_arr are optional (require JSONL cmd data).
    have_look = (look_fwd_xy is not None and look_fwd_xy.shape[0] >= n_frames)
    if look_fwd_xy is not None and not have_look:
        print(f"  [Viewer] look_fwd_xy length mismatch "
              f"(motion={n_frames}, look={look_fwd_xy.shape[0]}); look arrow disabled.")
        look_fwd_xy = None
        have_look = False
    have_desvel = (des_vel_arr is not None and des_vel_arr.shape[0] >= n_frames)
    if des_vel_arr is not None and not have_desvel:
        print(f"  [Viewer] des_vel_arr length mismatch "
              f"(motion={n_frames}, des_vel={des_vel_arr.shape[0]}); desvel arrow disabled.")
        des_vel_arr = None
        have_desvel = False
    # move_input → world requires look_fwd for the camera-yaw rotation.
    have_moveinput = (
        move_input_arr is not None
        and move_input_arr.shape[0] >= n_frames
        and have_look
    )
    if move_input_arr is not None and not have_moveinput:
        print(f"  [Viewer] move_input_arr disabled "
              f"(need both move_input data and look_fwd_xy).")
        move_input_arr = None
    show_orient_arrows = bool(show_orient_arrows)

    state = {
        "paused":      True,
        "fi":          0,
        "step":        0,
        "show_traj":   show_trajectory,
        "show_orient": show_orient_arrows,
        "show_vel":    False,
    }

    def key_cb(k):
        if k == 32:        # Space
            state["paused"] = not state["paused"]
        elif k == 262:     # Right
            state["step"] = 1
        elif k == 263:     # Left
            state["step"] = -1
        elif k == 259:     # Backspace
            state["fi"] = 0
        elif k == 84:      # T  → toggle trajectory overlay
            state["show_traj"] = not state["show_traj"]
        elif k == 79:      # O  → toggle orientation arrows
            state["show_orient"] = not state["show_orient"]
        elif k == 86:      # V  → toggle velocity arrows (green + red)
            state["show_vel"] = not state["show_vel"]

    viewer = mujoco.viewer.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
        key_callback=key_cb,
    )
    viewer.cam.lookat = np.array([0.0, 0.0, 0.85])
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 180

    last_t = time.time()
    dt = 1.0 / max(1.0, float(fps))
    full_traj_mode = (traj_window <= 0 or traj_window >= n_frames)
    # In full-mode the polyline + triads are static — bake them once into
    # user_scn and reserve the 3 trailing slots (current marker + 2 arrows).
    static_count = -1
    print(f"  Controls: Space=pause  Left/Right=step  Backspace=reset  "
          f"T=toggle trajectory  O=toggle orientation arrows  "
          f"V=toggle velocity arrows (green+red)")
    if full_traj_mode:
        print(f"  Trajectory: FULL clip ({n_frames} frames), polyline+triad every "
              f"{marker_step} frames ({'ON' if show_trajectory else 'OFF'} at start, baked once)")
    else:
        print(f"  Trajectory: ±{traj_window} frames, triad every {marker_step} frames "
              f"({'ON' if show_trajectory else 'OFF'} at start)")
    print(f"  Orientation arrows ({'ON' if show_orient_arrows else 'OFF'} at start):")
    print(f"    orange = Pelvis forward (qpos[3:7], fixed length)")
    print(f"    cyan   = Look/camera   (cmd.lookY"
          f"{'  ← N/A, no cmd data' if not have_look else ', fixed length'})")
    print(f"  Velocity arrows (OFF at start, toggle with V):")
    print(f"    green  = DesVel world  (cmd.des_vel m/s, length = |v|*"
          f"{_ORIENT_DESVEL_VIZ_SCALE} m/(m/s)"
          f"{'  ← N/A, no cmd data' if not have_desvel else ''})")
    print(f"    red    = MoveInput world (cmd.move_input × camera yaw, fixed length"
          f"{'  ← N/A, need look data' if not have_moveinput else ', ' + str(_ORIENT_MOVEINPUT_LENGTH_M) + 'm'})")
    print(f"  Playback fps={fps:.1f} (frame interval {dt*1000:.2f} ms)")
    print(f"  Started PAUSED. Press Space to play.")

    while viewer.is_running():
        if state["step"] != 0:
            state["fi"] = (state["fi"] + state["step"]) % n_frames
            state["step"] = 0
        elif not state["paused"]:
            now = time.time()
            if now - last_t >= dt:
                state["fi"] = (state["fi"] + 1) % n_frames
                last_t = now
            else:
                # Sleep at most until the next frame is due. Capped so we still
                # spin at >=1kHz for input responsiveness when fps is very high.
                wait = min(dt - (now - last_t), 0.001)
                if wait > 0:
                    time.sleep(wait)
                continue

        qpos = qpos_arr[state["fi"]].copy()
        qpos[:3] += xy_offset
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        # Per-frame world-frame vectors derived from cmd data.
        fi = state["fi"]

        # des_vel: body frame [fwd, left] → world via pelvis rotation.
        if have_desvel:
            dv = des_vel_arr[fi].astype(np.float64)           # [fwd, left]
            dv_world_xy = (rot_matrices[fi, :2, 0] * dv[0]   # pelvis +X * fwd
                         + rot_matrices[fi, :2, 1] * dv[1])  # pelvis +Y * left
        else:
            dv_world_xy = np.zeros(2)

        # move_input: camera input space [fwd, right] → world via camera yaw.
        # look_fwd = camera +X in world; camera right = [fy, -fx] (90° CW).
        if have_moveinput:
            mi = move_input_arr[fi].astype(np.float64)        # [fwd, right]
            lf = look_fwd_xy[fi].astype(np.float64)           # camera forward
            lr = np.array([lf[1], -lf[0]])                    # camera right
            mi_world_xy = lf * mi[0] + lr * mi[1]
        else:
            mi_world_xy = np.zeros(2)

        scn = viewer.user_scn
        if state["show_traj"]:
            if full_traj_mode:
                if static_count < 0:
                    scn.ngeom = 0
                    static_count = _build_static_full_traj(
                        scn, viz_positions, rot_matrices, marker_step,
                    )
                # Reserve [static_count : static_count+5]:
                #   +0 gold   current-frame dot
                #   +1 orange pelvis-orientation arrow (fixed length)
                #   +2 cyan   look-orientation  arrow (fixed length)
                #   +3 green  desired-velocity  arrow (length ∝ magnitude)
                #   +4 red    move_input world  arrow (fixed length)
                scn.ngeom = static_count + 5
                _set_current_marker(scn, static_count, viz_positions[fi])
                _emit_orientation_arrows(
                    scn,
                    slot_pelvis=static_count + 1,
                    slot_look=static_count + 2,
                    base_xy=viz_positions[fi, :2],
                    pelvis_dir_xy=pelvis_fwd_xy[fi],
                    look_dir_xy=look_fwd_xy[fi] if have_look else None,
                    enabled=state["show_orient"],
                )
                _set_desvel_arrow(
                    scn, static_count + 3,
                    viz_positions[fi, :2], dv_world_xy,
                    enabled=state["show_orient"] and state["show_vel"] and have_desvel,
                )
                _set_moveinput_arrow(
                    scn, static_count + 4,
                    viz_positions[fi, :2], mi_world_xy,
                    enabled=state["show_orient"] and state["show_vel"] and have_moveinput,
                )
            else:
                _add_traj_overlay(
                    viewer, fi, viz_positions, rot_matrices,
                    window=traj_window, marker_step=marker_step,
                )
                if state["show_orient"]:
                    _append_orientation_arrows(
                        scn,
                        base_xy=viz_positions[fi, :2],
                        pelvis_dir_xy=pelvis_fwd_xy[fi],
                        look_dir_xy=look_fwd_xy[fi] if have_look else None,
                    )
                    if state["show_vel"] and have_desvel:
                        _append_desvel_arrow(scn, viz_positions[fi, :2], dv_world_xy)
                    if state["show_vel"] and have_moveinput:
                        _append_moveinput_arrow(scn, viz_positions[fi, :2], mi_world_xy)
        else:
            scn.ngeom = 0
            if state["show_orient"]:
                _append_orientation_arrows(
                    scn,
                    base_xy=viz_positions[fi, :2],
                    pelvis_dir_xy=pelvis_fwd_xy[fi],
                    look_dir_xy=look_fwd_xy[fi] if have_look else None,
                )
                if state["show_vel"] and have_desvel:
                    _append_desvel_arrow(scn, viz_positions[fi, :2], dv_world_xy)
                if state["show_vel"] and have_moveinput:
                    _append_moveinput_arrow(scn, viz_positions[fi, :2], mi_world_xy)

        viewer.sync()

        if state["paused"] and state["step"] == 0:
            time.sleep(0.01)

    viewer.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--jsonl", default=DEFAULT_JSONL,
                    help="GASP animation JSONL to retarget")
    ap.add_argument("--meta",  default=DEFAULT_META)
    ap.add_argument("--stage", choices=["s1", "s3", "s5"], default="s5",
                    help="Pipeline stage from verify_pipeline: "
                         "s1=raw UE world wp(cm), s3=+UE->MuJoCo+Kabsch(m), "
                         "s5=full retarget + MuJoCo native viewer.")
    ap.add_argument("--data-scale", type=float, default=1.0,
                    help="Uniform scale factor applied about the UE world "
                         "origin to BOTH the source UE motion positions and "
                         "the (visualization) terrain, so a batch captured at "
                         "the original UE character height is shrunk to the G1 "
                         "character height. G1 ~= 0.77x the source UE character "
                         "height, so pass --data-scale 0.77 for ground/traversal "
                         "(original-height) recordings and 1.0 for stairs "
                         "(already G1-scaled) recordings. Default 1.0. Only "
                         "affects positions (rotations are scale-free); the "
                         "Kabsch/ball-align/retarget pipeline then runs exactly "
                         "as if the data had been captured at G1 scale.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Alignment config produced by gasp_bvh_calibrate.py")
    ap.add_argument("--src-human", type=str, default="bvh_ue5_g1scale",
                    choices=["bvh_ue5_native", "bvh_ue5_g1scale"],
                    help="GMR source skeleton key. 'bvh_ue5_g1scale' (default) "
                         "for a character scaled to G1 height (~1.32 m); "
                         "'bvh_ue5_native' for a 1.75 m human character.")
    ap.add_argument("--output-qpos", default=None,
                    help="Base path for the exported G1 qpos .npy "
                         "(shape: [n_frames, nq]). If omitted, it is derived "
                         "from --jsonl: <jsonl dir>/<jsonl stem>.npy (e.g. "
                         "WalkTurn_..._frames.jsonl → WalkTurn_..._frames.npy). "
                         "Sibling '_cmd.npy' (command) and '_scaled.npz' "
                         "(pre-IK skeleton) files are written alongside it (see "
                         "--no-output-cmd / --output-cmd / --no-save).")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not write any export files (qpos / cmd / scaled). "
                         "Useful for visualization-only runs.")
    ap.add_argument("--output-cmd", default=None,
                    help="Override the path of the auxiliary command/root "
                         "file (default: <output_qpos stem>_cmd.npy alongside "
                         "the motion .npy). Structured numpy array, one row "
                         "per motion frame; see COMMAND_DTYPE in this module.")
    ap.add_argument("--no-output-cmd", action="store_true",
                    help="Do not save the per-frame command/root .npy. "
                         "Has no effect when --output-qpos is unset.")
    ap.add_argument("--no-orientation-arrows", action="store_true",
                    help="Hide the cyan (Actor) and magenta (Look) ground "
                         "arrows in the viewer. Toggle at runtime with the "
                         "'O' key. The arrays are still computed so they can "
                         "be saved into the *_cmd.npy file.")
    ap.add_argument("--no-visualize", action="store_true",
                    help="Skip the MuJoCo viewer (useful for batch jobs)")
    ap.add_argument("--no-terrain", action="store_true",
                    help="Do not load / visualize the exported scene terrain.")
    ap.add_argument("--terrain", default=None,
                    help="Explicit path to a terrain_<hash>.json. Default: "
                         "resolved from the recording meta's `terrain_ref`.")
    ap.add_argument("--terrain-dir", default=None,
                    help="Directory holding terrain files. Default: a `terrain` "
                         "folder next to --meta (then next to --meta itself).")
    ap.add_argument("--terrain-ground-z0", type=float, default=0.0,
                    help="UE ground height (cm) used as the terrain z0 reference; "
                         "subtracted from UE Z so the floor maps onto the robot's "
                         "z=0 plane. Default 0.")
    ap.add_argument("--world-z0", type=float, default=86.0,
                    help="(Deprecated — S5 positions now come from joints[*].wp "
                         "(skel_mj), not load_gasp_data, so this value is "
                         "unused.)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier on the recording's native "
                         "sample rate (read from meta sample_rate_hz, default 60). "
                         "1.0 = original speed (default), 2.0 = 2x fast-forward, "
                         "0.25 = 4x slow-mo. Ignored if --fps is set.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Force exact playback FPS, overriding --speed and the "
                         "recording's native sample rate. Default: unset.")
    ap.add_argument("--no-ball-align", action="store_true",
                    help="Disable the S5 Root-Z (toe-to-toe) grounding shift on "
                         "the retargeted G1. Use to inspect raw GMR foot IK.")
    ap.add_argument("--extra-drop", type=float, default=0.031,
                    help="offset_human: human toe joint (ball) → ground (m). "
                         "Default 0.031 (3.1 cm). Used by S5 Root-Z + per-foot "
                         "grounding.")
    ap.add_argument("--no-scaled-ground", action="store_true",
                    help="(S5) disable method-3 support-foot pre-IK grounding "
                         "of the scaled-human IK targets.")
    ap.add_argument("--ground-softness", type=float, default=50.0,
                    help="(S5, method-3 pre-IK grounding) soft-min sharpness k "
                         "(1/m) for the support-foot grounding blend. Larger = "
                         "closer to a hard 'lowest foot' pick; smaller = "
                         "softer double-support handover (~1/k m band). "
                         "Default 50 (~2 cm band).")
    ap.add_argument("--per-foot-ground", action="store_true",
                    help="After Root-Z grounding, independently bend each leg so "
                         "each foot lands on its own target ground (handles "
                         "stairs / feet at different heights).")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="Hide the pelvis trajectory overlay in the viewer.")
    ap.add_argument("--traj-window", type=int, default=100,
                    help="Trajectory overlay window: ± this many frames around "
                         "the current frame. Default 100. Set <=0 (or use "
                         "--full-trajectory) to draw the entire motion.")
    ap.add_argument("--full-trajectory", action="store_true",
                    help="Draw the FULL motion trajectory (all frames) instead "
                         "of a sliding window. Equivalent to --traj-window=<huge>. "
                         "For very long clips you may want to bump --marker-step "
                         "to keep the overlay readable.")
    ap.add_argument("--marker-step", type=int, default=10,
                    help="Draw a position+orientation triad every N frames "
                         "inside the trajectory window. Default 10.")
    args = ap.parse_args()

    stage = args.stage
    need_kabsch = stage in ("s3", "s5")

    # S1 file read path (verify_pipeline): JSONL joints[*].wp in UE world cm.
    bone_names, _ = load_meta(args.meta)
    frames = load_frames(args.jsonl)
    n_frames = len(frames)
    print(f"[Human] frames={n_frames}, bones={len(bone_names)}, stage={stage}")
    skel_ue = np.zeros((n_frames, len(bone_names), 3), dtype=np.float64)
    for fi, fr in enumerate(frames):
        skel_ue[fi] = skeleton_world_direct(fr)

    # Uniformly scale the source UE motion about the UE world origin so a batch
    # captured at the original UE character height matches the G1 character
    # height (G1 ~= 0.77x the source UE character). The scaled positions
    # propagate to skel_mj / skel_kabsch / GMR input; the Kabsch config
    # (calibrated for G1-scale data) then applies as-is. See --data-scale.
    data_scale = float(args.data_scale)
    if abs(data_scale - 1.0) > 1e-9:
        skel_ue *= data_scale
        print(f"[Data scale] motion x{data_scale:.4f} (about UE origin) "
              f"-> G1-height matched")

    R_global = None
    t_global = None
    delta = {}
    cfg = {}
    skel_kabsch = None
    if need_kabsch:
        if not pathlib.Path(args.config).exists():
            raise FileNotFoundError(
                f"Alignment config not found: {args.config}\n"
                f"Run gasp_bvh_calibrate.py first to produce it."
            )
        print(f"[Config] {args.config}")
        R_global, t_global, delta, cfg = load_alignment_config(args.config)
        cm = cfg.get("calib_meta", {})
        print(f"  Calibrated from BVH = {cm.get('bvh')}")
        print(f"                  GASP = {cm.get('gasp_jsonl')}")
        print(f"                  frame= {cm.get('bvh_calib_frame')} (BVH) / "
              f"{cm.get('gasp_calib_frame')} (GASP)")
        print(f"  Anchors: {cm.get('anchors')}")
        print(f"  Kabsch RMS = {cm.get('kabsch_residual_rms_m', 0.0)*100:.2f} cm   "
              f"R_global ang = {cm.get('global_rotation_angle_deg', 0.0):.2f} deg")
        print(f"  Per-bone Δ count: {len(delta)}")

        # S3: UE -> MuJoCo + Kabsch.
        skel_mj = np.zeros_like(skel_ue)
        skel_kabsch = np.zeros_like(skel_ue)
        for fi in range(n_frames):
            skel_mj[fi] = ue_to_mujoco(skel_ue[fi])
            skel_kabsch[fi] = apply_kabsch(skel_mj[fi], R_global, t_global)

    if stage == "s1":
        z = skel_ue[:, :, 2]
        print(f"[S1] raw UE world wp loaded. Z range(cm): "
              f"{z.min():.2f} .. {z.max():.2f}")
        if args.output_qpos:
            out = pathlib.Path(args.output_qpos)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), skel_ue)
            print(f"  Saved S1 skeleton → {out}  (shape: {skel_ue.shape})")
        return

    if stage == "s3":
        assert skel_kabsch is not None
        z = skel_kabsch[:, :, 2]
        print(f"[S3] UE->MuJoCo+Kabsch loaded. Z range(m): "
              f"{z.min():.4f} .. {z.max():.4f}")
        if args.output_qpos:
            out = pathlib.Path(args.output_qpos)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), skel_kabsch)
            print(f"  Saved S3 skeleton → {out}  (shape: {skel_kabsch.shape})")
        return

    # stage == s5: use verify_pipeline's verified retarget reading path.
    # Positions come from wp (skel_mj, computed above from joints[*].wp →
    # UE→MuJoCo) — the SAME source as skel_kabsch used for grounding, so the
    # GMR input and grounding reference share one Z (no ground_z_cm mismatch).
    # Only the rotations are taken from load_gasp_data (cq + root.q → BVH).
    assert skel_kabsch is not None and skel_mj is not None
    print(f"\n[S5] Loading GASP rotations via vis_skeleton_compare ...")
    _, gasp_rot_all, _, _ = load_gasp_data(
        args.jsonl, args.meta,
        preserve_world_z=False, ground_z_cm=0.0,
    )
    print(f"  skel_mj shape = {skel_mj.shape} (wp-based positions for GMR)")

    # ── 3) Apply alignment + build GMR input ──
    # Positions use skel_mj (from wp); rotations use gasp_rot_all. Both are in
    # meta bone order (load_gasp_data names == meta bone_names), so we pass
    # bone_names as the shared name list.
    print(f"\n[Align + build GMR frames]")
    gmr_frames, root_xy = build_gmr_frames(
        skel_mj, gasp_rot_all, bone_names,
        R_global, t_global, delta,
        recenter_root_xy=True,
    )
    # Prefer the canonical BVH-rest height baked into the calibration config.
    # Falls back to a per-frame heuristic only if the config predates v1.1, in
    # which case the height may be wrong (causes feet-through-floor).
    h_cfg = cfg.get("actual_human_height_m")
    if h_cfg is not None:
        h = float(h_cfg)
        print(f"  Using calibrated human_height = {h:.3f} m  (from config)")
    else:
        h = estimate_human_height(gmr_frames)
        print(f"  [WARN] config has no 'actual_human_height_m' — falling back "
              f"to frame-0 estimate = {h:.3f} m. Re-run gasp_bvh_calibrate.py "
              f"to get the canonical BVH-rest height.")

    # ── 4a) Pre-IK grounding (verify_pipeline method 3) ──
    # GMR scales the human about the world origin, so on elevated terrain the
    # scaled IK targets sink. Re-anchor them in Z at the support foot (per-foot
    # toe pairing + soft-min handover) and back-project the shift onto the GMR
    # input (δ = dz/s_root reproduces the same pre-IK target grounding).
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    ball_idx = {n: name_to_idx[n] for n in ("ball_l", "ball_r")
                if n in name_to_idx}

    # The GMR "scale human data" is the IK *input* skeleton (scaled about the
    # root before the IK solve). We always capture it here so it can be exported
    # ("pre-IK skeleton", export part 1) alongside the post-IK G1 qpos. When
    # method-3 grounding runs, the exported skeleton is shifted by the same
    # dz_target so it matches the grounded IK target the solver actually sees.
    scaled_skel_export: np.ndarray | None = None   # (F, N, 3) scaled IK input
    scaled_names_export: list[str] | None = None    # joint names for the above
    try:
        assert skel_kabsch is not None
        _scaled_tmp, _scaled_names_tmp, _retargeter_probe, scaled_toe_tmp = \
            scaled_human_skeleton(gmr_frames, h, root_xy,
                                  src_human=args.src_human)
        scaled_skel_export = _scaled_tmp
        scaled_names_export = list(_scaled_names_tmp)

        if args.no_scaled_ground:
            print(f"\n[S5 pre-IK grounding] DISABLED (--no-scaled-ground); "
                  f"exporting raw (un-grounded) scaled skeleton.")
        else:
            print(f"\n[S5 pre-IK grounding] Computing grounded scaled target "
                  f"(method 3) ...")
            have_blue = "ball_l" in ball_idx and "ball_r" in ball_idx
            red_toe_ok = bool(np.isfinite(scaled_toe_tmp).all())
            if have_blue and red_toe_ok:
                k = float(args.ground_softness)
                blue_l = skel_kabsch[:, ball_idx["ball_l"], 2]
                blue_r = skel_kabsch[:, ball_idx["ball_r"], 2]
                red_l, red_r = scaled_toe_tmp[:, 0, 2], scaled_toe_tmp[:, 1, 2]
                dz_target = support_soft_ground_dz(blue_l, blue_r,
                                                   red_l, red_r, k)
                s_root = float(_retargeter_probe.human_scale_table.get(
                    _retargeter_probe.human_root_name, 1.0))
                if abs(s_root) > 1e-8:
                    dz_input = dz_target / s_root
                    for fi, fr in enumerate(gmr_frames):
                        shift = float(dz_input[fi])
                        for name in fr:
                            fr[name][0][2] += shift
                    # Keep the exported pre-IK skeleton in sync with the grounded
                    # IK target (same dz applied in target/world space).
                    scaled_skel_export = _scaled_tmp.copy()
                    scaled_skel_export[:, :, 2] += dz_target[:, None]
                    print(f"  method 3 applied to IK input: k={k:g}, "
                          f"s_root={s_root:.6f}")
                    print(f"  target Δz: min={dz_target.min():.4f}  "
                          f"med={np.median(dz_target):.4f}  "
                          f"max={dz_target.max():.4f}  std={dz_target.std():.4f}")
                else:
                    print("  skipped: invalid s_root≈0, cannot back-project dz.")
            else:
                print("  skipped: need blue ball_l/r and a finite scaled toe.")
    except Exception as e:
        print(f"  [S5 pre-IK grounding] FAILED ({type(e).__name__}: {e}); skip.")

    # ── 4b) GMR retarget ──
    print(f"\n[GMR retarget]")
    qpos_arr, retargeter = retarget_to_g1(gmr_frames, actual_human_height=h,
                                          src_human=args.src_human)
    g1_model = retargeter.model
    qpos_arr[:, 0] += root_xy[:, 0]
    qpos_arr[:, 1] += root_xy[:, 1]
    print(f"  Done. qpos shape = {qpos_arr.shape}")

    # ── 5) Foot grounding on the real G1 (Root-Z toe-to-toe + per-foot) ──
    print(f"\n[S5 grounding]")
    try:
        assert skel_kabsch is not None
        qpos_arr, _g1_foot_geoms = apply_s5_grounding(
            qpos_arr, g1_model, skel_kabsch, ball_idx, n_frames,
            extra_drop=args.extra_drop,
            no_ball_align=args.no_ball_align,
            per_foot=args.per_foot_ground,
        )
    except Exception as e:
        print(f"  [S5 grounding] FAILED ({type(e).__name__}: {e}); skip.")

    # ── 6) Build the per-frame command / root array (Actor & user orientation,
    #         WASD inputs, desired body-frame velocity, etc.). This is loaded
    #         independently from the motion array and never overwrites the
    #         motion .npy — see save_command_array.                                ──
    print(f"\n[Command] Loading cmd fields from JSONL")
    cmd_arr = None
    look_fwd_xy = None
    try:
        # First pass: read look_yaw_deg for compute_look_fwd_xy (not in COMMAND_DTYPE).
        look_yaw_raw: list[float] = []
        with open(args.jsonl, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _row = json.loads(_line)
                look_yaw_raw.append(float((_row.get("cmd") or {}).get("lookY", 0.0)))
        # Second pass via load_gasp_command_data (all coord conversions applied).
        cmd_arr_full = load_gasp_command_data(args.jsonl)
        n_motion = qpos_arr.shape[0]
        if cmd_arr_full.shape[0] != n_motion:
            m = min(cmd_arr_full.shape[0], n_motion)
            print(f"  [Command] aligning by truncation: jsonl={cmd_arr_full.shape[0]} "
                  f"motion={n_motion} -> {m}")
            cmd_arr = cmd_arr_full[:m].copy()
            look_yaw_arr = np.asarray(look_yaw_raw[:m], dtype=np.float32)
        else:
            cmd_arr = cmd_arr_full.copy()
            look_yaw_arr = np.asarray(look_yaw_raw, dtype=np.float32)
        look_fwd_xy = compute_look_fwd_xy(look_yaw_arr, R_global)
        print(f"  [Command] frames={cmd_arr.shape[0]}  "
              f"look_fwd_xy mean|={float(np.linalg.norm(look_fwd_xy.mean(axis=0))):.3f}")
    except Exception as e:
        print(f"  [Command] FAILED ({type(e).__name__}: {e}); orientation arrows disabled.")
        cmd_arr = None

    # ── Export ──────────────────────────────────────────────────────────────
    # Three sibling files, all named after the input JSONL stem so a clip's
    # outputs travel together (e.g. WalkTurn_C_8_DjMGcE36_frames.jsonl →
    # WalkTurn_C_8_DjMGcE36_frames.npy / _cmd.npy / _scaled.npz):
    #   <stem>.npy        part 2: IK-optimised G1 joint info (qpos, (T, nq))
    #   <stem>_cmd.npy    part 3: per-frame command/root (structured COMMAND_DTYPE)
    #   <stem>_scaled.npz part 1: scaled pre-IK skeleton (IK input positions+names)
    # The base path defaults to the input JSONL location; --output-qpos overrides
    # it. The three files are kept strictly separate so downstream consumers can
    # load any one independently.
    if args.no_save:
        print(f"\n[Export] SKIPPED (--no-save).")
    else:
        if args.output_qpos:
            out = pathlib.Path(args.output_qpos)
        else:
            jp = pathlib.Path(args.jsonl)
            out = jp.with_suffix(".npy")    # <jsonl stem>.npy, alongside the JSONL
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[Export] base = {out}")

        # part 2: G1 qpos (post-IK)
        np.save(str(out), qpos_arr)
        print(f"  Saved qpos → {out}  (shape: {qpos_arr.shape})")

        # part 1: scaled pre-IK skeleton
        if scaled_skel_export is not None and scaled_names_export is not None:
            scaled_out = out.with_name(out.stem + "_scaled.npz")
            try:
                save_scaled_skeleton(scaled_out, scaled_skel_export,
                                     scaled_names_export)
            except Exception as e:
                print(f"  [Scaled-save] FAILED ({type(e).__name__}: {e})")
        else:
            print(f"  [Scaled-save] SKIPPED (scaled skeleton unavailable).")

        # part 3: command / root
        if cmd_arr is not None and look_fwd_xy is not None and not args.no_output_cmd:
            cmd_out = (
                pathlib.Path(args.output_cmd) if args.output_cmd
                else out.with_name(out.stem + "_cmd" + out.suffix)
            )
            try:
                save_command_array(cmd_out, cmd_arr, look_fwd_xy)
            except Exception as e:
                print(f"  [Command-save] FAILED ({type(e).__name__}: {e})")

    # ── 7) Visualize ──
    if args.no_visualize:
        return
    print(f"\n[Visualize]")
    # Resolve playback fps: explicit --fps wins; otherwise meta sample rate * speed.
    source_fps = 60.0
    try:
        with open(args.meta, "r") as f:
            source_fps = float(json.load(f).get("sample_rate_hz", 60.0))
    except Exception:
        pass
    if args.fps is not None:
        effective_fps = float(args.fps)
        print(f"  Playback fps = {effective_fps:.1f} (forced via --fps)")
    else:
        effective_fps = source_fps * float(args.speed)
        print(f"  Playback fps = {effective_fps:.1f}  "
              f"(source {source_fps:.1f} Hz * speed {args.speed:g})")
    # --full-trajectory wins over --traj-window: 0 means "all frames" inside the overlay.
    effective_window = 0 if args.full_trajectory else args.traj_window

    # Load + transform the scene terrain through the SAME pipeline as the motion.
    terrain = None
    if not args.no_terrain:
        terrain_path = _resolve_terrain_path(args)
        if terrain_path:
            try:
                terrain = load_terrain_world(
                    terrain_path, R_global, t_global,
                    ground_z0_cm=args.terrain_ground_z0,
                    data_scale=data_scale,
                )
            except Exception as e:
                print(f"  [Terrain] FAILED ({type(e).__name__}: {e}); "
                      f"visualizing without terrain.")

    visualize_g1(
        qpos_arr,
        fps=effective_fps,
        traj_window=effective_window,
        marker_step=args.marker_step,
        show_trajectory=not args.no_trajectory,
        look_fwd_xy=look_fwd_xy,
        des_vel_arr=cmd_arr["des_vel"] if cmd_arr is not None else None,
        move_input_arr=cmd_arr["move_input"] if cmd_arr is not None else None,
        show_orient_arrows=not args.no_orientation_arrows,
        terrain=terrain,
    )
    print("Done.")


if __name__ == "__main__":
    main()
