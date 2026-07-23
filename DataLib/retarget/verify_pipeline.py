r"""Unified step-by-step pipeline verification (viser / MuJoCo).

Requires schema ≥ 2.4 data (each joint carries wp/wq world coordinates).

Each --stage runs all stages up to (and including) the specified one:

    s1  : Read bone world positions (wp) directly from JSONL. UE world cm
          (displayed as m). No X↔Y swap, no Kabsch, no ball-align.
    s3  : s1 + S2 UE→MuJoCo (X↔Y + cm→m) + S3 Kabsch (R_global/t_global).
    s4  : s3 + S4 per-frame ball-Z shift (simulated, same formula as the
          real retarget code).
    s5  : s3 + S5 GMR retarget (human → G1 qpos, root-XY recentered) +
          S4 ball-align on the real G1 via MuJoCo FK.  The IK input uses the
          same pre-IK grounded scaled-human targets as s5.1 (method 3). Shows
          the G1 robot alongside the human skeleton and terrain.

    The retarget itself is two sub-steps: (1) GMR first SCALES the human data
    about the root and offsets it ("scale human data"); this scaled skeleton is
    the IK *target*.  (2) IK then optimises the G1 joint rotations to match it.
    These two sub-steps can be inspected separately:

    s5.1 : show the "scale human data" skeleton — the IK input, i.e. the
           scaled human joints BEFORE IK runs (red), overlaid with the original
           un-scaled Kabsch full-body skeleton (blue) for a before/after-scaling
           comparison.  No G1 robot is drawn.  Because GMR scales about the
           world origin, the raw scaled skeleton sinks on elevated terrain; by
           default it is re-grounded in Z at the support foot (method 3:
           per-foot toe pairing + soft-min handover, see --ground-softness /
           --no-scaled-ground) so its feet rest smoothly on the ground without
           the height jumps a naive lowest-foot min would produce.
    s5.2 : alias of s5 (same method-3 pre-IK grounded targets and retarget
           output). Shows G1 robot + human skeleton.

Viewer backends (--viewer):
    viser  : web 3D viewer (default), works for all stages.
    mujoco : native MuJoCo viewer (s5/s5.2) — G1 robot + terrain + human
             skeleton overlay.  SPACE = pause, LEFT/RIGHT = step frame.

Example (Windows PowerShell)
----------------------------
    cd D:\tool\ue5\UnrealProjects\GASP\Scripts

    # S1: raw UE world (no --config needed)
    python .\UEGMR\Retargeting\debug\verify_pipeline.py `
        --jsonl .\data\WalkTurn_C_8_Qn38i0BC_frames.jsonl `
        --meta  .\data\WalkTurn_C_8_Qn38i0BC_meta.json `
        --stage s1

    # S3: + MuJoCo + Kabsch
    python .\UEGMR\Retargeting\debug\verify_pipeline.py `
        --jsonl .\data\temp\WalkTurnCrouch_C_11_34cm5kTm_frames.jsonl `
        --meta  .\data\temp\WalkTurnCrouch_C_11_34cm5kTm_meta.json `
        --config .\tools\gasp_bvh_alignment.json `
        --stage s3

    # S5: full retarget + MuJoCo native viewer
    python .\UEGMR\Retargeting\debug\verify_pipeline.py `
        --jsonl .\data\WalkTurn_C_8_Qn38i0BC_frames.jsonl `
        --meta  .\data\WalkTurn_C_8_Qn38i0BC_meta.json `
        --config .\tools\gasp_bvh_alignment.json `
        --stage s5 --viewer mujoco  --per-foot-ground

    # s5.1: 只看 IK 输入（缩放后的人体骨架）
    python .\UEGMR\Retargeting\debug\verify_pipeline.py `
        --jsonl .\data\WalkTurn_C_8_DjMGcE36_frames.jsonl `
        --meta  .\data\WalkTurn_C_8_DjMGcE36_meta.json `
        --config .\tools\gasp_bvh_alignment_g1_height.json `
        --src-human bvh_ue5_g1scale `
        --stage s5.1

    # s5.2: 看重映射结果
    python .\UEGMR\Retargeting\debug\verify_pipeline.py `
        --jsonl .\data\WalkTurn_C_8_DjMGcE36_frames.jsonl `
        --meta  .\data\WalkTurn_C_8_DjMGcE36_meta.json `
        --config .\tools\gasp_bvh_alignment_g1_height.json `
        --src-human bvh_ue5_g1scale `
        --stage s5.2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import viser
except ImportError:  # optional web viewer; --viewer mujoco works without it
    viser = None

HERE = pathlib.Path(__file__).resolve().parent

# Force UTF-8 stdout/stderr so non-ASCII help/prints (arrows, check-marks) don't
# crash on Windows GBK consoles. No-op on streams that don't support reconfigure.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Explicit skeleton connectivity (UE5 mannequin main chain only). Matches
# vis_skeleton_compare.BONE_CONNECTIONS. We draw bones from THIS curated list
# rather than meta's parent_indices so fingers / twist / IK / weapon bones are
# never drawn — those clutter the view and connect to misleading joints.
BONE_CONNECTIONS = [
    ("pelvis", "spine_01"), ("spine_01", "spine_02"), ("spine_02", "spine_03"),
    ("spine_03", "spine_04"), ("spine_04", "spine_05"),
    ("spine_05", "neck_01"), ("neck_01", "neck_02"), ("neck_02", "head"),
    ("spine_05", "clavicle_l"), ("clavicle_l", "upperarm_l"),
    ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
    ("spine_05", "clavicle_r"), ("clavicle_r", "upperarm_r"),
    ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"),
    ("pelvis", "thigh_l"), ("thigh_l", "calf_l"),
    ("calf_l", "foot_l"), ("foot_l", "ball_l"),
    ("pelvis", "thigh_r"), ("thigh_r", "calf_r"),
    ("calf_r", "foot_r"), ("foot_r", "ball_r"),
]

# Connectivity for the "scale human data" skeleton (--stage s5.1). GMR's
# scale_human_data only emits the bodies present in the IK config's
# human_scale_table (a reduced set), and the ankle target is "LeftFootMod" /
# "RightFootMod" rather than foot_*/ball_*. Edges are resolved by NAME, so any
# body missing from scaled_human_data is simply skipped.
SCALED_BONE_CONNECTIONS = [
    ("pelvis", "spine_05"),
    ("spine_05", "upperarm_l"), ("upperarm_l", "lowerarm_l"),
    ("lowerarm_l", "hand_l"),
    ("spine_05", "upperarm_r"), ("upperarm_r", "lowerarm_r"),
    ("lowerarm_r", "hand_r"),
    ("pelvis", "thigh_l"), ("thigh_l", "calf_l"), ("calf_l", "LeftFootMod"),
    ("pelvis", "thigh_r"), ("thigh_r", "calf_r"), ("calf_r", "RightFootMod"),
]

# Connectivity for the G1 robot FK skeleton, drawn as a yellow overlay so the
# user can intuitively compare the retargeted robot pose against the scaled
# human data (red). We connect the ANATOMICAL links (hip_pitch / shoulder_pitch
# = the real hip/shoulder joints) rather than the IK-matched *_yaw_link bodies,
# because the yaw links sit partway down the limb (chain order is
# pitch→roll→yaw→knee/elbow) and would make the stick figure look compressed and
# not overlay the robot mesh. Every body here is a real G1 link, so the skeleton
# co-locates exactly with the rendered robot.
G1_FK_BONE_CONNECTIONS = [
    ("pelvis", "torso_link"), ("torso_link", "head_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("left_ankle_roll_link", "left_toe_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("right_ankle_roll_link", "right_toe_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_alignment_config(path: str):
    cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    g = cfg["global"]
    Rg = R.from_quat(g["rotation_quat_wxyz"], scalar_first=True).as_matrix()
    tg = np.array(g["translation_xyz"], dtype=np.float64)
    return Rg, tg, cfg


def load_meta(meta_path: str):
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


# ── Transforms ───────────────────────────────────────────────────────────────

def skeleton_world_direct(frame: dict) -> np.ndarray:
    """Read bone world positions (wp) directly from frame (schema ≥ 2.4).
    Returns (B, 3) in UE world cm."""
    return np.asarray([j["wp"] for j in frame["joints"]], dtype=np.float64)


def ue_to_mujoco(p_cm: np.ndarray) -> np.ndarray:
    """S3: X<->Y swap + cm → m."""
    out = np.empty_like(p_cm)
    out[..., 0] = p_cm[..., 1]
    out[..., 1] = p_cm[..., 0]
    out[..., 2] = p_cm[..., 2]
    return out * 0.01


def apply_kabsch(p: np.ndarray, Rg: np.ndarray, tg: np.ndarray) -> np.ndarray:
    """S4: Kabsch global alignment."""
    return (Rg @ p.T).T + tg


# ── GMR Retarget ─────────────────────────────────────────────────────────────

def _ensure_gmr_on_path():
    candidates = [
        HERE.parent / "gmr",                          # release: DataLib/gmr
        HERE.parent / "GMR",                          # debug/../GMR  (unlikely)
        HERE.parent.parent / "UEGMR" / "GMR",        # Scripts/UEGMR/GMR
        HERE.parent.parent.parent / "UEGMR" / "GMR", # extra level
    ]
    for root in candidates:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _load_gasp_data_for_retarget(jsonl_path: str, meta_path: str,
                                 preserve_world_z: bool, ground_z_cm: float):
    """Load GASP data via the verified vis_skeleton_compare loader."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from vis_skeleton_compare import load_gasp_data  # noqa: E402
    return load_gasp_data(jsonl_path, meta_path,
                          preserve_world_z=preserve_world_z,
                          ground_z_cm=ground_z_cm)


def build_gmr_frames(gasp_pos_mj, gasp_rot_all, gasp_names,
                     Rg, tg, delta, root_name="pelvis",
                     recenter_root_xy=True):
    """Build GMR input frames.

    GMR's scale_human_data scales the root world position about the world
    ORIGIN (scaled_root = scale * root_pos, scale≈0.76). After Kabsch the
    human sits ~8 m from the origin, so this drags the robot ~2 m toward the
    origin, leaving the retargeted G1 offset from the input skeleton.

    To make the robot overlay the human, we translate every joint by the
    per-frame root XY so the root sits at XY=0 before GMR (Z untouched, kept
    for the natural crouch scaling + later ball-align). The removed offset is
    returned so the caller can add it back to qpos[:2] after retargeting.
    """
    identity3 = np.eye(3)
    n_frames = gasp_pos_mj.shape[0]
    frames = []
    root_xy = np.zeros((n_frames, 2), dtype=np.float64)
    for fi in range(n_frames):
        result = {}
        for gi, name in enumerate(gasp_names):
            pos_aligned = Rg @ gasp_pos_mj[fi, gi] + tg
            rot_aligned = Rg @ gasp_rot_all[fi][gi]
            D = delta.get(name, identity3)
            R_eq = rot_aligned @ D
            quat = R.from_matrix(R_eq).as_quat(scalar_first=True)
            result[name] = [pos_aligned, quat]

        if recenter_root_xy and root_name in result:
            off = result[root_name][0][:2].copy()
            root_xy[fi] = off
            for name in result:
                result[name][0][:2] -= off

        # Foot IK target: LeftFootMod = foot_* position (ankle) + ball_*
        # orientation. Foot pitch is driven by this ankle orientation task.
        if "foot_l" in result and "ball_l" in result:
            result["LeftFootMod"] = [result["foot_l"][0].copy(),
                                     result["ball_l"][1].copy()]
        if "foot_r" in result and "ball_r" in result:
            result["RightFootMod"] = [result["foot_r"][0].copy(),
                                      result["ball_r"][1].copy()]
        frames.append(result)
    return frames, root_xy


def estimate_human_height(gmr_frames):
    if not gmr_frames:
        return 1.7
    head = gmr_frames[0].get("head", [np.array([0, 0, 1.7])])[0]
    fl = gmr_frames[0].get("foot_l", [np.array([0, 0, 0])])[0]
    fr = gmr_frames[0].get("foot_r", [np.array([0, 0, 0])])[0]
    return float(head[2] - min(fl[2], fr[2]) + 0.09)


def retarget_to_g1(gmr_frames, human_height, warmup=20,
                   src_human="bvh_ue5_native"):
    _ensure_gmr_on_path()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
    retargeter = GMR(src_human=src_human, tgt_robot="unitree_g1",
                     actual_human_height=human_height)
    if warmup > 0 and gmr_frames:
        for _ in range(warmup):
            retargeter.retarget(gmr_frames[0])
    qpos_list = []
    for frame in gmr_frames:
        qpos_list.append(retargeter.retarget(frame).copy())
    return np.stack(qpos_list, axis=0), retargeter


def scaled_human_skeleton(gmr_frames, human_height, root_xy,
                          src_human="bvh_ue5_native"):
    """Extract the GMR "scale human data" — the IK *input* skeleton (--stage
    s5.1).

    GMR's pipeline first runs update_targets(), which scales the human joints
    about the root + applies the IK offsets, and stores the result in
    ``retargeter.scaled_human_data`` (this is exactly what the IK solver tries
    to match). We run only update_targets() per frame (no IK solve) and read
    those scaled positions back out.

    The GMR frames had their root XY recentered to the origin (see
    build_gmr_frames); we add the per-frame root XY back so the scaled skeleton
    lands in the same world frame as the Kabsch skeleton / terrain / G1.

    The scaled skeleton has NO toe (ball) joint — GMR's scale table only carries
    the ankle (LeftFootMod/RightFootMod). For toe-to-toe grounding we also need
    the scaled toe, so we scale the original ball_* about the root with the SAME
    leg/foot factor GMR uses for the ankle (human_scale_table["*FootMod"]),
    exactly reproducing GMR's per-joint scaling:
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

        # Reproduce GMR's scaling for the toe (ball_*), which is not in the
        # scale table. update_targets does not mutate the frame's positions, so
        # frame[root_name]/frame[ball_*] still hold the original (pre-scale) pos.
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
    """Method 3 grounding dz from per-foot pairing + soft-min support blend."""
    # per-foot residual (align red toe i -> blue ball i)
    res_l = blue_l_z - red_l_z
    res_r = blue_r_z - red_r_z
    # soft-min weights over red toe height (lower -> bigger weight).
    zmin = np.minimum(red_l_z, red_r_z)
    e_l = np.exp(-k * (red_l_z - zmin))
    e_r = np.exp(-k * (red_r_z - zmin))
    w_l = e_l / (e_l + e_r)
    w_r = 1.0 - w_l
    return w_l * res_l + w_r * res_r


def g1_fk_foot_z(model, qpos_arr, front_only=True):
    """Per-frame min foot contact sphere bottom Z via FK."""
    import mujoco as mj
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
    import mujoco as mj
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
    import mujoco as mj
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
    import mujoco as mj
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
    import mujoco as mj
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


# ── G1 Mesh Visualization ───────────────────────────────────────────────────

def _make_box_mesh(hx, hy, hz):
    v = np.array([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                  [-hx,-hy, hz],[hx,-hy, hz],[hx,hy, hz],[-hx,hy, hz]],
                 dtype=np.float32)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [0,4,7],[0,7,3],[1,2,6],[1,6,5]], dtype=np.int32)
    return v, f


def _make_sphere_mesh(radius, rings=8, sectors=16):
    verts = [[0, 0, radius]]
    for i in range(1, rings):
        phi = np.pi * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(sectors):
            th = 2 * np.pi * j / sectors
            verts.append([radius*sp*np.cos(th), radius*sp*np.sin(th), radius*cp])
    verts.append([0, 0, -radius])
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for j in range(sectors):
        faces.append([0, 1 + j, 1 + (j + 1) % sectors])
    for i in range(rings - 2):
        for j in range(sectors):
            a = 1 + i * sectors + j
            b = 1 + (i + 1) * sectors + j
            c = 1 + i * sectors + (j + 1) % sectors
            d = 1 + (i + 1) * sectors + (j + 1) % sectors
            faces.append([a, b, c])
            faces.append([c, b, d])
    south = len(verts) - 1
    base = 1 + (rings - 2) * sectors
    for j in range(sectors):
        faces.append([south, base + (j + 1) % sectors, base + j])
    return verts, np.array(faces, dtype=np.int32)


def _make_capsule_mesh(radius, half_len, rings=5, sectors=12):
    """Capsule along Z: cylinder [-half_len, +half_len] with hemisphere caps."""
    verts = [[0, 0, half_len + radius]]  # top pole
    for i in range(1, rings + 1):
        phi = np.pi / 2 * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        z = half_len + radius * cp
        r = radius * sp
        for j in range(sectors):
            th = 2 * np.pi * j / sectors
            verts.append([r * np.cos(th), r * np.sin(th), z])
    top_eq = 1 + (rings - 1) * sectors
    bot_eq = len(verts)
    for i in range(rings):
        phi = np.pi / 2 * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        z = -half_len - radius * sp
        r = radius * cp
        for j in range(sectors):
            th = 2 * np.pi * j / sectors
            verts.append([r * np.cos(th), r * np.sin(th), z])
    verts.append([0, 0, -half_len - radius])  # bottom pole
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for j in range(sectors):
        faces.append([0, 1 + (j + 1) % sectors, 1 + j])
    for i in range(rings - 1):
        for j in range(sectors):
            a, b = 1 + i * sectors + j, 1 + (i + 1) * sectors + j
            c, d = 1 + i * sectors + (j + 1) % sectors, 1 + (i + 1) * sectors + (j + 1) % sectors
            faces.append([a, b, d])
            faces.append([a, d, c])
    for j in range(sectors):
        a, b = top_eq + j, bot_eq + j
        c, d = top_eq + (j + 1) % sectors, bot_eq + (j + 1) % sectors
        faces.append([a, b, d])
        faces.append([a, d, c])
    for i in range(rings - 1):
        for j in range(sectors):
            a, b = bot_eq + i * sectors + j, bot_eq + (i + 1) * sectors + j
            c, d = bot_eq + i * sectors + (j + 1) % sectors, bot_eq + (i + 1) * sectors + (j + 1) % sectors
            faces.append([a, b, d])
            faces.append([a, d, c])
    south = len(verts) - 1
    last = bot_eq + (rings - 1) * sectors
    for j in range(sectors):
        faces.append([south, last + j, last + (j + 1) % sectors])
    return verts, np.array(faces, dtype=np.int32)


def _geom_local_mesh(model, geom_id):
    """Return (verts, faces) in geom's local frame, or None if unsupported."""
    import mujoco as mj
    gt = int(model.geom_type[geom_id])
    sz = model.geom_size[geom_id]
    if gt == mj.mjtGeom.mjGEOM_BOX:
        return _make_box_mesh(float(sz[0]), float(sz[1]), float(sz[2]))
    elif gt == mj.mjtGeom.mjGEOM_SPHERE:
        return _make_sphere_mesh(float(sz[0]))
    elif gt == mj.mjtGeom.mjGEOM_CAPSULE:
        return _make_capsule_mesh(float(sz[0]), float(sz[1]))
    elif gt == mj.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[geom_id])
        if mid < 0:
            return None
        vs, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fs, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        return (model.mesh_vert[vs:vs + vn].copy().astype(np.float32),
                model.mesh_face[fs:fs + fn].copy().astype(np.int32))
    return None


def setup_g1_visual(model, server, prefix="/g1"):
    """Create viser mesh nodes for every renderable MuJoCo geom.
    Returns list of (handle, geom_id)."""
    import mujoco as mj
    handles = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == 0:
            continue
        if int(model.geom_type[gid]) == mj.mjtGeom.mjGEOM_PLANE:
            continue
        result = _geom_local_mesh(model, gid)
        if result is None:
            continue
        verts, faces = result
        rgba = model.geom_rgba[gid]
        if float(rgba[3]) < 0.01:
            continue
        color = tuple(int(c * 255) for c in rgba[:3])
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
        h = server.scene.add_mesh_simple(
            f"{prefix}/{name}", verts, faces,
            color=color, opacity=float(rgba[3]),
            side="double", flat_shading=False)
        handles.append((h, gid))
    return handles


def update_g1_visual(data, handles, off_xy, visible=True):
    """Update viser mesh transforms from FK data."""
    for h, gid in handles:
        if not visible:
            h.visible = False
            continue
        h.visible = True
        pos = data.geom_xpos[gid].copy()
        pos[0] -= off_xy[0]
        pos[1] -= off_xy[1]
        h.position = pos.astype(np.float64)
        mat = data.geom_xmat[gid].reshape(3, 3)
        h.wxyz = R.from_matrix(mat).as_quat(scalar_first=True).astype(np.float64)


# ── MuJoCo native viewer ─────────────────────────────────────────────────────

_MJ_WRAPPER_TMPL = """<mujoco>
  <include file="{g1_xml_name}"/>
  <visual>
    <map znear="0.001" zfar="100"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.52 0.55 0.60" diffuse="0.30 0.30 0.32"
               specular="0.06 0.06 0.06"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".55 .55 .55"
             rgb2=".45 .45 .45" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.04"/>
{terrain_asset}  </asset>
  <worldbody>
    <geom name="viz_floor" type="plane" size="20 20 0.01" material="grid_mat"/>
{terrain_geom}  </worldbody>
</mujoco>
"""


def _build_terrain_xml_mj(terrain, xy_offset):
    """Bake transformed terrain (MuJoCo m) into inline <mesh>/<geom> XML."""
    if terrain is None:
        return "", ""
    verts, faces = terrain
    v = verts.astype(np.float64).copy()
    v[:, 0] -= xy_offset[0]
    v[:, 1] -= xy_offset[1]
    vtx_str = " ".join(f"{x:.5f}" for x in v.reshape(-1))
    face_str = " ".join(str(int(i)) for i in faces.reshape(-1))
    asset = (f'    <mesh name="ue_terrain" vertex="{vtx_str}" face="{face_str}"/>\n'
             '    <material name="ue_terrain_mat" rgba="0.42 0.66 0.92 1.0" '
             'specular="0.08" shininess="0.12" reflectance="0.0"/>\n')
    geom = ('    <geom name="ue_terrain_geom" type="mesh" mesh="ue_terrain" '
            'contype="0" conaffinity="0" group="2" material="ue_terrain_mat"/>\n')
    return asset, geom


def _draw_skeleton_user_scn(scn, joints_world, bone_pairs, foot_idx,
                            skip_set=None,
                            joint_color=(0.86, 0.12, 0.24, 1.0),
                            foot_color=(0.16, 0.86, 0.24, 1.0),
                            bone_color=(0.06, 0.06, 0.06, 1.0),
                            joint_r=0.018, foot_r=0.03, bone_r=0.008):
    """Append human skeleton (spheres + bone capsules) to viewer.user_scn."""
    import mujoco as mj
    foot_set = set(foot_idx or [])
    hide = skip_set or set()

    def _add_sphere(p, r, color):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE,
                        np.array([r, 0, 0], dtype=np.float64),
                        np.asarray(p, dtype=np.float64),
                        np.eye(3).reshape(-1),
                        np.asarray(color, dtype=np.float32))
        scn.ngeom += 1

    for i, p in enumerate(joints_world):
        if i in hide:
            continue
        if i in foot_set:
            _add_sphere(p, foot_r, foot_color)
        else:
            _add_sphere(p, joint_r, joint_color)

    for a, b in bone_pairs:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).reshape(-1),
                        np.asarray(bone_color, dtype=np.float32))
        mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, bone_r,
                         np.asarray(joints_world[a], dtype=np.float64),
                         np.asarray(joints_world[b], dtype=np.float64))
        scn.ngeom += 1


def _draw_ground_markers(scn, x, y, dist_human, dist_robot, dz):
    """Append grounding markers to user_scn: a green sphere at the human ground
    (min ball_z - offset_human), a blue sphere at the robot ground (min toe_z -
    offset_robot), and a vertical capsule whose length = Δz. Numeric values are
    attached as geom labels."""
    import mujoco as mj

    def _sphere(p, r, color, label):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE,
                        np.array([r, 0, 0], dtype=np.float64),
                        np.asarray(p, dtype=np.float64),
                        np.eye(3).reshape(-1),
                        np.asarray(color, dtype=np.float32))
        g.label = label.encode("utf-8")[:99]
        scn.ngeom += 1

    _sphere([x, y, dist_human], 0.025, (0.16, 0.86, 0.24, 1.0),
            f"human(min ball-off) z={dist_human:+.3f}")
    _sphere([x, y, dist_robot], 0.025, (0.20, 0.45, 0.95, 1.0),
            f"robot(min toe-off) z={dist_robot:+.3f}  dz={dz:+.3f}")
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).reshape(-1),
                        np.array([0.95, 0.7, 0.1, 1.0], dtype=np.float32))
        mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.006,
                         np.array([x, y, dist_human], dtype=np.float64),
                         np.array([x, y, dist_robot], dtype=np.float64))
        scn.ngeom += 1


def run_mujoco_viewer(args, n_frames, qpos_arr, g1_model, g1_foot_geoms,
                      skel_all, bone_pairs, foot_idx, terrain_raw, off_xy,
                      ik_set=None):
    """Native MuJoCo viewer: G1 robot + terrain + human skeleton overlay."""
    import mujoco as mj
    import mujoco.viewer
    from general_motion_retargeting.params import ROBOT_XML_DICT  # type: ignore

    g1_xml = pathlib.Path(str(ROBOT_XML_DICT["unitree_g1"]))
    wrapper_path = g1_xml.parent / "_verify_pipeline_wrapper.xml"

    terrain_asset, terrain_geom = _build_terrain_xml_mj(terrain_raw, off_xy)
    wrapper_path.write_text(
        _MJ_WRAPPER_TMPL.format(g1_xml_name=g1_xml.name,
                                terrain_asset=terrain_asset,
                                terrain_geom=terrain_geom),
        encoding="utf-8")
    try:
        model = mj.MjModel.from_xml_path(str(wrapper_path))
    except Exception as e:
        print(f"  [MuJoCo] terrain compile failed ({type(e).__name__}: {e}); "
              f"loading without terrain.")
        wrapper_path.write_text(
            _MJ_WRAPPER_TMPL.format(g1_xml_name=g1_xml.name,
                                    terrain_asset="", terrain_geom=""),
            encoding="utf-8")
        model = mj.MjModel.from_xml_path(str(wrapper_path))
    finally:
        wrapper_path.unlink(missing_ok=True)

    data = mj.MjData(model)

    # qpos with XY recentered to keep robot near origin (matches terrain shift)
    qpos_view = qpos_arr.copy()
    qpos_view[:, 0] -= off_xy[0]
    qpos_view[:, 1] -= off_xy[1]

    # human skeleton recentered the same way
    skel_view_all = skel_all.copy()
    skel_view_all[:, :, 0] -= off_xy[0]
    skel_view_all[:, :, 1] -= off_xy[1]

    # Grounding readout: human/robot ground levels + Δz, shown live as markers.
    offset_human = args.extra_drop
    offset_robot = g1_toe_standing_offset(model)
    toe_l_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_toe_link")
    toe_r_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_toe_link")
    ball_view_idx = list(foot_idx or [])

    print(f"\n[MuJoCo] Launching native viewer — {n_frames} frames.")
    print("  Red/green spheres + black capsules = human skeleton (GMR input).")
    print("  Robot mesh = G1 retarget output. Close the window to exit.")
    print(f"  Grounding markers: GREEN=human ground (min ball z-{offset_human:.3f}), "
          f"BLUE=robot ground (min toe z-{offset_robot:.3f}), yellow bar=Δz.")

    state = {"fi": 0, "paused": False, "fps": 30.0}

    def key_cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch == ' ':
            state["paused"] = not state["paused"]
        elif keycode == 262:  # right arrow
            state["fi"] = (state["fi"] + 1) % n_frames
        elif keycode == 263:  # left arrow
            state["fi"] = (state["fi"] - 1) % n_frames

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_cb,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        while viewer.is_running():
            fi = state["fi"]
            data.qpos[:] = qpos_view[fi]
            mj.mj_forward(model, data)

            viewer.user_scn.ngeom = 0
            _draw_skeleton_user_scn(viewer.user_scn, skel_view_all[fi],
                                    bone_pairs, foot_idx,
                                    skip_set=ik_set)

            # Live grounding readout (matches the displayed pose).
            if ball_view_idx and toe_l_id >= 0 and toe_r_id >= 0:
                ball_zs = skel_view_all[fi, ball_view_idx, 2]
                dist_human = float(ball_zs.min()) - offset_human
                tlz = float(data.xpos[toe_l_id, 2])
                trz = float(data.xpos[toe_r_id, 2])
                if tlz <= trz:
                    toe_z, txy = tlz, data.xpos[toe_l_id, :2]
                else:
                    toe_z, txy = trz, data.xpos[toe_r_id, :2]
                dist_robot = toe_z - offset_robot
                _draw_ground_markers(viewer.user_scn, float(txy[0]),
                                     float(txy[1]), dist_human, dist_robot,
                                     dist_human - dist_robot)
            viewer.sync()

            if not state["paused"]:
                state["fi"] = (fi + 1) % n_frames
                time.sleep(1.0 / max(1.0, state["fps"]))
            else:
                time.sleep(0.03)


# ── Terrain ──────────────────────────────────────────────────────────────────

def load_terrain_instances(terrain_path: str, gz0_cm: float,
                           skip_engine: bool = True):
    """Load terrain JSON → UE world (cm) with ground z0 subtracted."""
    data = json.loads(pathlib.Path(terrain_path).read_text(encoding="utf-8"))
    meshes, instances = data.get("meshes", []), data.get("instances", [])
    all_v, all_f, voff, skipped = [], [], 0, 0
    for inst in instances:
        mi = int(inst.get("mesh", -1))
        if mi < 0 or mi >= len(meshes):
            continue
        mesh = meshes[mi]
        if skip_engine and str(mesh.get("asset_path", "")).startswith("/Engine/"):
            skipped += 1
            continue
        V = np.asarray(mesh.get("vertices", []), dtype=np.float64).reshape(-1, 3)
        idx = np.asarray(mesh.get("indices", []), dtype=np.int64)
        if V.size == 0 or idx.size < 3:
            continue
        F = idx.reshape(-1, 3)
        loc = np.asarray(inst["location"], dtype=np.float64)
        Ri = R.from_quat(np.asarray(inst["rotation_quat_xyzw"],
                                     dtype=np.float64)).as_matrix()
        scl = np.asarray(inst["scale"], dtype=np.float64)
        Vw = (Ri @ (V * scl).T).T + loc
        Vw[:, 2] -= gz0_cm
        all_v.append(Vw)
        all_f.append(F + voff)
        voff += V.shape[0]
    if skipped:
        print(f"  [Terrain] skipped {skipped} /Engine/ instances")
    if not all_v:
        return None
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


def resolve_terrain_path(args) -> str | None:
    if args.terrain:
        return args.terrain
    ref = None
    try:
        ref = json.loads(pathlib.Path(args.meta).read_text(
            encoding="utf-8")).get("terrain_ref")
    except Exception:
        pass
    if not ref:
        print("  [Terrain] meta has no terrain_ref; use --terrain.")
        return None
    meta_dir = pathlib.Path(args.meta).parent
    cands = []
    if getattr(args, "terrain_dir", None):
        cands.append(pathlib.Path(args.terrain_dir) / ref)
    cands.append(meta_dir / "terrain" / ref)
    cands.append(meta_dir / ref)
    for c in cands:
        if c.is_file():
            return str(c)
    print(f"  [Terrain] could not find {ref} "
          f"(looked in {cands})")
    return None


# ── Visualisation ────────────────────────────────────────────────────────────

def main():
    # Line-buffer stdout so progress / grounding prints flush in real time even
    # when the (blocking) MuJoCo viewer opens or stdout is redirected to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--config", default=None,
                    help="gasp_bvh_alignment.json. Required for kabsch / ball-align.")
    ap.add_argument("--terrain", default=None,
                    help="Explicit path to a terrain_<hash>.json. Default: "
                         "resolved from the recording meta's terrain_ref, "
                         "searched in --terrain-dir then next to --meta.")
    ap.add_argument("--terrain-dir", default=None,
                    help="Directory holding terrain_<hash>.json files. "
                         "Default: a `terrain` folder next to --meta, then "
                         "--meta's own folder.")
    ap.add_argument("--stage",
                    choices=["s1", "s3", "s4", "s5", "s5.1", "s5.2"],
                    default="s5",
                    help="Run stages cumulatively. "
                         "s1=UE world (wp direct), s3=+MuJoCo+Kabsch, "
                         "s4=+ball-align(sim), s5=+GMR retarget with "
                         "pre-IK grounded scaled targets (default). "
                         "s5.1=show only the scaled human data (IK input) "
                         "skeleton; s5.2=same as s5.")
    ap.add_argument("--src-human", type=str, default="bvh_ue5_g1scale",
                    choices=["bvh_ue5_native", "bvh_ue5_g1scale"],
                    help="GMR source skeleton key. Use 'bvh_ue5_native' for "
                         "a 1.75 m human character, or "
                         "'bvh_ue5_g1scale' for a character scaled to G1 (default) "
                         "height (~1.32 m).")
    ap.add_argument("--world-z0", type=float, default=86.0,
                    help="(Deprecated — positions now come from wp, not "
                         "load_gasp_data, so this value is unused.)")
    ap.add_argument("--terrain-ground-z0", type=float, default=0.0)
    ap.add_argument("--extra-drop", type=float, default=0.031,
                    help="offset_human: human toe joint (ball) → ground (m). "
                         "Default 0.031 (3.1 cm).")
    ap.add_argument("--no-scaled-ground", action="store_true",
                    help="(s5/s5.1/s5.2) disable method-3 support-foot grounding "
                         "for scaled human targets. In s5.1 this shows raw "
                         "scaled data; in s5/s5.2 this disables pre-IK target "
                         "grounding.")
    ap.add_argument("--ground-softness", type=float, default=50.0,
                    help="(s5/s5.1/s5.2, method 3) soft-min sharpness k (1/m) for the "
                         "support-foot grounding blend. Larger = closer to a "
                         "hard 'lowest foot' pick (sharper handover); smaller = "
                         "softer/smoother double-support transition. The blend "
                         "band width is ~1/k m. Default 50 (~2 cm band).")
    ap.add_argument("--no-ball-align", action="store_true",
                    help="Disable the S4 ball-align Z shift on the retargeted "
                         "G1 (retarget stage). Use to inspect raw GMR foot IK.")
    ap.add_argument("--per-foot-ground", action="store_true",
                    help="After ball-align, independently bend each leg so each "
                         "foot lands on its own target ground (handles stairs / "
                         "feet at different heights).")
    ap.add_argument("--no-skip-engine", action="store_true")
    ap.add_argument("--joint-mapping", type=str, default=None,
                    help="joint_mapping.json for error-line robot↔human pairing. "
                         "Default: DataLib/retarget/joint_mapping.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--viewer", choices=["viser", "mujoco"], default="viser",
                    help="Visualisation backend. 'mujoco' opens the native "
                         "MuJoCo viewer with the G1 robot + terrain (retarget "
                         "stage only); 'viser' is the web 3D viewer (default).")
    args = ap.parse_args()

    # stage token may be "1"/"3"/"4"/"5"/"5.1"/"5.2"
    stage_num = int(float(args.stage[1:]))   # 1, 3, 4, or 5
    scaled_only = (args.stage == "s5.1")     # s5.1: show IK-input scaled human
    need_kabsch = stage_num >= 3

    Rg = tg = cfg = None
    delta = {}
    if need_kabsch:
        if not args.config:
            ap.error("--config is required for stages s3 / s4 / s5.")
        Rg, tg, cfg = load_alignment_config(args.config)
        delta = {
            name: R.from_quat(q, scalar_first=True).as_matrix()
            for name, q in cfg.get("per_bone_delta_quat_wxyz", {}).items()
        }
        cm = cfg.get("calib_meta", {})
        print(f"[Config] R_global ang={cm.get('global_rotation_angle_deg',0):.2f}°  "
              f"t_global={tg}")

    bone_names, _parents = load_meta(args.meta)
    frames = load_frames(args.jsonl)
    n_frames = len(frames)
    print(f"[Human] {n_frames} frames, {len(bone_names)} bones, "
          f"stage={args.stage}")

    # Build the skeleton from the explicit BONE_CONNECTIONS list (main chain
    # only). Bone edges are resolved by NAME → index, so the connectivity is
    # always correct regardless of meta's bone ordering, and fingers / twist /
    # IK / weapon bones are simply never referenced.
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    bone_pairs = np.array(
        [(name_to_idx[a], name_to_idx[b]) for a, b in BONE_CONNECTIONS
         if a in name_to_idx and b in name_to_idx],
        dtype=np.int64)

    # Joints that are actually part of the drawn skeleton (endpoints of edges).
    vis_names = {n for e in BONE_CONNECTIONS for n in e}
    vis_idx = sorted(name_to_idx[n] for n in vis_names if n in name_to_idx)
    # Everything else is hidden in the MuJoCo overlay (skip_set).
    hide_set = set(range(len(bone_names))) - set(vis_idx)

    foot_idx = [i for i, n in enumerate(bone_names)
                if n in ("ball_l", "ball_r")]
    ball_idx = {n: i for i, n in enumerate(bone_names)
                if n in ("ball_l", "ball_r")}

    # ── Build all skeletons per stage ──
    skel_all = np.zeros((n_frames, len(bone_names), 3), dtype=np.float64)

    # retarget-specific data (filled only when stage == retarget)
    qpos_arr = None
    g1_model = None
    g1_foot_geoms = []       # geom ids for foot Z readout

    # G1 FK skeleton (filled in s5/s5.2). Two sets:
    #  (1) anatomical stick figure that overlays the robot mesh (yellow)
    g1_fk_skel = None        # (F, N_body, 3) world-coord FK positions
    g1_fk_body_names = None  # list of N_body robot body names
    g1_fk_pairs = None       # (M, 2) bone connectivity indices for the figure
    #  (2) IK-matched body positions, paired to red targets for error lines
    g1_ik_fk_pts = None      # (F, K, 3) FK positions of IK-matched bodies
    g1_ik_human_names = None # list of K human joint names (one per IK body)
    g1_error_pairs = None    # (K, 2) → (ik_idx, skel_all_idx) for error lines

    # s5.1 overlay: original (un-scaled) Kabsch full-body skeleton (口径 B,
    # BONE_CONNECTIONS) drawn alongside the scaled IK-target skeleton so the
    # effect of GMR's root-relative scaling is visible. Filled only in s5.1.
    overlay_skel = None
    overlay_pairs = None
    overlay_vis_idx = None
    overlay_foot_idx = None
    # s5.1: scaled red toe (ball) world positions, (F, 2, 3) = [left, right].
    toe_world = None

    # S1: read bone world positions directly from wp (schema ≥ 2.4)
    skel_ue = np.zeros_like(skel_all)
    for fi, fr in enumerate(frames):
        skel_ue[fi] = skeleton_world_direct(fr)

    if stage_num <= 1:
        skel_all = skel_ue
        unit_label = "UE cm"
        scale = 0.01          # cm→m for viewer comfort
    else:
        # S2: UE → MuJoCo (X↔Y + cm→m)
        skel_mj = np.zeros_like(skel_all)
        for fi in range(n_frames):
            skel_mj[fi] = ue_to_mujoco(skel_ue[fi])
        # S3: Kabsch
        skel_kabsch = np.zeros_like(skel_all)
        for fi in range(n_frames):
            skel_kabsch[fi] = apply_kabsch(skel_mj[fi], Rg, tg)

        if stage_num == 3:
            skel_all = skel_kabsch
        elif stage_num == 4:
            # S4: ball-align simulation on the human skeleton
            skel_all = skel_kabsch.copy()
            if "ball_l" in ball_idx and "ball_r" in ball_idx:
                bl, br = ball_idx["ball_l"], ball_idx["ball_r"]
                ball_post = np.minimum(skel_kabsch[:, bl, 2],
                                       skel_kabsch[:, br, 2])
                ball_pre = np.minimum(skel_mj[:, bl, 2], skel_mj[:, br, 2])
                dz = ball_pre - args.extra_drop - ball_post
                skel_all[:, :, 2] += dz[:, None]
                print(f"[S4 Ball-align sim] extra_drop={args.extra_drop:.3f} m")
                print(f"  ue_ball_min_z (pre-Kabsch): "
                      f"min={ball_pre.min():.4f}  med={np.median(ball_pre):.4f}  "
                      f"max={ball_pre.max():.4f}")
                print(f"  ball_kabsch_z (post-Kabsch): "
                      f"min={ball_post.min():.4f}  med={np.median(ball_post):.4f}  "
                      f"max={ball_post.max():.4f}")
                print(f"  Δz: min={dz.min():.4f}  med={np.median(dz):.4f}  "
                      f"max={dz.max():.4f}  std={dz.std():.4f}")
                foot_z = np.minimum(skel_all[:, bl, 2], skel_all[:, br, 2])
                print(f"  foot Z after ball-align: "
                      f"min={foot_z.min():.4f}  med={np.median(foot_z):.4f}  "
                      f"max={foot_z.max():.4f}")
        elif stage_num >= 5:
            skel_all = skel_kabsch   # human skeleton stays at Kabsch stage

            # Rotations from load_gasp_data (cq + root.q → BVH coords).
            # Positions come from wp (skel_mj) — same source as the Kabsch
            # skeleton used for grounding, eliminating the ~6 cm Z mismatch
            # that the old load_gasp_data ground_z_cm path had.
            print(f"\n[Retarget] Loading GASP rotations via vis_skeleton_compare ...")
            _, gasp_rot_all, _, _ = \
                _load_gasp_data_for_retarget(
                    args.jsonl, args.meta,
                    preserve_world_z=False, ground_z_cm=0.0,
                )
            print(f"  skel_mj shape = {skel_mj.shape} (wp-based positions for GMR)")

            # Build GMR input frames: apply R_global/t_global + per-bone Δ.
            # Positions use skel_mj (from wp) so GMR input and grounding
            # reference (skel_kabsch) share the same Z — no ground_z_cm needed.
            print(f"[Retarget] Building GMR frames ...")
            gmr_frames, root_xy = build_gmr_frames(
                skel_mj, gasp_rot_all, bone_names, Rg, tg, delta,
                recenter_root_xy=True)
            # Human height is FIXED and known per source (ground/stairs at G1
            # height; traversal UE->0.77->G1), so after mesh.s*data-scale every
            # clip is at the G1-scale height baked into the config. Use that
            # constant instead of estimating from the motion (a crouched frame
            # 0 shrinks the whole skeleton -> the '矮' clip bug).
            h_cfg = cfg.get("actual_human_height_m") if cfg is not None else None
            if h_cfg is not None:
                h = float(h_cfg)
                print(f"  Human height = {h:.3f} m  (from config actual_human_height_m)")
            else:
                h = estimate_human_height(gmr_frames)
                print(f"  Estimated human height = {h:.3f} m  (config missing -> frame-0 fallback)")
            print(f"  root XY range: X[{root_xy[:,0].min():.2f},"
                  f"{root_xy[:,0].max():.2f}]  "
                  f"Y[{root_xy[:,1].min():.2f},{root_xy[:,1].max():.2f}] m "
                  f"(restored to qpos after GMR)")

            if scaled_only:
                # ── s5.1: visualise ONLY the "scale human data" (IK input) ──
                # Run GMR's scaling/offset step (no IK solve) and show the
                # resulting target skeleton. No G1 robot is built.
                print(f"[Retarget s5.1] Extracting scaled human data "
                      f"(IK input, pre-optimisation) ...")
                scaled_skel, scaled_names, _, scaled_toe = \
                    scaled_human_skeleton(gmr_frames, h, root_xy,
                                          src_human=args.src_human)
                print(f"  scaled skeleton: {scaled_skel.shape[1]} joints, "
                      f"{scaled_skel.shape[0]} frames")

                # ── method 3: support-foot-anchored vertical grounding ──
                # GMR scales about the world origin (z=0), so on elevated terrain
                # the scaled skeleton sinks. We re-anchor it in Z (single
                # per-frame shift; pose/scale unchanged) at the SUPPORT (stance)
                # foot's contact point instead of the origin.
                #
                # A single global dz has 1 DOF, so it can pin only ONE foot. We
                # therefore align each red toe to its OWN side's blue ball
                # (same foot, so no cross-foot aliasing):
                #     res_i = blue_ball_i_z - red_toe_i_z          (i = L, R)
                # and blend the two with a soft-min weight that favours the
                # LOWER (support) red toe:
                #     w_i = softmin_k(red_toe_z)_i,   dz = Σ_i w_i · res_i
                # The toe→ground offset is identical on both sides (toe-to-toe),
                # so it cancels and is omitted. Because the weights and per-foot
                # residuals are continuous in time, dz is continuous: a clear
                # stance foot gets w≈1 (grounded precisely), and the handover at
                # foot-switch / double-support is smoothed over a ~1/k band — no
                # height jumps. Larger k → sharper pick (--ground-softness).
                if args.no_scaled_ground:
                    print("[s5.1 grounding] DISABLED (--no-scaled-ground).")
                else:
                    have_blue = "ball_l" in ball_idx and "ball_r" in ball_idx
                    red_toe_ok = bool(np.isfinite(scaled_toe).all())
                    if have_blue and red_toe_ok:
                        k = float(args.ground_softness)
                        blue_l = skel_kabsch[:, ball_idx["ball_l"], 2]
                        blue_r = skel_kabsch[:, ball_idx["ball_r"], 2]
                        red_l, red_r = scaled_toe[:, 0, 2], scaled_toe[:, 1, 2]
                        dz = support_soft_ground_dz(blue_l, blue_r,
                                                    red_l, red_r, k)

                        scaled_skel[:, :, 2] += dz[:, None]
                        scaled_toe[:, :, 2] += dz[:, None]
                        toe_world = scaled_toe
                        print(f"[s5.1 grounding] method 3 (support-foot anchor, "
                              f"soft-min k={k:g})")
                        print(f"  Δz: min={dz.min():.4f}  med={np.median(dz):.4f}"
                              f"  max={dz.max():.4f}  std={dz.std():.4f}")

                        # ── smoothness diagnostic ──
                        # With per-foot pairing + soft-min, dz should be smooth.
                        # Report the largest frame-to-frame change and how often
                        # it exceeds 1 cm (was the source of the old jumps).
                        if n_frames > 1:
                            ddz = np.abs(np.diff(dz))
                            jump_fr = np.where(ddz > 0.01)[0]
                            print(f"  [smooth diag] max|Δdz/frame|="
                                  f"{ddz.max()*100:.2f} cm  med="
                                  f"{np.median(ddz)*100:.3f} cm  "
                                  f"|Δdz|>1cm: {jump_fr.size} frames")
                            if jump_fr.size:
                                preview = ", ".join(str(int(f))
                                                    for f in jump_fr[:12])
                                print(f"  [smooth diag] >1cm at frames "
                                      f"(first 12): {preview}")
                    else:
                        print("[s5.1 grounding] skipped (need blue ball_l/r "
                              "and a finite scaled toe).")

                # Keep the original (un-scaled) Kabsch full-body skeleton to
                # overlay for a before/after-scaling comparison (口径 B). These
                # bone_pairs / vis_idx / foot_idx still hold the full-body
                # BONE_CONNECTIONS values computed from meta above.
                overlay_skel = skel_kabsch
                overlay_pairs = bone_pairs
                overlay_vis_idx = vis_idx
                overlay_foot_idx = foot_idx
                # Override the visualised skeleton + connectivity with the
                # scaled human data, resolved against its own bone names.
                skel_all = scaled_skel
                bone_names = scaled_names
                name_to_idx = {n: i for i, n in enumerate(scaled_names)}
                bone_pairs = np.array(
                    [(name_to_idx[a], name_to_idx[b])
                     for a, b in SCALED_BONE_CONNECTIONS
                     if a in name_to_idx and b in name_to_idx],
                    dtype=np.int64)
                vis_idx = list(range(len(scaled_names)))
                hide_set = set()
                foot_idx = [name_to_idx[n]
                            for n in ("LeftFootMod", "RightFootMod")
                            if n in name_to_idx]
            else:
                # s5/s5.2: make IK input use the same grounded scaled-human
                # targets as s5.1. Method:
                #   1) compute method-3 dz on the scaled toe (world frame);
                #   2) back-project to gmr input by dividing with s_root.
                # A uniform input shift δ gives a scaled shift s_root*δ, so
                # δ = dz/s_root reproduces the same pre-IK target grounding.
                s52_scaled_skel = None
                s52_scaled_names = None
                s52_scaled_toe = None
                _scaled_tmp = _scaled_names_tmp = _retargeter_probe = scaled_toe_tmp = None
                need_scaled_probe = (args.stage == "s5.2") or (not args.no_scaled_ground)
                if need_scaled_probe:
                    _scaled_tmp, _scaled_names_tmp, _retargeter_probe, scaled_toe_tmp = \
                        scaled_human_skeleton(gmr_frames, h, root_xy,
                                              src_human=args.src_human)
                    if args.stage == "s5.2":
                        # s5.2 visualisation should match s5.1 colors:
                        # red=scaled, blue=un-scaled overlay.
                        s52_scaled_skel = _scaled_tmp.copy()
                        s52_scaled_names = list(_scaled_names_tmp)
                        s52_scaled_toe = scaled_toe_tmp.copy()

                if args.no_scaled_ground:
                    print(f"[{args.stage} pre-IK grounding] "
                          "DISABLED (--no-scaled-ground).")
                else:
                    print(f"[{args.stage} pre-IK grounding] Computing grounded "
                          "scaled target (same method 3 as s5.1) ...")
                    have_blue = "ball_l" in ball_idx and "ball_r" in ball_idx
                    red_toe_ok = bool(np.isfinite(scaled_toe_tmp).all()) if scaled_toe_tmp is not None else False
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
                            # keep s5.2 scaled visual in sync with grounded target
                            if s52_scaled_skel is not None and s52_scaled_toe is not None:
                                s52_scaled_skel[:, :, 2] += dz_target[:, None]
                                s52_scaled_toe[:, :, 2] += dz_target[:, None]
                            print(f"  method 3 applied to IK input: "
                                  f"k={k:g}, s_root={s_root:.6f}")
                            print(f"  target Δz: min={dz_target.min():.4f}  "
                                  f"med={np.median(dz_target):.4f}  "
                                  f"max={dz_target.max():.4f}  "
                                  f"std={dz_target.std():.4f}")
                            if n_frames > 1:
                                ddz = np.abs(np.diff(dz_target))
                                print(f"  [smooth diag] max|Δdz/frame|="
                                      f"{ddz.max()*100:.2f} cm  med="
                                      f"{np.median(ddz)*100:.3f} cm")
                        else:
                            print("  skipped: invalid s_root≈0, cannot back-project dz.")
                    else:
                        print("  skipped: need blue ball_l/r and a finite scaled toe.")

                # GMR retarget → G1 qpos
                print(f"[Retarget] Running GMR retarget ({n_frames} frames) ...")
                qpos_arr, retargeter = retarget_to_g1(
                    gmr_frames, h, src_human=args.src_human)
                g1_model = retargeter.model
                # restore absolute root XY removed before GMR
                qpos_arr[:, 0] += root_xy[:, 0]
                qpos_arr[:, 1] += root_xy[:, 1]
                print(f"  qpos shape = {qpos_arr.shape}")

                # Foot grounding on actual G1 qpos (using post-Kabsch ball Z).
                #   offset_human: human toe joint (ball) → ground (≈3cm, --extra-drop)
                #   offset_robot: G1 toe_link → ground in URDF standing pose
                import mujoco as mj
                offset_human = args.extra_drop
                offset_robot = g1_toe_standing_offset(g1_model)
                # GMR input positions and skel_kabsch both come from wp, so
                # their ball-Z values are consistent — no cross-source offset.
                have_balls = "ball_l" in ball_idx and "ball_r" in ball_idx
                if have_balls:
                    bl_s, br_s = ball_idx["ball_l"], ball_idx["ball_r"]
                    ball_l_kabsch = skel_kabsch[:, bl_s, :]
                    ball_r_kabsch = skel_kabsch[:, br_s, :]
                    ue_ball_min_z = np.minimum(ball_l_kabsch[:, 2],
                                               ball_r_kabsch[:, 2])

                # (1) Root-Z grounding: a single per-frame root Z shift so the
                #     robot's (lower) toe-to-ground matches the human's (lower)
                #     toe-to-ground:
                #        (toe_z + dz) - offset_robot = min(ball_z) - offset_human
                #     Keeps the retargeted motion intact, only fixes the global
                #     vertical float caused by ankle≠foot height. Skipped with
                #     --no-ball-align.
                if args.no_ball_align:
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

                # (2) Per-foot grounding: independently bend each leg so each
                #     toe_link lands on its own target (handles stairs / feet at
                #     different heights).
                if args.per_foot_ground and have_balls:
                    qpos_arr = per_foot_ground(
                        g1_model, qpos_arr, ball_l_kabsch[:, 2], ball_r_kabsch[:, 2],
                        offset_human, offset_robot)
                    # residual report (per foot)
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

                # Foot geom ids for per-frame Z readout in render loop
                g1_foot_geoms = _foot_geom_ids(g1_model)
                g1_foot_z_final = g1_fk_foot_z(g1_model, qpos_arr)
                print(f"  G1 foot Z (final): "
                      f"min={g1_foot_z_final.min():.4f}  "
                      f"med={np.median(g1_foot_z_final):.4f}  "
                      f"max={g1_foot_z_final.max():.4f}")
                print(f"  G1 ngeom={g1_model.ngeom}, foot contact geoms={len(g1_foot_geoms)}")

                # ── G1 FK skeleton: pre-compute body positions for vis ──
                # (1) Anatomical stick figure (overlays the robot mesh): unique
                #     bodies referenced by G1_FK_BONE_CONNECTIONS.
                _fig_names = []
                for a, b in G1_FK_BONE_CONNECTIONS:
                    for nm in (a, b):
                        if nm not in _fig_names and \
                                mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, nm) >= 0:
                            _fig_names.append(nm)
                g1_fk_body_names = _fig_names
                _fig_ids = [mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, nm)
                            for nm in g1_fk_body_names]
                _fig_n2i = {n: i for i, n in enumerate(g1_fk_body_names)}
                g1_fk_pairs = np.array(
                    [(_fig_n2i[a], _fig_n2i[b])
                     for a, b in G1_FK_BONE_CONNECTIONS
                     if a in _fig_n2i and b in _fig_n2i],
                    dtype=np.int64)

                # (2) IK-matched bodies → human joints (for magenta error lines).
                # Use joint_mapping.json for the robot↔human pairing.
                _jm_path = (pathlib.Path(args.joint_mapping) if args.joint_mapping
                            else HERE / "joint_mapping.json")
                if _jm_path.exists():
                    with open(_jm_path, "r", encoding="utf-8") as _f:
                        _jm_data = json.load(_f)
                    _ik_bodies = []
                    _ik_body_ids = []
                    g1_ik_human_names = []
                    for _m in _jm_data["mappings"]:
                        _rb = _m["robot"]
                        _hj = _m["human"]
                        _bid = mj.mj_name2id(g1_model, mj.mjtObj.mjOBJ_BODY, _rb)
                        if _bid >= 0:
                            _ik_bodies.append(_rb)
                            _ik_body_ids.append(_bid)
                            g1_ik_human_names.append(_hj)
                        else:
                            print(f"[G1 FK skel] joint_mapping: robot body "
                                  f"'{_rb}' not found in MuJoCo model, skipped")
                    print(f"[G1 FK skel] error-line mapping from: {_jm_path}")
                else:
                    print(f"[G1 FK skel] WARNING: joint_mapping not found at "
                          f"{_jm_path}, falling back to ik_match_table")
                    _ik_seen = {}
                    for _tbl in (retargeter.ik_match_table1,
                                 retargeter.ik_match_table2):
                        for _rb, _entry in _tbl.items():
                            _hj = _entry[0]
                            _bid = mj.mj_name2id(
                                g1_model, mj.mjtObj.mjOBJ_BODY, _rb)
                            if _bid >= 0:
                                _ik_seen[_rb] = (_hj, _bid)
                    _ik_bodies = list(_ik_seen.keys())
                    g1_ik_human_names = [_ik_seen[rb][0] for rb in _ik_bodies]
                    _ik_body_ids = [_ik_seen[rb][1] for rb in _ik_bodies]

                # FK all frames (both sets share one forward pass per frame)
                _fk_d = mj.MjData(g1_model)
                g1_fk_skel = np.zeros((n_frames, len(g1_fk_body_names), 3),
                                      dtype=np.float64)
                g1_ik_fk_pts = np.zeros((n_frames, len(_ik_bodies), 3),
                                        dtype=np.float64)
                for _fi in range(n_frames):
                    _fk_d.qpos[:] = qpos_arr[_fi]
                    mj.mj_forward(g1_model, _fk_d)
                    for _ji, _bid in enumerate(_fig_ids):
                        g1_fk_skel[_fi, _ji] = _fk_d.xpos[_bid]
                    for _ji, _bid in enumerate(_ik_body_ids):
                        g1_ik_fk_pts[_fi, _ji] = _fk_d.xpos[_bid]
                print(f"[G1 FK skel] figure: {len(g1_fk_body_names)} bodies, "
                      f"{g1_fk_pairs.shape[0]} bones; "
                      f"IK-matched: {len(_ik_bodies)} bodies; "
                      f"{n_frames} frames precomputed")

                # s5.2 visual overlay: same color semantics as s5.1
                #   red = scaled grounded skeleton, blue = original Kabsch
                if args.stage == "s5.2" and s52_scaled_skel is not None:
                    overlay_skel = skel_kabsch
                    overlay_pairs = bone_pairs
                    overlay_vis_idx = vis_idx
                    overlay_foot_idx = foot_idx
                    skel_all = s52_scaled_skel
                    toe_world = s52_scaled_toe
                    bone_names = s52_scaled_names
                    name_to_idx = {n: i for i, n in enumerate(s52_scaled_names)}
                    bone_pairs = np.array(
                        [(name_to_idx[a], name_to_idx[b])
                         for a, b in SCALED_BONE_CONNECTIONS
                         if a in name_to_idx and b in name_to_idx],
                        dtype=np.int64)
                    vis_idx = list(range(len(s52_scaled_names)))
                    hide_set = set()
                    foot_idx = [name_to_idx[n]
                                for n in ("LeftFootMod", "RightFootMod")
                                if n in name_to_idx]

                # Build error-line pairs: (ik_fk_idx, skel_all_idx).
                # Must be built AFTER skel_all / name_to_idx are finalised.
                if g1_ik_fk_pts is not None:
                    _ep = []
                    for _gi, _hname in enumerate(g1_ik_human_names):
                        if _hname in name_to_idx:
                            _ep.append((_gi, name_to_idx[_hname]))
                    g1_error_pairs = np.array(_ep, dtype=np.int64) if _ep \
                        else np.zeros((0, 2), dtype=np.int64)
                    print(f"[G1 FK skel] {g1_error_pairs.shape[0]} error-line pairs:")
                    for _k in range(g1_error_pairs.shape[0]):
                        _ik_i, _sk_i = int(g1_error_pairs[_k, 0]), int(g1_error_pairs[_k, 1])
                        _rbn = _ik_bodies[_ik_i]
                        _hjn = g1_ik_human_names[_ik_i]
                        _fk0 = g1_ik_fk_pts[0, _ik_i]
                        _sh0 = skel_all[0, _sk_i]
                        _d0 = float(np.linalg.norm(_fk0 - _sh0)) * 1000
                        print(f"  [{_k:2d}] robot={_rbn:30s} <-> "
                              f"human={_hjn:16s} (skel idx {_sk_i:2d})  "
                              f"frame0 err={_d0:.1f} mm")

        unit_label = "MuJoCo m"
        scale = 1.0

    # Foot stats (human)
    if foot_idx:
        fz = np.array([skel_all[fi, foot_idx, 2].min() for fi in range(n_frames)])
        if stage_num <= 1:
            print(f"[Human] ball(min) Z: min={fz.min():.1f}  "
                  f"med={np.median(fz):.1f}  max={fz.max():.1f} cm")
        else:
            print(f"[Human] ball(min) Z: min={fz.min():.4f}  "
                  f"med={np.median(fz):.4f}  max={fz.max():.4f} m")

    # ── Terrain ──
    terrain_ue = None
    tpath = resolve_terrain_path(args)
    if tpath:
        terrain_ue = load_terrain_instances(tpath, args.terrain_ground_z0,
                                            skip_engine=not args.no_skip_engine)

    # terrain_raw: same coord system as skel_all (cm for ue-world, m for others)
    terrain_raw = None
    if terrain_ue is not None:
        tv, tf = terrain_ue
        if stage_num <= 1:
            terrain_raw = (tv, tf)        # keep UE cm
        else:
            tvmj = ue_to_mujoco(tv)
            tvf = apply_kabsch(tvmj, Rg, tg)
            terrain_raw = (tvf, tf)
        tz = terrain_raw[0][:, 2]
        if stage_num <= 1:
            print(f"[Terrain] {tv.shape[0]} verts, {tf.shape[0]} tris; "
                  f"Z range [{tz.min():.1f}, {tz.max():.1f}] cm")
        else:
            print(f"[Terrain] {tv.shape[0]} verts, {tf.shape[0]} tris; "
                  f"Z range [{tz.min():.4f}, {tz.max():.4f}] m")

    # XY recenter (no Z shift). Everything goes through scale for the viewer.
    ref = terrain_raw[0] * scale if terrain_raw is not None else skel_all[0] * scale
    off_xy = ref[:, :2].mean(axis=0)

    def recenter(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).copy()
        p[..., 0] -= off_xy[0]
        p[..., 1] -= off_xy[1]
        return p.astype(np.float32)

    def skel_view(fi: int) -> np.ndarray:
        return recenter(skel_all[fi] * scale)

    def overlay_view(fi: int) -> np.ndarray:
        return recenter(overlay_skel[fi] * scale)

    def toe_view(fi: int) -> np.ndarray:
        return recenter(toe_world[fi] * scale)

    def g1_fk_view(fi: int) -> np.ndarray:
        return recenter(g1_fk_skel[fi] * scale)

    def g1_ik_fk_view(fi: int) -> np.ndarray:
        return recenter(g1_ik_fk_pts[fi] * scale)

    # ── MuJoCo native viewer branch ──
    if args.viewer == "mujoco":
        if stage_num < 5 or qpos_arr is None or g1_model is None:
            print("  [MuJoCo] viewer requires --stage s5/s5.2 (needs G1 qpos). "
                  "Falling back to viser.")
        else:
            run_mujoco_viewer(args, n_frames, qpos_arr, g1_model, g1_foot_geoms,
                              skel_all, bone_pairs, foot_idx, terrain_raw, off_xy,
                              ik_set=hide_set)
            return

    # ── viser ──
    if viser is None:
        print("[viser] 'viser' is not installed. Install it with `pip install viser` "
              "and re-run, or use `--viewer mujoco` for the native MuJoCo viewer.")
        return
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # ground plane at Z=0
    if terrain_raw is not None:
        tv_rc = recenter(terrain_raw[0] * scale)
        gx0, gy0 = tv_rc[:, 0].min(), tv_rc[:, 1].min()
        gx1, gy1 = tv_rc[:, 0].max(), tv_rc[:, 1].max()
    else:
        gx0, gy0, gx1, gy1 = -10, -10, 10, 10
    pad = 2.0
    gv = np.array([[gx0-pad, gy0-pad, 0], [gx1+pad, gy0-pad, 0],
                    [gx1+pad, gy1+pad, 0], [gx0-pad, gy1+pad, 0]], dtype=np.float32)
    gf = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    server.scene.add_mesh_simple("/ground", gv, gf, color=(150,150,150),
                                 opacity=0.45, side="double", flat_shading=True)
    server.scene.add_grid("/grid", width=float(gx1-gx0+2*pad),
                          height=float(gy1-gy0+2*pad),
                          position=(float((gx0+gx1)/2), float((gy0+gy1)/2), 0.0))

    if terrain_raw is not None:
        server.scene.add_mesh_simple("/terrain",
                                     recenter(terrain_raw[0] * scale),
                                     terrain_raw[1].astype(np.int32),
                                     color=(107,168,235), opacity=0.85,
                                     side="double", flat_shading=True)

    sk = skel_view(0)
    joints_h = server.scene.add_point_cloud("/human/joints", sk[vis_idx],
                                            colors=(220,30,60),
                                            point_size=0.025, point_shape="circle")
    feet_h = None
    if foot_idx:
        feet_h = server.scene.add_point_cloud("/human/feet", sk[foot_idx],
                                              colors=(40,220,60),
                                              point_size=0.05, point_shape="circle")
    seg = np.stack([sk[bone_pairs[:,0]], sk[bone_pairs[:,1]]], axis=1)
    bones_h = server.scene.add_line_segments("/human/bones", seg,
                                             colors=(15,15,15), line_width=2.0)

    # s5.1: original (un-scaled) Kabsch full-body skeleton overlay (blue/grey).
    ov_joints_h = ov_feet_h = ov_bones_h = None
    if overlay_skel is not None:
        ov0 = overlay_view(0)
        ov_joints_h = server.scene.add_point_cloud(
            "/orig/joints", ov0[overlay_vis_idx], colors=(60,120,235),
            point_size=0.02, point_shape="circle")
        if overlay_foot_idx:
            ov_feet_h = server.scene.add_point_cloud(
                "/orig/feet", ov0[overlay_foot_idx], colors=(60,200,200),
                point_size=0.045, point_shape="circle")
        ov_seg = np.stack([ov0[overlay_pairs[:,0]], ov0[overlay_pairs[:,1]]],
                          axis=1)
        ov_bones_h = server.scene.add_line_segments(
            "/orig/bones", ov_seg, colors=(120,120,120), line_width=2.0)

    # s5.1: scaled red toe (ball) markers + ankle→toe connectors (orange).
    toe_h = toe_seg_h = None
    if toe_world is not None and foot_idx:
        tv0 = toe_view(0)                       # (2, 3) = [left, right]
        toe_h = server.scene.add_point_cloud(
            "/scaled/toe", tv0, colors=(245,150,20),
            point_size=0.04, point_shape="circle")
        ank0 = sk[foot_idx]                     # red ankles [left, right]
        toe_seg_h = server.scene.add_line_segments(
            "/scaled/foot", np.stack([ank0, tv0], axis=1),
            colors=(245,150,20), line_width=2.0)

    # G1 robot mesh (retarget stage only)
    g1_handles = []
    g1_data_fk = None
    if g1_model is not None and qpos_arr is not None:
        import mujoco as mj
        g1_data_fk = mj.MjData(g1_model)
        print(f"[G1 Visual] Setting up {g1_model.ngeom} geoms in viser ...")
        g1_handles = setup_g1_visual(g1_model, server)
        print(f"  Created {len(g1_handles)} mesh nodes")

    # G1 FK skeleton overlay (yellow joints + bones, magenta error lines)
    fk_joints_h = fk_bones_h = fk_err_h = fk_ik_pts_h = None
    if g1_fk_skel is not None and g1_fk_pairs is not None:
        fk0 = g1_fk_view(0)
        fk_joints_h = server.scene.add_point_cloud(
            "/g1fk/joints", fk0, colors=(255, 200, 0),
            point_size=0.03, point_shape="circle")
        if g1_fk_pairs.shape[0] > 0:
            fk_seg = np.stack([fk0[g1_fk_pairs[:, 0]],
                               fk0[g1_fk_pairs[:, 1]]], axis=1)
            fk_bones_h = server.scene.add_line_segments(
                "/g1fk/bones", fk_seg, colors=(200, 160, 0), line_width=2.5)
        if g1_error_pairs is not None and g1_error_pairs.shape[0] > 0 \
                and g1_ik_fk_pts is not None:
            ik0 = g1_ik_fk_view(0)
            sk0 = skel_view(0)
            # Cyan markers at the IK-matched robot body positions (the
            # robot-side endpoints of the error lines). These are bodies
            # like hip_yaw_link / shoulder_yaw_link whose origins sit
            # INSIDE the limbs (not at the visible anatomical joints of the
            # yellow skeleton), so without markers the error lines would
            # appear to start from "nowhere".
            fk_ik_pts_h = server.scene.add_point_cloud(
                "/g1fk/ik_pts", ik0[g1_error_pairs[:, 0]],
                colors=(0, 220, 220), point_size=0.025, point_shape="circle")
            err_seg = np.stack([ik0[g1_error_pairs[:, 0]],
                                sk0[g1_error_pairs[:, 1]]], axis=1)
            fk_err_h = server.scene.add_line_segments(
                "/g1fk/errors", err_seg, colors=(220, 50, 220), line_width=1.5)

    g_frame = server.gui.add_slider("frame", 0, max(0, n_frames-1), 1, 0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", 1, 120, 1, 30)
    g_info = server.gui.add_text("ball Z", "")
    has_g1 = g1_data_fk is not None and qpos_arr is not None
    g_show_orig = None
    g_show_fk_skel = None
    g_show_err_lines = None
    if stage_num >= 5:
        g_show_human = server.gui.add_checkbox("show human", True)
        if has_g1:
            g_show_g1 = server.gui.add_checkbox("show G1", True)
        if fk_joints_h is not None:
            g_show_fk_skel = server.gui.add_checkbox("show G1 FK skeleton", True)
            g_show_err_lines = server.gui.add_checkbox("show error lines", True)
    if overlay_skel is not None:
        g_show_orig = server.gui.add_checkbox("show original (un-scaled)", True)
    if scaled_only or args.stage == "s5.2":
        server.gui.add_markdown(
            "**red** = scale human data (IK input) &nbsp; | &nbsp; "
            "**blue** = original un-scaled Kabsch skeleton &nbsp; | &nbsp; "
            "**orange** = scaled red toe (ground ref) &nbsp; | &nbsp; "
            "**yellow** = G1 FK skeleton &nbsp; | &nbsp; "
            "**cyan** = IK-matched robot bodies &nbsp; | &nbsp; "
            "**magenta** = error lines (cyan IK body -> red target)")
    server.gui.add_markdown(f"**Stage:** {args.stage} &nbsp; | &nbsp; **Units:** {unit_label}")

    def render(fi: int):
        # Human skeleton
        show_human = stage_num < 5 or g_show_human.value
        s = skel_view(fi)
        joints_h.points = s[vis_idx] if show_human else s[:0]
        if feet_h is not None:
            feet_h.points = s[foot_idx] if show_human else s[:0]
        if show_human:
            bones_h.points = np.stack([s[bone_pairs[:,0]], s[bone_pairs[:,1]]], axis=1)
        else:
            bones_h.points = np.zeros((0, 2, 3), dtype=np.float32)

        # Original un-scaled Kabsch full-body skeleton overlay (s5.1)
        if ov_joints_h is not None:
            show_orig = g_show_orig.value
            ov = overlay_view(fi)
            ov_joints_h.points = ov[overlay_vis_idx] if show_orig else ov[:0]
            if ov_feet_h is not None:
                ov_feet_h.points = ov[overlay_foot_idx] if show_orig else ov[:0]
            if show_orig:
                ov_bones_h.points = np.stack(
                    [ov[overlay_pairs[:,0]], ov[overlay_pairs[:,1]]], axis=1)
            else:
                ov_bones_h.points = np.zeros((0, 2, 3), dtype=np.float32)

        # Scaled red toe markers + ankle→toe connectors (s5.1), tied to the
        # red ("human") visibility toggle.
        if toe_h is not None:
            tv = toe_view(fi)
            toe_h.points = tv if show_human else tv[:0]
            if show_human:
                toe_seg_h.points = np.stack([s[foot_idx], tv], axis=1)
            else:
                toe_seg_h.points = np.zeros((0, 2, 3), dtype=np.float32)

        # G1 robot mesh
        g1fz = None
        if g1_data_fk is not None and qpos_arr is not None:
            show_g1 = g_show_g1.value
            g1_data_fk.qpos[:] = qpos_arr[fi]
            mj.mj_forward(g1_model, g1_data_fk)
            update_g1_visual(g1_data_fk, g1_handles, off_xy, visible=show_g1)
            if g1_foot_geoms:
                g1fz = min(float(g1_data_fk.geom_xpos[gid, 2])
                           - float(g1_model.geom_size[gid, 0])
                           for gid in g1_foot_geoms)

        # G1 FK skeleton overlay (yellow) + error lines (magenta)
        if fk_joints_h is not None:
            show_fk = g_show_fk_skel.value if g_show_fk_skel is not None else False
            fk = g1_fk_view(fi)
            fk_joints_h.points = fk if show_fk else fk[:0]
            if fk_bones_h is not None:
                if show_fk and g1_fk_pairs.shape[0] > 0:
                    fk_bones_h.points = np.stack(
                        [fk[g1_fk_pairs[:, 0]], fk[g1_fk_pairs[:, 1]]], axis=1)
                else:
                    fk_bones_h.points = np.zeros((0, 2, 3), dtype=np.float32)
            if fk_err_h is not None and g1_ik_fk_pts is not None:
                show_err = g_show_err_lines.value if g_show_err_lines is not None else False
                if show_err and g1_error_pairs.shape[0] > 0:
                    ik = g1_ik_fk_view(fi)
                    fk_err_h.points = np.stack(
                        [ik[g1_error_pairs[:, 0]], s[g1_error_pairs[:, 1]]], axis=1)
                    if fk_ik_pts_h is not None:
                        fk_ik_pts_h.points = ik[g1_error_pairs[:, 0]]
                else:
                    fk_err_h.points = np.zeros((0, 2, 3), dtype=np.float32)
                    if fk_ik_pts_h is not None:
                        fk_ik_pts_h.points = np.zeros((0, 3), dtype=np.float32)

        # Info text
        mean_err = None
        if g1_ik_fk_pts is not None and g1_error_pairs is not None \
                and g1_error_pairs.shape[0] > 0:
            ik_raw = g1_ik_fk_pts[fi]
            sk_raw = skel_all[fi]
            diffs = ik_raw[g1_error_pairs[:, 0]] - sk_raw[g1_error_pairs[:, 1]]
            mean_err = float(np.mean(np.linalg.norm(diffs, axis=1))) * 1000
        info_parts = []
        if foot_idx:
            bz_raw = skel_all[fi, foot_idx, 2].min()
            if stage_num <= 1:
                info_parts.append(f"ball: {bz_raw:.1f} cm")
            else:
                info_parts.append(f"ball: {bz_raw:.4f} m")
        if g1fz is not None:
            info_parts.append(f"G1: {g1fz:.4f} m")
            if foot_idx:
                diff = g1fz - skel_all[fi, foot_idx, 2].min()
                info_parts.append(f"Δ: {diff*100:+.2f} cm")
        if mean_err is not None:
            info_parts.append(f"FK err: {mean_err:.1f} mm")
        g_info.value = "  |  ".join(info_parts)

    g_frame.on_update(lambda _: render(int(g_frame.value)))
    if stage_num >= 5:
        g_show_human.on_update(lambda _: render(int(g_frame.value)))
        if has_g1:
            g_show_g1.on_update(lambda _: render(int(g_frame.value)))
    if g_show_orig is not None:
        g_show_orig.on_update(lambda _: render(int(g_frame.value)))
    if g_show_fk_skel is not None:
        g_show_fk_skel.on_update(lambda _: render(int(g_frame.value)))
    if g_show_err_lines is not None:
        g_show_err_lines.on_update(lambda _: render(int(g_frame.value)))
    render(0)

    _vp = getattr(server, "port", args.port)
    print(f"\n[viser] http://localhost:{_vp}  — stage={args.stage}")
    while True:
        if g_play.value and n_frames > 1:
            g_frame.value = (int(g_frame.value) + 1) % n_frames
            time.sleep(1.0 / max(1.0, float(g_fps.value)))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
