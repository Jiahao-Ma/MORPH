"""Compare three skeleton sources side-by-side in MuJoCo.

  LEFT  (Gold):  FBX Step1 (via Blender extraction) — Blender world coords, meters
  MID   (Green): BVH Step2 (parsed directly)        — BVH coords, cm
  RIGHT (Cyan):  GASP JSONL (Component → World)     — UE5 World coords, cm

All converted to MuJoCo frame (X=fwd, Y=left, Z=up, RH, meters) for display.

Coordinate systems verified:
  Blender/BVH: X=skel_left, Y=skel_fwd, Z=up  → MuJoCo: swap(X,Y)
  UE5 World:   X=fwd,       Y=right,    Z=up  → MuJoCo: negate(Y)

Usage:
  conda activate gmr
  python vis_skeleton_compare.py
  python vis_skeleton_compare.py --fbx <path> --bvh <path> --jsonl <path> --meta <path>

Controls:
  Space       Pause / Resume
  Right/Left  Step forward / backward
  Backspace   Reset to frame 0
  L           Toggle joint labels
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, cast

import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

mj_api = cast(Any, mj)
mjt_geom = cast(Any, mj).mjtGeom

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

DEFAULT_FBX = str(DATA_DIR / "UEGMR" / "step1_fbx_export" / "M_Neutral_Stand_Idle_Loop.fbx")
DEFAULT_BVH = str(DATA_DIR / "UEGMR" / "step2_bvh" / "M_Neutral_Stand_Idle_Loop.bvh")
DEFAULT_JSONL = str(DATA_DIR / "20260419_151033_frames.jsonl")
DEFAULT_META = str(DATA_DIR / "20260419_151033_meta.json")
# Calibration data for per-bone Δ between GASP and BVH world frames.
# Default: reuse the animation JSONL itself — Δ is a structural constant of the
# UE5 skeleton, so any frame where BVH and GASP are in approximately the same
# pose works (typical choice: frame 0, when GASP recording starts from idle).
DEFAULT_CALIB_JSONL = DEFAULT_JSONL
DEFAULT_CALIB_META  = DEFAULT_META

DEFAULT_BLENDER = r"D:\tool\blender\blender-4.0.0-windows-x64\blender.exe"

# ── Skeleton structure ───────────────────────────────────────────────────────

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

IMPORTANT_BONES = {
    "pelvis", "spine_01", "spine_03", "spine_05", "head",
    "neck_01", "clavicle_l", "clavicle_r",
    "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
    "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r",
    "foot_l", "foot_r", "ball_l", "ball_r",
}

HIDDEN_KEYWORDS = [
    "thumb", "index", "middle", "ring", "pinky",
    "metacarpal", "twist", "ik_hand", "ik_foot",
    "weapon", "attach",
]


# ── BVH Parser ───────────────────────────────────────────────────────────────

class BVHJoint:
    def __init__(self, name, parent_idx, offset, channels, channel_order):
        self.name = name
        self.parent_idx = parent_idx
        self.offset = np.array(offset, dtype=np.float64)
        self.channels = channels
        self.channel_order = channel_order


class BVHData:
    def __init__(self):
        self.joints: list[BVHJoint] = []
        self.frames: Any = []
        self.frame_time = 0.0
        self.num_frames = 0
        self._channel_specs: list[tuple[int, list[str]]] = []


def parse_bvh(filepath: str) -> BVHData:
    bvh = BVHData()
    joint_stack: list[int] = []
    channel_specs: list[tuple[int, list[str]]] = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    i = 0
    in_motion = False
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line == "HIERARCHY":
            continue
        if line == "MOTION":
            in_motion = True
            continue
        if in_motion:
            m = re.match(r"Frames:\s+(\d+)", line)
            if m:
                bvh.num_frames = int(m.group(1))
                continue
            m = re.match(r"Frame Time:\s+([\d.]+)", line)
            if m:
                bvh.frame_time = float(m.group(1))
                continue
            vals = list(map(float, line.split()))
            if vals:
                bvh.frames.append(vals)
            continue
        m = re.match(r"ROOT\s+(.+)", line)
        if m:
            j = BVHJoint(m.group(1).strip(), -1, [0, 0, 0], [], "")
            bvh.joints.append(j)
            joint_stack.append(len(bvh.joints) - 1)
            continue
        m = re.match(r"JOINT\s+(.+)", line)
        if m:
            pidx = joint_stack[-1] if joint_stack else -1
            j = BVHJoint(m.group(1).strip(), pidx, [0, 0, 0], [], "")
            bvh.joints.append(j)
            joint_stack.append(len(bvh.joints) - 1)
            continue
        if "End Site" in line:
            joint_stack.append(-999)
            continue
        if "{" in line:
            continue
        if "}" in line:
            if joint_stack:
                joint_stack.pop()
            continue
        m = re.match(r"OFFSET\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)", line)
        if m:
            if joint_stack and joint_stack[-1] >= 0:
                bvh.joints[joint_stack[-1]].offset = np.array(
                    [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                )
            continue
        m = re.match(r"CHANNELS\s+(\d+)\s+(.*)", line)
        if m:
            ch_names = m.group(2).split()
            if joint_stack and joint_stack[-1] >= 0:
                jidx = joint_stack[-1]
                bvh.joints[jidx].channels = ch_names
                bvh.joints[jidx].channel_order = "".join(
                    c[0].upper() for c in ch_names if "rotation" in c.lower()
                )
            channel_specs.append((joint_stack[-1] if joint_stack else -1, ch_names))
            continue
    bvh.frames = np.array(bvh.frames, dtype=np.float64)
    bvh._channel_specs = channel_specs
    return bvh


def compute_fk(bvh: BVHData, frame_idx: int):
    frame_data = bvh.frames[frame_idx]
    n = len(bvh.joints)
    gpos = np.zeros((n, 3))
    grot = [np.eye(3)] * n
    offset = 0
    for joint_idx, ch_names in bvh._channel_specs:
        if joint_idx < 0:
            offset += len(ch_names)
            continue
        joint = bvh.joints[joint_idx]
        local_pos = joint.offset.copy()
        euler = [0.0, 0.0, 0.0]
        for ch in ch_names:
            v = frame_data[offset]
            offset += 1
            if ch == "Xposition":
                local_pos[0] = v
            elif ch == "Yposition":
                local_pos[1] = v
            elif ch == "Zposition":
                local_pos[2] = v
            elif ch == "Xrotation":
                euler[0] = v
            elif ch == "Yrotation":
                euler[1] = v
            elif ch == "Zrotation":
                euler[2] = v
        rot_order = joint.channel_order or "ZYX"
        ax_map = {"X": euler[0], "Y": euler[1], "Z": euler[2]}
        local_rot = R.from_euler(rot_order, [ax_map[c] for c in rot_order], degrees=True).as_matrix()
        pidx = joint.parent_idx
        if pidx < 0:
            gpos[joint_idx] = local_pos
            grot[joint_idx] = local_rot
        else:
            gpos[joint_idx] = gpos[pidx] + grot[pidx] @ local_pos
            grot[joint_idx] = grot[pidx] @ local_rot
    return gpos, grot


# ── FBX extraction via Blender ───────────────────────────────────────────────

def extract_fbx_via_blender(fbx_path: str, blender_exe: str) -> dict:
    """Run Blender in background to extract bone positions. Uses cache."""
    cache_path = fbx_path.replace(".fbx", ".bones.json")
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(fbx_path):
        print(f"  Using cached FBX extraction: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    helper = str(HERE / "_blender_extract_fbx.py")
    print(f"  Extracting FBX via Blender...")
    cmd = [blender_exe, "--background", "--python", helper, "--", fbx_path, cache_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Blender stderr:\n{result.stderr}")
        raise RuntimeError(f"Blender extraction failed (exit {result.returncode})")

    for line in result.stdout.splitlines():
        if "[extract_fbx]" in line:
            print(f"  {line.strip()}")

    with open(cache_path, "r") as f:
        return json.load(f)


# ── GASP JSONL Loader ────────────────────────────────────────────────────────

def load_gasp_frames(jsonl_path: str) -> list[dict]:
    frames = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def gasp_component_to_world(
    joint_cps: np.ndarray,
    root_p: np.ndarray,
    root_q: np.ndarray,
    preserve_world_z: bool = False,
    ground_z_cm: float = 0.0,
) -> np.ndarray:
    """Component Space → UE5 World Space.

    MeshComponent has -90deg yaw relative to Actor.

    preserve_world_z:
        False (default): the mesh-component origin is pinned to world Z=0 — the
            historical flat-ground assumption used by the calibration tools.
        True: keep the actor's real world Z (minus ``ground_z_cm``) so the
            skeleton follows terrain / stairs instead of being flattened.
    ground_z_cm:
        UE ground reference (cm) subtracted from the actor Z when
        ``preserve_world_z`` is set, so the reference floor maps to Z=0.
    """
    actor_rot = R.from_quat(root_q)  # xyzw
    mesh_rel = R.from_euler("z", -90, degrees=True)
    mesh_rot = actor_rot * mesh_rel
    world_z = (root_p[2] - ground_z_cm) if preserve_world_z else 0.0
    mesh_world_pos = np.array([root_p[0], root_p[1], world_z])
    return mesh_rot.apply(joint_cps) + mesh_world_pos


# ── Coordinate Conversions ────────────────────────────────────────────────────
# MuJoCo has no fixed axis semantics — it's just a RH Z-up renderer.
# We render everything in BVH/Blender coordinates (X=skel_left, Y=skel_fwd, Z=up)
# exactly like step3_visualize_bvh.py does (no axis swap, only unit conversion).

def blender_m_to_mujoco(positions_m: np.ndarray) -> np.ndarray:
    """Blender world (X=skel_left, Y=skel_fwd, Z=up, m) — pass through."""
    return positions_m.copy()


def bvh_cm_to_mujoco(positions_cm: np.ndarray) -> np.ndarray:
    """BVH (X=skel_left, Y=skel_fwd, Z=up, cm) — cm to m only."""
    return positions_cm * 0.01


def ue5world_cm_to_mujoco(positions_cm: np.ndarray) -> np.ndarray:
    """GASP World (after scipy RH rotation of LH component-space joints) → BVH/Blender coords.

    Note: gasp_component_to_world() applies scipy (RH) rotation to LH UE5 data,
    which effectively mirrors left/right. To match BVH/FBX skeleton orientation
    (X=skel_left, Y=skel_fwd, Z=up, RH, m), we do NOT negate the Y axis.

    Mapping: new_X = +Y, new_Y = +X, new_Z = +Z
    """
    out = np.empty_like(positions_cm)
    out[..., 0] = positions_cm[..., 1]   # X(left) = Y
    out[..., 1] = positions_cm[..., 0]   # Y(fwd)  = X
    out[..., 2] = positions_cm[..., 2]   # Z(up)   = Z
    return out * 0.01


# GASP→BVH coord change matrix (matches the (X,Y,Z)→(Y,X,Z) swap used for
# positions in ue5world_cm_to_mujoco). M is symmetric so M.T == M.
_M_UE_TO_BVH = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=np.float64)


def load_gasp_data(jsonl_path: str, meta_path: str,
                   preserve_world_z: bool = False, ground_z_cm: float = 0.0):
    """Load GASP JSONL+meta and return (positions_mj, rotations_all, names, parents).

    positions_mj : (F, B, 3) float64 — world position in BVH/MuJoCo coords (m).
    rotations_all: list[F] of list[B] 3x3 rotation matrices in BVH/MuJoCo coords
                   (X=skel_left, Y=skel_fwd, Z=up, RH).
    names        : list[B] bone names (UE5 native).
    parents      : list[B] parent indices.

    preserve_world_z / ground_z_cm: forwarded to gasp_component_to_world so the
    skeleton can keep its real world Z (terrain / stair following) instead of
    being pinned to Z=0. See that function for details.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    frames = load_gasp_frames(jsonl_path)
    names = meta["bone_names"]
    parents = meta["parent_indices"]
    n_bones = len(names)
    n_frames = len(frames)

    positions_mj = np.zeros((n_frames, n_bones, 3))
    rotations_all: list[list[np.ndarray]] = []

    for fi, frame in enumerate(frames):
        root_p = np.array(frame["root"]["p"], dtype=np.float64)
        root_q = np.array(frame["root"]["q"], dtype=np.float64)  # xyzw
        joint_cps = np.zeros((n_bones, 3))
        joint_cqs = np.zeros((n_bones, 4))
        for ji, jdata in enumerate(frame["joints"]):
            joint_cps[ji] = jdata["cp"]
            joint_cqs[ji] = jdata["cq"]
        world_cm = gasp_component_to_world(
            joint_cps, root_p, root_q,
            preserve_world_z=preserve_world_z, ground_z_cm=ground_z_cm,
        )
        positions_mj[fi] = ue5world_cm_to_mujoco(world_cm)

        actor_rot = R.from_quat(root_q)
        mesh_rel = R.from_euler("z", -90, degrees=True)
        mesh_rot = actor_rot * mesh_rel

        rots: list[np.ndarray] = []
        for ji in range(n_bones):
            R_local = R.from_quat(joint_cqs[ji])
            R_world_mat = (mesh_rot * R_local).as_matrix()
            R_bvh_mat = _M_UE_TO_BVH @ R_world_mat @ _M_UE_TO_BVH.T
            rots.append(R_bvh_mat)
        rotations_all.append(rots)

    return positions_mj, rotations_all, names, parents


def compute_per_bone_delta(
    bvh_rotations_frame: list[np.ndarray],
    bvh_names: list[str],
    gasp_rotations_frame: list[np.ndarray],
    gasp_names: list[str],
) -> dict[str, np.ndarray]:
    """Δ_i = R_gasp_i^(-1) · R_bvh_i  (per-bone constant rotation in BVH coords).

    Returned as 3x3 matrices, one per shared bone name.
    """
    bvh_idx = {n: i for i, n in enumerate(bvh_names)}
    delta: dict[str, np.ndarray] = {}
    for gi, name in enumerate(gasp_names):
        if name not in bvh_idx:
            continue
        R_g = gasp_rotations_frame[gi]
        R_b = bvh_rotations_frame[bvh_idx[name]]
        delta[name] = R_g.T @ R_b   # R_g^(-1) is R_g.T for rotation matrices
    return delta


def report_delta_variance(
    bvh_rotations_all: list[list[np.ndarray]],
    bvh_names: list[str],
    gasp_rotations_all: list[list[np.ndarray]],
    gasp_names: list[str],
    bones_to_check: list[str],
    sample_frames: list[int],
):
    """Print, for each bone, the angular spread of Δ_i across `sample_frames`.

    A small spread (deg) means the per-bone delta is truly a structural constant
    (good calibration). A large spread means the BVH and GASP recordings are NOT
    in the same pose at those frames — the calibration is unreliable.
    """
    print(f"  Delta variance check across frames {sample_frames}:")
    bvh_idx = {n: i for i, n in enumerate(bvh_names)}
    gasp_idx = {n: i for i, n in enumerate(gasp_names)}
    for bone in bones_to_check:
        if bone not in bvh_idx or bone not in gasp_idx:
            continue
        deltas = []
        for fi in sample_frames:
            if fi >= len(bvh_rotations_all) or fi >= len(gasp_rotations_all):
                continue
            R_g = gasp_rotations_all[fi][gasp_idx[bone]]
            R_b = bvh_rotations_all[fi][bvh_idx[bone]]
            deltas.append(R_g.T @ R_b)
        if len(deltas) < 2:
            continue
        ref = deltas[0]
        max_deg = 0.0
        for D in deltas[1:]:
            dR = ref.T @ D
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
            if ang > max_deg:
                max_deg = ang
        flag = "OK" if max_deg < 2.0 else ("WARN" if max_deg < 10.0 else "BAD")
        print(f"    {bone:20s}  max Δ-angle spread = {max_deg:6.2f} deg  [{flag}]")


# ── MuJoCo Drawing ──────────────────────────────────────────────────────────

def _draw_sphere(scn, pos, size, rgba, label=""):
    if scn.ngeom >= scn.maxgeom - 1:
        return
    g = scn.geoms[scn.ngeom]
    mj_api.mjv_initGeom(
        g, type=mjt_geom.mjGEOM_SPHERE,
        size=[size, 0, 0], pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    if label:
        g.label = label
    scn.ngeom += 1


def _draw_capsule(scn, p0, p1, width, rgba):
    if scn.ngeom >= scn.maxgeom - 1:
        return
    g = scn.geoms[scn.ngeom]
    mj_api.mjv_initGeom(
        g, type=mjt_geom.mjGEOM_CAPSULE,
        size=[width, 0, 0], pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mj_api.mjv_connector(
        g, type=mjt_geom.mjGEOM_CAPSULE, width=width,
        from_=np.asarray(p0, dtype=np.float64),
        to=np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _draw_axes(scn, origin, length=0.25, width=0.003):
    o = np.asarray(origin, dtype=np.float64)
    _draw_capsule(scn, o, o + np.array([length, 0, 0]), width, [1, 0, 0, 0.9])
    _draw_capsule(scn, o, o + np.array([0, length, 0]), width, [0, 1, 0, 0.9])
    _draw_capsule(scn, o, o + np.array([0, 0, length]), width, [0, 0.3, 1, 0.9])
    _draw_sphere(scn, o + np.array([length + 0.02, 0, 0]), 0.008, [1, 0, 0, 0.9], "+X fwd")
    _draw_sphere(scn, o + np.array([0, length + 0.02, 0]), 0.008, [0, 1, 0, 0.9], "+Y left")
    _draw_sphere(scn, o + np.array([0, 0, length + 0.02]), 0.008, [0, 0.3, 1, 0.9], "+Z up")


def _draw_legend(scn, origin: np.ndarray, axis_length: float = 0.18, width: float = 0.006):
    """Draw a fixed-position legend showing what each axis color means.

    Three labeled colored capsules at ``origin`` indicating:
      X = red    Y = green    Z = blue
    """
    o = np.asarray(origin, dtype=np.float64)
    items = [
        (np.array([axis_length, 0, 0]),     [1.0, 0.0, 0.0, 1.0], "X axis = RED"),
        (np.array([0, axis_length, 0]),     [0.0, 1.0, 0.0, 1.0], "Y axis = GREEN"),
        (np.array([0, 0, axis_length]),     [0.0, 0.3, 1.0, 1.0], "Z axis = BLUE"),
    ]
    for tip_offset, rgba, label in items:
        if scn.ngeom >= scn.maxgeom - 2:
            return
        _draw_capsule(scn, o, o + tip_offset, width, rgba)
        _draw_sphere(scn, o + tip_offset * 1.15, 0.018, rgba, label)


def _draw_orientation_axes(
    scn,
    positions: np.ndarray,
    rotations: list,
    joint_names: list[str],
    offset: np.ndarray,
    axis_length: float = 0.06,
    width: float = 0.003,
    only_important: bool = True,
):
    """Draw per-joint local-frame RGB axes (X=red, Y=green, Z=blue).

    ``rotations`` is a list of 3x3 numpy rotation matrices (one per joint),
    expressing each joint's local frame in BVH/world coords. Same convention
    as step4's _draw_orientation_axes (which uses scalar-first quat).
    """
    o = np.asarray(offset, dtype=np.float64)
    axis_specs = [
        (np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.9]),
        (np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0, 0.9]),
        (np.array([0.0, 0.0, 1.0]), [0.0, 0.3, 1.0, 0.9]),
    ]
    n = len(joint_names)
    for i in range(n):
        name = joint_names[i]
        if not _is_visible(name):
            continue
        if only_important and (name not in IMPORTANT_BONES):
            continue
        base = positions[i] + o
        rot_m = rotations[i]
        for axis_vec, rgba in axis_specs:
            if scn.ngeom >= scn.maxgeom - 1:
                return
            tip = base + axis_length * (rot_m @ axis_vec)
            _draw_capsule(scn, base, tip, width, rgba)


def _is_visible(name: str) -> bool:
    return not any(kw in name.lower() for kw in HIDDEN_KEYWORDS)


def _draw_skeleton(
    scn,
    positions: np.ndarray,
    joint_names: list[str],
    offset: np.ndarray,
    jc_root, jc_imp, jc_other,
    bc_left, bc_right, bc_center,
    bone_width: float = 0.008,
    show_labels: bool = True,
):
    n = len(joint_names)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    for i in range(n):
        name = joint_names[i]
        if not _is_visible(name):
            continue
        is_imp = name in IMPORTANT_BONES
        rgba = jc_root if name == "pelvis" else (jc_imp if is_imp else jc_other)
        sz = 0.022 if name == "pelvis" else (0.016 if is_imp else 0.010)
        label = name if show_labels and (is_imp or name == "pelvis") else ""
        _draw_sphere(scn, positions[i] + offset, sz, rgba, label)

    for na, nb in BONE_CONNECTIONS:
        ia, ib = name_to_idx.get(na, -1), name_to_idx.get(nb, -1)
        if ia < 0 or ib < 0:
            continue
        is_l = nb.endswith("_l") or "_l" in nb
        is_r = nb.endswith("_r") or "_r" in nb
        rgba = bc_left if is_l else (bc_right if is_r else bc_center)
        _draw_capsule(scn, positions[ia] + offset, positions[ib] + offset, bone_width, rgba)


# ── Color Palettes ───────────────────────────────────────────────────────────

FBX_JC = [[1.0, 0.6, 0.0, 1.0], [1.0, 0.75, 0.1, 0.9], [0.7, 0.55, 0.3, 0.6]]
FBX_BC = [[0.9, 0.65, 0.0, 0.8], [1.0, 0.5, 0.0, 0.8], [0.8, 0.7, 0.1, 0.8]]

BVH_JC = [[0.1, 0.9, 0.1, 1.0], [0.2, 0.8, 0.2, 0.9], [0.4, 0.7, 0.4, 0.6]]
BVH_BC = [[0.2, 0.7, 0.2, 0.8], [0.2, 0.7, 0.2, 0.8], [0.2, 0.8, 0.2, 0.8]]

GASP_JC = [[0.0, 0.9, 0.9, 1.0], [0.1, 0.7, 0.9, 0.9], [0.3, 0.6, 0.7, 0.6]]
GASP_BC = [[0.0, 0.7, 0.9, 0.8], [0.0, 0.9, 0.7, 0.8], [0.1, 0.6, 0.8, 0.8]]


# ── Minimal MJCF ─────────────────────────────────────────────────────────────

MINIMAL_MJCF = """
<mujoco>
  <visual>
    <map znear="0.001" zfar="100"/>
    <quality shadowsize="2048"/>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.18 0.18 0.18" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".55 .55 .55" rgb2=".4 .4 .4"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="2 2 3" dir="-1 -1 -1" diffuse=".18 .18 .18" specular="0 0 0"/>
    <geom name="floor" type="plane" size="5 5 0.01" material="grid_mat" pos="1.5 0 0"/>
  </worldbody>
</mujoco>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare FBX / BVH / GASP skeletons")
    parser.add_argument("--fbx", default=DEFAULT_FBX)
    parser.add_argument("--bvh", default=DEFAULT_BVH)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL,
                        help="GASP animation JSONL to retarget and visualize")
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--gasp-calib-jsonl", default=DEFAULT_CALIB_JSONL,
                        help="GASP JSONL used to calibrate per-bone Δ vs --bvh frame "
                             "--calib-frame. Defaults to --jsonl (animation file). "
                             "Δ is a structural constant; any frame where BVH and "
                             "GASP are in approx. the same pose works.")
    parser.add_argument("--gasp-calib-meta", default=DEFAULT_CALIB_META)
    parser.add_argument("--calib-frame", type=int, default=0,
                        help="Frame index used in BOTH --bvh and --gasp-calib-jsonl "
                             "for Δ computation (default 0)")
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sep", type=float, default=0, help="X separation (m)")
    args = parser.parse_args()

    sep = args.sep

    # ── 1. FBX via Blender ────────────────────────────────────────────────
    print(f"=== Source 1: FBX (Step1) ===")
    print(f"  Path: {args.fbx}")
    fbx_data = extract_fbx_via_blender(args.fbx, args.blender)
    fbx_names = fbx_data["bone_names"]
    fbx_parents = fbx_data["parent_indices"]
    n_fbx_bones = len(fbx_names)
    n_fbx_frames = fbx_data["num_frames"]
    print(f"  Bones: {n_fbx_bones}, Frames: {n_fbx_frames}")

    fbx_positions_mj = np.zeros((n_fbx_frames, n_fbx_bones, 3))
    for fi in range(n_fbx_frames):
        pos_m = np.array(fbx_data["positions"][fi], dtype=np.float64)
        fbx_positions_mj[fi] = blender_m_to_mujoco(pos_m)

    # Verify key positions
    if "pelvis" in fbx_names:
        pi = fbx_names.index("pelvis")
        print(f"  pelvis frame0 (MuJoCo m): {fbx_positions_mj[0, pi]}")

    # ── 2. BVH (Step2) ───────────────────────────────────────────────────
    print(f"\n=== Source 2: BVH (Step2) ===")
    print(f"  Path: {args.bvh}")
    bvh = parse_bvh(args.bvh)
    bvh_names = [j.name for j in bvh.joints]
    bvh_parents = [j.parent_idx for j in bvh.joints]
    n_bvh_bones = len(bvh_names)
    n_bvh_frames = bvh.num_frames
    print(f"  Bones: {n_bvh_bones}, Frames: {n_bvh_frames}, FPS: {1/bvh.frame_time:.0f}")

    # Compute FK per frame: gpos (cm) and grot (3x3, world-frame rotation matrix)
    # Both expressed in BVH native coords: X=skel_left, Y=skel_fwd, Z=up, RH.
    # MuJoCo renders these directly without any axis swap (only cm -> m).
    bvh_positions_mj = np.zeros((n_bvh_frames, n_bvh_bones, 3))
    bvh_rotations_all: list[list[np.ndarray]] = []
    for fi in range(n_bvh_frames):
        gpos, grot = compute_fk(bvh, fi)
        bvh_positions_mj[fi] = bvh_cm_to_mujoco(gpos)
        bvh_rotations_all.append(grot)

    if "pelvis" in bvh_names:
        pi = bvh_names.index("pelvis")
        print(f"  pelvis frame0 (MuJoCo m): {bvh_positions_mj[0, pi]}")

    # ── 2b. Build retarget-ready frames (matches step4_retarget_bvh_to_g1.py) ──
    # Each frame is a dict: {bone_name: [pos_m (3,), quat_wxyz (4,)]}
    # Plus synthesized LeftFootMod/RightFootMod (foot pos + ball orientation) so
    # the GMR config "bvh_ue5_native" has the IK targets it expects.
    print(f"  Building retarget input dicts (pos+quat) ...")
    bvh_retarget_frames: list[dict] = []
    for fi in range(n_bvh_frames):
        gpos = bvh_positions_mj[fi] / 1.0  # already meters
        grot = bvh_rotations_all[fi]
        result: dict = {}
        for bi, joint in enumerate(bvh.joints):
            pos = gpos[bi]
            quat = R.from_matrix(grot[bi]).as_quat(scalar_first=True)  # wxyz
            result[joint.name] = [pos.copy(), quat]
        if "foot_l" in result and "ball_l" in result:
            result["LeftFootMod"] = [result["foot_l"][0].copy(), result["ball_l"][1].copy()]
        if "foot_r" in result and "ball_r" in result:
            result["RightFootMod"] = [result["foot_r"][0].copy(), result["ball_r"][1].copy()]
        bvh_retarget_frames.append(result)

    # Compute human height from rest pose (BVH offsets, identity rotations).
    # head bone Z + ~9cm gives top of skull. Same formula as step4.
    rest_pos = np.zeros((n_bvh_bones, 3))
    for bi, joint in enumerate(bvh.joints):
        if joint.parent_idx < 0:
            rest_pos[bi] = joint.offset
        else:
            rest_pos[bi] = rest_pos[joint.parent_idx] + joint.offset
    bvh_name_to_idx = {j.name: i for i, j in enumerate(bvh.joints)}
    head_rest_z = rest_pos[bvh_name_to_idx["head"]][2] * 0.01 if "head" in bvh_name_to_idx else 1.7
    bvh_human_height = head_rest_z + 0.09
    print(f"  Human height (head_z + 0.09m): {bvh_human_height:.3f}m")
    print(f"  Sample retarget keys (frame0, first 6): {list(bvh_retarget_frames[0].keys())[:6]} ...")
    pelvis_pq = bvh_retarget_frames[0].get("pelvis")
    if pelvis_pq is not None:
        print(f"  pelvis frame0  pos={pelvis_pq[0]}  quat(wxyz)={pelvis_pq[1]}")

    # ── 2c. Run GMR retarget BVH → G1 (optional, requires mink) ────────────
    bvh_qpos_all = None
    bvh_scaled_data_all: list[dict] | None = None
    try:
        # Try local sibling GMR first, then UEGMR-side GMR
        for gmr_root in (
            HERE.parent / "gmr",                        # release: DataLib/gmr
            HERE.parent / "UEGMR" / "GMR",
            HERE.parents[3] / "UEGMR" / "GMR",
        ):
            if gmr_root.exists() and str(gmr_root) not in sys.path:
                sys.path.insert(0, str(gmr_root))
        from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore

        print(f"\n  Initializing GMR (src=bvh_ue5_native, tgt=unitree_g1) ...")
        bvh_retargeter = GMR(
            src_human="bvh_ue5_native",
            tgt_robot="unitree_g1",
            actual_human_height=bvh_human_height,
        )
        ik_bones = sorted(bvh_retargeter.human_scale_table.keys())
        print(f"  IK bones expected: {ik_bones}")
        missing = [b for b in ik_bones if b not in bvh_retarget_frames[0]]
        if missing:
            print(f"  [WARN] Missing bones in retarget input: {missing}")

        # Warm up IK on frame 0 so the first stored qpos is converged.
        # Without this, frame 0 starts from G1's default rest pose (single IK
        # call doesn't fully converge) — both pose and pelvis xyz come out off,
        # which then poisons any offset computed from gasp_qpos_all[0][:3].
        print(f"  Warming up IK on frame 0 (BVH) ...")
        for _ in range(20):
            bvh_retargeter.retarget(bvh_retarget_frames[0])

        print(f"  Retargeting {n_bvh_frames} frames ...")
        bvh_qpos_all = []
        bvh_scaled_data_all = []
        for fi in range(n_bvh_frames):
            qpos = bvh_retargeter.retarget(bvh_retarget_frames[fi])
            bvh_qpos_all.append(qpos.copy())
            bvh_scaled_data_all.append(
                {k: [v[0].copy(), v[1].copy()] for k, v in bvh_retargeter.scaled_human_data.items()}
            )
        print(f"  Retarget OK. qpos dim = {len(bvh_qpos_all[0])}, frames = {len(bvh_qpos_all)}")
    except ModuleNotFoundError as e:
        print(f"  [WARN] GMR retarget skipped: missing dependency '{e.name}' (try: pip install mink)")
    except Exception as e:
        print(f"  [WARN] GMR retarget skipped: {type(e).__name__}: {e}")

    # ── 3. GASP JSONL → World Space ──────────────────────────────────────
    print(f"\n=== Source 3: GASP animation JSONL → World Space ===")
    print(f"  Path: {args.jsonl}")
    gasp_positions_mj, gasp_rotations_all, gasp_names, gasp_parents = (
        load_gasp_data(args.jsonl, args.meta)
    )
    n_gasp_bones = len(gasp_names)
    n_gasp_frames = gasp_positions_mj.shape[0]
    print(f"  Bones: {n_gasp_bones}, Frames: {n_gasp_frames}")

    if "pelvis" in gasp_names:
        pi = gasp_names.index("pelvis")
        print(f"  pelvis frame0 (MuJoCo m): {gasp_positions_mj[0, pi]}")

    # ── 3b. GASP calibration data (same animation as BVH) ─────────────────
    # Used to compute per-bone Δ_i = R_gasp_calib_i^(-1) · R_bvh_calib_i.
    # The animation in --gasp-calib-jsonl MUST match --bvh frame-for-frame
    # (typically: the M_Neutral_Stand_Idle_Loop animation recorded both ways).
    print(f"\n=== Source 3b: GASP calibration JSONL ===")
    print(f"  Path: {args.gasp_calib_jsonl}")
    bone_delta: dict[str, np.ndarray] | None = None
    try:
        if not os.path.exists(args.gasp_calib_jsonl):
            raise FileNotFoundError(args.gasp_calib_jsonl)
        calib_pos_mj, calib_rot_all, calib_names, _ = load_gasp_data(
            args.gasp_calib_jsonl, args.gasp_calib_meta
        )
        print(f"  Calib frames: {calib_pos_mj.shape[0]}, bones: {len(calib_names)}")

        cf = max(0, min(args.calib_frame, calib_pos_mj.shape[0] - 1, n_bvh_frames - 1))
        print(f"  Using calibration frame index = {cf}")

        bone_delta = compute_per_bone_delta(
            bvh_rotations_all[cf], bvh_names,
            calib_rot_all[cf],     calib_names,
        )
        print(f"  Computed Δ for {len(bone_delta)} shared bones.")

        # Sanity: print Δ-angle spread for the IK-relevant bones across a few frames
        ik_bones_to_check = [
            "pelvis", "spine_05",
            "thigh_l", "calf_l", "foot_l", "ball_l",
            "thigh_r", "calf_r", "foot_r", "ball_r",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
        ]
        max_n = min(len(calib_rot_all), n_bvh_frames)
        sample_frames = sorted({0, max_n // 4, max_n // 2, 3 * max_n // 4, max_n - 1})
        sample_frames = [f for f in sample_frames if f >= 0]
        report_delta_variance(
            bvh_rotations_all, bvh_names,
            calib_rot_all,    calib_names,
            ik_bones_to_check, sample_frames,
        )
    except FileNotFoundError as e:
        print(f"  [WARN] Calibration JSONL not found: {e}. Δ defaulting to identity.")
    except Exception as e:
        print(f"  [WARN] Calibration failed: {type(e).__name__}: {e}. Δ defaulting to identity.")

    # Sanity-check: pelvis quat from frame 0 of GASP-anim vs BVH-anim
    if "pelvis" in gasp_names and "pelvis" in bvh_names:
        pi = gasp_names.index("pelvis")
        pelvis_quat_wxyz = R.from_matrix(gasp_rotations_all[0][pi]).as_quat(scalar_first=True)
        print(f"  GASP pelvis frame0 quat(wxyz, BVH coords): {pelvis_quat_wxyz}")
        bp = bvh_name_to_idx["pelvis"]
        bvh_pelvis_quat = R.from_matrix(bvh_rotations_all[0][bp]).as_quat(scalar_first=True)
        print(f"  BVH  pelvis frame0 quat(wxyz, BVH coords): {bvh_pelvis_quat}")

    # ── 3c. Build GASP retarget input (apply Δ to "fake" BVH-equivalent) ──
    # quat_for_gmr_i = R.from_matrix(R_gasp_world_i · Δ_i)
    # pos_for_gmr_i  = gasp_positions_mj[fi, gi]   (already in BVH coords, m)
    # The bvh_ue5_native config is reused as-is.
    print(f"\n  Building GASP retarget input dicts (pos+quat) ...")
    identity3 = np.eye(3)
    gasp_retarget_frames: list[dict] = []
    for fi in range(n_gasp_frames):
        result: dict = {}
        for gi, name in enumerate(gasp_names):
            R_g = gasp_rotations_all[fi][gi]
            D = bone_delta.get(name, identity3) if bone_delta is not None else identity3
            R_eq = R_g @ D
            quat = R.from_matrix(R_eq).as_quat(scalar_first=True)  # wxyz
            result[name] = [gasp_positions_mj[fi, gi].copy(), quat]
        if "foot_l" in result and "ball_l" in result:
            result["LeftFootMod"] = [result["foot_l"][0].copy(), result["ball_l"][1].copy()]
        if "foot_r" in result and "ball_r" in result:
            result["RightFootMod"] = [result["foot_r"][0].copy(), result["ball_r"][1].copy()]
        gasp_retarget_frames.append(result)

    # ── 3d. Run GMR retarget on GASP frames ───────────────────────────────
    gasp_qpos_all: list[np.ndarray] | None = None
    try:
        # GMR module already imported above (when BVH retarget succeeded). If
        # it failed, this import will fail too — handled below.
        from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore

        print(f"\n  Initializing GMR for GASP (src=bvh_ue5_native, tgt=unitree_g1) ...")
        gasp_retargeter = GMR(
            src_human="bvh_ue5_native",
            tgt_robot="unitree_g1",
            actual_human_height=bvh_human_height,
        )
        ik_bones = sorted(gasp_retargeter.human_scale_table.keys())
        missing = [b for b in ik_bones if b not in gasp_retarget_frames[0]]
        if missing:
            print(f"  [WARN] Missing bones in GASP retarget input: {missing}")

        # Warm up IK on frame 0 (see BVH retarget above for rationale).
        print(f"  Warming up IK on frame 0 (GASP) ...")
        for _ in range(20):
            gasp_retargeter.retarget(gasp_retarget_frames[0])

        print(f"  Retargeting {n_gasp_frames} GASP frames ...")
        gasp_qpos_all = []
        for fi in range(n_gasp_frames):
            qpos = gasp_retargeter.retarget(gasp_retarget_frames[fi])
            gasp_qpos_all.append(qpos.copy())
        print(f"  GASP retarget OK. qpos dim = {len(gasp_qpos_all[0])}, frames = {len(gasp_qpos_all)}")
    except ModuleNotFoundError as e:
        print(f"  [WARN] GASP retarget skipped: missing dependency '{e.name}'")
    except Exception as e:
        print(f"  [WARN] GASP retarget skipped: {type(e).__name__}: {e}")

    # ── Center each skeleton at origin XY ────────────────────────────────
    def _pelvis_offset(positions_mj, names, x_shift):
        pi = names.index("pelvis") if "pelvis" in names else 0
        root0 = positions_mj[0, pi].copy()
        return np.array([-root0[0] + x_shift, -root0[1], 0.0])

    off_fbx  = _pelvis_offset(fbx_positions_mj, fbx_names, 0.0)
    off_bvh  = _pelvis_offset(bvh_positions_mj, bvh_names, sep)
    off_gasp = _pelvis_offset(gasp_positions_mj, gasp_names, sep * 2)

    print(f"\n  Offsets: FBX={off_fbx}, BVH={off_bvh}, GASP={off_gasp}")

    # ── Check FBX vs BVH match ───────────────────────────────────────────
    if "pelvis" in fbx_names and "pelvis" in bvh_names:
        fp = fbx_positions_mj[0, fbx_names.index("pelvis")]
        bp = bvh_positions_mj[0, bvh_names.index("pelvis")]
        diff = np.linalg.norm(fp - bp)
        print(f"  FBX vs BVH pelvis diff: {diff*100:.4f} cm {'OK' if diff < 0.005 else 'MISMATCH!'}")

    # ── MuJoCo Viewer ────────────────────────────────────────────────────
    total_frames = max(n_fbx_frames, n_bvh_frames, n_gasp_frames)
    # g1_source: "gasp" drives G1 from GASP-retargeted qpos; "bvh" from BVH-retargeted qpos.
    initial_g1_source = "gasp" if gasp_qpos_all is not None else "bvh"
    state = {
        "paused": True, "fi": 0, "step": 0,
        "labels": True, "axes": True, "g1": True,
        "g1_source": initial_g1_source,
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
        elif k == 76:      # L
            state["labels"] = not state["labels"]
        elif k == 79:      # O
            state["axes"] = not state["axes"]
            print(f"  [Orientation Axes {'ON' if state['axes'] else 'OFF'}]")
        elif k == 71:      # G
            state["g1"] = not state["g1"]
            print(f"  [G1 robot {'ON' if state['g1'] else 'OFF'}]")
        elif k == 86:      # V
            state["g1_source"] = "bvh" if state["g1_source"] == "gasp" else "gasp"
            print(f"  [G1 source = {state['g1_source'].upper()}]")

    # Try loading G1 robot XML as the main MJCF when *either* retarget succeeded.
    # We wrap it in a scene XML that adds a checker floor and lights, so the
    # 4 columns (FBX/BVH/GASP/G1) all share the same ground plane.
    g1_model = None
    # Per-source initial XY anchor — each source's frame-0 pelvis lands at column 4.
    bvh_g1_offset  = np.zeros(3)
    gasp_g1_offset = np.zeros(3)
    if bvh_qpos_all is not None:
        bvh_g1_offset  = np.array([sep * 3 - bvh_qpos_all[0][0],  -bvh_qpos_all[0][1],  0.0])
    if gasp_qpos_all is not None:
        gasp_g1_offset = np.array([sep * 3 - gasp_qpos_all[0][0], -gasp_qpos_all[0][1], 0.0])
    # Reference qpos used for nq sanity check.
    ref_qpos_all = bvh_qpos_all if bvh_qpos_all is not None else gasp_qpos_all
    if ref_qpos_all is not None:
        try:
            from general_motion_retargeting.params import ROBOT_XML_DICT  # type: ignore
            g1_xml_path = pathlib.Path(str(ROBOT_XML_DICT["unitree_g1"]))
            wrapper_xml = f"""
<mujoco>
  <include file=\"{g1_xml_path.name}\"/>
  <visual>
    <map znear=\"0.001\" zfar=\"100\"/>
    <quality shadowsize=\"2048\"/>
    <headlight diffuse=\"0.35 0.35 0.35\" ambient=\"0.18 0.18 0.18\" specular=\"0.05 0.05 0.05\"/>
  </visual>
  <asset>
    <texture name=\"grid\" type=\"2d\" builtin=\"checker\" rgb1=\".55 .55 .55\" rgb2=\".4 .4 .4\"
             width=\"512\" height=\"512\"/>
    <material name=\"grid_mat\" texture=\"grid\" texrepeat=\"8 8\" reflectance=\"0.05\"/>
  </asset>
  <worldbody>
    <light pos=\"2 2 3\" dir=\"-1 -1 -1\" diffuse=\".18 .18 .18\" specular=\"0 0 0\"/>
    <geom name=\"viz_floor\" type=\"plane\" size=\"6 4 0.01\"
          material=\"grid_mat\" pos=\"{sep * 1.5} 0 0\"/>
  </worldbody>
</mujoco>
"""
            wrapper_path = g1_xml_path.parent / "_vis_compare_g1_wrapper.xml"
            wrapper_path.write_text(wrapper_xml, encoding="utf-8")
            try:
                g1_model = mj_api.MjModel.from_xml_path(str(wrapper_path))
            finally:
                wrapper_path.unlink(missing_ok=True)
            assert g1_model.nq == len(ref_qpos_all[0]), \
                f"G1 nq={g1_model.nq} vs qpos len={len(ref_qpos_all[0])}"
            print(f"  Loaded G1 model (wrapped + floor): {g1_xml_path}")
            print(f"  G1 nq={g1_model.nq}, nu={g1_model.nu}")
            print(f"  BVH-G1  initial offset: {bvh_g1_offset}")
            print(f"  GASP-G1 initial offset: {gasp_g1_offset}")
        except Exception as e:
            print(f"  [WARN] Could not load G1 model: {type(e).__name__}: {e}")
            g1_model = None

    if g1_model is not None:
        model = g1_model
        data = mj_api.MjData(model)
    else:
        model = mj_api.MjModel.from_xml_string(MINIMAL_MJCF)
        data = mj_api.MjData(model)
    mj_api.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
        key_callback=key_cb,
    )

    # Camera: span across 4 columns when G1 is shown, else 3 columns.
    cam_x = sep * (1.5 if g1_model is not None else 1.0)
    viewer.cam.lookat = np.array([cam_x, 0.0, 0.85])
    viewer.cam.distance = 6.5 if g1_model is not None else 5.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 180

    frame_dt = 1.0 / args.fps
    last_t = time.time()

    print(f"\n  Controls: Space=pause  Left/Right=step  Backspace=reset")
    print(f"            L=labels  O=orientation axes  G=toggle G1  V=switch G1 source (BVH↔GASP)")
    print(f"  COL 1 (Gold)  = FBX Step1")
    print(f"  COL 2 (Green) = BVH Step2  (with per-joint RGB orientation axes; X=red Y=green Z=blue)")
    print(f"  COL 3 (Cyan)  = GASP World  (with per-joint RGB orientation axes)")
    if g1_model is not None:
        print(f"  COL 4 (G1 mesh) = retargeted to Unitree G1, source = {state['g1_source'].upper()}")
        print(f"                    (BVH-driven: idle loop;  GASP-driven: --jsonl animation via Δ calibration)")
    else:
        print(f"  [G1 column disabled — install mink and ensure GMR is importable]")
    print(f"  Started PAUSED. Press Space to play.\n")

    while viewer.is_running():
        if state["step"] != 0:
            state["fi"] = (state["fi"] + state["step"]) % total_frames
            state["step"] = 0
        elif not state["paused"]:
            now = time.time()
            if now - last_t >= frame_dt:
                state["fi"] = (state["fi"] + 1) % total_frames
                last_t = now
            else:
                time.sleep(0.001)
                continue

        fi = state["fi"]
        show = state["labels"]

        # Drive G1 robot qpos from the selected source (BVH-retargeted or GASP-retargeted)
        if g1_model is not None:
            src = state["g1_source"]
            if src == "gasp" and gasp_qpos_all is None:
                src = "bvh"
            if src == "bvh" and bvh_qpos_all is None:
                src = "gasp"

            qpos_src = gasp_qpos_all if src == "gasp" else bvh_qpos_all
            n_src = n_gasp_frames if src == "gasp" else n_bvh_frames
            # Per-source offset so each animation's frame 0 lands at column 4.
            off = gasp_g1_offset if src == "gasp" else bvh_g1_offset

            if qpos_src is not None and state["g1"]:
                qpos = qpos_src[fi % n_src].copy()
                qpos[:3] += off
                data.qpos[:] = qpos
                mj_api.mj_forward(model, data)
            elif qpos_src is not None:
                # Hide G1 by sending it far away
                data.qpos[:] = qpos_src[0]
                data.qpos[2] = -10.0
                mj_api.mj_forward(model, data)

        scn = viewer.user_scn
        if scn is not None:
            scn.ngeom = 0

            # World axes at each skeleton base
            _draw_axes(scn, off_fbx)
            _draw_axes(scn, np.array([sep, 0, 0]))
            _draw_axes(scn, np.array([sep * 2, 0, 0]))
            if g1_model is not None:
                _draw_axes(scn, np.array([sep * 3, 0, 0]))

            # Color legend (fixed position, top-left of scene relative to camera)
            _draw_legend(scn, np.array([-0.6, -0.6, 1.6]), axis_length=0.18, width=0.006)

            # 1. FBX skeleton (gold)
            _draw_skeleton(
                scn, fbx_positions_mj[fi % n_fbx_frames], fbx_names, off_fbx,
                *FBX_JC, *FBX_BC, show_labels=show,
            )

            # 2. BVH skeleton (green) + per-joint orientation axes (RGB)
            bvh_fi = fi % n_bvh_frames
            _draw_skeleton(
                scn, bvh_positions_mj[bvh_fi], bvh_names, off_bvh,
                *BVH_JC, *BVH_BC, show_labels=show,
            )
            if state["axes"]:
                _draw_orientation_axes(
                    scn,
                    bvh_positions_mj[bvh_fi],
                    bvh_rotations_all[bvh_fi],
                    bvh_names,
                    off_bvh,
                    axis_length=0.06,
                    width=0.003,
                    only_important=True,
                )

            # 3. GASP skeleton (cyan) + per-joint RGB orientation axes
            gasp_fi = fi % n_gasp_frames
            _draw_skeleton(
                scn, gasp_positions_mj[gasp_fi], gasp_names, off_gasp,
                *GASP_JC, *GASP_BC, show_labels=show,
            )
            if state["axes"]:
                _draw_orientation_axes(
                    scn,
                    gasp_positions_mj[gasp_fi],
                    gasp_rotations_all[gasp_fi],
                    gasp_names,
                    off_gasp,
                    axis_length=0.06,
                    width=0.003,
                    only_important=True,
                )

            # Title labels
            _draw_sphere(scn, off_fbx + np.array([0, 0, 1.75]),
                         0.008, [1, 0.7, 0, 0.9], "FBX Step1")
            _draw_sphere(scn, np.array([sep, 0, 1.75]),
                         0.008, [0.2, 0.8, 0.2, 0.9], "BVH Step2")
            _draw_sphere(scn, np.array([sep * 2, 0, 1.75]),
                         0.008, [0, 0.8, 0.8, 0.9], "GASP World")
            if g1_model is not None:
                src_label = state["g1_source"].upper()
                _draw_sphere(scn, np.array([sep * 3, 0, 1.75]),
                             0.008, [0.9, 0.2, 0.6, 0.9], f"G1 Retarget [src={src_label}]")

        viewer.sync()

        if state["paused"] and state["step"] == 0:
            time.sleep(0.01)

    viewer.close()
    print("Done.")


if __name__ == "__main__":
    main()
