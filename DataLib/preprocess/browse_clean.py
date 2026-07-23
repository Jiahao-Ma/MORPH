#!/usr/bin/env python3
"""Browse the cleaned MorphData_v1 segments in ONE persistent MuJoCo G1 viewer.

Reuses the MORPH retarget pipeline (DataLib/retarget/ue_world_skeleton_retarget.py)
to retarget each cleaned segment to G1 qpos on the fly, and plays them in a
single passive MuJoCo viewer with keys to JUMP between trajectories:

  Space       pause / play
  Left/Right  step one frame
  Backspace   reset to frame 0
  N           next trajectory        <-- the "jump to next clip" option
  P           previous trajectory
  T           toggle trajectory overlay
  O           toggle orientation arrows
  V           toggle velocity arrows (green + red)
  Esc / Q     quit

The next segment is retargeted in a background thread while the current one
plays, so jumping with N is usually instant (the viewer keeps rendering while
a not-yet-ready segment is being prepared). A checkerboard GROUND plane is
always shown. Pass --terrain-dir to also bake each clip's TERRAIN (from its
meta `terrain_ref`) into the scene; in terrain mode the viewer relaunches per
clip so the terrain matches the motion (the full terrain set is not vendored;
extract MorphDataTerrain_v1.zip into a folder and point --terrain-dir there).

Run from the Morph/ directory:
  python DataLib/preprocess/browse_clean.py --cat traversal_mantle
  python DataLib/preprocess/browse_clean.py --cat ground --limit 20
  python DataLib/preprocess/browse_clean.py --cat stairs --terrain-dir data/sample/terrain
  python DataLib/preprocess/browse_clean.py --cat traversal_vault --viewer viser \
      --terrain-dir data/MorphData_v1/terrain
  python DataLib/preprocess/browse_clean.py --all --shuffle

--viewer mujoco (default) opens a native MuJoCo window. --viewer viser starts a
web 3D server (http://localhost:<port>) that swaps terrain+robot in place on
trajectory switch (no window relaunch, even with terrain). Needs `pip install viser`.
"""
import sys
import json
import time
import glob
import pathlib
import argparse
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
RETARGET_DIR = HERE.parent / "retarget"
sys.path.insert(0, str(RETARGET_DIR))
import ue_world_skeleton_retarget as rt  # noqa: E402

DATA_ROOT = (HERE.parent.parent / "data" / "MorphData_v1").resolve()
CONFIG = RETARGET_DIR / "gasp_bvh_alignment_g1_height.json"
SRC_HUMAN = "bvh_ue5_g1scale"

# Per-category data-scale (see README §3): ground/stairs already G1 (1.0),
# traversal_* captured at original UE height -> 0.77.
CATEGORY_SCALE = {
    "ground": 1.0,
    "stairs": 1.0,
    "traversal_mantle": 0.77,
    "traversal_mantle_vault": 0.77,
    "traversal_vault": 0.77,
}
CATEGORIES = list(CATEGORY_SCALE.keys())


def retarget_segment(jsonl, meta, data_scale, smooth_win=9, per_foot=True,
                     height_from_data=True, terrain_dir=None,
                     terrain_ground_z0=0.0):
    """Run the S5 retarget flow (same as ue_world_skeleton_retarget.main s5)
    and return the artifacts needed for visualization. No file export.
    If terrain_dir is given, also loads the segment's terrain (from its
    meta `terrain_ref`) through the same transform pipeline. Returns dict:
    qpos, fps, look_fwd_xy, des_vel, move_input, terrain."""
    jsonl = str(jsonl); meta = str(meta)
    bone_names, _ = rt.load_meta(meta)
    frames = rt.load_frames(jsonl)
    n = len(frames)
    if n < 2:
        raise ValueError(f"too few frames ({n}) in {jsonl}")
    skel_ue = np.zeros((n, len(bone_names), 3), dtype=np.float64)
    for fi, fr in enumerate(frames):
        skel_ue[fi] = rt.skeleton_world_direct(fr)
    if abs(data_scale - 1.0) > 1e-9:
        skel_ue *= data_scale

    R_global, t_global, delta, cfg = rt.load_alignment_config(str(CONFIG))
    skel_mj = np.zeros_like(skel_ue)
    skel_kabsch = np.zeros_like(skel_ue)
    for fi in range(n):
        skel_mj[fi] = rt.ue_to_mujoco(skel_ue[fi])
        skel_kabsch[fi] = rt.apply_kabsch(skel_mj[fi], R_global, t_global)

    _, gasp_rot_all, _, _ = rt.load_gasp_data(
        jsonl, meta, preserve_world_z=False, ground_z_cm=0.0)
    gmr_frames, root_xy = rt.build_gmr_frames(
        skel_mj, gasp_rot_all, bone_names, R_global, t_global, delta,
        recenter_root_xy=True)
    if smooth_win >= 3 and root_xy.shape[0] >= 3:
        root_xy = root_xy.copy()
        root_xy[:, 0] = rt.smooth_1d(root_xy[:, 0], smooth_win)
        root_xy[:, 1] = rt.smooth_1d(root_xy[:, 1], smooth_win)

    if height_from_data:
        h = rt.estimate_human_height(gmr_frames)
    else:
        h_cfg = cfg.get("actual_human_height_m")
        h = float(h_cfg) if h_cfg is not None else rt.estimate_human_height(gmr_frames)

    name_to_idx = {nm: i for i, nm in enumerate(bone_names)}
    ball_idx = {nm: name_to_idx[nm] for nm in ("ball_l", "ball_r") if nm in name_to_idx}
    scaled_skel = None
    scaled_names = None
    scaled_toe = None
    try:
        scaled_skel, scaled_names, probe, scaled_toe = rt.scaled_human_skeleton(
            gmr_frames, h, root_xy, src_human=SRC_HUMAN)
        if "ball_l" in ball_idx and "ball_r" in ball_idx and np.isfinite(scaled_toe).all():
            k = 50.0
            blue_l = skel_kabsch[:, ball_idx["ball_l"], 2]
            blue_r = skel_kabsch[:, ball_idx["ball_r"], 2]
            red_l, red_r = scaled_toe[:, 0, 2], scaled_toe[:, 1, 2]
            dz = rt.support_soft_ground_dz(blue_l, blue_r, red_l, red_r, k)
            if smooth_win >= 3:
                dz = rt.smooth_1d(dz, smooth_win)
            # Ground the visualised scaled skeleton + toe in Z (world units),
            # matching verify_pipeline s5.1, so the red IK-input sits on the
            # terrain instead of sinking through it.
            scaled_skel[:, :, 2] += dz[:, None]
            scaled_toe[:, :, 2] += dz[:, None]
            s_root = float(probe.human_scale_table.get(probe.human_root_name, 1.0))
            if abs(s_root) > 1e-8:
                dz_input = dz / s_root
                for fi, fr in enumerate(gmr_frames):
                    sh = float(dz_input[fi])
                    for nm in fr:
                        fr[nm][0][2] += sh
    except Exception as e:
        print(f"  [pre-IK grounding] FAILED ({type(e).__name__}: {e}); skip.")

    qpos_arr, retargeter = rt.retarget_to_g1(
        gmr_frames, actual_human_height=h, src_human=SRC_HUMAN)
    g1_model = retargeter.model
    qpos_arr[:, 0] += root_xy[:, 0]
    qpos_arr[:, 1] += root_xy[:, 1]
    try:
        qpos_arr, _ = rt.apply_s5_grounding(
            qpos_arr, g1_model, skel_kabsch, ball_idx, n,
            extra_drop=0.031, no_ball_align=False, per_foot=per_foot,
            ground_softness=50.0, smooth_win=smooth_win)
    except Exception as e:
        print(f"  [S5 grounding] FAILED ({type(e).__name__}: {e}); skip.")

    look_fwd_xy = None; des_vel = None; move_input = None
    try:
        look_yaw = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                look_yaw.append(float((json.loads(line).get("cmd") or {}).get("lookY", 0.0)))
        cmd_arr_full = rt.load_gasp_command_data(jsonl)
        m = min(cmd_arr_full.shape[0], qpos_arr.shape[0])
        cmd_arr = cmd_arr_full[:m].copy()
        look_yaw_arr = np.asarray(look_yaw[:m], dtype=np.float32)
        look_fwd_xy = rt.compute_look_fwd_xy(look_yaw_arr, R_global)
        des_vel = cmd_arr["des_vel"]
        move_input = cmd_arr["move_input"]
    except Exception as e:
        print(f"  [cmd] FAILED ({type(e).__name__}: {e}); arrows disabled.")

    fps = 60.0
    try:
        fps = float(json.load(open(meta)).get("sample_rate_hz", 60.0))
    except Exception:
        pass

    terrain = None
    if terrain_dir:
        ns = argparse.Namespace(terrain=None, terrain_dir=str(terrain_dir),
                                meta=meta)
        tpath = rt._resolve_terrain_path(ns)
        if tpath:
            try:
                terrain = rt.load_terrain_world(
                    tpath, R_global, t_global,
                    ground_z0_cm=terrain_ground_z0,
                    data_scale=data_scale)
            except Exception as e:
                print(f"  [terrain] FAILED ({type(e).__name__}: {e}); "
                      f"visualizing without terrain.")
                terrain = None
    return dict(qpos=qpos_arr, fps=fps, look_fwd_xy=look_fwd_xy,
                des_vel=des_vel, move_input=move_input, g1_model=g1_model,
                terrain=terrain,
                scaled_skel=scaled_skel, scaled_names=scaled_names,
                scaled_toe=scaled_toe,
                kabsch_skel=skel_kabsch, kabsch_names=bone_names)


def build_playlist(cats, data_root=DATA_ROOT, limit=None, shuffle=False, seed=0):
    items = []
    for cat in cats:
        scale = CATEGORY_SCALE[cat]
        files = sorted(glob.glob(str(data_root / cat / "*_frames.jsonl")))
        for fp in files:
            meta = fp.replace("_frames.jsonl", "_meta.json")
            if not pathlib.Path(meta).exists():
                continue
            items.append(dict(cat=cat, jsonl=fp, meta=meta, scale=scale,
                              name=pathlib.Path(fp).stem))
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(items)
    if limit:
        items = items[:limit]
    return items


class Prefetcher:
    """Retarget segment indices in a single background thread. The main viewer
    thread never runs retarget concurrently, so GMR is single-threaded."""

    def __init__(self, playlist, terrain_dir=None, terrain_ground_z0=0.0):
        self.playlist = playlist
        self.terrain_dir = terrain_dir
        self.terrain_ground_z0 = terrain_ground_z0
        self.cache = {}            # idx -> result dict | Exception
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.wanted = None        # idx the worker should retarget next
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _retarget(self, idx):
        it = self.playlist[idx]
        return retarget_segment(it["jsonl"], it["meta"], it["scale"],
                                terrain_dir=self.terrain_dir,
                                terrain_ground_z0=self.terrain_ground_z0)

    def _loop(self):
        while True:
            with self.cond:
                while self.wanted is None or self.wanted in self.cache:
                    self.cond.wait()
                idx = self.wanted
            try:
                res = self._retarget(idx)
            except Exception as e:
                res = e
                print(f"  [prefetch] retarget failed for #{idx} "
                      f"({self.playlist[idx]['name']}): {type(e).__name__}: {e}")
            with self.cond:
                self.cache[idx] = res
                self.cond.notify_all()

    def request(self, idx):
        if idx is None or idx < 0 or idx >= len(self.playlist):
            return
        with self.cond:
            if idx not in self.cache and self.wanted != idx:
                self.wanted = idx
                self.cond.notify_all()

    def has(self, idx):
        with self.cond:
            return idx in self.cache and not isinstance(self.cache[idx], Exception)

    def get(self, idx):
        with self.cond:
            if idx not in self.cache:
                raise KeyError(idx)
            res = self.cache[idx]
        if isinstance(res, Exception):
            raise res
        return res


def build_model(terrain=None, xy_offset=None):
    """Build the G1 MuJoCo model. A checkerboard ground plane is always
    included; if `terrain` (verts, faces) is given it is baked in as a mesh,
    shifted by xy_offset so it stays aligned with the recentered clip."""
    rt._ensure_gmr_on_path()
    from general_motion_retargeting.params import ROBOT_XML_DICT  # type: ignore
    g1_xml = pathlib.Path(str(ROBOT_XML_DICT["unitree_g1"]))
    if xy_offset is None:
        xy_offset = np.zeros(3)
    terrain_asset, terrain_geom = rt._build_terrain_xml(terrain, xy_offset)
    wrapper = g1_xml.parent / "_browse_clean_wrapper.xml"
    wrapper.write_text(
        rt._WRAPPER_XML_TMPL.format(g1_xml_name=g1_xml.name,
                                    terrain_asset=terrain_asset,
                                    terrain_geom=terrain_geom),
        encoding="utf-8")
    import mujoco as mj  # type: ignore
    try:
        return mj.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink(missing_ok=True)


def make_seg(res):
    """Build a per-clip view dict (recentered to pelvis origin) from a
    retarget_segment result. Shared by the MuJoCo and viser viewers."""
    q = res["qpos"]
    xy = np.array([-q[0, 0], -q[0, 1], 0.0], dtype=np.float64)
    qw = q[:, 3:7]
    qxyzw = np.stack([qw[:, 1], qw[:, 2], qw[:, 3], qw[:, 0]], axis=1)
    rot = R.from_quat(qxyzw).as_matrix()
    n = q.shape[0]
    lk = res["look_fwd_xy"]
    have_look = lk is not None and lk.shape[0] >= n
    dv = res["des_vel"]
    have_dv = dv is not None and dv.shape[0] >= n
    mi = res["move_input"]
    have_mi = mi is not None and mi.shape[0] >= n and have_look

    def _rc(arr):
        """Recenter an (..., 3) world-frame array to the pelvis origin (XY only)."""
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64).copy()
        a[..., 0] += xy[0]
        a[..., 1] += xy[1]
        return a

    return dict(qpos=q, fps=res["fps"], n=n, xy_offset=xy,
                viz_pos=q[:, :3] + xy, rot=rot,
                pelvis_fwd=rot[:, :2, 0].astype(np.float32),
                look=lk if have_look else None,
                des_vel=dv if have_dv else None,
                move_input=mi if have_mi else None,
                have_look=have_look, have_desvel=have_dv,
                have_moveinput=have_mi, terrain=res.get("terrain"),
                scaled_skel=_rc(res.get("scaled_skel")),
                scaled_names=res.get("scaled_names"),
                scaled_toe=_rc(res.get("scaled_toe")),
                kabsch_skel=_rc(res.get("kabsch_skel")),
                kabsch_names=res.get("kabsch_names"))


def run_viewer(playlist, start=0, traj_window=100, marker_step=10,
               terrain_dir=None, terrain_ground_z0=0.0):
    """Play the playlist in a MuJoCo G1 viewer with N/P to jump trajectories.

    terrain_dir=None  -> one persistent viewer, ground plane only (instant N).
    terrain_dir set   -> per-segment viewer with that clip's terrain baked in
                         (the window relaunches on N/P so the terrain matches).
    A checkerboard ground plane is always present."""
    import mujoco as mj  # type: ignore
    import mujoco.viewer  # type: ignore

    pf = Prefetcher(playlist, terrain_dir=terrain_dir,
                    terrain_ground_z0=terrain_ground_z0)

    def render_frame(viewer, data, model, seg, state):
        qpos = seg["qpos"][state["fi"]].copy()
        qpos[:3] += seg["xy_offset"]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        fi = state["fi"]
        if seg["have_desvel"]:
            dv = seg["des_vel"][fi].astype(np.float64)
            dvw = (seg["rot"][fi, :2, 0] * dv[0] + seg["rot"][fi, :2, 1] * dv[1])
        else:
            dvw = np.zeros(2)
        if seg["have_moveinput"]:
            mi = seg["move_input"][fi].astype(np.float64)
            lf = seg["look"][fi].astype(np.float64)
            lr = np.array([lf[1], -lf[0]])
            miw = lf * mi[0] + lr * mi[1]
        else:
            miw = np.zeros(2)
        scn = viewer.user_scn
        scn.ngeom = 0
        if state["show_traj"]:
            rt._add_traj_overlay(viewer, fi, seg["viz_pos"], seg["rot"],
                                 window=traj_window, marker_step=marker_step)
        if state["show_orient"]:
            rt._append_orientation_arrows(
                scn, base_xy=seg["viz_pos"][fi, :2],
                pelvis_dir_xy=seg["pelvis_fwd"][fi],
                look_dir_xy=seg["look"][fi] if seg["have_look"] else None)
            if state["show_vel"] and seg["have_desvel"]:
                rt._append_desvel_arrow(scn, seg["viz_pos"][fi, :2], dvw)
            if state["show_vel"] and seg["have_moveinput"]:
                rt._append_moveinput_arrow(scn, seg["viz_pos"][fi, :2], miw)
        viewer.sync()

    def make_key_cb(state, on_next, on_prev):
        def cb(k):
            if k == 32:            # Space
                state["paused"] = not state["paused"]
            elif k == 262:         # Right
                state["step"] = 1
            elif k == 263:         # Left
                state["step"] = -1
            elif k == 259:         # Backspace
                state["fi"] = 0
            elif k in (78, 110):   # N / n
                on_next()
            elif k in (80, 112):   # P / p
                on_prev()
            elif k == 84:          # T
                state["show_traj"] = not state["show_traj"]
            elif k == 79:          # O
                state["show_orient"] = not state["show_orient"]
            elif k == 86:          # V
                state["show_vel"] = not state["show_vel"]
            elif k in (256, 113):  # Esc / Q
                state["quit"] = True
        return cb

    def announce(idx, seg):
        it = playlist[idx]
        if seg.get("terrain") is not None:
            terr = "  +terrain"
        elif terrain_dir:
            terr = "  (no terrain found -> ground only)"
        else:
            terr = ""
        print(f"\n[#{idx+1}/{len(playlist)}] {it['name']} ({it['cat']})  "
              f"n={seg['n']}  dur={seg['n']/seg['fps']:.1f}s  "
              f"fps={seg['fps']:.1f}{terr}")

    def step_logic(state, seg, last_t):
        if state["step"] != 0:
            state["fi"] = (state["fi"] + state["step"]) % seg["n"]
            state["step"] = 0
        elif not state["paused"]:
            now = time.time()
            dt = 1.0 / max(1.0, float(seg["fps"]))
            if now - last_t >= dt:
                state["fi"] = (state["fi"] + 1) % seg["n"]
                last_t = now
            else:
                wait = min(dt - (now - last_t), 0.001)
                if wait > 0:
                    time.sleep(wait)
                return last_t, False
        return last_t, True

    CTRL = ("  Controls: Space=pause  Left/Right=step  Backspace=reset  "
            "N=next  P=prev  T=traj  O=orient  V=vel  Esc=quit")

    # ── Mode A: persistent single viewer (ground plane only) ──
    if terrain_dir is None:
        model = build_model(None)
        data = mj.MjData(model)
        print(f"\n[retarget #{start}] {playlist[start]['name']} ...")
        first = retarget_segment(playlist[start]["jsonl"], playlist[start]["meta"],
                                 playlist[start]["scale"])
        with pf.lock:
            pf.cache[start] = first
            pf.cond.notify_all()
        pf.request(start + 1)
        cur = {"seg": make_seg(first)}
        state = {"idx": start, "paused": True, "fi": 0, "step": 0,
                 "show_traj": True, "show_orient": True, "show_vel": False,
                 "pending": None, "quit": False}

        def switch_to(idx):
            if idx < 0 or idx >= len(playlist):
                print(f"  [boundary] {'start' if idx < 0 else 'end'} of playlist.")
                return
            if pf.has(idx):
                try:
                    cur["seg"] = make_seg(pf.get(idx))
                    state["idx"] = idx
                    state["pending"] = None
                    pf.request(idx + 1)
                    announce(idx, cur["seg"])
                except Exception as e:
                    print(f"  [switch] failed #{idx}: {e}; skipping.")
                    state["pending"] = None
            else:
                pf.request(idx)
                state["pending"] = idx
                if not state.get("_la"):
                    print(f"  [loading next trajectory #{idx+1} "
                          f"{playlist[idx]['name']} ...]")
                    state["_la"] = True

        key_cb = make_key_cb(state,
                             lambda: switch_to(state["idx"] + 1),
                             lambda: switch_to(state["idx"] - 1))
        viewer = mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
            key_callback=key_cb)
        viewer.cam.lookat = np.array([0.0, 0.0, 0.85])
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 180
        announce(start, cur["seg"])
        print(CTRL)
        print("  Started PAUSED. Press Space to play, N to jump to the next clip.")
        last_t = time.time()
        while viewer.is_running() and not state["quit"]:
            if state["pending"] is not None and pf.has(state["pending"]):
                switch_to(state["pending"])
            last_t, proceed = step_logic(state, cur["seg"], last_t)
            if not proceed:
                continue
            render_frame(viewer, data, model, cur["seg"], state)
            if state["paused"] and state["step"] == 0 and state["pending"] is None:
                time.sleep(0.01)
        viewer.close()
        return

    # ── Mode B: per-segment viewer with terrain baked in ──
    print(f"\n[retarget #{start}] {playlist[start]['name']} ...")
    first = retarget_segment(playlist[start]["jsonl"], playlist[start]["meta"],
                             playlist[start]["scale"], terrain_dir=terrain_dir,
                             terrain_ground_z0=terrain_ground_z0)
    with pf.lock:
        pf.cache[start] = first
        pf.cond.notify_all()
    pf.request(start + 1)
    idx = start
    while 0 <= idx < len(playlist):
        seg = make_seg(pf.get(idx))
        model = build_model(seg.get("terrain"), seg["xy_offset"])
        data = mj.MjData(model)
        state = {"idx": idx, "paused": True, "fi": 0, "step": 0,
                 "show_traj": True, "show_orient": True, "show_vel": False,
                 "advance": None, "quit": False, "_la": False}
        start_idx = idx

        def request_advance(target):
            if target < 0 or target >= len(playlist):
                print(f"  [boundary] {'start' if target < 0 else 'end'} of playlist.")
                return
            state["advance"] = target

        key_cb = make_key_cb(state,
                             lambda: request_advance(state["idx"] + 1),
                             lambda: request_advance(state["idx"] - 1))
        viewer = mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
            key_callback=key_cb)
        viewer.cam.lookat = np.array([0.0, 0.0, 0.85])
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 180
        announce(idx, seg)
        print(CTRL)
        print("  Terrain mode: N/P relaunch the window with the next clip's "
              "terrain. Started PAUSED.")
        last_t = time.time()
        advanced = False
        while viewer.is_running() and not state["quit"]:
            if state["advance"] is not None:
                target = state["advance"]
                if pf.has(target):
                    idx = target
                    state["advance"] = None
                    viewer.close()
                    advanced = True
                    break
                else:
                    pf.request(target)
                    if not state["_la"]:
                        print(f"  [loading next trajectory #{target+1} "
                              f"{playlist[target]['name']} ...]")
                        state["_la"] = True
            last_t, proceed = step_logic(state, seg, last_t)
            if not proceed:
                continue
            render_frame(viewer, data, model, seg, state)
            if state["paused"] and state["step"] == 0 and state["advance"] is None:
                time.sleep(0.01)
        if not advanced:
            # window closed manually or quit key
            try:
                viewer.close()
            except Exception:
                pass
            return
        # else loop relaunches for the new idx


def run_viser_viewer(playlist, start=0, traj_window=100, marker_step=10,
                     terrain_dir=None, terrain_ground_z0=0.0, port=8080):
    """Browse the cleaned segments in a viser web 3D viewer.

    One persistent viser server; N/P (or the clip slider) switch trajectory by
    swapping the G1 mesh, terrain mesh and trajectory line in-place — no window
    relaunch, even with terrain. Ground plane always shown."""
    import mujoco as mj  # type: ignore
    import viser
    from verify_pipeline import (setup_g1_visual, update_g1_visual,
                                SCALED_BONE_CONNECTIONS, BONE_CONNECTIONS)

    def _bone_pairs(connections, names):
        """Resolve a list of (a, b) bone-name pairs to index pairs."""
        n2i = {nm: i for i, nm in enumerate(names)} if names is not None else {}
        pairs = [(n2i[a], n2i[b]) for a, b in connections
                 if a in n2i and b in n2i]
        return np.asarray(pairs, dtype=np.int64) if pairs else None

    # Retarget the first clip BEFORE starting the viser server. GMR's C-extension
    # init is sensitive to other threads running concurrently, so we do the
    # (single-threaded) retarget first, then bring up the web server.
    pf = Prefetcher(playlist, terrain_dir=terrain_dir,
                    terrain_ground_z0=terrain_ground_z0)
    print(f"\n[retarget #{start}] {playlist[start]['name']} ...")
    first = retarget_segment(playlist[start]["jsonl"], playlist[start]["meta"],
                            playlist[start]["scale"], terrain_dir=terrain_dir,
                            terrain_ground_z0=terrain_ground_z0)
    with pf.lock:
        pf.cache[start] = first
        pf.cond.notify_all()
    pf.request(start + 1)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    # Ground plane + grid (fixed, large).
    gv = np.array([[-12, -12, 0], [12, -12, 0], [12, 12, 0], [-12, 12, 0]],
                  dtype=np.float32)
    gf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    server.scene.add_mesh_simple("/ground", gv, gf, color=(150, 150, 150),
                                 opacity=0.45, side="double", flat_shading=True)
    server.scene.add_grid("/grid", width=24.0, height=24.0)


    cur = {"idx": start, "seg": None, "n": 0, "g1_handles": [], "g1_model": None,
           "g1_data": None, "terrain_h": None, "traj_h": None, "cur_h": None,
           "pending": None,
           "scaled_joints_h": None, "scaled_bones_h": None, "scaled_toe_h": None,
           "scaled_bp": None,
           "kabsch_joints_h": None, "kabsch_bones_h": None, "kabsch_bp": None}

    def make_seg_local(res):
        return make_seg(res)

    def _remove_skel_handles():
        for k in ("scaled_joints_h", "scaled_bones_h", "scaled_toe_h",
                  "kabsch_joints_h", "kabsch_bones_h"):
            if cur[k] is not None:
                cur[k].remove(); cur[k] = None

    def load_segment(idx):
        res = pf.get(idx)
        seg = make_seg_local(res)
        # remove previous clip's handles
        if cur["terrain_h"] is not None:
            cur["terrain_h"].remove(); cur["terrain_h"] = None
        if cur["traj_h"] is not None:
            cur["traj_h"].remove(); cur["traj_h"] = None
        if cur["cur_h"] is not None:
            cur["cur_h"].remove(); cur["cur_h"] = None
        for h, _gid in cur["g1_handles"]:
            h.remove()
        _remove_skel_handles()
        # G1 robot mesh (retargeter's model, robot-only)
        g1_model = res["g1_model"]
        g1_data = mj.MjData(g1_model)
        cur["g1_handles"] = setup_g1_visual(g1_model, server, prefix="/g1")
        cur["g1_model"] = g1_model
        cur["g1_data"] = g1_data
        # terrain
        if seg.get("terrain") is not None:
            v = seg["terrain"][0].astype(np.float32).copy()
            v[:, 0] += seg["xy_offset"][0]
            v[:, 1] += seg["xy_offset"][1]
            f = seg["terrain"][1].astype(np.int32)
            cur["terrain_h"] = server.scene.add_mesh_simple(
                "/terrain", v, f, color=(107, 168, 235), opacity=0.85,
                side="double", flat_shading=True)
        # full-clip trajectory line (static) + gold current marker
        vp = seg["viz_pos"].astype(np.float32)
        if vp.shape[0] >= 2:
            seg_pts = np.stack([vp[:-1], vp[1:]], axis=1)
            cur["traj_h"] = server.scene.add_line_segments(
                "/traj", seg_pts, colors=(0.0, 0.75, 0.75), line_width=2.0)
        cur["cur_h"] = server.scene.add_point_cloud(
            "/cur", vp[:1], colors=(255, 180, 40), point_size=0.05,
            point_shape="circle")
        # scaled "IK-input" human skeleton (red) + un-scaled Kabsch overlay (blue)
        # — same as verify_pipeline --stage s5.1.
        sk = seg.get("scaled_skel")
        if sk is not None and seg.get("scaled_names"):
            cur["scaled_bp"] = _bone_pairs(SCALED_BONE_CONNECTIONS,
                                           seg["scaled_names"])
            cur["scaled_joints_h"] = server.scene.add_point_cloud(
                "/scaled/joints", sk[0].astype(np.float32),
                colors=(220, 30, 60), point_size=0.025, point_shape="circle")
            if cur["scaled_bp"] is not None:
                bp = cur["scaled_bp"]
                cur["scaled_bones_h"] = server.scene.add_line_segments(
                    "/scaled/bones",
                    np.stack([sk[0, bp[:, 0]], sk[0, bp[:, 1]]], axis=1)
                    .astype(np.float32),
                    colors=(220, 30, 60), line_width=2.0)
            toe = seg.get("scaled_toe")
            if toe is not None and np.isfinite(toe[0]).all():
                cur["scaled_toe_h"] = server.scene.add_point_cloud(
                    "/scaled/toe", toe[0].astype(np.float32),
                    colors=(245, 150, 20), point_size=0.04, point_shape="circle")
        kb = seg.get("kabsch_skel")
        if kb is not None and seg.get("kabsch_names"):
            cur["kabsch_bp"] = _bone_pairs(BONE_CONNECTIONS, seg["kabsch_names"])
            cur["kabsch_joints_h"] = server.scene.add_point_cloud(
                "/kabsch/joints", kb[0].astype(np.float32),
                colors=(60, 120, 235), point_size=0.02, point_shape="circle")
            if cur["kabsch_bp"] is not None:
                bp = cur["kabsch_bp"]
                cur["kabsch_bones_h"] = server.scene.add_line_segments(
                    "/kabsch/bones",
                    np.stack([kb[0, bp[:, 0]], kb[0, bp[:, 1]]], axis=1)
                    .astype(np.float32),
                    colors=(120, 120, 120), line_width=1.5)
        cur["idx"] = idx
        cur["seg"] = seg
        cur["n"] = seg["n"]
        g_frame.value_max = max(0, seg["n"] - 1)
        g_frame.value = 0
        g_clip.value = idx + 1
        g_name.value = playlist[idx]["name"]
        g_dur.value = f"{seg['n']/seg['fps']:.1f}s  ({playlist[idx]['cat']})"
        if seg.get("terrain") is not None:
            g_terrain.value = "on"
        elif terrain_dir:
            g_terrain.value = "ground only (no terrain found)"
        else:
            g_terrain.value = "off (pass --terrain-dir)"

    def render(fi):
        seg = cur["seg"]
        if seg is None:
            return
        fi = max(0, min(fi, seg["n"] - 1))
        # Recenter exactly like the working reference (visualize_g1): shift the
        # pelvis by xy_offset BEFORE fk, then render geoms with no extra offset.
        # (Passing xy_offset to update_g1_visual would SUBTRACT it — wrong sign —
        # and strand the robot at 2*frame0_xy, far from the terrain/trajectory.)
        qpos = seg["qpos"][fi].copy()
        qpos[:3] += seg["xy_offset"]
        cur["g1_data"].qpos[:] = qpos
        mj.mj_forward(cur["g1_model"], cur["g1_data"])
        update_g1_visual(cur["g1_data"], cur["g1_handles"],
                         np.zeros(2), visible=g_show_g1.value)
        if cur["cur_h"] is not None:
            cur["cur_h"].points = seg["viz_pos"][fi:fi+1].astype(np.float32)
        if cur["traj_h"] is not None:
            cur["traj_h"].visible = g_show_traj.value
        # scaled IK-input skeleton (red) + toe (orange)
        if cur["scaled_joints_h"] is not None and seg.get("scaled_skel") is not None:
            s = seg["scaled_skel"][fi].astype(np.float32)
            show = g_show_scaled.value
            cur["scaled_joints_h"].points = s if show else s[:0]
            if cur["scaled_bones_h"] is not None and cur["scaled_bp"] is not None:
                bp = cur["scaled_bp"]
                if show:
                    cur["scaled_bones_h"].points = np.stack(
                        [s[bp[:, 0]], s[bp[:, 1]]], axis=1).astype(np.float32)
                else:
                    cur["scaled_bones_h"].points = np.zeros((0, 2, 3),
                                                             dtype=np.float32)
            if cur["scaled_toe_h"] is not None and seg.get("scaled_toe") is not None:
                tv = seg["scaled_toe"][fi].astype(np.float32)
                cur["scaled_toe_h"].points = tv if show else tv[:0]
        # un-scaled Kabsch overlay (blue)
        if cur["kabsch_joints_h"] is not None and seg.get("kabsch_skel") is not None:
            k = seg["kabsch_skel"][fi].astype(np.float32)
            show = g_show_kabsch.value
            cur["kabsch_joints_h"].points = k if show else k[:0]
            if cur["kabsch_bones_h"] is not None and cur["kabsch_bp"] is not None:
                bp = cur["kabsch_bp"]
                if show:
                    cur["kabsch_bones_h"].points = np.stack(
                        [k[bp[:, 0]], k[bp[:, 1]]], axis=1).astype(np.float32)
                else:
                    cur["kabsch_bones_h"].points = np.zeros((0, 2, 3),
                                                             dtype=np.float32)
        g_frame.value = fi

    def request_clip(target):
        if target < 0 or target >= len(playlist):
            return
        if pf.has(target):
            cur["pending"] = target
        else:
            pf.request(target)
            cur["pending"] = target
            g_name.value = f"loading #{target+1} {playlist[target]['name']} ..."

    # GUI controls.
    g_clip = server.gui.add_slider("clip", 1, len(playlist), 1, start + 1)
    g_frame = server.gui.add_slider("frame", 0, max(0, first["qpos"].shape[0] - 1), 1, 0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", 1, 120, 1, 30)
    g_show_traj = server.gui.add_checkbox("trajectory", True)
    g_show_g1 = server.gui.add_checkbox("show G1", True)
    g_show_scaled = server.gui.add_checkbox("show scaled (IK input)", True)
    g_show_kabsch = server.gui.add_checkbox("show Kabsch (un-scaled)", False)
    g_name = server.gui.add_text("clip", "")
    g_dur = server.gui.add_text("info", "")
    g_terrain = server.gui.add_text("terrain", "")
    server.gui.add_button("next (N)").on_click(lambda _: request_clip(cur["idx"] + 1))
    server.gui.add_button("prev (P)").on_click(lambda _: request_clip(cur["idx"] - 1))
    server.gui.add_markdown(
        "**red** = scaled human (IK input) &nbsp; | &nbsp; **orange** = scaled toe "
        "&nbsp; | &nbsp; **blue** = un-scaled Kabsch &nbsp; | &nbsp; **gold** = "
        "current pelvis &nbsp; | &nbsp; **cyan line** = trajectory &nbsp; | &nbsp; "
        "**blue mesh** = terrain &nbsp; | &nbsp; **grey** = ground")

    g_clip.on_update(lambda _: request_clip(int(g_clip.value) - 1))

    load_segment(start)
    render(0)

    _vp = getattr(server, "port", port)
    print(f"\n[viser] http://localhost:{_vp}  — {len(playlist)} clips")
    print("  Use the clip slider or next/prev buttons to jump trajectories. "
          "Close with Ctrl-C.")

    try:
        while True:
            if cur["pending"] is not None and pf.has(cur["pending"]):
                t = cur["pending"]
                cur["pending"] = None
                load_segment(t)
                pf.request(t + 1)
                render(0)
            fi = int(g_frame.value)
            if g_play.value and cur["n"] > 1:
                fi = (fi + 1) % cur["n"]
                g_frame.value = fi
            render(fi)
            time.sleep(1.0 / float(g_fps.value) if g_play.value else 0.05)
    except KeyboardInterrupt:
        print("\n[viser] shutting down.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cat", choices=CATEGORIES,
                   help="single category to browse")
    g.add_argument("--all", action="store_true",
                   help="browse all categories back-to-back")
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="root of (cleaned) MorphData_v1")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of segments in the playlist")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the playlist (seed 0)")
    ap.add_argument("--start", type=int, default=0,
                    help="0-based index of the first segment to show")
    ap.add_argument("--traj-window", type=int, default=100,
                    help="trajectory overlay window (frames). <=0 = full clip")
    ap.add_argument("--marker-step", type=int, default=10)
    ap.add_argument("--terrain-dir", default=None,
                    help="Directory holding terrain_<hash>.json files. When set, "
                         "each clip's terrain (from its meta `terrain_ref`) is "
                         "baked into the scene and the viewer relaunches per "
                         "clip so the terrain matches. Without it, only the "
                         "ground plane is shown in one persistent viewer.")
    ap.add_argument("--terrain-ground-z0", type=float, default=0.0,
                    help="UE ground height (cm) used as the terrain z0 "
                         "reference (subtracted from UE Z). Default 0.")
    ap.add_argument("--viewer", choices=["mujoco", "viser"], default="mujoco",
                    help="Viewer backend. 'mujoco' = native MuJoCo window "
                         "(default). 'viser' = web 3D viewer (needs `pip install "
                         "viser`); one persistent server, terrain+robot swap in "
                         "place on trajectory switch (no window relaunch).")
    ap.add_argument("--port", type=int, default=8080,
                    help="viser server port (default 8080).")
    args = ap.parse_args()

    cats = CATEGORIES if args.all else [args.cat]
    playlist = build_playlist(cats, data_root=pathlib.Path(args.data_root),
                             limit=args.limit, shuffle=args.shuffle)
    if not playlist:
        print(f"No segments found under {args.data_root} for {cats}.")
        sys.exit(1)
    print(f"[playlist] {len(playlist)} segments across {cats}")
    if args.terrain_dir:
        print(f"[terrain] dir={args.terrain_dir}  (terrain + ground)")
    else:
        print("[terrain] off (ground plane only). Pass --terrain-dir to add terrain.")
    start = min(args.start, len(playlist) - 1)
    if args.viewer == "viser":
        run_viser_viewer(playlist, start=start,
                         traj_window=args.traj_window,
                         marker_step=args.marker_step,
                         terrain_dir=args.terrain_dir,
                         terrain_ground_z0=args.terrain_ground_z0,
                         port=args.port)
    else:
        run_viewer(playlist, start=start,
                   traj_window=args.traj_window, marker_step=args.marker_step,
                   terrain_dir=args.terrain_dir,
                   terrain_ground_z0=args.terrain_ground_z0)


if __name__ == "__main__":
    main()
