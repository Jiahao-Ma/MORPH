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
a not-yet-ready segment is being prepared). Terrain is not loaded by default
(the full terrain set is not vendored); pass --terrain-dir to bake each
segment's terrain into the scene (rebuilds the model on switch).

Run from the Morph/ directory:
  python DataLib/preprocess/browse_clean.py --cat traversal_mantle
  python DataLib/preprocess/browse_clean.py --cat ground --limit 20
  python DataLib/preprocess/browse_clean.py --all --shuffle
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
                     height_from_data=True):
    """Run the S5 retarget flow (same as ue_world_skeleton_retarget.main s5)
    and return the artifacts needed for visualization. No file export, no
    terrain. Returns dict: qpos, fps, look_fwd_xy, des_vel, move_input."""
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
    try:
        _scaled, _names, probe, toe = rt.scaled_human_skeleton(
            gmr_frames, h, root_xy, src_human=SRC_HUMAN)
        if "ball_l" in ball_idx and "ball_r" in ball_idx and np.isfinite(toe).all():
            k = 50.0
            blue_l = skel_kabsch[:, ball_idx["ball_l"], 2]
            blue_r = skel_kabsch[:, ball_idx["ball_r"], 2]
            red_l, red_r = toe[:, 0, 2], toe[:, 1, 2]
            dz = rt.support_soft_ground_dz(blue_l, blue_r, red_l, red_r, k)
            if smooth_win >= 3:
                dz = rt.smooth_1d(dz, smooth_win)
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
    return dict(qpos=qpos_arr, fps=fps, look_fwd_xy=look_fwd_xy,
                des_vel=des_vel, move_input=move_input, g1_model=g1_model)


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

    def __init__(self, playlist):
        self.playlist = playlist
        self.cache = {}            # idx -> result dict | Exception
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.wanted = None        # idx the worker should retarget next
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _retarget(self, idx):
        it = self.playlist[idx]
        return retarget_segment(it["jsonl"], it["meta"], it["scale"])

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


def _build_g1_model():
    rt._ensure_gmr_on_path()
    from general_motion_retargeting.params import ROBOT_XML_DICT  # type: ignore
    g1_xml = pathlib.Path(str(ROBOT_XML_DICT["unitree_g1"]))
    wrapper = g1_xml.parent / "_browse_clean_wrapper.xml"
    wrapper.write_text(
        rt._WRAPPER_XML_TMPL.format(g1_xml_name=g1_xml.name,
                                    terrain_asset="", terrain_geom=""),
        encoding="utf-8")
    import mujoco as mj  # type: ignore
    try:
        return mj.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink(missing_ok=True)


def run_viewer(playlist, start=0, traj_window=100, marker_step=10):
    import mujoco as mj  # type: ignore
    import mujoco.viewer  # type: ignore

    model = _build_g1_model()
    data = mj.MjData(model)

    pf = Prefetcher(playlist)
    # Retarget the first segment (blocking) before opening the viewer.
    print(f"\n[retarget #{start}] {playlist[start]['name']} "
          f"({playlist[start]['cat']}) ...")
    first = retarget_segment(playlist[start]["jsonl"], playlist[start]["meta"],
                            playlist[start]["scale"])
    with pf.lock:
        pf.cache[start] = first
        pf.cond.notify_all()
    pf.request(start + 1)

    state = {
        "idx": start, "paused": True, "fi": 0, "step": 0,
        "show_traj": True, "show_orient": True, "show_vel": False,
        "pending": None,   # idx we are switching to once ready
        "loading": False,
    }

    # Per-trajectory derived arrays (recomputed on switch).
    seg = {"qpos": None, "fps": 60.0, "n": 0, "xy_offset": None,
           "viz_pos": None, "rot": None, "pelvis_fwd": None,
           "look": None, "des_vel": None, "move_input": None,
           "have_look": False, "have_desvel": False, "have_moveinput": False}

    def load_into_seg(res):
        q = res["qpos"]
        seg["qpos"] = q
        seg["fps"] = res["fps"]
        seg["n"] = q.shape[0]
        seg["xy_offset"] = np.array([-q[0, 0], -q[0, 1], 0.0], dtype=np.float64)
        seg["viz_pos"] = q[:, :3] + seg["xy_offset"]
        qw = q[:, 3:7]
        qxyzw = np.stack([qw[:, 1], qw[:, 2], qw[:, 3], qw[:, 0]], axis=1)
        seg["rot"] = R.from_quat(qxyzw).as_matrix()
        seg["pelvis_fwd"] = seg["rot"][:, :2, 0].astype(np.float32)
        lk = res["look_fwd_xy"]
        seg["have_look"] = lk is not None and lk.shape[0] >= seg["n"]
        seg["look"] = lk if seg["have_look"] else None
        dv = res["des_vel"]
        seg["have_desvel"] = dv is not None and dv.shape[0] >= seg["n"]
        seg["des_vel"] = dv if seg["have_desvel"] else None
        mi = res["move_input"]
        seg["have_moveinput"] = mi is not None and mi.shape[0] >= seg["n"] and seg["have_look"]
        seg["move_input"] = mi if seg["have_moveinput"] else None
        state["fi"] = 0

    load_into_seg(first)

    def switch_to(idx):
        if idx < 0 or idx >= len(playlist):
            print(f"  [boundary] already at {'start' if idx < 0 else 'end'} "
                  f"of playlist (#{state['idx']+1}/{len(playlist)}).")
            return False
        if pf.has(idx):
            try:
                load_into_seg(pf.get(idx))
                state["idx"] = idx
                state["pending"] = None
                state["loading"] = False
                pf.request(idx + 1)
                it = playlist[idx]
                dur = seg["n"] / max(1.0, seg["fps"])
                print(f"\n[#{idx+1}/{len(playlist)}] {it['name']} "
                      f"({it['cat']})  n={seg['n']}  dur={dur:.1f}s  "
                      f"fps={seg['fps']:.1f}")
                return True
            except Exception as e:
                print(f"  [switch] failed #{idx}: {e}; skipping.")
                state["pending"] = None
                state["loading"] = False
                return False
        else:
            pf.request(idx)
            state["pending"] = idx
            state["loading"] = True
            if not state.get("_loading_announced"):
                print(f"  [loading next trajectory #{idx+1} "
                      f"{playlist[idx]['name']} ...]")
                state["_loading_announced"] = True
            return False

    def key_cb(k):
        if k == 32:        # Space
            state["paused"] = not state["paused"]
        elif k == 262:     # Right
            state["step"] = 1
        elif k == 263:     # Left
            state["step"] = -1
        elif k == 259:     # Backspace
            state["fi"] = 0
        elif k == 78 or k == 110:   # N / n
            switch_to(state["idx"] + 1)
        elif k == 80 or k == 112:   # P / p
            switch_to(state["idx"] - 1)
        elif k == 84:      # T
            state["show_traj"] = not state["show_traj"]
        elif k == 79:      # O
            state["show_orient"] = not state["show_orient"]
        elif k == 86:      # V
            state["show_vel"] = not state["show_vel"]
        elif k in (256, 113):  # Esc / Q
            state["quit"] = True

    viewer = mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
        key_callback=key_cb)
    viewer.cam.lookat = np.array([0.0, 0.0, 0.85])
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 180

    it = playlist[start]
    print(f"\n[#{start+1}/{len(playlist)}] {it['name']} ({it['cat']})  "
          f"n={seg['n']}  dur={seg['n']/seg['fps']:.1f}s")
    print("  Controls: Space=pause  Left/Right=step  Backspace=reset  "
          "N=next  P=prev  T=traj  O=orient  V=vel  Esc=quit")
    print("  Started PAUSED. Press Space to play, N to jump to the next clip.")

    last_t = time.time()
    while viewer.is_running() and not state.get("quit"):
        # Resolve a pending switch when the target is ready.
        if state["pending"] is not None:
            if pf.has(state["pending"]):
                switch_to(state["pending"])
            # else keep rendering current while loading

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
                continue

        qpos = seg["qpos"][state["fi"]].copy()
        qpos[:3] += seg["xy_offset"]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        fi = state["fi"]
        if seg["have_desvel"]:
            dv = seg["des_vel"][fi].astype(np.float64)
            dv_world_xy = (seg["rot"][fi, :2, 0] * dv[0]
                           + seg["rot"][fi, :2, 1] * dv[1])
        else:
            dv_world_xy = np.zeros(2)
        if seg["have_moveinput"]:
            mi = seg["move_input"][fi].astype(np.float64)
            lf = seg["look"][fi].astype(np.float64)
            lr = np.array([lf[1], -lf[0]])
            mi_world_xy = lf * mi[0] + lr * mi[1]
        else:
            mi_world_xy = np.zeros(2)

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
                rt._append_desvel_arrow(scn, seg["viz_pos"][fi, :2], dv_world_xy)
            if state["show_vel"] and seg["have_moveinput"]:
                rt._append_moveinput_arrow(scn, seg["viz_pos"][fi, :2], mi_world_xy)
        viewer.sync()

        if state["paused"] and state["step"] == 0 and state["pending"] is None:
            time.sleep(0.01)

    viewer.close()


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
    args = ap.parse_args()

    cats = CATEGORIES if args.all else [args.cat]
    playlist = build_playlist(cats, data_root=pathlib.Path(args.data_root),
                             limit=args.limit, shuffle=args.shuffle)
    if not playlist:
        print(f"No segments found under {args.data_root} for {cats}.")
        sys.exit(1)
    print(f"[playlist] {len(playlist)} segments across {cats}")
    run_viewer(playlist, start=min(args.start, len(playlist) - 1),
               traj_window=args.traj_window, marker_step=args.marker_step)


if __name__ == "__main__":
    main()
