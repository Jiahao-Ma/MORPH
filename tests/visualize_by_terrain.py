#!/usr/bin/env python3
"""Randomly play Stage 2/Stage 3 motions assigned to one terrain in Viser.

The Stage 2 UE skeletons and saved Stage 3 G1 qpos are loaded from the Stage 3
segments manifest. Every clip starts at local frame zero on a shared clock and
loops independently at its source FPS. All coordinates use the same transform
as ``verify_pipeline_scaled.py``:

    Stage 2 wp * data_scale -> UE-to-MuJoCo -> global Kabsch

Stage 3 qpos is already in that final world frame, so it is never transformed
again. A common XY display offset is applied to terrain, humans, and robots.

Large terrain groups can require parsing several GB of JSONL. The first run
therefore creates an immutable, terrain-specific visualization cache. Later
runs memory-map that cache and start quickly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RETARGET_DIR = REPO / "DataLib" / "retarget"
GMR_DIR = REPO / "DataLib" / "gmr"
DEFAULT_MANIFEST = pathlib.Path(
    "/media/disk0/parc_data/"
    "MorphData_v1_g1_forward_only_min3_balanced/segments_manifest.jsonl"
)
DEFAULT_CACHE_ROOT = pathlib.Path(
    "/media/disk0/parc_data/MorphData_v1_visual_cache/by_terrain"
)
DEFAULT_CONFIG = RETARGET_DIR / "gasp_bvh_alignment_g1_height.json"
DEFAULT_IK_CONFIG = (
    GMR_DIR
    / "general_motion_retargeting"
    / "ik_configs"
    / "bvh_ue5_g1scale_to_g1.json"
)
CACHE_VERSION = 3
HUMAN_COLOR = np.asarray((45, 165, 255), dtype=np.uint8)
G1_COLOR = np.asarray((255, 155, 35), dtype=np.uint8)
MATCH_GROUPS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "pelvis": ("pelvis", (255, 65, 65)),
    "thigh": ("hip", (70, 230, 90)),
    "calf": ("knee", (255, 220, 45)),
    "foot": ("ankle", (185, 90, 255)),
    "upperarm": ("shoulder", (255, 75, 180)),
    "lowerarm": ("elbow", (40, 230, 230)),
    "hand": ("hand", (250, 250, 250)),
}
HUMAN_POINT_ALIASES = {
    "LeftFootMod": "foot_l",
    "RightFootMod": "foot_r",
}
ROBOT_POINT_ALIASES = {
    "left_hip_yaw_link": "left_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_pitch_link",
}

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(GMR_DIR) not in sys.path:
    sys.path.insert(0, str(GMR_DIR))

from DataLib.retarget import verify_pipeline_scaled as verify  # noqa: E402


HUMAN_CONNECTIONS = verify.BONE_CONNECTIONS
G1_CONNECTIONS = verify.G1_FK_BONE_CONNECTIONS


@dataclass(frozen=True)
class Motion:
    seg_id: str
    category: str
    n_frames: int
    fps: float
    data_scale: float
    stage2_jsonl: pathlib.Path
    stage2_meta: pathlib.Path
    qpos_npy: pathlib.Path

    def to_json(self) -> dict[str, Any]:
        return {
            "seg_id": self.seg_id,
            "category": self.category,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "data_scale": self.data_scale,
            "stage2_jsonl": str(self.stage2_jsonl),
            "stage2_meta": str(self.stage2_meta),
            "qpos_npy": str(self.qpos_npy),
        }


@dataclass
class Prepared:
    cache_dir: pathlib.Path | None
    metadata: dict[str, Any]
    human: np.ndarray
    g1: np.ndarray
    terrain_vertices: np.ndarray
    terrain_faces: np.ndarray


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_position_matches(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load every GMR correspondence with a non-zero position objective."""
    cfg = read_json(path)
    matched: dict[str, str] = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = cfg.get(table_name, {})
        if not isinstance(table, dict):
            raise ValueError(f"{path}: {table_name} must be an object")
        for robot_name, entry in table.items():
            if not isinstance(entry, list) or len(entry) < 3:
                raise ValueError(
                    f"{path}: invalid {table_name} entry for {robot_name}"
                )
            human_name = str(entry[0])
            position_weight = float(entry[1])
            if position_weight <= 0.0:
                continue
            previous = matched.setdefault(str(robot_name), human_name)
            if previous != human_name:
                raise ValueError(
                    f"{path}: {robot_name} maps to both {previous} and "
                    f"{human_name}"
                )

    result: list[dict[str, Any]] = []
    for target_robot_name, target_human_name in matched.items():
        human_name = HUMAN_POINT_ALIASES.get(
            target_human_name, target_human_name
        )
        robot_name = ROBOT_POINT_ALIASES.get(
            target_robot_name, target_robot_name
        )
        key = target_human_name.lower()
        group_key = next((name for name in MATCH_GROUPS if name in key), None)
        if group_key is None:
            raise ValueError(
                f"{path}: no marker color group for {target_human_name}"
            )
        label, color = MATCH_GROUPS[group_key]
        result.append(
            {
                "human": human_name,
                "target_human": target_human_name,
                "robot": robot_name,
                "target_robot": target_robot_name,
                "group": label,
                "color": list(color),
            }
        )
    if not result:
        raise ValueError(f"{path}: no non-zero position matches found")
    return result


def stat_signature(path: pathlib.Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def unique_joint_names(connections: list[tuple[str, str]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for a, b in connections:
        for name in (a, b):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def connection_indices(
    names: list[str], connections: list[tuple[str, str]]
) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(names)}
    pairs = [
        (lookup[a], lookup[b])
        for a, b in connections
        if a in lookup and b in lookup
    ]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def load_matching_motions(
    manifest_path: pathlib.Path,
    terrain_name: str,
    *,
    limit: int | None,
) -> tuple[list[Motion], int]:
    """Resolve and strictly validate manifest rows for one terrain filename."""
    candidates: list[Motion] = []
    invalid: list[str] = []
    scanned = 0
    with manifest_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            scanned += 1
            row = json.loads(line)
            meta_path = pathlib.Path(str(row.get("washed_meta", "")))
            if not meta_path.is_file():
                continue
            try:
                meta = read_json(meta_path)
            except Exception as exc:
                invalid.append(
                    f"line {line_no}: cannot read {meta_path}: {type(exc).__name__}"
                )
                continue
            if meta.get("terrain_ref") != terrain_name:
                continue

            stage2_jsonl = pathlib.Path(str(row.get("washed_jsonl", "")))
            qpos_path = pathlib.Path(str(row.get("qpos_npy", "")))
            n_frames = int(row.get("n_frames", 0))
            if not stage2_jsonl.is_file() or not qpos_path.is_file():
                invalid.append(
                    f"line {line_no}: missing Stage 2 JSONL or Stage 3 qpos"
                )
                continue
            try:
                qpos = np.load(qpos_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                invalid.append(
                    f"line {line_no}: cannot load {qpos_path}: {type(exc).__name__}"
                )
                continue
            if qpos.shape != (n_frames, 36):
                invalid.append(
                    f"line {line_no}: qpos shape {qpos.shape}, expected "
                    f"({n_frames}, 36)"
                )
                continue
            fps = float(row.get("fps_src") or meta.get("sample_rate_hz") or 60.0)
            candidates.append(
                Motion(
                    seg_id=str(row["seg_id"]),
                    category=str(row.get("category", "")),
                    n_frames=n_frames,
                    fps=fps,
                    data_scale=float(row["data_scale"]),
                    stage2_jsonl=stage2_jsonl.resolve(),
                    stage2_meta=meta_path.resolve(),
                    qpos_npy=qpos_path.resolve(),
                )
            )

    if invalid:
        preview = "\n  ".join(invalid[:10])
        suffix = "" if len(invalid) <= 10 else f"\n  ... {len(invalid) - 10} more"
        raise RuntimeError(
            f"{len(invalid)} matching manifest row(s) are invalid:\n  "
            f"{preview}{suffix}"
        )
    candidates.sort(key=lambda m: (m.category, m.seg_id))
    total = len(candidates)
    if limit is not None:
        candidates = candidates[:limit]
    print(
        f"[Manifest] scanned {scanned:,} rows; terrain={terrain_name}: "
        f"{total:,} valid motion(s)"
    )
    if limit is not None and limit < total:
        print(f"[Manifest] --limit keeps the first {len(candidates):,} motion(s)")
    return candidates, total


def cache_fingerprint(
    terrain_path: pathlib.Path,
    config_path: pathlib.Path,
    ik_config_path: pathlib.Path,
    motions: list[Motion],
    *,
    terrain_ground_z0: float,
    skip_engine: bool,
) -> str:
    h = hashlib.sha256()
    h.update(f"visualize_by_terrain_cache_v{CACHE_VERSION}\n".encode())
    h.update((stat_signature(terrain_path) + "\n").encode())
    h.update((stat_signature(config_path) + "\n").encode())
    h.update((stat_signature(ik_config_path) + "\n").encode())
    h.update(f"terrain_ground_z0={terrain_ground_z0:.12g}\n".encode())
    h.update(f"skip_engine={int(skip_engine)}\n".encode())
    for motion in motions:
        h.update((motion.seg_id + "\n").encode())
        h.update((stat_signature(motion.stage2_jsonl) + "\n").encode())
        h.update((stat_signature(motion.stage2_meta) + "\n").encode())
        h.update((stat_signature(motion.qpos_npy) + "\n").encode())
        h.update(
            f"{motion.n_frames}|{motion.fps:.9g}|{motion.data_scale:.9g}\n".encode()
        )
    return h.hexdigest()


def choose_common_scale(motions: list[Motion]) -> float:
    scales = np.asarray([m.data_scale for m in motions], dtype=np.float64)
    if float(np.ptp(scales)) > 1e-6:
        values = sorted({round(float(v), 8) for v in scales})
        raise ValueError(
            "motions assigned to one terrain have inconsistent data_scale "
            f"values: {values}"
        )
    return float(scales[0])


def load_human_clip(
    motion: Motion,
    wanted_names: list[str],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    meta = read_json(motion.stage2_meta)
    bone_names = list(meta["bone_names"])
    lookup = {name: i for i, name in enumerate(bone_names)}
    missing = [name for name in wanted_names if name not in lookup]
    if missing:
        raise ValueError(f"{motion.seg_id}: missing human joints {missing}")
    indices = [lookup[name] for name in wanted_names]

    out = np.empty((motion.n_frames, len(indices), 3), dtype=np.float32)
    count = 0
    with motion.stage2_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if count >= motion.n_frames:
                raise ValueError(
                    f"{motion.seg_id}: Stage 2 has more than "
                    f"{motion.n_frames} non-empty frames"
                )
            frame = json.loads(line)
            joints = frame["joints"]
            try:
                out[count] = [joints[i]["wp"] for i in indices]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{motion.seg_id}: invalid joints/wp at frame {count}"
                ) from exc
            count += 1
    if count != motion.n_frames:
        raise ValueError(
            f"{motion.seg_id}: Stage 2 has {count} frames, "
            f"manifest says {motion.n_frames}"
        )

    # Uniform data scale in UE world cm, then UE X/Y swap and cm -> m.
    out *= np.float32(motion.data_scale * 0.01)
    out = out[..., [1, 0, 2]]
    # Row-vector form of verify.apply_kabsch: R @ p -> p @ R.T.
    out = out @ rotation.T.astype(np.float32)
    out += translation.astype(np.float32)
    return out


def load_g1_model():
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    return mujoco, model


def g1_clip_fk(
    qpos_path: pathlib.Path,
    n_frames: int,
    mujoco_module,
    model,
    body_ids: np.ndarray,
) -> np.ndarray:
    qpos = np.load(qpos_path, mmap_mode="r", allow_pickle=False)
    if qpos.shape != (n_frames, model.nq):
        raise ValueError(
            f"{qpos_path}: qpos shape {qpos.shape}, "
            f"expected ({n_frames}, {model.nq})"
        )
    data = mujoco_module.MjData(model)
    out = np.empty((n_frames, len(body_ids), 3), dtype=np.float32)
    for frame_i in range(n_frames):
        data.qpos[:] = qpos[frame_i]
        mujoco_module.mj_forward(model, data)
        out[frame_i] = data.xpos[body_ids]
    return out


def prepare_arrays(
    terrain_path: pathlib.Path,
    config_path: pathlib.Path,
    ik_config_path: pathlib.Path,
    motions: list[Motion],
    *,
    terrain_ground_z0: float,
    skip_engine: bool,
) -> Prepared:
    rotation, translation, _ = verify.load_alignment_config(str(config_path))
    human_names = unique_joint_names(HUMAN_CONNECTIONS)
    g1_names = unique_joint_names(G1_CONNECTIONS)
    retarget_matches = load_position_matches(ik_config_path)
    for match in retarget_matches:
        if match["human"] not in human_names:
            raise ValueError(
                f"retarget human point {match['human']} is absent from the "
                "Stage2 skeleton"
            )
        if match["robot"] not in g1_names:
            g1_names.append(match["robot"])
    human_pairs = connection_indices(human_names, HUMAN_CONNECTIONS)
    g1_pairs = connection_indices(g1_names, G1_CONNECTIONS)
    data_scale = choose_common_scale(motions)

    mujoco_module, model = load_g1_model()
    g1_body_ids = np.asarray(
        [
            mujoco_module.mj_name2id(
                model, mujoco_module.mjtObj.mjOBJ_BODY, name
            )
            for name in g1_names
        ],
        dtype=np.int32,
    )
    if bool((g1_body_ids < 0).any()):
        missing = [
            name for name, body_id in zip(g1_names, g1_body_ids) if body_id < 0
        ]
        raise ValueError(f"G1 model is missing bodies: {missing}")

    offsets = np.zeros(len(motions) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([m.n_frames for m in motions], dtype=np.int64)
    total_frames = int(offsets[-1])
    human = np.empty((total_frames, len(human_names), 3), dtype=np.float32)
    g1 = np.empty((total_frames, len(g1_names), 3), dtype=np.float32)

    print(
        f"[Prepare] {len(motions):,} motion(s), {total_frames:,} total frames; "
        "streaming Stage 2 JSONL and computing Stage 3 FK"
    )
    started = time.monotonic()
    for motion_i, motion in enumerate(motions):
        begin, end = int(offsets[motion_i]), int(offsets[motion_i + 1])
        human[begin:end] = load_human_clip(
            motion, human_names, rotation, translation
        )
        g1[begin:end] = g1_clip_fk(
            motion.qpos_npy,
            motion.n_frames,
            mujoco_module,
            model,
            g1_body_ids,
        )
        done = motion_i + 1
        if done == 1 or done == len(motions) or done % 10 == 0:
            elapsed = time.monotonic() - started
            print(
                f"  [{done:>4}/{len(motions)}] {motion.seg_id} "
                f"({end:,}/{total_frames:,} frames, {elapsed:.1f}s)"
            )

    terrain_ue = verify.load_terrain_instances(
        str(terrain_path),
        float(terrain_ground_z0),
        skip_engine=skip_engine,
        traj_xy=None,
    )
    if terrain_ue is None:
        raise ValueError(f"terrain contains no renderable instances: {terrain_path}")
    terrain_vertices, terrain_faces = terrain_ue
    terrain_vertices = terrain_vertices.astype(np.float32) * np.float32(data_scale)
    terrain_vertices *= np.float32(0.01)
    terrain_vertices = terrain_vertices[..., [1, 0, 2]]
    terrain_vertices = terrain_vertices @ rotation.T.astype(np.float32)
    terrain_vertices += translation.astype(np.float32)
    terrain_faces = terrain_faces.astype(np.int32)

    # One common display translation preserves the Stage2/G1/terrain alignment.
    scene_xy = terrain_vertices[:, :2].mean(axis=0).astype(np.float32)
    pelvis_h = human_names.index("pelvis")
    pelvis_g = g1_names.index("pelvis")
    pelvis_error = np.linalg.norm(
        human[:, pelvis_h] - g1[:, pelvis_g], axis=1
    )
    metadata: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "config": str(config_path.resolve()),
        "ik_config": str(ik_config_path.resolve()),
        "data_scale": data_scale,
        "motion_count": len(motions),
        "total_frames": total_frames,
        "offsets": offsets.tolist(),
        "human_names": human_names,
        "human_pairs": human_pairs.tolist(),
        "g1_names": g1_names,
        "g1_pairs": g1_pairs.tolist(),
        "retarget_position_matches": retarget_matches,
        "scene_xy": scene_xy.tolist(),
        "motions": [m.to_json() for m in motions],
        "alignment_diagnostic_m": {
            "human_to_g1_pelvis_median": float(np.median(pelvis_error)),
            "human_to_g1_pelvis_p95": float(np.percentile(pelvis_error, 95)),
            "human_to_g1_pelvis_max": float(pelvis_error.max()),
        },
        "terrain": {
            "path": str(terrain_path.resolve()),
            "vertices": int(terrain_vertices.shape[0]),
            "triangles": int(terrain_faces.shape[0]),
            "z_min": float(terrain_vertices[:, 2].min()),
            "z_max": float(terrain_vertices[:, 2].max()),
        },
    }
    return Prepared(
        cache_dir=None,
        metadata=metadata,
        human=human,
        g1=g1,
        terrain_vertices=terrain_vertices,
        terrain_faces=terrain_faces,
    )


def write_cache(prepared: Prepared, cache_dir: pathlib.Path) -> Prepared:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=cache_dir.parent)
    )
    try:
        np.save(temp_dir / "human.npy", prepared.human, allow_pickle=False)
        np.save(temp_dir / "g1.npy", prepared.g1, allow_pickle=False)
        np.save(
            temp_dir / "terrain_vertices.npy",
            prepared.terrain_vertices,
            allow_pickle=False,
        )
        np.save(
            temp_dir / "terrain_faces.npy",
            prepared.terrain_faces,
            allow_pickle=False,
        )
        with (temp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(prepared.metadata, f, indent=2, ensure_ascii=False)
            f.write("\n")
        try:
            os.replace(temp_dir, cache_dir)
        except OSError:
            # Another process may have completed this exact immutable cache.
            if not (cache_dir / "metadata.json").is_file():
                raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    print(f"[Cache] wrote {cache_dir}")
    return load_cache(cache_dir)


def load_cache(cache_dir: pathlib.Path) -> Prepared:
    required = [
        "metadata.json",
        "human.npy",
        "g1.npy",
        "terrain_vertices.npy",
        "terrain_faces.npy",
    ]
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete cache {cache_dir}: missing {missing}")
    metadata = read_json(cache_dir / "metadata.json")
    if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(f"unsupported cache version in {cache_dir}")
    print(f"[Cache] loading {cache_dir}")
    return Prepared(
        cache_dir=cache_dir,
        metadata=metadata,
        human=np.load(cache_dir / "human.npy", mmap_mode="r", allow_pickle=False),
        g1=np.load(cache_dir / "g1.npy", mmap_mode="r", allow_pickle=False),
        terrain_vertices=np.load(
            cache_dir / "terrain_vertices.npy", mmap_mode="r", allow_pickle=False
        ),
        terrain_faces=np.load(
            cache_dir / "terrain_faces.npy", mmap_mode="r", allow_pickle=False
        ),
    )


def get_prepared(
    args: argparse.Namespace,
    terrain_path: pathlib.Path,
    motions: list[Motion],
) -> Prepared:
    fingerprint = cache_fingerprint(
        terrain_path,
        args.config,
        args.ik_config,
        motions,
        terrain_ground_z0=args.terrain_ground_z0,
        skip_engine=not args.no_skip_engine,
    )
    suffix = "-limited" if args.limit is not None else ""
    cache_dir = (
        args.cache_root
        / f"{terrain_path.stem}{suffix}"
        / fingerprint[:20]
    )
    if not args.no_cache and (cache_dir / "metadata.json").is_file():
        return load_cache(cache_dir)
    if not args.no_cache and cache_dir.exists():
        raise RuntimeError(
            f"incomplete visualization cache exists at {cache_dir}; "
            "move it aside and rerun"
        )

    prepared = prepare_arrays(
        terrain_path,
        args.config,
        args.ik_config,
        motions,
        terrain_ground_z0=args.terrain_ground_z0,
        skip_engine=not args.no_skip_engine,
    )
    prepared.metadata["fingerprint"] = fingerprint
    if args.no_cache:
        print("[Cache] disabled; prepared arrays remain in memory only")
        return prepared
    return write_cache(prepared, cache_dir)


def layer_colors(count: int, color: np.ndarray) -> np.ndarray:
    """Return a stable per-layer color instead of coloring both layers alike."""
    if count <= 0 or color.shape != (3,):
        raise ValueError("layer color count must be positive and RGB-shaped")
    return np.broadcast_to(color, (count, 3)).copy()


def random_motion_indices(
    motion_count: int,
    visible_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a viewer subset without replacement, preserving random order."""
    if motion_count <= 0:
        raise ValueError("motion_count must be positive")
    if visible_count <= 0 or visible_count > motion_count:
        raise ValueError("visible_count must lie in [1, motion_count]")
    return np.asarray(
        rng.choice(motion_count, size=visible_count, replace=False),
        dtype=np.int64,
    )


def scene_frame(
    prepared: Prepared,
    clock_s: float,
    motion_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = prepared.metadata
    motions = metadata["motions"]
    offsets = np.asarray(metadata["offsets"], dtype=np.int64)
    lengths = np.diff(offsets)
    fps = np.asarray([float(row["fps"]) for row in motions], dtype=np.float64)
    if motion_indices is None:
        motion_indices = np.arange(len(motions), dtype=np.int64)
    else:
        motion_indices = np.asarray(motion_indices, dtype=np.int64)
        if motion_indices.ndim != 1 or motion_indices.size == 0:
            raise ValueError("motion_indices must be a non-empty vector")
        if bool((motion_indices < 0).any()) or bool(
            (motion_indices >= len(motions)).any()
        ):
            raise ValueError("motion_indices contains an out-of-range index")
    offsets = offsets[motion_indices]
    lengths = lengths[motion_indices]
    fps = fps[motion_indices]
    local = np.floor(clock_s * fps).astype(np.int64) % lengths
    indices = offsets + local
    return prepared.human[indices], prepared.g1[indices], local


def line_geometry(
    poses: np.ndarray,
    pairs: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    segments = poses[:, pairs, :].reshape(-1, 2, 3).astype(np.float32)
    segment_colors = np.broadcast_to(
        colors[:, None, None, :],
        (poses.shape[0], pairs.shape[0], 2, 3),
    ).reshape(-1, 2, 3)
    return segments, segment_colors


def recenter(points: np.ndarray, scene_xy: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    out[..., 0] -= scene_xy[0]
    out[..., 1] -= scene_xy[1]
    return out


def run_viewer(prepared: Prepared, args: argparse.Namespace) -> None:
    try:
        import mujoco
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "visualization dependencies are missing; run in the morph conda "
            "environment with `viser` and `mujoco` installed"
        ) from exc

    metadata = prepared.metadata
    count = int(metadata["motion_count"])
    offsets = np.asarray(metadata["offsets"], dtype=np.int64)
    lengths = np.diff(offsets)
    source_fps = np.asarray(
        [float(row["fps"]) for row in metadata["motions"]], dtype=np.float64
    )
    durations = lengths / source_fps
    max_duration = max(float(durations.max()), 1e-3)
    scene_xy = np.asarray(metadata["scene_xy"], dtype=np.float32)
    human_pairs = np.asarray(metadata["human_pairs"], dtype=np.int32)
    g1_pairs = np.asarray(metadata["g1_pairs"], dtype=np.int32)
    human_name_to_index = {
        name: index for index, name in enumerate(metadata["human_names"])
    }
    g1_name_to_index = {
        name: index for index, name in enumerate(metadata["g1_names"])
    }
    position_matches = metadata["retarget_position_matches"]
    human_match_indices = np.asarray(
        [human_name_to_index[row["human"]] for row in position_matches],
        dtype=np.int64,
    )
    g1_match_indices = np.asarray(
        [g1_name_to_index[row["robot"]] for row in position_matches],
        dtype=np.int64,
    )
    match_colors = np.asarray(
        [row["color"] for row in position_matches], dtype=np.uint8
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    terrain_vertices = recenter(prepared.terrain_vertices, scene_xy)
    terrain_node = server.scene.add_mesh_simple(
        "/terrain",
        terrain_vertices,
        np.asarray(prepared.terrain_faces, dtype=np.int32),
        color=(107, 168, 235),
        opacity=0.82,
        side="double",
        flat_shading=True,
    )
    xmin, ymin = terrain_vertices[:, :2].min(axis=0)
    xmax, ymax = terrain_vertices[:, :2].max(axis=0)
    server.scene.add_grid(
        "/grid",
        width=float(xmax - xmin + 4.0),
        height=float(ymax - ymin + 4.0),
        position=(float((xmin + xmax) / 2), float((ymin + ymax) / 2), 0.0),
        cell_size=0.5,
        section_size=2.0,
    )

    selector_max = min(count, int(args.max_visible))
    initial_visible = min(int(args.display_count), selector_max)
    rng = np.random.default_rng(int(args.selection_seed))
    selected_indices = random_motion_indices(count, initial_visible, rng)

    skeleton_nodes: dict[str, Any] = {}

    def install_skeleton_nodes(
        human_pose: np.ndarray, g1_pose: np.ndarray
    ) -> None:
        """Recreate colored nodes when the visible subset size changes."""
        for handle in skeleton_nodes.values():
            handle.remove()
        human_palette = layer_colors(len(human_pose), HUMAN_COLOR)
        g1_palette = layer_colors(len(g1_pose), G1_COLOR)
        human_seg, human_seg_colors = line_geometry(
            human_pose, human_pairs, human_palette
        )
        g1_seg, g1_seg_colors = line_geometry(g1_pose, g1_pairs, g1_palette)
        skeleton_nodes["human_bones"] = server.scene.add_line_segments(
            "/stage2/bones",
            human_seg,
            human_seg_colors,
            line_width=2.0,
        )
        skeleton_nodes["human_joints"] = server.scene.add_point_cloud(
            "/stage2/joints",
            human_pose.reshape(-1, 3),
            np.broadcast_to(
                human_palette[:, None, :], human_pose.shape
            ).reshape(-1, 3),
            point_size=0.018,
            point_shape="circle",
            precision="float32",
        )
        skeleton_nodes["g1_bones"] = server.scene.add_line_segments(
            "/stage3/g1_fk_bones",
            g1_seg,
            g1_seg_colors,
            line_width=2.5,
        )
        skeleton_nodes["g1_joints"] = server.scene.add_point_cloud(
            "/stage3/g1_fk_joints",
            g1_pose.reshape(-1, 3),
            np.broadcast_to(g1_palette[:, None, :], g1_pose.shape).reshape(-1, 3),
            point_size=0.022,
            point_shape="circle",
            precision="float32",
        )
        repeated_match_colors = np.broadcast_to(
            match_colors[None, :, :],
            (len(human_pose), len(position_matches), 3),
        ).reshape(-1, 3)
        human_match_points = human_pose[:, human_match_indices, :].reshape(
            -1, 3
        )
        g1_match_points = g1_pose[:, g1_match_indices, :].reshape(-1, 3)
        skeleton_nodes["human_match_outlines"] = server.scene.add_point_cloud(
            "/retarget_matches/human_outline",
            human_match_points,
            repeated_match_colors,
            point_size=0.018,
            point_shape="circle",
            precision="float32",
        )
        skeleton_nodes["human_match_centers"] = server.scene.add_point_cloud(
            "/retarget_matches/human_center",
            human_match_points,
            np.broadcast_to(HUMAN_COLOR, repeated_match_colors.shape).copy(),
            point_size=0.014,
            point_shape="circle",
            precision="float32",
        )
        skeleton_nodes["g1_match_outlines"] = server.scene.add_point_cloud(
            "/retarget_matches/g1_outline",
            g1_match_points,
            repeated_match_colors,
            point_size=0.022,
            point_shape="circle",
            precision="float32",
        )
        skeleton_nodes["g1_match_centers"] = server.scene.add_point_cloud(
            "/retarget_matches/g1_center",
            g1_match_points,
            np.broadcast_to(G1_COLOR, repeated_match_colors.shape).copy(),
            point_size=0.018,
            point_shape="circle",
            precision="float32",
        )

    human0, g10, _ = scene_frame(prepared, 0.0, selected_indices)
    install_skeleton_nodes(
        recenter(human0, scene_xy), recenter(g10, scene_xy)
    )

    # Allocate reusable mesh display slots. Random selection swaps qpos files
    # into these slots instead of creating hundreds of permanent mesh nodes.
    if args.mesh_limit == 0:
        mesh_capacity = 0
    elif args.mesh_limit < 0:
        mesh_capacity = selector_max
    else:
        mesh_capacity = min(selector_max, int(args.mesh_limit))
    mesh_slots: list[dict[str, Any]] = []
    if mesh_capacity:
        _, model = load_g1_model()
        print(
            f"[G1 mesh] creating {mesh_capacity:,} reusable display slot(s), "
            f"{model.ngeom} MuJoCo geoms each"
        )
        for slot_index in range(mesh_capacity):
            handles = verify.setup_g1_visual(
                model,
                server,
                prefix=f"/stage3/g1_mesh/slot_{slot_index:03d}",
            )
            mesh_slots.append(
                {
                    "model": model,
                    "data": mujoco.MjData(model),
                    "handles": handles,
                    "motion_index": None,
                    "qpos": None,
                }
            )

    def bind_mesh_slots(selection: np.ndarray) -> None:
        for slot_index, slot in enumerate(mesh_slots):
            if slot_index >= len(selection):
                slot["motion_index"] = None
                slot["qpos"] = None
            for handle, _geom_id in slot["handles"]:
                handle.visible = False
                continue
            motion_index = int(selection[slot_index])
            if slot["motion_index"] != motion_index:
                row = metadata["motions"][motion_index]
                slot["motion_index"] = motion_index
                slot["qpos"] = np.load(
                    row["qpos_npy"], mmap_mode="r", allow_pickle=False
                )

    bind_mesh_slots(selected_indices)

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("play", True)
        clock_slider = server.gui.add_slider(
            "shared time (s)",
            min=0.0,
            max=max_duration,
            step=min(1.0 / 120.0, max_duration),
            initial_value=0.0,
        )
        speed = server.gui.add_slider(
            "speed", min=0.05, max=4.0, step=0.05, initial_value=args.speed
        )
        reset = server.gui.add_button("reset")
    with server.gui.add_folder("Motion selection"):
        visible_count = server.gui.add_slider(
            "visible motion count",
            min=1,
            max=selector_max,
            step=1,
            initial_value=initial_visible,
        )
        randomize = server.gui.add_button("randomize motions")
        selection_info = server.gui.add_markdown("")
    with server.gui.add_folder("Layers"):
        show_human = server.gui.add_checkbox("Stage2 human skeletons", True)
        show_g1 = server.gui.add_checkbox("Stage3 G1 FK skeletons", True)
        show_joints = server.gui.add_checkbox("joint points", True)
        show_matches = server.gui.add_checkbox(
            "retarget position markers", True
        )
        show_mesh = server.gui.add_checkbox(
            f"full G1 meshes ({mesh_capacity} slots)", bool(mesh_capacity)
        )
        show_terrain = server.gui.add_checkbox("terrain", True)
    info = server.gui.add_markdown(
        f"**Terrain:** `{pathlib.Path(metadata['terrain']['path']).name}`  \n"
        f"**Motion pool:** {count:,} · **frames:** "
        f"{int(metadata['total_frames']):,}  \n"
        "All selected clips share time zero and loop independently.  \n"
        "**Blue:** Stage2 human · **Orange:** Stage3 G1"
    )
    del info
    match_legend = server.gui.add_markdown(
        "**Retarget position markers**  \n"
        "The same color marks the corresponding Human ↔ G1 points.  \n"
        "🔴 pelvis · 🟢 hip · 🟡 knee · 🟣 ankle  \n"
        "🩷 shoulder · 🩵 elbow · ⚪ hand"
    )
    del match_legend

    state = {
        "clock": 0.0,
        "reset": False,
        "randomize": False,
        "selected_indices": selected_indices,
    }

    def update_selection_info(selection: np.ndarray) -> None:
        labels = [
            f"{slot + 1}. `{metadata['motions'][int(index)]['seg_id']}`"
            for slot, index in enumerate(selection)
        ]
        selection_info.content = "**Selected motions**  \n" + "  \n".join(labels)

    update_selection_info(selected_indices)

    @reset.on_click
    def _(_event) -> None:
        state["reset"] = True

    @randomize.on_click
    def _(_event) -> None:
        state["randomize"] = True

    @visible_count.on_update
    def _(_event) -> None:
        state["randomize"] = True

    last_tick = time.monotonic()
    print(
        f"[Viewer] http://localhost:{args.port} — showing {initial_visible:,} "
        f"random motion(s) from a pool of {count:,}; Ctrl-C to stop"
    )
    try:
        while True:
            frame_started = time.monotonic()
            dt = frame_started - last_tick
            last_tick = frame_started
            wanted = int(visible_count.value)
            selection = np.asarray(state["selected_indices"], dtype=np.int64)
            if state["randomize"] or len(selection) != wanted:
                selection = random_motion_indices(count, wanted, rng)
                state["selected_indices"] = selection
                state["randomize"] = False
                state["clock"] = 0.0
                human0, g10, _ = scene_frame(prepared, 0.0, selection)
                install_skeleton_nodes(
                    recenter(human0, scene_xy), recenter(g10, scene_xy)
                )
                bind_mesh_slots(selection)
                update_selection_info(selection)
                print(
                    "[Viewer] selected "
                    + ", ".join(
                        metadata["motions"][int(index)]["seg_id"]
                        for index in selection
                    )
                )

            if state["reset"]:
                state["clock"] = 0.0
                state["reset"] = False
            elif play.value:
                state["clock"] = (
                    state["clock"] + dt * float(speed.value)
                ) % max_duration
            else:
                state["clock"] = float(clock_slider.value)
            clock_slider.value = float(state["clock"])

            human_pose, g1_pose, local_frames = scene_frame(
                prepared, state["clock"], selection
            )
            human_pose = recenter(human_pose, scene_xy)
            g1_pose = recenter(g1_pose, scene_xy)
            human_seg, _ = line_geometry(
                human_pose,
                human_pairs,
                layer_colors(len(selection), HUMAN_COLOR),
            )
            g1_seg, _ = line_geometry(
                g1_pose,
                g1_pairs,
                layer_colors(len(selection), G1_COLOR),
            )

            with server.atomic():
                human_bones = skeleton_nodes["human_bones"]
                human_joints = skeleton_nodes["human_joints"]
                g1_bones = skeleton_nodes["g1_bones"]
                g1_joints = skeleton_nodes["g1_joints"]
                human_match_handles = (
                    skeleton_nodes["human_match_outlines"],
                    skeleton_nodes["human_match_centers"],
                )
                g1_match_handles = (
                    skeleton_nodes["g1_match_outlines"],
                    skeleton_nodes["g1_match_centers"],
                )
                human_bones.points = human_seg
                human_joints.points = human_pose.reshape(-1, 3)
                g1_bones.points = g1_seg
                g1_joints.points = g1_pose.reshape(-1, 3)
                human_match_points = human_pose[
                    :, human_match_indices, :
                ].reshape(-1, 3)
                g1_match_points = g1_pose[:, g1_match_indices, :].reshape(
                    -1, 3
                )
                for handle in human_match_handles:
                    handle.points = human_match_points
                for handle in g1_match_handles:
                    handle.points = g1_match_points
                human_bones.visible = bool(show_human.value)
                human_joints.visible = bool(
                    show_human.value and show_joints.value
                )
                g1_bones.visible = bool(show_g1.value)
                g1_joints.visible = bool(show_g1.value and show_joints.value)
                show_human_matches = bool(
                    show_human.value and show_matches.value
                )
                show_g1_matches = bool(show_g1.value and show_matches.value)
                for handle in human_match_handles:
                    handle.visible = show_human_matches
                for handle in g1_match_handles:
                    handle.visible = show_g1_matches
                show_full_mesh = bool(show_g1.value and show_mesh.value)
                for slot_index, slot in enumerate(mesh_slots):
                    if slot["qpos"] is None:
                        for handle, _geom_id in slot["handles"]:
                            handle.visible = False
                        continue
                    slot["data"].qpos[:] = slot["qpos"][
                        int(local_frames[slot_index])
                    ]
                    mujoco.mj_forward(slot["model"], slot["data"])
                    verify.update_g1_visual(
                        slot["data"],
                        slot["handles"],
                        scene_xy,
                        visible=show_full_mesh,
                    )
                terrain_node.visible = bool(show_terrain.value)

            elapsed = time.monotonic() - frame_started
            time.sleep(max(0.0, 1.0 / args.update_fps - elapsed))
    except KeyboardInterrupt:
        print("\n[Viewer] stopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--terrain",
        type=pathlib.Path,
        required=True,
        help="Exact terrain_*.json path. Matching uses its filename.",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help=f"Stage 3 segments manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help="Global Kabsch alignment config used by Stage 3.",
    )
    parser.add_argument(
        "--ik-config",
        type=pathlib.Path,
        default=DEFAULT_IK_CONFIG,
        help=(
            "GMR IK config used to derive the highlighted position "
            "correspondences."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=DEFAULT_CACHE_ROOT,
        help="New visualization-cache stage; source datasets are never modified.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the motion pool after sorting by category and seg_id. The web "
            "viewer randomizes within this pool."
        ),
    )
    parser.add_argument(
        "--display-count",
        type=int,
        default=4,
        help="Initial number of randomly selected motions shown (default: 4).",
    )
    parser.add_argument(
        "--max-visible",
        type=int,
        default=16,
        help=(
            "Maximum selectable simultaneous motions in the web UI "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="Seed for reproducible motion selections (default: 0).",
    )
    parser.add_argument(
        "--mesh-limit",
        type=int,
        default=8,
        help=(
            "Number of full G1 mesh display slots. 0 disables meshes; -1 "
            "requests meshes for every visible motion (default: 8)."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build/validate the cache without opening the web viewer.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build in memory and do not read/write the visualization cache.",
    )
    parser.add_argument("--terrain-ground-z0", type=float, default=0.0)
    parser.add_argument(
        "--no-skip-engine",
        action="store_true",
        help="Also render /Engine/ helper meshes from the terrain JSON.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--update-fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.display_count <= 0:
        parser.error("--display-count must be greater than zero")
    if args.max_visible <= 0:
        parser.error("--max-visible must be greater than zero")
    if args.mesh_limit < -1:
        parser.error("--mesh-limit must be -1, 0, or a positive integer")
    if args.update_fps <= 0:
        parser.error("--update-fps must be greater than zero")
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    for label in ("terrain", "manifest", "config", "ik_config"):
        path = getattr(args, label)
        if not path.is_file():
            parser.error(f"--{label.replace('_', '-')} does not exist: {path}")
        setattr(args, label, path.resolve())
    args.cache_root = args.cache_root.resolve()

    motions, full_count = load_matching_motions(
        args.manifest, args.terrain.name, limit=args.limit
    )
    if not motions:
        parser.error(
            f"terrain {args.terrain.name} has no valid Stage2+Stage3 motion. "
            "Choose a non-zero terrain from .codex/terrain_motion_counts.md."
        )
    print(
        f"[Input] selected {len(motions):,}/{full_count:,} motion(s), "
        f"{sum(m.n_frames for m in motions):,} frames"
    )
    prepared = get_prepared(args, args.terrain, motions)
    diag = prepared.metadata["alignment_diagnostic_m"]
    print(
        "[Alignment] human pelvis -> G1 pelvis distance: "
        f"median={diag['human_to_g1_pelvis_median']:.3f} m, "
        f"p95={diag['human_to_g1_pelvis_p95']:.3f} m, "
        f"max={diag['human_to_g1_pelvis_max']:.3f} m"
    )
    if args.prepare_only:
        print("[Done] cache preparation and validation completed")
        return 0
    run_viewer(prepared, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
