"""Convert UE terrain JSON exports to MuJoCo MJCF and IsaacLab-ready formats.

Reads all terrain_<hash>.json files from the terrain directory, applies the
same UE->MuJoCo coordinate transform used by the retarget pipeline
(X<->Y swap + cm->m + Kabsch R_global/t_global), and writes:

  - terrain_mujoco/<hash>.xml          standalone MJCF with inline mesh
  - terrain_mujoco/<hash>_mesh.obj     OBJ for external loading
  - terrain_isaaclab/<hash>.obj        OBJ (same mesh, Z-up right-hand m)
  - terrain_isaaclab/<hash>.npz        {vertices, faces, height_field, ...}

Usage:
  cd <repo>/Morph
  python DataLib/terrain/convert_terrain.py \
      --terrain-dir data/sample/stairs \
      --config DataLib/retarget/gasp_bvh_alignment_g1_height.json \
      --output-dir data/RetargetOutputs \
      --name stairs1

By default a flat GROUND plane is appended to every exported terrain, because
the UE terrain export only contains the stair / box geometry and no floor
(--no-ground disables; --ground-margin / --ground-z tune it).
"""
import argparse
import json
import pathlib
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Force UTF-8 stdout/stderr so non-ASCII prints don't crash on Windows GBK consoles.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_M_UE_TO_BVH = np.array([[0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)


def load_alignment(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    g = cfg["global"]
    qwxyz = np.asarray(g["rotation_quat_wxyz"], dtype=np.float64)
    qxyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
    R_global = R.from_quat(qxyzw).as_matrix()
    t_global = np.asarray(g["translation_xyz"], dtype=np.float64)
    return R_global, t_global


def load_terrain_json(path: pathlib.Path, R_global, t_global,
                      ground_z0_cm=0.0, skip_engine=True):
    data = json.loads(path.read_text(encoding="utf-8"))
    meshes = data.get("meshes", [])
    instances = data.get("instances", [])
    if not meshes or not instances:
        return None

    all_v, all_f = [], []
    voff = 0
    for inst in instances:
        mi = int(inst.get("mesh", -1))
        if mi < 0 or mi >= len(meshes):
            continue
        mesh = meshes[mi]
        if skip_engine and str(mesh.get("asset_path", "")).startswith("/Engine/"):
            continue
        V = np.asarray(mesh["vertices"], dtype=np.float64).reshape(-1, 3)
        idx = np.asarray(mesh["indices"], dtype=np.int64)
        if V.size == 0 or idx.size < 3:
            continue
        F = idx.reshape(-1, 3)
        Nloc = np.asarray(mesh.get("normals", []), dtype=np.float64)
        Nloc = Nloc.reshape(-1, 3) if Nloc.size == V.size else None

        loc = np.asarray(inst["location"], dtype=np.float64)
        quat_xyzw = np.asarray(inst["rotation_quat_xyzw"], dtype=np.float64)
        scl = np.asarray(inst["scale"], dtype=np.float64)
        Rinst = R.from_quat(quat_xyzw).as_matrix()

        Vw = (Rinst @ (V * scl).T).T + loc
        Vw[:, 2] -= ground_z0_cm
        Vmj = Vw[:, [1, 0, 2]] * 0.01
        Vf = (R_global @ Vmj.T).T + t_global

        lin = R_global @ (0.01 * _M_UE_TO_BVH) @ Rinst @ np.diag(scl)
        tri = Vf[F]
        gnorm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if Nloc is not None:
            Nf = (lin @ Nloc.T).T
            ref = Nf[F].sum(axis=1)
            flip = np.einsum("ij,ij->i", gnorm, ref) < 0.0
        else:
            flip = np.full(F.shape[0], np.linalg.det(lin) < 0.0)
        if flip.any():
            F = F.copy()
            F[flip] = F[flip][:, ::-1]

        all_v.append(Vf.astype(np.float32))
        all_f.append((F + voff).astype(np.int32))
        voff += V.shape[0]

    if not all_v:
        return None
    verts = np.concatenate(all_v, axis=0)
    faces = np.concatenate(all_f, axis=0)
    return verts, faces


def write_obj(path: pathlib.Path, verts, faces):
    with open(path, "w") as f:
        f.write(f"# terrain mesh: {verts.shape[0]} verts, {faces.shape[0]} tris\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def write_mujoco_xml(path: pathlib.Path, verts, faces, obj_name: str):
    vtx_str = " ".join(f"{x:.5f}" for x in verts.reshape(-1))
    face_str = " ".join(str(int(i)) for i in faces.reshape(-1))
    xml = f"""<mujoco model="terrain">
  <compiler angle="radian"/>
  <asset>
    <mesh name="terrain" vertex="{vtx_str}" face="{face_str}"/>
    <material name="terrain_mat" rgba="0.42 0.66 0.92 1.0"
              specular="0.08" shininess="0.12" reflectance="0.0"/>
  </asset>
  <worldbody>
    <geom name="terrain_geom" type="mesh" mesh="terrain"
          material="terrain_mat" contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")


def make_ground_quad(verts, margin=5.0, ground_z=None):
    """A large flat ground quad (2 tris) covering the terrain XY bbox expanded
    by `margin` metres, at z = ground_z.

    ground_z defaults to the terrain's min Z (the base floor), so the ground
    sits strictly at / below every stair and never clips through the geometry.
    Returns (gv (4,3), gf (2,3)) or None if verts is empty.
    """
    if verts.shape[0] == 0:
        return None
    z = float(verts[:, 2].min()) if ground_z is None else float(ground_z)
    x0 = float(verts[:, 0].min()) - margin
    x1 = float(verts[:, 0].max()) + margin
    y0 = float(verts[:, 1].min()) - margin
    y1 = float(verts[:, 1].max()) + margin
    gv = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
                  dtype=verts.dtype)
    gf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return gv, gf


def rasterize_height_field(verts, faces, resolution=0.02, fill_z=None):
    """Rasterize trimesh to a regular grid height field (IsaacLab format).

    Returns (height_field (H,W) float32, origin (x,y) float64, resolution float).
    Height at each cell = max Z of any triangle covering that XY cell.
    Cells not covered by any triangle are filled with `fill_z` (default: the
    terrain min Z) — this is the surrounding GROUND plane, so the height field
    has a continuous floor around/below the stairs instead of a void.
    """
    xmin, ymin = verts[:, 0].min(), verts[:, 1].min()
    xmax, ymax = verts[:, 0].max(), verts[:, 1].max()
    margin = resolution
    xmin -= margin; ymin -= margin
    xmax += margin; ymax += margin
    W = int(np.ceil((xmax - xmin) / resolution))
    H = int(np.ceil((ymax - ymin) / resolution))

    hf = np.full((H, W), np.nan, dtype=np.float32)
    tri_v = verts[faces]

    for tri in tri_v:
        txmin = tri[:, 0].min(); txmax = tri[:, 0].max()
        tymin = tri[:, 1].min(); tymax = tri[:, 1].max()
        ci0 = max(0, int((txmin - xmin) / resolution))
        ci1 = min(W - 1, int((txmax - xmin) / resolution))
        ri0 = max(0, int((tymin - ymin) / resolution))
        ri1 = min(H - 1, int((tymax - ymin) / resolution))

        v0, v1, v2 = tri[0], tri[1], tri[2]
        for ri in range(ri0, ri1 + 1):
            for ci in range(ci0, ci1 + 1):
                px = xmin + (ci + 0.5) * resolution
                py = ymin + (ri + 0.5) * resolution
                # barycentric test
                d00 = v1[0] - v0[0]; d01 = v2[0] - v0[0]
                d10 = v1[1] - v0[1]; d11 = v2[1] - v0[1]
                dp0 = px - v0[0]; dp1 = py - v0[1]
                det = d00 * d11 - d01 * d10
                if abs(det) < 1e-12:
                    continue
                u = (dp0 * d11 - d01 * dp1) / det
                v = (d00 * dp1 - dp0 * d10) / det
                if u >= -1e-6 and v >= -1e-6 and (u + v) <= 1.0 + 1e-6:
                    z = v0[2] + u * (v1[2] - v0[2]) + v * (v2[2] - v0[2])
                    if np.isnan(hf[ri, ci]) or z > hf[ri, ci]:
                        hf[ri, ci] = z

    fill = float(verts[:, 2].min()) if fill_z is None else float(fill_z)
    hf[np.isnan(hf)] = fill
    return hf, np.array([xmin, ymin], dtype=np.float64), resolution


def main():
    ap = argparse.ArgumentParser(description="Convert UE terrain to MuJoCo / IsaacLab formats")
    ap.add_argument("--terrain-dir", required=True)
    ap.add_argument("--config", required=True, help="Alignment config JSON (gasp_bvh_alignment_g1_height.json)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ground-z0-cm", type=float, default=0.0)
    ap.add_argument("--hf-resolution", type=float, default=0.02,
                    help="Height field grid resolution in meters (default 0.02 = 2cm)")
    ap.add_argument("--name", default="terrain",
                    help="Name prefix for the two output subdirs: "
                         "<output-dir>/<name>_mujoco and <output-dir>/<name>_isaaclab "
                    "(default 'terrain' -> terrain_mujoco / terrain_isaaclab). "
                    "Use e.g. --name stairs1 to match an existing stairs1_* layout.")
    ap.add_argument("--no-ground", action="store_true",
                    help="Do NOT append a flat ground plane. By default a large "
                         "ground quad is added at --ground-z (default 0.0, the "
                         "world floor where the character's feet land) covering "
                         "the XY bbox + --ground-margin, because the UE terrain "
                         "export only contains the stair/box geometry and NO "
                         "floor — without this the exported terrain has no "
                         "ground.")
    ap.add_argument("--ground-margin", type=float, default=5.0,
                    help="Metres the ground plane extends beyond the terrain XY "
                         "bbox on every side (default 5.0).")
    ap.add_argument("--ground-z", type=float, default=None,
                    help="Z (m, MuJoCo frame) of the ground plane. Default 0.0 "
                         "(the world floor — the retarget pipeline puts the foot "
                         "soles on z=0; the stair mesh dips below 0, so its min "
                         "is NOT the floor). Set e.g. -0.5 for a sunken floor.")
    args = ap.parse_args()

    terrain_dir = pathlib.Path(args.terrain_dir)
    output_dir = pathlib.Path(args.output_dir)
    mj_dir = output_dir / f"{args.name}_mujoco"
    il_dir = output_dir / f"{args.name}_isaaclab"
    mj_dir.mkdir(parents=True, exist_ok=True)
    il_dir.mkdir(parents=True, exist_ok=True)

    R_global, t_global = load_alignment(args.config)

    terrain_files = sorted(terrain_dir.glob("terrain_*.json"))
    if not terrain_files:
        print(f"No terrain_*.json found in {terrain_dir}")
        return

    print(f"Found {len(terrain_files)} terrain file(s) in {terrain_dir}")
    print(f"Output: {output_dir}\n")

    for tf in terrain_files:
        hash_id = tf.stem.replace("terrain_", "")
        print(f"--- {tf.name} (hash={hash_id}) ---")

        result = load_terrain_json(tf, R_global, t_global,
                                   ground_z0_cm=args.ground_z0_cm)
        if result is None:
            print("  [SKIP] empty or invalid terrain\n")
            continue
        verts, faces = result
        print(f"  {verts.shape[0]} verts, {faces.shape[0]} tris")
        print(f"  Z range: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}] m")

        # Detect a ground plane already baked into the source JSON by the UE
        # exporter (asset "/MotionCapture/Synthetic/GroundPlane"). When present,
        # skip our own injection so we don't add a SECOND ground plane.
        try:
            _raw = json.loads(tf.read_text(encoding="utf-8"))
            has_source_ground = any(
                "Synthetic/GroundPlane" in str(m.get("asset_path", ""))
                for m in _raw.get("meshes", []))
        except Exception:
            has_source_ground = False

        # The UE terrain export only contains the stair / box geometry — there
        # is NO floor mesh (unless the UE exporter already added a synthetic
        # ground, detected above). Append a flat ground quad so the exported
        # terrain has a continuous ground (base floor) around / below the
        # stairs. The ground is added to the MESH (obj / xml / npz vertices+
        # faces); the height field is rasterized from the stair-only geometry
        # and its outside cells are filled with ground_z (equivalent, but far
        # faster than rasterizing two bbox-spanning triangles).
        verts_hf, faces_hf = verts, faces
        # Ground is at the world floor Z. In the MuJoCo frame this is z=0: the
        # retarget pipeline places the character so the foot soles rest on
        # z=0 (verified from the motion: foot strikes land at z~0.04-0.07 m,
        # i.e. sole thickness above 0). The stair MESH dips below 0 (its min
        # is the underside of the stair base, e.g. -0.579 m), so using the
        # mesh min as the floor would sink the ground ~0.6 m below where the
        # feet actually walk. Override with --ground-z for non-zero floors.
        ground_z = 0.0 if args.ground_z is None else float(args.ground_z)
        if args.no_ground:
            print(f"  [Ground] DISABLED (--no-ground)")
        elif has_source_ground:
            print(f"  [Ground] already present in source JSON "
                  f"(Synthetic/GroundPlane) — not adding a second one")
        else:
            g = make_ground_quad(verts, args.ground_margin, ground_z)
            if g is not None:
                gv, gf = g
                n_v = verts.shape[0]
                verts = np.concatenate([verts, gv], axis=0)
                faces = np.concatenate([faces, gf + n_v], axis=0)
                print(f"  [Ground] added ground quad at z={ground_z:.4f} m, "
                      f"margin={args.ground_margin:g} m "
                      f"(verts {n_v} -> {verts.shape[0]}, "
                      f"tris {faces_hf.shape[0]} -> {faces.shape[0]})")

        mj_xml = mj_dir / f"{hash_id}.xml"
        mj_obj = mj_dir / f"{hash_id}.obj"
        write_mujoco_xml(mj_xml, verts, faces, hash_id)
        write_obj(mj_obj, verts, faces)
        print(f"  [MuJoCo]   {mj_xml.name}  +  {mj_obj.name}")

        il_obj = il_dir / f"{hash_id}.obj"
        il_npz = il_dir / f"{hash_id}.npz"
        write_obj(il_obj, verts, faces)
        print(f"  [IsaacLab] {il_obj.name}  (rasterizing height field...)")
        hf, origin, res = rasterize_height_field(verts_hf, faces_hf,
                                                 args.hf_resolution,
                                                 fill_z=ground_z)
        np.savez_compressed(il_npz,
                            vertices=verts,
                            faces=faces,
                            height_field=hf,
                            origin_xy=origin,
                            resolution=res,
                            coordinate_system="right_hand_z_up_meters")
        print(f"  [IsaacLab] {il_npz.name}  height_field shape={hf.shape}  "
              f"res={res:.3f}m")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
