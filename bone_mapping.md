# UEFN_Mannequin 骨骼 → Unitree G1 映射说明

## 1. "UEFN_Mannequin 骨骼 → G1" 映射代码的位置

需要先澄清一点：仓库里没有字面量 `UEFN_Mannequin` 这个标识符（全仓 grep 0 命中）。这里说的"UE5 Mannequin 骨骼"在代码里被当作 **`bvh_ue5_native`**（原始 1.75 m 人形）和 **`bvh_ue5_g1scale`**（已缩放到 G1 高度的人形）两种 src-human 来处理，骨骼名就是 UE Mannequin 的标准命名（`pelvis / spine_01..05 / thigh_l / calf_l / foot_l / ball_l / upperarm_l / lowerarm_l / hand_l …`）。映射到 G1 的对应关系分散在三个层次：

### (a) 骨骼层级 / 重要骨骼定义

`Scripts/tools/vis_skeleton_compare.py` 里 `BONE_CONNECTIONS` 和 `IMPORTANT_BONES` 定义了源人形骨架结构（UE Mannequin 命名）：

```python
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
```

### (b) G1 ↔ 人形 关节配对（对应关系表）

`Scripts/joint_mapping.json` 是一张 robot↔human 配对表（12 对 + 根），用于误差评估：

```json
{
    "description": "Robot-to-human joint mapping (G1 <-> scaled human)",
    "robot": "unitree_g1",
    "human_src": "bvh_ue5_g1scale",
    "mappings": [
        { "robot": "left_hip_pitch_link",  "human": "thigh_l" },
        { "robot": "right_hip_pitch_link", "human": "thigh_r" },
        { "robot": "left_knee_link",      "human": "calf_l" },
        { "robot": "right_knee_link",      "human": "calf_r" },
        { "robot": "left_ankle_roll_link", "human": "LeftFootMod" },
        { "robot": "right_ankle_roll_link","human": "RightFootMod" },
        { "robot": "left_shoulder_pitch_link","human": "upperarm_l" },
        { "robot": "right_shoulder_pitch_link","human": "upperarm_r" },
        { "robot": "left_elbow_link",     "human": "lowerarm_l" },
        { "robot": "right_elbow_link",    "human": "lowerarm_r" },
        { "robot": "left_wrist_yaw_link", "human": "hand_l" },
        { "robot": "right_wrist_yaw_link","human": "hand_r" }
    ]
}
```

### (c) IK 重定向配置（真正的映射 + 缩放 + 初始对齐四元数）

GMR 引擎使用的 IK 配置 JSON，是"骨骼映射"的真正落点：

- `Scripts/UEGMR/Retargeting/gasp_bvh_alignment.json` / `gasp_bvh_alignment_g1_height.json` — 全局 R/t + 每骨 Δ
- `Scripts/UEGMR/GMR/general_motion_retargeting/ik_configs/bvh_ue5_native_to_g1.json`（1.75 m 源）
- `Scripts/UEGMR/GMR/general_motion_retargeting/ik_configs/bvh_ue5_g1scale_to_g1.json`（G1-scale 源）

`bvh_ue5_native_to_g1.json` 的 `ik_match_table1/2` 就是逐骨的 `robot_body → human_bone` 映射，含位置偏移、权重和初始对齐四元数：

```json
"ik_match_table1": {
    "pelvis":                  ["pelvis",       0,  10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "left_hip_yaw_link":       ["thigh_l",      0,  10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "left_knee_link":          ["calf_l",       0,  10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "left_ankle_roll_link":    ["LeftFootMod",  50, 10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "right_hip_yaw_link":      ["thigh_r",      0,  10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "right_knee_link":         ["calf_r",       0,  10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "right_ankle_roll_link":   ["RightFootMod", 50, 10,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "torso_link":              ["spine_05",     0, 100,  [0,0,0], [ 0.7071, 0, 0, -0.7071]],
    "left_shoulder_yaw_link":  ["upperarm_l",   0, 100,  [0,0,0], [ 0.6601, 0.2967, -0.2829, -0.6294]],
    "left_elbow_link":         ["lowerarm_l",   0,  10,  [0,0,0], [ 0.5660, 0.7066, 0.0266, -0.4238]],
    "left_wrist_yaw_link":     ["hand_l",       0,  10,  [0,0,0], [ 0.5660, 0.7066, 0.0266, -0.4238]],
    "right_shoulder_yaw_link": ["upperarm_r",   0, 100,  [0,0,0], [-0.6294, 0.2829, -0.2967,  0.6601]],
    "right_elbow_link":        ["lowerarm_r",   0,  10,  [0,0,0], [ 0.4238, 0.0266,  0.7066, -0.5660]],
    "right_wrist_yaw_link":    ["hand_r",       0,  10,  [0,0,0], [ 0.4238, 0.0266,  0.7066, -0.5660]]
}
```

### (d) 调用映射的 Python 代码

`Scripts/tools/ue_world_skeleton_retarget.py` 是主入口（`Scripts/UEGMR/Retargeting/ue_world_skeleton_retarget.py` 是同一份的另一处副本）。它做三件事：把 GASP JSONL 装成 `{bone_name: [pos, quat_wxyz]}`、合成 `LeftFootMod/RightFootMod`（foot 位置 + ball 朝向），再用 `src_human="bvh_ue5_native" / "bvh_ue5_g1scale"`、`tgt_robot="unitree_g1"` 调 GMR：

```python
for gi, name in enumerate(gasp_names):
    # Step 2a: global rigid transform
    pos_aligned = R_global @ gasp_pos_mj[fi, gi] + t_global
    rot_aligned = R_global @ gasp_rot_all[fi][gi]
    # Step 2b: per-bone delta (residual local-frame mismatch)
    D = delta.get(name, identity3)
    R_eq = rot_aligned @ D
    quat = R.from_matrix(R_eq).as_quat(scalar_first=True)  # wxyz
    result[name] = [pos_aligned, quat]

# Synthesized IK targets used by bvh_ue5_native_to_g1 config
if "foot_l" in result and "ball_l" in result:
    result["LeftFootMod"]  = [result["foot_l"][0].copy(), result["ball_l"][1].copy()]
if "foot_r" in result and "ball_r" in result:
    result["RightFootMod"] = [result["foot_r"][0].copy(), result["ball_r"][1].copy()]
```

```python
print(f"  Init GMR (src=bvh_ue5_native, tgt=unitree_g1, height={actual_human_height:.3f}m)")
retargeter = GMR(
    src_human="bvh_ue5_native",
    tgt_robot="unitree_g1",
    actual_human_height=actual_human_height,
)
```

## 2. 是否已整合到 `D:\tool\ue5\UnrealProjects\GASP\Morph`

**是的，已经完整整合**。`Morph/` 是把 `Scripts/UEGMR` 这套重定向 + 地形导出工具抽出来做成的一个自包含子仓库（README 里明确说"trimmed, vendored copy of GMR"）。对应关系如下：

| Scripts 中的位置 | Morph 中的对应位置 |
|---|---|
| `Scripts/tools/ue_world_skeleton_retarget.py` | `Morph/DataLib/retarget/ue_world_skeleton_retarget.py` |
| `Scripts/tools/vis_skeleton_compare.py` | `Morph/DataLib/retarget/vis_skeleton_compare.py` |
| `Scripts/tools/gasp_bvh_calibrate.py` | `Morph/DataLib/retarget/gasp_bvh_calibrate.py` |
| `Scripts/tools/gasp_bvh_alignment.json` | `Morph/DataLib/retarget/gasp_bvh_alignment.json` |
| `Scripts/tools/gasp_bvh_alignment_g1_height.json` | `Morph/DataLib/retarget/gasp_bvh_alignment_g1_height.json` |
| `Scripts/joint_mapping.json` | `Morph/DataLib/retarget/joint_mapping.json` |
| `Scripts/UEGMR/GMR/general_motion_retargeting/` | `Morph/DataLib/gmr/general_motion_retargeting/`（vendored 精简版） |
| `Scripts/UEGMR/GMR/.../ik_configs/bvh_ue5_native_to_g1.json` | `Morph/DataLib/gmr/general_motion_retargeting/ik_configs/bvh_ue5_native_to_g1.json` |
| `Scripts/UEGMR/GMR/.../ik_configs/bvh_ue5_g1scale_to_g1.json` | `Morph/DataLib/gmr/general_motion_retargeting/ik_configs/bvh_ue5_g1scale_to_g1.json` |
| `Scripts/UEGMR/GMR/assets/unitree_g1/` | `Morph/DataLib/gmr/assets/unitree_g1/` |

`Morph/README.md` 第 9–13 行也写明了这种整合关系：

> The repository is self-contained: the retargeting engine is a **trimmed,
> vendored copy** of the [GMR](https://github.com/YanjieZe/GMR) library under
> `DataLib/gmr/`, so no external `general_motion_retargeting` install is needed.
> You only supply your own recordings (drop them under `data/`) and run the
> scripts for post-processing, export, and visualization.

`bvh_ue5_g1scale_to_g1.json` 里 `human_height_assumption=1.32`、`human_scale_table` 缩放系数 1.1932/1.0606，与 README §3 描述的 G1-scale 配置完全吻合。

## 3. 小结

- **映射代码**：`Scripts/tools/ue_world_skeleton_retarget.py` + `Scripts/UEGMR/GMR/general_motion_retargeting/` 下的 IK 配置 JSON（`bvh_ue5_native_to_g1.json` / `bvh_ue5_g1scale_to_g1.json`）+ `Scripts/joint_mapping.json`。
- **对应关系**：根 `pelvis↔pelvis`，躯干 `torso_link↔spine_05`，腿 `hip_pitch/knee/ankle_roll_link ↔ thigh_l/calf_l/LeftFootMod`（右同），臂 `shoulder_yaw/elbow/wrist_yaw_link ↔ upperarm_l/lowerarm_l/hand_l`（右同）。`LeftFootMod/RightFootMod` 是合成的（foot 位置 + ball 朝向）。
- **是否整合到 Morph**：是，`Morph/DataLib/retarget/` 和 `Morph/DataLib/gmr/` 已经把整套映射代码、IK 配置、G1 资产都 vendor 进去了，`Morph/` 可独立运行，不依赖 `Scripts/`。
