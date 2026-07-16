"""General Motion Retargeting (GMR) — trimmed vendored subset.

Only the pieces required by the UE-world-skeleton -> Unitree G1 retarget
pipeline are exported here, so the release does NOT pull in the heavy optional
dependencies of the upstream package (torch / imageio / loop_rate_limiters /
opencv / xrobotoolkit). Everything the retarget scripts import is provided:

    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import ROBOT_XML_DICT, ...

If you need the upstream viewer / streaming features, install the full package
from https://github.com/YanjieZe/GMR instead.
"""
from .params import (
    IK_CONFIG_ROOT,
    ASSET_ROOT,
    ROBOT_XML_DICT,
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from .motion_retarget import GeneralMotionRetargeting
from .neck_retarget import human_head_to_robot_neck
from .data_loader import load_robot_motion

__all__ = [
    "IK_CONFIG_ROOT",
    "ASSET_ROOT",
    "ROBOT_XML_DICT",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
    "GeneralMotionRetargeting",
    "human_head_to_robot_neck",
    "load_robot_motion",
]
