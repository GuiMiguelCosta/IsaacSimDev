import os
from pathlib import Path

import numpy as np

_DEFAULT_IILAB_ASSETS_DIR = Path.home() / "Documents" / "git" / "iilab_usd_digitaltwin"
IILAB_ASSETS_DIR = Path(os.getenv("IILAB_ASSETS_DIR", str(_DEFAULT_IILAB_ASSETS_DIR))).expanduser()
ISAACLAB_ROOT = Path(os.getenv("ISAACLAB_ROOT", str(Path.home() / "IsaacLab"))).expanduser()


def _asset_path(env_var_name: str, relative_path: str) -> str:
    """Resolve an asset path from an environment variable or default root.

    Inputs are the environment variable name and relative fallback path; the
    output is an expanded filesystem path string. This exists so all configurable
    USD assets follow the same override behavior.
    """

    return str(Path(os.getenv(env_var_name, str(IILAB_ASSETS_DIR / relative_path))).expanduser())


ROBOT_USD_PATH = str(
    Path(
        os.getenv(
            "IILAB_KUKA_ROBOT_USD",
            str(
                IILAB_ASSETS_DIR
                / "Models"
                / "Equipments"
                / "Robots"
                / "kuka_lbr_iiwa"
                / "lbr_iiwa_14_r820"
                / "lbr_iiwa_14_r820.usd"
            ),
        )
    ).expanduser()
)
TABLE_USD_PATH = _asset_path(
    "IILAB_IBOT_CELL_USD",
    "Models/Equipments/LearningByDemonstration/ibotCell.usdc",
)
CONTAINER_USD_PATH = _asset_path(
    "IILAB_CONTAINER_USD",
    "Models/Assets/container_h20/container_h20.usd",
)
BOTTOM_HOUSING_USD_PATH = _asset_path(
    "IILAB_BOTTOM_HOUSING_USD",
    "Models/Assets/ibot_assets/BottomHousing.usdc",
)
TOP_BEARING_USD_PATH = _asset_path(
    "IILAB_TOP_BEARING_USD",
    "Models/Assets/ibot_assets/TopBearing.usdc",
)
AXIS_USD_PATH = _asset_path(
    "IILAB_AXIS_USD",
    "Models/Assets/ibot_assets/Axis.usdc",
)

ROBOT_PRIM_PATH = "/World/Robot"
GROUND_PRIM_PATH = "/World/GroundPlane"
LIGHT_PRIM_PATH = "/World/light"
TABLE_PRIM_PATH = "/World/Table"
CONTAINER_PRIM_PATH = "/World/Container"
CUBE_1_PRIM_PATH = "/World/Cube_1"
CUBE_2_PRIM_PATH = "/World/Cube_2"
CUBE_3_PRIM_PATH = "/World/Cube_3"

IDENTITY_ROTATION = (1.0, 0.0, 0.0, 0.0)
GROUND_PLANE_POSITION = (0.0, 0.0, -1.05)
TABLE_POSITION = (0.0, 0.0, 0.0)
TABLE_ROTATION = IDENTITY_ROTATION
ROBOT_POSITION = (0.0, 0.0, 0.0)
ROBOT_ROTATION = IDENTITY_ROTATION
CONTAINER_POSITION = (0.5, -0.25, -0.09)
CONTAINER_ROTATION = IDENTITY_ROTATION
CONTAINER_SCALE = (0.5, 0.5, 1.0)
CUBE_1_POSITION = (0.46, 0.08, -0.065)
CUBE_2_POSITION = (0.56, 0.18, -0.065)
CUBE_3_POSITION = (0.52, 0.32, -0.065)
CUBE_1_ROTATION = IDENTITY_ROTATION
CUBE_2_ROTATION = IDENTITY_ROTATION
CUBE_3_ROTATION = (0.7071068, 0.7071068, 0.0, 0.0)
PIECE_POSITION_X_RANGE = (0.4, 0.6)
PIECE_POSITION_Y_RANGE = (0.0, 0.4)
PIECE_POSITION_Z_RANGE = (-0.065, -0.065)
PIECE_YAW_RANGE = (-1.0, 1.0)
PIECE_MIN_SEPARATION = 0.12
PIECE_MAX_SAMPLE_TRIES = 5000
LIGHT_COLOR = (0.75, 0.75, 0.75)
LIGHT_INTENSITY = 3000.0

ROBOT_EE_LINK_NAME = "link_7"
ROBOT_EE_OFFSET = (0.0, 0.0, 0.210)
GRIPPER_OPEN_COMMAND = 0.0
GRIPPER_CLOSE_COMMAND = 0.8
TASK_SUCCESS_X_THRESHOLD = 0.16
TASK_SUCCESS_Y_THRESHOLD = 0.10
TASK_SUCCESS_Z_MIN_OFFSET = -0.02
TASK_SUCCESS_Z_MAX_OFFSET = 0.10
TASK_GRIPPER_OPEN_THRESHOLD = 0.05
POLICY_CONTROL_DT = 0.05
DEFAULT_ROBOMIMIC_PYTHON = os.getenv("IILAB_ROBOMIMIC_PYTHON", "").strip()
DEFAULT_ROBOMIMIC_CHECKPOINT = os.getenv("IILAB_ROBOMIMIC_CHECKPOINT", "").strip()
DEFAULT_ROBOMIMIC_NORM_FACTOR_MIN = os.getenv("IILAB_ROBOMIMIC_NORM_FACTOR_MIN", "").strip()
DEFAULT_ROBOMIMIC_NORM_FACTOR_MAX = os.getenv("IILAB_ROBOMIMIC_NORM_FACTOR_MAX", "").strip()

ROBOT_JOINT_NAMES = (
    "joint_a1",
    "joint_a2",
    "joint_a3",
    "joint_a4",
    "joint_a5",
    "joint_a6",
    "joint_a7",
    "finger_joint",
)
DEFAULT_ROBOT_JOINTS = np.array([[0.0, 0.0, 0.0, -1.309, 0.0, 1.745, 0.0, 0.0]], dtype=np.float32)
DEFAULT_ROBOT_JOINTS_BY_NAME = {
    joint_name: joint_value
    for joint_name, joint_value in zip(ROBOT_JOINT_NAMES, DEFAULT_ROBOT_JOINTS[0].tolist())
}
