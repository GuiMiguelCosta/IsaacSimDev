from __future__ import annotations

import math
import random

from .constants import (
    PIECE_MAX_SAMPLE_TRIES,
    PIECE_MIN_SEPARATION,
    PIECE_POSITION_X_RANGE,
    PIECE_POSITION_Y_RANGE,
    PIECE_POSITION_Z_RANGE,
    PIECE_YAW_RANGE,
)
from .scene_model import IILABScene
from .scene_objects import get_scene_object_spec
from .scene_physics import set_xform, zero_rigid_body_velocity
from .scene_robot import reset_robot_pose


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles into Isaac's (w, x, y, z) quaternion order.

    Inputs are roll, pitch, and yaw in radians; the output is a quaternion.
    This exists so randomized piece yaw can be authored without pulling in a
    heavier transform dependency.
    """

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def sample_piece_positions(piece_count: int) -> list[tuple[float, float, float]]:
    """Sample non-overlapping object spawn positions when possible.

    Input is the number of pieces; the output is a list of positions.
    This exists to match the Isaac Lab task's randomized spawn ranges while
    preventing obvious initial object overlaps.
    """

    positions: list[tuple[float, float, float]] = []
    for piece_index in range(piece_count):
        for attempt_index in range(PIECE_MAX_SAMPLE_TRIES):
            position = (
                random.uniform(*PIECE_POSITION_X_RANGE),
                random.uniform(*PIECE_POSITION_Y_RANGE),
                random.uniform(*PIECE_POSITION_Z_RANGE),
            )
            if piece_index == 0 or attempt_index == PIECE_MAX_SAMPLE_TRIES - 1:
                positions.append(position)
                break
            if all(math.dist(position, existing_position) > PIECE_MIN_SEPARATION for existing_position in positions):
                positions.append(position)
                break
    return positions


def randomize_piece_poses(scene: IILABScene) -> None:
    """Randomize table object poses and clear their velocities.

    Input is the loaded scene; there is no output.
    This exists to reset object layout between episodes and after object
    add/remove/import operations.
    """

    positions = sample_piece_positions(len(scene.objects))
    for scene_object, position in zip(scene.objects, positions):
        spec = get_scene_object_spec(scene_object.key)
        orientation = quat_from_euler_xyz(spec.random_roll, 0.0, random.uniform(*PIECE_YAW_RANGE))
        set_xform(scene_object.prim_path, translation=position, orientation=orientation)
        zero_rigid_body_velocity(scene_object.prim_path)


def reset_scene(scene: IILABScene) -> None:
    """Reset the editor scene to an Isaac-Lab-like episode layout.

    Input is the scene object; there is no output.
    This exists as the shared reset operation after scene creation, object
    changes, queue import, and policy restarts.
    """

    randomize_piece_poses(scene)
    reset_robot_pose(scene.robot)
