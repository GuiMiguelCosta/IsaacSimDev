from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation

from .constants import (
    CONTAINER_POSITION,
    CONTAINER_PRIM_PATH,
    CONTAINER_ROTATION,
    CONTAINER_SCALE,
    CONTAINER_USD_PATH,
    GROUND_PLANE_POSITION,
    GROUND_PRIM_PATH,
    LIGHT_COLOR,
    LIGHT_INTENSITY,
    LIGHT_PRIM_PATH,
    ROBOT_POSITION,
    ROBOT_PRIM_PATH,
    ROBOT_ROTATION,
    ROBOT_USD_PATH,
    TABLE_POSITION,
    TABLE_PRIM_PATH,
    TABLE_ROTATION,
    TABLE_USD_PATH,
)
from .scene_model import IILABScene
from .scene_objects import DEFAULT_OBJECTS, add_scene_object_reference, find_next_object_indices, make_default_object_prim_path
from .scene_physics import (
    add_reference,
    configure_static_scene_asset,
    configure_static_scene_asset_with_authored_collision_fallback,
    create_physics_scene,
)
from .scene_reset import reset_scene
from .scene_robot import author_robot_joint_defaults, configure_robot_physics


def create_scene_lighting() -> None:
    """Create and configure scene lighting.

    There are no inputs or outputs. This exists so the stage builder does not
    mix lighting details with asset references and physics setup.
    """

    light = DomeLight(LIGHT_PRIM_PATH)
    light.set_colors([LIGHT_COLOR])
    light.set_intensities([LIGHT_INTENSITY])


def build_scene(robot_usd_path: str = ROBOT_USD_PATH) -> IILABScene:
    """Build the full IILAB task scene on a fresh USD stage.

    Input is the robot USD path; the output is an IILABScene handle.
    This exists as the single public scene-construction entry point used by the
    extension and queue import path.
    """

    stage_utils.create_new_stage(template="empty")
    create_physics_scene()
    create_scene_lighting()

    GroundPlane(GROUND_PRIM_PATH, positions=[GROUND_PLANE_POSITION])

    add_reference(TABLE_PRIM_PATH, TABLE_USD_PATH, translation=TABLE_POSITION, orientation=TABLE_ROTATION)
    configure_static_scene_asset(TABLE_PRIM_PATH)
    add_reference(
        CONTAINER_PRIM_PATH,
        CONTAINER_USD_PATH,
        translation=CONTAINER_POSITION,
        orientation=CONTAINER_ROTATION,
        scale=CONTAINER_SCALE,
    )
    configure_static_scene_asset_with_authored_collision_fallback(CONTAINER_PRIM_PATH)
    scene_objects = [
        add_scene_object_reference(object_key, make_default_object_prim_path(object_key))
        for object_key in DEFAULT_OBJECTS
    ]
    add_reference(ROBOT_PRIM_PATH, robot_usd_path, translation=ROBOT_POSITION, orientation=ROBOT_ROTATION)
    author_robot_joint_defaults(ROBOT_PRIM_PATH)
    robot = Articulation(ROBOT_PRIM_PATH)

    configure_robot_physics(robot)
    scene = IILABScene(robot=robot, objects=scene_objects, next_object_indices=find_next_object_indices(scene_objects))
    reset_scene(scene)

    return scene
