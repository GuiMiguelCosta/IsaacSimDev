from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from omni.physx.scripts import utils as physx_utils
from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .constants import (
    AXIS_USD_PATH,
    BOTTOM_HOUSING_USD_PATH,
    CONTAINER_POSITION,
    CONTAINER_PRIM_PATH,
    CONTAINER_ROTATION,
    CONTAINER_SCALE,
    CONTAINER_USD_PATH,
    CUBE_1_POSITION,
    CUBE_1_PRIM_PATH,
    CUBE_1_ROTATION,
    CUBE_2_POSITION,
    CUBE_2_PRIM_PATH,
    CUBE_2_ROTATION,
    CUBE_3_POSITION,
    CUBE_3_PRIM_PATH,
    CUBE_3_ROTATION,
    DEFAULT_ROBOT_JOINTS,
    DEFAULT_ROBOT_JOINTS_BY_NAME,
    GROUND_PLANE_POSITION,
    GROUND_PRIM_PATH,
    IDENTITY_ROTATION,
    LIGHT_COLOR,
    LIGHT_INTENSITY,
    LIGHT_PRIM_PATH,
    PIECE_MAX_SAMPLE_TRIES,
    PIECE_MIN_SEPARATION,
    PIECE_POSITION_X_RANGE,
    PIECE_POSITION_Y_RANGE,
    PIECE_POSITION_Z_RANGE,
    PIECE_YAW_RANGE,
    ROBOT_POSITION,
    ROBOT_PRIM_PATH,
    ROBOT_ROTATION,
    ROBOT_USD_PATH,
    TABLE_POSITION,
    TABLE_PRIM_PATH,
    TABLE_ROTATION,
    TABLE_USD_PATH,
    TOP_BEARING_USD_PATH,
)


@dataclass
class IILABScene:
    robot: Articulation


def _create_physics_scene() -> None:
    stage = stage_utils.get_current_stage(backend="usd")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)


def _require_existing_asset(asset_path: str) -> str:
    path = Path(asset_path)
    if not path.exists():
        raise FileNotFoundError(f"USD asset not found: {path}")
    return str(path)


def _get_prim(prim_path: str) -> Usd.Prim:
    stage = stage_utils.get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return prim


def _make_default_joint_targets(robot: Articulation) -> np.ndarray:
    targets = np.zeros((1, robot.num_dofs), dtype=np.float32)

    name_to_index = {name: idx for idx, name in enumerate(robot.dof_names)}
    names_applied = 0
    for dof_name, joint_value in DEFAULT_ROBOT_JOINTS_BY_NAME.items():
        if dof_name in name_to_index:
            targets[0, name_to_index[dof_name]] = joint_value
            names_applied += 1

    if names_applied == 0:
        size = min(robot.num_dofs, DEFAULT_ROBOT_JOINTS.shape[1])
        targets[:, :size] = DEFAULT_ROBOT_JOINTS[:, :size]

    return targets


def _arm_dof_indices(robot: Articulation) -> list[int]:
    name_to_index = {name: idx for idx, name in enumerate(robot.dof_names)}
    indices = [name_to_index[f"joint_a{i + 1}"] for i in range(7) if f"joint_a{i + 1}" in name_to_index]
    return indices if indices else list(range(min(7, robot.num_dofs)))


def _finger_dof_indices(robot: Articulation) -> list[int]:
    name_to_index = {name: idx for idx, name in enumerate(robot.dof_names)}
    return [name_to_index["finger_joint"]] if "finger_joint" in name_to_index else []


def _configure_robot_physics(robot: Articulation) -> None:
    arm_dofs = _arm_dof_indices(robot)
    finger_dofs = _finger_dof_indices(robot)

    try:
        robot.set_solver_iteration_counts(position_counts=[64], velocity_counts=[16])
    except Exception:
        pass

    try:
        robot.set_link_enabled_gravities([False])
    except Exception:
        pass

    try:
        robot.set_dof_gains(stiffnesses=[400.0], dampings=[80.0], dof_indices=arm_dofs)
    except Exception:
        pass

    try:
        robot.set_dof_gains(stiffnesses=[400.0], dampings=[30.0], dof_indices=finger_dofs)
    except Exception:
        pass


def _set_xform(
    prim_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = IDENTITY_ROTATION,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    prim = _get_prim(prim_path)
    xformable = UsdGeom.Xformable(prim)
    prop_names = prim.GetPropertyNames()
    xformable.ClearXformOpOrder()

    if "xformOp:translate" not in prop_names:
        translate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
    else:
        translate_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        orient_op = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    else:
        orient_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))

    if "xformOp:scale" not in prop_names:
        scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
    else:
        scale_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        translate_op.Set(Gf.Vec3f(*translation))
    else:
        translate_op.Set(Gf.Vec3d(*translation))

    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
    else:
        orient_op.Set(Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:])))

    if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        scale_op.Set(Gf.Vec3f(*scale))
    else:
        scale_op.Set(Gf.Vec3d(*scale))

    xformable.SetXformOpOrder([translate_op, orient_op, scale_op])


def _add_reference(
    prim_path: str,
    usd_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = IDENTITY_ROTATION,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    stage_utils.add_reference_to_stage(usd_path=_require_existing_asset(usd_path), path=prim_path)
    _set_xform(prim_path, translation=translation, orientation=orientation, scale=scale)


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return a quaternion in Isaac's expected (w, x, y, z) order."""

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


def _ensure_rigid_body(prim_path: str, *, kinematic: bool, disable_gravity: bool) -> None:
    prim = _get_prim(prim_path)

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_enabled_attr = rigid_body_api.GetRigidBodyEnabledAttr()
    if rigid_body_enabled_attr:
        rigid_body_enabled_attr.Set(True)
    else:
        rigid_body_api.CreateRigidBodyEnabledAttr(True)

    kinematic_enabled_attr = rigid_body_api.GetKinematicEnabledAttr()
    if kinematic_enabled_attr:
        kinematic_enabled_attr.Set(kinematic)
    else:
        rigid_body_api.CreateKinematicEnabledAttr(kinematic)

    physx_rigid_body_api = (
        PhysxSchema.PhysxRigidBodyAPI(prim)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    )
    disable_gravity_attr = physx_rigid_body_api.GetDisableGravityAttr()
    if disable_gravity_attr:
        disable_gravity_attr.Set(disable_gravity)
    else:
        physx_rigid_body_api.CreateDisableGravityAttr(disable_gravity)


def _set_mass(prim_path: str, mass: float) -> None:
    prim = _get_prim(prim_path)
    mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr:
        mass_attr.Set(mass)
    else:
        mass_api.CreateMassAttr(mass)


def _configure_static_scene_asset(prim_path: str) -> None:
    prim = _get_prim(prim_path)
    physx_utils.setColliderSubtree(prim, approximationShape="none")
    _ensure_rigid_body(prim_path, kinematic=True, disable_gravity=True)


def _subtree_has_authored_collision(prim: Usd.Prim) -> bool:
    for child_prim in Usd.PrimRange(prim):
        if child_prim.HasAPI(UsdPhysics.CollisionAPI) or child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            return True
        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            return True
    return False


def _configure_static_scene_asset_with_authored_collision_fallback(prim_path: str) -> None:
    prim = _get_prim(prim_path)
    if not _subtree_has_authored_collision(prim):
        physx_utils.setColliderSubtree(prim, approximationShape="none")
    _ensure_rigid_body(prim_path, kinematic=True, disable_gravity=True)


def _configure_dynamic_piece(prim_path: str, *, mass: float = 0.05) -> None:
    prim = _get_prim(prim_path)
    physx_utils.setColliderSubtree(prim, approximationShape="convexDecomposition")
    _ensure_rigid_body(prim_path, kinematic=False, disable_gravity=False)
    _set_mass(prim_path, mass)


def _zero_rigid_body_velocity(prim_path: str) -> None:
    prim = _get_prim(prim_path)
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None
    if rigid_body_api is None:
        return

    linear_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
    linear_velocity_attr = rigid_body_api.GetVelocityAttr()
    if linear_velocity_attr:
        linear_velocity_attr.Set(linear_velocity)
    else:
        rigid_body_api.CreateVelocityAttr(linear_velocity)

    angular_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
    angular_velocity_attr = rigid_body_api.GetAngularVelocityAttr()
    if angular_velocity_attr:
        angular_velocity_attr.Set(angular_velocity)
    else:
        rigid_body_api.CreateAngularVelocityAttr(angular_velocity)


def _sample_piece_positions() -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for piece_index in range(3):
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


def _randomize_piece_poses() -> None:
    """Match the Isaac Lab task's piece spawn ranges instead of using a single fixed layout."""

    positions = _sample_piece_positions()
    piece_poses = (
        (CUBE_1_PRIM_PATH, positions[0], _quat_from_euler_xyz(0.0, 0.0, random.uniform(*PIECE_YAW_RANGE))),
        (CUBE_2_PRIM_PATH, positions[1], _quat_from_euler_xyz(0.0, 0.0, random.uniform(*PIECE_YAW_RANGE))),
        (CUBE_3_PRIM_PATH, positions[2], _quat_from_euler_xyz(math.pi / 2.0, 0.0, random.uniform(*PIECE_YAW_RANGE))),
    )
    for prim_path, position, orientation in piece_poses:
        _set_xform(prim_path, translation=position, orientation=orientation)
        _zero_rigid_body_velocity(prim_path)


def _set_float_attr(prim: Usd.Prim, attr_name: str, value: float) -> None:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
    attr.Set(float(value))


def _author_joint_pose(joint_prim: Usd.Prim, joint_position: float) -> None:
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_type = "angular"
        authored_position = np.rad2deg(joint_position)
        position_attr_name = "state:angular:physics:position"
        velocity_attr_name = "state:angular:physics:velocity"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_type = "linear"
        authored_position = joint_position
        position_attr_name = "state:linear:physics:position"
        velocity_attr_name = "state:linear:physics:velocity"
    else:
        return

    _set_float_attr(joint_prim, position_attr_name, authored_position)
    _set_float_attr(joint_prim, velocity_attr_name, 0.0)

    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
    if not drive_api:
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)

    target_position_attr = drive_api.GetTargetPositionAttr()
    if target_position_attr:
        target_position_attr.Set(authored_position)
    else:
        drive_api.CreateTargetPositionAttr(authored_position)


def _author_robot_joint_defaults(robot_prim_path: str) -> None:
    stage = stage_utils.get_current_stage(backend="usd")
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {robot_prim_path}")

    authored_joint_names = set()
    for prim in Usd.PrimRange(robot_prim):
        joint_name = prim.GetName()
        if joint_name not in DEFAULT_ROBOT_JOINTS_BY_NAME:
            continue
        _author_joint_pose(prim, DEFAULT_ROBOT_JOINTS_BY_NAME[joint_name])
        authored_joint_names.add(joint_name)

    missing_joint_names = sorted(set(DEFAULT_ROBOT_JOINTS_BY_NAME) - authored_joint_names)
    if missing_joint_names:
        raise RuntimeError(f"Could not find robot joint prims for: {', '.join(missing_joint_names)}")


def _create_scene_lighting() -> None:
    light = DomeLight(LIGHT_PRIM_PATH)
    light.set_colors([LIGHT_COLOR])
    light.set_intensities([LIGHT_INTENSITY])


def build_scene(robot_usd_path: str = ROBOT_USD_PATH) -> IILABScene:
    stage_utils.create_new_stage(template="empty")
    _create_physics_scene()
    _create_scene_lighting()

    GroundPlane(GROUND_PRIM_PATH, positions=[GROUND_PLANE_POSITION])

    _add_reference(TABLE_PRIM_PATH, TABLE_USD_PATH, translation=TABLE_POSITION, orientation=TABLE_ROTATION)
    _configure_static_scene_asset(TABLE_PRIM_PATH)
    _add_reference(
        CONTAINER_PRIM_PATH,
        CONTAINER_USD_PATH,
        translation=CONTAINER_POSITION,
        orientation=CONTAINER_ROTATION,
        scale=CONTAINER_SCALE,
    )
    _configure_static_scene_asset_with_authored_collision_fallback(CONTAINER_PRIM_PATH)
    _add_reference(
        CUBE_1_PRIM_PATH,
        BOTTOM_HOUSING_USD_PATH,
        translation=CUBE_1_POSITION,
        orientation=CUBE_1_ROTATION,
    )
    _configure_dynamic_piece(CUBE_1_PRIM_PATH)
    _add_reference(
        CUBE_2_PRIM_PATH,
        TOP_BEARING_USD_PATH,
        translation=CUBE_2_POSITION,
        orientation=CUBE_2_ROTATION,
    )
    _configure_dynamic_piece(CUBE_2_PRIM_PATH)
    _add_reference(
        CUBE_3_PRIM_PATH,
        AXIS_USD_PATH,
        translation=CUBE_3_POSITION,
        orientation=CUBE_3_ROTATION,
    )
    _configure_dynamic_piece(CUBE_3_PRIM_PATH)
    _add_reference(ROBOT_PRIM_PATH, robot_usd_path, translation=ROBOT_POSITION, orientation=ROBOT_ROTATION)
    _author_robot_joint_defaults(ROBOT_PRIM_PATH)
    robot = Articulation(ROBOT_PRIM_PATH)

    _configure_robot_physics(robot)
    scene = IILABScene(robot=robot)
    reset_scene(scene)

    return scene


def reset_scene(scene: IILABScene) -> None:
    """Reset the editor scene to an Isaac-Lab-like episode layout."""

    _randomize_piece_poses()
    reset_robot_pose(scene.robot)


def reset_robot_pose(robot: Articulation) -> None:
    default_targets = _make_default_joint_targets(robot)
    robot.set_default_state(dof_positions=default_targets)
    robot.set_dof_position_targets(default_targets)
    try:
        robot.set_dof_positions(default_targets)
    except Exception:
        pass
