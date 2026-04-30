from __future__ import annotations

import re

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import Articulation
from pxr import Usd, UsdPhysics

from .constants import DEFAULT_ROBOT_JOINTS, DEFAULT_ROBOT_JOINTS_BY_NAME
from .scene_physics import set_float_attr

ROBOT_SOLVER_POSITION_ITERATIONS = 32
ROBOT_SOLVER_VELOCITY_ITERATIONS = 1
ROBOT_ARM_RESET_NOISE_STD = 0.02


def make_default_joint_targets(robot: Articulation) -> np.ndarray:
    """Build the robot's default joint target vector.

    Input is an articulation; the output is a (1, dof_count) float array.
    This exists to prefer named joint defaults while preserving an index-based
    fallback for asset variants that expose different DOF names.
    """

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


def named_dof_indices(robot: Articulation, joint_names: list[str]) -> list[int]:
    """Resolve articulation DOF names into indices.

    Inputs are an articulation and desired joint names; the output is matching
    indices. This exists so physics tuning can tolerate missing optional joints.
    """

    name_to_index = {name: idx for idx, name in enumerate(robot.dof_names)}
    return [name_to_index[joint_name] for joint_name in joint_names if joint_name in name_to_index]


def arm_dof_indices(robot: Articulation) -> list[int]:
    """Return the seven arm DOF indices, falling back to leading DOFs.

    Input is an articulation; the output is a list of indices.
    This exists because policy control needs stable arm joints even when an
    imported articulation omits expected names.
    """

    indices = named_dof_indices(robot, [f"joint_a{i + 1}" for i in range(7)])
    return indices if indices else list(range(min(7, robot.num_dofs)))


def finger_dof_indices(robot: Articulation) -> list[int]:
    """Return the directly actuated gripper finger DOF indices.

    Input is an articulation; the output is a list of indices.
    This exists to isolate gripper actuation naming from controller code.
    """

    return named_dof_indices(robot, ["finger_joint"])


def inner_finger_dof_indices(robot: Articulation) -> list[int]:
    """Return inner finger joint DOFs used by the imported gripper.

    Input is an articulation; the output is a list of indices.
    This exists so gripper observation and tuning can handle mimic-style joints.
    """

    return [index for index, name in enumerate(robot.dof_names) if re.fullmatch(r".*_inner_finger_joint", name)]


def passive_gripper_dof_indices(robot: Articulation) -> list[int]:
    """Return passive gripper linkage DOFs that should not fight actuation.

    Input is an articulation; the output is a list of indices.
    This exists to set low gains on passive gripper joints and reduce solver
    instability during manipulation.
    """

    return [
        index
        for index, name in enumerate(robot.dof_names)
        if re.fullmatch(r".*_inner_finger_knuckle_joint", name) or name == "right_outer_knuckle_joint"
    ]


def set_robot_dof_limits(
    robot: Articulation,
    dof_indices: list[int],
    *,
    max_effort: float,
    max_velocity: float,
) -> None:
    """Apply effort and velocity limits to selected DOFs.

    Inputs are an articulation, DOF indices, and limits; there is no output.
    This exists to keep robot tuning grouped and tolerant of Isaac API variants.
    """

    if not dof_indices:
        return

    try:
        robot.set_max_efforts(np.full((1, len(dof_indices)), max_effort, dtype=np.float32), joint_indices=dof_indices)
    except Exception:
        pass

    try:
        robot.set_max_joint_velocities(
            np.full((1, len(dof_indices)), max_velocity, dtype=np.float32),
            joint_indices=dof_indices,
        )
    except Exception:
        pass


def configure_robot_physics(robot: Articulation) -> None:
    """Apply articulation solver, gravity, limit, and gain settings.

    Input is the robot articulation; there is no output.
    This exists to keep robot-specific PhysX tuning out of generic scene build
    code and make joint group handling explicit.
    """

    shoulder_dofs = named_dof_indices(robot, [f"joint_a{i + 1}" for i in range(4)])
    forearm_dofs = named_dof_indices(robot, [f"joint_a{i + 1}" for i in range(4, 7)])
    arm_dofs = arm_dof_indices(robot)
    finger_dofs = finger_dof_indices(robot)
    inner_finger_dofs = inner_finger_dof_indices(robot)
    passive_gripper_dofs = passive_gripper_dof_indices(robot)

    try:
        robot.set_solver_iteration_counts(
            position_counts=[ROBOT_SOLVER_POSITION_ITERATIONS],
            velocity_counts=[ROBOT_SOLVER_VELOCITY_ITERATIONS],
        )
    except Exception:
        pass

    try:
        robot.set_link_enabled_gravities([False])
    except Exception:
        pass

    set_robot_dof_limits(robot, shoulder_dofs, max_effort=87.0, max_velocity=2.175)
    set_robot_dof_limits(robot, forearm_dofs, max_effort=12.0, max_velocity=2.61)
    set_robot_dof_limits(robot, finger_dofs, max_effort=2.0, max_velocity=2.61)
    set_robot_dof_limits(robot, inner_finger_dofs, max_effort=1650.0, max_velocity=5.0)
    set_robot_dof_limits(robot, passive_gripper_dofs, max_effort=1000.0, max_velocity=10.0)

    if arm_dofs:
        try:
            robot.set_dof_gains(stiffnesses=[400.0], dampings=[80.0], dof_indices=arm_dofs)
        except Exception:
            pass

    if finger_dofs:
        try:
            robot.set_dof_gains(stiffnesses=[400.0], dampings=[30.0], dof_indices=finger_dofs)
        except Exception:
            pass

    if inner_finger_dofs:
        try:
            robot.set_dof_gains(stiffnesses=[1000.0], dampings=[100.0], dof_indices=inner_finger_dofs)
        except Exception:
            pass

    if passive_gripper_dofs:
        try:
            robot.set_dof_gains(stiffnesses=[0.0], dampings=[0.0], dof_indices=passive_gripper_dofs)
        except Exception:
            pass


def author_joint_pose(joint_prim: Usd.Prim, joint_position: float) -> None:
    """Author initial state and drive target on a USD joint prim.

    Inputs are a joint prim and position in SI units/radians; there is no output.
    This exists so the USD stage itself starts from the same pose that the
    articulation controller later commands.
    """

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

    set_float_attr(joint_prim, position_attr_name, authored_position)
    set_float_attr(joint_prim, velocity_attr_name, 0.0)

    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
    if not drive_api:
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)

    target_position_attr = drive_api.GetTargetPositionAttr()
    if target_position_attr:
        target_position_attr.Set(authored_position)
    else:
        drive_api.CreateTargetPositionAttr(authored_position)


def author_robot_joint_defaults(robot_prim_path: str) -> None:
    """Author default joint positions into the robot USD subtree.

    Input is the robot prim path; there is no output.
    This exists because imported articulations can otherwise initialize from USD
    defaults before the extension has a chance to command targets.
    """

    stage = stage_utils.get_current_stage(backend="usd")
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {robot_prim_path}")

    authored_joint_names = set()
    for prim in Usd.PrimRange(robot_prim):
        joint_name = prim.GetName()
        if joint_name not in DEFAULT_ROBOT_JOINTS_BY_NAME:
            continue
        author_joint_pose(prim, DEFAULT_ROBOT_JOINTS_BY_NAME[joint_name])
        authored_joint_names.add(joint_name)

    missing_joint_names = sorted(set(DEFAULT_ROBOT_JOINTS_BY_NAME) - authored_joint_names)
    if missing_joint_names:
        raise RuntimeError(f"Could not find robot joint prims for: {', '.join(missing_joint_names)}")


def reset_robot_pose(robot: Articulation) -> None:
    """Reset robot targets and current DOF positions around the default pose.

    Input is the robot articulation; there is no output.
    This exists to create a fresh episode pose while adding small arm noise that
    matches the task's randomized starts.
    """

    default_targets = make_default_joint_targets(robot)
    arm_dofs = arm_dof_indices(robot)
    if arm_dofs:
        default_targets = default_targets.copy()
        default_targets[0, arm_dofs] += np.random.normal(0.0, ROBOT_ARM_RESET_NOISE_STD, size=len(arm_dofs)).astype(
            np.float32
        )

    robot.set_default_state(dof_positions=default_targets)
    robot.set_dof_position_targets(default_targets)
    try:
        robot.set_dof_positions(default_targets)
    except Exception:
        pass
