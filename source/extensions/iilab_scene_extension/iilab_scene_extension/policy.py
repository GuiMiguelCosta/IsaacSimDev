from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import (
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_OPEN_COMMAND,
    POLICY_CONTROL_DT,
    ROBOT_EE_LINK_NAME,
    ROBOT_EE_OFFSET,
    TASK_GRIPPER_OPEN_THRESHOLD,
    TASK_SUCCESS_X_THRESHOLD,
    TASK_SUCCESS_Y_THRESHOLD,
    TASK_SUCCESS_Z_MAX_OFFSET,
    TASK_SUCCESS_Z_MIN_OFFSET,
)
from .policy_observations import (
    PolicyMetadata,
    PolicyObservationAdapter,
    PolicyObservationSnapshot,
    resolve_gripper_observation_layout,
)
from .pose_math import (
    POSE_ERROR_EPS,
    axis_angle_to_quat,
    combine_pose,
    compute_dls_joint_targets,
    compute_pose_error,
    get_world_pose,
    normalize_quat,
    quat_multiply,
    subtract_frame_position,
    to_numpy,
)
from .robomimic_client import RobomimicInferenceWorker, guess_latest_checkpoint
from .scene_model import IILABScene
from .scene_robot import make_default_joint_targets

__all__ = ["RobomimicInferenceWorker", "RobomimicPolicyController", "guess_latest_checkpoint"]


@dataclass
class RobomimicPolicyController:
    """Run one robomimic policy against the loaded Isaac scene."""

    scene: IILABScene
    worker: RobomimicInferenceWorker
    target_object_prim_path: str
    target_position_prim_path: str
    status_callback: Callable[[str], None] | None = None
    control_dt: float = POLICY_CONTROL_DT
    completion_confirmation_steps: int = 5
    _accumulated_dt: float = field(default=0.0, init=False, repr=False)
    _default_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False, repr=False)
    _last_action: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False, repr=False)
    _policy_metadata: PolicyMetadata | None = field(default=None, init=False, repr=False)
    _observation_adapter: PolicyObservationAdapter | None = field(default=None, init=False, repr=False)
    _completion_confirmation_count: int = field(default=0, init=False, repr=False)
    _task_complete_latched: bool = field(default=False, init=False, repr=False)
    _completion_status_message: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate targets, start the worker, and prepare control state.

        Inputs are the dataclass fields; there is no output.
        This exists to fail before simulation starts if the robot, target prims,
        worker metadata, or observation adapter cannot be prepared.
        """

        name_to_index = {name: index for index, name in enumerate(self.scene.robot.dof_names)}
        self._arm_dof_indices = [name_to_index[f"joint_a{i + 1}"] for i in range(7) if f"joint_a{i + 1}" in name_to_index]
        self._finger_dof_indices = [name_to_index["finger_joint"]] if "finger_joint" in name_to_index else []
        self._gripper_observation_dof_indices, self._gripper_observation_signs = resolve_gripper_observation_layout(
            list(self.scene.robot.dof_names)
        )

        if len(self._arm_dof_indices) != 7:
            raise RuntimeError("Could not resolve the seven KUKA arm joints on the articulation.")

        try:
            self._ee_link_index = self.scene.robot.link_names.index(ROBOT_EE_LINK_NAME)
        except ValueError as exc:
            raise RuntimeError(f"Could not find robot end-effector link '{ROBOT_EE_LINK_NAME}'.") from exc

        self._ee_jacobian_index = max(self._ee_link_index - 1, 0)
        self._ee_link_path = self.scene.robot.link_paths[0][self._ee_link_index]
        self._default_joint_positions = make_default_joint_targets(self.scene.robot).reshape(-1).astype(np.float32)

        if self.target_object_prim_path == self.target_position_prim_path:
            raise RuntimeError("Target object prim and target position prim must be different.")

        get_world_pose(self.target_object_prim_path)
        get_world_pose(self.target_position_prim_path)

        self.worker.start()
        self._policy_metadata = self.worker.metadata
        self._last_action = np.zeros(self._policy_metadata.action_dim, dtype=np.float32)
        self._observation_adapter = PolicyObservationAdapter(
            metadata=self._policy_metadata,
            target_object_prim_path=self.target_object_prim_path,
            target_position_prim_path=self.target_position_prim_path,
        )
        self._make_observation()
        self.worker.reset_episode()

    def reset(self) -> None:
        """Reset controller timing, last action, completion latch, and worker state.

        There are no inputs or outputs. This exists when the timeline resets but
        the selected policy should remain attached to the scene.
        """

        self._accumulated_dt = 0.0
        if self._last_action.size > 0:
            self._last_action.fill(0.0)
        self._completion_confirmation_count = 0
        self._task_complete_latched = False
        self._completion_status_message = ""
        self.worker.reset_episode()

    def close(self) -> None:
        """Close the robomimic worker owned by this controller.

        There are no inputs or outputs. This exists so extension stop/shutdown
        code does not need to know the worker lifecycle details.
        """

        self.worker.close()

    def on_physics_step(self, step_dt: float) -> None:
        """Advance policy control on accumulated physics timestep time.

        Input is the physics step duration; there is no output.
        This exists to run policy inference at control_dt even when PhysX steps
        at a different frequency.
        """

        if self._task_complete_latched:
            return

        self._accumulated_dt += float(step_dt)
        while self._accumulated_dt + POSE_ERROR_EPS >= self.control_dt:
            self._accumulated_dt -= self.control_dt

            if self._update_completion_state():
                break

            self._step_policy_once()
            if self._update_completion_state():
                break

    def _step_policy_once(self) -> None:
        """Build observations, infer one action, and apply it to the robot.

        There are no inputs or outputs. This exists as the single-cycle policy
        control primitive used by the timestep accumulator.
        """

        observation = self._make_observation()
        action = self.worker.infer(observation)
        self._last_action = action.astype(np.float32)
        self._apply_action(self._last_action)

    def hold_current_pose(self) -> None:
        """Command the robot to hold its current arm and gripper positions.

        There are no inputs or outputs. This exists to make stopping a policy
        leave the robot stable instead of falling back to stale targets.
        """

        arm_positions = to_numpy(self.scene.robot.get_dof_positions(dof_indices=self._arm_dof_indices)).reshape(1, -1)
        self.scene.robot.set_dof_position_targets(arm_positions.astype(np.float32), dof_indices=self._arm_dof_indices)

        if self._finger_dof_indices:
            finger_positions = to_numpy(
                self.scene.robot.get_dof_positions(dof_indices=self._finger_dof_indices)
            ).reshape(1, -1)
            self.scene.robot.set_dof_position_targets(
                finger_positions.astype(np.float32),
                dof_indices=self._finger_dof_indices,
            )

    def is_task_complete(self) -> bool:
        """Return whether the target object has reached the target container.

        There are no inputs; the output is a boolean.
        This exists so the extension can advance policy queues when the current
        manipulation objective has completed.
        """

        if self._task_complete_latched:
            return True

        self._update_completion_state()
        if self._task_complete_latched:
            self.hold_current_pose()
            return True

        return False

    @property
    def completion_status_message(self) -> str:
        """Return the most recent task-completion status text.

        There are no inputs; the output is a status string.
        This exists so queue advancement can include the completed policy target
        in the extension status panel.
        """

        return self._completion_status_message

    def _update_completion_state(self) -> bool:
        """Update the debounced completion latch.

        There are no inputs; the output is True once the task is latched done.
        This exists to avoid advancing queues from a single noisy success sample.
        """

        if self._task_complete_latched:
            return True

        if not self._compute_task_complete():
            self._completion_confirmation_count = 0
            return False

        self._completion_confirmation_count += 1
        if self._completion_confirmation_count < self.completion_confirmation_steps:
            return False

        self._task_complete_latched = True
        self._completion_status_message = (
            "Task complete.\n"
            f"Target object: {self.target_object_prim_path}\n"
            f"Target position: {self.target_position_prim_path}"
        )
        if self.status_callback is not None:
            self.status_callback(self._completion_status_message)
        self.hold_current_pose()
        return True

    def _compute_task_complete(self) -> bool:
        """Evaluate the semantic task-completion condition.

        There are no inputs; the output is a boolean.
        This exists to combine gripper-open state with object-inside-container
        geometry into one reusable predicate.
        """

        if not self._is_gripper_open():
            return False

        target_position, target_quat = get_world_pose(self.target_position_prim_path)
        return self._is_prim_inside_container(self.target_object_prim_path, target_position, target_quat)

    def _make_observation(self) -> dict[str, np.ndarray]:
        """Build the policy observation dictionary for the current frame.

        There are no inputs; the output maps observation keys to arrays.
        This exists to validate adapter initialization and delegate shape
        adaptation to PolicyObservationAdapter.
        """

        if self._observation_adapter is None:
            raise RuntimeError("Observation adapter is not initialized.")

        snapshot = self._make_observation_snapshot()
        return self._observation_adapter.build(snapshot)

    def _make_observation_snapshot(self) -> PolicyObservationSnapshot:
        """Sample robot, target object, target position, and last-action state.

        There are no inputs; the output is a PolicyObservationSnapshot.
        This exists to gather simulation data once per inference step before the
        adapter maps it into policy-specific observation keys.
        """

        joint_positions = to_numpy(self.scene.robot.get_dof_positions()).reshape(-1).astype(np.float32, copy=False)
        if joint_positions.shape != self._default_joint_positions.shape:
            raise RuntimeError(
                "Robot joint position shape does not match the authored default joint shape: "
                f"{joint_positions.shape} vs {self._default_joint_positions.shape}."
            )

        joint_velocities = np.zeros_like(joint_positions)
        get_dof_velocities = getattr(self.scene.robot, "get_dof_velocities", None)
        if callable(get_dof_velocities):
            try:
                sampled_joint_velocities = to_numpy(get_dof_velocities()).reshape(-1).astype(np.float32, copy=False)
                if sampled_joint_velocities.shape == joint_positions.shape:
                    joint_velocities = sampled_joint_velocities
            except Exception:
                pass

        ee_position, ee_quat = self._get_tcp_pose()
        target_object_position, target_object_quat = get_world_pose(self.target_object_prim_path)
        target_position, target_quat = get_world_pose(self.target_position_prim_path)

        if self._gripper_observation_dof_indices:
            gripper_positions = to_numpy(
                self.scene.robot.get_dof_positions(dof_indices=self._gripper_observation_dof_indices)
            ).reshape(-1)
            gripper_observation = (gripper_positions * self._gripper_observation_signs).astype(np.float32, copy=False)
            if gripper_observation.size == 1:
                gripper_observation = np.repeat(gripper_observation, 2)
        else:
            gripper_observation = np.zeros(2, dtype=np.float32)

        return PolicyObservationSnapshot(
            joint_pos=(joint_positions - self._default_joint_positions).astype(np.float32, copy=False),
            joint_vel=joint_velocities.astype(np.float32, copy=False),
            ee_pos=ee_position.astype(np.float32, copy=False),
            ee_quat=ee_quat.astype(np.float32, copy=False),
            gripper_pos=gripper_observation.astype(np.float32, copy=False),
            target_object_pos=target_object_position.astype(np.float32, copy=False),
            target_object_quat=target_object_quat.astype(np.float32, copy=False),
            target_position_pos=target_position.astype(np.float32, copy=False),
            target_position_quat=target_quat.astype(np.float32, copy=False),
            last_action=self._last_action.astype(np.float32, copy=False),
        )

    def _apply_action(self, action: np.ndarray) -> None:
        """Convert a policy action into robot arm and gripper targets.

        Input is a flat policy action; there is no output.
        This exists to support the known 7D relative and 8D absolute KUKA action
        formats while keeping IK and gripper command logic together.
        """

        current_joint_positions = to_numpy(
            self.scene.robot.get_dof_positions(dof_indices=self._arm_dof_indices)
        ).reshape(-1)
        jacobians = to_numpy(self.scene.robot.get_jacobian_matrices())
        jacobian = np.take(jacobians[0, self._ee_jacobian_index], self._arm_dof_indices, axis=-1).astype(np.float32)

        current_position, current_quat = self._get_tcp_pose()
        if action.shape[0] == 8:
            target_position = action[0:3].astype(np.float32)
            target_quat = normalize_quat(action[3:7].astype(np.float32))
            gripper_command = float(action[-1])
        elif action.shape[0] == 7:
            delta_position = action[0:3].astype(np.float32)
            delta_quat = axis_angle_to_quat(action[3:6].astype(np.float32))
            target_position = current_position + delta_position
            target_quat = normalize_quat(quat_multiply(delta_quat, current_quat))
            gripper_command = float(action[-1])
        else:
            raise RuntimeError(
                "The extension currently supports 7D relative or 8D absolute KUKA pose actions, "
                f"but the policy produced shape {action.shape}."
            )

        pose_error = compute_pose_error(current_position, current_quat, target_position, target_quat)
        arm_targets = compute_dls_joint_targets(jacobian, current_joint_positions, pose_error)

        self.scene.robot.set_dof_position_targets(arm_targets.reshape(1, -1), dof_indices=self._arm_dof_indices)

        if self._finger_dof_indices:
            finger_target = GRIPPER_CLOSE_COMMAND if gripper_command < 0.0 else GRIPPER_OPEN_COMMAND
            self.scene.robot.set_dof_position_targets(
                np.full((1, len(self._finger_dof_indices)), finger_target, dtype=np.float32),
                dof_indices=self._finger_dof_indices,
            )

    @property
    def policy_observation_keys(self) -> tuple[str, ...]:
        """Return the active policy's observation keys.

        There are no inputs; the output is a tuple of key strings.
        This exists for user-facing status text when a policy starts.
        """

        return self._policy_metadata.observation_keys if self._policy_metadata is not None else ()

    def _is_gripper_open(self) -> bool:
        """Check whether the gripper is open enough for task completion.

        There are no inputs; the output is a boolean.
        This exists so object-in-container success is only accepted after release.
        """

        if self._finger_dof_indices:
            finger_position = float(
                to_numpy(self.scene.robot.get_dof_positions(dof_indices=self._finger_dof_indices)).reshape(-1)[0]
            )
            return finger_position <= TASK_GRIPPER_OPEN_THRESHOLD

        if self._gripper_observation_dof_indices:
            gripper_positions = to_numpy(
                self.scene.robot.get_dof_positions(dof_indices=self._gripper_observation_dof_indices)
            ).reshape(-1)
            gripper_positions = gripper_positions * self._gripper_observation_signs
            return bool(np.all(gripper_positions <= TASK_GRIPPER_OPEN_THRESHOLD))

        return True

    def _is_prim_inside_container(
        self,
        prim_path: str,
        container_position: np.ndarray,
        container_quat: np.ndarray,
    ) -> bool:
        """Check whether a prim position lies inside the configured container zone.

        Inputs are a prim path and container pose; the output is a boolean.
        This exists to make the success bounds relative to the target container
        instead of hard-coding world axes.
        """

        prim_position, _ = get_world_pose(prim_path)
        prim_position_in_container = subtract_frame_position(container_position, container_quat, prim_position)
        return (
            abs(float(prim_position_in_container[0])) < TASK_SUCCESS_X_THRESHOLD
            and abs(float(prim_position_in_container[1])) < TASK_SUCCESS_Y_THRESHOLD
            and TASK_SUCCESS_Z_MIN_OFFSET < float(prim_position_in_container[2]) < TASK_SUCCESS_Z_MAX_OFFSET
        )

    def _get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool-center-point pose derived from the end-effector link.

        There are no inputs; the output is (position, quaternion).
        This exists because policies are trained against TCP pose, not the raw
        robot link frame.
        """

        link_position, link_quat = get_world_pose(self._ee_link_path)
        return combine_pose(
            link_position,
            link_quat,
            np.array(ROBOT_EE_OFFSET, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
