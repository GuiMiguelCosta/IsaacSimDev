from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


def resolve_gripper_observation_layout(dof_names: list[str]) -> tuple[list[int], np.ndarray]:
    """Resolve gripper DOFs and signs for policy observations.

    Input is the articulation DOF name list; the output is indices and signs.
    This exists to support both direct finger joints and two-sided inner-finger
    layouts while presenting a stable observation to robomimic policies.
    """

    inner_finger_entries = [
        (name, index) for index, name in enumerate(dof_names) if re.fullmatch(r".*_inner_finger_joint", name)
    ]
    if len(inner_finger_entries) == 2:
        left_entries = [entry for entry in inner_finger_entries if "left" in entry[0]]
        right_entries = [entry for entry in inner_finger_entries if "right" in entry[0]]
        if len(left_entries) == 1 and len(right_entries) == 1:
            ordered_entries = [left_entries[0], right_entries[0]]
        else:
            ordered_entries = inner_finger_entries
        return [entry[1] for entry in ordered_entries], np.array([1.0, -1.0], dtype=np.float32)

    name_to_index = {name: index for index, name in enumerate(dof_names)}
    if "finger_joint" in name_to_index:
        return [name_to_index["finger_joint"]], np.array([1.0], dtype=np.float32)

    return [], np.zeros(0, dtype=np.float32)


def prim_path_to_observation_token(prim_path: str) -> str:
    """Convert a prim path into a snake-case observation token.

    Input is a prim path; the output is a lowercase token.
    This exists so policies can use object-specific keys such as
    bottom_housing_pos without hard-coding every prim name.
    """

    prim_name = Path(prim_path).name or prim_path.rsplit("/", 1)[-1]
    token = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", prim_name)
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token)
    token = re.sub(r"[^0-9A-Za-z_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower() or "prim"


@dataclass(frozen=True)
class PolicyMetadata:
    """Observation/action schema reported by the robomimic worker."""

    observation_shapes: dict[str, tuple[int, ...]]
    action_dim: int

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the policy observation keys in reported order.

        There are no inputs; the output is a tuple of key strings.
        This exists for status messages and adapter iteration.
        """

        return tuple(self.observation_shapes.keys())

    @classmethod
    def from_worker_message(cls, message: dict) -> "PolicyMetadata":
        """Build metadata from the robomimic worker's ready message.

        Input is a decoded worker message; the output is PolicyMetadata.
        This exists to validate subprocess startup before the controller starts
        sending observations.
        """

        raw_shapes = message.get("obs_shapes", {})
        if not isinstance(raw_shapes, dict) or not raw_shapes:
            raise RuntimeError("Robomimic worker did not report any observation metadata.")

        action_dim = int(message.get("ac_dim", 0))
        if action_dim <= 0:
            raise RuntimeError("Robomimic worker reported an invalid action dimension.")

        observation_shapes = {
            str(key): tuple(int(dim) for dim in np.asarray(shape, dtype=np.int64).reshape(-1).tolist())
            for key, shape in raw_shapes.items()
        }
        return cls(observation_shapes=observation_shapes, action_dim=action_dim)


@dataclass(frozen=True)
class PolicyObservationSnapshot:
    """Raw simulation state sampled for one policy inference step."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper_pos: np.ndarray
    target_object_pos: np.ndarray
    target_object_quat: np.ndarray
    target_position_pos: np.ndarray
    target_position_quat: np.ndarray
    last_action: np.ndarray


class PolicyObservationAdapter:
    """Adapt simulation snapshots into policy-specific observation dictionaries."""

    def __init__(self, metadata: PolicyMetadata, target_object_prim_path: str, target_position_prim_path: str):
        """Create an adapter for a policy's declared observation schema.

        Inputs are policy metadata and target prim paths; there is no output.
        This exists so learned policies can use multiple common observation key
        conventions without changing controller code.
        """

        self._metadata = metadata
        self._target_object_token = prim_path_to_observation_token(target_object_prim_path)
        self._target_position_token = prim_path_to_observation_token(target_position_prim_path)

    def build(self, snapshot: PolicyObservationSnapshot) -> dict[str, np.ndarray]:
        """Build a robomimic observation dict from a simulation snapshot.

        Input is a PolicyObservationSnapshot; the output maps keys to arrays.
        This exists to enforce policy-declared shapes before crossing the
        subprocess boundary.
        """

        observation = {}
        for key, expected_shape in self._metadata.observation_shapes.items():
            value = self._resolve_observation(key, expected_shape, snapshot)
            if value is None:
                raise RuntimeError(
                    "Could not adapt simulation observations to the policy input "
                    f"'{key}' with shape {expected_shape}."
                )
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            if tuple(value.shape) != tuple(expected_shape):
                raise RuntimeError(
                    f"Observation '{key}' expected shape {expected_shape}, "
                    f"but the adapter produced shape {tuple(value.shape)}."
                )
            observation[key] = value
        return observation

    def _resolve_observation(
        self,
        key: str,
        expected_shape: tuple[int, ...],
        snapshot: PolicyObservationSnapshot,
    ) -> np.ndarray | None:
        """Resolve a single policy observation key from a snapshot.

        Inputs are the key, expected shape, and snapshot; the output is an array
        or None. This exists to keep all key aliases and packed-object layouts in
        one place.
        """

        ee_pose = np.concatenate((snapshot.ee_pos, snapshot.ee_quat), axis=0).astype(np.float32, copy=False)
        gripper_to_object = snapshot.target_object_pos - snapshot.ee_pos
        gripper_to_target = snapshot.target_position_pos - snapshot.ee_pos
        object_to_target = snapshot.target_object_pos - snapshot.target_position_pos

        direct_features = {
            "joint_pos": snapshot.joint_pos,
            "joint_vel": snapshot.joint_vel,
            "ee_pose": ee_pose,
            "eef_pose": ee_pose,
            "ee_pos": snapshot.ee_pos,
            "eef_pos": snapshot.ee_pos,
            "ee_quat": snapshot.ee_quat,
            "eef_quat": snapshot.ee_quat,
            "gripper_pos": snapshot.gripper_pos,
            "actions": snapshot.last_action,
            "object_pos": snapshot.target_object_pos,
            "object_quat": snapshot.target_object_quat,
            "target_object_pos": snapshot.target_object_pos,
            "target_object_quat": snapshot.target_object_quat,
            "target_pos": snapshot.target_position_pos,
            "target_quat": snapshot.target_position_quat,
            "target_position_pos": snapshot.target_position_pos,
            "target_position_quat": snapshot.target_position_quat,
            "gripper_to_object": gripper_to_object,
            "gripper_to_target": gripper_to_target,
            "gripper_to_target_position": gripper_to_target,
            "object_to_target": object_to_target,
            "object_to_target_position": object_to_target,
        }
        if key in direct_features:
            return direct_features[key]

        if key == "object":
            packed_object = np.concatenate(
                (
                    snapshot.target_object_pos,
                    snapshot.target_object_quat,
                    gripper_to_object,
                    snapshot.target_position_pos,
                    snapshot.target_position_quat,
                    gripper_to_target,
                    object_to_target,
                ),
                axis=0,
            ).astype(np.float32, copy=False)
            if tuple(packed_object.shape) == tuple(expected_shape):
                return packed_object
            return None

        if key.endswith("_pos"):
            role = self._classify_pose_role(key[:-4])
            if role == "target_position":
                return snapshot.target_position_pos
            if role == "target_object":
                return snapshot.target_object_pos
            return None

        if key.endswith("_quat"):
            role = self._classify_pose_role(key[:-5])
            if role == "target_position":
                return snapshot.target_position_quat
            if role == "target_object":
                return snapshot.target_object_quat
            return None

        return None

    def _classify_pose_role(self, prefix: str) -> str | None:
        """Classify an observation key prefix as object or target-position data.

        Input is a key prefix; the output is a role string or None.
        This exists to make object-specific keys and generic goal keys converge
        on the same snapshot fields.
        """

        normalized_prefix = prefix.strip().lower()

        target_object_prefixes = {
            self._target_object_token,
            f"target_{self._target_object_token}",
            "object",
            "target_object",
        }
        target_position_prefixes = {
            self._target_position_token,
            f"target_{self._target_position_token}",
            "target",
            "goal",
            "target_position",
            "goal_position",
        }

        if normalized_prefix in target_object_prefixes:
            return "target_object"
        if normalized_prefix in target_position_prefixes:
            return "target_position"
        if normalized_prefix.startswith("target_") or normalized_prefix.startswith("goal_"):
            return "target_position"
        return "target_object"
