from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from pxr import Usd, UsdGeom

from .constants import (
    DEFAULT_ROBOMIMIC_PYTHON,
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_OPEN_COMMAND,
    ISAACLAB_ROOT,
    POLICY_CONTROL_DT,
    ROBOT_EE_LINK_NAME,
    ROBOT_EE_OFFSET,
    TASK_GRIPPER_OPEN_THRESHOLD,
    TASK_SUCCESS_X_THRESHOLD,
    TASK_SUCCESS_Y_THRESHOLD,
    TASK_SUCCESS_Z_MAX_OFFSET,
    TASK_SUCCESS_Z_MIN_OFFSET,
)
from .scene import IILABScene, _make_default_joint_targets

try:
    import warp as wp
except Exception:
    wp = None

_POSE_ERROR_EPS = 1.0e-6
_DLS_DAMPING = 0.05
_ROBOMIMIC_IMPORT_CHECK = "import robomimic, torch"


def guess_latest_checkpoint() -> str:
    checkpoint_paths = []
    logs_root = ISAACLAB_ROOT / "logs" / "robomimic"
    if logs_root.exists():
        checkpoint_paths = sorted(logs_root.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(checkpoint_paths[0]) if checkpoint_paths else ""


def _iter_robomimic_python_candidates(configured_python: str | None) -> list[str]:
    if configured_python:
        return [configured_python]

    repo_root = Path(__file__).resolve().parents[4]
    raw_candidates = []
    for pattern in (
        "_build/*/release/python.sh",
        "_build/*/debug/python.sh",
        "_build/*/release/kit/python/bin/python3",
        "_build/*/debug/kit/python/bin/python3",
    ):
        raw_candidates.extend(str(path) for path in sorted(repo_root.glob(pattern)))

    current_executable = Path(sys.executable)
    if current_executable.name.startswith("python"):
        raw_candidates.append(str(current_executable))

    for executable_name in ("python3", "python"):
        resolved_path = shutil.which(executable_name)
        if resolved_path:
            raw_candidates.append(resolved_path)

    candidates = []
    seen = set()
    for candidate in raw_candidates:
        normalized_candidate = str(Path(candidate).expanduser())
        if normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        candidates.append(normalized_candidate)
    return candidates


@lru_cache(maxsize=8)
def _resolve_robomimic_python_executable(configured_python: str | None) -> str:
    failures = []
    for candidate in _iter_robomimic_python_candidates(configured_python):
        try:
            completed = subprocess.run(
                [candidate, "-c", _ROBOMIMIC_IMPORT_CHECK],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except FileNotFoundError:
            reason = "interpreter was not found"
        except OSError as exc:
            reason = str(exc)
        except subprocess.TimeoutExpired:
            reason = "module check timed out"
        else:
            if completed.returncode == 0:
                return candidate
            reason = completed.stderr.strip() or completed.stdout.strip() or f"returned exit code {completed.returncode}"

        failures.append(f"{candidate} ({reason})")
        if configured_python:
            break

    if configured_python:
        raise RuntimeError(
            "Configured robomimic Python is missing dependencies. "
            f"Checked {configured_python}: {failures[0] if failures else 'unknown error'}"
        )

    checked_paths = ", ".join(failures) if failures else "no candidates found"
    raise RuntimeError(
        "Could not find a Python interpreter with both torch and robomimic installed. "
        "Set IILAB_ROBOMIMIC_PYTHON to a working interpreter if needed. "
        f"Checked: {checked_paths}"
    )


def _to_numpy(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if wp is not None:
        try:
            return wp.to_numpy(data)
        except Exception:
            pass
    if hasattr(data, "numpy"):
        try:
            return data.numpy()
        except Exception:
            pass
    return np.asarray(data)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < _POSE_ERROR_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = quat_a
    bw, bx, by, bz = quat_b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def _quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_quat = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float32)
    rotated = _quat_multiply(_quat_multiply(quat, vector_quat), _quat_conjugate(quat))
    return rotated[1:]


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    sin_half = np.linalg.norm(quat[1:])
    if sin_half < _POSE_ERROR_EPS:
        return np.zeros(3, dtype=np.float32)
    axis = quat[1:] / sin_half
    angle = 2.0 * np.arctan2(sin_half, quat[0])
    return axis * angle


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < _POSE_ERROR_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis = axis_angle / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return np.array([np.cos(half_angle), *(axis * sin_half)], dtype=np.float32)


def _combine_pose(
    parent_position: np.ndarray,
    parent_quat: np.ndarray,
    child_position: np.ndarray,
    child_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_position = parent_position + _quat_apply(parent_quat, child_position)
    world_quat = _normalize_quat(_quat_multiply(parent_quat, child_quat))
    return world_position, world_quat


def _subtract_frame_position(
    frame_position: np.ndarray,
    frame_quat: np.ndarray,
    point_position: np.ndarray,
) -> np.ndarray:
    return _quat_apply(_quat_conjugate(_normalize_quat(frame_quat)), point_position - frame_position)


def _compute_pose_error(
    current_position: np.ndarray,
    current_quat: np.ndarray,
    target_position: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    position_error = target_position - current_position
    rotation_error = _quat_multiply(_normalize_quat(target_quat), _quat_conjugate(_normalize_quat(current_quat)))
    return np.concatenate((position_error, _quat_to_axis_angle(rotation_error))).astype(np.float32)


def _compute_dls_joint_targets(
    jacobian: np.ndarray,
    current_joint_positions: np.ndarray,
    pose_error: np.ndarray,
    damping: float = _DLS_DAMPING,
) -> np.ndarray:
    expected_shape = (pose_error.shape[0], current_joint_positions.shape[0])
    if jacobian.shape == expected_shape[::-1]:
        jacobian = jacobian.T
    elif jacobian.shape != expected_shape:
        raise RuntimeError(
            "Unexpected Jacobian shape for IK. "
            f"Expected {expected_shape} or {expected_shape[::-1]}, received {jacobian.shape}."
        )

    damping_matrix = (damping**2) * np.eye(jacobian.shape[0], dtype=np.float32)
    pseudo_inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
    delta_joint_positions = pseudo_inverse @ pose_error
    return (current_joint_positions + delta_joint_positions).astype(np.float32)


def _get_world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    stage = stage_utils.get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transform = xform_cache.GetLocalToWorldTransform(prim)
    translation = np.array(transform.ExtractTranslation(), dtype=np.float32)
    rotation = transform.ExtractRotationQuat()
    quat = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float32)
    return translation, _normalize_quat(quat)


def _resolve_gripper_observation_layout(dof_names: list[str]) -> tuple[list[int], np.ndarray]:
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


def _prim_path_to_observation_token(prim_path: str) -> str:
    prim_name = Path(prim_path).name or prim_path.rsplit("/", 1)[-1]
    token = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", prim_name)
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token)
    token = re.sub(r"[^0-9A-Za-z_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower() or "prim"


@dataclass(frozen=True)
class PolicyMetadata:
    observation_shapes: dict[str, tuple[int, ...]]
    action_dim: int

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return tuple(self.observation_shapes.keys())

    @classmethod
    def from_worker_message(cls, message: dict) -> "PolicyMetadata":
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
    def __init__(self, metadata: PolicyMetadata, target_object_prim_path: str, target_position_prim_path: str):
        self._metadata = metadata
        self._target_object_token = _prim_path_to_observation_token(target_object_prim_path)
        self._target_position_token = _prim_path_to_observation_token(target_position_prim_path)

    def build(self, snapshot: PolicyObservationSnapshot) -> dict[str, np.ndarray]:
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


@dataclass
class RobomimicInferenceWorker:
    checkpoint_path: str
    norm_factor_min: float | None = None
    norm_factor_max: float | None = None
    python_executable: str = DEFAULT_ROBOMIMIC_PYTHON
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _metadata: PolicyMetadata | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        worker_script_path = Path(__file__).with_name("robomimic_worker.py")
        resolved_python_executable = _resolve_robomimic_python_executable(self.python_executable)
        command = [resolved_python_executable, str(worker_script_path), "--checkpoint", self.checkpoint_path]
        if self.norm_factor_min is not None and self.norm_factor_max is not None:
            command.extend(
                ["--norm-factor-min", str(self.norm_factor_min), "--norm-factor-max", str(self.norm_factor_max)]
            )

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready_message = self._read_message()
        if ready_message.get("status") != "ready":
            raise RuntimeError(ready_message.get("message", "Robomimic worker failed to start."))
        self._metadata = PolicyMetadata.from_worker_message(ready_message)

    def reset_episode(self) -> None:
        self._request({"cmd": "reset"})

    def infer(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        response = self._request({"cmd": "infer", "obs": {key: value.tolist() for key, value in observation.items()}})
        action = np.asarray(response.get("action", []), dtype=np.float32)
        if action.ndim != 1:
            action = action.reshape(-1)
        return action

    def close(self) -> None:
        if self._process is None:
            return

        try:
            self._request({"cmd": "close"})
        except Exception:
            pass

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

        self._process = None
        self._metadata = None

    @property
    def metadata(self) -> PolicyMetadata:
        if self._metadata is None:
            raise RuntimeError("Robomimic worker metadata is not available because the worker is not running.")
        return self._metadata

    def _request(self, payload: dict) -> dict:
        process = self._require_process()
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
        return self._read_message()

    def _read_message(self) -> dict:
        process = self._require_process()
        assert process.stdout is not None
        skipped_output_lines: list[str] = []

        while True:
            line = process.stdout.readline()
            if not line:
                stderr_output = ""
                if process.poll() is not None and process.stderr is not None:
                    stderr_output = process.stderr.read().strip()

                diagnostic_output = "\n".join(skipped_output_lines).strip()
                if stderr_output and diagnostic_output:
                    raise RuntimeError(f"{stderr_output}\nWorker stdout:\n{diagnostic_output}")
                if stderr_output:
                    raise RuntimeError(stderr_output)
                if diagnostic_output:
                    raise RuntimeError(f"Robomimic worker exited unexpectedly.\nWorker stdout:\n{diagnostic_output}")
                raise RuntimeError("Robomimic worker exited unexpectedly.")

            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                message = json.loads(stripped_line)
            except json.JSONDecodeError:
                skipped_output_lines.append(stripped_line)
                skipped_output_lines = skipped_output_lines[-20:]
                continue

            if message.get("status") == "error":
                error_message = message.get("message", "Robomimic worker reported an error.")
                diagnostic_output = "\n".join(skipped_output_lines).strip()
                if diagnostic_output:
                    error_message = f"{error_message}\nWorker stdout:\n{diagnostic_output}"
                raise RuntimeError(error_message)
            return message

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("Robomimic worker is not running.")
        return self._process


@dataclass
class RobomimicPolicyController:
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
        name_to_index = {name: index for index, name in enumerate(self.scene.robot.dof_names)}
        self._arm_dof_indices = [name_to_index[f"joint_a{i + 1}"] for i in range(7) if f"joint_a{i + 1}" in name_to_index]
        self._finger_dof_indices = [name_to_index["finger_joint"]] if "finger_joint" in name_to_index else []
        self._gripper_observation_dof_indices, self._gripper_observation_signs = _resolve_gripper_observation_layout(
            list(self.scene.robot.dof_names)
        )

        if len(self._arm_dof_indices) != 7:
            raise RuntimeError("Could not resolve the seven KUKA arm joints on the articulation.")

        self._ee_link_index = self.scene.robot.link_names.index(ROBOT_EE_LINK_NAME)
        self._ee_jacobian_index = max(self._ee_link_index - 1, 0)
        self._ee_link_path = self.scene.robot.link_paths[0][self._ee_link_index]
        self._default_joint_positions = _make_default_joint_targets(self.scene.robot).reshape(-1).astype(np.float32)

        if self.target_object_prim_path == self.target_position_prim_path:
            raise RuntimeError("Target object prim and target position prim must be different.")

        _get_world_pose(self.target_object_prim_path)
        _get_world_pose(self.target_position_prim_path)

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
        self._accumulated_dt = 0.0
        if self._last_action.size > 0:
            self._last_action.fill(0.0)
        self._completion_confirmation_count = 0
        self._task_complete_latched = False
        self._completion_status_message = ""
        self.worker.reset_episode()

    def close(self) -> None:
        self.worker.close()

    def on_physics_step(self, step_dt: float) -> None:
        if self._task_complete_latched:
            return

        self._accumulated_dt += float(step_dt)
        while self._accumulated_dt + _POSE_ERROR_EPS >= self.control_dt:
            self._accumulated_dt -= self.control_dt

            if self._update_completion_state():
                break

            self._step_policy_once()
            if self._update_completion_state():
                break

    def _step_policy_once(self) -> None:
        observation = self._make_observation()
        action = self.worker.infer(observation)
        self._last_action = action.astype(np.float32)
        self._apply_action(self._last_action)

    def hold_current_pose(self) -> None:
        arm_positions = _to_numpy(self.scene.robot.get_dof_positions(dof_indices=self._arm_dof_indices)).reshape(1, -1)
        self.scene.robot.set_dof_position_targets(arm_positions.astype(np.float32), dof_indices=self._arm_dof_indices)

        if self._finger_dof_indices:
            finger_positions = _to_numpy(
                self.scene.robot.get_dof_positions(dof_indices=self._finger_dof_indices)
            ).reshape(1, -1)
            self.scene.robot.set_dof_position_targets(
                finger_positions.astype(np.float32),
                dof_indices=self._finger_dof_indices,
            )

    def is_task_complete(self) -> bool:
        if self._task_complete_latched:
            return True

        self._update_completion_state()
        if self._task_complete_latched:
            self.hold_current_pose()
            return True

        return False

    @property
    def completion_status_message(self) -> str:
        return self._completion_status_message

    def _update_completion_state(self) -> bool:
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
        if not self._is_gripper_open():
            return False

        target_position, target_quat = _get_world_pose(self.target_position_prim_path)
        return self._is_prim_inside_container(self.target_object_prim_path, target_position, target_quat)

    def _make_observation(self) -> dict[str, np.ndarray]:
        if self._observation_adapter is None:
            raise RuntimeError("Observation adapter is not initialized.")

        snapshot = self._make_observation_snapshot()
        return self._observation_adapter.build(snapshot)

    def _make_observation_snapshot(self) -> PolicyObservationSnapshot:
        joint_positions = _to_numpy(self.scene.robot.get_dof_positions()).reshape(-1).astype(np.float32, copy=False)
        if joint_positions.shape != self._default_joint_positions.shape:
            raise RuntimeError(
                "Robot joint position shape does not match the authored default joint shape: "
                f"{joint_positions.shape} vs {self._default_joint_positions.shape}."
            )

        joint_velocities = np.zeros_like(joint_positions)
        get_dof_velocities = getattr(self.scene.robot, "get_dof_velocities", None)
        if callable(get_dof_velocities):
            try:
                sampled_joint_velocities = _to_numpy(get_dof_velocities()).reshape(-1).astype(np.float32, copy=False)
                if sampled_joint_velocities.shape == joint_positions.shape:
                    joint_velocities = sampled_joint_velocities
            except Exception:
                pass

        ee_position, ee_quat = self._get_tcp_pose()
        target_object_position, target_object_quat = _get_world_pose(self.target_object_prim_path)
        target_position, target_quat = _get_world_pose(self.target_position_prim_path)

        if self._gripper_observation_dof_indices:
            gripper_positions = _to_numpy(
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
        current_joint_positions = _to_numpy(
            self.scene.robot.get_dof_positions(dof_indices=self._arm_dof_indices)
        ).reshape(-1)
        jacobians = _to_numpy(self.scene.robot.get_jacobian_matrices())
        jacobian = np.take(jacobians[0, self._ee_jacobian_index], self._arm_dof_indices, axis=-1).astype(np.float32)

        current_position, current_quat = self._get_tcp_pose()
        if action.shape[0] == 8:
            target_position = action[0:3].astype(np.float32)
            target_quat = _normalize_quat(action[3:7].astype(np.float32))
            gripper_command = float(action[-1])
        elif action.shape[0] == 7:
            delta_position = action[0:3].astype(np.float32)
            delta_quat = _axis_angle_to_quat(action[3:6].astype(np.float32))
            target_position = current_position + delta_position
            target_quat = _normalize_quat(_quat_multiply(delta_quat, current_quat))
            gripper_command = float(action[-1])
        else:
            raise RuntimeError(
                "The extension currently supports 7D relative or 8D absolute KUKA pose actions, "
                f"but the policy produced shape {action.shape}."
            )

        pose_error = _compute_pose_error(current_position, current_quat, target_position, target_quat)
        arm_targets = _compute_dls_joint_targets(jacobian, current_joint_positions, pose_error)

        self.scene.robot.set_dof_position_targets(arm_targets.reshape(1, -1), dof_indices=self._arm_dof_indices)

        if self._finger_dof_indices:
            finger_target = GRIPPER_CLOSE_COMMAND if gripper_command < 0.0 else GRIPPER_OPEN_COMMAND
            self.scene.robot.set_dof_position_targets(
                np.full((1, len(self._finger_dof_indices)), finger_target, dtype=np.float32),
                dof_indices=self._finger_dof_indices,
            )

    @property
    def policy_observation_keys(self) -> tuple[str, ...]:
        return self._policy_metadata.observation_keys if self._policy_metadata is not None else ()

    def _is_gripper_open(self) -> bool:
        if self._finger_dof_indices:
            finger_position = float(
                _to_numpy(self.scene.robot.get_dof_positions(dof_indices=self._finger_dof_indices)).reshape(-1)[0]
            )
            return finger_position <= TASK_GRIPPER_OPEN_THRESHOLD

        if self._gripper_observation_dof_indices:
            gripper_positions = _to_numpy(
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
        prim_position, _ = _get_world_pose(prim_path)
        prim_position_in_container = _subtract_frame_position(container_position, container_quat, prim_position)
        return (
            abs(float(prim_position_in_container[0])) < TASK_SUCCESS_X_THRESHOLD
            and abs(float(prim_position_in_container[1])) < TASK_SUCCESS_Y_THRESHOLD
            and TASK_SUCCESS_Z_MIN_OFFSET < float(prim_position_in_container[2]) < TASK_SUCCESS_Z_MAX_OFFSET
        )

    def _get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        link_position, link_quat = _get_world_pose(self._ee_link_path)
        return _combine_pose(
            link_position,
            link_quat,
            np.array(ROBOT_EE_OFFSET, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
