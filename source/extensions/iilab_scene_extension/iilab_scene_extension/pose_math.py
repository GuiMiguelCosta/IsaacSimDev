from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from pxr import Usd, UsdGeom

POSE_ERROR_EPS = 1.0e-6
DLS_DAMPING = 0.01

try:
    import warp as wp
except Exception:
    wp = None


def to_numpy(data) -> np.ndarray:
    """Convert Isaac, Torch, Warp, or array-like data to a NumPy array.

    Input is an arbitrary numeric container; the output is a NumPy array.
    This exists to keep controller math independent from whichever backend Isaac
    returns for a given articulation query.
    """

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


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize a quaternion in (w, x, y, z) order.

    Input is a quaternion-like array; the output is a normalized float array.
    This exists to avoid NaNs and zero-length rotations in pose control.
    """

    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < POSE_ERROR_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion.

    Input is a quaternion in (w, x, y, z) order; the output is its conjugate.
    This exists for frame transforms and relative rotation error calculations.
    """

    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) order.

    Inputs are two quaternions; the output is their product.
    This exists to compose poses and integrate relative orientation actions.
    """

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


def quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.

    Inputs are a quaternion and 3D vector; the output is the rotated vector.
    This exists for composing TCP offsets and converting points between frames.
    """

    vector_quat = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float32)
    rotated = quat_multiply(quat_multiply(quat, vector_quat), quat_conjugate(quat))
    return rotated[1:]


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion into an axis-angle vector.

    Input is a quaternion; the output is a 3D axis-angle vector.
    This exists because damped least-squares IK consumes a 6D pose error.
    """

    quat = normalize_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    sin_half = np.linalg.norm(quat[1:])
    if sin_half < POSE_ERROR_EPS:
        return np.zeros(3, dtype=np.float32)
    axis = quat[1:] / sin_half
    angle = 2.0 * np.arctan2(sin_half, quat[0])
    return axis * angle


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector into a quaternion.

    Input is a 3D axis-angle vector; the output is a quaternion.
    This exists to interpret 7D relative policy actions as pose deltas.
    """

    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < POSE_ERROR_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis = axis_angle / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return np.array([np.cos(half_angle), *(axis * sin_half)], dtype=np.float32)


def combine_pose(
    parent_position: np.ndarray,
    parent_quat: np.ndarray,
    child_position: np.ndarray,
    child_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a child pose with a parent pose.

    Inputs are parent and child position/quaternion pairs; the output is the
    child pose in world coordinates. This exists to derive the TCP pose from the
    robot link pose plus the end-effector offset.
    """

    world_position = parent_position + quat_apply(parent_quat, child_position)
    world_quat = normalize_quat(quat_multiply(parent_quat, child_quat))
    return world_position, world_quat


def subtract_frame_position(
    frame_position: np.ndarray,
    frame_quat: np.ndarray,
    point_position: np.ndarray,
) -> np.ndarray:
    """Express a world-space point position in a frame's local coordinates.

    Inputs are frame pose and world point; the output is local xyz.
    This exists for container success checks that should follow the container's
    orientation.
    """

    return quat_apply(quat_conjugate(normalize_quat(frame_quat)), point_position - frame_position)


def compute_pose_error(
    current_position: np.ndarray,
    current_quat: np.ndarray,
    target_position: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """Build a 6D position and rotation error vector.

    Inputs are current and target poses; the output is xyz plus axis-angle error.
    This exists as the task-space error passed into damped least-squares IK.
    """

    position_error = target_position - current_position
    rotation_error = quat_multiply(normalize_quat(target_quat), quat_conjugate(normalize_quat(current_quat)))
    return np.concatenate((position_error, quat_to_axis_angle(rotation_error))).astype(np.float32)


def compute_dls_joint_targets(
    jacobian: np.ndarray,
    current_joint_positions: np.ndarray,
    pose_error: np.ndarray,
    damping: float = DLS_DAMPING,
) -> np.ndarray:
    """Solve damped least-squares IK for new joint targets.

    Inputs are the end-effector Jacobian, current joints, pose error, and
    damping; the output is joint targets. This exists to convert policy pose
    actions into articulation position commands without adding a full IK stack.
    """

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


def get_world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a prim's world-space position and quaternion.

    Input is a prim path; the output is (position, quaternion).
    This exists to provide a single validated USD transform path for policy
    observations, target checks, and TCP pose composition.
    """

    stage = stage_utils.get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transform = xform_cache.GetLocalToWorldTransform(prim)
    translation = np.array(transform.ExtractTranslation(), dtype=np.float32)
    rotation = transform.ExtractRotationQuat()
    quat = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float32)
    return translation, normalize_quat(quat)
