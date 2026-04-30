from __future__ import annotations

from pathlib import Path

from .constants import (
    CONTAINER_PRIM_PATH,
    DEFAULT_ROBOMIMIC_NORM_FACTOR_MAX,
    DEFAULT_ROBOMIMIC_NORM_FACTOR_MIN,
)
from .policy_config import QueuedPolicyConfig
from .policy_discovery import discover_policies
from .scene_model import IILABScene


def parse_optional_float(value: str, label: str) -> float | None:
    """Parse an optional UI/env float field.

    Inputs are the raw string and user-facing label; the output is a float or
    None. This exists so default normalization factors fail with clear messages.
    """

    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def default_norm_factors() -> tuple[float | None, float | None]:
    """Resolve optional robomimic action normalization factors.

    There are no inputs; the output is (min, max), each float or None.
    This exists to validate the pair once before constructing many policy configs.
    """

    norm_factor_min = parse_optional_float(DEFAULT_ROBOMIMIC_NORM_FACTOR_MIN, "Norm Min")
    norm_factor_max = parse_optional_float(DEFAULT_ROBOMIMIC_NORM_FACTOR_MAX, "Norm Max")
    if (norm_factor_min is None) != (norm_factor_max is None):
        raise ValueError(
            "Set both IILAB_ROBOMIMIC_NORM_FACTOR_MIN and IILAB_ROBOMIMIC_NORM_FACTOR_MAX or leave both empty."
        )
    return norm_factor_min, norm_factor_max


def available_policy_display_name(policy_config: QueuedPolicyConfig) -> str:
    """Build the visible label for a compatible policy/object pair.

    Input is a queued policy config; the output is a UI display string.
    This exists to keep available-policy combo boxes and summaries in sync.
    """

    object_label = Path(policy_config.target_object_prim_path).name
    task_label = policy_config.task_type or "task"
    robot_label = policy_config.robot_type or "robot"
    return f"{policy_config.policy_name} [{task_label}/{robot_label}] -> {object_label}"


def build_available_policy_configs(
    scene: IILABScene | None,
    policies_root: Path | None,
) -> tuple[list[QueuedPolicyConfig], str]:
    """Create policy/object pairings compatible with the current scene.

    Inputs are the loaded scene and policies root; the output is a config list
    plus a summary string. This exists to keep scan logic independent from Omni
    UI widgets and policy queue mutation.
    """

    if scene is None:
        return [], "Load the scene before scanning compatible policies."
    if policies_root is None:
        return [], "Set the policies folder path."
    if not policies_root.exists() or not policies_root.is_dir():
        return [], f"Policies folder not found: {policies_root}"

    norm_factor_min, norm_factor_max = default_norm_factors()
    discovered_policies = discover_policies(policies_root)
    if not discovered_policies:
        return [], f"No checkpoint files found under {policies_root} for known object types."

    scene_objects_by_key = {}
    for scene_object in scene.objects:
        scene_objects_by_key.setdefault(scene_object.key, []).append(scene_object)

    available_policy_configs: list[QueuedPolicyConfig] = []
    for discovered_policy in discovered_policies:
        for scene_object in scene_objects_by_key.get(discovered_policy.object_key, []):
            available_policy_configs.append(
                QueuedPolicyConfig(
                    checkpoint_path=discovered_policy.checkpoint_path,
                    target_object_prim_path=scene_object.prim_path,
                    target_position_prim_path=CONTAINER_PRIM_PATH,
                    policy_name=discovered_policy.policy_name,
                    task_type=discovered_policy.task_type,
                    robot_type=discovered_policy.robot_type,
                    object_type=discovered_policy.object_key,
                    norm_factor_min=norm_factor_min,
                    norm_factor_max=norm_factor_max,
                )
            )

    if not available_policy_configs:
        object_names = ", ".join(Path(scene_object.prim_path).name for scene_object in scene.objects)
        return [], f"No discovered policies match the objects on the table: {object_names or 'none'}."

    return available_policy_configs, f"Found {len(available_policy_configs)} compatible policy/object pairing(s)."
