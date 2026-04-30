from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueuedPolicyConfig:
    """Serializable configuration for one policy/object execution step."""

    checkpoint_path: str
    target_object_prim_path: str
    target_position_prim_path: str
    policy_name: str
    task_type: str
    robot_type: str
    object_type: str
    norm_factor_min: float | None = None
    norm_factor_max: float | None = None

    @property
    def display_name(self) -> str:
        """Build a compact queue label from the checkpoint and target paths.

        Inputs are the instance fields; the output is a human-readable string.
        This exists so UI summaries and status messages use one consistent name.
        """

        checkpoint_label = self.policy_name or Path(self.checkpoint_path).name or self.checkpoint_path
        return (
            f"{checkpoint_label} | "
            f"{Path(self.target_object_prim_path).name} -> {Path(self.target_position_prim_path).name}"
        )
