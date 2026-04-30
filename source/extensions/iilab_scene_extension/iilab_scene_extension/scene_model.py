from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaacsim.core.experimental.prims import Articulation

if TYPE_CHECKING:
    from .scene_objects import SceneObjectInstance


@dataclass
class IILABScene:
    """Runtime handles for the loaded IILAB scene."""

    robot: Articulation
    objects: list["SceneObjectInstance"]
    next_object_indices: dict[str, int] = field(default_factory=dict)
