from __future__ import annotations

from dataclasses import dataclass
import math
import re

import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import Sdf

from .constants import (
    AXIS_USD_PATH,
    BOTTOM_HOUSING_USD_PATH,
    CUBE_1_POSITION,
    CUBE_1_ROTATION,
    CUBE_2_POSITION,
    CUBE_2_ROTATION,
    CUBE_3_POSITION,
    CUBE_3_ROTATION,
    TOP_BEARING_USD_PATH,
)
from .scene_physics import add_reference, configure_dynamic_piece


@dataclass(frozen=True)
class SceneObjectSpec:
    """Static asset and reset configuration for one table object type."""

    key: str
    display_name: str
    prim_prefix: str
    usd_path: str
    default_position: tuple[float, float, float]
    default_orientation: tuple[float, float, float, float]
    random_roll: float = 0.0
    fallback_collision_approximation: str = "convexDecomposition"


@dataclass(frozen=True)
class SceneObjectInstance:
    """Runtime identity of one dynamic table object in the stage."""

    key: str
    prim_path: str


SCENE_OBJECT_SPECS: dict[str, SceneObjectSpec] = {
    "bottom_housing": SceneObjectSpec(
        key="bottom_housing",
        display_name="Bottom Housing",
        prim_prefix="BottomHousing",
        usd_path=BOTTOM_HOUSING_USD_PATH,
        default_position=CUBE_1_POSITION,
        default_orientation=CUBE_1_ROTATION,
        fallback_collision_approximation="convexDecomposition",
    ),
    "axis": SceneObjectSpec(
        key="axis",
        display_name="Axis",
        prim_prefix="Axis",
        usd_path=AXIS_USD_PATH,
        default_position=CUBE_3_POSITION,
        default_orientation=CUBE_3_ROTATION,
        random_roll=math.pi / 2.0,
        fallback_collision_approximation="convexHull",
    ),
    "top_bearing": SceneObjectSpec(
        key="top_bearing",
        display_name="Top Bearing",
        prim_prefix="TopBearing",
        usd_path=TOP_BEARING_USD_PATH,
        default_position=CUBE_2_POSITION,
        default_orientation=CUBE_2_ROTATION,
        fallback_collision_approximation="convexDecomposition",
    ),
}
SCENE_OBJECT_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (spec.key, spec.display_name) for spec in SCENE_OBJECT_SPECS.values()
)
DEFAULT_OBJECTS: tuple[str, ...] = ("bottom_housing", "top_bearing", "axis")


def get_scene_object_spec(object_key: str) -> SceneObjectSpec:
    """Return the validated spec for a table object type.

    Input is an object key; the output is the matching SceneObjectSpec.
    This exists to centralize error messages for unknown object types.
    """

    try:
        return SCENE_OBJECT_SPECS[object_key]
    except KeyError as exc:
        valid_names = ", ".join(spec.display_name for spec in SCENE_OBJECT_SPECS.values())
        raise ValueError(f"Unknown scene object '{object_key}'. Available objects: {valid_names}.") from exc


def add_scene_object_reference(object_key: str, prim_path: str) -> SceneObjectInstance:
    """Reference and configure one dynamic scene object.

    Inputs are the object key and target prim path; the output is the object
    instance metadata. This exists so build, add, and import workflows share the
    same asset and physics setup.
    """

    spec = get_scene_object_spec(object_key)
    add_reference(
        prim_path,
        spec.usd_path,
        translation=spec.default_position,
        orientation=spec.default_orientation,
    )
    configure_dynamic_piece(
        prim_path,
        fallback_collision_approximation=spec.fallback_collision_approximation,
    )
    return SceneObjectInstance(key=object_key, prim_path=prim_path)


def make_default_object_prim_path(object_key: str) -> str:
    """Return the default prim path for the first instance of an object type.

    Input is an object key; the output is an absolute prim path.
    This exists to keep initial scene object naming consistent with later added
    instances.
    """

    spec = get_scene_object_spec(object_key)
    return f"/World/{spec.prim_prefix}_1"


def make_object_prim_path(scene, object_key: str) -> str:
    """Choose the next unused prim path for a new scene object.

    Inputs are the scene and object key; the output is an absolute prim path.
    This exists to avoid collisions when users add multiple objects of the same
    type or import queues with custom prim paths.
    """

    spec = get_scene_object_spec(object_key)
    stage = stage_utils.get_current_stage(backend="usd")
    next_object_index = scene.next_object_indices.get(object_key, 1)
    while True:
        prim_path = f"/World/{spec.prim_prefix}_{next_object_index}"
        next_object_index += 1
        scene.next_object_indices[object_key] = next_object_index
        if not stage.GetPrimAtPath(prim_path).IsValid():
            return prim_path


def find_next_object_indices(objects: list[SceneObjectInstance]) -> dict[str, int]:
    """Compute next per-object numeric suffixes from existing objects.

    Input is the current object list; the output maps object keys to next suffix.
    This exists so imported/custom prim paths do not cause future add operations
    to overwrite or reuse an occupied name.
    """

    next_object_indices = {object_key: 1 for object_key in SCENE_OBJECT_SPECS}
    for scene_object in objects:
        spec = get_scene_object_spec(scene_object.key)
        match = re.fullmatch(rf"/World/{re.escape(spec.prim_prefix)}_(\d+)", scene_object.prim_path)
        if match:
            next_object_indices[scene_object.key] = max(
                next_object_indices.get(scene_object.key, 1),
                int(match.group(1)) + 1,
            )
    return next_object_indices


def format_scene_object(scene_object: SceneObjectInstance) -> str:
    """Format a scene object for UI summaries.

    Input is an object instance; the output is a display string.
    This exists so object add/remove controls use the same labels.
    """

    spec = get_scene_object_spec(scene_object.key)
    return f"{spec.display_name} ({scene_object.prim_path})"


def add_scene_object(scene, object_key: str) -> SceneObjectInstance:
    """Add one object to a loaded scene and reset the episode layout.

    Inputs are the scene and object key; the output is the new object instance.
    This exists as the high-level mutation entry point for the UI add button.
    """

    from .scene_reset import reset_scene

    scene_object = add_scene_object_reference(object_key, make_object_prim_path(scene, object_key))
    scene.objects.append(scene_object)
    reset_scene(scene)
    return scene_object


def remove_scene_object(scene, prim_path: str) -> SceneObjectInstance:
    """Remove one object from a loaded scene and reset the episode layout.

    Inputs are the scene and prim path; the output is the removed object.
    This exists as the high-level mutation entry point for the UI remove button.
    """

    from .scene_reset import reset_scene

    scene_object = next((item for item in scene.objects if item.prim_path == prim_path), None)
    if scene_object is None:
        raise ValueError(f"Scene object not found: {prim_path}")

    stage = stage_utils.get_current_stage(backend="usd")
    stage.RemovePrim(Sdf.Path(prim_path))
    scene.objects.remove(scene_object)
    reset_scene(scene)
    return scene_object


def set_scene_objects(scene, object_specs: list[tuple[str, str]]) -> list[SceneObjectInstance]:
    """Replace all dynamic objects in a scene from validated specs.

    Inputs are the scene and (object_key, prim_path) pairs; the output is the new
    object list. This exists for queue import, where the stage must match the
    serialized policy targets before execution.
    """

    from .scene_reset import reset_scene

    stage = stage_utils.get_current_stage(backend="usd")
    requested_objects: list[tuple[str, str]] = []
    seen_prim_paths: set[str] = set()
    current_object_paths = {scene_object.prim_path for scene_object in scene.objects}

    for object_key, prim_path in object_specs:
        get_scene_object_spec(object_key)
        if not Sdf.Path.IsValidPathString(prim_path) or not prim_path.startswith("/"):
            raise ValueError(f"Object prim path must be an absolute valid USD path: {prim_path}")
        if prim_path in seen_prim_paths:
            raise ValueError(f"Duplicate object prim path in scene object config: {prim_path}")
        if stage.GetPrimAtPath(prim_path).IsValid() and prim_path not in current_object_paths:
            raise ValueError(f"Cannot create scene object because a prim already exists at: {prim_path}")
        seen_prim_paths.add(prim_path)
        requested_objects.append((object_key, prim_path))

    for scene_object in list(scene.objects):
        if stage.GetPrimAtPath(scene_object.prim_path).IsValid():
            stage.RemovePrim(Sdf.Path(scene_object.prim_path))

    scene.objects.clear()
    for object_key, prim_path in requested_objects:
        scene.objects.append(add_scene_object_reference(object_key, prim_path))

    scene.next_object_indices = find_next_object_indices(scene.objects)
    reset_scene(scene)
    return scene.objects
