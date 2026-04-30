from __future__ import annotations

from pathlib import Path

import isaacsim.core.experimental.utils.stage as stage_utils
from omni.physx.scripts import utils as physx_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .constants import IDENTITY_ROTATION

PHYSICS_TIMESTEPS_PER_SECOND = 100.0
PHYSICS_BOUNCE_THRESHOLD_VELOCITY = 0.01
PHYSICS_FRICTION_CORRELATION_DISTANCE = 0.00625
DYNAMIC_PIECE_MASS = 0.05
DYNAMIC_PIECE_SOLVER_POSITION_ITERATIONS = 16
DYNAMIC_PIECE_SOLVER_VELOCITY_ITERATIONS = 1
DYNAMIC_PIECE_MAX_ANGULAR_VELOCITY = 1000.0
DYNAMIC_PIECE_MAX_LINEAR_VELOCITY = 1000.0
DYNAMIC_PIECE_MAX_DEPENETRATION_VELOCITY = 5.0


def create_physics_scene() -> None:
    """Author the USD physics scene with the extension's solver settings.

    There are no inputs or outputs. This exists so stage construction has one
    dedicated place for global gravity and PhysX timestep configuration.
    """

    stage = stage_utils.get_current_stage(backend="usd")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())

    time_steps_attr = physx_scene_api.GetTimeStepsPerSecondAttr()
    if time_steps_attr:
        time_steps_attr.Set(PHYSICS_TIMESTEPS_PER_SECOND)
    else:
        physx_scene_api.CreateTimeStepsPerSecondAttr(PHYSICS_TIMESTEPS_PER_SECOND)

    bounce_threshold_attr = physx_scene_api.GetBounceThresholdAttr()
    if bounce_threshold_attr:
        bounce_threshold_attr.Set(PHYSICS_BOUNCE_THRESHOLD_VELOCITY)
    else:
        physx_scene_api.CreateBounceThresholdAttr(PHYSICS_BOUNCE_THRESHOLD_VELOCITY)

    friction_correlation_attr = physx_scene_api.GetFrictionCorrelationDistanceAttr()
    if friction_correlation_attr:
        friction_correlation_attr.Set(PHYSICS_FRICTION_CORRELATION_DISTANCE)
    else:
        physx_scene_api.CreateFrictionCorrelationDistanceAttr(PHYSICS_FRICTION_CORRELATION_DISTANCE)


def require_existing_asset(asset_path: str) -> str:
    """Validate a USD asset path before referencing it into the stage.

    Input is an asset path string; the output is the expanded string path.
    This exists to fail early with a clear file error instead of a silent USD
    reference that later produces an invalid prim.
    """

    path = Path(asset_path)
    if not path.exists():
        raise FileNotFoundError(f"USD asset not found: {path}")
    return str(path)


def get_prim(prim_path: str) -> Usd.Prim:
    """Fetch a valid USD prim from the current stage.

    Input is an absolute prim path; the output is a valid Usd.Prim.
    This exists to make missing-prim errors explicit at call sites that require
    authored scene content.
    """

    stage = stage_utils.get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    return prim


def set_xform(
    prim_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = IDENTITY_ROTATION,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Set translate, orient, and scale ops on a prim in a stable order.

    Inputs are a prim path and transform components; there is no output.
    This exists to normalize referenced asset transforms without accumulating
    duplicate xform ops across resets or imports.
    """

    prim = get_prim(prim_path)
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


def add_reference(
    prim_path: str,
    usd_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = IDENTITY_ROTATION,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Reference a USD asset and immediately author its transform.

    Inputs are target prim path, USD file path, and transform components; there
    is no output. This exists to make scene builder code express intent instead
    of repeating reference and transform boilerplate.
    """

    stage_utils.add_reference_to_stage(usd_path=require_existing_asset(usd_path), path=prim_path)
    set_xform(prim_path, translation=translation, orientation=orientation, scale=scale)


def set_schema_attr(get_attr, create_attr, value) -> None:
    """Set a USD schema attribute, creating it when missing.

    Inputs are schema getter/creator callables and the value; there is no output.
    This exists to avoid repeated get-or-create blocks across PhysX APIs.
    """

    if value is None:
        return
    attr = get_attr()
    if attr:
        attr.Set(value)
    else:
        create_attr(value)


def configure_rigid_body_prim(
    prim: Usd.Prim,
    *,
    rigid_body_enabled: bool,
    kinematic: bool,
    disable_gravity: bool,
    solver_position_iteration_count: int | None = None,
    solver_velocity_iteration_count: int | None = None,
    max_angular_velocity: float | None = None,
    max_linear_velocity: float | None = None,
    max_depenetration_velocity: float | None = None,
) -> None:
    """Apply rigid-body and PhysX body attributes to a prim.

    Inputs are a prim and body settings; there is no output. This exists so
    static assets, dynamic pieces, and fallback rigid bodies share one schema
    authoring path.
    """

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(prim)
    set_schema_attr(rigid_body_api.GetRigidBodyEnabledAttr, rigid_body_api.CreateRigidBodyEnabledAttr, rigid_body_enabled)
    set_schema_attr(rigid_body_api.GetKinematicEnabledAttr, rigid_body_api.CreateKinematicEnabledAttr, kinematic)

    physx_rigid_body_api = (
        PhysxSchema.PhysxRigidBodyAPI(prim)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    )
    set_schema_attr(physx_rigid_body_api.GetDisableGravityAttr, physx_rigid_body_api.CreateDisableGravityAttr, disable_gravity)
    set_schema_attr(
        physx_rigid_body_api.GetSolverPositionIterationCountAttr,
        physx_rigid_body_api.CreateSolverPositionIterationCountAttr,
        solver_position_iteration_count,
    )
    set_schema_attr(
        physx_rigid_body_api.GetSolverVelocityIterationCountAttr,
        physx_rigid_body_api.CreateSolverVelocityIterationCountAttr,
        solver_velocity_iteration_count,
    )
    set_schema_attr(
        physx_rigid_body_api.GetMaxAngularVelocityAttr,
        physx_rigid_body_api.CreateMaxAngularVelocityAttr,
        max_angular_velocity,
    )
    set_schema_attr(
        physx_rigid_body_api.GetMaxLinearVelocityAttr,
        physx_rigid_body_api.CreateMaxLinearVelocityAttr,
        max_linear_velocity,
    )
    set_schema_attr(
        physx_rigid_body_api.GetMaxDepenetrationVelocityAttr,
        physx_rigid_body_api.CreateMaxDepenetrationVelocityAttr,
        max_depenetration_velocity,
    )


def subtree_rigid_body_prims(root_prim: Usd.Prim) -> list[Usd.Prim]:
    """Collect rigid-body prims under a subtree root.

    Input is a root prim; the output is a list of prims with RigidBodyAPI.
    This exists because imported assets may author rigid bodies on child meshes
    instead of their root prim.
    """

    return [child_prim for child_prim in Usd.PrimRange(root_prim) if child_prim.HasAPI(UsdPhysics.RigidBodyAPI)]


def ensure_rigid_body(prim_path: str, *, kinematic: bool, disable_gravity: bool) -> None:
    """Ensure a prim has a rigid body with the requested mode.

    Inputs are a prim path and body flags; there is no output.
    This exists as a fallback for assets that do not already author any rigid
    body prims in their subtree.
    """

    prim = get_prim(prim_path)
    configure_rigid_body_prim(prim, rigid_body_enabled=True, kinematic=kinematic, disable_gravity=disable_gravity)


def set_mass(prim_path: str, mass: float) -> None:
    """Author mass on a prim, creating MassAPI when needed.

    Inputs are a prim path and mass value; there is no output.
    This exists to keep dynamic table objects light and predictable in PhysX.
    """

    prim = get_prim(prim_path)
    mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr:
        mass_attr.Set(mass)
    else:
        mass_api.CreateMassAttr(mass)


def configure_rigid_body_subtree(
    prim_path: str,
    *,
    rigid_body_enabled: bool,
    kinematic: bool,
    disable_gravity: bool,
    solver_position_iteration_count: int | None = None,
    solver_velocity_iteration_count: int | None = None,
    max_angular_velocity: float | None = None,
    max_linear_velocity: float | None = None,
    max_depenetration_velocity: float | None = None,
) -> list[Usd.Prim]:
    """Apply rigid-body settings to authored bodies under a prim.

    Inputs are a prim path and body settings; the output is the configured body
    prims. This exists to preserve asset-authored body topology while still
    providing a root fallback for simpler USD assets.
    """

    root_prim = get_prim(prim_path)
    rigid_body_prims = subtree_rigid_body_prims(root_prim)
    if not rigid_body_prims:
        ensure_rigid_body(prim_path, kinematic=kinematic, disable_gravity=disable_gravity)
        rigid_body_prims = [root_prim]

    for rigid_body_prim in rigid_body_prims:
        configure_rigid_body_prim(
            rigid_body_prim,
            rigid_body_enabled=rigid_body_enabled,
            kinematic=kinematic,
            disable_gravity=disable_gravity,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            max_angular_velocity=max_angular_velocity,
            max_linear_velocity=max_linear_velocity,
            max_depenetration_velocity=max_depenetration_velocity,
        )

    return rigid_body_prims


def subtree_has_authored_collision(prim: Usd.Prim) -> bool:
    """Check whether a subtree already contains collision APIs.

    Input is a root prim; the output is True when collision is authored.
    This exists to avoid overwriting authored collision setups with fallback
    collider generation.
    """

    for child_prim in Usd.PrimRange(prim):
        if child_prim.HasAPI(UsdPhysics.CollisionAPI) or child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            return True
        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            return True
    return False


def configure_static_scene_asset(
    prim_path: str,
    *,
    fallback_collision_approximation: str | None = "none",
) -> None:
    """Configure a referenced table or fixture as a kinematic scene asset.

    Inputs are the prim path and optional fallback collider mode; there is no
    output. This exists so static scene geometry participates in collision
    without being moved by gravity or policy contacts.
    """

    prim = get_prim(prim_path)
    if fallback_collision_approximation and not subtree_has_authored_collision(prim):
        physx_utils.setColliderSubtree(prim, approximationShape=fallback_collision_approximation)
    configure_rigid_body_subtree(prim_path, rigid_body_enabled=True, kinematic=True, disable_gravity=True)


def configure_static_scene_asset_with_authored_collision_fallback(prim_path: str) -> None:
    """Configure a static asset that should keep authored collision when present.

    Input is a prim path; there is no output. This exists for container-like
    assets where authored collision is preferable and mesh fallback should be
    conservative.
    """

    configure_static_scene_asset(prim_path, fallback_collision_approximation="none")


def configure_dynamic_piece(
    prim_path: str,
    *,
    mass: float = DYNAMIC_PIECE_MASS,
    fallback_collision_approximation: str | None = None,
) -> None:
    """Configure a table object as a dynamic manipulation target.

    Inputs are the prim path, mass, and optional fallback collider mode; there is
    no output. This exists to tune solver iterations, velocities, and mass in one
    place for all movable pieces.
    """

    prim = get_prim(prim_path)
    if fallback_collision_approximation and not subtree_has_authored_collision(prim):
        physx_utils.setColliderSubtree(prim, approximationShape=fallback_collision_approximation)
    rigid_body_prims = configure_rigid_body_subtree(
        prim_path,
        rigid_body_enabled=True,
        kinematic=False,
        disable_gravity=False,
        solver_position_iteration_count=DYNAMIC_PIECE_SOLVER_POSITION_ITERATIONS,
        solver_velocity_iteration_count=DYNAMIC_PIECE_SOLVER_VELOCITY_ITERATIONS,
        max_angular_velocity=DYNAMIC_PIECE_MAX_ANGULAR_VELOCITY,
        max_linear_velocity=DYNAMIC_PIECE_MAX_LINEAR_VELOCITY,
        max_depenetration_velocity=DYNAMIC_PIECE_MAX_DEPENETRATION_VELOCITY,
    )
    if rigid_body_prims:
        set_mass(rigid_body_prims[0].GetPath().pathString, mass)


def zero_rigid_body_velocity(prim_path: str) -> None:
    """Clear linear and angular velocity on every body under a prim.

    Input is a prim path; there is no output. This exists so randomized resets
    start from rest instead of preserving velocities from the previous episode.
    """

    prim = get_prim(prim_path)
    rigid_body_prims = subtree_rigid_body_prims(prim)
    if not rigid_body_prims:
        return

    linear_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
    angular_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
    for rigid_body_prim in rigid_body_prims:
        rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)

        linear_velocity_attr = rigid_body_api.GetVelocityAttr()
        if linear_velocity_attr:
            linear_velocity_attr.Set(linear_velocity)
        else:
            rigid_body_api.CreateVelocityAttr(linear_velocity)

        angular_velocity_attr = rigid_body_api.GetAngularVelocityAttr()
        if angular_velocity_attr:
            angular_velocity_attr.Set(angular_velocity)
        else:
            rigid_body_api.CreateAngularVelocityAttr(angular_velocity)


def set_float_attr(prim: Usd.Prim, attr_name: str, value: float) -> None:
    """Set or create a float-valued USD attribute.

    Inputs are the prim, attribute name, and value; there is no output.
    This exists for robot joint state authoring where Isaac expects named float
    attributes on joint prims.
    """

    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
    attr.Set(float(value))
