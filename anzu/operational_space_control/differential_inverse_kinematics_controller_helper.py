"""
A re-implementation of MakeDifferentialInverseKinematicsControllerForStation
in Python, so that the open-source LBM Eval can set up differential IK without
relying on giant piles of native C++ code from Anzu.
"""
import functools
import math
from pathlib import Path

import numpy as np

from pydrake.geometry import (
    CollisionFilterDeclaration,
    ProximityProperties,
    SceneGraph,
    Sphere,
)
from pydrake.math import RigidTransform
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsController,
    DifferentialInverseKinematicsSystem,
)
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.plant import CoulombFriction, MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex, ScopedName
from pydrake.planning import (
    CollisionCheckerParams,
    DofMask,
    JointLimits,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

from anzu.common.anzu_model_directives import MakeDefaultAnzuPackageMap

_SUPPORTED_STATION_NAMES = ("cabot", "riverway")

# -----------------------------------------------------------------------------
# Begin Python transmogrification of make_planning_robot.cc.


# The panda control box has an internal collision model that is *very*
# conservative. Failure to satisfy *that* model causes faults. Our model (in
# the .urdf file) is a tight fit and would admit commands that will trigger
# faults. So, we introduce some empirically determined geometries to keep us
# away from the fault domain.
def _MakePandaCollisionCheckerMatchFrankas(
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
        model_instance: ModelInstanceIndex):

    default_properties = ProximityProperties()
    # N.B. For now we just lie.
    default_properties.AddProperty("material", "coulomb_friction",
                                   CoulombFriction(1.0, 1.0))

    link2_collision_sphere = Sphere(0.15)
    X_L2So = RigidTransform(np.zeros(3))

    link5_collision_sphere = Sphere(0.075)
    X_L5So = RigidTransform(np.array([0.0, 0.06, -0.125]))

    link0 = plant.GetBodyByName("panda_link0", model_instance)
    link1 = plant.GetBodyByName("panda_link1", model_instance)
    link2 = plant.GetBodyByName("panda_link2", model_instance)
    link3 = plant.GetBodyByName("panda_link3", model_instance)
    link4 = plant.GetBodyByName("panda_link4", model_instance)
    link5_lower = plant.GetBodyByName("panda_link5_lower", model_instance)
    link5_upper = plant.GetBodyByName("panda_link5_upper", model_instance)
    link6 = plant.GetBodyByName("panda_link6", model_instance)
    link7 = plant.GetBodyByName("panda_link7", model_instance)

    plant.RegisterCollisionGeometry(link2, X_L2So, link2_collision_sphere,
                                    "panda_link2_extra", default_properties)
    plant.RegisterCollisionGeometry(
        link5_lower, X_L5So, link5_collision_sphere, "panda_link5_extra",
        default_properties)

    # Reproduce the filters provided by the panda model that are affected by
    # the new collision geometry.

    # group_link0123
    set_L0123 = plant.CollectRegisteredGeometries([link0, link1, link2, link3])
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(set_L0123))

    # group_link1234
    set_L1234 = plant.CollectRegisteredGeometries([link1, link2, link3, link4])
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(set_L1234))

    # group_link3456
    set_L3456 = plant.CollectRegisteredGeometries(
        [link3, link4, link5_lower, link5_upper, link6])
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(set_L3456))

    # group_link567
    set_L567 = plant.CollectRegisteredGeometries(
        [link5_lower, link5_upper, link6, link7])
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(set_L567))


# Customizes collisions specific to whichever model may be present (with hacky
# existence checks). We do this directly on the MultibodyPlant so that any
# subsequent collision checker operations on the plant's bodies will include
# the modifications.
def _CustomizePlantCollisions(
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
        robot_models: list[ModelInstanceIndex]):
    for model_instance in robot_models:
        is_panda_arm = plant.HasBodyNamed("panda_link0", model_instance)
        if is_panda_arm:
            _MakePandaCollisionCheckerMatchFrankas(
                plant, scene_graph, model_instance)


def _MakeCollisionChecker(*, dmd_url: str) -> SceneGraphCollisionChecker:
    # Parse from directives using the anzu package map.
    model_builder = RobotDiagramBuilder()
    model_builder.parser().package_map().AddMap(MakeDefaultAnzuPackageMap())
    model_builder.parser().AddModelsFromUrl(dmd_url)

    # Identify model instances belonging to the robot.
    known_robot_model_instance_names = [
        "left::panda",
        "left::panda_hand",
        "left::framos_wrist_assembly",
        "left::panda_wrist_lucid_right",
        "left::panda_wrist_lucid_left",
        "right::panda",
        "right::panda_hand",
        "right::framos_wrist_assembly",
        "right::panda_wrist_lucid_right",
        "right::panda_wrist_lucid_left",
    ]
    robot_models = [
        model_builder.plant().GetModelInstanceByName(name)
        for name in known_robot_model_instance_names
        if model_builder.plant().HasModelInstanceNamed(name)
    ]

    # TODO(eric / siyuan): this should be done by model directives.
    _CustomizePlantCollisions(model_builder.plant(),
                              model_builder.scene_graph(), robot_models)

    # Create the collision checker around the model.
    model = model_builder.Build()
    checker_params = CollisionCheckerParams(
        model=model,
        robot_model_instances=robot_models,
        edge_step_size=0.05,
        env_collision_padding=0.0,
        self_collision_padding=0.0,
        implicit_context_parallelism=False,
        # We explicitly request the default C-space distance function. We're
        # not doing edge queries, so the default is fine.
        distance_and_interpolation_provider=None,
    )
    return SceneGraphCollisionChecker(checker_params)

# End Python transmogrification of make_planning_robot.cc.
# -----------------------------------------------------------------------------


def _ApplyJointLimitModifications(
        plant: MultibodyPlant, active_dof: DofMask,
        position_limits: dict[str: tuple[float | None, float | None]],
        velocity_limit_scale: float) -> JointLimits:
    # Start with the current limits.
    plant_limits = JointLimits(plant, True, True, True)
    position_lower = plant_limits.position_lower()
    position_upper = plant_limits.position_upper()
    velocity_lower = plant_limits.velocity_lower()
    velocity_upper = plant_limits.velocity_upper()
    acceleration_lower = plant_limits.acceleration_lower()
    acceleration_upper = plant_limits.acceleration_upper()

    # Mix in the position_limits dict.
    for name, (new_lower, new_upper) in position_limits.items():
        scoped_name = ScopedName.Parse(name)
        joint = plant.GetJointByName(
            scoped_name.get_element(),
            plant.GetModelInstanceByName(scoped_name.get_namespace()),
        )
        assert joint.num_positions() == 1
        if new_lower is not None:
            position_lower[joint.position_start()] = new_lower
        if new_upper is not None:
            position_upper[joint.position_start()] = new_upper

    # Mix in the velocity_limit_scale.
    velocity_lower *= velocity_limit_scale
    velocity_upper *= velocity_limit_scale

    # Pack the limits back into the class, returning only the active_dof.
    full_dof_result = JointLimits(
        position_lower=position_lower,
        position_upper=position_upper,
        velocity_lower=velocity_lower,
        velocity_upper=velocity_upper,
        acceleration_lower=acceleration_lower,
        acceleration_upper=acceleration_upper,
    )
    return JointLimits(full_dof_result, active_dof)


def _GetStationDmdUrl(station_name: str, postfix: list[str]) -> str:
    # This URL convention mimics anzu/common/make_robot_configuration.h.
    dmd_url = "package://lbm_eval_models/stations/"
    dmd_url += f"{station_name}/add_{station_name}"
    for item in postfix:
        if item:
            dmd_url += f"_{item}"
    dmd_url += ".dmd.yaml"
    return dmd_url


def _MakeDiffIkControllerForStationWithIngredients(
        station_name: str,
        postfix: list[str],
        active_model_instances: list[str],
        joint_limits: None,
        timestep: float,
        config_file: Path,
        allow_finger_table_collision: bool) -> (
            DifferentialInverseKinematicsController):
    """Constructs the diff ik controller for the named station. This also
    returns the ingredients that went into the diff ik system to aid in
    testing."""
    if station_name not in _SUPPORTED_STATION_NAMES:
        raise NotImplementedError(f"Unknown station {station_name}")
    if joint_limits is not None:
        raise NotImplementedError()
    if allow_finger_table_collision is not True:
        raise NotImplementedError()
    collision_checker = _MakeCollisionChecker(
        dmd_url=_GetStationDmdUrl(
            station_name=station_name,
            postfix=postfix,
        ),
    )
    plant = collision_checker.plant()
    active_dof = functools.reduce(DofMask.Union, [
        DofMask.MakeFromModel(plant, name)
        for name in active_model_instances
    ])

    # The magic numbers and strings below are the common values that appeared
    # in all supported configurations for Panda robot stations. The comparison
    # test for this module checks that the resulting controllers match the ones
    # produced by the original C++ implementations.
    DiffIkSystem = DifferentialInverseKinematicsSystem
    cartesian_axis_masks = {
        "right::panda::panda_link8": np.ones(6),
        "left::panda::panda_link8": np.ones(6),
    }
    spatial_velocity_as_vec6 = np.array(
        [math.radians(90.0), math.radians(90.0), math.radians(90.0),
         0.4, 0.4, 0.4])

    ingredients = []

    def add(Ingredient, **kwargs):
        ingredients.append(Ingredient(Ingredient.Config(**kwargs)))

    add(DiffIkSystem.LeastSquaresCost,
        cartesian_qp_weight=100.0,
        cartesian_axis_masks=cartesian_axis_masks,
        # See anzu#17024.
        use_legacy_implementation=True)
    add(DiffIkSystem.JointCenteringCost,
        posture_gain=1.0,
        cartesian_axis_masks=cartesian_axis_masks)
    add(DiffIkSystem.CartesianPositionLimitConstraint,
        p_TG_next_lower=np.array([-0.375, -0.7, 0.015]),
        p_TG_next_upper=np.array([0.2, 0.7, 0.9]))
    add(DiffIkSystem.CartesianVelocityLimitConstraint,
        V_next_TG_limit=spatial_velocity_as_vec6)
    limits = _ApplyJointLimitModifications(
        plant, active_dof,
        {"left::panda::panda_joint4": (None, math.radians(-35.0)),
         "right::panda::panda_joint4": (None, math.radians(-35.0))},
        velocity_limit_scale=0.9)
    Ingredient = DiffIkSystem.JointVelocityLimitConstraint
    ingredients.append(Ingredient(Ingredient.Config(), limits))
    add(DiffIkSystem.CollisionConstraint,
        safety_distance=0.01, influence_distance=0.05)

    recipe = DiffIkSystem.Recipe()
    for ingredient in ingredients:
        recipe.AddIngredient(ingredient)

    system = DiffIkSystem(
        recipe=recipe, task_frame="manipuland_table::table_top_center",
        collision_checker=collision_checker,
        active_dof=active_dof, time_step=timestep, K_VX=0.1,
        Vd_TG_limit=SpatialVelocity(spatial_velocity_as_vec6))
    result = DifferentialInverseKinematicsController(
        system, planar_rotation_dof_indices=[])
    return result, ingredients


def MakeDifferentialInverseKinematicsControllerForStation(
        station_name: str,
        postfix: list[str],
        active_model_instances: list[str],
        joint_limits: None,
        timestep: float,
        config_file: Path,
        allow_finger_table_collision: bool) -> (
            DifferentialInverseKinematicsController):
    # Normal usage strips away the ingredients.
    controller, _ = _MakeDiffIkControllerForStationWithIngredients(
        station_name=station_name,
        postfix=postfix,
        active_model_instances=active_model_instances,
        joint_limits=joint_limits,
        timestep=timestep,
        config_file=config_file,
        allow_finger_table_collision=allow_finger_table_collision)
    return controller
