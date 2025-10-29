import numpy as np

from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.tree import (
    BodyIndex,
    FrameIndex,
    JacobianWrtVariable,
    JointActuatorIndex,
    JointIndex,
    ModelInstanceIndex,
    PrismaticJoint,
    RevoluteJoint,
)

from anzu.common.containers import take_first


def _get_plant_aggregate(num_func, get_func, index_cls, model_instances=None,
                         ignore_ephemeral_elements=False):
    items = []
    for i in range(num_func()):
        item = get_func(index_cls(i))
        if (ignore_ephemeral_elements and hasattr(item, "is_ephemeral") and
                item.is_ephemeral()):
            continue
        if model_instances is None or item.model_instance() in model_instances:
            items.append(item)
    return items


def get_model_instances(plant):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_model_instances, lambda x: x,
        ModelInstanceIndex)


def get_bodies(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_bodies, plant.get_body, BodyIndex, model_instances)


def get_frames(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    # Unconditionally ignore ephemeral frames (frames added during Finalize()).
    # Those are used by Drake internally for performance reasons but of no
    # interest to Anzu.
    return _get_plant_aggregate(
        plant.num_frames, plant.get_frame, FrameIndex,
        model_instances=model_instances,
        ignore_ephemeral_elements=True)


def get_frames_attached_to(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    frames = []
    for frame in get_frames(plant):
        if frame.is_ephemeral():
            continue
        if frame.body() in bodies:
            frames.append(frame)
    return frames


def get_joints(plant, model_instances=None, ignore_ephemeral_joints=False):
    # TODO(eric.cousineau): Hoist this somewhere?
    # Optionally ignore ephemeral joints (joints added during Finalize() to
    # connect free bodies and trees to World). These can be quaternion or RPY
    # floating joints or weld joints depending on chosen MultibodyPlant
    # options.
    return _get_plant_aggregate(
        plant.num_joints, plant.get_joint, JointIndex,
        model_instances=model_instances,
        ignore_ephemeral_elements=ignore_ephemeral_joints)


def is_joint_solely_connected_to(joint, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    parent = joint.parent_body()
    child = joint.child_body()
    return parent in bodies and child in bodies


def get_joints_solely_connected_to(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    return [
        joint for joint in get_joints(plant)
        if is_joint_solely_connected_to(joint, bodies)]


def get_joint_actuators(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_actuators, plant.get_joint_actuator,
        JointActuatorIndex)


def get_joint_actuators_affecting_joints(plant, joints):
    # TODO(eric.cousineau): Hoist this somewhere?
    joint_actuators = []
    for joint_actuator in get_joint_actuators(plant):
        if joint_actuator.joint() in joints:
            joint_actuators.append(joint_actuator)
    return joint_actuators


def get_or_add_model_instance(plant, name):
    # TODO(eric.cousineau): Hoist this somewhere?
    if not plant.HasModelInstanceNamed(name):
        return plant.AddModelInstance(name)
    else:
        return plant.GetModelInstanceByName(name)


def get_geometries(plant, scene_graph, bodies=None):
    """Returns all GeometryId's attached to bodies. Assumes corresponding
    FrameId's have been added."""
    if bodies is None:
        bodies = get_bodies(plant)
    geometry_ids = []
    inspector = scene_graph.model_inspector()
    for body in list(bodies):
        frame_id = plant.GetBodyFrameIdOrThrow(body.index())
        body_geometry_ids = inspector.GetGeometries(frame_id)
        geometry_ids.extend(body_geometry_ids)
    # N.B. `inspector.GetGeometries` returns the ids in a consistent (sorted)
    # order, but we re-sort here just in case the geometries have been mutated.
    return sorted(geometry_ids, key=lambda x: x.get_value())


def get_joint_positions(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    return q[start:start + count].copy()


def set_joint_positions(plant, context, joint, qj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    q[start:start + count] = qj
    plant.SetPositions(context, q)


def get_joint_velocities(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    return v[start:start + count].copy()


def set_joint_velocities(plant, context, joint, vj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    v[start:start + count] = vj
    plant.SetVelocities(context, v)


def elements_sorted(xs):
    # TODO(eric.cousineau): Bind `__lt__` for sorting these types, and then
    # just use sorted().
    # Use https://github.com/RobotLocomotion/drake/pull/13489
    xs = list(xs)
    if len(xs) == 0:
        return xs
    x0 = take_first(xs)
    # TypeSafeIndex.
    try:
        int(x0)
        return sorted(xs, key=lambda x: int(x))
    except TypeError as e:
        if "int() argument" not in str(e):
            raise
    # MultibodyPlant element.
    try:
        int(x0.index())
        return sorted(xs, key=lambda x: int(x.index()))
    except AttributeError as e:
        if "has no attribute 'index'" not in str(e):
            raise
    # Geometry identifier.
    try:
        x0.get_value()
        return sorted(xs, key=lambda x: int(x.get_value()))
    except AttributeError as e:
        if "has no attribute 'get_value'" not in str(e):
            raise
    assert False


def get_frame_pose(plant, context, frame_T, frame_F):
    """Gets the pose of a frame."""
    X_TF = plant.CalcRelativeTransform(context, frame_T, frame_F)
    return X_TF


def set_default_frame_pose(plant, frame_F, X_WF):
    assert frame_F.body().is_floating()
    X_FB = frame_F.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WF @ X_FB
    plant.SetDefaultFreeBodyPose(frame_F.body(), X_WB)


def set_frame_pose(plant, context, frame_T, frame_F, X_TF):
    """Sets the pose of a frame attached to floating body."""
    if frame_T is None:
        frame_T = plant.world_frame()
    X_WT = plant.CalcRelativeTransform(context, plant.world_frame(), frame_T)
    assert frame_F.body().is_floating()
    X_FB = frame_F.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WT @ X_TF @ X_FB
    plant.SetFreeBodyPose(context, frame_F.body(), X_WB)


def get_frame_spatial_velocity(plant, context, frame_T, frame_F, frame_E=None):
    """
    Returns:
        SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
        (which is frame T if unspecified).
    """
    if frame_E is None:
        frame_E = frame_T
    Jv_TF_E = plant.CalcJacobianSpatialVelocity(
        context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_F,
        p_BoBp_B=[0, 0, 0],
        frame_A=frame_T,
        frame_E=frame_E,
    )
    v = plant.GetVelocities(context)
    V_TF_E = SpatialVelocity(Jv_TF_E @ v)
    return V_TF_E


def set_frame_spatial_velocity(
    plant, context, frame_T, frame_F, V_TF_E, frame_E=None
):
    if frame_E is None:
        frame_E = frame_T
    R_WE = plant.CalcRelativeTransform(
        context, plant.world_frame(), frame_E
    ).rotation()
    V_TF_W = V_TF_E.Rotate(R_WE)
    X_WT = plant.CalcRelativeTransform(context, plant.world_frame(), frame_T)
    V_WT = get_frame_spatial_velocity(
        plant, context, plant.world_frame(), frame_T
    )
    V_WF = V_WT.ComposeWithMovingFrameVelocity(X_WT.translation(), V_TF_W)
    body_B = frame_F.body()
    R_WB = plant.CalcRelativeTransform(
        context, plant.world_frame(), body_B.body_frame()
    ).rotation()
    p_BF = frame_F.GetFixedPoseInBodyFrame().translation()
    p_BF_W = R_WB @ p_BF
    V_WBf = V_WF.Shift(p_BF_W)
    plant.SetFreeBodySpatialVelocity(body_B, V_WBf, context)


def remove_role_from_geometries(plant, scene_graph, *, role, bodies=None):
    if bodies is None:
        bodies = get_bodies(plant)
    source_id = plant.get_source_id()
    for geometry_id in get_geometries(plant, scene_graph, bodies):
        scene_graph.RemoveRole(source_id, geometry_id, role)


def filter_all_collisions(plant, scene_graph):
    bodies = get_bodies(plant)
    geometries = get_geometries(plant, scene_graph, bodies)
    filter_manager = scene_graph.collision_filter_manager()
    geometry_set = GeometrySet(geometries)
    declaration = CollisionFilterDeclaration()
    declaration.ExcludeWithin(geometry_set)
    filter_manager.Apply(declaration)


def remove_joint_limits(plant):
    # TODO(eric.cousineau): Handle actuator limits when Drake supports mutating
    # them.
    for joint in get_joints(plant):
        num_q = joint.num_positions()
        num_v = joint.num_velocities()
        joint.set_position_limits(
            np.full(num_q, -np.inf), np.full(num_q, np.inf)
        )
        joint.set_velocity_limits(
            np.full(num_v, -np.inf), np.full(num_v, np.inf)
        )
        joint.set_acceleration_limits(
            np.full(num_v, -np.inf), np.full(num_v, np.inf)
        )


def remove_joint_damping(plant, model_instances=None):
    count = 0
    for joint in get_joints(plant, model_instances=model_instances):
        if isinstance(joint, (RevoluteJoint, PrismaticJoint)):
            joint.set_default_damping(0.0)
            count += 1
    return count
