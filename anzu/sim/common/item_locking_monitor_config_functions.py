from collections import defaultdict
import dataclasses
import itertools
import logging

import numpy as np

from pydrake.geometry import (
    Aabb,
    Box,
    CollisionFilterDeclaration,
    GeometryId,
    GeometryInstance,
    GeometrySet,
    Obb,
    ProximityProperties,
    QueryObject,
    Role,
    SceneGraph,
    SceneGraphInspector,
)
from pydrake.math import RigidTransform
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (
    BodyIndex,
    DeformableBody,
    DeformableBodyIndex,
    JointIndex,
    ModelInstanceIndex,
    RigidBody,
    ScopedName,
)
from pydrake.systems.framework import Context, EventStatus

from anzu.sim.common.hardware_station_monitor import (
    CompositeMonitor,
    HardwareStationMonitor,
)
from anzu.sim.common.item_locking_monitor_config import (
    ItemLockingMonitorConfig,
)


def _ObbToAabb(obb_G: Obb, X_RG: RigidTransform) -> Aabb:
    """Converts an object aligned bounding box (OBB) in its geometry frame G
    to an axis aligned bounding box (AABB) in the rigid body frame R.

    Args:
        obb_G: The oriented bounding box of the geometry in the geometry frame
            G.
        X_RG: The pose of the geometry frame G in the rigid body frame R.
    Returns:
        An AABB in the rigid body frame R. This is not necessarily the AABB of
        geometry G. It is the AABB of the original OBB and will likely be
        larger than the fit AABB of the geometry G.
    """
    # The center of the OBB, Bo, in the geometry frame G.
    p_GBo = obb_G.center()
    # The center of the AABB, Bo, in the rigid body frame R (aka, p_RBo).
    aabb_center_R = X_RG @ p_GBo
    X_GB = obb_G.pose()
    # The pose of the bounding box B in the rigid body frame R.
    X_RB = X_RG @ X_GB
    # Rather than creating all points and looking for the element-wise maximum
    # of those points, we observe that the x coordinate of the rotated points
    # will follow the pattern   +/- r11 * e_x +/- r12 * e_y +/- r13 * e_z   .
    # Thus the maximum of this value will occur when all of the terms are
    # positive, which is guaranteed to happen given the +/- in the pattern
    # above.  Thus we can directly skip to the final result by taking the
    # absolute value of the elements in the rotation matrix and multiplying
    # them with the size vector which is already known to be positive by
    # definition. The same holds for the y and z components.
    # See https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/ for
    # a more detailed breakdown.
    aabb_half_size_R = np.abs(X_RB.rotation().matrix()) @ obb_G.half_width()
    return Aabb(p_HoBo=aabb_center_R, half_width=aabb_half_size_R)


def _MergeAabbs(aabb_list: list[Aabb]) -> Aabb:
    """Given a list of AABBs that are measured and expressed in the same frame,
    returns the AABB (measured and expressed in the same frame) of the union of
    all the AABBs.
    """
    count = len(aabb_list)
    max_points = np.ndarray(shape=(3, count))
    min_points = np.ndarray(shape=(3, count))
    for i, bbox in enumerate(aabb_list):
        half_size = bbox.half_width()
        max_points[:, i] = bbox.center() + half_size
        min_points[:, i] = bbox.center() - half_size
    max_point = np.max(max_points, axis=1)
    min_point = np.min(min_points, axis=1)
    center = 0.5 * (max_point + min_point)
    size = max_point - min_point
    return Aabb(p_HoBo=center, half_width=size * 0.5)


def _GetProximityGeometriesForBody(inspector: SceneGraphInspector,
                                   body: RigidBody):
    """Returns the proximity geometries for the given body in its body frame.
    """
    frame_id = body.GetParentPlant().GetBodyFrameIdOrThrow(body.index())
    return inspector.GetGeometries(frame_id, role=Role.kProximity)


def _CalcAabb(inspector: SceneGraphInspector, body: RigidBody):
    """Calculates the AABB of the given body in its body frame."""
    aabb_B_list = list()
    for geom_id in _GetProximityGeometriesForBody(inspector, body):
        obb_G = inspector.GetObbInGeometryFrame(geom_id)
        # Bound the geometry OBB (expressed in the geometry frame), with an
        # AABB in the body frame (so that they can be aggregated) in the
        # common body frame.
        X_BG = inspector.GetPoseInFrame(geom_id)
        aabb_B = _ObbToAabb(obb_G, X_BG)
        aabb_B_list.append(aabb_B)
    return _MergeAabbs(aabb_B_list)


def get_contact_pairs_scoped_names(
    plant: MultibodyPlant, context: Context
) -> list[tuple[ScopedName, ScopedName]]:
    """Returns a list of tuples of scoped names for all rigid or deformable
    body pairs that are in contact. Point contact, hydroelastic contact, and
    deformable contact pairs are included.
    """

    contact_results = plant.get_contact_results_output_port().Eval(context)
    result = []

    # Point contact pairs.
    for i in range(contact_results.num_point_pair_contacts()):
        contact_result = contact_results.point_pair_contact_info(i)
        index_A = contact_result.bodyA_index()
        index_B = contact_result.bodyB_index()
        name_A = plant.get_body(index_A).scoped_name()
        name_B = plant.get_body(index_B).scoped_name()
        result.append((name_A, name_B))

    # Hydroelastic contact pairs.
    query_object = plant.get_geometry_query_input_port().Eval(context)
    inspector = query_object.inspector()
    for i in range(contact_results.num_hydroelastic_contacts()):
        contact_result = contact_results.hydroelastic_contact_info(i)
        geo_id_M = contact_result.contact_surface().id_M()
        geo_id_N = contact_result.contact_surface().id_N()
        frame_id_M = inspector.GetFrameId(geo_id_M)
        frame_id_N = inspector.GetFrameId(geo_id_N)
        body_M = plant.GetBodyFromFrameId(frame_id_M)
        body_N = plant.GetBodyFromFrameId(frame_id_N)
        result.append((body_M.scoped_name(), body_N.scoped_name()))

    deformable_model = plant.deformable_model()
    for i in range(contact_results.num_deformable_contacts()):
        deformable_contact = contact_results.deformable_contact_info(i)
        geometry_id_A = deformable_contact.id_A()
        geometry_id_B = deformable_contact.id_B()
        body_id_A = deformable_model.GetBodyId(geometry_id_A)
        body_A = deformable_model.GetBody(body_id_A)
        name_A = body_A.scoped_name()
        # A is always deformable. B might be rigid or deformable.
        is_B_deformable = inspector.IsDeformableGeometry(geometry_id_B)
        if is_B_deformable:
            body_id_B = deformable_model.GetBodyId(geometry_id_B)
            body_B = deformable_model.GetBody(body_id_B)
            name_B = body_B.scoped_name()
        else:
            index_B = plant.GetBodyFromFrameId(
                inspector.GetFrameId(geometry_id_B)
            ).index()
            body_B = plant.get_body(index_B)
            name_B = body_B.scoped_name()
        result.append((name_A, name_B))

    return result


def _contact_pairs_as_dict(
    contact_pairs: list[tuple[ScopedName, ScopedName]],
) -> dict[str: list[ScopedName]]:
    """Returns a dictionary of the form {name_A: [name_B, ...], ...} where
    name_A is a string of the fully qualified name of one of the bodies in
    contact and name_B's are the ScopedNames of the bodies in contact with body
    A.
    """
    contact_dict = defaultdict(list)
    for name_A, name_B in contact_pairs:
        contact_dict[name_A.get_full()].append(name_B)
        contact_dict[name_B.get_full()].append(name_A)
    return contact_dict


def _get_body_from_index(
    plant: MultibodyPlant, body_index: BodyIndex | DeformableBodyIndex
) -> RigidBody | DeformableBody:
    """Returns the body from the body index."""
    if isinstance(body_index, BodyIndex):
        return plant.get_body(body_index)
    return plant.deformable_model().GetBody(body_index)


def _get_body_from_scoped_name(
    plant: MultibodyPlant, scoped_name: ScopedName
) -> RigidBody | DeformableBody:
    """Returns the body from the scoped name. Throws an error if the body is
    not found.

    In the unlikely event that a deformable body and a rigid body both have the
    supplied name, the rigid body will be returned.
    """
    deformable_model = plant.deformable_model()
    namespace = scoped_name.get_namespace()
    element = scoped_name.get_element()
    if len(namespace):
        instance = plant.GetModelInstanceByName(namespace)
        if plant.HasBodyNamed(element, instance):
            return plant.GetBodyByName(element, instance)
        if deformable_model.HasBodyNamed(element, instance):
            return deformable_model.GetBodyByName(element, instance)
    else:
        if plant.HasBodyNamed(element):
            return plant.GetBodyByName(element)
        if deformable_model.HasBodyNamed(element):
            return deformable_model.GetBodyByName(element)
    raise ValueError(f"Body {scoped_name} not found in plant.")


# TODO(dale.mcconachie) Hoist this to Drake.
def _GetScopedGeometryByName(
    plant: MultibodyPlant, inspector: SceneGraphInspector, full_body_name: str
) -> list[GeometryId]:
    """If the name of a rigid body is given, returns the geometry ids of all
    the collision geometries associated with that body. If the name of a
    deformable body is given, returns the sole geometry id associated with that
    deformable body.

    In the unlikely event that a deformable body and a rigid body both have the
    supplied name, the rigid body will be returned.
    """
    scoped_name = ScopedName.Parse(full_body_name)
    body = _get_body_from_scoped_name(plant, scoped_name)
    if isinstance(body, DeformableBody):
        return [body.geometry_id()]
    return _GetProximityGeometriesForBody(inspector, body)


@dataclasses.dataclass(kw_only=True)
class _DeformableBodyData:
    """Stores the body index and geometry id of a deformable target."""

    body_index: DeformableBodyIndex
    geometry_id: GeometryId


@dataclasses.dataclass(kw_only=True)
class _RigidBodyData:
    """Stores the body index, proximity bounding box id, and geometry ids of a
    rigid target.
    """

    # The body index of the target.
    body_index: BodyIndex
    # The id of the proximity bounding box of the target.
    bounding_box_id: GeometryId
    # The geometry ids of the proximity geometries of the body.
    geometry_ids: list[GeometryId]
    # The index of the inboard joint of the target.
    joint_index: JointIndex


@dataclasses.dataclass(kw_only=True)
class _TargetDetail:
    """A helper class for ItemLockingMonitor that collects all relevant
    information about each model subject to locking.
    """

    model_instance_index: ModelInstanceIndex
    model_instance_name: str
    is_deformable: bool = False
    data: _DeformableBodyData | _RigidBodyData
    is_locked: bool = False
    want_locked: bool = False
    colliding_bodies_when_fell_asleep: list[ScopedName] | None = None


def _FindAllTargets(plant) -> list[_TargetDetail]:
    """Returns the "target" model instances the plant that are subject to
    locking (along with some helpful additional details about each model).
    A target is:
    - a model instance with only a single rigid body, where that body has no
      outboard joints, and is mobilized by a single 6dof inboard joint, and the
      joint's inboard parent body is anchored to the world.
    - a model instance with only a single deformable body.
    In the future, we might extend the definition (e.g., to model instances
    with more than one body, but still welded as an isolated subgraph).
    """
    # Precompute all world-affixed bodies.
    world_body_indices = set(
        [
            body.index()
            for body in plant.GetBodiesWeldedTo(plant.GetBodyByName("world"))
        ]
    )

    # Find all model instances that contain exactly one rigid body.
    rigid_candidates: dict[BodyIndex, _TargetDetail] = dict()
    for i in range(plant.num_model_instances()):
        i = ModelInstanceIndex(i)
        body_indices = plant.GetBodyIndices(i)
        if len(body_indices) != 1:
            continue
        body_index = body_indices[0]
        rigid_candidates[body_index] = _TargetDetail(
            model_instance_name=plant.GetModelInstanceName(i),
            model_instance_index=i,
            is_deformable=False,
            data=_RigidBodyData(
                body_index=body_index,
                geometry_ids=plant.GetCollisionGeometriesForBody(
                    plant.get_body(body_index)
                ),
                # We'll set joint_index later on within this function.
                joint_index=None,
                # We'll set bounding_box_id when ItemLockingMonitor is
                # initialized.
                bounding_box_id=None,
            ),
        )

    # Precompute all inboard and outboard joints for those bodies.
    inboards: dict[BodyIndex, list[JointIndex]] = dict()
    outboards: dict[BodyIndex, list[JointIndex]] = dict()
    for j in plant.GetJointIndices():
        joint = plant.get_joint(j)
        outboards.setdefault(joint.parent_body().index(), list()).append(j)
        inboards.setdefault(joint.child_body().index(), list()).append(j)

    # Find rigid bodies that meet our criteria.
    matches = []
    for body_index in rigid_candidates.keys():
        # Skip bodies that are inboard of something else.
        if len(outboards.get(body_index, [])) > 0:
            continue
        # Skip bodies without a single 6dof inboard joint.
        if len(inboards.get(body_index, [])) != 1:
            continue
        joint_index = inboards[body_index][0]
        joint = plant.get_joint(joint_index)
        if joint.num_positions() < 6:
            continue
        # Skip bodies with a non-world-affixed parent.
        if joint.parent_body().index() not in world_body_indices:
            continue
        rigid_candidates[body_index].data.joint_index = joint_index
        matches.append(body_index)

    result = [rigid_candidates[body_index] for body_index in matches]

    # Find deformable bodies that meet our criteria and add them to the result.
    deformable_model = plant.deformable_model()
    for i in range(plant.num_model_instances()):
        i = ModelInstanceIndex(i)
        deformable_body_ids = deformable_model.GetBodyIds(i)
        for deformable_body_id in deformable_body_ids:
            deformable_body = deformable_model.GetBody(deformable_body_id)
            detail = _TargetDetail(
                model_instance_index=i,
                model_instance_name=plant.GetModelInstanceName(i),
                is_deformable=True,
                data=_DeformableBodyData(
                    body_index=deformable_body.index(),
                    geometry_id=deformable_body.geometry_id(),
                ),
            )
            result.append(detail)

    return result


def _deformable_target_overlaps_rigid_target(
    deformable_target: _DeformableBodyData,
    rigid_target: _RigidBodyData,
    distance_threshold: float,
    query_object: QueryObject,
) -> bool:
    """Checks if the given deformable target might be within the distance
    threshold of the given rigid target."""
    aabb_i_W = query_object.ComputeAabbInWorld(deformable_target.geometry_id)
    padded_aabb_i_W = Aabb(
        p_HoBo=aabb_i_W.center(),
        half_width=aabb_i_W.half_width() + distance_threshold,
    )
    for geometry_id in rigid_target.geometry_ids:
        obb_j_W = query_object.ComputeObbInWorld(geometry_id)
        if Obb.HasOverlap(obb_j_W, padded_aabb_i_W, RigidTransform.Identity()):
            return True
    return False


def _are_targets_overlapping(
    target_i: _TargetDetail,
    target_j: _TargetDetail,
    distance_threshold: float,
    query_object: QueryObject,
) -> bool:
    """Checks if the given targets are approximately within the distance
    threshold of each other (approximate in the sense that the targets might
    be a bit farther apart than the distance threshold, but still close)."""
    if target_i.is_deformable and target_j.is_deformable:
        aabb_i_W = query_object.ComputeAabbInWorld(target_i.data.geometry_id)
        padded_aabb_i_W = Aabb(
            p_HoBo=aabb_i_W.center(),
            half_width=aabb_i_W.half_width() + distance_threshold,
        )
        aabb_j_W = query_object.ComputeAabbInWorld(target_j.data.geometry_id)
        return Aabb.HasOverlap(
            padded_aabb_i_W, aabb_j_W, RigidTransform.Identity()
        )
    if target_i.is_deformable:
        # i is deformable, j is rigid.
        return _deformable_target_overlaps_rigid_target(
            target_i.data,
            target_j.data,
            distance_threshold,
            query_object,
        )
    if target_j.is_deformable:
        # j is deformable, i is rigid.
        return _deformable_target_overlaps_rigid_target(
            target_j.data,
            target_i.data,
            distance_threshold,
            query_object,
        )
    # Both are rigid. Use the bounding box of the rigid target.
    signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(
        target_i.data.bounding_box_id,
        target_j.data.bounding_box_id,
    )
    return signed_distance_pair.distance <= distance_threshold


# TODO(dale.mcconachie) Convert this to a LeafSystem with inputs and outputs
# once joints can be locked/unlocked from within the systems framework and
# collision filters are similarly supported.
# https://github.com/RobotLocomotion/drake/issues/20571
# TODO(jeremy.nimmer) We've outgrown the name "item", so should adopt something
# more conventional. The typical term of art here is "sleeping".
class ItemLockingMonitor(HardwareStationMonitor):
    """This class is a simulation monitor designed to lock rigid manipulands
    when they are far away from the grippers, unlocking when the grippers
    approach that manipuland. Designed purely for simulation speed improvements
    rather than any physical process. No deformable bodies are ever locked.
    """

    def __init__(
        self,
        config: ItemLockingMonitorConfig,
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
    ):
        super().__init__()
        assert plant.is_finalized()
        self._t_start = config.t_start
        self._distance_threshold = config.distance_threshold
        self._speed_threshold = config.speed_threshold
        inspector = scene_graph.model_inspector()
        self._unlock_near_ids: list[GeometryId] = list(
            itertools.chain(
                *[
                    _GetScopedGeometryByName(plant, inspector, name)
                    for name in config.unlock_near_geometry
                ]
            )
        )
        self._plant = plant
        self._scene_graph = scene_graph
        self._targets = _FindAllTargets(plant)
        self._plant_rigid_state = None
        self._plant_deformable_state = None
        self._geometry_filter_id = None

        # Add a proximity bounding box helper geometry for each lockable rigid
        # body. The shape of deformable bodies change over time, so we can't
        # use a fixed bounding box for deformable bodies over time. Instead,
        # we obtain the bounding box from QueryObject at each time step.
        for detail in self._targets:
            if detail.is_deformable:
                continue
            # Calculate a (not necessarily tight) AABB encompassing all the
            # rigid body's proximity geometry (in body frame B).
            body = plant.get_body(detail.data.body_index)
            aabb_B = _CalcAabb(inspector, body)

            # Add the bounding box as a proximity shape; it must be a proximity
            # shape to be able to call ComputeSignedDistancePairClosestPoints.
            # (Registering a geometry directly on the scene graph instead of
            # laundering it through the plant would be a problem if the added
            # geometry were to collide with anything; however, we will be
            # filtering out all collisions involving these added geometries,
            # thus there will never be any collisions to be resolved.)
            bounding_box_id = scene_graph.RegisterGeometry(
                source_id=plant.get_source_id(),
                frame_id=plant.GetBodyFrameIdIfExists(detail.data.body_index),
                geometry=GeometryInstance(
                    X_PG=RigidTransform(aabb_B.center()),
                    shape=Box(aabb_B.half_width() * 2.0),
                    name=body.name() + "_aabb_for_joint_locking",
                ),
            )
            scene_graph.AssignRole(
                source_id=plant.get_source_id(),
                geometry_id=bounding_box_id,
                properties=ProximityProperties(),
            )
            detail.data.bounding_box_id = bounding_box_id

        # Filter out all collisions between our added bounding box geometries
        # and everything else (including themselves).
        filter_manager = scene_graph.collision_filter_manager()
        filter_manager.Apply(
            CollisionFilterDeclaration().ExcludeBetween(
                GeometrySet(
                    [
                        detail.data.bounding_box_id
                        for detail in self._targets
                        if not detail.is_deformable
                    ]
                ),
                GeometrySet(inspector.GetAllGeometryIds()),
            )
        )

    def _is_target_close_to_unlock_geometries(
        self,
        detail: _TargetDetail,
        query_object,
    ) -> bool:
        """Checks if the given target is close to any of the unlock
        geometries."""
        if detail.is_deformable:
            # For deformable bodies, we get its updated bounding box from
            # QueryObject at each time step and compare it with the bounding
            # box of the unlock near geometries.
            aabb_W = query_object.ComputeAabbInWorld(detail.data.geometry_id)
            # Pad the AABB with the distance threshold and test for overlaps
            # with the OBBs of the unlock near geometries.
            padded_aabb_W = Aabb(
                p_HoBo=aabb_W.center(),
                half_width=aabb_W.half_width() + self._distance_threshold / 2,
            )
            for id in self._unlock_near_ids:
                unlock_near_obb_W = query_object.ComputeObbInWorld(id)
                if Obb.HasOverlap(
                    unlock_near_obb_W,
                    padded_aabb_W,
                    RigidTransform.Identity(),
                ):
                    return True
        else:
            # For rigid bodies, we get the signed distance between the rigid
            # body and the unlock near geometries. (We could just estimate the
            # distances using the bounding boxes similar to how deformables
            # are handled, but we keep this behavior for backward
            # compatibility.)
            for unlock_near_id in self._unlock_near_ids:
                signed_distance_pair = (
                    query_object.ComputeSignedDistancePairClosestPoints(
                        detail.data.bounding_box_id, unlock_near_id
                    )
                )
                if signed_distance_pair.distance <= self._distance_threshold:
                    return True
        return False

    def Monitor(self, root_context: Context) -> EventStatus:
        """Callback for Simulator.set_monitor that computes the desired joint
        locking status for monitored targets, and breaks out of the simulation
        in case any has changed. The desired status is stored as member data;
        users should call the SetItemLockStates() method to update the joint
        locking state.
        """
        if root_context.get_time() < self._t_start:
            return EventStatus.DidNothing()

        plant_context = self._plant.GetMyContextFromRoot(root_context)
        query_object = self._plant.get_geometry_query_input_port().Eval(
            plant_context
        )

        # To improve simulation throughput, we early-return if the
        # MultibodyPlant state hasn't changed since the last call.

        # Check if rigid body states have changed.
        new_plant_rigid_state = self._plant.GetPositionsAndVelocities(
            plant_context
        )
        rigid_state_changed = not np.array_equal(
            new_plant_rigid_state, self._plant_rigid_state
        )

        # Check if any deformable body states have changed.
        deformable_model = self._plant.deformable_model()
        new_plant_deformable_state = []
        for i in range(deformable_model.num_bodies()):
            body_index = DeformableBodyIndex(i)
            deformable_body = deformable_model.GetBody(body_index)
            new_state_i = deformable_body.GetPositionsAndVelocities(
                plant_context
            )
            new_plant_deformable_state.extend(new_state_i)
        deformable_state_changed = not np.array_equal(
            new_plant_deformable_state, self._plant_deformable_state
        )

        # Early return if no state changes detected.
        state_changed = rigid_state_changed or deformable_state_changed
        if not state_changed:
            return EventStatus.DidNothing()

        # Update cached state.
        self._plant_rigid_state = new_plant_rigid_state
        self._plant_deformable_state = new_plant_deformable_state

        # Default to locking everything, unlocking only if it is close to a
        # finger. This considers each target independently; we'll consider
        # any target-on-target interaction in a second pass, below.
        velocities_port = self._plant.get_body_spatial_velocities_output_port()
        velocities = velocities_port.Eval(plant_context)
        for detail in self._targets:
            detail.want_locked = True
            # Check the translational speed limit.
            # TODO(jeremy-nimmer) This only considers translation. For rigid
            # bodies, there might be cases where rotational speed should also
            # prevent locking even if the target is translationally stationary.
            # For deformable bodies, a body that is deforming around its center
            # of mass, such that the center of mass is stationary, should not
            # be locked.
            if detail.is_deformable:
                deformable_body = deformable_model.GetBody(
                    detail.data.body_index
                )
                v_WBcom = (
                    deformable_body
                    .CalcCenterOfMassTranslationalVelocityInWorld(
                        plant_context
                    )
                )
                speed = np.linalg.norm(v_WBcom)
            else:
                spatial_velocity = velocities[detail.data.body_index]
                speed = np.linalg.norm(spatial_velocity.translational())

            if not speed < self._speed_threshold:
                detail.want_locked = False
                continue
            # Check the distance limit.
            if self._is_target_close_to_unlock_geometries(detail,
                                                          query_object):
                detail.want_locked = False

        # Targets that are (conservatively) in contact with each other must
        # either all lock at once, or none of them can lock at all. First,
        # we'll compute the (conservative) collision status for all target
        # pairs (excluding self-collisions).
        num_targets = len(self._targets)
        contacts = np.zeros(shape=(num_targets, num_targets))
        for i in range(num_targets):
            target_i = self._targets[i]
            for j in range(i + 1, num_targets):
                target_j = self._targets[j]
                contact = (
                    1
                    if _are_targets_overlapping(
                        target_i,
                        target_j,
                        self._distance_threshold,
                        query_object,
                    )
                    else 0
                )
                contacts[i, j] = contact
                contacts[j, i] = contact
        # Second, we'll propagate the "unlocked" attribute across collisions.
        unlocked_to_visit = set(
            [i for i, x in enumerate(self._targets) if not x.want_locked]
        )
        unlocked_visited = set()
        while unlocked_to_visit:
            i = unlocked_to_visit.pop()
            unlocked_visited.add(i)
            target_i = self._targets[i]
            assert not target_i.want_locked
            for j in range(num_targets):
                if contacts[i, j]:
                    target_j = self._targets[j]
                    if target_j.want_locked:
                        assert j not in unlocked_to_visit
                        assert j not in unlocked_visited
                        target_j.want_locked = False
                        unlocked_to_visit.add(j)

        if self._needs_update():
            return EventStatus.ReachedTermination(None, "Joint locking update")
        else:
            return EventStatus.DidNothing()

    def _needs_update(self) -> bool:
        """Returns True iff we need to stop the sim to change the locking."""
        return any([x.is_locked != x.want_locked for x in self._targets])

    def HandleExternalUpdates(self, root_context: Context) -> None:
        """Updates the context with the lock states computed by the most recent
        call to the Monitor() method.
        """
        if not self._needs_update():
            # It's extremely important not to change the context when locking
            # details are quiescent. Changes are non-trivially expensive.
            return

        plant_context = self._plant.GetMyContextFromRoot(root_context)
        query_object = self._plant.get_geometry_query_input_port().Eval(
            plant_context
        )
        inspector = query_object.inspector()
        contact_pairs = get_contact_pairs_scoped_names(
            self._plant, plant_context
        )
        contact_pairs_dict = _contact_pairs_as_dict(contact_pairs)
        # Update the MultibodyPlant context.
        plant_context = self._plant.GetMyMutableContextFromRoot(root_context)
        locked_geom_ids = []

        for detail in self._targets:
            body = _get_body_from_index(self._plant, detail.data.body_index)
            if detail.want_locked:
                if detail.is_deformable:
                    locked_geom_ids.append(body.geometry_id())
                else:
                    locked_geom_ids.extend(
                        _GetProximityGeometriesForBody(inspector, body)
                    )
                if not detail.is_locked:
                    # TODO(dale.mcconachie) Consider quieting this.
                    logging.info(f"Locking {detail.model_instance_name}")
                    if detail.is_deformable:
                        body.Disable(plant_context)
                    else:
                        joint = self._plant.get_joint(detail.data.joint_index)
                        joint.Lock(plant_context)
                    detail.is_locked = True
                    detail.colliding_bodies_when_fell_asleep = (
                        contact_pairs_dict[body.scoped_name().get_full()]
                    )
                    # TODO(dale.mcconachie) We should make this really check
                    # for obj being static for more than a fixed period of
                    # time. Right now, this could lock items in mid air with
                    # epsilon velocity.
            else:
                if detail.is_locked:
                    # TODO(dale.mcconachie) Consider quieting this.
                    logging.info(f"Unlocking {detail.model_instance_name}")

                    if detail.is_deformable:
                        body.Enable(plant_context)
                    else:
                        joint = self._plant.get_joint(detail.data.joint_index)
                        joint.Unlock(plant_context)
                    detail.is_locked = False
                    detail.colliding_bodies_when_fell_asleep = None

        # Update the SceneGraph context.
        new_filter = CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(locked_geom_ids),
            GeometrySet(query_object.inspector().GetAllGeometryIds()),
        )
        filter_manager = self._scene_graph.collision_filter_manager(
            self._scene_graph.GetMyMutableContextFromRoot(root_context)
        )
        if self._geometry_filter_id is not None:
            filter_manager.RemoveDeclaration(self._geometry_filter_id)
        self._geometry_filter_id = filter_manager.ApplyTransient(new_filter)

    def get_colliding_bodies_when_fell_asleep(self):
        """Returns the list of sleeping contact pairs (pairs of bodies that are
        in contact and are both sleeping) in the same format as returned by
        `get_contact_pairs_scoped_names(...)`.
        """
        contact_pairs = []
        for detail in self._targets:
            if detail.colliding_bodies_when_fell_asleep is not None:
                body_A = _get_body_from_index(
                    self._plant, detail.data.body_index
                )
                name_A = body_A.scoped_name()
                for name_B in detail.colliding_bodies_when_fell_asleep:
                    if name_A.get_full() < name_B.get_full():
                        contact_pairs.append((name_A, name_B))
                    else:
                        contact_pairs.append((name_B, name_A))
        return contact_pairs


def ApplyItemLockingMonitorConfig(
    config: ItemLockingMonitorConfig,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    monitor: CompositeMonitor,
) -> None:
    """Prepares the given plant and scene_graph for free body locking.
    Adds the ItemLockingMonitor to the given CompositeMonitor.
    @pre The plant is finalized.
    """
    monitor.add_monitor(ItemLockingMonitor(config, plant, scene_graph))
