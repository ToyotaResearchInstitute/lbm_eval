"""
Provides subgraph functionality for MultibodyPlant. For the definition of
"subgraph", please see the MultibodyPlantSubgraph class documentation.

The MultibodyPlantSubgraph class was designed for the following workflows:

- Identifying a subgraph of a source MultibodyPlant, and *possibly* a
  SceneGraph. It does not matter if the plant is finalized or not.

- Extracting a subgraph (breaking a big model apart) and copying it into a new
  MultibodyPlant, and *possibly* a new SceneGraph. This can be loosely defined
  as "decomposition", but it's done by copies, not decomposing the original
  element.
    - This can be used for specifying controllers from an existing "super"
    plant.

- Composition (making a big model) by taking subgraphs from existing plants and
  add them to a new plant.
    - This can be used to assemble scenes in a modular fashion.
    - This can also be used to "cache" parsing results; rather than trying
      to remember which model instance came from which URDF / SDFormat
      file, where it was welded, etc. Instead, just take the relevant portion
      of the plant / graph.

- Creating an "un-finalized" instance of a MultibodyPlant (or portion) by
  making a subgraph consisting of the portion of interest and copying it into a
  new plant.
    - It can be used to replace joints and/or floating bodies with welds which
      can be useful for fixing degrees of freedom for controllers, etc.

- Creating a "finalized" copy of an unfinalized MultibodyPlant as means to
  compute kinematics using an unfinalized plant.

- Sometimes, a (MultibodyPlant, SceneGraph) might be part of a diagram that
  cannot be converted, e.g., .ToAutoDiffXd(), due to limitations of other
  systems in the diagram (e.g. LCM publishers). The (MultibodyPlant,
  SceneGraph) pair can be copied to a new diagram with only those elements, so
  that they can be used in optimization.

For examples of these workflows, please see `multibody_plant_subgraph_test.py`
in Anzu.

Currently out of scope for this design:

- Any scalar types other than T=float (T=double in C++). This is meant to
  "mirror" parsing code, and only deal with MultibodyPlant_[float].

- This is *only* used for copying subgraphs of a MultibodyPlant (and possibly a
  SceneGraph). It makes no steps towards trying to identify and copy subsets of
  Systems Framework Diagrams.
"""

from collections import OrderedDict
import copy
from functools import partial

from pydrake.geometry import GeometryId, SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import (
    BallRpyJoint,
    FixedOffsetFrame,
    Frame,
    Joint,
    JointActuator,
    PrismaticJoint,
    QuaternionFloatingJoint,
    RevoluteJoint,
    RigidBody,
    RigidBodyFrame,
    UniversalJoint,
    WeldJoint,
    default_model_instance,
    world_model_instance,
)
from pydrake.systems.framework import Context, DiagramBuilder

from anzu.common import multibody_extras as me
from anzu.common.containers import SortedDict, SortedSet


def model_instance_remap_to_default(plant_src, model_instance_src, plant_dest):
    """Remap to default model instance (intended for plant_dest)."""
    return default_model_instance()


def create_model_instance_remap_by_name(func):
    """Creates remap based on transforming model instance name.
    Example:

        @create_model_instance_remap_by_name
        def model_instance_remap_with_prefix(name_src):
            name_dest = f"my_custom_prefix::{name_src}"
            return name_dest
    """

    def remap(plant_src, model_instance_src, plant_dest):
        name = plant_src.GetModelInstanceName(model_instance_src)
        new_name = func(name)
        return me.get_or_add_model_instance(plant_dest, new_name)

    return remap


@create_model_instance_remap_by_name
def model_instance_remap_same_name(name):
    return name


_FIELDS = (
    'model_instances',
    'bodies',
    'frames',
    'joints',
    'joint_actuators',
    'geometry_ids',
)


class _MultibodyPlantElementsBase:
    """Provides a container for elements that MultibodyPlantSubgraph allows
    the user to interact with."""
    def __init__(self, container_cls):
        self.model_instances = container_cls()
        self.bodies = container_cls()
        self.frames = container_cls()
        self.joints = container_cls()
        self.joint_actuators = container_cls()
        # TODO(eric.cousineau): How to handle force elements?
        self.geometry_ids = container_cls()
        # TODO(eric.cousineau): How to handle collision filters?

    def _copy_to(self, new):
        new.model_instances = copy.copy(self.model_instances)
        new.bodies = copy.copy(self.bodies)
        new.frames = copy.copy(self.frames)
        new.joints = copy.copy(self.joints)
        new.joint_actuators = copy.copy(self.joint_actuators)
        new.geometry_ids = copy.copy(self.geometry_ids)

    def __copy__(self):
        """Makes a "level 2" shallow copy."""
        new = _MultibodyPlantElementsBase()
        self._copy_to(new)
        return new

    def _apply(self, container_func, other):
        container_func(self.model_instances, other.model_instances)
        container_func(self.bodies, other.bodies)
        container_func(self.frames, other.frames)
        container_func(self.joints, other.joints)
        container_func(self.joint_actuators, other.joint_actuators)
        container_func(self.geometry_ids, other.geometry_ids)
        return self

    def __eq__(self, other):
        return (
            self.model_instances == other.model_instances and
            self.bodies == other.bodies and
            self.frames == other.frames and
            self.joints == other.joints and
            self.joint_actuators == other.joint_actuators and
            self.geometry_ids == other.geometry_ids)

    def as_tuple(self):
        return (
            ('model_instances', self.model_instances),
            ('bodies', self.bodies),
            ('frames', self.frames),
            ('joints', self.joints),
            ('joint_actuators', self.joint_actuators),
            ('geometry_ids', self.geometry_ids),
        )

    def __repr__(self):
        s = "(\n  " + ",\n  ".join(str(x) for x in self.as_tuple()) + ",\n)"
        return s


def _exclusive_set_update(dest, src):
    assert len(src) == len(set(src))  # Unique source material
    assert dest.isdisjoint(src)  # No common elements.
    dest.update(src)


class MultibodyPlantElements(_MultibodyPlantElementsBase):
    """
    Aggregates elements from a MultibodyPlant (and optional SceneGraph).

    This does *not* check for any invariants. Use MultibodyPlantSubgraph for
    more thorough bookkeeping instead.
    """
    def __init__(self, plant, scene_graph=None):
        sorted_factory = partial(SortedSet, sorted=me.elements_sorted)
        super().__init__(sorted_factory)
        assert isinstance(plant, MultibodyPlant)
        if scene_graph is not None:
            assert isinstance(scene_graph, SceneGraph)
        self.plant = plant
        self.scene_graph = scene_graph

    def __copy__(self):
        new = MultibodyPlantElements(self.plant, self.scene_graph)
        self._copy_to(new)
        return new

    def _check(self, other):
        # Ensure that these are consistent.
        assert self.plant == other.plant
        assert self.scene_graph == other.scene_graph

    def __iadd__(self, other):
        self._check(other)
        self._apply(_exclusive_set_update, other)
        return self

    def __eq__(self, other):
        return (
            self.plant is other.plant and
            self.scene_graph is other.scene_graph and
            super().__eq__(other))


def get_elements_from_bodies(
        plant, bodies, scene_graph=None, model_instances=None):
    elem = MultibodyPlantElements(plant, scene_graph)
    elem.bodies.update(bodies)
    if model_instances is None:
        elem.model_instances = {x.model_instance() for x in elem.bodies}
    else:
        elem.model_instances.update(model_instances)
    elem.joints.update(me.get_joints_solely_connected_to(
        plant, elem.bodies))
    elem.joint_actuators.update(me.get_joint_actuators_affecting_joints(
            plant, elem.joints))
    elem.frames.update(me.get_frames_attached_to(plant, elem.bodies))
    if scene_graph is not None:
        elem.geometry_ids.update(
            me.get_geometries(plant, scene_graph, elem.bodies))
    return elem


def get_elements_from_plant(plant, scene_graph=None, model_instances=None):
    if model_instances is None:
        model_instances = me.get_model_instances(plant)
    return get_elements_from_bodies(
        plant=plant,
        bodies=me.get_bodies(plant, model_instances),
        scene_graph=scene_graph,
        model_instances=model_instances,
    )


def _add_item(container, key, value):
    # Adds an item, ensuring that it does not already exist.
    assert key not in container, key
    container[key] = value


def _get_cls_or_factory(container):
    cls = type(container)
    if issubclass(cls, SortedDict):
        return partial(cls, sorted_keys=container._sorted_keys)
    elif issubclass(cls, SortedSet):
        return partial(cls, sorted=container._sorted)
    else:
        return cls


def _inverse(container, strict):
    cls = _get_cls_or_factory(container)
    new = cls()
    for src, dest in container.items():
        if strict:
            assert dest not in new
        new[dest] = src
    return new


class MultibodyPlantElementsMap(_MultibodyPlantElementsBase):
    """
    Handles both the copying of elements from `plant_src` (and
    `scene_graph_src`), keeping track of those associations, and handling of
    transferring state.

    This does *not* check for any invariants. Use MultibodyPlantSubgraph for
    bookkeeping instead.
    """

    # TODO(eric.cousineau): All formulations of "copy-to-with-associations"
    # should ideally be formulated individually (e.g. `Joint.CopyTo(...)`),
    # rather than aggregating all of it here.

    def __init__(
            self, plant_src, plant_dest,
            scene_graph_src=None, scene_graph_dest=None):
        sorted_factory = partial(SortedDict, sorted_keys=me.elements_sorted)
        super().__init__(sorted_factory)
        self.plant_src = plant_src
        self.scene_graph_src = scene_graph_src
        self.plant_dest = plant_dest
        self.scene_graph_dest = scene_graph_dest
        # These elements are allowed to "repeat".
        self._builtins_src = self.make_empty_elements_src()

    def __eq__(self, other):
        return (
            self.plant_src is other.plant_src and
            self.scene_graph_src is other.scene_graph_src and
            self.plant_dest is other.plant_dest and
            self.scene_graph_dest is other.scene_graph_dest and
            super().__eq__(other))

    def make_empty_elements_src(self):
        return MultibodyPlantElements(self.plant_src, self.scene_graph_src)

    def make_empty_elements_dest(self):
        return MultibodyPlantElements(self.plant_dest, self.scene_graph_dest)

    def inverse(self, strict=True):
        """Reverses associations. If strict, then this must be a injective
        mapping (e.g. each source element maps to a unique destination
        element)."""
        new = MultibodyPlantElementsMap(
            self.plant_dest, self.plant_src,
            self.scene_graph_dest, self.scene_graph_src)
        new.model_instances = _inverse(self.model_instances, strict)
        new.bodies = _inverse(self.bodies, strict)
        new.frames = _inverse(self.frames, strict)
        new.joints = _inverse(self.joints, strict)
        new.joint_actuators = _inverse(self.joint_actuators, strict)
        new.geometry_ids = _inverse(self.geometry_ids, strict)
        return new

    def register_world_body_and_frame(self):
        """Registers the world body and frame."""
        plant_src = self.plant_src
        plant_dest = self.plant_dest
        _add_item(
            self.bodies,
            plant_src.world_body(), plant_dest.world_body())
        self._builtins_src.bodies.add(plant_src.world_body())
        _add_item(
            self.frames,
            plant_src.world_frame(), plant_dest.world_frame())
        self._builtins_src.frames.add(plant_src.world_frame())

    def register_model_instance(self, model_instance_src, model_instance_dest):
        """Register a ModelInstanceIndex for both the source and destination
        plant."""
        _add_item(
            self.model_instances,
            model_instance_src, model_instance_dest)
        return model_instance_dest

    def copy_body(self, body_src):
        """Copies a Body from the source plant to the destination plant.

        Note: The world body is handled by `register_world_body_and_frame()`,
        and is ignored by this method.
        """
        if body_src in self._builtins_src.bodies:
            return
        assert type(body_src) is RigidBody, (
            f"Body must only be RigidBody: {body_src}")
        plant_src = self.plant_src
        plant_dest = self.plant_dest
        model_instance_src = body_src.model_instance()
        model_instance_dest = self.model_instances[
            model_instance_src]
        body_dest = plant_dest.AddRigidBody(
            name=body_src.name(),
            M_BBo_B=body_src.default_spatial_inertia(),
            model_instance=model_instance_dest,
        )
        _add_item(self.bodies, body_src, body_dest)
        # Set default state.
        X_WB = plant_src.GetDefaultFreeBodyPose(body_src)
        plant_dest.SetDefaultFreeBodyPose(body_dest, X_WB)
        # Register body frame as a builtin.
        frame_src = body_src.body_frame()
        frame_dest = body_dest.body_frame()
        self._builtins_src.frames.add(frame_src)
        _add_item(self.frames, frame_src, frame_dest)
        return frame_src

    def copy_frame(self, frame_src):
        """Copies a Frame to be added to the destination plant.

        Note: RigidBodyFrame's are handled by `copy_body`, and are ignored by
        this method.
        """
        if frame_src in self._builtins_src.frames:
            return
        plant_dest = self.plant_dest
        assert not isinstance(frame_src, RigidBodyFrame), (
            f"{frame_src}, {frame_src.name()}")
        parent_frame_src = frame_src.body().body_frame()
        parent_frame_dest = self.frames[parent_frame_src]
        model_instance_src = frame_src.model_instance()
        model_instance_dest = (
            self.model_instances[model_instance_src])
        frame_dest = FixedOffsetFrame(
            name=frame_src.name(), P=parent_frame_dest,
            X_PF=frame_src.GetFixedPoseInBodyFrame(),
            model_instance=model_instance_dest,
        )
        frame_dest = plant_dest.AddFrame(frame_dest)
        _add_item(self.frames, frame_src, frame_dest)

    def copy_joint(self, joint_src):
        """Copies a joint to be added to the destination plant."""
        assert isinstance(joint_src, Joint)
        plant_dest = self.plant_dest
        frame_on_parent_dest = self.frames[joint_src.frame_on_parent()]
        frame_on_child_dest = self.frames[joint_src.frame_on_child()]
        # N.B. We use `type(x) == cls`, not `isinstance(x, cls)`, so that we
        # know we recreate the exact types.
        if type(joint_src) is BallRpyJoint:
            joint_dest = BallRpyJoint(
                name=joint_src.name(),
                frame_on_parent=frame_on_parent_dest,
                frame_on_child=frame_on_child_dest,
                damping=joint_src.default_damping(),
            )
        elif type(joint_src) is PrismaticJoint:
            joint_dest = PrismaticJoint(
                name=joint_src.name(),
                frame_on_parent=frame_on_parent_dest,
                frame_on_child=frame_on_child_dest,
                axis=joint_src.translation_axis(),
                damping=joint_src.default_damping(),
            )
        elif type(joint_src) is RevoluteJoint:
            joint_dest = RevoluteJoint(
                name=joint_src.name(),
                frame_on_parent=frame_on_parent_dest,
                frame_on_child=frame_on_child_dest,
                axis=joint_src.revolute_axis(),
                damping=joint_src.default_damping(),
            )
        elif type(joint_src) is UniversalJoint:
            joint_dest = UniversalJoint(
                name=joint_src.name(),
                frame_on_parent=frame_on_parent_dest,
                frame_on_child=frame_on_child_dest,
                damping=joint_src.default_damping(),
            )
        elif type(joint_src) is WeldJoint:
            joint_dest = WeldJoint(
                name=joint_src.name(),
                frame_on_parent_F=frame_on_parent_dest,
                frame_on_child_M=frame_on_child_dest,
                X_FM=joint_src.X_FM(),
            )
        elif type(joint_src) is QuaternionFloatingJoint:
            joint_dest = QuaternionFloatingJoint(
                name=joint_src.name(),
                frame_on_parent=frame_on_parent_dest,
                frame_on_child=frame_on_child_dest,
                angular_damping=joint_src.default_angular_damping(),
                translational_damping=(
                    joint_src.default_translational_damping()
                ),
            )
        else:
            assert False, f"Cannot clone: {type(joint_src)}"
        joint_dest.set_position_limits(
            joint_src.position_lower_limits(),
            joint_src.position_upper_limits())
        joint_dest.set_velocity_limits(
            joint_src.velocity_lower_limits(),
            joint_src.velocity_upper_limits())
        joint_dest.set_acceleration_limits(
            joint_src.acceleration_lower_limits(),
            joint_src.acceleration_upper_limits())
        joint_dest.set_default_positions(
            joint_src.default_positions())
        joint_dest = plant_dest.AddJoint(joint_dest)
        _add_item(self.joints, joint_src, joint_dest)

    def copy_joint_actuator(self, joint_actuator_src):
        """Copies a JointActuator to be added to the destination plant."""
        assert type(joint_actuator_src) is JointActuator
        plant_dest = self.plant_dest
        joint_src = joint_actuator_src.joint()
        joint_dest = self.joints[joint_src]
        joint_actuator_dest = plant_dest.AddJointActuator(
            joint_actuator_src.name(), joint_dest,
            effort_limit=joint_actuator_src.effort_limit())
        joint_actuator_dest.set_default_rotor_inertia(
            joint_actuator_src.default_rotor_inertia())
        joint_actuator_dest.set_default_gear_ratio(
            joint_actuator_src.default_gear_ratio())
        _add_item(
            self.joint_actuators,
            joint_actuator_src, joint_actuator_dest)

    def copy_geometry_by_id(self, geometry_id_src):
        """Copes the geometry represented by geometry_id_src (and its roles /
        properties) to be added to the destination plant / scene_graph.

        At present, collision geometries that are to be used for
        simulation must be registered with the plant (rather than
        queried). For this reason, all collision geometries must *only*
        be collision geometries, and have to go through the plant.
        """
        assert isinstance(geometry_id_src, GeometryId)
        plant_src = self.plant_src
        scene_graph_src = self.scene_graph_src
        plant_dest = self.plant_dest
        scene_graph_dest = self.scene_graph_dest
        assert scene_graph_src is not None
        assert scene_graph_dest is not None
        inspector_src = scene_graph_src.model_inspector()
        frame_id_src = inspector_src.GetFrameId(geometry_id_src)
        body_src = plant_src.GetBodyFromFrameId(frame_id_src)
        assert body_src is not None
        body_dest = self.bodies[body_src]
        frame_id_dest = plant_dest.GetBodyFrameIdOrThrow(body_dest.index())
        assert frame_id_dest is not None
        geometry_instance_dest = inspector_src.CloneGeometryInstance(
            geometry_id_src)

        # Use new "scoped" name.
        if body_src.model_instance() != world_model_instance():
            model_instance_name_src = plant_src.GetModelInstanceName(
                body_src.model_instance())
            prefix_src = f"{model_instance_name_src}::"
        else:
            prefix_src = ""
        if body_dest.model_instance() != world_model_instance():
            model_instance_name_dest = plant_dest.GetModelInstanceName(
                body_dest.model_instance())
            prefix_dest = f"{model_instance_name_dest}::"
        else:
            prefix_dest = ""
        scoped_name_src = geometry_instance_dest.name()
        assert scoped_name_src.startswith(prefix_src), (
            f"'{scoped_name_src}' should start with '{prefix_src}'")
        unscoped_name = scoped_name_src[len(prefix_src):]
        scoped_name_dest = f"{prefix_dest}{unscoped_name}"
        geometry_instance_dest.set_name(scoped_name_dest)

        # TODO(eric.cousineau): How to relax this constraint? How can we
        # register with SceneGraph only?
        # See: https://github.com/RobotLocomotion/drake/issues/13445
        # TODO(eric.cousineau): Try Ale's potential fix here:
        # https://github.com/RobotLocomotion/drake/pull/13371
        proximity_properties = (
            geometry_instance_dest.proximity_properties())
        if proximity_properties is not None:
            assert geometry_instance_dest.perception_properties() is None
            assert geometry_instance_dest.illustration_properties() is None
            geometry_id_dest = plant_dest.RegisterCollisionGeometry(
                body=body_dest,
                X_BG=geometry_instance_dest.pose(),
                shape=geometry_instance_dest.shape(),
                name=unscoped_name,
                properties=proximity_properties)
        else:
            # Register as normal.
            source_id_dest = plant_dest.get_source_id()
            geometry_id_dest = scene_graph_dest.RegisterGeometry(
                source_id_dest, frame_id_dest, geometry_instance_dest)
        _add_item(self.geometry_ids, geometry_id_src, geometry_id_dest)

    def copy_state(self, context_src, context_dest):
        """Copies the (physical) state from context_src to context_dest."""
        assert isinstance(context_src, Context)
        assert isinstance(context_dest, Context)
        plant_src = self.plant_src
        plant_dest = self.plant_dest
        for body_src, body_dest in self.bodies.items():
            # TODO(eric.cousineau): Remove this since we now have access to the
            # floating joint coordinates?
            if body_dest.is_floating():
                X_WB = plant_src.CalcRelativeTransform(
                    context_src, plant_src.world_frame(),
                    body_src.body_frame())
                V_WB = plant_src.EvalBodySpatialVelocityInWorld(
                    context_src, body_src)
                plant_dest.SetFreeBodyPose(context_dest, body_dest, X_WB)
                plant_dest.SetFreeBodySpatialVelocity(
                    body_dest, V_WB, context_dest)
        for joint_src, joint_dest in self.joints.items():
            if type(joint_src) is not type(joint_dest):
                # This joint may have been welded (and part of a inverse map).
                # Skip.
                continue
            qj = me.get_joint_positions(plant_src, context_src, joint_src)
            me.set_joint_positions(plant_dest, context_dest, joint_dest, qj)
            vj = me.get_joint_velocities(plant_src, context_src, joint_src)
            me.set_joint_velocities(plant_dest, context_dest, joint_dest, vj)


class SubgraphPolicy:
    """Provides a policy to mutate a subgraph, and then mutate a "destination
    plant" after the subgraph is added to the model."""

    def mutate_subgraph(self, subgraph):
        """Use this to either remove source elements, add in existing source
        elements, etc.

        Returns:
            (elem_added, elem_removed), each are MultibodyPlantElements's.
        """
        raise NotImplementedError()

    def after_add(self, subgraph):
        """Use this to add additional destination elements, e.g. adding
        joints.

        Returns:
            (elem_added, elem_removed), each are MultibodyPlantElements's.
        """
        raise NotImplementedError()


class RemoveFinalizedFloatingJoints(SubgraphPolicy):
    def mutate_subgraph(self, subgraph):
        plant_src = subgraph.plant_src
        elem_added = None
        elem_removed = None
        if not plant_src.is_finalized():
            return elem_added, elem_removed
        elem_removed = subgraph.make_empty_elements()
        elem_src = subgraph.elements_src
        for joint_src in elem_src.joints:
            if type(joint_src) is QuaternionFloatingJoint:
                elem_removed += subgraph.remove_joint(joint_src)
        return elem_added, elem_removed

    def after_add(self, subgraph):
        elem_added = None
        elem_removed = None
        return elem_added, elem_removed


class FreezeJointSubgraphPolicy(SubgraphPolicy):
    """Defines a "policy" on subgraphs to allow a set of joints to be replaced
    with a weld."""

    def __init__(self, freeze_joints_src, context_src):
        # TODO(eric.cousineau): Should also pass in bodies to freeze if they
        # are floating?
        # TODO(eric.cousineau): If the source plant is not yet finalized, an
        # workaround: Create subgraph copy of plant (no geometry), finalize it,
        # and use that plant.
        self._freeze_joints_src = freeze_joints_src
        self._context_src = context_src

    @classmethod
    def from_plant(cls, plant, context_src, model_instances=None):
        # Get all non-weld joints.
        joints_src = me.get_joints(plant, model_instances)
        for joint in list(joints_src):
            if joint.num_positions() == 0:
                joints_src.remove(joint)
        return cls(joints_src, context_src)

    def mutate_subgraph(self, subgraph):
        """Remove joints from subgraph. (They must already be part of the
        subgraph)."""
        elem_added = None
        elem_removed = subgraph.make_empty_elements()
        for joint_src in self._freeze_joints_src:
            elem_removed += subgraph.remove_joint(joint_src)
        return elem_added, elem_removed

    def after_add(self, src_to_dest):
        """Replace joints with welds. plant_src must be finalized, with the
        provided context."""
        elem_added = src_to_dest.make_empty_elements_dest()
        elem_removed = None
        for joint_src in self._freeze_joints_src:
            # Ensure that this joint was removed.
            assert joint_src not in src_to_dest.joints
            # Replicate kinematics, but frozen.
            frame_P_src = joint_src.parent_body().body_frame()
            frame_C_src = joint_src.child_body().body_frame()
            X_PC = src_to_dest.plant_src.CalcRelativeTransform(
                self._context_src, frame_P_src, frame_C_src)
            frame_P_dest = src_to_dest.frames[frame_P_src]
            frame_C_dest = src_to_dest.frames[frame_C_src]
            weld = src_to_dest.plant_dest.WeldFrames(
                frame_P_dest, frame_C_dest, X_PC)
            src_to_dest.joints[joint_src] = weld
            elem_added.joints.add(weld)
        return elem_added, elem_removed


class MultibodyPlantSubgraph:
    # TODO(eric.cousineau): Consider renaming to `MultibodyPlantCopier`?
    # Move subgraph constraints / documentation to the
    # `check_subgraph_invariants` function?
    """
    Defines the subgraph of a *single* source MultibodyPlant (and possibly
    SceneGraph). This subgraph can then be copied onto *any* unfinalized
    destination MultibodyPlant (and possibly SceneGraph), and return the
    associations between the source (MbP, *SG) and a given destination (MbP,
    *SG).

    Note: The abbreviation "(MbP, *SG)" is intended to imply a MultibodyPlant
    and possibly SceneGraph; if the SceneGraph is specified, it *must* be
    registered with the given plant.

    This MultibodyPlantSubgraph only identifies topology; computations
    themselves are only done by (MbP, *SG).

    For more information about workflows, see the module-level docstring.

    Note:
        It does *not* matter if the source plant is finalized or not.

    It is important to note that model instances (ModelInstanceIndex) in Drake
    should not be strictly interpreted as physical things; i.e. some model
    instances could correspond to a mechanical subsystem (e.g. an arm), where
    other model instances could identify a group of disconnected bodies,
    joints, frames, etc. Instead, they're more of a "tagging" mechanism. Given
    that they are part of the public API for MultibodyPlant, and their names
    are an important facet of this, they are incorporated into the subgraph
    (defined below) as an effective "container" for the objects that are
    "tagged" with this instance.

    As used here, the term "subgraph" is purely conceptual. The vertices of the
    graph are (MbP, *SG) elements (e.g., bodies, frames, joints, geometries,
    etc.) and model instances, and graph itself is directed. A directed edge is
    deemed to exist from one vertex A to another vertex B iff A refers to B in
    a way that is required for A to "exist" specific to the MultibodyPlant. For
    example:

    - A model instance I can be in the graph without having any bodies that
      belong to it, but a body B cannot be in the subgraph if model instance I
      is not part of the subgraph -- because I is specifically required for
      defining B.
    - Two bodies B1 and B2 connected by a joint J in the original plant can be
      in the subgraph, regardless of whether J is in it. However, B cannot be
      in the subgraph if either B1 or B2 are not in it.
    - A geometry G *could* be defined in the SceneGraph without being attached
      to any body in the MultibodyPlant. However, for the purposes of this
      class, it is deemed necessary for the geometry G to be associated with
      the body B in order to facilitate association after the geometry is
      copied to a destination plant.

    The (directed) subgraph must connected in a specific way; the connectivity
    is as strict as or more strict than being weakly connected (all vertices
    have a reachable path between each other), but not as strict as being
    unilaterally connected (all vertices are connected to each other). This is
    because some vertices may require multiple connections, e.g. the joint
    example illustrated above.

    The concrete invariants for elements in this subgraph:

    - All subgraph model instances must be part of the subgraph's
      MultibodyPlant.

    - All subgraph bodies must be part of the subgraph model instances.

    - All subgraph frames must be attached to the subgraph bodies.

    - All subgraph joints must be solely connected to the subgraph bodies.

    - All subgraph joint actuators must act solely on the subgraph joints.

    - All subgraph geometries must be attached to the subgraph bodies, and
      must be part of the subgraph's SceneGraph (if specified).

      Geometries have additional (hack-ish) constraints that are not enforced
      as subgraph invariants, but rather when adding the subgraph to a new
      MultibodyPlant (and possibly SceneGraph):

      - The subgraph geometry must use the "scoped name" of
        "{body_model_instance}::{geometry_name}". This is necessary so
        that the copies of geometry can be renamed to the new model instance
        name given as part of the copy operation. Any geometry not using this
        pattern will raise an error.
      - If a geometry has a collision role in addition to other roles, an error
        will be raised. This is due to a constraint in how `MultibodyPlant`
        handles collision geometry (drake#13445).
      - This class only deals with SceneGraph's registered model geometry; it
        does not affect copies of that model in any allocated Contexts.
    """
    # TODO(eric.cousineau): Relax the error constraints for geometry and
    # possibly ignore them.

    def __init__(self, elem, *, remove_finalized_floating_joints=True):
        """
        Arguments:
            elem:
                Elements to track via MultibodyPlantElements.
            remove_finalized_floating_joints:
                If true, will not copy (possibly implicit) floating joints from
                the source plant if it is finalized. This is useful if you want
                to copy from a finalized plant (which may have implicit
                floating joints) to then weld or mobilize the plant.
        """
        assert isinstance(elem, MultibodyPlantElements)
        check_subgraph_invariants(elem)
        self._elem_src = copy.copy(elem)
        self._policies = []
        if remove_finalized_floating_joints:
            self.apply_policy(RemoveFinalizedFloatingJoints())

    def __copy__(self):
        """Defines a shallow "level-2" copy."""
        policies = list(self._policies)
        new = MultibodyPlantSubgraph(elem=self.elements_src)
        # Do not apply these policies to re-mutate the subgrahp.
        new._policies = policies
        return new

    def apply_policy(self, policy):
        """Adds a policy, and mutates this subgraph according to it."""
        assert isinstance(policy, SubgraphPolicy)
        elem_added, elem_removed = policy.mutate_subgraph(self)
        self._policies.append(policy)
        return elem_added, elem_removed

    @property
    def plant_src(self):
        """Returns source MultibodyPlant."""
        return self._elem_src.plant

    @property
    def scene_graph_src(self):
        """Returns source SceneGraph. May be None."""
        return self._elem_src.scene_graph

    @property
    def elements_src(self):
        """Returns a shallow copy of elements contained within subgraph."""
        return copy.copy(self._elem_src)

    def make_empty_elements(self):
        return MultibodyPlantElements(self.plant_src, self.scene_graph_src)

    def add_model_instance(self, model_instance, include_dependents=False):
        raise NotImplemented

    def remove_model_instance(self, model_instance):
        elem_src = self._elem_src
        elem_removed = self.make_empty_elements()
        elem_src.model_instances.remove(model_instance)
        elem_removed.model_instances.add(model_instance)
        for body in me.get_bodies(self.plant_src):
            if body.model_instance() == model_instance:
                elem_removed += self.remove_body(body)
        return elem_removed

    def add_body(self, body, include_dependents=False):
        raise NotImplemented

    def remove_body(self, body, include_welded_bodies=False):
        """
        Removes body and all associated elements from this subgraph.

        Args:
            body: Body to be removed.
            include_welded_bodies: If True, will also remove any bodies in this
                subgraph that are welded to ``body``.
        Returns:
            MultibodyPlantElements containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = self.make_empty_elements()
        elem_src.bodies.remove(body)
        elem_removed.bodies.add(body)
        frames = [
            x for x in elem_src.frames
            if x.body() is body]
        for x in frames:
            elem_removed += self.remove_frame(x)
        joints = [
            x for x in elem_src.joints
            if body in (x.parent_body(), x.child_body())]
        for x in joints:
            elem_removed += self.remove_joint(x)
        if self.scene_graph_src is not None:
            geometry_ids = me.get_geometries(
                self.plant_src, self.scene_graph_src, [body])
            for x in geometry_ids:
                elem_removed += self.remove_geometry_id(x)
        if include_welded_bodies:
            for welded_body in self.plant_src.GetBodiesWeldedTo(body):
                if welded_body in elem_src.bodies:
                    elem_removed += self.remove_body(welded_body)
        return elem_removed

    def add_frame(self, frame, include_dependents=False):
        raise NotImplemented

    def remove_frame(self, frame):
        """Removes a frame from this subgraph.
        Returns:
            MultibodyPlantElements containing all removed elements.
        """
        # N.B. Since most elements interface with bodies, not frames, then it
        # is OK to remove a frame and not really have other dependent elements.
        elem_src = self._elem_src
        elem_removed = self.make_empty_elements()
        elem_src.frames.remove(frame)
        elem_removed.frames.add(frame)
        return elem_removed

    def add_joint(self, joint, include_dependents=False):
        raise NotImplemented

    def remove_joint(self, joint):
        """Removes a joint and its associated actuators from this subgraph.
        Returns:
            MultibodyPlantElements containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = self.make_empty_elements()
        elem_src.joints.remove(joint)
        elem_removed.joints.add(joint)
        joint_actuators = [
            x for x in elem_src.joint_actuators
            if x.joint() is joint]
        for x in joint_actuators:
            elem_removed += self.remove_joint_actuator(x)
        return elem_removed

    def add_joint_actuator(self, joint_actuator):
        raise NotImplemented

    def remove_joint_actuator(self, joint_actuator):
        """Removes a joint actuator from this subgraph.
        Returns:
            MultibodyPlantElements containing all removed elements.
        """
        elem_src = self._elem_src
        elem_removed = self.make_empty_elements()
        elem_src.joint_actuators.remove(joint_actuator)
        elem_removed.joint_actuators.add(joint_actuator)
        return elem_removed

    def add_geometry_id(self, geometry_id):
        raise NotImplemented

    def remove_geometry_id(self, geometry_id):
        """Removes a geometry_id id from this subgraph.
        Returns:
            MultibodyPlantElements containing all removed elements.
        """
        elem_src = self._elem_src
        elem_src.geometry_ids.remove(geometry_id)
        elem_removed = self.make_empty_elements()
        elem_removed.geometry_ids.add(geometry_id)
        return elem_removed

    def add_to(
            self,
            plant_dest,
            scene_graph_dest=None,
            model_instance_remap=model_instance_remap_same_name):
        """Adds this subgraph onto a given plant_dest and scene_graph_dest.

        Args:
            plant_dest: "Destination plant".
            scene_graph_dest: "Destination scene graph".
                If this is specified, there must be a source scene_graph.
            model_instance_remap:
                Either a function of the form, which remaps source model
                instances to a set of destination model instance:

                    func(plant_src, model_instance_src, plant_dest)
                        -> model_instance_dest

                Or a string, which in this function will be used to remap all
                source model instances to a destination model instance of the
                given name (which may be added if it does not already exist).
        Returns:
            src_to_dest (MultibodyPlantElementsMap) used to copy elements and
                record the associations from this subgraph's plant_src (and
                scene_graph_src) to plant_dest (and scene_graph_dest).
        """
        assert isinstance(plant_dest, MultibodyPlant)
        if isinstance(model_instance_remap, str):
            model_instance_name = model_instance_remap
            model_instance_remap = create_model_instance_remap_by_name(
                lambda x: model_instance_name)

        plant_src = self.plant_src
        scene_graph_src = self.scene_graph_src
        elem_src = self._elem_src

        if scene_graph_dest is not None:
            assert scene_graph_src is not None

        # Create mapping.
        src_to_dest = MultibodyPlantElementsMap(
            plant_src,
            plant_dest,
            scene_graph_src=scene_graph_src,
            scene_graph_dest=scene_graph_dest,
        )

        # Remap and register model instances.
        for model_instance_src in me.elements_sorted(elem_src.model_instances):
            model_instance_dest = model_instance_remap(
                plant_src, model_instance_src, plant_dest)
            src_to_dest.register_model_instance(
                model_instance_src, model_instance_dest)

        # Register world body and frame if we're using that source model
        # instance.
        if world_model_instance() in src_to_dest.model_instances:
            src_to_dest.register_world_body_and_frame()

        # Copy bodies.
        for body_src in me.elements_sorted(elem_src.bodies):
            src_to_dest.copy_body(body_src)

        # Copy frames.
        for frame_src in me.elements_sorted(elem_src.frames):
            src_to_dest.copy_frame(frame_src)

        # Copy joints.
        for joint_src in me.elements_sorted(elem_src.joints):
            src_to_dest.copy_joint(joint_src)

        # Copy joint actuators.
        for joint_actuator_src in me.elements_sorted(elem_src.joint_actuators):
            src_to_dest.copy_joint_actuator(joint_actuator_src)

        # Copy geometries (if applicable).
        if scene_graph_dest is not None:
            for geometry_id_src in me.elements_sorted(elem_src.geometry_ids):
                src_to_dest.copy_geometry_by_id(geometry_id_src)

        # Apply policies to new mapping.
        for policy in self._policies:
            policy.after_add(src_to_dest)

        return src_to_dest


def _check_plant_aggregate(get_func, item):
    assert get_func(item.index()) is item, (f"{item}: {item.name()}")


def check_subgraph_invariants(elem):
    # Ensures that current elements / topology satisifes subgraph invariants.
    plant = elem.plant
    scene_graph = elem.scene_graph

    plant_model_instances = me.get_model_instances(plant)
    assert set(elem.model_instances) <= set(plant_model_instances)

    # Check bodies.
    for body in elem.bodies:
        assert isinstance(body, RigidBody)
        _check_plant_aggregate(plant.get_body, body)
        assert body.model_instance() in elem.model_instances

    # Check frames.
    for frame in elem.frames:
        assert isinstance(frame, Frame)
        _check_plant_aggregate(plant.get_frame, frame)
        assert frame.body() in elem.bodies
        assert frame.model_instance() in elem.model_instances

    # Check joints.
    for joint in elem.joints:
        assert isinstance(joint, Joint)
        _check_plant_aggregate(plant.get_joint, joint)
        assert me.is_joint_solely_connected_to(joint, elem.bodies)
        assert joint.model_instance() in elem.model_instances

    # Check actuators.
    for joint_actuator in elem.joint_actuators:
        assert isinstance(joint_actuator, JointActuator), joint_actuator
        _check_plant_aggregate(plant.get_joint_actuator, joint_actuator)
        assert joint_actuator.joint() in elem.joints
        assert joint_actuator.model_instance() in elem.model_instances

    # Check geometries.
    if scene_graph is not None:
        assert plant.geometry_source_is_registered()
        inspector = scene_graph.model_inspector()
        for geometry_id in elem.geometry_ids:
            assert isinstance(geometry_id, GeometryId)
            frame_id = inspector.GetFrameId(geometry_id)
            body = plant.GetBodyFromFrameId(frame_id)
            assert body in elem.bodies
    else:
        assert elem.geometry_ids == set(), elem.geometry_ids


def parse_as_multibody_plant_subgraph(
        model_file, with_scene_graph=True, parse_model_into=None):
    """Parses a MultibodyPlant into subgraph for later use.

    This can be used to "cache" parsing results, and make it so that you an add
    a model multiple times without reparsing it, as well as being able to add
    models to a new scene without reparsing it.

    Args:
        model_file: File to parse.
        with_scene_graph:
            If true, will add a scene_graph via a diagram.
            Otherwise, will use plant directly.
        make_parser:
            How to construct the Parser given the plant. Override this if you
            need package paths.
    Returns:
        subgraph, model_instance
    """
    # TODO(eric.cousineau): Handle adding multiple models?

    if parse_model_into is None:

        def parse_model_into(plant, some_model_file):
            model_instance_index, = Parser(plant).AddModels(some_model_file)
            return model_instance_index

    arbitrary_dt = 0.1
    if with_scene_graph:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=arbitrary_dt)
    else:
        plant = MultibodyPlant(time_step=arbitrary_dt)
        scene_graph = None
    model = parse_model_into(plant, model_file)
    # N.B. Finalize and Build are not necessary.
    subgraph = MultibodyPlantSubgraph(
        get_elements_from_plant(plant, scene_graph))
    return (subgraph, model)


class SubgraphTemplate:
    """Caches a loaded model, and allows it to then be added to other models
    using MultibodyPlantSubgraph."""
    def __init__(self, model_file):
        self._subgraph, self._model = (
            parse_as_multibody_plant_subgraph(model_file))

    def add_to(self, plant, scene_graph, name):
        to_new = self._subgraph.add_to(
            plant, scene_graph, model_instance_remap=name)
        return to_new.model_instances[self._model]


def copy_to_finalized_plant(plant_src):
    """Copies plant_src. This can be used to compute kinematics on a model by
    simply finalizing the copied version.

    Returns:
        plant_dest, src_to_dest
    """
    plant_dest = MultibodyPlant(plant_src.time_step())
    subgraph = MultibodyPlantSubgraph(get_elements_from_plant(plant_src))
    src_to_dest = subgraph.add_to(plant_dest)
    plant_dest.Finalize()
    return plant_dest, src_to_dest


def copy_to_finalized_plant_and_scene_graph(plant_src, scene_graph_src):
    """Copies (plant_src, scene graph) into a new diagram.

    This can be used to copy a plant+scene_graph specifically for using in
    optimization (where you cannot have visualizers, etc. attached).

    Returns:
        plant, scene_graph, src_to_dest, diagram
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, plant_src.time_step())
    subgraph = MultibodyPlantSubgraph(
        get_elements_from_plant(plant_src, scene_graph_src))
    src_to_dest = subgraph.add_to(plant, scene_graph)
    plant.Finalize()
    diagram = builder.Build()
    return plant, scene_graph, src_to_dest, diagram


def add_plant_with_articulated_subset_to(
        plant_src, articulated_models_src, plant_dest,
        scene_graph_src=None, context_src=None, scene_graph_dest=None,
        model_instances_src=None):
    # TODO(eric.cousineau): Is this more complicated than its worth?
    assert articulated_models_src is not None
    if context_src is None:
        context_src = plant_src.CreateDefaultContext()
    if model_instances_src is None:
        model_instances_src = me.get_model_instances(plant_src)
    freeze_models_src = list(
        set(model_instances_src) - set(articulated_models_src))
    subgraph = MultibodyPlantSubgraph(get_elements_from_plant(
        plant_src, scene_graph_src, model_instances_src))
    subgraph.apply_policy(
        FreezeJointSubgraphPolicy.from_plant(
            plant_src, context_src, model_instances=freeze_models_src))
    src_to_dest = subgraph.add_to(
        plant_dest, scene_graph_dest=scene_graph_dest)
    return src_to_dest
