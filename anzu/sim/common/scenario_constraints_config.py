import dataclasses
import logging
import re

import numpy as np

from pydrake.geometry import GeometrySet, Role
from pydrake.multibody.parsing import GetScopedFrameByName
from pydrake.multibody.tree import BodyIndex, DeformableBodyIndex, FrameIndex


def _CollidesAndSelected(forbidden_set: set, ident_A, ident_B,
                         actual_contact_force, contact_force_limit) -> bool:
    """Returns True if both the actual contact force meets or exceeds the
    limit, and if both colliding objects are in the forbidden set.
    """
    # Carefully crafted to allow some callers to use BodyIndex as the
    # comparison universe, and some to use GeometryId. For this to work, all of
    # `forbidden_set`, `ident_A`, and `ident_B` must contain values of the same
    # type.
    logging.debug(f"Found collision between {ident_A} and {ident_B}...")
    if actual_contact_force < contact_force_limit:
        logging.debug("...and ignored it (contact force too low).")
    elif ident_A in forbidden_set and ident_B in forbidden_set:
        logging.debug(
            f"...which is flagged as a collision: F={actual_contact_force}.")
        return True
    else:
        logging.debug("...and ignored it (not selected).")
    return False


def _PointPairCollides(plant, point_pair_contact, bodies: set[BodyIndex],
                       contact_force_limit) -> bool:
    """Returns the result of _CollidesAndSelected() for a collision pair
    reported as point contact."""
    body_A = plant.get_body(point_pair_contact.bodyA_index())
    body_B = plant.get_body(point_pair_contact.bodyB_index())
    actual_contact_force = np.linalg.norm(point_pair_contact.contact_force())
    # Selection will be entirely in body index space.
    return _CollidesAndSelected(bodies, body_A.index(), body_B.index(),
                                actual_contact_force, contact_force_limit)


def _HydroelasticPairCollides(plant, inspector, hydroelastic_contact,
                              bodies: set[BodyIndex], contact_force_limit):
    """Returns the result of _CollidesAndSelected() for a collision pair
    reported as hydroelastic contact."""
    geo_id_M = hydroelastic_contact.contact_surface().id_M()
    geo_id_N = hydroelastic_contact.contact_surface().id_N()
    frame_id_M = inspector.GetFrameId(geo_id_M)
    frame_id_N = inspector.GetFrameId(geo_id_N)
    body_M = plant.GetBodyFromFrameId(frame_id_M)
    body_N = plant.GetBodyFromFrameId(frame_id_N)
    # TODO(dale.mcconachie) This discards the torque elements. Do we care?
    actual_contact_force = np.linalg.norm(
        hydroelastic_contact.F_Ac_W().translational())
    # Selection will be entirely in body index space.
    return _CollidesAndSelected(bodies, body_M.index(), body_N.index(),
                                actual_contact_force, contact_force_limit)


def _DeformablePairCollides(plant, inspector, deformable_contact,
                            rigid_bodies: set[BodyIndex],
                            deformable_bodies: set[DeformableBodyIndex],
                            contact_force_limit) -> bool:
    """Returns the result of _CollidesAndSelected() for a collision pair
    reported as deformable contact."""
    # Selection will be entirely in geometry id space.
    geo_id_A = deformable_contact.id_A()
    geo_id_B = deformable_contact.id_B()
    # Add all the geoms from the deformable bodies to the forbidden set.
    deformable_model = plant.deformable_model()
    forbidden_geoms = set(deformable_model.GetBody(index).geometry_id()
                          for index in deformable_bodies)
    # Add all the geoms from the rigid bodies to the forbidden set.
    forbidden_frames = [plant.GetBodyFrameIdOrThrow(body_index)
                        for body_index in rigid_bodies]
    set_from_frames = GeometrySet(forbidden_frames)
    geoms_from_frames = inspector.GetGeometryIds(set_from_frames,
                                                 Role.kProximity)
    forbidden_geoms |= geoms_from_frames
    # TODO(dale.mcconachie) This discards the torque elements. Do we care?
    actual_contact_force = np.linalg.norm(
        deformable_contact.F_Ac_W().translational())
    return _CollidesAndSelected(forbidden_geoms, geo_id_A,
                                geo_id_B, actual_contact_force,
                                contact_force_limit)


def _Collides(
        plant,
        plant_context,
        rigid_bodies: set[BodyIndex],
        deformable_bodies: set[DeformableBodyIndex],
        contact_force_limit) -> bool:
    """Returns True if any collisions are found where the actual force meets or
    exceeds the `contact_force_limit`, and both bodies in collision are in one
    of the sets described by `rigid_bodies` or `deformable_bodies`.
    """
    query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = query_object.inspector()
    contact_results = plant.get_contact_results_output_port().Eval(
        plant_context)

    num_contacts = (contact_results.num_point_pair_contacts() +
                    contact_results.num_hydroelastic_contacts() +
                    contact_results.num_deformable_contacts())
    if num_contacts == 0:
        logging.debug(
            f"Configuration is collision-free ({len(rigid_bodies)} rigid "
            f"bodies and {len(deformable_bodies)} deformable bodies).")
        return False

    # Point-pair contacts.
    for k in range(contact_results.num_point_pair_contacts()):
        point_pair_contact = contact_results.point_pair_contact_info(k)
        if _PointPairCollides(plant, point_pair_contact, rigid_bodies,
                              contact_force_limit):
            return True

    # Hydroelastic contacts.
    for k in range(contact_results.num_hydroelastic_contacts()):
        hydroelastic_contact = contact_results.hydroelastic_contact_info(k)
        if _HydroelasticPairCollides(plant, inspector, hydroelastic_contact,
                                     rigid_bodies, contact_force_limit):
            return True

    # Deformable contacts.
    for k in range(contact_results.num_deformable_contacts()):
        deformable_contact = contact_results.deformable_contact_info(k)
        if _DeformablePairCollides(plant, inspector, deformable_contact,
                                   rigid_bodies, deformable_bodies,
                                   contact_force_limit):
            return True
    return False


@dataclasses.dataclass(kw_only=True)
class NonCollisionConstraint:
    """Describes a constraint that certain bodies may not be in collision."""

    contact_force_limit: float = 0.0
    """The maximum permitted force between the bodies.  Because Drake bodies
    are almost never in static equilibrium without either a collision or a
    joint limit constraint, you may wish to set this relative to the mass of
    the objects in question.
    """

    forbid_collision_between: list[str] = dataclasses.field(
        default_factory=list)
    """A list of regular expressions matching the scoped names of bodies
    (rigid or deformable) in the plant.

    Note that collision filtering is respected here, so e.g. including all
    bodies of an arm will correctly filter adjacent-link collisions.

    To forbid collisions between models, write them as regular expressions on
    the model namespace; that is: `spatula::.*` instead of `spatula` to match
    bodies of the spatula model.
    """

    def Satisfied(self, *, plant, plant_context) -> bool:
        """Returns `True` iff `plant` at the state of `plant_context` satisfies
        the constraint.
        """
        constrained_body_res = list(self.forbid_collision_between)
        constrained_body_re_matched = [False] * len(constrained_body_res)
        constrained_rigid_bodies = set()
        constrained_deformable_bodies = set()
        # First, find all the rigid bodies that match the regular expressions.
        for body_index in range(plant.num_bodies()):
            name = plant.get_body(
                BodyIndex(body_index)).scoped_name().to_string()
            for re_index, regex in enumerate(constrained_body_res):
                if re.fullmatch(regex, name):
                    constrained_rigid_bodies.add(BodyIndex(body_index))
                    constrained_body_re_matched[re_index] = True
        # Then, find all the deformable bodies that match the regular
        # expressions.
        deformable_model = plant.deformable_model()
        for index in range(deformable_model.num_bodies()):
            body = deformable_model.GetBody(DeformableBodyIndex(index))
            name = body.scoped_name().to_string()
            for re_index, regex in enumerate(constrained_body_res):
                if re.fullmatch(regex, name):
                    constrained_deformable_bodies.add(
                        DeformableBodyIndex(index))
                    constrained_body_re_matched[re_index] = True
        for re_index, matched in enumerate(constrained_body_re_matched):
            if not matched:
                logging.warning(
                    f"Constraint body regular expression"
                    f" '{constrained_body_res[re_index]}'"
                    f" did not match any bodies.")

        return not _Collides(plant,
                             plant_context,
                             constrained_rigid_bodies,
                             constrained_deformable_bodies,
                             self.contact_force_limit)


@dataclasses.dataclass(kw_only=True)
class PointsInRelativeBoxConstraint:
    """Constrain that points in frame A should lie within a box in frame B.
    """

    frame_A_name: str = ""
    """Matches the name of frame_A."""

    p_AP: list[np.ndarray] = dataclasses.field(default_factory=list)
    """The points P in frame A."""

    frame_B_name: str = ""
    """Matches the name of frame_B."""

    box_lo: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3))
    """The lower corner of the box in frame B."""

    box_up: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3))
    """The upper corner of the box in frame B."""

    def Satisfied(self, *, plant, plant_context) -> bool:
        """Returns `True` iff `plant` at the state of `plant_context` satisfies
        the constraint.
        """
        num_pts = len(self.p_AP)
        p_BP = np.zeros((3, num_pts))
        p_AP = np.zeros((3, num_pts))
        for i in range(num_pts):
            p_AP[:, i] = self.p_AP[i]
        p_BP = plant.CalcPointsPositions(
            plant_context, GetScopedFrameByName(plant, self.frame_A_name),
            p_AP, GetScopedFrameByName(plant, self.frame_B_name))
        # Reshape 1d vectors to column vectors.
        box_up = self.box_up.reshape((-1, 1))
        box_lo = self.box_lo.reshape((-1, 1))
        return ((p_BP <= np.tile(box_up, (1, num_pts))).all() and
                (p_BP >= np.tile(box_lo, (1, num_pts))).all())


@dataclasses.dataclass(kw_only=True)
class XYCircle:
    """Describes a circle inscribed on the X-Y plane."""

    center: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(2))
    """The center of the circle."""

    radius: float = 0.0
    """The radius of the circle."""

    # Returns True iff point is within the interior of the circle.
    def InRegion(self, *, point: np.ndarray) -> bool:
        distance_sq = np.linalg.norm(point - self.center)
        radius_sq = self.radius * self.radius
        return distance_sq < radius_sq


@dataclasses.dataclass(kw_only=True)
class XYRectangle:
    """Describes a rectangle inscribed on the X-Y plane."""

    min: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(2))
    """The lower corner of the rectangle."""

    max: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(2))
    """The upper corner of the rectangle."""

    # Returns True iff point is within the interior of the rectangle.
    def InRegion(self, *, point: np.ndarray) -> bool:
        min_satisfied = all(self.min < point)
        max_satisfied = all(point < self.max)
        return min_satisfied and max_satisfied


def _OutOfRegion(plant, plant_context, frame_As: set[FrameIndex], frame_B_name,
                 region_B: XYCircle | XYRectangle) -> bool:
    """Returns True iff the origin of all frame As is outside of the region
    specified by region_B in frame_B.
    """
    frame_B = GetScopedFrameByName(plant, frame_B_name)
    for frame_A_index in frame_As:
        frame_A = plant.get_frame(frame_A_index)
        X_BA = plant.CalcRelativeTransform(plant_context, frame_B, frame_A)
        in_region = region_B.InRegion(point=X_BA.translation()[:2])
        if in_region:
            return False
    return True


@dataclasses.dataclass(kw_only=True)
class FramesNotInXYRegionConstraint:
    """Constrain that the origin of all frames named matching any of the
    regular expressions in frame_A_names must not in the interior of a region
    inscribed on the XY plane of frame B.
    """

    frame_A_names: list[str] = dataclasses.field(default_factory=list)
    """A list of regular expressions matching frame names of frames in the
    plant.
    """

    frame_B_name: str = "world"
    """The name of frame B."""

    region_B: XYCircle | XYRectangle = dataclasses.field(
        default_factory=XYCircle)
    """The region in frame B that must not contain the frame."""

    def Satisfied(self, *, plant, plant_context) -> bool:
        """Returns `True` iff `plant` at the state of `plant_context` satisfies
        the constraint.
        """
        constrained_frame_res = list(self.frame_A_names)
        constrained_frame_re_matched = [False] * len(constrained_frame_res)
        constrained_frames = set()
        for frame_index in range(plant.num_frames()):
            name = plant.get_frame(
                FrameIndex(frame_index)).scoped_name().to_string()
            for re_index, regex in enumerate(constrained_frame_res):
                if (re.fullmatch(regex, name)):
                    constrained_frames.add(FrameIndex(frame_index))
                    constrained_frame_re_matched[re_index] = True
        for re_index, matched in enumerate(constrained_frame_re_matched):
            if not matched:
                logging.warning(
                    f"Constraint frame regular expression"
                    f" '{self.frame_A_names[re_index]}' did not match any"
                    f" frames.")
        return _OutOfRegion(plant,
                            plant_context,
                            constrained_frames,
                            self.frame_B_name,
                            self.region_B)


# N.B. These are not alphabetical so that we preserve legacy behavior where
# NonCollisionConstraint is the default.
ConcreteConstraintVariant = (NonCollisionConstraint |
                             PointsInRelativeBoxConstraint |
                             FramesNotInXYRegionConstraint)


@dataclasses.dataclass(kw_only=True)
class ScenarioConstraints:
    """Specifies validity constraints on a scenario, which can be addressed by
    resampling (rejection sampling) or in the future (TODO(ggould)) by using
    a solver.
    """

    num_sample_attempts: int = 100
    """If rejection sampling is required to meet the constraints, the number
    of sample attempts allowed before declaring the scenario infeasible.
    """

    constraints: dict[str, ConcreteConstraintVariant] = dataclasses.field(
        default_factory=dict)
    """Any number of named constraints."""

    def Satisfied(self, *, plant, plant_context) -> bool:
        """Returns `True` iff `plant` at the state of `plant_context` satisfies
        the constraints described by `constraints`.
        """
        result = True
        for constraint in self.constraints.values():
            # TODO(dale.mcconachie,ggould) Reorder this loop so that
            # constraints that are cheaper to evaluate happen before more
            # expensive ones (non-collision).
            result = (result and
                      constraint.Satisfied(plant=plant,
                                           plant_context=plant_context))
        return result
