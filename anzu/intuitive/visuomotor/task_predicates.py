from collections import defaultdict
import logging
import math
import re
import typing

import numpy as np

from pydrake.multibody.parsing import GetScopedFrameByName
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (
    DeformableBody,
    Frame,
    FrameIndex,
    ModelInstanceIndex,
    RigidBody,
    ScopedName,
)
from pydrake.systems.framework import Context, Diagram

import anzu.intuitive.common.math_safe as math_safe
from anzu.sim.common.item_locking_monitor_config_functions import (
    ItemLockingMonitor,
    get_contact_pairs_scoped_names,
)


def _get_contact_pairs(
    plant, item_locking_monitor, root_context
) -> list[tuple[ScopedName, ScopedName]]:
    """Returns the list of all bodies pairs that are in contact in the format
    of a list of tuples of two ScopedNames, one for each body in contact. This
    includes both bodies that are active and bodies that are sleeping.
    """
    # Extract the contact pairs that have not been filtered out due to item
    # locking.
    plant_context = plant.GetMyContextFromRoot(root_context)
    active_contact_pairs = get_contact_pairs_scoped_names(plant, plant_context)
    # Extract any memorized sleeping contact pairs if item locking is enabled.
    sleeping_contact_pairs = []
    if item_locking_monitor is not None:
        sleeping_contact_pairs = (
            item_locking_monitor.get_colliding_bodies_when_fell_asleep()
        )
    # Return the combination of the two.
    return active_contact_pairs + sleeping_contact_pairs


def _get_model_instances_by_regexp(
    plant: MultibodyPlant, model_name_regexp: str
) -> list[ModelInstanceIndex]:
    match_model_instances = []
    for model_instance_index in range(plant.num_model_instances()):
        index = ModelInstanceIndex(model_instance_index)
        fullmatch = re.fullmatch(
            model_name_regexp, plant.GetModelInstanceName(index)
        )
        if fullmatch is not None:
            match_model_instances.append(index)
    # Warn the user if the regular expression is not matched.
    if len(match_model_instances) == 0:
        logging.getLogger("drake").warn(
            "Model instance name regular expression"
            f"'{model_name_regexp}' did not match any model instances."
        )
    return match_model_instances


# TODO(dale.mcconachie) De-duplicate this.
def _get_frames_by_regexp(
    plant: MultibodyPlant, frame_name_regexp: str
) -> list[Frame]:
    if frame_name_regexp == "world":
        return [plant.world_frame()]
    match_frames = []
    for frame_index in range(plant.num_frames()):
        frame = plant.get_frame(FrameIndex(frame_index))
        frame_name = frame.scoped_name().get_full()
        fullmatch = re.fullmatch(frame_name_regexp, frame_name)
        if fullmatch is not None:
            match_frames.append(frame)
    # Warn the user if the regular expression is not matched.
    if len(match_frames) == 0:
        logging.getLogger("drake").warn(
            "Frame name regular expression "
            f"'{frame_name_regexp}' did not match any frames."
        )
    return match_frames


def _get_model_instance_by_scoped_name(
    plant, scoped_name
) -> ModelInstanceIndex:
    model_name = scoped_name.get_namespace()
    return plant.GetModelInstanceByName(model_name)


class TaskPredicate:
    """Base class for all task predicates; used to store common data elements
    and define the basic API.
    """

    def __init__(
        self,
        name: str,
        diagram: Diagram,
        item_locking_monitor: ItemLockingMonitor | None,
    ):
        # The `diagram` must be the root Diagram.
        if not name:
            raise ValueError("`name` must not be empty")
        self._name = name
        self._diagram = diagram

        # Try some known locations for the plant, and throw otherwise.
        sim_diagram = diagram
        if self._diagram.HasSubsystemNamed("station"):
            sim_diagram = diagram.GetSubsystemByName("station")
        self._plant = sim_diagram.GetSubsystemByName("plant")
        self._item_locking_monitor = item_locking_monitor
        if self._item_locking_monitor is not None:
            assert isinstance(self._item_locking_monitor, ItemLockingMonitor)

    @property
    def name(self):
        return self._name

    def eval(self, root_context: Context):
        return self.eval_detailed(root_context)[self.name]

    def eval_detailed(self, root_context: Context) -> dict:
        """Evaluates the task predicate function as well as any
        intermediate/sub-components of the predicate.

        Args:
            root_context: The Context for the diagram passed in to the
                constructor.
        Returns:
            A dictionary of named predicate components and values for those
            components. Must contain at least one predicate with `self.name` as
            the key and a boolean value as the item.
        """
        raise NotImplementedError()


# TODO(dale.mcconachie) Duplicated from `rewards.py`
def _raise_if_name_overlap(evaluators: list[TaskPredicate]):
    names = [x.name for x in evaluators]
    overlap = [x for x in names if names.count(x) > 1]
    if overlap:
        raise RuntimeError(f"Overlap in evaluator names: {overlap}")


# TODO(dale.mcconachie) Duplicated from `rewards.py`
def _raise_if_key_overlap(dict1: dict, dict2: dict):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    overlap = keys1.intersection(keys2)
    if overlap:
        raise RuntimeError(f"Overlap in dict keys: {overlap}")


def _get_rigid_or_deformable_body(
    plant: MultibodyPlant, body_scoped_name: ScopedName
) -> tuple[bool, DeformableBody | None, RigidBody | None]:
    """Helper method to return a body as either deformable or rigid based on
    its scoped name.

    Returns:
        Tuple of (is_deformable, deformable_body, rigid_body) where
        `is_deformable` is a boolean, and `deformable_body` and
        `rigid_body` are either None or the corresponding body objects.
    """
    body_instance = plant.GetModelInstanceByName(
        body_scoped_name.get_namespace()
    )
    deformable_model = plant.deformable_model()
    is_deformable = deformable_model.HasBodyNamed(
        body_scoped_name.get_element(), body_instance
    )

    if is_deformable:
        deformable_body = deformable_model.GetBodyByName(
            body_scoped_name.get_element(), body_instance
        )
        rigid_body = None
    else:
        rigid_body = plant.GetBodyByName(
            body_scoped_name.get_element(), body_instance
        )
        deformable_body = None

    return is_deformable, deformable_body, rigid_body


def _check_deformable_body_velocity_bounds(
    deformable_body: DeformableBody,
    plant_context: Context,
    translational_velocity_bound: float,
    angular_velocity_bound: float,
    result_prefix: str,
) -> dict:
    """Helper function to check deformable body velocity bounds. The
    translational velocity is measured at the center of mass of the body in the
    world frame. The angular velocity is an "effective" angular velocity for
    deformable bodies, which measures how fast the body is rotating about an
    axis passing through its center of mass. See
    drake::multibody::DeformableBody::CalcEffectiveAngularVelocity().

    Args:
        deformable_body: The deformable body to check.
        plant_context: The plant context.
        translational_velocity_bound: Maximum allowed translational velocity
            magnitude in meters per second. Must be non-negative.
        angular_velocity_bound: Maximum allowed "effective" angular velocity
            magnitude in radians per second. Must be non-negative.
        result_prefix: Prefix for the result dictionary keys.

    Returns:
        Dictionary with velocity values and satisfaction flags.
    """
    v_translational = (
        deformable_body.CalcCenterOfMassTranslationalVelocityInWorld(
            plant_context
        )
    )
    w_rotational = deformable_body.CalcEffectiveAngularVelocity(plant_context)

    v_bound_satisfied = bool(
        v_translational.dot(v_translational)
        <= translational_velocity_bound**2
    )
    w_bound_satisfied = bool(
        w_rotational.dot(w_rotational) <= angular_velocity_bound**2
    )

    return {
        f"{result_prefix}.v_WD": v_translational,
        f"{result_prefix}.w_WD": w_rotational,
        f"{result_prefix}.v_WD_bound_satisfied": v_bound_satisfied,
        f"{result_prefix}.w_WD_bound_satisfied": w_bound_satisfied,
    }


class AlwaysSatisfiedPredicate(TaskPredicate):
    def eval_detailed(self, root_context: Context) -> dict:
        return {self.name: True}


class NeverSatisfiedPredicate(TaskPredicate):
    def eval_detailed(self, root_context: Context) -> dict:
        return {self.name: False}


class AtLeastOneSatisfiedPredicate(TaskPredicate):
    def __init__(self, predicates: list[TaskPredicate], **kwargs):
        super().__init__(**kwargs)
        _raise_if_name_overlap(predicates)
        self._predicates = predicates

    def eval_detailed(self, root_context: Context) -> dict:
        satisfied = False
        aggregate_details = dict()
        for predicate in self._predicates:
            detailed_result = predicate.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_result)
            aggregate_details.update(detailed_result)
            satisfied |= detailed_result[predicate.name]
        aggregate_details[self.name] = satisfied
        return aggregate_details


class ExactlyOneSatisfiedPredicate(TaskPredicate):
    def __init__(self, predicates: list[TaskPredicate], **kwargs):
        super().__init__(**kwargs)
        _raise_if_name_overlap(predicates)
        self._predicates = predicates

    def eval_detailed(self, root_context: Context) -> dict:
        individual_satisfied = list()
        aggregate_details = dict()
        for predicate in self._predicates:
            detailed_result = predicate.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_result)
            aggregate_details.update(detailed_result)
            individual_satisfied.append(detailed_result[predicate.name])
        satisfied = sum(individual_satisfied) == 1
        aggregate_details[self.name] = satisfied
        return aggregate_details


class AllSatisfiedPredicate(TaskPredicate):
    def __init__(self, predicates: list[TaskPredicate], **kwargs):
        super().__init__(**kwargs)
        _raise_if_name_overlap(predicates)
        self._predicates = predicates

    def eval_detailed(self, root_context: Context) -> dict:
        satisfied = True
        aggregate_details = dict()
        for predicate in self._predicates:
            detailed_result = predicate.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_result)
            aggregate_details.update(detailed_result)
            satisfied &= detailed_result[predicate.name]
        aggregate_details[self.name] = satisfied
        return aggregate_details


class NoneSatisfiedPredicate(TaskPredicate):
    def __init__(self, predicates: list[TaskPredicate], **kwargs):
        super().__init__(**kwargs)
        _raise_if_name_overlap(predicates)
        self._predicates = predicates

    def eval_detailed(self, root_context: Context) -> dict:
        satisfied = True
        aggregate_details = dict()
        for predicate in self._predicates:
            detailed_result = predicate.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_result)
            aggregate_details.update(detailed_result)
            satisfied &= not detailed_result[predicate.name]
        aggregate_details[self.name] = satisfied
        return aggregate_details


class RelativeXYDistancePredicate(TaskPredicate):
    """Defines a predicate based on the translation distance between two
    frames (frame A and frame B), using only the X-Y plane of frame M to
    measure distance. The distance used is relative to points specified in
    frame A and frame B.
    """

    def __init__(
        self,
        frame_A_name: str,
        frame_B_name: str,
        frame_M_name: str,
        dist_threshold: float,
        p_AGa: np.ndarray,
        p_BGb: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert p_AGa.shape == (3,), p_AGa
        assert p_BGb.shape == (3,), p_BGb
        self._frame_A = GetScopedFrameByName(self._plant, frame_A_name)
        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        self._frame_M = GetScopedFrameByName(self._plant, frame_M_name)
        self._dist_threshold = dist_threshold
        self._p_AGa = p_AGa
        self._p_BGb = p_BGb

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_MA = self._plant.CalcRelativeTransform(
            plant_context, self._frame_M, self._frame_A
        )
        X_MB = self._plant.CalcRelativeTransform(
            plant_context, self._frame_M, self._frame_B
        )
        p_MGa = X_MA.multiply(self._p_AGa)
        p_MGb = X_MB.multiply(self._p_BGb)
        distance = np.linalg.norm(p_MGa[0:2] - p_MGb[0:2])
        satisfied = distance < self._dist_threshold
        return {
            self.name: satisfied,
            f"{self.name}.dist": distance,
        }


class RelativeTranslationDistancePredicate(TaskPredicate):
    """Defines a predicate based on the translation distance between two
    frames. The distance used is relative to points specified in those frames.
    """

    def __init__(
        self,
        frame_a_name: str,
        frame_b_name: str,
        dist_threshold: float,
        p_AGa: np.ndarray,
        p_BGb: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert p_AGa.shape == (3,), p_AGa
        assert p_BGb.shape == (3,), p_BGb
        self._frame_a = GetScopedFrameByName(self._plant, frame_a_name)
        self._frame_b = GetScopedFrameByName(self._plant, frame_b_name)
        self._dist_threshold = dist_threshold
        self._p_AGa = p_AGa
        self._p_BGb = p_BGb

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_AB = self._plant.CalcRelativeTransform(
            plant_context, self._frame_a, self._frame_b
        )
        p_AGb = X_AB.multiply(self._p_BGb)
        distance = np.linalg.norm(self._p_AGa - p_AGb)
        satisfied = distance < self._dist_threshold
        return {
            self.name: satisfied,
            f"{self.name}.dist": distance,
        }


class RelativeRotationDistancePredicate(TaskPredicate):
    def __init__(
        self,
        frame_a_name: str,
        frame_b_name: str,
        threshold_rad: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_a = GetScopedFrameByName(self._plant, frame_a_name)
        self._frame_b = GetScopedFrameByName(self._plant, frame_b_name)
        self._threshold_rad = threshold_rad

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        R_AB = self._plant.CalcRelativeTransform(
            plant_context, self._frame_a, self._frame_b
        ).rotation()
        # Compute the angle derived from Rodrigues' formula
        # R = I + sinθ*K + (1−cosθ)*K².
        alignment = (R_AB.matrix().trace() - 1.0) / 2.0
        angle_error_rad = math_safe.acos_safe(alignment)
        satisfied = angle_error_rad < self._threshold_rad
        return {
            self.name: satisfied,
            f"{self.name}.angle_error_rad": angle_error_rad,
        }


class RelativePoseDistancePredicate(TaskPredicate):
    def __init__(
        self,
        frame_a_name: str,
        frame_b_name: str,
        p_AGa: np.ndarray,
        p_BGb: np.ndarray,
        translation_dist_threshold: float,
        rotation_threshold_rad: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        translation = RelativeTranslationDistancePredicate(
            name=f"{self.name}.translation",
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            frame_a_name=frame_a_name,
            frame_b_name=frame_b_name,
            dist_threshold=translation_dist_threshold,
            p_AGa=p_AGa,
            p_BGb=p_BGb,
        )
        rotation = RelativeRotationDistancePredicate(
            name=f"{self.name}.rotation",
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            frame_a_name=frame_a_name,
            frame_b_name=frame_b_name,
            threshold_rad=rotation_threshold_rad,
        )
        self._evaluator_internal = AllSatisfiedPredicate(
            name=self.name,
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=[translation, rotation],
        )

    def eval_detailed(self, root_context: Context) -> dict:
        return self._evaluator_internal.eval_detailed(root_context)


# TODO(dale.mcconachie) Update this to use VelocityBoundPredicate.
class OnTopPredicate(TaskPredicate):
    def __init__(
        self,
        top_object_name: str,
        top_object_frame_name: str,
        bottom_object_name: str,
        bottom_object_frame_name: str,
        p_TGt: np.ndarray,
        p_BGb: np.ndarray,
        z_threshold: float,
        xy_threshold: float,
        require_support: bool,
        require_no_fingers_touching: bool,
        top_object_velocity_bound: typing.Optional[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert p_TGt.shape == (3,), p_TGt
        assert p_BGb.shape == (3,), p_BGb
        self._top_object_name = top_object_name
        self._top_object_instance = self._plant.GetModelInstanceByName(
            top_object_name
        )
        self._top_frame = self._plant.GetFrameByName(
            top_object_frame_name, self._top_object_instance
        )
        self._bottom_object_name = bottom_object_name
        self._bottom_object_instance = self._plant.GetModelInstanceByName(
            bottom_object_name
        )
        self._bottom_frame = self._plant.GetFrameByName(
            bottom_object_frame_name, self._bottom_object_instance
        )
        self._p_TGt = p_TGt
        self._p_BGb = p_BGb
        self._xy_threshold = xy_threshold
        self._z_threshold = z_threshold
        self._require_support = require_support
        self._require_no_fingers_touching = require_no_fingers_touching
        self._top_object_velocity_bound = top_object_velocity_bound

    def _check_distance(self, root_context: Context):
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_WT = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), self._top_frame
        )
        X_WB = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), self._bottom_frame
        )
        p_WGt = X_WT.multiply(self._p_TGt)
        p_WGb = X_WB.multiply(self._p_BGb)
        error = p_WGt - p_WGb
        xy_dist = np.linalg.norm(error[:2])
        z_dist = error[2]
        xy_satisfied = xy_dist < self._xy_threshold
        # TODO(dale.mcconachie) Do we want to keep the "z_dist is positive"
        # constraint? Small penetrations can cause this to be false if surface
        # frames are used.
        z_satisfied = z_dist >= 0.0 and z_dist < self._z_threshold

        return {
            f"{self.name}.xy_dist": xy_dist,
            f"{self.name}.xy_satisfied": xy_satisfied,
            f"{self.name}.z_dist": z_dist,
            f"{self.name}.z_satisfied": z_satisfied,
        }

    def _check_bodies_in_contact(
        self, contact_pairs: list[tuple[ScopedName, ScopedName]]
    ):
        bodies_in_contact = False
        fingers_in_contact = False

        for body_A, body_B in contact_pairs:
            if (
                body_A.get_namespace() == self._top_object_name
                and body_B.get_namespace() == self._bottom_object_name
            ):
                bodies_in_contact = True
            if (
                body_B.get_namespace() == self._top_object_name
                and body_A.get_namespace() == self._bottom_object_name
            ):
                bodies_in_contact = True

            if (
                body_A.get_namespace() == self._top_object_name
                and "finger" in body_B.get_element()
            ):
                fingers_in_contact = True
            if (
                body_B.get_namespace() == self._top_object_name
                and "finger" in body_A.get_element()
            ):
                fingers_in_contact = True

        return bodies_in_contact, fingers_in_contact

    def _check_top_object_velocity_bound(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        v_WT = self._top_frame.CalcSpatialVelocityInWorld(plant_context)
        # TODO(dale.mcconachie) Use limit^2 instead of sqrt().
        # We cast to bool to convert the np.bool_ to a straight bool.
        velocity_bound_satisfied = bool(
            np.sqrt(
                v_WT.rotational().dot(v_WT.rotational())
                + v_WT.translational().dot(v_WT.translational())
            )
            <= self._top_object_velocity_bound
        )
        return {
            # TODO(dale.mcconachie) Don't embed a SpatialVelocity object here.
            f"{self.name}.top_object_velocity": v_WT,
            f"{self.name}.top_object_velocity_bound_satisfied": velocity_bound_satisfied,  # noqa
        }

    def eval_detailed(self, root_context: Context) -> dict:
        distance_results = self._check_distance(root_context)
        xy_satisfied = distance_results[f"{self.name}.xy_satisfied"]
        z_satisfied = distance_results[f"{self.name}.z_satisfied"]

        contact_pairs = _get_contact_pairs(
            self._plant, self._item_locking_monitor, root_context
        )
        bodies_in_contact, fingers_in_contact = self._check_bodies_in_contact(
            contact_pairs
        )

        satisfied = xy_satisfied & z_satisfied
        if self._require_support:
            satisfied &= bodies_in_contact
        if self._require_no_fingers_touching:
            satisfied &= not fingers_in_contact

        if self._top_object_velocity_bound is not None:
            top_object_velocity_result = self._check_top_object_velocity_bound(
                root_context
            )
            satisfied &= top_object_velocity_result[
                f"{self.name}.top_object_velocity_bound_satisfied"
            ]
        else:
            top_object_velocity_result = {}

        return {
            self.name: satisfied,
            **distance_results,
            **top_object_velocity_result,
            f"{self.name}.bodies_in_contact": bodies_in_contact,
            f"{self.name}.fingers_in_contact": fingers_in_contact,
        }


class OnTopFrameListsPredicate(TaskPredicate):
    def __init__(
        self,
        top_object_name: str,
        top_object_frame_name_list: list[str],
        bottom_object_name: str,
        bottom_object_frame_name_list: list[str],
        p_TGt_list: list[np.ndarray],
        p_BGb_list: list[np.ndarray],
        z_threshold: float,
        xy_threshold: float,
        require_support: bool,
        require_no_fingers_touching: bool,
        top_object_velocity_bound: typing.Optional[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        for p_TGt in p_TGt_list:
            assert p_TGt.shape == (3,), p_TGt
        for p_BGb in p_BGb_list:
            assert p_BGb.shape == (3,), p_BGb
        assert len(top_object_frame_name_list) == len(p_TGt_list)
        assert len(set(top_object_frame_name_list)) == len(
            top_object_frame_name_list
        ), "The items in top_object_frame_name_list must be unique"
        assert len(bottom_object_frame_name_list) == len(p_BGb_list)
        assert len(set(bottom_object_frame_name_list)) == len(
            bottom_object_frame_name_list
        ), "The items in bottom_object_frame_name_list must be unique"

        self._evaluators = []
        for top_object_frame_name, p_TGt in zip(
            top_object_frame_name_list, p_TGt_list
        ):
            for bottom_object_frame_name, p_BGb in zip(
                bottom_object_frame_name_list, p_BGb_list
            ):
                name = (
                    f"{self.name}.{top_object_frame_name}"
                    f".{bottom_object_frame_name}"
                )
                self._evaluators.append(
                    OnTopPredicate(
                        name=name,
                        diagram=self._diagram,
                        item_locking_monitor=self._item_locking_monitor,
                        top_object_name=top_object_name,
                        bottom_object_name=bottom_object_name,
                        z_threshold=z_threshold,
                        xy_threshold=xy_threshold,
                        require_support=require_support,
                        require_no_fingers_touching=require_no_fingers_touching,  # noqa
                        top_object_frame_name=top_object_frame_name,
                        bottom_object_frame_name=bottom_object_frame_name,
                        p_TGt=p_TGt,
                        p_BGb=p_BGb,
                        top_object_velocity_bound=top_object_velocity_bound,
                    )
                )

        self._require_support = require_support
        self._require_no_fingers_touching = require_no_fingers_touching

    def eval_detailed(self, root_context: Context) -> dict:
        # Note that we explicitly evaluate each evaluator explicitly rather
        # than rely on AllSatisfiedPredicate in order to avoid checking contact
        # results multiple times.

        # Check distance.
        aggregate_distance_results = dict()
        distance_satisfied = False
        for evaluator in self._evaluators:
            distance_results = evaluator._check_distance(root_context)
            # We don't have to check for overlapping names due to the way we
            # construct the evaluators - we are guaranteed to have unique
            # evaluator names.
            aggregate_distance_results.update(distance_results)
            xy_satisfied = distance_results[f"{evaluator.name}.xy_satisfied"]
            z_satisfied = distance_results[f"{evaluator.name}.z_satisfied"]
            if xy_satisfied and z_satisfied:
                distance_satisfied = True

        # Check contacts. Note that it doesn't matter which evaluator we use
        # because all evaluators are using just the object names, not the frame
        # names or points for this check.
        contact_pairs = _get_contact_pairs(
            self._plant, self._item_locking_monitor, root_context
        )
        evaluator = self._evaluators[0]
        (
            bodies_in_contact,
            fingers_in_contact,
        ) = evaluator._check_bodies_in_contact(contact_pairs)

        satisfied = distance_satisfied
        if self._require_support:
            satisfied &= bodies_in_contact
        if self._require_no_fingers_touching:
            satisfied &= not fingers_in_contact

        return {
            self.name: satisfied,
            f"{self.name}.distance_satisfied": distance_satisfied,
            **aggregate_distance_results,
            f"{self.name}.bodies_in_contact": bodies_in_contact,
            f"{self.name}.fingers_in_contact": fingers_in_contact,
        }


class OnlyMakeContactWithPredicate(TaskPredicate):
    """
    Body A is making contact with only body B (but B can make contact with
    other bodies).
    Optionally, the predicate also requires that the velocity V_PA (body A's
    spatial velocity measured and expressed in frame P) is below a certain
    threshold.

    Note that if body A is deformable, we require the frame P to be the world
    frame if it is supplied.
    """

    def __init__(
        self,
        body_A_scoped_name: str,
        body_B_scoped_name: str,
        frame_P_name: typing.Optional[str],
        v_PA_bound: typing.Optional[float],
        w_PA_bound: typing.Optional[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._body_A_scoped_name = ScopedName.Parse(body_A_scoped_name)
        self._body_B_scoped_name = ScopedName.Parse(body_B_scoped_name)

        # Setup body A (deformable or rigid)
        (
            self._body_A_is_deformable,
            self._deformable_body_A,
            self._rigid_body_A,
        ) = _get_rigid_or_deformable_body(
            self._plant, self._body_A_scoped_name
        )

        # Setup body B (deformable or rigid)
        (
            self._body_B_is_deformable,
            self._deformable_body_B,
            self._rigid_body_B,
        ) = _get_rigid_or_deformable_body(
            self._plant, self._body_B_scoped_name
        )

        self._frame_P_name = frame_P_name
        assert (
            (frame_P_name is None)
            == (v_PA_bound is None)
            == (w_PA_bound is None)
        )
        if frame_P_name is not None:
            self._frame_P = GetScopedFrameByName(self._plant, frame_P_name)
            assert v_PA_bound > 0, v_PA_bound
            assert w_PA_bound > 0, w_PA_bound
            self._requires_V_PA_bound = True
            # We require P to be the world frame if the body is deformable.
            if self._body_A_is_deformable:
                assert self._frame_P.name() == "world", (
                    "Frame P must be the world frame if the body is "
                    f"deformable: {self._frame_P.name()}"
                )
        else:
            self._frame_P = None
            self._requires_V_PA_bound = False
        self._v_PA_bound = v_PA_bound
        self._w_PA_bound = w_PA_bound

    def _check_only_make_contact(
        self, root_context: Context
    ) -> typing.Tuple[bool, set[str]]:
        """
        Returns if body A is only in contact with body B, and all the bodies in
        contact with body A.
        """
        contact_pairs = _get_contact_pairs(
            self._plant, self._item_locking_monitor, root_context
        )
        body_A_contact_bodies = set()
        for body1, body2 in contact_pairs:
            if body1.get_full() == self._body_A_scoped_name.get_full():
                body_A_contact_bodies.add(body2.get_full())
            elif body2.get_full() == self._body_A_scoped_name.get_full():
                body_A_contact_bodies.add(body1.get_full())
        body_A_only_contact_body_B = len(body_A_contact_bodies) == 1 & all(
            [
                body == self._body_B_scoped_name.get_full()
                for body in body_A_contact_bodies
            ]
        )
        return body_A_only_contact_body_B, body_A_contact_bodies

    def _check_body_A_velocity_bound(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)

        if self._body_A_is_deformable:
            result = _check_deformable_body_velocity_bounds(
                self._deformable_body_A,
                plant_context,
                self._v_PA_bound,
                self._w_PA_bound,
                self.name,
            )

            # Rename keys from WD to PA
            renames = {
                "w_WD": "w_PA",
                "v_WD": "v_PA",
                "v_WD_bound_satisfied": "v_PA_bound_satisfied",
                "w_WD_bound_satisfied": "w_PA_bound_satisfied",
            }

            renamed_result = {}
            for old_key, new_key in renames.items():
                old_full_key = f"{self.name}.{old_key}"
                new_full_key = f"{self.name}.{new_key}"
                renamed_result[new_full_key] = result[old_full_key]

            return renamed_result
        else:
            V_PA = self._rigid_body_A.body_frame().CalcSpatialVelocity(
                plant_context, self._frame_P, self._frame_P
            )
            v_PA_bound_satisfied = bool(
                V_PA.translational().dot(V_PA.translational())
                <= self._v_PA_bound**2
            )
            w_PA_bound_satisfied = bool(
                V_PA.rotational().dot(V_PA.rotational())
                <= self._w_PA_bound**2
            )
            return {
                f"{self.name}.v_PA": V_PA.translational(),
                f"{self.name}.w_PA": V_PA.rotational(),
                f"{self.name}.v_PA_bound_satisfied": v_PA_bound_satisfied,
                f"{self.name}.w_PA_bound_satisfied": w_PA_bound_satisfied,
            }

    def eval_detailed(self, root_context: Context) -> dict:
        (
            body_A_only_contact_body_B,
            body_A_contact_bodies,
        ) = self._check_only_make_contact(root_context)
        satisfied = body_A_only_contact_body_B
        if self._requires_V_PA_bound:
            velocity_bound_result = self._check_body_A_velocity_bound(
                root_context
            )
            satisfied &= (
                velocity_bound_result[f"{self.name}.v_PA_bound_satisfied"]
                & velocity_bound_result[f"{self.name}.w_PA_bound_satisfied"]
            )
        else:
            velocity_bound_result = {}
        return {
            self.name: satisfied,
            **velocity_bound_result,
            # YAML doesn't really do sets, so convert to a canonical list.
            f"{self.name}.body_A_contact_bodies": sorted(
                list(body_A_contact_bodies)
            ),
        }


class PointsInRelativeBoxPredicate(TaskPredicate):
    """
    Point Pᵢ attached to frame A is within a box p_BP_lo <= p_BP <= p_BP_up
    specified in frame B.
    """

    def __init__(
        self,
        frame_A_name: str,
        p_AP: np.ndarray,
        frame_B_name: str,
        p_BP_lo: np.ndarray,
        p_BP_hi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_A = GetScopedFrameByName(self._plant, frame_A_name)
        assert p_AP.shape[0] == 3
        self._p_AP = p_AP
        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        assert p_BP_lo.shape == (3,)
        assert p_BP_hi.shape == (3,)
        assert np.all(p_BP_lo <= p_BP_hi)
        self._p_BP_lo = p_BP_lo
        self._p_BP_hi = p_BP_hi

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        p_BP = self._plant.CalcPointsPositions(
            plant_context, self._frame_A, self._p_AP, self._frame_B
        )
        error = (
            np.maximum(p_BP.T - self._p_BP_hi, np.zeros_like(p_BP.T)).sum()
            + np.maximum(self._p_BP_lo - p_BP.T, np.zeros_like(p_BP.T)).sum()
        ).item()
        satisfied = bool(error == 0)
        return {
            self.name: satisfied,
            f"{self.name}.error": error,
            f"{self.name}.p_BP": p_BP,
        }


class FramesInRelativeBoxPredicate(TaskPredicate):
    """
    The origin of the frames specified by frame_A_names is within a box
    p_BP_lo <= p_BP <= p_BP_up specified in frame B.

    frame_A_names is a list of regular expressions of scoped frame names.
    """

    def __init__(
        self,
        frame_A_names: list[str],
        frame_B_name: str,
        p_BP_lo: np.ndarray,
        p_BP_hi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Extract all frames that we will be tracking.
        frame_As = []
        for frame_A_name in frame_A_names:
            frame_As.extend(_get_frames_by_regexp(self._plant, frame_A_name))
        assert len(frame_As) > 0, f"Empty frames list: {frame_A_names}"
        # Build evaluators for each individual frame.
        evaluators = []
        for frame_A in frame_As:
            evaluator_name = f"{self.name}.{frame_A.scoped_name()}"
            evaluators.append(
                PointsInRelativeBoxPredicate(
                    name=evaluator_name,
                    diagram=self._diagram,
                    item_locking_monitor=self._item_locking_monitor,
                    frame_A_name=frame_A.scoped_name().get_full(),
                    p_AP=np.array([[0, 0, 0]]).T,
                    frame_B_name=frame_B_name,
                    p_BP_lo=p_BP_lo,
                    p_BP_hi=p_BP_hi,
                )
            )
        # Compile all the evaluators together into the AND of all individual
        # evaluators.
        self._evaluator = AllSatisfiedPredicate(
            name=self.name,
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=evaluators,
        )

    def eval_detailed(self, root_context: Context) -> dict:
        return self._evaluator.eval_detailed(root_context)


class FramesNotInRelativeBoxPredicate(TaskPredicate):
    """
    The origin of the frames specified by frame_A_names is NOT within a box
    p_BP_lo <= p_BP <= p_BP_up specified in frame B.

    frame_A_names is a list of regular expressions of scoped frame names.
    """

    def __init__(
        self,
        frame_A_names: list[str],
        frame_B_name: str,
        p_BP_lo: np.ndarray,
        p_BP_hi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Extract all frames that we will be tracking.
        frame_As = []
        for frame_A_name in frame_A_names:
            frame_As.extend(_get_frames_by_regexp(self._plant, frame_A_name))
        assert len(frame_As) > 0, f"Empty frames list: {frame_A_names}"
        # Build evaluators for each individual frame.
        evaluators = []
        for frame_A in frame_As:
            evaluator_name = f"{self.name}.{frame_A.scoped_name()}"
            evaluators.append(
                PointsInRelativeBoxPredicate(
                    name=evaluator_name,
                    diagram=self._diagram,
                    item_locking_monitor=self._item_locking_monitor,
                    frame_A_name=frame_A.scoped_name().get_full(),
                    p_AP=np.array([[0, 0, 0]]).T,
                    frame_B_name=frame_B_name,
                    p_BP_lo=p_BP_lo,
                    p_BP_hi=p_BP_hi,
                )
            )

        # Compile all the evaluators together into the AND NOT of all
        # individual evaluators.
        self._evaluator = NoneSatisfiedPredicate(
            name=self.name,
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=evaluators,
        )

    def eval_detailed(self, root_context: Context) -> dict:
        return self._evaluator.eval_detailed(root_context)


class AxisPointsInDirectionPredicate(TaskPredicate):
    def __init__(
        self,
        frame_name: str,
        axis_F: np.ndarray,
        direction_W: np.ndarray,
        threshold_rad: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame = GetScopedFrameByName(self._plant, frame_name)
        # TODO(dale.mcconachie) Consider relaxing this and normalizing
        # internally.
        assert math.isclose(np.linalg.norm(axis_F), 1.0)
        assert math.isclose(np.linalg.norm(direction_W), 1.0)
        self._axis_F = axis_F
        self._direction_W = direction_W
        self._threshold_rad = threshold_rad

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_WF = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), self._frame
        )
        axis_W = X_WF.rotation() @ self._axis_F
        alignment = np.dot(self._direction_W, axis_W)
        angle_error_rad = math_safe.acos_safe(alignment)
        satisfied = angle_error_rad < self._threshold_rad
        return {
            f"{self.name}.angle_error_rad": angle_error_rad,
            self.name: satisfied,
        }


class AxisPointsInRelativeDirectionPredicate(TaskPredicate):
    def __init__(
        self,
        frame_A_name: str,
        axis_A: np.ndarray,
        frame_B_name: str,
        axis_B: np.ndarray,
        threshold_rad: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_A = GetScopedFrameByName(self._plant, frame_A_name)
        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        # TODO(hongkai.dai) Consider relaxing this and normalizing
        # internally.
        assert math.isclose(np.linalg.norm(axis_A), 1.0)
        assert math.isclose(np.linalg.norm(axis_B), 1.0)
        self._axis_A = axis_A
        self._axis_B = axis_B
        self._threshold_rad = threshold_rad

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        n_B = (
            self._plant.CalcRelativeTransform(
                plant_context, self._frame_B, self._frame_A
            ).rotation()
            @ self._axis_A
        )
        alignment = np.dot(self._axis_B, n_B)
        angle_error_rad = math_safe.acos_safe(alignment)
        satisfied = angle_error_rad < self._threshold_rad
        return {
            f"{self.name}.angle_error_rad": angle_error_rad,
            self.name: satisfied,
        }


class FingersContactPredicate(TaskPredicate):
    """
    A predicate that checks if robot fingers are in contact with the specified
    object.
    """

    def __init__(
        self,
        body_name: str,
        expect_contact: bool,
        finger_namespace: typing.Optional[str],
        **kwargs,
    ):
        """
        Constructs a FingersContactPredicate.

        Args:
        body_name: The name of the body to check contact with. The namespace
            portion of the scoped name of the body is expected here.
        expect_contact: Whether the finger is expected to be in contact with
            the object.
        finger_namespace: The namespace of the finger used to specify which
            finger is being checked. If None, *any* finger can be used to check
            for contact.
        """
        super().__init__(**kwargs)
        self._body_name = body_name
        self._expect_contact = expect_contact
        self._finger_namespace = finger_namespace

    def _finger_match(self, finger_body_name: ScopedName) -> bool:
        found = True
        if (
            self._finger_namespace is not None
            and finger_body_name.get_namespace() != self._finger_namespace
        ):
            found = False
        if "finger" not in finger_body_name.get_element():
            found = False
        return found

    def eval_detailed(self, root_context: Context) -> dict:
        contact_pairs = _get_contact_pairs(
            self._plant, self._item_locking_monitor, root_context
        )
        # TODO(hongkai): should clean up the logic here.
        fingers_in_contact = False
        for body_A, body_B in contact_pairs:
            if (
                body_A.get_namespace() == self._body_name
                and self._finger_match(body_B)
            ):
                fingers_in_contact = True
            if (
                body_B.get_namespace() == self._body_name
                and self._finger_match(body_A)
            ):
                fingers_in_contact = True

        satisfied = fingers_in_contact == self._expect_contact

        return {
            f"{self.name}.contact": fingers_in_contact,
            self.name: satisfied,
        }


class FingersContactBodiesPredicate(TaskPredicate):
    """
    The contact state between the fingers and all specified bodies match with
    the expected value.
    """

    def __init__(
        self,
        body_names: list[str],
        expect_contact: bool,
        finger_namespace: typing.Optional[str],
        **kwargs,
    ):
        """
        Constructs a FingersContactBodiesPredicate.

        Args:
            body_names: The names of the bodies to check contact with. The
                namespace portion of the scoped name of the body is expected
                here.
            expect_contact: Whether the fingers are expected to be in contact
                with the bodies.
            finger_namespace: The namespace of the finger used to specify which
                finger is being checked. If None, *any* finger can be used to
                check for contact.
        """
        super().__init__(**kwargs)
        evaluators = []
        for body_name in body_names:
            model_instance_names = [
                self._plant.GetModelInstanceName(instance)
                for instance in _get_model_instances_by_regexp(
                    self._plant, body_name
                )
            ]
            for model_instance_name in model_instance_names:
                evaluator_name = f"{self.name}.{model_instance_name}"
                evaluators.append(
                    FingersContactPredicate(
                        name=evaluator_name,
                        diagram=self._diagram,
                        item_locking_monitor=self._item_locking_monitor,
                        body_name=model_instance_name,
                        expect_contact=expect_contact,
                        finger_namespace=finger_namespace,
                    )
                )
        self._evaluator = AllSatisfiedPredicate(
            name=self.name,
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=evaluators,
        )

    def eval_detailed(self, root_context) -> dict:
        return self._evaluator.eval_detailed(root_context)


class ContactFlagPredicate(TaskPredicate):
    """
    A predicate that checks if existence of contacts between objects matches
    the expectation.
    """

    def __init__(
        self,
        body_1_name: str,
        body_2_name: str,
        bodies_in_contact_expected: bool,
        fingers_in_contact_1_expected: typing.Optional[bool],
        fingers_in_contact_2_expected: typing.Optional[bool],
        **kwargs,
    ):
        """
        Constructs a ContactFlagPredicate.

        Args:
            body_1_name: The name of the first body to check contact with. The
                namespace portion of the scoped name of the body is expected
                here.
            body_2_name: The name of the second body to check contact with. The
                namespace portion of the scoped name of the body is expected
                here.
            bodies_in_contact_expected: Whether the bodies 1 and 2 are
                expected to be in contact.
            fingers_in_contact_1_expected: Whether the body 1 is expected to
                be in contact with any finger.
            fingers_in_contact_2_expected: Whether the body 2 is expected to be
                in contact with any finger.
        """
        super().__init__(**kwargs)
        self._body_1_name = body_1_name
        self._body_2_name = body_2_name
        self._bodies_in_contact_expected = bodies_in_contact_expected
        self._fingers_in_contact_1_expected = fingers_in_contact_1_expected
        self._fingers_in_contact_2_expected = fingers_in_contact_2_expected

    def _check_bodies_in_contact(
        self, contact_pairs: list[tuple[ScopedName, ScopedName]]
    ):
        bodies_in_contact = False
        fingers_in_contact_1 = False
        fingers_in_contact_2 = False

        for body_A, body_B in contact_pairs:
            if (
                body_A.get_namespace() == self._body_1_name
                and body_B.get_namespace() == self._body_2_name
            ):
                bodies_in_contact = True
            if (
                body_B.get_namespace() == self._body_1_name
                and body_A.get_namespace() == self._body_2_name
            ):
                bodies_in_contact = True

            if (
                body_A.get_namespace() == self._body_1_name
                and "finger" in body_B.get_element()
            ):
                fingers_in_contact_1 = True
            if (
                body_B.get_namespace() == self._body_1_name
                and "finger" in body_A.get_element()
            ):
                fingers_in_contact_1 = True

            if (
                body_A.get_namespace() == self._body_2_name
                and "finger" in body_B.get_element()
            ):
                fingers_in_contact_2 = True
            if (
                body_B.get_namespace() == self._body_2_name
                and "finger" in body_A.get_element()
            ):
                fingers_in_contact_2 = True

        return bodies_in_contact, fingers_in_contact_1, fingers_in_contact_2

    def eval_detailed(self, root_context: Context) -> dict:
        contact_pairs = _get_contact_pairs(
            self._plant,
            self._item_locking_monitor,
            root_context,
        )
        (
            bodies_in_contact,
            fingers_in_contact_1,
            fingers_in_contact_2,
        ) = self._check_bodies_in_contact(contact_pairs)
        satisfied = bodies_in_contact == self._bodies_in_contact_expected
        if self._fingers_in_contact_1_expected is not None:
            satisfied &= (
                fingers_in_contact_1 == self._fingers_in_contact_1_expected
            )
        if self._fingers_in_contact_2_expected is not None:
            satisfied &= (
                fingers_in_contact_2 == self._fingers_in_contact_2_expected
            )

        return {
            self.name: satisfied,
            f"{self.name}.bodies_in_contact": bodies_in_contact,
            f"{self.name}.fingers_in_contact_1": fingers_in_contact_1,
            f"{self.name}.fingers_in_contact_2": fingers_in_contact_2,
        }


class ExclusiveContactPredicate(TaskPredicate):
    """
    For each body body_A (model instance) in set A, the set of bodies (model
    instances) that body_A is in contact with, is a subset of set B. And each
    body body_A (model instance) in set A must be in contact with at least one
    body in in set B.
    (But B can be in contact with other bodies).
    """

    def __init__(
        self,
        body_A_names: list[str],
        body_B_names: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._models_A: set[ModelInstanceIndex] = set()
        self._models_B: set[ModelInstanceIndex] = set()

        for body_A_name in body_A_names:
            self._models_A.update(
                _get_model_instances_by_regexp(self._plant, body_A_name)
            )
        for body_B_name in body_B_names:
            self._models_B.update(
                _get_model_instances_by_regexp(self._plant, body_B_name)
            )

    def eval_detailed(self, root_context: Context) -> dict:
        contact_pairs = _get_contact_pairs(
            self._plant, self._item_locking_monitor, root_context
        )

        # Maps the body in self._models_A to the set of bodies that makes
        # contact with the body.
        MAPPING_TYPE = dict[ModelInstanceIndex, set[ModelInstanceIndex]]
        models_in_contact_with_A: MAPPING_TYPE = defaultdict(set)
        for body_M, body_N in contact_pairs:
            index_M = _get_model_instance_by_scoped_name(self._plant, body_M)
            index_N = _get_model_instance_by_scoped_name(self._plant, body_N)
            if index_M in self._models_A:
                models_in_contact_with_A[index_M].add(index_N)
            if index_N in self._models_A:
                models_in_contact_with_A[index_N].add(index_M)

        # Now that we have a unique set of contacts for each body in set A,
        # check that we are in contact with something that we are supposed to
        # be in contact with, and not in contact with everything else.
        result = {}
        all_satisfied = True
        for model_A in self._models_A:
            contacts = models_in_contact_with_A[model_A]
            valid_contact = not contacts.isdisjoint(self._models_B)
            invalid_contact = not contacts.issubset(self._models_B)
            satisfied = valid_contact and not invalid_contact

            model_name = self._plant.GetModelInstanceName(model_A)
            name = f"{self.name}.{model_name}"
            # YAML doesn't know about sets, so convert to a canonical list.
            result[f"{name}.contacts"] = sorted(
                [self._plant.GetModelInstanceName(index) for index in contacts]
            )
            result[f"{name}.valid_contact"] = valid_contact
            result[f"{name}.invalid_contact"] = invalid_contact
            result[name] = satisfied
            all_satisfied &= satisfied
        result[self.name] = all_satisfied
        return result


class MugOnBranchPredicate(TaskPredicate):
    def __init__(
        self,
        branch_frame_name: str,
        mug_handle_frame_names: list[str],
        mug_center_frame_name: str,
        branch_length: float,
        handle_length: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # One of the frames origin on the mug handle is above the branch, and
        # the mug center is below the branch.

        # The x axis of the branch frame always points perpendicular from the
        # branch and goes downward.
        p_BP_lo = np.array([-0.02, -handle_length / 2, 0])
        p_BP_hi = np.array([0.0, handle_length / 2, branch_length * 0.9])
        self._handle_above_branch = AtLeastOneSatisfiedPredicate(
            name=f"{self.name}.handle_above_branch",
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=[
                PointsInRelativeBoxPredicate(
                    name=f"{self.name}.{handle_frame_name}_above_branch",
                    diagram=self._diagram,
                    item_locking_monitor=self._item_locking_monitor,
                    frame_A_name=handle_frame_name,
                    p_AP=np.zeros((3, 1)),
                    frame_B_name=branch_frame_name,
                    p_BP_lo=p_BP_lo,
                    p_BP_hi=p_BP_hi,
                )
                for handle_frame_name in mug_handle_frame_names
            ],
        )
        self._mug_center_below_branch = PointsInRelativeBoxPredicate(
            name=f"{self.name}.mug_center_below_branch",
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            frame_A_name=mug_center_frame_name,
            p_AP=np.zeros((3, 1)),
            frame_B_name=branch_frame_name,
            p_BP_lo=np.array([0.01, -handle_length, 0]),
            p_BP_hi=np.array([0.1, handle_length, branch_length * 0.9]),
        )

    def eval_detailed(self, root_context: Context) -> dict:
        handle_above_branch_ret = self._handle_above_branch.eval_detailed(
            root_context
        )
        mug_center_below_branch_ret = (
            self._mug_center_below_branch.eval_detailed(root_context)
        )
        satisfied = (
            handle_above_branch_ret[self._handle_above_branch.name]
            and mug_center_below_branch_ret[self._mug_center_below_branch.name]
        )
        ret = {self.name: satisfied}
        ret.update(handle_above_branch_ret)
        ret.update(mug_center_below_branch_ret)

        return ret


class MugOnMugHolderPredicate(TaskPredicate):
    """
    The mug hangs on one of the mug holder.
    """

    def __init__(
        self,
        branch_frame_names: list[str],
        mug_handle_frame_names: list[str],
        mug_center_frame_name: str,
        branch_length: float,
        handle_length: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._mug_above_branch = AtLeastOneSatisfiedPredicate(
            name=f"{self.name}.above_branch",
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=[
                MugOnBranchPredicate(
                    name=f"{self.name}.above_{branch_frame_name}",
                    diagram=self._diagram,
                    item_locking_monitor=self._item_locking_monitor,
                    branch_frame_name=branch_frame_name,
                    mug_handle_frame_names=mug_handle_frame_names,
                    mug_center_frame_name=mug_center_frame_name,
                    branch_length=branch_length,
                    handle_length=handle_length,
                )
                for branch_frame_name in branch_frame_names
            ],
        )

    def eval_detailed(self, root_context: Context) -> dict:
        ret = self._mug_above_branch.eval_detailed(root_context)
        satisfied = ret[self._mug_above_branch.name]
        ret[self._name] = satisfied
        return ret


class OneToOneOnTopPredicate(TaskPredicate):
    """
    Given N top objects and N bottom objects, each top_object is on top of one
    bottom_object.
    """

    def __init__(
        self,
        top_object_names: list[str],
        top_frame_names: list[str],
        bottom_object_names: list[str],
        bottom_frame_names: list[str],
        p_TGt: np.ndarray,
        p_BGb: np.ndarray,
        xy_threshold: float,
        z_threshold: float,
        require_support: bool,
        require_no_fingers_touching: bool,
        top_object_velocity_bound: typing.Optional[float],
        **kwargs,
    ):
        """
        The point p_TGt on the frame top_frame_names[i] on the object
        top_object_names[i] is within xy_threshold to the point p_BGb on the
        frame bottom_frame_names[j] on the object bottom_object_names[j],
        within z_threshold.
        """
        super().__init__(**kwargs)
        assert (
            len(top_object_names)
            == len(top_frame_names)
            == len(bottom_object_names)
            == len(bottom_frame_names)
        )
        self.num_objects = len(top_object_names)
        self._top_on_bottom_predicates = [
            [
                OnTopPredicate(
                    name=f"{self._name}.{top_object_names[i]}"
                    f"_on_{bottom_object_names[j]}",
                    diagram=self._diagram,
                    item_locking_monitor=self._item_locking_monitor,
                    top_object_name=top_object_names[i],
                    top_object_frame_name=top_frame_names[i],
                    bottom_object_name=bottom_object_names[j],
                    bottom_object_frame_name=bottom_frame_names[j],
                    p_TGt=p_TGt,
                    p_BGb=p_BGb,
                    z_threshold=z_threshold,
                    xy_threshold=xy_threshold,
                    require_support=require_support,
                    require_no_fingers_touching=require_no_fingers_touching,
                    top_object_velocity_bound=top_object_velocity_bound,
                )
                for j in range(self.num_objects)
            ]
            for i in range(self.num_objects)
        ]
        _raise_if_name_overlap(
            [
                self._top_on_bottom_predicates[i][j]
                for i in range(self.num_objects)
                for j in range(self.num_objects)
            ]
        )

    def eval_detailed(self, root_context: Context) -> dict:
        aggregate_details = dict()
        predicate_results = [
            [
                self._top_on_bottom_predicates[i][j].eval_detailed(
                    root_context
                )
                for j in range(self.num_objects)
            ]
            for i in range(self.num_objects)
        ]
        satisfied_matrix = np.array(
            [
                [
                    predicate_results[i][j][
                        self._top_on_bottom_predicates[i][j].name
                    ]
                    for j in range(self.num_objects)
                ]
                for i in range(self.num_objects)
            ]
        )
        satisfied = (satisfied_matrix.sum(axis=0) == 1).all() and (
            satisfied_matrix.sum(axis=1) == 1
        ).all()
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                aggregate_details.update(predicate_results[i][j])
        aggregate_details[self.name] = satisfied
        return aggregate_details


class VelocityBoundPredicate(TaskPredicate):
    """
    The velocity v_BP (point P's velocity measured in frame B) is within a
    bound.
    """

    def __init__(
        self,
        frame_A_name: str,
        p_AP: np.ndarray,
        frame_B_name: str,
        velocity_bound: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_A = GetScopedFrameByName(self._plant, frame_A_name)
        self._p_AP = p_AP
        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        assert velocity_bound >= 0, f"velocity_bound={velocity_bound}"
        self._velocity_bound = velocity_bound

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        v_BP = self._frame_A.CalcSpatialVelocity(
            plant_context, self._frame_B, self._frame_B
        )
        # TODO(dale.mcconachie) Use limit^2 instead of sqrt().
        # We cast to bool to convert the np.bool_ to a straight bool.
        satisfied = bool(
            np.sqrt(
                v_BP.rotational().dot(v_BP.rotational())
                + v_BP.translational().dot(v_BP.translational())
            )
            <= self._velocity_bound
        )
        return {
            # TODO(dale.mcconachie) Don't embed a SpatialVelocity object here.
            f"{self.name}.v_BP": v_BP,
            self.name: satisfied,
        }


class VelocitiesBoundsPredicate(TaskPredicate):
    """
    v_BP is within the bound. Note that for every pair of frame (A, B), we
    compute its v_BP, and require each v_BP is within the bound.
    """

    def __init__(
        self,
        frame_A_names: list[str],
        p_AP: np.ndarray,
        frame_B_names: list[str],
        velocity_bound: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_A_list = []
        self._frame_B_list = []
        for frame_A_name in frame_A_names:
            self._frame_A_list.extend(
                _get_frames_by_regexp(self._plant, frame_A_name)
            )
        for frame_B_name in frame_B_names:
            self._frame_B_list.extend(
                _get_frames_by_regexp(self._plant, frame_B_name)
            )

        evaluators = []
        for frame_A in self._frame_A_list:
            for frame_B in self._frame_B_list:
                frame_A_name = frame_A.scoped_name().get_full()
                frame_B_name = frame_B.scoped_name().get_full()
                evaluator_name = f"{self.name}.{frame_A_name}_{frame_B_name}"
                evaluators.append(
                    VelocityBoundPredicate(
                        name=evaluator_name,
                        diagram=self._diagram,
                        item_locking_monitor=self._item_locking_monitor,
                        frame_A_name=frame_A_name,
                        p_AP=p_AP,
                        frame_B_name=frame_B_name,
                        velocity_bound=velocity_bound,
                    )
                )
        self._evaluator = AllSatisfiedPredicate(
            name=self.name,
            diagram=self._diagram,
            item_locking_monitor=self._item_locking_monitor,
            predicates=evaluators,
        )

    def eval_detailed(self, root_context) -> dict:
        return self._evaluator.eval_detailed(root_context)


class DeformableVelocityBoundPredicate(TaskPredicate):
    """
    The translational and angular velocity of the deformable body are within
    bounds.
    """

    def __init__(
        self,
        body_scoped_name: str,
        translational_velocity_bound: float,
        angular_velocity_bound: float,
        **kwargs,
    ):
        """Initializes a predicate that checks if a deformable body's
         velocities are within bounds.

        Args:
            body_scoped_name: The scoped name of the deformable body.
            translational_velocity_bound: The maximum allowed translational
                velocity magnitude in meters per second. Must be non-negative.
                The velocity is measured at the center of mass of the body in
                the world frame.
            angular_velocity_bound: The maximum allowed "effective" angular
                velocity magnitude in radians per second. Must be non-negative.
                The "effective" angular velocity measures how fast the body is
                rotating about an axis passing through its center of mass.
        """
        super().__init__(**kwargs)
        scoped_name = ScopedName.Parse(body_scoped_name)
        model_instance = self._plant.GetModelInstanceByName(
            scoped_name.get_namespace()
        )
        self._deformable_body = self._plant.deformable_model().GetBodyByName(
            scoped_name.get_element(), model_instance
        )
        assert (
            translational_velocity_bound >= 0
        ), f"translational_velocity_bound={translational_velocity_bound}"
        assert (
            angular_velocity_bound >= 0
        ), f"angular_velocity_bound={angular_velocity_bound}"
        self._translational_velocity_bound = translational_velocity_bound
        self._angular_velocity_bound = angular_velocity_bound

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)

        # Use the shared helper function for deformable body velocity bounds.
        velocity_results = _check_deformable_body_velocity_bounds(
            self._deformable_body,
            plant_context,
            self._translational_velocity_bound,
            self._angular_velocity_bound,
            self.name,
        )

        # Extract the satisfaction flags and compute overall satisfaction.
        satisfied = bool(
            velocity_results[f"{self.name}.v_WD_bound_satisfied"]
            and velocity_results[f"{self.name}.w_WD_bound_satisfied"]
        )

        # Update the result keys to match the existing pattern (v_WD, w_WD).
        return {
            self.name: satisfied,
            f"{self.name}.v_WD": velocity_results[f"{self.name}.v_WD"],
            f"{self.name}.w_WD": velocity_results[f"{self.name}.w_WD"],
        }


class CenterOfMassInRelativeBoxPredicate(TaskPredicate):
    """
    The center of mass of a body A (rigid or deformable) is within a box
    specified in the B frame.
    """

    def __init__(
        self,
        body_A_scoped_name: str,
        frame_B_name: str,
        p_BAcm_lo: np.ndarray,
        p_BAcm_hi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._body_A_scoped_name = ScopedName.Parse(body_A_scoped_name)
        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        assert p_BAcm_lo.shape == (3,)
        assert p_BAcm_hi.shape == (3,)
        assert np.all(p_BAcm_lo <= p_BAcm_hi)
        self._p_BAcm_lo = p_BAcm_lo
        self._p_BAcm_hi = p_BAcm_hi

        # Check if the body is a deformable body.
        (
            self._is_deformable,
            self._deformable_body,
            self._rigid_body,
        ) = _get_rigid_or_deformable_body(
            self._plant, self._body_A_scoped_name
        )

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)

        if self._is_deformable:
            p_WAcm = self._deformable_body.CalcCenterOfMassPositionInWorld(
                plant_context
            )
        else:
            p_AoAcm = self._rigid_body.CalcCenterOfMassInBodyFrame(
                plant_context
            )
            # Convert from p_AoAcm_A to p_WAcm_W
            X_WA = self._plant.EvalBodyPoseInWorld(
                plant_context, self._rigid_body
            )
            p_WAcm = X_WA @ p_AoAcm
        p_BAcm = self._plant.CalcPointsPositions(
            plant_context, self._plant.world_frame(), p_WAcm, self._frame_B
        )
        error = (
            np.maximum(
                p_BAcm.T - self._p_BAcm_hi, np.zeros_like(p_BAcm.T)
            ).sum()
            + np.maximum(
                self._p_BAcm_lo - p_BAcm.T, np.zeros_like(p_BAcm.T)
            ).sum()
        ).item()
        satisfied = bool(error == 0)
        return {
            self.name: satisfied,
            f"{self.name}.error": error,
            f"{self.name}.p_BAcm": p_BAcm,
        }
