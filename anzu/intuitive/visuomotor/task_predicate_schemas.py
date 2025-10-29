import dataclasses as dc
import typing

import numpy as np

from pydrake.systems.framework import Diagram

from anzu.intuitive.visuomotor.task_predicates import (
    AllSatisfiedPredicate,
    AlwaysSatisfiedPredicate,
    AtLeastOneSatisfiedPredicate,
    AxisPointsInDirectionPredicate,
    AxisPointsInRelativeDirectionPredicate,
    CenterOfMassInRelativeBoxPredicate,
    ContactFlagPredicate,
    DeformableVelocityBoundPredicate,
    ExactlyOneSatisfiedPredicate,
    ExclusiveContactPredicate,
    FingersContactBodiesPredicate,
    FingersContactPredicate,
    FramesInRelativeBoxPredicate,
    FramesNotInRelativeBoxPredicate,
    MugOnBranchPredicate,
    MugOnMugHolderPredicate,
    NeverSatisfiedPredicate,
    OnTopFrameListsPredicate,
    OnTopPredicate,
    OneToOneOnTopPredicate,
    OnlyMakeContactWithPredicate,
    PointsInRelativeBoxPredicate,
    RelativePoseDistancePredicate,
    RelativeRotationDistancePredicate,
    RelativeTranslationDistancePredicate,
    RelativeXYDistancePredicate,
    VelocitiesBoundsPredicate,
    VelocityBoundPredicate,
)
from anzu.sim.common.item_locking_monitor_config_functions import (
    ItemLockingMonitor,
)

# TODO(dale.mcconachie) Convert documentation comments to python docstring
# format.


@dc.dataclass
class PredicateConfig:
    name: str = ""

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        raise NotImplementedError()


def _to_rad(deg: float) -> float:
    return deg / 180.0 * np.pi


@dc.dataclass
class AlwaysSatisfiedPredicateConfig(PredicateConfig):
    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return AlwaysSatisfiedPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
        )


@dc.dataclass
class NeverSatisfiedPredicateConfig(PredicateConfig):
    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return NeverSatisfiedPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
        )


@dc.dataclass
class RelativeXYDistancePredicateConfig(PredicateConfig):
    # Frames to measure the distance between.
    frame_A_name: str = ""
    frame_B_name: str = ""
    # Frame that defines the X-Y plane.
    frame_M_name: str = "world"
    dist_threshold: float = 0.0
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_AGa: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return RelativeXYDistancePredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_name=self.frame_A_name,
            frame_B_name=self.frame_B_name,
            frame_M_name=self.frame_M_name,
            dist_threshold=self.dist_threshold,
            p_AGa=self.p_AGa,
            p_BGb=self.p_BGb,
        )


@dc.dataclass
class RelativeTranslationDistancePredicateConfig(PredicateConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""
    dist_threshold: float = 0.0
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_AGa: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return RelativeTranslationDistancePredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
            dist_threshold=self.dist_threshold,
            p_AGa=self.p_AGa,
            p_BGb=self.p_BGb,
        )


@dc.dataclass
class RelativeRotationDistancePredicateConfig(PredicateConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""
    # The rotation deviation threshold measured in degrees.
    threshold_deg: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return RelativeRotationDistancePredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
            threshold_rad=_to_rad(self.threshold_deg),
        )


@dc.dataclass
class RelativePoseDistancePredicateConfig(PredicateConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_AGa: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. Note that this may be zero
    # in many circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    translation_dist_threshold: float = 0.0
    # The rotation deviation threshold measured in degrees.
    rotation_threshold_deg: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return RelativePoseDistancePredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
            p_AGa=self.p_AGa,
            p_BGb=self.p_BGb,
            translation_dist_threshold=self.translation_dist_threshold,
            rotation_threshold_rad=_to_rad(self.rotation_threshold_deg),
        )


@dc.dataclass
class OnTopPredicateConfig(PredicateConfig):
    # Model instance name for the object that should be on top.
    top_object_name: str = ""
    # Name of the frame associated with `top_object_name`. Note that this
    # should not be a scoped name, just the direct name of the frame in
    # question.
    top_object_frame_name: str = ""
    # Model instance name for the object that should be on the bottom.
    bottom_object_name: str = ""
    # Name of the frame associated with `bottom_object_name`. Note that this
    # should not be a scoped name, just the direct name of the frame in
    # question.
    bottom_object_frame_name: str = ""
    # Offset in top object frame (T) for the goal position. I.e.; the point
    # relative to frame T that should be aligned with frame B. Note that this
    # may be zero in many circumstances. This can be done by adding frames in
    # the model files/directives so that we keep all similar information in one
    # logical place in a scenario definition file.
    p_TGt: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in bottom object frame (B) for the goal position. I.e.; the point
    # relative to frame B that should be aligned with frame T. Note that this
    # may be zero in many circumstances. This can be done by adding frames in
    # the model files/directives so that we keep all similar information in one
    # logical place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    xy_threshold: float = 0.0
    z_threshold: float = 0.0
    require_support: bool = True
    require_no_fingers_touching: bool = True
    top_object_velocity_bound: typing.Optional[float] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return OnTopPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            top_object_name=self.top_object_name,
            top_object_frame_name=self.top_object_frame_name,
            bottom_object_name=self.bottom_object_name,
            bottom_object_frame_name=self.bottom_object_frame_name,
            xy_threshold=self.xy_threshold,
            z_threshold=self.z_threshold,
            require_support=self.require_support,
            require_no_fingers_touching=self.require_no_fingers_touching,
            p_TGt=self.p_TGt,
            p_BGb=self.p_BGb,
            top_object_velocity_bound=self.top_object_velocity_bound,
        )


@dc.dataclass
class OnTopFrameListsPredicateConfig(PredicateConfig):
    # Model instance name for the object that should be on top.
    top_object_name: str = ""
    # List of frame names associated with `top_object_name`. Note that these
    # should not be scoped names, just the direct name of the frames in
    # question.
    top_object_frame_name_list: list[str] = dc.field(default_factory=list)
    # Model instance name for the object that should be on the bottom.
    bottom_object_name: str = ""
    # List of frame names associated with `bottom_object_name`. Note that these
    # should not be scoped names, just the direct name of the frames in
    # question.
    bottom_object_frame_name_list: list[str] = dc.field(default_factory=list)
    # Offset in top object frame (T) for the goal position. I.e.; the point
    # relative to frame T that should be aligned with frame B. Note that this
    # may be zero in many circumstances. This can be done by adding frames in
    # the model files/directives so that we keep all similar information in one
    # logical place in a scenario definition file. If this is specified it will
    # be used for all frames and p_TGt_list must be None.
    p_TGt: typing.Optional[np.ndarray] = None
    # A list of the same length as top_object_frame_name_list. If this is
    # unspecified then p_TGt will be used for all frames. If this is specified
    # then p_TGt must be None.
    p_TGt_list: typing.Optional[list[np.ndarray]] = None
    # Offset in bottom object frame (B) for the goal position. I.e.; the point
    # relative to frame B that should be aligned with frame T. Note that this
    # may be zero in many circumstances. This can be done by adding frames in
    # the model files/directives so that we keep all similar information in one
    # logical place in a scenario definition file.
    p_BGb: typing.Optional[np.ndarray] = None
    # A list of the same length as frame_b_name_list. If this is unspecified
    # then p_BGb will be used for all frames. If this is specified then p_BGb
    # must be None.
    p_BGb_list: typing.Optional[list[np.ndarray]] = None
    xy_threshold: float = 0.0
    z_threshold: float = 0.0
    require_support: bool = True
    require_no_fingers_touching: bool = True
    top_object_velocity_bound: typing.Optional[float] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        if self.p_TGt is not None:
            assert (
                not self.p_TGt_list
            ), "Only one of p_TGt or p_TGt_list can be specified"
            self.p_TGt_list = [self.p_TGt] * len(
                self.top_object_frame_name_list
            )
            self.p_TGt = None
        elif self.p_TGt_list:
            assert len(self.p_TGt_list) == len(self.top_object_frame_name_list)
        else:
            raise ValueError("One of p_TGt or p_TGt_list must be specified")

        if self.p_BGb is not None:
            assert (
                not self.p_BGb_list
            ), "Only one of p_BGb or p_BGb_list can be specified"
            self.p_BGb_list = [self.p_BGb] * len(
                self.bottom_object_frame_name_list
            )
            self.p_BGb = None
        elif self.p_BGb_list:
            assert len(self.p_BGb_list) == len(
                self.bottom_object_frame_name_list
            )
        else:
            raise ValueError("One of p_BGb or p_BGb_list must be specified")

        return OnTopFrameListsPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            top_object_name=self.top_object_name,
            top_object_frame_name_list=self.top_object_frame_name_list,
            bottom_object_name=self.bottom_object_name,
            bottom_object_frame_name_list=self.bottom_object_frame_name_list,
            p_TGt_list=self.p_TGt_list,
            p_BGb_list=self.p_BGb_list,
            xy_threshold=self.xy_threshold,
            z_threshold=self.z_threshold,
            require_support=self.require_support,
            require_no_fingers_touching=self.require_no_fingers_touching,
            top_object_velocity_bound=self.top_object_velocity_bound,
        )


@dc.dataclass
class OnlyMakeContactWithPredicateConfig(PredicateConfig):
    body_A_scoped_name: str = ""
    body_B_scoped_name: str = ""
    frame_P_name: typing.Optional[str] = "world"
    v_PA_bound: typing.Optional[float] = 0.1
    w_PA_bound: typing.Optional[float] = 0.1

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> OnlyMakeContactWithPredicate:
        return OnlyMakeContactWithPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_A_scoped_name=self.body_A_scoped_name,
            body_B_scoped_name=self.body_B_scoped_name,
            frame_P_name=self.frame_P_name,
            v_PA_bound=self.v_PA_bound,
            w_PA_bound=self.w_PA_bound,
        )


@dc.dataclass
class ExclusiveContactPredicateConfig(PredicateConfig):
    body_A_names: list[str] = dc.field(default_factory=list)
    body_B_names: list[str] = dc.field(default_factory=list)

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> ExclusiveContactPredicate:
        return ExclusiveContactPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_A_names=self.body_A_names,
            body_B_names=self.body_B_names,
        )


@dc.dataclass
class AxisPointsInDirectionPredicateConfig(PredicateConfig):
    frame_name: str = ""
    # Axis expressed in `frame_name`'s frame that should be pointed in the
    # given direction.
    axis_F: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    # Desired direction for the `axis_f` of `frame_name` to point measured in
    # world frame.
    direction_W: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    # The rotation deviation threshold measured in degrees.
    threshold_deg: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return AxisPointsInDirectionPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_name=self.frame_name,
            axis_F=self.axis_F,
            direction_W=self.direction_W,
            threshold_rad=_to_rad(self.threshold_deg),
        )


@dc.dataclass
class AxisPointsInRelativeDirectionPredicateConfig(PredicateConfig):
    frame_A_name: str = ""
    # axis_A expressed in `frame_A_name`'s frame that should be pointed in the
    # direction of axis_B expressed in `frame_B_name`'s frame.
    axis_A: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    frame_B_name: str = ""
    axis_B: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    # The rotation deviation threshold measured in degrees.
    threshold_deg: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return AxisPointsInRelativeDirectionPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_name=self.frame_A_name,
            axis_A=self.axis_A,
            frame_B_name=self.frame_B_name,
            axis_B=self.axis_B,
            threshold_rad=_to_rad(self.threshold_deg),
        )


@dc.dataclass
class PointsInRelativeBoxPredicateConfig(PredicateConfig):
    frame_A_name: str = ""
    p_AP: list[np.ndarray] = dc.field(default_factory=list)
    frame_B_name: str = ""
    p_BP_lo: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    p_BP_hi: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return PointsInRelativeBoxPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_name=self.frame_A_name,
            p_AP=np.array(self.p_AP).T,
            frame_B_name=self.frame_B_name,
            p_BP_lo=self.p_BP_lo,
            p_BP_hi=self.p_BP_hi,
        )


@dc.dataclass
class FramesInRelativeBoxPredicateConfig(PredicateConfig):
    # List of regular expressions matching the scoped names of frames in the
    # plant.
    frame_A_names: list[str] = dc.field(default_factory=list)
    # Frame to measure the position of the origin of frame A in.
    frame_B_name: str = ""
    # Lower bound of a box defined in frame B.
    p_BP_lo: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Upper bound of a box defined in frame B.
    p_BP_hi: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return FramesInRelativeBoxPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_names=self.frame_A_names,
            frame_B_name=self.frame_B_name,
            p_BP_lo=self.p_BP_lo,
            p_BP_hi=self.p_BP_hi,
        )


@dc.dataclass
class FramesNotInRelativeBoxPredicateConfig(PredicateConfig):
    # List of regular expressions matching the scoped names of frames in the
    # plant.
    frame_A_names: list[str] = dc.field(default_factory=list)
    # Frame to measure the position of the origin of frame A in.
    frame_B_name: str = ""
    # Lower bound of a box defined in frame B.
    p_BP_lo: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Upper bound of a box defined in frame B.
    p_BP_hi: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return FramesNotInRelativeBoxPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_names=self.frame_A_names,
            frame_B_name=self.frame_B_name,
            p_BP_lo=self.p_BP_lo,
            p_BP_hi=self.p_BP_hi,
        )


@dc.dataclass
class FingersContactPredicateConfig(PredicateConfig):
    body_name: str = ""
    expect_contact: bool = True
    # TODO (hongkai): add this to all predicates that use fingers, and hoist
    # the match logic to some shared location.
    finger_namespace: typing.Optional[str] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> FingersContactPredicate:
        return FingersContactPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_name=self.body_name,
            expect_contact=self.expect_contact,
            finger_namespace=self.finger_namespace,
        )


@dc.dataclass
class FingersContactBodiesPredicateConfig(PredicateConfig):
    body_names: list[str] = dc.field(default_factory=list)
    expect_contact: bool = True
    finger_namespace: typing.Optional[str] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> FingersContactBodiesPredicate:
        return FingersContactBodiesPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_names=self.body_names,
            expect_contact=self.expect_contact,
            finger_namespace=self.finger_namespace,
        )


@dc.dataclass
class ContactFlagPredicateConfig(PredicateConfig):
    body_1_name: str = ""
    body_2_name: str = ""
    bodies_in_contact_expected: bool = True
    fingers_in_contact_1_expected: typing.Optional[bool] = None
    fingers_in_contact_2_expected: typing.Optional[bool] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> ContactFlagPredicate:
        return ContactFlagPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_1_name=self.body_1_name,
            body_2_name=self.body_2_name,
            bodies_in_contact_expected=self.bodies_in_contact_expected,
            fingers_in_contact_1_expected=self.fingers_in_contact_1_expected,
            fingers_in_contact_2_expected=self.fingers_in_contact_2_expected,
        )


@dc.dataclass
class OneToOneOnTopPredicateConfig(PredicateConfig):
    """
    Given N top objects and N bottom objects, each top_object is on top of one
    bottom_object.

    The point p_TGt on the frame top_frame_names[i] on the object
    top_object_names[i] is within xy_threshold to the point p_BGb on the
    frame bottom_frame_names[j] on the object bottom_object_names[j], within
    z_threshold.
    """

    top_object_names: list[str] = dc.field(default_factory=list)
    top_frame_names: list[str] = dc.field(default_factory=list)
    bottom_object_names: list[str] = dc.field(default_factory=list)
    bottom_frame_names: list[str] = dc.field(default_factory=list)
    p_TGt: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    xy_threshold: float = 0.1
    z_threshold: float = 0.1
    require_support: bool = True
    require_no_fingers_touching: bool = True
    top_object_velocity_bound: typing.Optional[float] = None

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> OneToOneOnTopPredicate:
        return OneToOneOnTopPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            top_object_names=self.top_object_names,
            top_frame_names=self.top_frame_names,
            bottom_object_names=self.bottom_object_names,
            bottom_frame_names=self.bottom_frame_names,
            p_TGt=self.p_TGt,
            p_BGb=self.p_BGb,
            xy_threshold=self.xy_threshold,
            z_threshold=self.z_threshold,
            require_support=self.require_support,
            require_no_fingers_touching=self.require_no_fingers_touching,
            top_object_velocity_bound=self.top_object_velocity_bound,
        )


@dc.dataclass
class MugOnBranchPredicateConfig(PredicateConfig):
    branch_frame_name: str = ""
    mug_handle_frame_names: list[str] = dc.field(default_factory=list)
    mug_center_frame_name: str = ""
    branch_length: float = 0.1
    handle_length: float = 0.05

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> MugOnBranchPredicate:
        return MugOnBranchPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            branch_frame_name=self.branch_frame_name,
            mug_handle_frame_names=self.mug_handle_frame_names,
            mug_center_frame_name=self.mug_center_frame_name,
            branch_length=self.branch_length,
            handle_length=self.handle_length,
        )


@dc.dataclass
class MugOnMugHolderPredicateConfig(PredicateConfig):
    branch_frame_names: list[str] = dc.field(default_factory=list)
    mug_handle_frame_names: list[str] = dc.field(default_factory=list)
    mug_center_frame_name: str = ""
    branch_length: float = 0.1
    handle_length: float = 0.05

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> MugOnMugHolderPredicate:
        return MugOnMugHolderPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            branch_frame_names=self.branch_frame_names,
            mug_handle_frame_names=self.mug_handle_frame_names,
            mug_center_frame_name=self.mug_center_frame_name,
            branch_length=self.branch_length,
            handle_length=self.handle_length,
        )


@dc.dataclass
class VelocityBoundPredicateConfig(PredicateConfig):
    frame_A_name: str = ""
    p_AP: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    frame_B_name: str = ""
    velocity_bound: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> VelocityBoundPredicate:
        return VelocityBoundPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_name=self.frame_A_name,
            p_AP=self.p_AP,
            frame_B_name=self.frame_B_name,
            velocity_bound=self.velocity_bound,
        )


@dc.dataclass
class VelocitiesBoundsPredicateConfig(PredicateConfig):
    # Regex of frame names within the plant.
    frame_A_names: list[str] = dc.field(default_factory=list)
    p_AP: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Regex of frame names within the plant.
    frame_B_names: list[str] = dc.field(default_factory=list)
    velocity_bound: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> VelocitiesBoundsPredicate:
        return VelocitiesBoundsPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            frame_A_names=self.frame_A_names,
            p_AP=self.p_AP,
            frame_B_names=self.frame_B_names,
            velocity_bound=self.velocity_bound,
        )


@dc.dataclass
class DeformableVelocityBoundPredicateConfig(PredicateConfig):
    body_scoped_name: str = ""
    # The translational velocity bound for the center of mass of the deformable
    # body, measured in the world frame.
    translational_velocity_bound: float = 0.0
    # The velocity bound for the "effective angular velocity" of the deformable
    # body. This measures how fast the body is rotating about an axis passing
    # through its center of mass.
    angular_velocity_bound: float = 0.0

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ) -> DeformableVelocityBoundPredicate:
        return DeformableVelocityBoundPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_scoped_name=self.body_scoped_name,
            translational_velocity_bound=self.translational_velocity_bound,
            angular_velocity_bound=self.angular_velocity_bound,
        )


@dc.dataclass
class CenterOfMassInRelativeBoxPredicateConfig(PredicateConfig):
    body_A_scoped_name: str = ""
    frame_B_name: str = ""
    # Lower bound of a box defined in frame B.
    p_BAcm_lo: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Upper bound of a box defined in frame B.
    p_BAcm_hi: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        return CenterOfMassInRelativeBoxPredicate(
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
            body_A_scoped_name=self.body_A_scoped_name,
            frame_B_name=self.frame_B_name,
            p_BAcm_lo=self.p_BAcm_lo,
            p_BAcm_hi=self.p_BAcm_hi,
        )


ConcretePredicateConfigVariant = typing.Union[
    AlwaysSatisfiedPredicateConfig,
    AxisPointsInDirectionPredicateConfig,
    AxisPointsInRelativeDirectionPredicateConfig,
    CenterOfMassInRelativeBoxPredicateConfig,
    ContactFlagPredicateConfig,
    DeformableVelocityBoundPredicateConfig,
    ExclusiveContactPredicateConfig,
    FingersContactPredicateConfig,
    FingersContactBodiesPredicateConfig,
    FramesInRelativeBoxPredicateConfig,
    MugOnBranchPredicateConfig,
    MugOnMugHolderPredicateConfig,
    NeverSatisfiedPredicateConfig,
    OnlyMakeContactWithPredicateConfig,
    OneToOneOnTopPredicateConfig,
    OnTopFrameListsPredicateConfig,
    OnTopPredicateConfig,
    PointsInRelativeBoxPredicateConfig,
    RelativePoseDistancePredicateConfig,
    RelativeRotationDistancePredicateConfig,
    RelativeTranslationDistancePredicateConfig,
    RelativeXYDistancePredicateConfig,
    VelocityBoundPredicateConfig,
    VelocitiesBoundsPredicateConfig,
]


@dc.dataclass
class AtLeastOneSatisfiedPredicateConfig(PredicateConfig):
    predicate_configs: list[ConcretePredicateConfigVariant] = dc.field(
        default_factory=list
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        predicates = [
            config.create(diagram, item_locking_monitor=item_locking_monitor)
            for config in self.predicate_configs
        ]
        return AtLeastOneSatisfiedPredicate(
            predicates,
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
        )


@dc.dataclass
class ExactlyOneSatisfiedPredicateConfig(PredicateConfig):
    predicate_configs: list[ConcretePredicateConfigVariant] = dc.field(
        default_factory=list
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        predicates = [
            config.create(diagram, item_locking_monitor=item_locking_monitor)
            for config in self.predicate_configs
        ]
        return ExactlyOneSatisfiedPredicate(
            predicates,
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
        )


@dc.dataclass
class AllSatisfiedPredicateConfig(PredicateConfig):
    predicate_configs: list[ConcretePredicateConfigVariant] = dc.field(
        default_factory=list
    )

    def create(
        self, diagram: Diagram, item_locking_monitor: ItemLockingMonitor | None
    ):
        predicates = [
            config.create(diagram, item_locking_monitor=item_locking_monitor)
            for config in self.predicate_configs
        ]
        return AllSatisfiedPredicate(
            predicates,
            name=self.name,
            diagram=diagram,
            item_locking_monitor=item_locking_monitor,
        )
