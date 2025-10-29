import dataclasses as dc
import typing

import numpy as np

from pydrake.systems.framework import Diagram

from anzu.intuitive.visuomotor.rewards import (
    AxisPointsInDirectionReward,
    AxisPointsInRelativeDirectionReward,
    ConstantReward,
    MaxReward,
    MugOnBranchReward,
    MugOnMugHolderReward,
    PointsInRelativeBoxCost,
    RelativeFrameListTranslationDistanceCost,
    RelativePoseDistanceCost,
    RelativeRotationDistanceCost,
    RelativeTranslationDistanceCost,
)

# TODO(dale.mcconachie) Convert documentation comments to python docstring
# format.


@dc.dataclass
class RewardConfig:
    name: str = ""
    weight: float = 1.0

    def create(self, diagram: Diagram):
        raise NotImplementedError()


@dc.dataclass
class ConstantRewardConfig(RewardConfig):
    value: float = 1.0

    def create(self, diagram: Diagram):
        return ConstantReward(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            value=self.value,
        )


@dc.dataclass
class RelativeTranslationDistanceCostConfig(RewardConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_AGa: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(self, diagram: Diagram):
        return RelativeTranslationDistanceCost(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
            p_AGa=self.p_AGa,
            p_BGb=self.p_BGb,
        )


@dc.dataclass
class RelativeRotationDistanceCostConfig(RewardConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""

    def create(self, diagram: Diagram):
        return RelativeRotationDistanceCost(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
        )


@dc.dataclass
class RelativePoseDistanceCostConfig(RewardConfig):
    frame_a_name: str = ""
    frame_b_name: str = ""
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_AGa: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file.
    p_BGb: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    rotation_weight: float = 1.0

    def create(self, diagram: Diagram):
        return RelativePoseDistanceCost(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_a_name=self.frame_a_name,
            frame_b_name=self.frame_b_name,
            p_AGa=self.p_AGa,
            p_BGb=self.p_BGb,
            rotation_weight=self.rotation_weight,
        )


@dc.dataclass
class RelativeFrameListTranslationDistanceCostConfig(RewardConfig):
    frame_a_name_list: list[str] = dc.field(default_factory=list)
    frame_b_name_list: list[str] = dc.field(default_factory=list)
    # Offset in frame A for the goal position. I.e.; the point relative to
    # frame A that should be aligned with frame B. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file. If this is specified it will be used
    # for all frames and p_AGa_list must be None.
    p_AGa: typing.Optional[np.ndarray] = None
    # A list of the same length as frame_a_name_list. If this is unspecified
    # then p_AGa will be used for all frames. If this is specified then p_AGa
    # must be None.
    p_AGa_list: typing.Optional[list[np.ndarray]] = None
    # Offset in frame B for the goal position. I.e.; the point relative to
    # frame B that should be aligned with frame A. This should be set to zero
    # in most circumstances. This can be done by adding frames in the model
    # files/directives so that we keep all similar information in one logical
    # place in a scenario definition file. If this is specified it will be used
    # for all frames and p_BGb_list must be None.
    p_BGb: typing.Optional[np.ndarray] = None
    # A list of the same length as frame_b_name_list. If this is unspecified
    # then p_BGb will be used for all frames. If this is specified then p_BGb
    # must be None.
    p_BGb_list: typing.Optional[list[np.ndarray]] = None

    def create(self, diagram: Diagram):
        if self.p_AGa is not None:
            assert (
                not self.p_AGa_list
            ), "Only one of p_AGa or p_AGa_list can be specified"
            self.p_AGa_list = [self.p_AGa] * len(self.frame_a_name_list)
            self.p_AGa = None
        elif self.p_AGa_list:
            assert len(self.p_AGa_list) == len(self.frame_a_name_list)
        else:
            raise ValueError("One of p_AGa or p_AGa_list must be specified")

        if self.p_BGb is not None:
            assert (
                not self.p_BGb_list
            ), "Only one of p_BGb or p_BGb_list can be specified"
            self.p_BGb_list = [self.p_BGb] * len(self.frame_b_name_list)
            self.p_BGb = None
        elif self.p_BGb_list:
            assert len(self.p_BGb_list) == len(self.frame_b_name_list)
        else:
            raise ValueError("One of p_BGb or p_BGb_list must be specified")

        return RelativeFrameListTranslationDistanceCost(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_a_name_list=self.frame_a_name_list,
            frame_b_name_list=self.frame_b_name_list,
            p_AGa_list=self.p_AGa_list,
            p_BGb_list=self.p_BGb_list,
        )


@dc.dataclass
class AxisPointsInDirectionRewardConfig(RewardConfig):
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

    def create(self, diagram: Diagram):
        return AxisPointsInDirectionReward(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_name=self.frame_name,
            axis_F=self.axis_F,
            direction_W=self.direction_W,
        )


@dc.dataclass
class AxisPointsInRelativeDirectionRewardConfig(RewardConfig):
    # axis_A in frame_A should be pointing in the direction of axis_B in
    # frame B.
    frame_A_name: str = ""
    axis_A: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    frame_B_name: str = ""
    # given direction.
    axis_B: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )

    def create(self, diagram: Diagram):
        return AxisPointsInRelativeDirectionReward(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_A_name=self.frame_A_name,
            axis_A=self.axis_A,
            frame_B_name=self.frame_B_name,
            axis_B=self.axis_B,
        )


@dc.dataclass
class PointsInRelativeBoxCostConfig(RewardConfig):
    """
    Point Pᵢ attached to frame A should lie within a box
    p_BP_lo <= p_BP <= p_BP_hi specified in frame B.
    """

    frame_A_name: str = ""
    p_AP: list[np.ndarray] = dc.field(
        default_factory=lambda: [np.array([0.0, 0.0, 0.0])]
    )
    frame_B_name: str = ""
    p_BP_lo: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    p_BP_hi: np.ndarray = dc.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def create(self, diagram: Diagram):
        return PointsInRelativeBoxCost(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            frame_A_name=self.frame_A_name,
            p_AP=np.array(self.p_AP).T,
            frame_B_name=self.frame_B_name,
            p_BP_lo=self.p_BP_lo,
            p_BP_hi=self.p_BP_hi,
        )


@dc.dataclass
class MugOnBranchRewardConfig(RewardConfig):
    branch_frame_name: str = ""
    mug_handle_frame_names: list[str] = dc.field(default_factory=list)
    mug_center_frame_name: str = ""
    branch_length: float = 0.1
    handle_length: float = 0.05

    def create(self, diagram: Diagram) -> MugOnBranchReward:
        return MugOnBranchReward(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            branch_frame_name=self.branch_frame_name,
            mug_handle_frame_names=self.mug_handle_frame_names,
            mug_center_frame_name=self.mug_center_frame_name,
            branch_length=self.branch_length,
            handle_length=self.handle_length,
        )


@dc.dataclass
class MugOnMugHolderRewardConfig(RewardConfig):
    branch_frame_names: list[str] = dc.field(default_factory=list)
    mug_handle_frame_names: list[str] = dc.field(default_factory=list)
    mug_center_frame_name: str = ""
    branch_length: float = 0.1
    handle_length: float = 0.05

    def create(self, diagram: Diagram) -> MugOnMugHolderReward:
        return MugOnMugHolderReward(
            name=self.name,
            weight=self.weight,
            diagram=diagram,
            branch_frame_names=self.branch_frame_names,
            mug_handle_frame_names=self.mug_handle_frame_names,
            mug_center_frame_name=self.mug_center_frame_name,
            branch_length=self.branch_length,
            handle_length=self.handle_length,
        )


# I do not include MaxRewardConfig here because the class is not defined. As a
# result, we cannot create MaxRewardConfig nested inside MaxRewardConfig.
ConcreteRewardConfigVariant = typing.Union[
    AxisPointsInDirectionRewardConfig,
    AxisPointsInRelativeDirectionRewardConfig,
    ConstantRewardConfig,
    MugOnBranchRewardConfig,
    MugOnMugHolderRewardConfig,
    PointsInRelativeBoxCostConfig,
    RelativeTranslationDistanceCostConfig,
    RelativeRotationDistanceCostConfig,
    RelativeFrameListTranslationDistanceCostConfig,
    RelativePoseDistanceCostConfig,
]


@dc.dataclass
class MaxRewardConfig(RewardConfig):
    evaluator_configs: list[ConcreteRewardConfigVariant] = dc.field(
        default_factory=list
    )

    def create(self, diagram: Diagram):
        evaluators = [
            config.create(diagram) for config in self.evaluator_configs
        ]
        return MaxReward(
            evaluators, name=self.name, weight=self.weight, diagram=diagram
        )
