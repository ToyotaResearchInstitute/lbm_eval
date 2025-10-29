import logging
import math
import re

import numpy as np

from pydrake.multibody.parsing import GetScopedFrameByName
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Frame, FrameIndex
from pydrake.systems.framework import Context, Diagram

import anzu.intuitive.common.math_safe as math_safe


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


class Reward:
    """Base class for all rewards; used to store common data elements and
    define the basic API.
    """

    def __init__(self, name: str, weight: float, diagram: Diagram):
        if not name:
            raise ValueError("`name` must not be empty")
        self._name = name
        self._weight = weight
        self._diagram = diagram

        # Try some known locations for the plant, and throw otherwise.
        sim_diagram = diagram
        if self._diagram.HasSubsystemNamed("station"):
            sim_diagram = diagram.GetSubsystemByName("station")
        self._plant = sim_diagram.GetSubsystemByName("plant")

    @property
    def name(self):
        return self._name

    def eval(self, root_context: Context):
        return self.eval_detailed(root_context)[self.name]

    def eval_detailed(self, root_context: Context):
        """Evaluates the reward function as well as any
        intermediate/sub-components of the reward.

        Args:
            context: The Drake Context for the plant passed in to the
                constructor.
        Returns:
            A dictionary of named reward components and values for those
            components. Must contain at least one reward with `self.name` as
            the key and a scalar value as the item.
        """
        # TODO(dale.mcconachie): Add action as an input to this function
        # when we need that affordance.
        raise NotImplementedError


def _raise_if_name_overlap(evaluators: list[Reward]):
    names = [x.name for x in evaluators]
    overlap = [x for x in names if names.count(x) > 1]
    if overlap:
        raise RuntimeError(f"Overlap in evaluator names: {overlap}")


def _raise_if_key_overlap(dict1: dict, dict2: dict):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    overlap = keys1.intersection(keys2)
    if overlap:
        raise RuntimeError(f"Overlap in dict keys: {overlap}")


class MaxReward(Reward):
    """Performs a 'hard-max' operation on the results from each individual
    evaluator.
    """

    def __init__(
        self,
        evaluators: list[Reward],
        **kwargs,
    ):
        super().__init__(**kwargs)
        _raise_if_name_overlap(evaluators)
        self._evaluators = evaluators

    def eval_detailed(self, root_context: Context):
        aggregate_rewards = list()
        aggregate_details = dict()
        for evaluator in self._evaluators:
            detailed_reward = evaluator.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_reward)
            aggregate_details.update(detailed_reward)
            aggregate_rewards.append(detailed_reward[evaluator.name])
        aggregate_details[self.name] = max(aggregate_rewards)
        return aggregate_details


class SoftMaxReward(Reward):
    """Performs a 'soft-max' operation on the results from each individual
    evaluator using the log-sum-exp function.
    """

    def __init__(
        self,
        evaluators: list[Reward],
        **kwargs,
    ):
        super().__init__(**kwargs)
        _raise_if_name_overlap(evaluators)
        self._evaluators = evaluators

    def eval_detailed(self, root_context: Context):
        aggregate_exp_rewards = 0
        aggregate_details = dict()
        for evaluator in self._evaluators:
            detailed_reward = evaluator.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_reward)
            aggregate_details.update(detailed_reward)
            aggregate_exp_rewards += math.exp(detailed_reward[evaluator.name])
        aggregate_details[self.name] = math.log(aggregate_exp_rewards)
        return aggregate_details


class TotalReward(Reward):
    def __init__(
        self,
        evaluators: list[Reward],
        **kwargs,
    ):
        super().__init__(**kwargs)
        _raise_if_name_overlap(evaluators)
        self._evaluators = evaluators

    def eval_detailed(self, root_context: Context):
        total_reward = 0.0
        aggregate_details = dict()
        for evaluator in self._evaluators:
            detailed_reward = evaluator.eval_detailed(root_context)
            _raise_if_key_overlap(aggregate_details, detailed_reward)
            aggregate_details.update(detailed_reward)
            total_reward += detailed_reward[evaluator.name]
        aggregate_details[self.name] = self._weight * total_reward
        return aggregate_details


class ConstantReward(Reward):
    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self._value = value

    def eval_detailed(self, root_context: Context):
        return {self.name: self._weight * self._value}


class RelativeTranslationDistanceCost(Reward):
    """Defines a cost based on the translation distance between two frames.
    The distance used is relative to points specified in those frames.
    """

    def __init__(
        self,
        frame_a_name: str,
        frame_b_name: str,
        p_AGa: np.ndarray,
        p_BGb: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert p_AGa.shape == (3,), p_AGa
        assert p_BGb.shape == (3,), p_BGb
        self._frame_a = GetScopedFrameByName(self._plant, frame_a_name)
        self._frame_b = GetScopedFrameByName(self._plant, frame_b_name)
        self._p_AGa = p_AGa
        self._p_BGb = p_BGb

    def eval_detailed(self, root_context: Context):
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_AB = self._plant.CalcRelativeTransform(
            plant_context, self._frame_a, self._frame_b
        )
        p_AGb = X_AB.multiply(self._p_BGb)
        distance = np.linalg.norm(self._p_AGa - p_AGb)
        return {self.name: -self._weight * distance}


class RelativeRotationDistanceCost(Reward):
    """Defines a cost based on the rotation difference between two frames
    (measured in radians)."""

    def __init__(
        self,
        frame_a_name,
        frame_b_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_a = GetScopedFrameByName(self._plant, frame_a_name)
        self._frame_b = GetScopedFrameByName(self._plant, frame_b_name)

    def eval_detailed(self, root_context: Context):
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        R_AB = self._plant.CalcRelativeTransform(
            plant_context, self._frame_a, self._frame_b
        ).rotation()
        # Compute the angle derived from Rodrigues' formula
        # R = I + sinθ*K + (1−cosθ)*K².
        angle = math_safe.acos_safe((R_AB.matrix().trace() - 1.0) / 2.0)
        return {self.name: -self._weight * angle}


class RelativePoseDistanceCost(Reward):
    def __init__(
        self,
        frame_a_name,
        frame_b_name,
        p_AGa: np.ndarray,
        p_BGb: np.ndarray,
        rotation_weight,
        **kwargs,
    ):
        super().__init__(**kwargs)
        translation = RelativeTranslationDistanceCost(
            name=f"{self.name}.translation",
            weight=1.0,
            diagram=self._diagram,
            frame_a_name=frame_a_name,
            frame_b_name=frame_b_name,
            p_AGa=p_AGa,
            p_BGb=p_BGb,
        )
        rotation = RelativeRotationDistanceCost(
            name=f"{self.name}.rotation",
            weight=rotation_weight,
            diagram=self._diagram,
            frame_a_name=frame_a_name,
            frame_b_name=frame_b_name,
        )
        self._evaluator_internal = TotalReward(
            name=self.name,
            weight=self._weight,
            diagram=self._diagram,
            evaluators=[translation, rotation],
        )

    def eval_detailed(self, root_context: Context):
        return self._evaluator_internal.eval_detailed(root_context)


class RelativeFrameListTranslationDistanceCost(Reward):
    """Defines a cost based on the minimum translation distance between two
    sets of frames. The distance used is relative to points specified in those
    frames.
    """

    def __init__(
        self,
        frame_a_name_list: list[str],
        frame_b_name_list: list[str],
        p_AGa_list: list[np.ndarray],
        p_BGb_list: list[np.ndarray],
        **kwargs,
    ):
        super().__init__(**kwargs)
        for p_AGa in p_AGa_list:
            assert p_AGa.shape == (3,), p_AGa
        for p_BGb in p_BGb_list:
            assert p_BGb.shape == (3,), p_BGb
        assert len(frame_a_name_list) == len(p_AGa_list)
        assert len(set(frame_a_name_list)) == len(
            frame_a_name_list
        ), "The items in frame_a_name_list must be unique"
        assert len(frame_b_name_list) == len(p_BGb_list)
        assert len(set(frame_b_name_list)) == len(
            frame_b_name_list
        ), "The items in frame_b_name_list must be unique"

        evaluators = []
        for frame_a_name, p_AGa in zip(frame_a_name_list, p_AGa_list):
            for frame_b_name, p_BGb in zip(frame_b_name_list, p_BGb_list):
                name = f"{self.name}.{frame_a_name}.{frame_b_name}"
                evaluators.append(
                    RelativeTranslationDistanceCost(
                        name=name,
                        weight=self._weight,
                        diagram=self._diagram,
                        frame_a_name=frame_a_name,
                        frame_b_name=frame_b_name,
                        p_AGa=p_AGa,
                        p_BGb=p_BGb,
                    )
                )
        self._evaluator_internal = MaxReward(
            name=self.name,
            weight=1.0,
            diagram=self._diagram,
            evaluators=evaluators,
        )

    def eval_detailed(self, root_context: Context):
        return self._evaluator_internal.eval_detailed(root_context)


class AxisPointsInDirectionReward(Reward):
    def __init__(
        self,
        frame_name: str,
        axis_F: np.ndarray,
        direction_W: np.ndarray,
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

    def eval_detailed(self, root_context: Context):
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        X_WF = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), self._frame
        )
        axis_W = X_WF.rotation() @ self._axis_F
        alignment = np.dot(self._direction_W, axis_W)
        return {self.name: self._weight * alignment}


class AxisPointsInRelativeDirectionReward(Reward):
    def __init__(
        self,
        frame_A_name: str,
        axis_A: np.ndarray,
        frame_B_name: str,
        axis_B: np.ndarray,
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

    def eval_detailed(self, root_context: Context):
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        # Compute the axis_A expressed in frame_b
        n_B = (
            self._plant.CalcRelativeTransform(
                plant_context, self._frame_B, self._frame_A
            ).rotation()
            @ self._axis_A
        )
        alignment = np.dot(self._axis_B, n_B)
        return {self.name: self._weight * alignment}


class PointsInRelativeBoxCost(Reward):
    """
    Points Pᵢ attached to a frame A should lie within a box specified in frame
    B. This cost computes the violation of the box constraint along each
    dimension of the box, and sum up the violation.

    If frame_A_name is a regular expression, then this operation is performed
    on all frames that match and results are added together.
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
        """
        The box in frame B is p_BP_lo <= p_BP <= p_BP_hi
        p_AP should be of shape (3,) or (3, N).
        """
        super().__init__(**kwargs)
        self._frame_A = _get_frames_by_regexp(self._plant, frame_A_name)
        assert len(self._frame_A) > 0, f"No frames found: {frame_A_name}"

        self._frame_B = GetScopedFrameByName(self._plant, frame_B_name)
        assert p_AP.shape == (3,) or p_AP.shape[0] == 3
        self._p_AP = p_AP
        assert p_BP_lo.shape == (3,)
        assert p_BP_hi.shape == (3,)
        assert np.all(p_BP_lo <= p_BP_hi)
        self._p_BP_lo = p_BP_lo
        self._p_BP_hi = p_BP_hi

    def _eval_violation(self, plant_context: Context, frame_A: Frame) -> float:
        p_BP = self._plant.CalcPointsPositions(
            plant_context, frame_A, self._p_AP, self._frame_B
        )

        violation = (
            np.maximum(p_BP.T - self._p_BP_hi, np.zeros_like(p_BP.T)).sum()
            + np.maximum(self._p_BP_lo - p_BP.T, np.zeros_like(p_BP.T)).sum()
        )
        return violation

    def eval_detailed(self, root_context: Context) -> dict:
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        result = {}
        total = 0.0
        for frame_A in self._frame_A:
            violation = self._eval_violation(plant_context, frame_A)
            total += violation
            result[f"{self.name}.{frame_A.scoped_name()}"] = violation
        result[self.name] = -self._weight * total
        return result


class MugOnBranchReward(Reward):
    """
    Mug hangs on a branch.
    """

    def __init__(
        self,
        branch_frame_name: str,
        mug_handle_frame_names: list[str],
        mug_center_frame_name: str,
        branch_length: float,
        handle_length: float,
        **kwargs,
    ):
        """
        We assume that the z axis of the branch frame points outward along the
        branch, and the branch frame is at the root of that branch.

        Args:
          branch_frame_name: Name of the branch frame.
          mug_handle_middle_frame_name: Name of the frame on the middle of the
            mug handle.
          mug_handle_frame_names: Names of frames on the handle. One of the mug
            handle frame origins must be above the branch.
          branch_length: The length of the branch.
          handle_length: The length of the handle (along the axis mug axis
            direction).
        """
        super().__init__(**kwargs)

        # The x axis of the branch frame always points perpendicular from the
        # branch and goes downward.
        p_BP_lo = np.array([-0.02, -handle_length / 2, 0])
        p_BP_hi = np.array([0.0, handle_length / 2, branch_length * 0.9])
        self._handle_above_branch = MaxReward(
            name=f"{self.name}.handle_above_branch",
            diagram=self._diagram,
            weight=self._weight,
            evaluators=[
                PointsInRelativeBoxCost(
                    name=f"{self.name}.{handle_frame_name}_above_branch",
                    weight=1.0,
                    diagram=self._diagram,
                    frame_A_name=handle_frame_name,
                    p_AP=np.zeros((3, 1)),
                    frame_B_name=branch_frame_name,
                    p_BP_lo=p_BP_lo,
                    p_BP_hi=p_BP_hi,
                )
                for handle_frame_name in mug_handle_frame_names
            ],
        )
        self._mug_center_below_branch = PointsInRelativeBoxCost(
            name=f"{self.name}.mug_center_below_branch",
            weight=1.0,
            diagram=self._diagram,
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
        reward = (
            handle_above_branch_ret[self._handle_above_branch.name]
            + self._weight
            * mug_center_below_branch_ret[self._mug_center_below_branch.name]
        )
        ret = {self.name: reward}
        ret.update(handle_above_branch_ret)
        ret.update(mug_center_below_branch_ret)

        return ret


class MugOnMugHolderReward(Reward):
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
        self._mug_above_branch = MaxReward(
            name=f"{self.name}.above_branch",
            weight=self._weight,
            diagram=self._diagram,
            evaluators=[
                MugOnBranchReward(
                    name=f"{self.name}.above_{branch_frame_name}",
                    weight=1,
                    diagram=self._diagram,
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
        reward = ret[self._mug_above_branch.name]
        ret[self._name] = reward
        return ret
