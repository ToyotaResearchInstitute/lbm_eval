#! /usr/bin/env python3
"""
ggould's "useless policy" example to wave the robot arms.

Originally from anzu/intuitive/lbm_eval_dev/lbm_eval_sandbox.ipynb
"""
import argparse
import copy
import uuid
import warnings

import numpy as np

from grpc_workspace.lbm_policy_server import (
    LbmPolicyServerConfig,
    run_policy_server,
)
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.policy import Policy, PolicyMetadata


def _get_policy_metadata():
    return PolicyMetadata(
        name="WaveAround",
        skill_type="Undefined",
        checkpoint_path="None",
        git_repo="Unknown",
        git_sha="Undefined",
    )


class WaveAround(Policy):
    """A trivial demonstration policy that waves the arms."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._counter = 0
        self._initial_poses = None

    def get_policy_metadata(self):
        return _get_policy_metadata()

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        if self._initial_poses is None:
            self._initial_poses = copy.deepcopy(observation.robot.actual.poses)

        grippers = copy.deepcopy(observation.robot.actual.grippers)
        poses = copy.deepcopy(self._initial_poses)
        offset = np.sin(self._counter * 0.1 * np.array([0.02, 0.03, 0.05]))
        for robot_name, pose in self._initial_poses.items():
            observed_xyz = self._initial_poses[robot_name].translation()
            poses[robot_name].set_translation(observed_xyz + offset)
        self._counter += 1
        return PosesAndGrippers(poses=poses, grippers=grippers)


class WaveAroundBatch(Policy):
    """A trivial demonstration policy that waves the arms.

    Supports the gRPC batch interface.
    """

    def __init__(self):
        # # An internal client identifier to be used when this policy is
        # called through the non-batch interface.
        self._internal_uuid = uuid.uuid4()
        # A mapping from UUID to the individual policy per UUID.
        self._sub_policies: dict[uuid.UUID, WaveAround] = {}

    def reset(self, seed: int | None, options=None):
        self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(
        self, seeds: dict[uuid.UUID, int | None], options=None
    ) -> None:
        for one_uuid, one_seed in seeds.items():
            if one_seed is not None:
                warnings.warn(f"We ignore the seed for {one_uuid}!")

            # Handle the internal state for each UUID.
            self._sub_policies[one_uuid] = WaveAround()

    def get_policy_metadata(self):
        return _get_policy_metadata()

    def step(self, observation):
        batch_actions = self.step_batch({self._internal_uuid: observation})
        return batch_actions[observation]

    def step_batch(
        self, observations: dict[uuid.UUID, MultiarmObservation]
    ) -> dict[uuid.UUID, PosesAndGrippers]:
        batch_actions = {}
        for one_uuid, observation in observations.items():
            sub_policy = self._sub_policies[one_uuid]
            batch_actions[one_uuid] = sub_policy.step(observation)
        return batch_actions


def main():
    parser = argparse.ArgumentParser()
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()
    policy = WaveAroundBatch()
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
