"""
Common configuration and factory methods.
"""

import dataclasses as dc
from typing import Any, Union

import numpy as np

from pydrake.math import RigidTransform

from anzu.intuitive.typing_ import maybe_delete_paths
from anzu.perception.pose_util import rpy_deg, to_xyz_rpy_deg


@dc.dataclass
class UniformVector:
    min: np.ndarray
    max: np.ndarray

    def sample(self):
        return np.random.uniform(self.min, self.max)


@dc.dataclass
class Uniform:
    min: float
    max: float

    def sample(self):
        return np.random.uniform(self.min, self.max)


@dc.dataclass
class Rpy:
    deg: Union[np.ndarray, UniformVector]

    def sample(self):
        return rpy_deg(sample(self.deg))

    def deterministic(self):
        return rpy_deg(deterministic(self.deg))


def sample(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, str):
        return x
    elif hasattr(x, "sample"):
        return x.sample()
    else:
        assert False, type(x)


def deterministic(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif hasattr(x, "deterministic"):
        return x.deterministic()
    else:
        assert False, type(x)


@dc.dataclass
class Transform:
    # TODO(eric.cousineau): Make this work more like C++ schemas.
    translation: Union[np.ndarray, UniformVector]
    rotation: Union[Rpy]

    def sample(self):
        return RigidTransform(
            R=sample(self.rotation),
            p=sample(self.translation),
        )

    def deterministic(self):
        return RigidTransform(
            R=deterministic(self.rotation),
            p=deterministic(self.translation),
        )

    @classmethod
    def Identity(cls):
        return cls(
            translation=np.zeros(3),
            rotation=Rpy(deg=np.zeros(3)),
        )

    @classmethod
    def from_rigid_transform(cls, X):
        xyz, rpy_deg = to_xyz_rpy_deg(X)
        return cls(translation=xyz, rotation=Rpy(deg=rpy_deg))


@dc.dataclass
class PolicyConfig:
    def create(self):
        raise NotImplementedError()


@dc.dataclass
class EnvConfig:
    pass


@dc.dataclass
class ControllerConfig:
    policy: Any

    def create(self):
        return self.policy.create()

    @staticmethod
    def from_dict_heuristic(value):
        maybe_delete_paths(
            value,
            [
                "observation",
                "action",
                "feature_map",
                "checkpoint_file",
            ],
        )
        return value


@dc.dataclass
class ScenarioConfig:
    # Base Config.
    env: EnvConfig
    controller: ControllerConfig

    def get_scenario_factory(self):
        raise NotImplementedError()

    def create(self):
        env = self.env.create()
        policy = self.controller.create()
        return env, policy

    @staticmethod
    def from_dict_heuristic(value):
        maybe_delete_paths(
            value,
            [
                "aux_scenario",
                "feature_map",
                "plotting_config",
                "train",
            ],
        )
        return value
