"""A re-implementation of multi_frame_pose_stream_param.h in Python, so that we
can load a matching yaml file without the need for any native C++ code.
"""

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(kw_only=True)
class SpatialVelocityLimit:
    rotation_deg: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3))
    translation: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3))

    def ToVector6(self) -> np.ndarray:
        V_abs_max = np.array(list(self.rotation_deg * math.pi / 180.0)
                             + list(self.translation))
        return V_abs_max


@dataclasses.dataclass(kw_only=True)
class VectorLimits:
    min: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0))
    max: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0))


@dataclasses.dataclass(kw_only=True)
class AngleLimit:
    min_deg: float | None = None
    max_deg: float | None = None


_PositionLimitAdjustment = dict[str, AngleLimit]


@dataclasses.dataclass(kw_only=True)
class FrameConfig:
    name: str = ""
    cartesian_axis_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(6))


@dataclasses.dataclass(kw_only=True)
class MultiFramePoseStreamParam:
    # These type aliases match the bound c++ implementation.
    VectorLimits = VectorLimits
    AngleLimit = AngleLimit
    FrameConfig = FrameConfig

    task_frame: str = ""
    control_frames: list[FrameConfig] = dataclasses.field(
        default_factory=list)
    K_VX: float = 1.0
    V_limit: SpatialVelocityLimit = dataclasses.field(
        default_factory=SpatialVelocityLimit)
    unconstrained_degrees_of_freedom_velocity_limit: float = 0.0
    p_TG: VectorLimits = dataclasses.field(
        default_factory=VectorLimits)
    enable_collision_avoidance: bool = True
    influence_distance: float = 0.0
    safety_distance: float = 0.0
    position_limits: _PositionLimitAdjustment = dataclasses.field(
        default_factory=_PositionLimitAdjustment)
    velocity_limit_scale: float = 0.9
    planar_rotation_dof_indices: list[int] = dataclasses.field(
        default_factory=list)
    throw_on_qp_failure: bool | None = None
