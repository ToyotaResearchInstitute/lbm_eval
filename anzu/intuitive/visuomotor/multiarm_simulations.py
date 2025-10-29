"""
Multiarm simulation environments and configurations. This library
also contains classes and functions to enable these simulations to
be configured and run with Scenario files compatible with
//sim/station/hardware_station_simulation.

"""
from contextlib import nullcontext
import copy
import dataclasses as dc
import difflib
import importlib
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from robot_gym.multiarm_spaces import (
    CURRENT_VERSION,
    CameraDepthImage,
    CameraImageSet,
    CameraImageSetMap,
    CameraLabelImage,
    CameraRgbImage,
    MultiarmObservation,
    PosesAndGrippers,
    PosesAndGrippersActualAndDesired,
)
from robot_gym.multiarm_spaces_conversions import action_get_stop_info

from pydrake.common import RandomGenerator
from pydrake.common.value import Value
from pydrake.common.yaml import yaml_dump, yaml_dump_typed, yaml_load_typed
from pydrake.geometry import Meshcat, RenderLabel
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import GetScopedFrameByName
from pydrake.multibody.tree import (
    Frame,
    JacobianWrtVariable,
    ModelInstanceIndex,
)
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator
from pydrake.systems.framework import (
    DiagramBuilder,
    FixedInputPortValue,
    InputPort,
    OutputPortIndex,
)
from pydrake.systems.primitives import (
    BusCreator,
    ConstantVectorSource,
    Multiplexer,
    Selector,
    SelectorParams,
    ZeroOrderHold,
)

from anzu.common.path_util import resolve_path
from anzu.common.plant_multiplexers import (
    MakeCommandDemultiplexer,
    MakeStateMultiplexer,
)
from anzu.common.profiling.perf_controller import scoped_perf_sampling
from anzu.common.tmpfile import AutoTemporaryDirectory
from anzu.intuitive.s3_utils import assert_s3_access, exec_s3_sync, is_s3_path
from anzu.intuitive.skill_defines import SkillType
from anzu.intuitive.station_defines import PLATFORM_TYPE_LBM_DUAL_PANDA
from anzu.intuitive.time_util import get_datetime_now
from anzu.intuitive.typing_ import copy_field
from anzu.intuitive.visuomotor.bases import (
    AnzuEnv,
    DistributionShiftInfo,
    EnvDomain,
    EnvMetadata,
    TimeStep,
)
from anzu.intuitive.visuomotor.camera_semantic_names import (
    load_camera_id_to_semantic_name,
    rename_camera_image_set,
)
from anzu.intuitive.visuomotor.common_schemas import EnvConfig
from anzu.intuitive.visuomotor.common_spaces import PolicyInstant
from anzu.intuitive.visuomotor.episode import TimingInfo
from anzu.intuitive.visuomotor.multiarm_user_input import (
    UserInputFootSwitchConfig,
    UserInputJoystickConfig,
    UserInputNoneConfig,
)
from anzu.intuitive.visuomotor.reward_schemas import (
    AxisPointsInDirectionRewardConfig,
    AxisPointsInRelativeDirectionRewardConfig,
    ConstantRewardConfig,
    MaxRewardConfig,
    MugOnBranchRewardConfig,
    MugOnMugHolderRewardConfig,
    PointsInRelativeBoxCostConfig,
    RelativeFrameListTranslationDistanceCostConfig,
    RelativePoseDistanceCostConfig,
    RelativeRotationDistanceCost,
    RelativeTranslationDistanceCostConfig,
)
from anzu.intuitive.visuomotor.rewards import TotalReward
from anzu.intuitive.visuomotor.simulation_bases import (
    translate_drake_simulation_failures,
)
from anzu.intuitive.visuomotor.station_scenario_yaml_processing import (
    CameraOverrides,
    ScenarioResolutionIntermediaries,
    apply_scenario_overrides_and_sample,
)
from anzu.intuitive.visuomotor.task_predicate_schemas import (
    AllSatisfiedPredicateConfig,
    AlwaysSatisfiedPredicateConfig,
    AtLeastOneSatisfiedPredicateConfig,
    AxisPointsInDirectionPredicateConfig,
    AxisPointsInRelativeDirectionPredicateConfig,
    CenterOfMassInRelativeBoxPredicateConfig,
    ContactFlagPredicateConfig,
    DeformableVelocityBoundPredicateConfig,
    ExactlyOneSatisfiedPredicateConfig,
    ExclusiveContactPredicateConfig,
    FingersContactBodiesPredicateConfig,
    FingersContactPredicateConfig,
    FramesInRelativeBoxPredicateConfig,
    FramesNotInRelativeBoxPredicateConfig,
    MugOnBranchPredicateConfig,
    MugOnMugHolderPredicateConfig,
    NeverSatisfiedPredicateConfig,
    OnTopFrameListsPredicateConfig,
    OnTopPredicateConfig,
    OneToOneOnTopPredicateConfig,
    OnlyMakeContactWithPredicateConfig,
    PointsInRelativeBoxPredicateConfig,
    RelativePoseDistancePredicateConfig,
    RelativeRotationDistancePredicateConfig,
    RelativeTranslationDistancePredicateConfig,
    RelativeXYDistancePredicateConfig,
    VelocitiesBoundsPredicateConfig,
    VelocityBoundPredicateConfig,
)
from anzu.intuitive.visuomotor.task_predicates import AllSatisfiedPredicate
from anzu.mobile.control.head_control import make_head_control_system
from anzu.sim.common.control import SlewRateSystem
from anzu.sim.common.deformable_sim_config_functions import (
    ApplyDeformableSimConfig,
)
from anzu.sim.common.item_locking_monitor_config_functions import (
    ItemLockingMonitor,
)
from anzu.sim.common.primitive_systems import fix_port
from anzu.sim.common.take_scenario_sample import CommonScenarioConfig

from anzu.intuitive.visuomotor.demo.visuomotor_multi_frame_pose_stream_config_functions import (  # noqa
    load_pose_stream_params,
)

try:
    # By default, we'll aim to use the C++ implementation.
    from anzu.robot_bridge.cc import MultiFramePoseStreamParam
except ImportError:
    # However, when that's missing (for open-source LBM Eval) we'll fall back
    # to its re-implemented twin in pure Python.
    from anzu.robot_bridge.multi_frame_pose_stream_param import (
        MultiFramePoseStreamParam,
    )

_BASE_COMMAND_SYSTEM_NAME = "BaseRainbowRby1CommandSystem"


# TODO(dale.mcconachie) Deduplicate information in this schema from the
# visuomotor_multi_frame_pose_stream files (i.e.; the file referenced in
# DifferentialInverseKinematicsConfig). At the very least assert equality of
# content.
@dc.dataclass(kw_only=True)
class MultiarmRobotSetup:
    """The Robot configuration used to specify the control modes, controlled
    frames, and a task frame associated with a scenario."""

    diff_ik_names: List[str] = dc.field(default_factory=list)
    """The model names to be controlled by the DiffIK controller, e.g., robot
    arms. However, it must specifically exclude any models which are to be
    controlled by other means, e.g., gripper models.
    """

    diff_ik_frames: List[str] = dc.field(default_factory=list)
    """The names of the frames that are exposed for a policy to command via
    DiffIK."""

    # The head is handled separately from the other frames, so create an
    # explicit separate entry for it.
    head_frame: Optional[str] = None
    """The name of the head frame used for independent head tracking (if any).
    If specified; this frame must not be in `diff_ik_frames`.
    """

    gripper_names: List[str] = dc.field(default_factory=list)
    """The model names of the grippers that are exposed for a policy to
    command."""

    # The base requires specific additional wiring; so split it out from the
    # other models.
    base_name: Optional[str] = None

    # TODO(#16792) Enable having a different value for the task frame at the
    # robot gym API level vs the mid level controller API level.

    # TODO(dale.mcconachie) Extract this from the model directives or otherwise
    # remove the ability for there to be a an unintended difference between the
    # planning and simulation models.
    # TODO(dale.mcconachie) This default is only valid for panda stations; on
    # RBY1 stations it is explicitly invalid.
    station_postfix: str = "nominal"

    def validate(self):
        if self.head_frame in self.diff_ik_frames:
            raise ValueError(
                "If head_frame is specified, this means we are doing "
                "independent head tracking. This means that the head frame "
                "cannot be in the list of diff IK frames."
            )

        for gripper_name in self.gripper_names:
            if gripper_name in self.diff_ik_names:
                raise ValueError(
                    "Grippers cannot be controlled by the DiffIK controller."
                )

    def __post_init__(self):
        self.validate()


@dc.dataclass(kw_only=True)
class DifferentialInverseKinematicsConfig:
    # TODO(dale.mcconachie) Add some mechanism to ensure that if/when params
    # are added to `{robot,rby1}_control_main.cc` they are added here
    # automatically if they are relevant.

    config_file: str = ""
    """Configuration file to be used for the pose stream component of the
    diff IK system.
    """

    # TODO(dale.mcconachie) This option should be merged with the above
    # `config_file` via a `Union` and the cross check removed. Doing so
    # requires more surgery throughout the codebase to properly support having
    # this data specified either way so we defer that work. Also of note,
    # //intuitive/haptic:schemas has a `!ExtractYaml` tag that could
    # potentially be of use, and/or replaced/merged with
    # //sim/common:take_scenario_sample's `!IncludeFile` affordance.
    pose_stream_config: Optional[MultiFramePoseStreamParam] = None
    """Configuration that was used by the the diff IK pose stream system. If
    this is `None` it will be populated from the contents of `config_file`. If
    it is not `None` it will be compared to the contents and an exception will
    be thrown if the contents are different.
    """

    allow_finger_table_collision: Optional[bool] = False

    def validate(self):
        assert self.config_file != "", "Must specify a file to load"

        pose_stream_config = load_pose_stream_params(
            filename=resolve_path(self.config_file)
        )
        if self.pose_stream_config is not None:
            # Launder through strings as the underlying C++ types don't all
            # have equality operators.
            stored_string = yaml_dump_typed(self.pose_stream_config)
            loaded_string = yaml_dump_typed(pose_stream_config)

            if stored_string != loaded_string:
                diff = "\n".join(
                    difflib.context_diff(
                        stored_string.splitlines(), loaded_string.splitlines()
                    )
                )
                raise ValueError(
                    "Stored pose stream configuration does not match the "
                    "contents on disk. This likely means that you are using a "
                    "different version of anzu than was used to generate the "
                    "you are loading. Proceed with caution only if you know "
                    "exactly what is different and what it affects.\n\n" + diff
                )
        else:
            self.pose_stream_config = pose_stream_config

    def __post_init__(self):
        self.validate()


RewardConfigVariant = Union[
    AxisPointsInDirectionRewardConfig,
    AxisPointsInRelativeDirectionRewardConfig,
    ConstantRewardConfig,
    MaxRewardConfig,
    MugOnBranchRewardConfig,
    MugOnMugHolderRewardConfig,
    PointsInRelativeBoxCostConfig,
    RelativeFrameListTranslationDistanceCostConfig,
    RelativePoseDistanceCostConfig,
    RelativeRotationDistanceCost,
    RelativeTranslationDistanceCostConfig,
]
TaskPredicateConfigVariant = Union[
    AllSatisfiedPredicateConfig,
    AlwaysSatisfiedPredicateConfig,
    AtLeastOneSatisfiedPredicateConfig,
    AxisPointsInDirectionPredicateConfig,
    AxisPointsInRelativeDirectionPredicateConfig,
    CenterOfMassInRelativeBoxPredicateConfig,
    ContactFlagPredicateConfig,
    DeformableVelocityBoundPredicateConfig,
    ExactlyOneSatisfiedPredicateConfig,
    ExclusiveContactPredicateConfig,
    FingersContactPredicateConfig,
    FingersContactBodiesPredicateConfig,
    FramesInRelativeBoxPredicateConfig,
    FramesNotInRelativeBoxPredicateConfig,
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
UserInputVariant = Union[
    UserInputNoneConfig, UserInputFootSwitchConfig, UserInputJoystickConfig
]


@dc.dataclass(kw_only=True)
class HardwareStationScenarioSimulationEnvConfig(EnvConfig):
    # TODO(dale.mcconachie) This config (and the associated __init__ method)
    # have too many arguments.
    # TODO(dale.mcconachie): The `Optional` flags for this config do not match
    # the `__init__` method of the environment class itself.
    # TODO(dale.mcconachie) Reorder this config to put related items together.
    # At the same time; update the doc string for the environment itself, sort
    # the environment's __init__ arguments to match the order here; and sort
    # the YAML files to match. This applies to all sub-configs as well.
    # TODO(dale.mcconachie) Give every field a default value.
    robot_setup: MultiarmRobotSetup

    # This governs the schema class used to load the simulation scenario. It is
    # the python package name where the scenario module can be imported, i.e.,
    # `{simulation_scenario_package}.hardware_station_simulation_scenario.Scenario`
    # will be the dataclass used to load `simulation_scenario_config` data, and
    # `{simulation_scenario_package}.hardware_station.MakeHardwareStation` will
    # be the function used to instantiate the scenario.
    simulation_scenario_package: str = "anzu.sim.station"

    simulation_scenario_config: CommonScenarioConfig

    diff_ik_config: DifferentialInverseKinematicsConfig
    # TODO(dale.mcconachie): This contains elements that are most closely
    # related to the simulation scenario, so it should be moved to next to it.
    # However this would require that `diff_ik_config` have a default value.
    camera_overrides: CameraOverrides = dc.field(
        default_factory=CameraOverrides
    )
    # TODO(dale.mcconachie) Should this be enabled and the simulator_config
    # value disabled for all teleop simulations? The default visualization rate
    # is 20 Hz, which is only double the step rate, so drake could be slowing
    # things down more than is needed in some cases.
    target_realtime_rate: float = 0.0
    """If positive, slow the simulation down to this realtime rate. This is
    handled at the environment-step level rather than the drake simulation
    level to allow for an entire `dt` worth of time (including camera
    rendering) to be completed before realtime rate is evaluated.
    """
    dt: float = 0.1
    # This is used to resolve our semantic name mapping. This can be explicitly
    # set to None if the scenario truly does not correspond to a physical
    # station.
    station_name: Optional[str] = None

    # This is used to populate EnvMetadata. Ideally, this won't be optional,
    # but is necessary for backward compatibility reasons.
    skill: Optional[SkillType] = None

    # Similar to MultiarmLcmEnvConfig.default_language_instruction, this field
    # should be filled for tasks / skills whose goals / predicates are clear.
    # Unlike real hardware configs, these simulation configs mostly should
    # have well-defined goal / task predicates at authoring time. So this
    # field should really be present for most configs. There could still be
    # exceptions like in test specific configs or generic sandbox configs,
    # where the language instruction can be omitted or overwritten at run time.
    # Once provided, at run time, the MultiarmLcmEnv will incorporate this
    # as the "default" language instruction as part of the observation space
    # to any policy. In the future, we plan to add mechanisms to further change
    # language instructions in the observation space to support dynamic tasks.
    default_language_instruction: Optional[str] = None

    # Distribution shift information that was used to define the values in
    # `simulation_scenario_config`.
    distribution_shift_info: DistributionShiftInfo = dc.field(
        default_factory=DistributionShiftInfo
    )

    reward_configs: List[RewardConfigVariant] = dc.field(
        default_factory=lambda: [
            ConstantRewardConfig(name="unused", weight=0, value=0)
        ]
    )
    task_predicate_configs: List[TaskPredicateConfigVariant] = dc.field(
        default_factory=lambda: [NeverSatisfiedPredicateConfig(name="unused")]
    )
    user_input: UserInputVariant = copy_field(UserInputJoystickConfig())
    use_camera_semantic_names: bool = True

    # If the simulation time exceeds this value, the step will fail with a stop
    # reason of "Timeout ...". If not provided, there's no timeout enforcement.
    # The reset_and_record_pre_episode_snapshot options may override this.
    t_max: float | None = None

    def create(self):
        return HardwareStationScenarioSimulationEnv(**vars(self))

    def __post_init__(self):
        # Ensure that the rewards all have unique names to avoid later name
        # clashes.
        reward_names = [config.name for config in self.reward_configs]
        assert (
            "total" not in reward_names
        ), "'total' is reserved for the aggregate of all rewards"
        assert len(reward_names) == len(set(reward_names)), reward_names

        # Ensure that the task predicates all have unique names to avoid later
        # name clashes.
        predicate_names = [
            predicate.name for predicate in self.task_predicate_configs
        ]
        assert (
            "all_satisfied" not in predicate_names
        ), "'all_satisfied' is reserved for the aggregate of all predicates"
        assert len(predicate_names) == len(
            set(predicate_names)
        ), predicate_names


def _resolve_summary_dir(
    save_dir: str | None, demonstration_index: int | None
) -> tuple[str, str | None, AutoTemporaryDirectory | None]:
    """Determines the appropriate logging directory for saving
    demonstration artifacts. If syncing to S3 is desired, the
    function creates a temporary local directory during runtime and
    syncs to S3 when the demonstration finishes.

    If `save_dir` is specified, `demonstration_index` must also be specified.

        If no save dir is specified, data is logged to the TEST_TMPDIR, or to
        `/tmp/sim_summary`.

    Return:
        local_summary_dir: The path where logs are saved locally.
        s3_summary_dir: The destination S3 path, if applicable; else, None.
        tmpdir_holder: A temporary directory holder if syncing to S3 is
            desired; else, None.
    """
    tmpdir_holder = None
    s3_summary_dir = None

    if save_dir is None:
        # Even if `save_dir` is not specified, i.e., a user doesn't want
        # logging, we still store the artifacts for reference.
        local_summary_dir = os.environ.get("TEST_TMPDIR", "/tmp/sim_summary")
    else:
        if demonstration_index is None:
            raise ValueError(
                "If `save_dir` is specified, `demonstration_index` must also "
                "be specified."
            )

        if is_s3_path(save_dir):
            tmpdir_holder = AutoTemporaryDirectory(prefix="sim_summary")
            local_summary_dir = tmpdir_holder.tempdir()

            if "TEST_TMPDIR" not in os.environ:
                # We can currently only test this when we are not under CI.
                assert_s3_access(save_dir, aws_profile=None)
            else:
                logging.warn(
                    f"`bazel test` cannot save to s3 path {save_dir}; "
                    f"test results will be left in {local_summary_dir} instead"
                )

            # Copies summary_dir pattern from
            # https://github.shared-services.aws.tri.global/robotics/anzu/blob/65003fb/intuitive/visuomotor/demonstrate.py#L406-L408
            s3_summary_dir = os.path.join(
                save_dir, f"demonstration_{demonstration_index}"
            )
        else:
            local_summary_dir = os.path.join(
                save_dir, f"demonstration_{demonstration_index}"
            )

    os.makedirs(local_summary_dir, exist_ok=True)
    assert os.path.isdir(local_summary_dir)
    return local_summary_dir, s3_summary_dir, tmpdir_holder


class HardwareStationScenarioSimulationEnv(AnzuEnv):
    """
    Defines an environment for simulating a robot in a scene using a specified
    Scenario to build the simulation.

    Arguments:
        robot_setup: A MultiarmRobotSetup instance dictating model and relevant
            model instances and model frames.
        simulation_scenario_package: Governs the schema class used to load the
            simulation scenario. See HardwareStationScenarioSimulationEnvConfig
            documentation for details.
        simulation_scenario_config: A SimulationScenarioConfig instance
            containing the scenario_file, scenario_name, scenario_text, and /
            or the random_seed to be used when constructing a simulation.
        diff_ik_config: A DifferentialInverseKinematicsConfig instance to
            configure the Diff IK controller for a specific robot embodiment,
            e.g., q and v limits and task frames.
        dt: A float to determine the time step for simulation.
        reward_configs: A list of reward configurations that must be a variant
            contained in RewardConfigVariant
        task_predicate_configs: A list of task predicate configurations that
            must be a variant of TaskPredicateConfigVariant.
        user_input: A user input configuration (such as joystick or footpedal)
            used to control the flow of the environment's state machine,
            allowing the user to handle pauses, resets.
        TODO(dale.mcconachie) This list has not been kept up to date.

    Optional `options` fields:
        "save_dir":
            Used to update HardwareStationScenarioSimulation.save_dir().
        "demonstration_index":
            Episode index. This is *required* if "save_dir" is supplied.
            Otherwise, will default to 0.
    """

    # TODO(dale.mcconachie) Sort the class functions to put related functions
    # together, and have the order that the code is written match the order
    # the code is executed in as much as practical.

    def __init__(
        self,
        *,
        robot_setup: MultiarmRobotSetup,
        simulation_scenario_package: str,
        simulation_scenario_config: CommonScenarioConfig,
        diff_ik_config: DifferentialInverseKinematicsConfig,
        camera_overrides: CameraOverrides,
        target_realtime_rate: float,
        dt: float,
        reward_configs: List[RewardConfigVariant],
        task_predicate_configs: List[TaskPredicateConfigVariant],
        user_input: UserInputVariant,
        skill: SkillType,
        default_language_instruction: Optional[str] = None,
        distribution_shift_info: Optional[DistributionShiftInfo] = None,
        station_name: str,
        use_camera_semantic_names: bool = True,
        t_max: float | None = None,
    ):
        if dt <= 0.0:
            raise ValueError("The simulation time step (dt) must be positive.")

        self._simulation_scenario_config = simulation_scenario_config
        self._simulation_scenario_package = simulation_scenario_package
        self._sim = HardwareStationScenarioSimulation(
            robot_setup=robot_setup,
            station_name=station_name,
            diff_ik_config=diff_ik_config,
            dt=dt,
        )

        self._camera_overrides = camera_overrides
        self._dt = dt

        self._target_step_dt = 0.0
        if target_realtime_rate > 0:
            self._target_step_dt = dt / target_realtime_rate
        elif target_realtime_rate < 0:
            raise ValueError(
                f"target_realtime_rate must be >= 0.0: {target_realtime_rate}"
            )
        # TODO(dale.mcconachie) Ideally we assert early in the construction
        # process that if `target_realtime_rate` is set, then
        # `simulation_scenario_config.target_realtime_rate` is not set. However
        # doing so involves introspecting the scenario config with all the
        # overlays/overrides applied. We defer this validation to
        # `_resolve_scenario` instead for now.

        self._reward_configs = reward_configs
        self._task_predicate_configs = task_predicate_configs

        self._skill = skill
        if self._skill is None:
            logging.warn(
                "WARNING: skill is set to None, please provide a proper value"
            )
        self._default_language_instruction = default_language_instruction
        self._distribution_shift_info = distribution_shift_info
        if self._distribution_shift_info is None:
            self._distribution_shift_info = DistributionShiftInfo()

        # User Input.
        self.user_input = user_input.create()

        self._station_name = station_name
        if self._station_name is not None:
            self._use_camera_semantic_names = use_camera_semantic_names
            self._camera_id_to_semantic_name = load_camera_id_to_semantic_name(
                self._station_name
            )
        else:
            self._use_camera_semantic_names = False
            self._camera_id_to_semantic_name = None

        # The reset_and_record_pre_episode_snapshot options may override this.
        self._t_max_per_constructor_config = t_max

        self._invalidate()

    def _maybe_update_stop_info(self, stop_info, default_reason):
        if stop_info is None:
            return
        if len(stop_info) == 0:
            return
        self._is_success = stop_info.get("is_success", None)
        self._is_retry = stop_info.get("is_retry", None)
        self._stop_reason = stop_info.get("stop_reason", default_reason)

    def get_env_metadata(self):
        return EnvMetadata(
            domain=EnvDomain.Sim,
            skill=self._skill,
            station_name=self._station_name,
            hardware_platform_type=PLATFORM_TYPE_LBM_DUAL_PANDA,
            camera_id_to_semantic_name=self._camera_id_to_semantic_name,
            distribution_shift_info=self._distribution_shift_info,
        )

    @translate_drake_simulation_failures
    def step(self, action):
        self._sim.set_env_input(action)
        self._sim.step()

        # Check user signals.
        user_stop_info = self.user_input.check_success_signals()
        self._maybe_update_stop_info(user_stop_info, "User")

        # Check action info.
        action_stop_info = action_get_stop_info(action)
        self._maybe_update_stop_info(action_stop_info, "Action")

        step_info = self._get_time_step()

        if self._target_step_dt > 0:
            # Note that we are deliberately doing additive time here, not
            # start_time + steps_taken * dt. We do this because we don't want
            # to "speed ahead" at some steps. I.e.; if step N takes dt + 0.05
            # seconds, we don't want step N+1 to take dt - 0.05 seconds; we
            # want it to take dt seconds.
            wall_t_next = self._wall_t_last + self._target_step_dt
            sleep_time = max(0, wall_t_next - time.time())
            time.sleep(sleep_time)
        self._wall_t_last = time.time()

        return step_info

    def _get_time_step(self) -> TimeStep:
        info = {}
        info["time"] = self.get_time()

        detailed_rewards = self._reward_evaluator.eval_detailed(
            root_context=self._sim.root_context
        )
        info["detailed_rewards"] = detailed_rewards
        reward = detailed_rewards[self._reward_evaluator.name]

        detailed_task_predicates = self._task_predicate.eval_detailed(
            root_context=self._sim.root_context
        )
        info["detailed_task_predicates"] = detailed_task_predicates
        plant_state = self._sim.get_plant_state()

        info["plant_state"] = plant_state
        plant_named_position = self._sim.get_plant_named_position()
        info["position"] = plant_named_position
        task_done = detailed_task_predicates[self._task_predicate.name]

        terminated = False
        truncated = False
        is_success = False
        is_retry = False

        if task_done:
            terminated = True
            # N.B. Predicate being done indicates success.
            is_success = True
            self._stop_reason = f"task_predicate[{self._task_predicate.name}]"
        elif self._is_success is not None:
            terminated = True
            is_success = self._is_success
        elif self._is_retry:
            truncated = True
            is_retry = self._is_retry
        elif self._t_max is not None and self.get_time() >= self._t_max:
            truncated = True
            is_success = False
            self._stop_reason = f"Timeout {self.get_time()} >= {self._t_max}"

        info["is_success"] = is_success
        info["is_retry"] = is_retry
        if self._stop_reason is not None:
            info["stop_reason"] = self._stop_reason

        obs = self._get_observation()

        return TimeStep(obs, reward, terminated, truncated, info)

    def _get_language_instruction(self):
        # TODO(sfeng): expand this to connect to an external source to support
        # varying language instructions.
        return self._default_language_instruction

    @translate_drake_simulation_failures
    def _get_observation(self):
        """Retrieves and then combines the robot and image components of an
        observation into a single object. If cameras have been disabled or
        replaced with ROS cameras, the image component will contain no data.
        """
        # When we're replacing cameras with ROS cameras it is explicitly so
        # that we can use the AVP pipeline to perform teleoperation; thus we
        # disable images as part of the observation to reduce overall
        # computation.
        if self._camera_overrides.replace_cameras_with_ros:
            image_set = CameraImageSetMap()
        else:
            image_set = self._sim.get_images()
            if self._use_camera_semantic_names:
                image_set = rename_camera_image_set(
                    self._camera_id_to_semantic_name, image_set
                )

        return MultiarmObservation(
            robot=self._sim.get_robot_observation(),
            visuo=image_set,
            language_instruction=self._get_language_instruction(),
        )

    def set_plant_state(self, state: dict):
        self._sim.set_plant_state(state)

    def render(self, instant: PolicyInstant = None):
        """TODO(imcmahon) this is not implemented and always returns False.
        Either implement this function or remove it completely.

        This function is intended to render the environment to a cv window if
        this environment has rendering active and returns True. Otherwise
        returns False.
        """
        return False

    def get_time(self):
        return self._sim.get_time()

    def reset_and_record_pre_episode_snapshot(
        self, *, seed: int | None = None, options: Dict[str, Any] = None
    ):
        """Resets the environment and simulation to a new state.

        Supported `options`:
          - "save_dir": A directory to save the simulation artifacts to. A
            subdirectory will be created within this folder named
            demonstration_<index> where artifacts will be stored.
          - "demonstration_index": An integer index for the demonstration
            being run. Must be set if "save_dir" is set.
        """
        if options is None:
            options = {}

        self._invalidate()

        self._is_success = None
        self._is_retry = None
        self._stop_reason = None
        self._t_max = options.get("t_max", self._t_max_per_constructor_config)

        scenario, make_hardware_station = self._resolve_scenario(
            seed=seed, options=options
        )
        self._sim.reset(scenario, make_hardware_station)

        self._reward_evaluator = TotalReward(
            name="total",
            weight=1.0,
            diagram=self._sim.diagram,
            evaluators=[
                config.create(self._sim.diagram)
                for config in self._reward_configs
            ],
        )
        item_locking_monitor = None
        if self._sim._monitor:
            item_locking_monitor = self._sim._monitor.get_monitor_by_type(
                ItemLockingMonitor
            )
        self._task_predicate = AllSatisfiedPredicate(
            name="all_satisfied",
            diagram=self._sim.diagram,
            item_locking_monitor=item_locking_monitor,
            predicates=[
                predicate.create(
                    self._sim.diagram,
                    item_locking_monitor=item_locking_monitor,
                )
                for predicate in self._task_predicate_configs
            ],
        )

        self._timing_info = TimingInfo.make_empty()
        self._timing_info.setup_start_time = get_datetime_now()
        self.user_input.reset()
        logging.info("Done moving. Wait for user")
        self.user_input.wait_for_start()
        self._timing_info.setup_end_time = get_datetime_now()

        self.user_input.print_success_signals()
        logging.info("Running")

    def _invalidate(self):
        """Sets to `None` all elements of this object that will be set by
        `reset_and_record_pre_episode_snapshot()`.
        """
        # NOTE: This is sorted by the order in which the fields are set in
        # `reset()`. In theory this is useful to ensure that we don't miss
        # something - we can scan top down in both places.
        fields = [
            # `reset_and_record_pre_episode_snapshot()` directly
            "_is_success",
            "_is_retry",
            "_stop_reason",
            "_t_max",
            "_timing_info",
            # _resolve_scenario()`
            "_summary_dir",
            "_s3_summary_dir",
            "_tempdir_holder",
            # `reset_and_record_pre_episode_snapshot()` directly
            "_reward_evaluator",
            "_task_predicate",
            "_timing_info",
        ]
        for field in fields:
            setattr(self, field, None)

    def _resolve_scenario(
        self, seed: int | None, options: Dict[str, Any]
    ) -> Any:
        summary_dir, s3_summary_dir, tempdir_holder = _resolve_summary_dir(
            save_dir=options.get("save_dir", None),
            demonstration_index=options.get("demonstration_index", None),
        )
        self._summary_dir = summary_dir
        self._s3_summary_dir = s3_summary_dir
        self._tempdir_holder = tempdir_holder

        (
            scenario_schema,
            make_hardware_station,
        ) = self._schema_and_make_hardware_station()

        simulation_scenario_config = copy.deepcopy(
            self._simulation_scenario_config
        )
        if seed is not None:
            simulation_scenario_config.random_seed = seed

        intermediaries = apply_scenario_overrides_and_sample(
            common_scenario_config=simulation_scenario_config,
            scenario_schema=scenario_schema,
            make_hardware_station=make_hardware_station,
            logging_dir=self._summary_dir,
            keyframe_logging_fps=(1.0 / self._dt),
            camera_overrides=self._camera_overrides,
        )
        self._write_scenario_intermediaries_to_disk(intermediaries)
        simulation_scenario = intermediaries.simulation_scenario

        # TODO(dale.mcconachie): Ideally this check is in `__init__` so that
        # we fail fast, but we put it here for expediency for now.
        if self._target_step_dt != 0.0:
            simulator_config = simulation_scenario.simulator_config
            if simulator_config.target_realtime_rate != 0.0:
                raise ValueError(
                    "The environment has a target_realtime_rate set; when this"
                    " is set, the simulator config must be set to 'as fast as "
                    "possible' (i.e.; target_realtime_rate=0.0). Recorded "
                    f"value: {simulator_config.target_realtime_rate}"
                )

        return simulation_scenario, make_hardware_station

    def _schema_and_make_hardware_station(self) -> tuple[type, Callable]:
        # TODO(dale.mcconachie): Move this to
        # `apply_scenario_overrides_and_sample()`?
        scenario_module = importlib.import_module(
            name="{}.hardware_station_simulation_scenario".format(
                self._simulation_scenario_package
            )
        )
        station_module = importlib.import_module(
            name="{}.hardware_station".format(
                self._simulation_scenario_package
            )
        )
        return scenario_module.Scenario, station_module.MakeHardwareStation

    def _write_scenario_intermediaries_to_disk(
        self, scenario_intermediaries: ScenarioResolutionIntermediaries
    ):
        yaml_dump(
            scenario_intermediaries.random_scenario,
            filename=os.path.join(self._summary_dir, "random_scenario.yaml"),
        )
        yaml_dump(
            scenario_intermediaries.resolved_scenario,
            filename=os.path.join(self._summary_dir, "resolved_scenario.yaml"),
        )
        yaml_dump_typed(
            scenario_intermediaries.simulation_scenario,
            filename=os.path.join(
                self._summary_dir, "simulation_scenario.yaml"
            ),
        )

    def start_episode(self):
        self._timing_info.episode_start_time = get_datetime_now()
        step_info = self._get_time_step()
        self._wall_t_last = time.time()
        return step_info

    def stop_episode(self):
        self._timing_info.episode_end_time = get_datetime_now()

    def abort_episode(self):
        pass

    def finish_and_record_post_episode_snapshot(self):
        self._sim.finish()

        if self._s3_summary_dir is not None:
            assert self._tempdir_holder is not None
            # Note that we are deliberately setting aws_profile=None to enable
            # this on lbm_eval cluster nodes which don't use AWS profiles.
            exec_s3_sync(
                src=self._summary_dir,
                dest=self._s3_summary_dir,
                aws_profile=None,
            )
            # Explicitly delete the directory to trigger removal on disk.
            del self._tempdir_holder
            self._tempdir_holder = None

        self._timing_info.teardown_start_time = get_datetime_now()
        self._timing_info.teardown_end_time = get_datetime_now()
        self._timing_info.assert_valid()
        # To match hardware.
        final_info = {
            "timing_info": self._timing_info,
            "station_name": self._station_name,
            "camera_id_to_semantic_name": self._camera_id_to_semantic_name,
            "use_camera_semantic_names": self._use_camera_semantic_names,
            "pre_experiment_data": None,
            "post_experiment_data": None,
        }
        return final_info

    def close(self):
        pass


@dc.dataclass(kw_only=True)
class _ControlledFrame:
    """Runtime data structure corresponding to one of MultiarmRobotSetup's
    specified frames to control."""

    name: str
    """The frame name given in MultiarmRobotSetup.{diff_ik_frame,head_frame}.
    """

    frame: Frame
    """The Frame object from the sim station's plant."""

    input_port: InputPort
    """The RigidTransform-valued input port for the desired pose of the
    frame, which feeds into some downstream system."""

    input_port_value: FixedInputPortValue | None = None
    """The object connected to the input port to set a value. This will be None
    until after Context-creation time."""


@dc.dataclass(kw_only=True)
class _RobotPart:
    """Runtime data structure corresponding to one of MultiarmRobotSetup's
    specified model instances that can report more than just position. I.e.;
    everything except the grippers. This dataclass should only be instantiated
    after the sim plant is finalized to ensure the values stay consistent. Note
    that not all robots and model instances will report full proprioception.
    """

    # TODO(dale.mcconachie) Discuss with Policy and Platform if grippers can be
    # folded into `joint_position`, `joint_velocity`, etc. in
    # `PosesAndGrippers` and thus the distinction between grippers and other
    # model instances can be removed.

    model_name: str
    """The model name."""

    model_instance: ModelInstanceIndex
    """The model instance in the sim plant given `model_name`."""

    tau_port: OutputPortIndex | None = None
    """The port ID for the model's measured torque. This port will only be
    assigned if the model exposes `torque_measured`. For robots without torque
    sensing, this field will remain None after reset()."""

    ext_tau_port: OutputPortIndex | None = None
    """The port ID for the model's external torque. This port will only be
    assigned if the model exposes `torque_external`. For robots without torque
    sensing, this field will remain None after reset()."""

    actual_q_port: OutputPortIndex | None = None
    """The port ID for the model's actual position. This port will always be
    assigned after reset()."""

    actual_v_port: OutputPortIndex | None = None
    """The port ID for the model's actual velocity. This port will always be
    assigned after reset()."""

    desired_q_port: OutputPortIndex | None = None
    """The port ID for the model's commanded position. Depending on the
    driver's control mode, this field, `desired_v_port`, or both should have a
    valid value after reset()."""

    desired_v_port: OutputPortIndex | None = None
    """The port ID for the model's commanded velocity. Depending on the
    driver's control mode, this field, `desired_q_port`, or both should have a
    valid value after reset()."""


# TODO(dale.mcconachie): The only thing that this class preserves through a
# reset is some stored configuration values, a meshcat instance, and
# _perf_sampling_times. Possibly this class should be rebuilt (taking meshcat
# as an input) instead of reset to avoid incomplete initialization errors etc.
class HardwareStationScenarioSimulation:
    """
    Defines a diagram for simulating a robot in a scene. Note that this class
    is not fully initialized until reset() is called for the first time.

    N.B. we do not call reset in the initializer because we have not yet been
    passed a random seed or save dir via set_random_seed() and set_save_dir()
    respectively. For architecture reasons beyond the scope of this comment we
    cannot pass those values in to the constructor.

    Arguments:
        robot_setup: A MultiarmRobotSetup instance dictating model and relevant
            model instances and model frames.
        station_name: The name of the station being simulated.
        diff_ik_config: A DifferentialInverseKinematicsConfig instance to
            configure the Diff IK controller for a specific robot embodiment,
            e.g., q and v limits and task frames.
        dt: How long the simulation should advance per call to step().
    """

    def __init__(
        self,
        *,
        robot_setup: MultiarmRobotSetup,
        station_name: str,
        diff_ik_config: DifferentialInverseKinematicsConfig,
        dt: float,
    ):
        # Check instances for required fields.
        assert isinstance(robot_setup, MultiarmRobotSetup)
        assert isinstance(diff_ik_config, DifferentialInverseKinematicsConfig)
        robot_setup.validate()
        diff_ik_config.validate()

        self._robot_setup = robot_setup
        self._station_name = station_name
        self._diff_ik_config = diff_ik_config
        self._dt = dt

        self._meshcat = Meshcat()

        # The environment variable ANZU_PERF_SAMPLING_TIMES is expected to be a
        # comma-separated list of 0 or more floating point numbers. The numbers
        # are the start times of simulation time steps that should be performed
        # with Linux `perf` sampling turned on.
        env_key = "ANZU_PERF_SAMPLING_TIMES"
        self._perf_sampling_times = set(
            float(x) for x in os.environ.get(env_key, "").split(",") if x
        )

        self._invalidate()

    def _should_sample(self, sim_t):
        """Returns True if any of the perf sampling times parsed from the
        environment variable ANZU_PERF_SAMPLING_TIMES are sufficiently close to
        the current simulation time. "Sufficiently close" means less the one
        quarter of `dt` away from the current time.
        """
        # TODO(dale.mcconachie) Revisit if this logic around self._dt is still
        # needed or if we can remove it and add appropriate unit testing.
        return any(
            [
                (abs(x - sim_t) < (self._dt / 4))
                for x in self._perf_sampling_times
            ]
        )

    @property
    def diagram(self):
        return self._diagram

    @property
    def plant(self):
        return self._plant

    @property
    def scene_graph(self):
        return self._scene_graph

    @property
    def root_context(self):
        return self._diagram_context

    @property
    def plant_context(self):
        return self._plant.GetMyContextFromRoot(self._diagram_context)

    # NOTE: All functions in between `reset()` and `step()` are used by
    # `reset()` to (re)set the simulation. They are sorted first by operating
    # on similar components, and second by the order in which they are used
    # within `reset()`.

    def reset(self, scenario: Any, make_hardware_station: Callable):
        """Resets the simulation to a new initial state."""

        # TODO(imcmahon): This reset() function very long and needs to be
        # refactored for readability. Ideally components are broken out into
        # separate classes that can be initialized independently so that there
        # are fewer cases where objects are in an intermediary invalid state.

        logging.info("Reset simulation")
        self._invalidate()

        self._scenario = scenario
        builder = DiagramBuilder()

        # Populates:
        # - self._station
        # - self._logging
        # - self._monitor
        # - self._plant
        # - self._scene_graph
        self._setup_station(
            builder=builder,
            make_hardware_station=make_hardware_station,
        )

        # Per the requested data from MultiarmRobotSetup, a mapping from each
        # model name to the corresponding _RobotPart record with the details.
        # The ports will be populated during the setup process.
        self._robot_parts: dict[str, _RobotPart] = dict()
        # Per the requested list of MultiarmRobotSetup.controlled_frame names,
        # a mapping from each frame's model instance name to the corresponding
        # _ControlledFrame record with the details. (Note that each model
        # instance can have at most one controlled frame.) This dict will be
        # populated by _setup_differential_ik().
        self._controlled_frames: dict[str, _ControlledFrame] = dict()
        # Populates:
        # - Nothing. But adds a system to the diagram that
        #   _setup_differential_ik() requires.
        self._setup_base_control(builder)
        # Populates:
        # - self._frame_T
        # - self._controlled_frames[N].name
        # - self._controlled_frames[N].frame
        # - self._controlled_frames[N].input_port
        # - self._robot_parts[N].model_name
        # - self._robot_parts[N].model_instance
        self._setup_differential_ik(builder)
        # Populates:
        # - self._robot_parts[N].model_name
        # - self._robot_parts[N].model_instance
        self._setup_head_control(builder)
        # Populates:
        # - self._robot_parts[N].ext_tau_port
        # - self._robot_parts[N].tau_port
        self._setup_and_export_torques(builder)
        # Populates:
        # - self._robot_parts[N].actual_q_port
        # - self._robot_parts[N].actual_v_port
        self._export_actual_positions_and_velocities(builder)
        # Populates:
        # - self._robot_parts[N].desired_q_port
        # - self._robot_parts[N].desired_v_port
        self._export_commanded_positions_and_velocities(builder)
        # Populates:
        # - self._gripper_instances
        # - self._gripper_input_ports
        # - self._gripper_output_ports
        # TODO(dale.mcconachie) `_gripper_instances`, `_gripper_input_ports`,
        # `_gripper_output_ports`, and `_gripper_commands` should all be
        # combined into a single dataclass similar to `_RobotPart` and/or
        # `_ControlledFrame`.
        self._setup_grippers(builder)

        self._diagram = builder.Build()
        self._diagram_context = self._diagram.CreateDefaultContext()
        self._plant_context = self._plant.GetMyContextFromRoot(
            self._diagram_context
        )
        # Populates:
        # - self._controlled_frames[N].input_port_value
        # - self._gripper_commands[N]
        self._add_policy_action_input_ports(builder)

        # Sample any remaining random elements of the context.
        generator = RandomGenerator(self._scenario.random_seed)
        self._diagram.SetRandomContext(self._diagram_context, generator)
        self._simulator = Simulator(self._diagram, self._diagram_context)
        # TODO(imcmahon): Refactor the simulator configuration, item locking,
        # and deformable sim config in
        # //sim/station/hardware_station_simulation.py into a reusable function
        # to be called here.
        ApplySimulatorConfig(self._scenario.simulator_config, self._simulator)
        # Apply deformable simulation configurations if they are present. It's
        # important to do this *after* the scenario is resolved so that the
        # no threads are spawned for deformable simulation while sampling the
        # scenario (which itself might be using thread parallelism).
        if self._scenario.deformable_sim_config is not None:
            ApplyDeformableSimConfig(
                self._scenario.deformable_sim_config, self._plant
            )
        if self._monitor:
            self._simulator.set_monitor(self._monitor.Monitor)

        # Set the initial inputs for all systems to the values in the fully
        # sampled scenario.
        self._set_diff_ik_initial_position()
        self._initialize_policy_action_input_ports()
        self._commands_initialized = False

        # Tracks the configured simulation start time and the total number of
        # steps that we have taken since that start time. Used to ensure that
        # we don't get dither when we step the simulator.
        self._start_time = self._diagram_context.get_time()
        self._steps_taken = 0

        # TODO(dale.mcconachie) Consider making this part of the files logged
        # into env._summary_dir.
        if logging.getLogger("drake").getEffectiveLevel() <= logging.DEBUG:
            graphviz_filename = "/tmp/multiarm_simulations.dot"
            with open(graphviz_filename, "w", encoding="utf-8") as f:
                # TODO(dale.mcconachie) What is the right syntax to split the
                # plant I/O into two blocks?
                f.write(self._diagram.GetGraphvizString())

    def _invalidate(self):
        """Sets to `None` all elements of this object that will be set by
        `reset()`.
        """
        # NOTE: This is sorted by the order in which the fields are set in
        # `reset()`. In theory this is useful to ensure that we don't miss
        # something - we can scan top down in both places.
        fields = [
            # `reset()` directly
            "_scenario",
            # setup_station()
            "_station",
            "_logging",
            "_monitor",
            "_plant",
            "_scene_graph",
            # reset() directly creates these, then populated elsewhere.
            "_robot_parts",
            "_controlled_frames",
            # _setup_differential_ik()
            "_frame_T",
            # _setup_grippers()
            "_gripper_instances",
            "_gripper_input_ports",
            "_gripper_output_ports",
            # reset() directly
            "_diagram",
            "_diagram_context",
            "_plant_context",
            # _add_policy_action_input_ports()
            "_gripper_commands",
            # reset() directly
            "_simulator",
            "_commands_initialized",
            "_start_time",
            "_steps_taken",
        ]
        for field in fields:
            setattr(self, field, None)

    def _setup_station(self, *, builder, make_hardware_station):
        self._station, self._logging, self._monitor = make_hardware_station(
            self._scenario, meshcat=self._meshcat, export_ports=True
        )
        self._plant = self._station.plant()
        self._scene_graph = self._station.scene_graph()

        builder.AddNamedSystem("station", self._station)
        builder.ExportOutput(
            self._station.GetOutputPort("plant.contact_results"),
            "contact_results",
        )

    def _setup_base_control(self, builder):
        """Sets up the base control system for the robot, if applicable. Must
        be called before `_setup_differential_ik()`.
        """
        if self._robot_setup.base_name is not None:
            # NOTE: We use a deferred import here to avoid problems with the
            # open sourcing effort.
            from anzu.robot_bridge.hub.cc import RainbowRby1BaseDriver
            from anzu.sim.rainbow_rby1.cc import RainbowRby1BaseCommandSystem

            logging.info("Setting up base control system.")

            base_command_system = builder.AddNamedSystem(
                _BASE_COMMAND_SYSTEM_NAME,
                RainbowRby1BaseCommandSystem(
                    # TODO(dale.mcconachie) Add CI test that ensures that this
                    # value and the value consumed by rby1_control_main (loaded
                    # by hub builder) cannot accidentally diverge.
                    # NOTE: This is a different config from the the model
                    # driver config, though they share the same name within
                    # their respective namespaces.
                    RainbowRby1BaseDriver().position_error_gain
                ),
            )

            base_name = self._robot_setup.base_name
            builder.Connect(
                base_command_system.get_output_port(),
                self._station.GetInputPort(f"{base_name}.body_velocity"),
            )
            builder.Connect(
                self._station.GetOutputPort(f"{base_name}.position_measured"),
                base_command_system.get_position_measured_input_port(),
            )

    def _setup_differential_ik(self, builder):
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._plant
        assert self._controlled_frames is not None

        logging.info(
            "Setting up DifferentialInverseKinematics controller "
            "with collision avoidance."
        )

        # TODO(#16792) Enable the task frame used to communicate with a policy
        # to be different from the frame used to communicate with mid-level
        # controllers.
        self._frame_T = GetScopedFrameByName(
            self.plant, self._diff_ik_config.pose_stream_config.task_frame
        )

        desired_poses_bus = BusCreator()
        for frame_name in self._robot_setup.diff_ik_frames:
            frame = GetScopedFrameByName(self._plant, frame_name)
            model_instance = frame.model_instance()
            model_name = self._plant.GetModelInstanceName(model_instance)
            assert model_name not in self._controlled_frames, model_name
            input_port = desired_poses_bus.DeclareAbstractInputPort(
                name=frame_name,
                model_value=Value[RigidTransform](),
            )
            self._controlled_frames[model_name] = _ControlledFrame(
                name=frame_name,
                frame=frame,
                input_port=input_port,
            )
        builder.AddNamedSystem("DesiredPosesBus", desired_poses_bus)

        for model_name in self._robot_setup.diff_ik_names:
            self._robot_parts[model_name] = _RobotPart(
                model_name=model_name,
                model_instance=self._plant.GetModelInstanceByName(model_name),
            )

        # Choose which differential IK implementation to use. For open-source
        # stations, we'll always use the open-source differential IK even when
        # the full Anzu dependencies are available. This improves our test
        # uniformity of simulation results and test leverage no matter the
        # runtime environment.
        is_open_source = self._station_name in ["cabot", "riverway"]
        if is_open_source:
            helper = importlib.import_module(
                name="anzu.operational_space_control.differential_inverse_kinematics_controller_helper"  # noqa
            )
        else:
            helper = importlib.import_module(
                name="anzu.operational_space_control.cc",
            )
        diff_ik_controller = builder.AddNamedSystem(
            "DifferentialIK",
            helper.MakeDifferentialInverseKinematicsControllerForStation(
                station_name=self._station_name,
                postfix=[self._robot_setup.station_postfix],
                active_model_instances=self._robot_setup.diff_ik_names,
                joint_limits=None,
                timestep=self._plant.time_step(),
                config_file=resolve_path(self._diff_ik_config.config_file),
                allow_finger_table_collision=(
                    self._diff_ik_config.allow_finger_table_collision
                ),
            ),
        )

        diff_ik_sys = diff_ik_controller.differential_inverse_kinematics()
        diff_ik_plant = diff_ik_sys.plant()
        state_mux = self._setup_state_multiplexer(builder, diff_ik_plant)

        command_demux = builder.AddNamedSystem(
            "ComandDemultiplexer",
            MakeCommandDemultiplexer(
                plant=diff_ik_plant,
                model_instance_names=self._robot_setup.diff_ik_names,
            ),
        )
        for model_name in self._robot_setup.diff_ik_names:
            if model_name == self._robot_setup.base_name:
                # The base model is position and velocity controlled by a
                # mid-level controller, so we have to wire it differently than
                # other models.
                command_system = builder.GetSubsystemByName(
                    _BASE_COMMAND_SYSTEM_NAME
                )
                builder.Connect(
                    command_demux.GetOutputPort(model_name + ".position"),
                    command_system.get_position_input_port(),
                )
                builder.Connect(
                    command_demux.GetOutputPort(model_name + ".velocity"),
                    command_system.get_velocity_input_port(),
                )
            else:
                # For all other models, we connect the command demux directly
                # to the implicit PD control inputs of the multibody plant.
                builder.Connect(
                    command_demux.GetOutputPort(model_name + ".position"),
                    self._station.GetInputPort(model_name + ".position"),
                )

        nominal_posture = builder.AddNamedSystem(
            "NominalPosture",
            ConstantVectorSource(
                self._get_initial_posture_for_differential_ik(diff_ik_plant)
            ),
        )
        builder.Connect(
            nominal_posture.get_output_port(),
            diff_ik_controller.GetInputPort("nominal_posture"),
        )
        builder.Connect(
            diff_ik_controller.GetOutputPort("commanded_position"),
            command_demux.GetInputPort("position"),
        )
        builder.Connect(
            diff_ik_controller.GetOutputPort("commanded_velocity"),
            command_demux.GetInputPort("velocity"),
        )
        builder.Connect(
            state_mux.GetOutputPort("state"),
            diff_ik_controller.GetInputPort("estimated_state"),
        )
        builder.Connect(
            desired_poses_bus.get_output_port(),
            diff_ik_controller.GetInputPort("desired_poses"),
        )
        return diff_ik_controller

    def _setup_state_multiplexer(self, builder, diff_ik_plant):
        """Sets up a multiplexer that takes the output from each model driver
        and combines them into a single state vector for the differential IK
        controller.
        """
        state_mux = builder.AddNamedSystem(
            "StateMultiplexer", MakeStateMultiplexer(plant=diff_ik_plant)
        )

        # Connect each actuated model's state port with `state_mux`.
        for i in range(state_mux.num_input_ports()):
            input_port = state_mux.get_input_port(i)
            if input_port.size() > 0:
                input_port_name = input_port.get_name()
                logging.debug(f"Connecting input port {input_port_name}")
                assert input_port_name.endswith(".state")
                model_name = input_port_name[: -len(".state")]

                # NOTE: The DiffIk plant can be (and sometimes is) different
                # than the station (simulation) plant due to things like a
                # simplified base model (planar). Thus we take the driver state
                # output rather than than taking state directly from the
                # station plant when we can.

                if model_name in self._robot_setup.gripper_names:
                    # The gripper model drivers output a 2 element state vector
                    # (i.e.; 1 DoF) while the plant state for the gripper is 4
                    # elements (i.e.; 2 DoF - one per finger). Thus we must
                    # take the plant state directly.
                    gripper_port = self._station.GetOutputPort(
                        f"plant.{model_name}_state"
                    )
                    builder.Connect(gripper_port, input_port)
                else:
                    position_port = self._station.GetOutputPort(
                        f"{model_name}.position_measured"
                    )

                    velocity_names = [
                        # The Panda/IIWA driver exports "estimated" velocity.
                        f"{model_name}.velocity_estimated",
                        # The Rainbow drivers export "measured" velocity.
                        f"{model_name}.velocity_measured",
                    ]
                    for name in velocity_names:
                        if self._station.HasOutputPort(name):
                            velocity_port = self._station.GetOutputPort(name)
                            break
                    else:
                        available_ports = "\n"
                        for i in range(self._station.num_output_ports()):
                            port = self._station.get_output_port(i)
                            available_ports += f" - {port.get_name()}\n"
                        raise RuntimeError(
                            "None of the expected velocity ports "
                            f"{velocity_names} were found for {model_name} "
                            f"Available ports:{available_ports}"
                        )

                    mux_sizes = [position_port.size(), velocity_port.size()]
                    mux = builder.AddNamedSystem(
                        f"DiffIKStateMux.{model_name}", Multiplexer(mux_sizes)
                    )
                    builder.Connect(position_port, mux.get_input_port(0))
                    builder.Connect(velocity_port, mux.get_input_port(1))
                    builder.Connect(mux.get_output_port(), input_port)
            else:
                logging.debug(f"Skipping input port {input_port.get_name()}")
        return state_mux

    def _get_initial_posture_for_differential_ik(self, control_plant):
        """Uses the initial positions of the robot models in the sim plant to
        get the initial posture for the controlled models of the differential
        IK plant. This is necessary because the differential IK controller does
        not have an affordance for setting the initial posture of the
        controlled models via YAML or any other such configuration.
        """
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._plant
        assert self._robot_parts

        # TODO(dale.mcconachie) Is using this default context correct?
        # Shouldn't we be using the initial positions defined in the
        # simulation scenario? (Maybe the plant already has those set by this
        # point and has sampled any remaining randomness?)
        sim_plant_default_context = self._plant.CreateDefaultContext()
        initial_posture = np.zeros(control_plant.num_positions())

        for model_name, robot_part in self._robot_parts.items():
            if model_name == self._robot_setup.base_name:
                q0 = self._extract_base_se2_pose(
                    sim_plant_default_context, model_name
                )
            else:
                q0 = self._plant.GetPositions(
                    sim_plant_default_context,
                    robot_part.model_instance,
                )
            control_plant.SetPositionsInArray(
                control_plant.GetModelInstanceByName(model_name),
                q0,
                initial_posture,
            )

        return initial_posture

    def _extract_base_se2_pose(self, sim_plant_default_context, model_name):
        """Project the simulation's Cartesian SE(3) pose, from the sim
        plant's default context for the given model name, down to SE(2)
        along/about its Z axis to (+x: forward, +y: left,
        tetha: ccw-from-forward)).
        This uses the base driver to extract the SE(2) pose from the
        full sim plant state.
        """
        state_output_port = self._plant.get_state_output_port()
        state = state_output_port.Eval(sim_plant_default_context)
        driver = self._station.GetSubsystemByName(model_name)
        driver_context = driver.CreateDefaultContext()
        estimated_state_port = driver.GetInputPort("estimated_state")
        fix_port(estimated_state_port, driver_context, state, resolve=False)
        pose = driver.GetOutputPort("position_measured").Eval(driver_context)
        assert len(pose) == 3, f"Expected a 3-element SE(2) pose, got {pose}."
        return pose

    def _set_diff_ik_initial_position(self):
        """Sets the initial position of the diff IK plant to the same position
        as the initial position of the robot models in the sim plant. Should
        only be called after all randomization has been resolved.
        """
        # TODO(dale.mcconachie) Is there a way to enforce the above "should"?

        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._diagram
        assert self._diagram_context

        controller = self._diagram.GetSubsystemByName("DifferentialIK")
        context = controller.GetMyMutableContextFromRoot(self._diagram_context)
        initial_position = self._get_initial_posture_for_differential_ik(
            controller.differential_inverse_kinematics().plant()
        )
        controller.set_initial_position(context, initial_position)

    def _setup_head_control(self, builder):
        if self._robot_setup.head_frame is not None:
            frame = GetScopedFrameByName(
                self._plant, self._robot_setup.head_frame
            )
            model_instance = frame.model_instance()
            model_name = self._plant.GetModelInstanceName(model_instance)
            assert model_name not in self._controlled_frames, model_name

            self._robot_parts[model_name] = _RobotPart(
                model_name=model_name,
                model_instance=model_instance,
            )

            head_control = builder.AddNamedSystem(
                "HeadControl",
                make_head_control_system(self._plant, model_name),
            )
            input_port = head_control.GetInputPort("X_TH_desired")
            self._controlled_frames[model_name] = _ControlledFrame(
                name=self._robot_setup.head_frame,
                frame=frame,
                input_port=input_port,
            )
            builder.Connect(
                self._station.GetOutputPort(f"{model_name}.position_measured"),
                head_control.GetInputPort("q_current"),
            )
            builder.Connect(
                head_control.get_output_port(),
                self._station.GetInputPort("head.position"),
            )

    def _setup_and_export_torques(self, builder):
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._robot_parts
        assert self._station
        assert self._plant

        for model_name, robot_part in self._robot_parts.items():

            def add_zoh_system_and_export(
                output_port_name,
                exported_output_name,
            ):
                system_output_port = self._station.GetOutputPort(
                    output_port_name
                )
                zoh = builder.AddSystem(
                    ZeroOrderHold(
                        self._plant.time_step(), system_output_port.size()
                    )
                )
                builder.Connect(system_output_port, zoh.get_input_port())
                return builder.ExportOutput(
                    zoh.get_output_port(),
                    exported_output_name,
                )

            # To be similar to hardware where torque values are discretely
            # sampled, add a ZOH on both arm_ext_tau and arm_tau.
            # Note: This is relevant to Anzu issue 12109.
            # https://github.shared-services.aws.tri.global/robotics/anzu/issues/12109
            if self._station.HasOutputPort(
                f"{model_name}.torque_external"
            ) and self._station.HasOutputPort(f"{model_name}.torque_measured"):
                robot_part.ext_tau_port = add_zoh_system_and_export(
                    f"{model_name}.torque_external",
                    f"{model_name}.arm_ext_tau",
                )
                robot_part.tau_port = add_zoh_system_and_export(
                    f"{model_name}.torque_measured", f"{model_name}.arm_tau"
                )

    def _export_actual_positions_and_velocities(self, builder):
        for model_name, robot_part in self._robot_parts.items():
            position_name = f"{model_name}.position_measured"
            if self._station.HasOutputPort(position_name):
                robot_part.actual_q_port = builder.ExportOutput(
                    self._station.GetOutputPort(position_name), position_name
                )
            else:
                logging.info(f"No position_measured port for {model_name}")

            velocity_names = [
                # The Panda/IIWA driver exports "estimated" velocity.
                f"{model_name}.velocity_estimated",
                # The Rainbow drivers export "measured" velocity.
                f"{model_name}.velocity_measured",
            ]
            for velocity_name in velocity_names:
                if self._station.HasOutputPort(velocity_name):
                    robot_part.actual_v_port = builder.ExportOutput(
                        self._station.GetOutputPort(velocity_name),
                        velocity_name,
                    )
                    break
            else:
                raise RuntimeError(
                    f"None of the expected velocity ports {velocity_names} "
                    f"were found for {model_name}."
                )

    def _export_commanded_positions_and_velocities(self, builder):
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._robot_parts
        assert self._station

        for model_name, robot_part in self._robot_parts.items():
            position_name = f"{model_name}.position_commanded"
            if self._station.HasOutputPort(position_name):
                robot_part.desired_q_port = builder.ExportOutput(
                    self._station.GetOutputPort(position_name), position_name
                )
            velocity_name = f"{model_name}.velocity_commanded"
            if self._station.HasOutputPort(velocity_name):
                robot_part.desired_v_port = builder.ExportOutput(
                    self._station.GetOutputPort(velocity_name), velocity_name
                )

    def _setup_grippers(self, builder):
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._plant
        assert self._station

        self._gripper_instances = {
            gripper_name: self._plant.GetModelInstanceByName(gripper_name)
            for gripper_name in self._robot_setup.gripper_names
        }

        # Setup the input ports.
        self._gripper_input_ports = dict()
        for gripper_name in self._robot_setup.gripper_names:
            # Note: The purpose of attaching a SlewRateSystem to a gripper is
            # to close the Sim2Real gap as the real hardware couldn't do that
            # as it does in simulations. Per discussion in the lbm-sim channel,
            # we (Jeremy, Zach) decided to leave those numbers hard-coded now.
            period_sec = 1 / 200.0
            width_dot_max = 0.25  # m/s
            slew_rate = builder.AddSystem(
                SlewRateSystem(
                    period_sec,
                    v_lower=[-width_dot_max],
                    v_upper=[width_dot_max],
                )
            )
            builder.Connect(
                slew_rate.GetOutputPort("q_limited"),
                self._station.GetInputPort(gripper_name + ".position"),
            )
            self._gripper_input_ports[gripper_name] = slew_rate.GetInputPort(
                "q"
            )

        # Setup the output ports used to communicate with a policy.
        self._gripper_output_ports = dict()
        for gripper_name in self._robot_setup.gripper_names:
            # The model driver outputs the gripper state as a 1-DoF joint
            # rather than the 2-DoF joint used by the plant, but it doesn't
            # export the position and velocity measured as separate ports like
            # other drivers do so extract the position element of the gripper
            # state for each gripper.
            # TODO(dale.mcconachie) Update the gripper driver to match the
            # other drivers port output formats.
            state_port = self._station.GetOutputPort(gripper_name + ".state")
            assert state_port.size() == 2
            # 1 input port of size 2; 1 output port of size 1 that extracts
            # the first element of the input port.
            # TODO(dale.mcconachie) Consider exporting the velocity as well to
            # match the exports for the arms.
            selector_params_yaml = {
                "inputs": [{"name": "state", "size": 2}],
                "outputs": [
                    {
                        "name": "position",
                        "selections": [
                            {"input_port_index": 0, "input_offset": 0}
                        ],
                    }
                ],
            }
            selector_params = yaml_load_typed(
                data=yaml_dump(selector_params_yaml), schema=SelectorParams
            )
            selector = builder.AddNamedSystem(
                f"GripperPositionSelector.{gripper_name}",
                Selector(selector_params),
            )

            builder.Connect(state_port, selector.get_input_port())
            self._gripper_output_ports[gripper_name] = builder.ExportOutput(
                selector.get_output_port(), f"{gripper_name}.position_measured"
            )

    def _add_policy_action_input_ports(self, builder):
        """Adds `FixedInputPortValue` objects that will consume inputs from a
        policy. The initial values of these ports is irrelevant as they will be
        set to the correct values in `_initialize_policy_action_input_ports()`
        once the full context has been sampled.
        """
        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._controlled_frames
        assert self._diagram_context
        assert self._gripper_input_ports

        for controlled_frame in self._controlled_frames.values():
            controlled_frame.input_port_value = fix_port(
                controlled_frame.input_port,
                self._diagram_context,
                RigidTransform(),
            )
        self._gripper_commands = dict()
        for gripper_name in self._robot_setup.gripper_names:
            self._gripper_commands[gripper_name] = fix_port(
                self._gripper_input_ports[gripper_name],
                self._diagram_context,
                np.zeros(self._gripper_input_ports[gripper_name].size()),
            )

    def _initialize_policy_action_input_ports(self):
        """Sets the initial commanded values for all policy inputs to the
        current state of the simulation. Should only be called after all
        randomization has been resolved.

        This is required to be called before the first call to
        `get_robot_observation()` so that the reported desired values match the
        current state of the simulation.
        """
        # TODO(dale.mcconachie) Is there a way to enforce the above "should"?

        # This function is called as part of the reset() process, so the object
        # as a whole is in an invalid state. Assert that the components that we
        # expect to be set up are indeed set up.
        assert self._controlled_frames
        assert self._plant
        assert self._plant_context
        assert self._frame_T
        assert self._gripper_instances
        assert self._gripper_commands

        for controlled_frame in self._controlled_frames.values():
            # The DiffIk system's input is in task frame, so make sure that we
            # do that transformation.
            X_TC = self._plant.CalcRelativeTransform(
                self._plant_context, self._frame_T, controlled_frame.frame
            )
            controlled_frame.input_port_value.GetMutableData().set_value(X_TC)
        for gripper_name in self._robot_setup.gripper_names:
            # TODO(imcmahon): Consolidate this gripper calculation with
            # the one performed in get_robot_observation().
            gripper_positions = self._plant.GetPositions(
                self._plant_context,
                self._gripper_instances[gripper_name],
            )
            assert len(gripper_positions) == 2
            self._gripper_commands[gripper_name].GetMutableData().set_value(
                gripper_positions[1] - gripper_positions[0]
            )

    # NOTE: All functions in between `reset()` and `step()` are used by
    # `reset()` to (re)set the simulation. They are sorted first by operating
    # on similar components, and second by the order in which they are used
    # within `reset()`.

    def step(self):
        assert self._commands_initialized
        t_now = self._diagram_context.get_time()
        self._steps_taken += 1
        t_next = self._start_time + self._steps_taken * self._dt
        assert t_next > t_now, (t_now, t_next, self._steps_taken)
        # Note that the monitor can cause `AdvanceTo` to terminate early, thus
        # we must check for that happening and call `AdvanceTo` again until the
        # full time has elapsed.
        with (
            scoped_perf_sampling()
            if self._should_sample(t_now)
            else nullcontext()
        ):
            while self._diagram_context.get_time() < t_next:
                if self._monitor:
                    self._monitor.HandleExternalUpdates(
                        self._simulator.get_mutable_context()
                    )
                self._simulator.AdvanceTo(t_next)
            self._commands_initialized = False

    def get_time(self):
        return self._diagram_context.get_time()

    def get_images(self):
        images = CameraImageSetMap()
        for camera in self._scenario.cameras:
            rgb = None
            depth = None
            label = None
            if camera.rgb:
                rgb = self._get_one_image(
                    camera=camera,
                    port_name="color_image",
                )
            if camera.depth:
                depth = self._get_one_image(
                    camera=camera,
                    port_name="depth_image_16u",
                )
            if camera.label:
                label = self._get_one_image(
                    camera=camera,
                    port_name="label_image",
                )
            image = CameraImageSet(
                rgb=rgb,
                depth=depth,
                label=label,
            )
            images[camera.name] = image
        return images

    def _get_one_image(self, *, camera, port_name):
        # Compute X_TC (the sensor pose in the task frame).
        X_TW = self._plant.CalcRelativeTransform(
            self._plant_context, self._frame_T, self._plant.world_frame()
        )
        camera_sys = self._station.GetSubsystemByName(
            f"rgbd_sensor_{camera.name}"
        )
        X_WB = camera_sys.body_pose_in_world_output_port().Eval(
            camera_sys.GetMyContextFromRoot(self._diagram_context),
        )
        if port_name.startswith("depth"):
            # TODO(jeremy.nimmer) The "default" camera is unsound, we should be
            # using the kinematic parameters from the context.
            render_camera = camera_sys.default_depth_render_camera().core()
        else:
            # TODO(jeremy.nimmer) Ditto.
            render_camera = camera_sys.default_color_render_camera().core()
        # TODO(dale.mcconachie) Have drake do all this math internally via a
        # single call to CalcRelativeTransform if possible.
        X_TC = X_TW @ X_WB @ render_camera.sensor_pose_in_camera_body()

        # Render the image.
        if camera.fisheye_distortion is not None:
            image_sys = self._station.GetSubsystemByName(
                f"fisheye_distortion_{camera.name}"
            )
        else:
            image_sys = camera_sys
        image_drake = image_sys.GetOutputPort(port_name).Eval(
            image_sys.GetMyContextFromRoot(self._diagram_context),
        )

        # Look up K (the intrinsics).
        if camera.fisheye_distortion is not None:
            K = camera.fisheye_distortion.get_K()
            # TODO(sfeng): add D to CameraImageSet as well.
        else:
            K = render_camera.intrinsics().intrinsic_matrix().copy()

        # Return the required struct type and image shape.
        if port_name == "color_image":
            rgb = image_drake.data[..., :3].copy()
            return CameraRgbImage(
                array=rgb,
                K=K,
                X_TC=X_TC,
            )
        if port_name == "depth_image_16u":
            depth = image_drake.data.squeeze(2).copy()
            return CameraDepthImage(
                array=depth,
                K=K,
                X_TC=X_TC,
            )
        if port_name == "label_image":
            # Replace kEmpty with zero.
            label = image_drake.data.squeeze(2).copy()
            empty = label == int(RenderLabel.kEmpty)
            label[empty] = int(RenderLabel(0))
            return CameraLabelImage(
                array=label,
                K=K,
                X_TC=X_TC,
            )
        raise NotImplementedError(port_name)

    def _calculate_wrench(self, *, model_name, model_instance, tau):
        # Follows the behavior found here:
        # https://github.shared-services.aws.tri.global/robotics/anzu/blob/70c7b27/intuitive/visuomotor/demo/robot_control_main.py#L392-L405
        Jv_TG_full = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self._controlled_frames[model_name].frame,
            p_BoBp_B=[0, 0, 0],
            frame_A=self._frame_T,
            frame_E=self._frame_T,
        )

        # Downselect to just the velocities (in line with
        # JacobianWrtVariable.kV).
        selector_matrix = self._plant.MakeStateSelectorMatrix(
            self._plant.GetJointIndices(model_instance)
        )
        plant_dof = self._plant.num_positions()
        model_dof = self._plant.num_positions(model_instance)
        selector_matrix = selector_matrix[model_dof:, plant_dof:]
        # Downselect the Jacobian to just the DOF associated with this model.
        Jv_TG_transpose = selector_matrix @ Jv_TG_full.T

        return np.linalg.lstsq(Jv_TG_transpose, tau, rcond=-1)[0]

    def get_robot_observation(self):
        """Retrieves the robot's proprioception - i.e.; its own joint
        positions, velocities torques, wrenches, forward kinematics, etc. as
        well as commanded versions of the same.
        """
        # Note: Not every robot has everything calculated; some of them will be
        # left as empty dictionaries if the calculation is irrelevant.
        q_actual = dict()
        v_actual = dict()
        tau_actual = dict()
        tau_ext_actual = dict()
        gripper_widths_actual = dict()
        # Frame-based.
        poses_actual = dict()
        wrench_actual = dict()
        external_wrench_actual = dict()

        q_desired = dict()
        v_desired = dict()
        tau_desired = dict()
        tau_ext_desired = dict()
        gripper_widths_desired = dict()
        # Frame-based.
        poses_desired = dict()
        wrench_desired = dict()
        external_wrench_desired = dict()

        for model_name, controlled_frame in self._controlled_frames.items():
            poses_actual[model_name] = self._plant.CalcRelativeTransform(
                self._plant_context, self._frame_T, controlled_frame.frame
            )
            # The poses that were passed in to the input port are already in
            # task frame.
            poses_desired[model_name] = copy.deepcopy(
                controlled_frame.input_port_value.GetMutableData().get_value()
            )

        for model_name, robot_part in self._robot_parts.items():
            # q and v.
            q_actual[model_name] = self._diagram.get_output_port(
                robot_part.actual_q_port
            ).Eval(self._diagram_context)
            if robot_part.desired_q_port is not None:
                q_desired[model_name] = self._diagram.get_output_port(
                    robot_part.desired_q_port
                ).Eval(self._diagram_context)
            v_actual[model_name] = self._diagram.get_output_port(
                robot_part.actual_v_port
            ).Eval(self._diagram_context)
            if robot_part.desired_v_port is not None:
                v_desired[model_name] = self._diagram.get_output_port(
                    robot_part.desired_v_port
                ).Eval(self._diagram_context)

            # Tau and external tau for torque-control robots only.
            if (
                robot_part.tau_port is not None
                and robot_part.ext_tau_port is not None
            ):
                tau_actual[model_name] = self._diagram.get_output_port(
                    robot_part.tau_port
                ).Eval(self._diagram_context)
                tau_ext_actual[model_name] = self._diagram.get_output_port(
                    robot_part.ext_tau_port
                ).Eval(self._diagram_context)

                # Follows the behavior found here:
                # https://github.shared-services.aws.tri.global/robotics/anzu/blob/70c7b27/intuitive/visuomotor/demo/robot_control_main.py#L888-L891
                wrench_actual[model_name] = self._calculate_wrench(
                    model_name=model_name,
                    model_instance=robot_part.model_instance,
                    tau=tau_actual[model_name],
                )
                external_wrench_actual[model_name] = self._calculate_wrench(
                    model_name=model_name,
                    model_instance=robot_part.model_instance,
                    tau=tau_ext_actual[model_name],
                )

                # Follows the behavior found here:
                # https://github.shared-services.aws.tri.global/robotics/anzu/blob/70c7b27/intuitive/visuomotor/demo/robot_control_main.py#L874-L875
                tau_nan = np.full_like(q_desired[model_name], np.nan)
                tau_desired[model_name] = tau_nan
                tau_ext_desired[model_name] = tau_nan
                # Follows the behavior found here:
                # https://github.shared-services.aws.tri.global/robotics/anzu/blob/70c7b27/intuitive/visuomotor/demo/robot_control_main.py#L898-L900
                F_nan = np.full(6, np.nan)
                wrench_desired[model_name] = F_nan
                external_wrench_desired[model_name] = F_nan

        for gripper_name in self._robot_setup.gripper_names:
            port = self._diagram.get_output_port(
                self._gripper_output_ports[gripper_name]
            )
            # The port outputs a vector of size 1, so strip that down to just
            # the scalar value.
            gripper_widths_actual[gripper_name] = port.Eval(
                self._diagram_context
            ).item()

            # N.B. We don't need to make a copy of the data here because
            # we are extracting a scalar directly. If that ever changes to
            # a vector, we will presumably need to be making a deep copy here.
            gripper_widths_desired[gripper_name] = (
                self._gripper_commands[gripper_name]
                .GetMutableData()
                .get_value()[0]
            )

        return PosesAndGrippersActualAndDesired(
            actual=PosesAndGrippers(
                poses=poses_actual,
                joint_position=q_actual,
                joint_velocity=v_actual,
                joint_torque=tau_actual,
                joint_torque_external=tau_ext_actual,
                grippers=gripper_widths_actual,
                wrench=wrench_actual,
                external_wrench=external_wrench_actual,
            ),
            desired=PosesAndGrippers(
                poses=poses_desired,
                joint_position=q_desired,
                joint_velocity=v_desired,
                joint_torque=tau_desired,
                joint_torque_external=tau_ext_desired,
                grippers=gripper_widths_desired,
                wrench=wrench_desired,
                external_wrench=external_wrench_desired,
            ),
            version=CURRENT_VERSION,
        )

    def set_env_input(self, env_input):
        assert isinstance(env_input, PosesAndGrippers)

        for model_name, controlled_frame in self._controlled_frames.items():
            # We assume that the poses given to us are already in task frame.
            controlled_frame.input_port_value.GetMutableData().set_value(
                env_input.poses[model_name]
            )

        for gripper_name in self._robot_setup.gripper_names:
            gripper_width = env_input.grippers[gripper_name]
            # TODO(imcmahon): Obtain the gripper minimum width from the
            # MultibodyPlant or URDF / SDF.
            if gripper_width < 0.0:
                gripper_width = 0.0
            self._gripper_commands[gripper_name].GetMutableData().set_value(
                [gripper_width]
            )

        self._commands_initialized = True

    def get_plant_named_position(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Return the mapping
        {model_instance_name:
            {joint_name_1: joint_1_values,
             joint_name_2: joint_2_values,
                     ...
            }
        }
        This can be used in setting the initial position in the scenario.
        """
        ret = {}
        plant = self._plant
        q = plant.GetPositions(self._plant_context)
        for i in range(plant.num_model_instances()):
            model_instance_index = ModelInstanceIndex(i)
            model_instance_name = plant.GetModelInstanceName(
                model_instance_index
            )
            named_model_q = {}
            for joint_index in plant.GetJointIndices(model_instance_index):
                joint = plant.get_joint(joint_index)
                if joint.num_positions() > 0:
                    named_model_q[joint.name()] = q[
                        joint.position_start() : joint.position_start()
                        + joint.num_positions()
                    ].tolist()
            if len(named_model_q) > 0:
                ret[model_instance_name] = named_model_q
        return ret

    def get_plant_state(self) -> dict:
        """
        Return the state (q, v) of the simulator.
        """
        plant = self._plant
        context = self._plant_context
        q = plant.GetPositions(context)
        v = plant.GetVelocities(context)
        # The plant also contains an abstract state, but I am not sure of its
        # type or content.
        return {"q": q, "v": v}

    def set_plant_state(self, plant_state: dict):
        plant = self._plant
        context = self._plant_context
        plant.SetPositions(context, plant_state["q"])
        plant.SetVelocities(context, plant_state["v"])
        # The value at the desired ports depend on self._plant_context
        self._set_desired_ports()

    def finish(self):
        if self._logging is not None:
            self._logging.Finish()
            self._logging = None


class SimWithLanguageTestEnv(HardwareStationScenarioSimulationEnv):
    """
    Test only env to wiggle dynamic language instructions during runtime.
    The env will set language instruction to
    self._default_language_instruction at the first tick, then '0' for the
    next 2 ticks, then '2' for the next 2 ticks, and '4' till the end of
    the rollout.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_and_record_pre_episode_snapshot(
        self, *, seed=None, options=None
    ):
        super().reset_and_record_pre_episode_snapshot(
            seed=seed, options=options
        )
        self._tick_counter = 0
        assert self._default_language_instruction is not None

    def step(self, action):
        self._tick_counter += 1
        return super().step(action)

    def _get_language_instruction(self):
        # dynamic language instructions during runtime.
        if self._tick_counter > 4:
            return "4"
        elif self._tick_counter > 2:
            return "2"
        elif self._tick_counter == 0:
            return self._default_language_instruction
        return "0"


@dc.dataclass
class SimWithLanguageTestEnvConfig(HardwareStationScenarioSimulationEnvConfig):
    def create(self):
        return SimWithLanguageTestEnv(**vars(self))
