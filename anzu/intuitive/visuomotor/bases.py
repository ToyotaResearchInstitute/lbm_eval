"""
Base types for Gym interface.

For example pseudo code of what stepping on hardware (and other domains) may
look like, please see the documentation for `GymEnvWrappingAnzuEnv`.
"""

import contextlib
import dataclasses as dc
from enum import Enum
import logging
import typing
from typing import Dict, Optional

import gymnasium as gym
from robot_gym.policy import Policy

from anzu.intuitive.skill_defines import SkillType

# Minor type annotations
ObsType = typing.NewType("ObsType", typing.Any)
ActType = typing.NewType("ActType", typing.Any)


class EnvDomain(Enum):
    """
    Used to signal trajectory is collected in sim or real world.
    """

    Sim = "sim"
    Real = "real"
    Umi = "umi"


class EpisodeType(Enum):
    """
    Used to signal use cases for logged trajectory. Currently should match
    the lbm visuomotor folder structure:
    .../data/tasks/{skill}/{station}/{domain}/bc/{episode_type}/...
    """

    Teleop = "teleop"
    Rollout = "rollout"
    # Dummy episode that captures initial condition for evaluation
    EvalInitialCondition = "eval_initial_condition"
    # Dummy episode that captures initial condition for teleop
    TeleopInitialCondition = "teleop_initial_condition"
    # Intended for dagger like data: mixture sources for actions
    ExpertIntervention = "expert_intervention"


@dc.dataclass
class DistributionShiftInfo:
    """
    Map of distribution shift options to their setting. Boolean values are
    expressed as "enabled" or "disabled". Distribution shifts that can take on
    a range of values are expressed as some value taken from {"disabled",
    "low", "medium", "high"}.
    """

    scene_extrinsics: str = "disabled"
    scene_intrinsics: str = "disabled"
    wrist_extrinsics: str = "disabled"
    wrist_intrinsics: str = "disabled"
    environment_map: str = "disabled"
    lighting: str = "disabled"
    distractor_textures: str = "disabled"
    manipuland_textures: str = "disabled"
    table_top_texture: str = "disabled"
    distractor_models: str = "disabled"
    manipuland_pose: str = "disabled"

    def __post_init__(self):
        ALLOWED_VALUES = ["disabled", "enabled", "low", "medium", "high"]
        for field in dc.fields(DistributionShiftInfo):
            assert getattr(self, field.name) in ALLOWED_VALUES


@dc.dataclass
class EnvMetadata:
    """
    Intended to capture env related meta data information that are independent
    of any rollout related information like index or seed.
    """

    domain: EnvDomain
    skill: SkillType

    station_name: str
    hardware_platform_type: str
    camera_id_to_semantic_name: Optional[Dict[str, str]] = None

    distribution_shift_info: Optional[DistributionShiftInfo] = None

    def __post_init__(self):
        if self.skill is None:
            logging.warn(
                "WARNING: skill in EnvMetadata is set to None, "
                "defaulting to Undefined."
            )
            self.skill = SkillType.Undefined

        assert isinstance(self.skill, SkillType), self.skill
        assert isinstance(self.domain, EnvDomain), self.domain


# Same as `TimeStep` in `dm_env` / `dm_control`.
TimeStep = typing.NamedTuple(
    "TimeStep",
    [
        # Observation.
        ("obs", ObsType),
        # Reward at given instant.
        ("reward", float | None),
        # MDP terminated (successfully or unsucessfully).
        ("terminated", bool),
        # MDP was truncated (did not terminate).
        ("truncated", bool),
        # Metadata information.
        ("info", dict | None),
    ],
)


def is_terminal_time_step(time_step):
    done = time_step.terminated or time_step.truncated
    return done


class AnzuEnv:
    """
    A *distinct* base class from the OpenAI / gymnasium Gym environment.

    For actual stepping usage, please see:
        - GymEnvWrappingAnzuEnv
        - collect_episode_gym()
    """

    def __init__(self):
        # TODO(eric.cousineau): Have this leverage observation / action spaces.
        pass

    def get_env_metadata(self) -> EnvMetadata:
        raise NotImplementedError()

    def step(self, action) -> TimeStep:
        """
        Args:
            act: Action
        Returns:
            TimeStep

        For more details, see:
        https://gym.openai.com/docs/#observations
        """
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def close(self):
        """Free resources, stop robots, etc."""
        raise NotImplementedError()

    def reset_and_record_pre_episode_snapshot(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """
        Resets the environment. May re-randomize the initial state.

        This can take as long as needed.

        Note: As opposed to nominal gym API, this *does not* return anything.
        """
        raise NotImplementedError()

    def start_episode(self) -> TimeStep:
        """
        Called immediately after `recorder.start_recording()` has been called
        to denote the start of an episode.

        Internal state such as wall-clock start time should be updated.
        Anything that is received from middleware should be confirmed to be new
        so that recording will have seen those messages.

        This should take the minimal amount of time possible.

        Returns:
            TimeStep
        """
        raise NotImplementedError()

    def stop_episode(self):
        """
        Called right before `recorder.stop_recording()`. Meant to record any
        information regarding the ending episode.

        This should take the minimal amount of time possible.
        """
        raise NotImplementedError()

    def finish_and_record_post_episode_snapshot(self) -> dict:
        """
        Called after `recorder.stop_recording()`. Meant to help teardown, etc.

        This can take as long as needed.

        Returns:
            final_info
        """
        raise NotImplementedError()

    def abort_episode(self):
        """
        Aborts an episode. This may be called if an exception occurs at any
        point in time such that the terminal state is not actually marked as
        terminal.
        """
        raise NotImplementedError()


class Recorder:
    """
    Records start, stop, and each step of an episode for an AnzuEnv.

    For more details, see GymEnvWrappingAnzuEnv.
    """

    def start_recording(self, *, seed=None, options=None):
        """
        This may have non-trivial start-up time for hardware; we want to
        start recordering before calling `env.start_episode()`.

        `seed` and `options` are generally the same as what is passed to
        `env.reset(options)`.

        If using middleware recording ,(e.g., LCM, ROS), the related
        `env.start_episode()` should wait for fresh messages to ensure
        we are as precise as possible.
        """
        raise NotImplementedError()

    def record_initial(self, time_step):
        """
        Records initial TimeStep immediately after `env.reset()` (or
        `anzu_env.start_episode()`).
        """
        raise NotImplementedError()

    def record_step(
        self,
        time_step_prev: TimeStep,
        act: ActType,
        time_step: TimeStep,
    ):
        """
        Records a step.

        Arguments:
            time_step_prev: Prior time step, where `time_step_prev.obs` was
                used to produce `act`.
            act: Action fed into the environment.
            time_step: TimeStep after taking `act`.
        """
        raise NotImplementedError()

    def stop_recording(self):
        """
        Called before `env.finish_episode()`. Meant to stop any high-throughput
        logging.
        """
        raise NotImplementedError()

    def save_recording(self, final_info):
        """
        Called *after*
            final_info = env.finish_episode()
        Meant to then save data to the disk, especially if there is any
        post-experiment data that was collected via the environment, user
        input, etc.
        """
        raise NotImplementedError()

    def abort_recording(self):
        """
        Called if an exception occurs at any point in time such that the
        terminal state is not actually marked as terminal.
        """
        raise NotImplementedError()

    def close(self):
        """
        Free up any resources related to this Recorder.
        """
        raise NotImplementedError()


class NoopRecorder(Recorder):
    """A recorder that is complete no-op."""

    def start_recording(self, *, seed=None, options=None):
        pass

    def record_initial(self, time_step):
        pass

    def record_step(
        self,
        time_step_prev,
        act,
        time_step,
    ):
        pass

    def stop_recording(self):
        pass

    def save_recording(self, final_info):
        pass

    def abort_recording(self):
        pass

    def close(self):
        pass


class GymEnvWrappingAnzuEnv(gym.Env):
    """
    Wrapper of AnzuEnv and Recorder class to provide more vanilla gym-like
    semantics.

    A episode will start upon `.reset()`. The episode will be ended either upon
    next call to `.reset()`, `.stop_episode()`, or upon `.close()` of this
    instance.

    When called with `collect_episode_gym()`, we have the following *rough*
    sequence of events:

        # Setup.
        policy.reset(seed, options)
        anzu_env.reset_and_record_pre_episode_snapshot(seed, options)
        recorder.start_recording(seed, options)
        time_step = anzu_env.start_episode()
        recorder.record_initial(time_step)

        # Running loop.
        while not is_terminal_time_step(time_step):
            time_step_prev = time_step
            act = policy.step(time_step_prev)
            time_step = anzu_env.step(act)
            recorder.record_step(
                time_step_prev=time_step_prev,
                act=act,
                time_step=time_step,
            )
            anzu_env.render()

        if <no fatal errors>:
            # Nominal completion of episode.
            anzu_env.stop_episode()
            recorder.stop_recording()
            final_info = anzu_env.finish_and_record_post_episode_snapshot()
            recorder.save_recording(final_info)
        else:
            # Fatal error, either from exception or premature ending of
            # stepping loop.
            anzu_env.abort_episode()
            recorder.abort_recording()
    """

    def __init__(
        self,
        anzu_env: AnzuEnv,
        recorder: Recorder | None = None,
        *,
        owning=True,
    ):
        """
        Arugments:
            anzu_env: AnzuEnv instance
            recorder: Recorder instance
            owning: If True, will close anzu_env and recorder via `.close()`.
        """
        super().__init__()
        self.anzu_env = anzu_env
        if recorder is None:
            recorder = NoopRecorder()
        self.recorder = recorder
        self._owning = owning

        self._time_step_prev = None
        self._closed = False

    @property
    def _started(self):
        return self._time_step_prev is not None

    def _start(self, *, seed=None, options=None):
        assert not self._started
        self.anzu_env.reset_and_record_pre_episode_snapshot(
            seed=seed, options=options
        )
        self.recorder.start_recording(seed=seed, options=options)
        time_step = self.anzu_env.start_episode()
        self.recorder.record_initial(time_step)
        self._time_step_prev = time_step
        return time_step

    def _maybe_stop_or_abort(self):
        if not self._started:
            return
        was_proper_stop = is_terminal_time_step(self._time_step_prev)
        if was_proper_stop:
            # Note: This does *not* call `anzu_env.close()` or
            # `recorder.close()`.
            self.anzu_env.stop_episode()
            self.recorder.stop_recording()
            final_info = (
                self.anzu_env.finish_and_record_post_episode_snapshot()
            )
            self.recorder.save_recording(final_info)
        else:
            # We did not signal a "true" stop, assume we should abort.
            self.anzu_env.abort_episode()
            self.recorder.abort_recording()
        self._time_step_prev = None

    def get_env_metadata(self):
        return self.anzu_env.get_env_metadata()

    def reset(self, *, seed=None, options=None):
        assert not self._closed
        self._maybe_stop_or_abort()
        time_step = self._start(seed=seed, options=options)
        return time_step

    def step(self, act):
        assert self._started and not self._closed, (
            self._started,
            self._closed,
        )
        time_step = self.anzu_env.step(act)
        self.recorder.record_step(
            time_step_prev=self._time_step_prev,
            act=act,
            time_step=time_step,
        )
        self._time_step_prev = time_step
        return time_step

    def render(self):
        assert self._started and not self._closed, (
            self._started,
            self._closed,
        )
        return self.anzu_env.render()

    def stop_episode(self):
        """
        Non-gym API to signal that we have finished an episode, but may still
        reset once more.

        This is provided so that we do not have to strictly close this
        environment to get the last episode.
        """
        assert not self._closed
        self._maybe_stop_or_abort()

    def close(self):
        if self._closed:
            return
        # Finish episode, if one was started.
        self._maybe_stop_or_abort()
        # Close resources if they are owned.
        if self._owning:
            self.anzu_env.close()
            self.recorder.close()
        self._closed = True


def collect_episode_gym(
    env: gym.Env,
    policy: Policy,
    *,
    seed: int | None = None,
    options: dict | None = None,
):
    """
    Collects a single episode with an *gym* env and one our Policy
    implementations.

    It is the caller's responsibility to call `env.close()` and
    `policy.close()`. This is very important to handle error behavior and
    ensure resources are freed / robots are stopped, etc, when errors occur.

    Note: This is generally meant to be used with `GymEnvWrappingAnzuEnv`,
    which does the relevant recording.
    """

    policy_metadata = policy.get_policy_metadata()
    if options is None:
        options = {}
    options["policy_metadata"] = policy_metadata
    if hasattr(env, "get_env_metadata"):
        env_metadata = env.get_env_metadata()
        options["env_metadata"] = env_metadata

    policy.reset(seed=seed, options=options)
    time_step = env.reset(seed=seed, options=options)

    try:
        while not is_terminal_time_step(time_step):
            act = policy.step(time_step.obs)
            time_step = env.step(act)
            env.render()
    finally:
        if isinstance(env, GymEnvWrappingAnzuEnv):
            # TODO(eric.cousineau): How to make more generic?
            env.stop_episode()


def closing_multiple(*things):
    """
    Plural version of `contextlib.closing(thing)`.

    If any exceptions occur, it will be stored, and all `things` will still be
    attempted to be closed. The latest exception will be rethrown.
    """
    exit_stack = contextlib.ExitStack()
    for thing in things:
        if thing is not None:
            closing_context = contextlib.closing(thing)
            exit_stack.enter_context(closing_context)
    return exit_stack


class ManualRecoveryNeeded(RuntimeError):
    """
    Indicates an error that occurs during `env.reset*()` and `env.step()`, and
    may require manual intervention before we can resume.

    Examples are losing connections from the env (e.g., cameras) or the
    policy (e.g., teleop via a headset device). The user can then reestablish a
    connection, and then continue collecting data once they indicate that
    recovery is done.
    """

    pass
