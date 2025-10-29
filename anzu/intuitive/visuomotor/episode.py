import dataclasses as dc
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from robot_gym.policy import PolicyMetadata

from anzu.intuitive.visuomotor.bases import EnvMetadata, EpisodeType
from anzu.intuitive.visuomotor.common_spaces import TerminalInstant
from anzu.intuitive.visuomotor.trajectory import Trajectory


@dc.dataclass
class TimingInfo:
    setup_start_time: datetime
    setup_end_time: datetime

    episode_start_time: datetime
    episode_end_time: datetime

    teardown_start_time: datetime
    teardown_end_time: datetime

    @staticmethod
    def make_empty():
        return TimingInfo(
            setup_start_time=None,
            setup_end_time=None,
            episode_start_time=None,
            episode_end_time=None,
            teardown_start_time=None,
            teardown_end_time=None,
        )

    @staticmethod
    def make_dummy():
        now = datetime.now(timezone.utc)
        return TimingInfo(
            setup_start_time=now,
            setup_end_time=now,
            episode_start_time=now,
            episode_end_time=now,
            teardown_start_time=now,
            teardown_end_time=now,
        )

    def assert_valid(self):
        assert self.setup_start_time.tzinfo is not None
        assert self.setup_end_time.tzinfo is not None
        assert self.episode_start_time.tzinfo is not None
        assert self.episode_end_time.tzinfo is not None
        assert self.teardown_start_time.tzinfo is not None
        assert self.teardown_end_time.tzinfo is not None


@dc.dataclass
class TimedLanguageInstruction:
    """
    Used to capture a time bounded language instruction.
    """

    # These are in Env clock
    start_time: float
    end_time: float

    # Associated frame indices
    start_frame_index: int
    end_frame_index: int

    language_instruction: str


@dc.dataclass
class DemonstrationInfo:
    # (For training / evaluation) The seed value used from `anzu...rng.seed()`
    # for demonstration.
    seed: int

    # Scenario configuration used. This should be a raw dictionary.
    # TODO(eric.cousineau): Given how complex data structures are, it makes no
    # sense to require this to be raw. Require actual config structure at some
    # point, or rename this field.
    scenario_config: dict

    ###########################################################################
    # TODO(eric.cousineau): Make these required if/when backwards compatibility
    # doesn't invalidate old `*.pkl` files. They are currently duplicated in
    # other fields
    is_eval: bool = False
    # Mapping from camera name to semantic name.
    camera_id_to_semantic_name: Optional[Dict[str, str]] = None
    ###########################################################################

    # Summarizes dense language instructions using a list of
    # TimedLanguageInstruction.
    timed_language_instructions: Optional[
        List[TimedLanguageInstruction]
    ] = None

    # Teleop vs rollouts vs dagger vs initial conditions, etc
    episode_type: Optional[EpisodeType] = None

    # Initial condition overlay
    overlay_snapshot_path: Optional[str] = None

    # Demonstration index.
    index: Optional[int] = None

    # Wall clock timestamp for events like start and end of an episode, etc
    timing_info: Optional[TimingInfo] = None

    # Env metadata
    env_metadata: Optional[EnvMetadata] = None

    # Policy metadata
    policy_metadata: Optional[PolicyMetadata] = None

    # Operator related bookkeeping information
    operator_name: Optional[str] = None
    host_name: Optional[str] = None
    # Eval cluster specific
    lbm_eval_cluster_name: Optional[str] = None
    lbm_eval_cluster_email: Optional[str] = None

    # Runtime information
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    lbm_release: Optional[str] = None

    # For offline randomization, we store the full path of the original source
    # episode that was used to produce this randomized episode.
    source_episode: Optional[str] = None

    # The types of data augmentations performed for this episode.
    augmentation_types: Optional[list[str]] = None

    # Used to store custom labels that describe the episode. The currently
    # expected values are lists with keys "eval" and "teleop".
    tags: Optional[dict[str, list[str]]] = None

    # Used to store controller configuration used to collect data for the
    # episode.
    controller_config: Optional[dict] = None

    def __post_init__(self):
        if not isinstance(self.scenario_config, dict):
            raise RuntimeError(
                f"scenario_config must actually be a raw dict; got "
                f"{type(self.scenario_config)} instead"
            )

    def __repr__(self):
        # Make an abbreviated representation.
        attributes = ""
        for key in self.__dict__:
            if key != "scenario_config":
                attributes += f"{key}={repr(getattr(self, key))}, "
            else:
                attributes += "scenario_config=<dict>, "

        return f"{self.__class__.__name__}({attributes})"


@dc.dataclass
class EpisodeStatus:
    is_successful: bool
    message: Optional[str] = None

    @classmethod
    def Success(cls, message=None):
        return cls(is_successful=True, message=message)

    @classmethod
    def Failure(cls, message):
        return cls(is_successful=False, message=message)

    @classmethod
    def Nothing(cls):
        return cls(is_successful=False, message=None)

    def __repr__(self):
        inner = ""
        if self.message is not None:
            inner = repr(self.message)
        if self.is_successful:
            return f"EpisodeStatus.Success({inner})"
        elif self.message is not None:
            return f"EpsiodeStatus.Failure({inner})"
        else:
            return f"EpisodeStatus.Nothing({inner})"

    @property
    def is_timeout(self):
        return not self.is_successful and self.message.startswith(
            "Failure: Not enough time"
        )


@dc.dataclass
class Episode:
    """Similar to ImitationEpisode from key_dynam repository."""

    trajectory: Trajectory  # of PolicyInstant

    terminal_instant: Optional[TerminalInstant]

    status: EpisodeStatus

    demonstration_info: DemonstrationInfo

    pre_experiment_data: Any = None
    post_experiment_data: Any = None
