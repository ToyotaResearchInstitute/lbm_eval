import dataclasses as dc
import typing
from typing import Dict, Optional

# Minor type annotations
ObsType = typing.NewType("ObsType", typing.Any)
ActType = typing.NewType("ActType", typing.Any)


@dc.dataclass
class PolicyMetadata:
    """
    This struct is supposed to capture time / obs invariant meta information
    about the policy itself. Any dense information like debugging information
    or language instructions should be returned as part of action.
    """

    # descriptive name for what this policy is. e.g. clip_vit_b_unet_dp
    name: str
    # camel case SkillType that matches anzu
    # (e.g. BimanualPlaceFruitFromBowlIntoBin).
    # Multi task skill has dedicated SkillType as well.
    # robot side will parse and throw if this field is set incorrectly.
    skill_type: str
    # efs or s3 file path to ckpt
    checkpoint_path: str

    # Should be true for language conditioned policies. Optional for backward
    # compability reasons.
    is_language_conditioned: Optional[bool] = None

    # These are supposed to capture the code state used at *inference* time,
    # not training time.
    git_repo: Optional[str] = None
    git_sha: Optional[str] = None

    # Raw yaml config used to construct the policy
    raw_policy_config: dict = dc.field(default_factory=dict)

    # Note: dict used to capture all relevant system / runtime env that we
    # might want. since it's hard to capture everything a priori, i am
    # proposing we use a dict. once we somewhat settle on a minimal set of
    # runtime info for reproduction, let's lift those information out of the
    # catchall (e.g. we decided to go with containerization, use the
    # appropriate id for images.)
    runtime_information: Dict[str, str] = dc.field(default_factory=dict)


class Policy:
    """
    Base interface for a policy.

    Warning:
        This is a Work in Progress. Please talk to Siyuan, Sammy, and Eric
        before depending heavily on this interface.
    """

    def get_policy_metadata(self):
        """
        Return a PolicyMetadata struct.
        """
        raise NotImplementedError()

    def reset(self, *, seed=None, options=None):
        """
        Resets the state of the policy.

        Arguments:
            seed: Generally the same as what is passed into `gym.Env.reset()`.
            options: Generally the same as what is passed into
                `gym.Env.reset()`.
        """
        raise NotImplementedError()

    def step(self, obs: ObsType) -> ActType:
        """
        Takes an observation and produces an action.

        Can include debug information in the action itself.
        """
        raise NotImplementedError()

    def close(self):
        """
        Cleans up resources when fully done with policy (after all episodes).
        """
        pass


class PolicyConfig:
    def create(self):
        raise NotImplementedError()
