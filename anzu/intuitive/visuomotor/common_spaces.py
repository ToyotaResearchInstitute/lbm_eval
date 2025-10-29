from collections import namedtuple
import dataclasses as dc
from typing import Any, Optional

from anzu.intuitive.skill_defines import SkillType
from anzu.intuitive.visuomotor.bases import is_terminal_time_step

StepInfo = namedtuple("StepInfo", ("observation", "reward", "done", "info"))


def time_step_to_step_info(time_step):
    return StepInfo(
        observation=time_step.obs,
        reward=time_step.reward,
        done=is_terminal_time_step(time_step),
        info=time_step.info,
    )


@dc.dataclass
class PolicyInstantDebug:
    """Intended for information that is not strictly part of the visuomotor
    pipeline."""

    policy_step_info: Optional[Any] = None
    # TODO(dale.mcconachie) Consider adding a policy_state entry here for any
    # policies that have internal state.


@dc.dataclass
class PolicyInstant:
    t: float
    env_state: Any
    observation: Any
    feature: Any
    action: Any
    env_input: Any
    step_info: StepInfo
    transforms: Optional[Any] = None
    debug: Optional[PolicyInstantDebug] = None
    skill_type: Optional[SkillType] = None

    def __post_init__(self):
        if self.skill_type is not None:
            assert isinstance(self.skill_type, SkillType)


@dc.dataclass
class TerminalInstant:
    t: float
    env_state: Any
    observation: Any
    step_info: StepInfo
    skill_type: Optional[SkillType] = None

    def __post_init__(self):
        if self.skill_type is not None:
            assert isinstance(self.skill_type, SkillType)
