from enum import Enum
import os
from pathlib import Path
import re
from typing import List, Sequence

from anzu.common.runfiles import Rlocation


def _valid_skill_name(name):
    # Ensure there are no unexpected characters;
    # only [a-z], [A-Z], and [0-9] are allowed.
    if re.search("[^a-zA-Z0-9]", name):
        return False, f"Invalid skill type '{name}'; invalid character found."
    # Each line must start with a capital letter to be a valid
    # CamelCase skill name.
    if not re.search("^[A-Z]", name):
        return False, (
            f"Invalid skill type '{name}'; skill name must be in CamelCase."
        )
    return True, None


def _load_file_contents(filepath):
    skill_types = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip()
            # Blank lines are okay.
            if not line:
                continue
            # Comments are okay.
            if line[0] == "#":
                continue
            valid, reason = _valid_skill_name(line)
            if not valid:
                raise ValueError(
                    f"{reason} File: {os.path.basename(filepath)}"
                )
            skill_types.append(line)
    return skill_types


def _load_skill_types_from_file(root_path=None):
    # Note that specifying a root_path is primarily intended as a test
    # affordance.
    if not root_path:
        root_path = Rlocation("anzu/intuitive/skill_types/.empty")
        root_path = os.path.split(root_path)[0]
    filepaths = Path(root_path).glob("*.txt")
    skill_types = []
    for filepath in filepaths:
        skill_types.extend(_load_file_contents(filepath))
    duplicates = [x for x in skill_types if skill_types.count(x) > 1]
    if duplicates:
        raise ValueError(f"Duplicates in skill type lists: {duplicates}")
    return sorted(skill_types)


def _camel_case_to_snake_case(camel_case: str) -> str:
    # Prefix any length of uppercase characters with an underscore.
    camel_case_with_underscores = re.sub("([A-Z]+)", r"_\1", camel_case)
    # Note that we need to remove a leading underscore.
    return camel_case_with_underscores[1:].lower()


def _camel_cases_to_snake_cases(values: Sequence[str]) -> List[str]:
    result = []
    for camel_case in values:
        if camel_case == "WriteTRIJarod":
            result.append("write_tri_jarod")
            continue
        if camel_case == "FoldTowel":
            # TODO(dale.mcconachie) Verify that the config files also has
            # the same matching typo and fix both at the same time.
            result.append("fold_towl")
            continue
        if camel_case == "PutMarkerInMugGS360":
            result.append("put_marker_in_mug_gs360")
            continue
        result.append(_camel_case_to_snake_case(camel_case))
    return result


def _create_skill_type_enum():
    camel_case_skill_types = _load_skill_types_from_file()
    snake_case_skill_types = _camel_cases_to_snake_cases(
        camel_case_skill_types
    )
    generated_class = Enum(
        "SkillType", zip(camel_case_skill_types, snake_case_skill_types)
    )
    return generated_class


SkillType = _create_skill_type_enum()


def force_create_skill_type_from_skill_name(skill_name: str) -> "SkillType":
    snake_case = _camel_case_to_snake_case(skill_name)
    generated_class = Enum("SkillType", [(skill_name, snake_case)])
    return generated_class


def get_skill_by_camel_case_name(name):
    if name not in SkillType.__members__:
        raise ValueError(
            f"{name} is not registered. Please check spelling, "
            "ensure you have the correct branch up-to-date, and/or add to "
            "anzu/intuitive/skill_types/*.txt."
        )
    return SkillType[name]


def sort_skills(skills: list[SkillType]) -> list[SkillType]:
    def key(skill: SkillType):
        return skill.name

    return sorted(skills, key=key)


_SNAKE_CASE_TO_SKILL = {skill.value: skill for skill in SkillType}


def get_skill_by_snake_case_name(name):
    skill = _SNAKE_CASE_TO_SKILL.get(name)
    if skill is None:
        raise ValueError(
            f"{name} is not registered. Please check spelling, "
            "ensure you have the correct branch up-to-date, and/or add the "
            "camel case version of it to anzu/intuitive/skill_types/*.txt."
        )
    return skill


def get_skill_by_name(name: str) -> SkillType:
    # TODO(hongkai.dai) replace get_skill_by_snake_case_name and
    # get_skill_by_camel_case_name with get_skill_by_name.
    if name in SkillType.__members__:
        return SkillType[name]
    if name in _SNAKE_CASE_TO_SKILL:
        return _SNAKE_CASE_TO_SKILL[name]
    raise ValueError(
        f"{name} is not registered. Please check spelling, "
        "ensure you have the correct branch up-to-date, and/or add the "
        "camel case version of it to anzu/intuitive/skill_types/*.txt."
    )
