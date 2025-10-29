from pydrake.common.yaml import yaml_load_typed

try:
    from anzu.common.cc import (
        GetStationNameFromString,
        IsDualArmPanda,
        IsRainbow,
        StationName,
    )
    from anzu.robot_bridge.cc import MultiFramePoseStreamParam
except ImportError:
    # When in open-source mode, we don't have access to the `.cc` modules.
    StationName = None
    from anzu.robot_bridge.multi_frame_pose_stream_param import (
        MultiFramePoseStreamParam,
    )

from anzu.common.runfiles import Rlocation


def _get_pose_stream_params_filename_from_station_name(
    *,
    station_name: str | StationName,
) -> str:
    if StationName is None:
        # The `.cc` imports didn't work, which means that we're in open source
        # mode with only two stations supported. We'll confirm that's true; if
        # this fails, then a `deps = ...` is probably missing.
        assert isinstance(station_name, str)
        assert station_name in ("cabot", "riverway")
        species = "dual_arm_panda"
    else:
        if not isinstance(station_name, StationName):
            station_name = GetStationNameFromString(station_name)
        if IsRainbow(station_name):
            species = "rainbow"
        elif IsDualArmPanda(station_name):
            species = "dual_arm_panda"
        else:
            raise ValueError("Unknown robot species")
    filename = Rlocation(
        "anzu/intuitive/visuomotor/demo/"
        f"visuomotor_multi_frame_pose_stream_{species}.yaml"
    )
    return filename


def load_pose_stream_params(
    *,
    filename: str | None = None,
    station_name: str | StationName | None = None,
) -> MultiFramePoseStreamParam:
    if not ((filename is None) ^ (station_name is None)):
        raise ValueError(
            "Exactly one of filename or station_name must be provided"
        )

    if filename is None:
        filename = _get_pose_stream_params_filename_from_station_name(
            station_name=station_name,
        )

    return yaml_load_typed(
        schema=MultiFramePoseStreamParam,
        filename=filename,
        child_name="multi_frame_pose_stream",
    )
