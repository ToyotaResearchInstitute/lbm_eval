import socket

from anzu.mobile.common.rby1_station_defines import VALID_RBY1_STATION_NAMES

TELEOP_OPERATORS = [
    # for rollout
    "robot",
    # for test
    "test",
]


WEST_COAST_STATION_NAMES = {
    "wollaston",
    "davis",
    "ruggles",
    "worcester",
}


PLATFORM_TYPE_LBM_DUAL_PANDA = "LbmDualPanda"
PLATFORM_TYPE_LBM_RAINBOW_RBY1 = "LbmRainbowRby1"


def resolve_operator_name(operator_name: str):
    """
    Tries to extend operator_name to operator_name in TELEOP_OPERATORS
    the expected types of operator_name are
    such as "Masayuki M", "masayuki masuda"
    """
    if type(operator_name) is str:
        for _operator_name in TELEOP_OPERATORS:
            if operator_name.lower() in _operator_name.lower():
                return _operator_name

    raise RuntimeError(
        f"operator_name doesn't match TELEOP_OPERATORS. "
        f"operator_name: {operator_name}, "
        f"TELEOP_OPERATORS: {TELEOP_OPERATORS}"
    )


# Placeholder station name for legacy data / or data without any station
# dependent information.
ARBITRARY_STATION_NAME = "ArbitraryStation"


# N.B. Please keep these ordering in sync with anzu::common::StationName in
# make_robot_configuration.h.
# TODO(kuni) Consider deriving this list from that define. The current list
# includes only the stations compatible to diffusion policy learning.
VALID_PANDA_STATION_NAMES = [
    "cabot",
    "davis",
    "hersey",
    "maverick",
    "milton",
    "riverway",
    "ruggles",
    "salem",
    "stony_brook",
    "testbed",  # N.B. Not part of StationName.
    "wollaston",
    "wood_island",
    "worcester",
]

VALID_STATION_NAMES = VALID_PANDA_STATION_NAMES + VALID_RBY1_STATION_NAMES


def assert_valid_station_name(station_name):
    if station_name not in VALID_STATION_NAMES:
        raise RuntimeError(
            f"Invalid station: {repr(station_name)}\n"
            f"Available choices: {repr(VALID_STATION_NAMES)}"
        )


def infer_host_station_name():
    """
    Tries to infer station name from host name.

    This does not perform a validity check.
    """
    station_name = socket.gethostname()
    assert station_name, "Failed to resolve station name from hostname."
    station_name = station_name.replace("-", "_")
    return station_name


def resolve_station_name(station_name):
    """
    Resolves a station name, and ensures that it is a valid name.

    If station_name is None, will use infer
    """
    if station_name is None:
        station_name = infer_host_station_name()
        print(f"Resolved station_name={repr(station_name)}")
    assert_valid_station_name(station_name)
    return station_name


def resolve_hardware_platform(station_name_in):
    station_name = resolve_station_name(station_name_in)
    if station_name in VALID_PANDA_STATION_NAMES:
        return PLATFORM_TYPE_LBM_DUAL_PANDA
    assert station_name in VALID_RBY1_STATION_NAMES
    return PLATFORM_TYPE_LBM_RAINBOW_RBY1
