import copy
import dataclasses as dc
import logging
import os
from pathlib import Path
from typing import Any, Callable

from pydrake.common.yaml import yaml_dump, yaml_load_typed
from pydrake.visualization import VisualizationConfig

# TODO(dale.mcconachie) We are using this to resolve non-model directive
# files. Thus this helper should probably move outside of
# `anzu_model_directives`.
from anzu.common.anzu_model_directives import MakeDefaultAnzuPackageMap
from anzu.sim.common.take_scenario_sample import (
    CommonScenarioConfig,
    do_sample,
    ingest_yaml,
)

# TODO(dale.mcconachie) Throughout this file we are repeatedly ingesting the
# YAML as files/strings, and then re-emitting it back to a file/string. We
# should stop doing this.


def _resolve_scenario_filename(
    common_scenario_config: CommonScenarioConfig,
) -> CommonScenarioConfig:
    common_scenario_config = copy.deepcopy(common_scenario_config)
    if common_scenario_config.scenario_file is not None:
        if common_scenario_config.scenario_file.startswith("package:"):
            scenario_file_path = MakeDefaultAnzuPackageMap().ResolveUrl(
                common_scenario_config.scenario_file
            )
            common_scenario_config.scenario_file = scenario_file_path
    return common_scenario_config


def _get_or_default_visualization_period(scenario_yaml: dict) -> float:
    """Extract the visualization publish period from the scenario YAML, or
    return the default.
    """
    try:
        return scenario_yaml["visualization"]["publish_period"]
    except KeyError:
        return VisualizationConfig().publish_period


def _add_logging_config(
    *,
    common_scenario_config: CommonScenarioConfig,
    logging_dir: str,
    keyframe_logging_fps: float,
) -> CommonScenarioConfig:
    """Given the logging directory, replace the logging stanza with a
    KeyframesLogging, HtmlRecording and DeformableLogging configuration.
    """
    # TODO(dale.mcconachie) Convert this function to use
    # hardware_station_simulation_scenario.Scenario as the input and/or output
    # format instead? This would require resolving all randomness from the
    # source YAML before we do this munging as we have randomness in our YAMLs
    # that Scenario cannot express. It would also introduce an explicit
    # dependency on a C++ bound library. `Scenario` has that dependency
    # transitively, but this option would include it directly.

    scenario_yaml = ingest_yaml(**vars(common_scenario_config))

    # NOTE: At the time this function was written; the `logging` stanza of all
    # YAMLs that we intend to use with this function were empty. We assert that
    # this is still the case so that if this assumption ever changes, we don't
    # silently get unexpected behavior that is hard to diagnose.
    if "logging" in scenario_yaml:
        if scenario_yaml["logging"] != {}:
            raise RuntimeError(
                "The `logging` stanza of the scenario YAML must be empty, "
                "but it is not. This function assumes that the logging stanza "
                "is empty and will replace it with a new configuration. If "
                "you have a use case where this is not true; revisit this "
                "function and make sure that it works as intended for all "
                "use cases."
            )

    # TODO(dale.mcconachie): Use `log_dir` instead of embedding the full
    # directory directory in `filename`s below.
    logging_dir = Path(logging_dir)
    visualization_period = _get_or_default_visualization_period(scenario_yaml)

    # Add a logging stanza with our new logging configuration.
    scenario_yaml["logging"] = {
        "logs": [
            {
                "_tag": "!KeyframesLogging",
                "filename": os.path.join(logging_dir, "keyframes.txt"),
                "frequency_fps": keyframe_logging_fps,
            },
            {
                "_tag": "!DeformableLogging",
                "filename": os.path.join(logging_dir, "deformable.h5"),
                # Since we will render each keyframe with the deformable object
                # state, we will use the same fps as keyframe logging.
                "frequency_fps": keyframe_logging_fps,
            },
            {
                "_tag": "!HtmlRecording",
                "filename": os.path.join(logging_dir, "recording.html"),
                "frequency_fps": 1.0 / visualization_period,
            },
        ],
    }

    # We have ingested all information in the source common_scenario_config,
    # but that data structure is the language we're going to use to pass data
    # around, so dump back to that format.
    return CommonScenarioConfig(
        scenario_file=None,
        scenario_name=None,
        scenario_text=yaml_dump(scenario_yaml),
        random_seed=common_scenario_config.random_seed,
        num_sample_processes=common_scenario_config.num_sample_processes,
    )


@dc.dataclass(kw_only=True)
class CameraOverrides:
    """Overrides for the `cameras` portion of a hardware station simulation
    scenario.
    """

    remove_cameras: bool = False
    """Remove cameras from the scenario before creating it, but keep the
    cameras in the resolved scenario. Primarily used for teleop with space mice
    or the Oculus.
    """

    replace_cameras_with_ros: list[str] = dc.field(default_factory=list)
    """Replace the given cameras with ROS rgb-only cameras. Any other cameras
    will be removed from the scenario. This also removes any fisheye distortion
    and disables images as part of an observation. I.e. observation.visuo will
    be empty. Primarily used for teleop with the `robot_camera_server` and the
    Apple Vision Pro.
    """

    def validate(self):
        if self.remove_cameras and self.replace_cameras_with_ros:
            raise ValueError(
                "Mutually exclusive options; we cannot remove cameras and "
                "replace them with ROS cameras at the same time."
            )

    def __post_init__(self):
        self.validate()


def _apply_camera_overrides(
    scenario: Any, camera_overrides: CameraOverrides
) -> Any:
    scenario = copy.deepcopy(scenario)

    if camera_overrides.remove_cameras:
        logging.info("Removing cameras to improve simulation performance.")
        scenario.cameras = []

    if camera_overrides.replace_cameras_with_ros:
        logging.info("Replacing the cameras with ROS cameras.")

        # We must delay this import to avoid a dependency on native code in
        # cases where it's unavailable (e.g., open source).
        from anzu.sim.camera.cc import CameraConfigRos

        new_cameras = []
        for camera in scenario.cameras:
            if camera.name in camera_overrides.replace_cameras_with_ros:
                # Extract the drake portion; and remove the LCM, depth, and
                # label fields.
                drake_camera = camera.drake
                drake_camera.lcm_bus = ""
                drake_camera.depth = False
                drake_camera.label = False
                # Convert the rest to a ROS camera config.
                ros_camera = CameraConfigRos(
                    drake=drake_camera, rgb_topic="/" + drake_camera.name
                )
                new_cameras.append(ros_camera)

        scenario.cameras = new_cameras
    return scenario


@dc.dataclass(kw_only=True, frozen=True)
class ScenarioResolutionIntermediaries:
    random_scenario: dict[str, Any]
    """The scenario YAML that was sampled from, before any overrides."""
    resolved_scenario: dict[str, Any]
    """The resolved scenario YAML before any camera overrides."""
    simulation_scenario: Any
    """The final scenario object as a full first-class type with all overrides
    applied and randomness sampled."""


def apply_scenario_overrides_and_sample(
    common_scenario_config: CommonScenarioConfig,
    scenario_schema: type,
    make_hardware_station: Callable,
    logging_dir: str | Path | None,
    keyframe_logging_fps: float,
    camera_overrides: CameraOverrides | None = None,
) -> ScenarioResolutionIntermediaries:
    """Apply overrides to a common scenario configuration and samples the
    resulting scenario. Returns intermediary YAMLs during the override process
    and the final Scenario object.
    """
    # Apply the overrides/transformations that we can while still in YAML
    # format.
    common_scenario_config = _resolve_scenario_filename(common_scenario_config)

    if logging_dir is not None:
        # Ensure that we get an interpretable error rather than a downstream
        # YAML parsing error if the scenario schema does not have a
        # `logging` field but logging is requested.
        field_names = [field.name for field in dc.fields(scenario_schema)]
        if "logging" not in field_names:
            raise ValueError(
                "The scenario schema does not have a `logging` field, so "
                "logging cannot be configured. If you want to use logging, "
                "add a `logging` field to the scenario schema."
            )

        common_scenario_config = _add_logging_config(
            common_scenario_config=common_scenario_config,
            logging_dir=logging_dir,
            keyframe_logging_fps=keyframe_logging_fps,
        )

    random_scenario_yaml = ingest_yaml(**vars(common_scenario_config))
    # Convert to schema format and apply the rest of the overrides.
    resolved_scenario_yaml = do_sample(
        common_scenario_config,
        schema=scenario_schema,
        make_hardware_station=make_hardware_station,
    )
    simulation_scenario = yaml_load_typed(
        schema=scenario_schema, data=yaml_dump(resolved_scenario_yaml)
    )
    if camera_overrides is not None:
        simulation_scenario = _apply_camera_overrides(
            simulation_scenario, camera_overrides
        )
    # Return the intermediary YAMLs and the final scenario.
    return ScenarioResolutionIntermediaries(
        random_scenario=random_scenario_yaml,
        resolved_scenario=resolved_scenario_yaml,
        simulation_scenario=simulation_scenario,
    )
