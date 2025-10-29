import dataclasses
import pathlib

from pydrake.planning import RobotDiagramBuilder

from anzu.sim.common.plant_pose_recorder import PlantPoseRecorder


# Configures the logging of keyframes data to a text file.
@dataclasses.dataclass(kw_only=True)
class KeyframesLogging:
    # Filename, relative to `log_dir` if relative.
    filename: str = "keyframes.txt"

    # Frequency, in fps, of keyframe recording.
    frequency_fps: float = 30.0


def ApplyLogConfig(*,
                   config: KeyframesLogging,
                   log_dir: pathlib.Path,
                   robot_builder: RobotDiagramBuilder,
                   **kwargs):
    """Enables keyframes logging on the given `robot_builder`."""
    txt_filename = log_dir / config.filename
    txt_filename.parent.mkdir(parents=True, exist_ok=True)
    plant = robot_builder.plant()
    builder = robot_builder.builder()
    pose_recorder = builder.AddSystem(
        PlantPoseRecorder(
            plant=plant,
            filename=txt_filename,
            period=(1.0 / config.frequency_fps),
        )
    )
    builder.Connect(plant.get_state_output_port(),
                    pose_recorder.get_input_port())
