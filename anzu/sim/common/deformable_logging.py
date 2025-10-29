import dataclasses
import pathlib

from pydrake.planning import RobotDiagramBuilder

from anzu.sim.common.deformable_configuration_recorder import (
    DeformableConfigurationRecorder,
)


# Configures the logging of deformable data to an h5 file.
@dataclasses.dataclass(kw_only=True)
class DeformableLogging:
    # Filename, relative to `log_dir` if relative.
    filename: str = "deformable.h5"

    # Frequency, in fps, of deformable recording.
    frequency_fps: float = 30.0


def ApplyLogConfig(
    *,
    config: DeformableLogging,
    log_dir: pathlib.Path,
    robot_builder: RobotDiagramBuilder,
    **kwargs,
):
    """Enables deformable state logging on the given `robot_builder`."""
    h5_filename = log_dir / config.filename
    h5_filename.parent.mkdir(parents=True, exist_ok=True)
    plant = robot_builder.plant()
    builder = robot_builder.builder()
    deformable_logger = builder.AddSystem(
        DeformableConfigurationRecorder(
            plant=plant,
            filename=h5_filename,
            period=1.0 / config.frequency_fps,
        )
    )
    builder.Connect(
        plant.get_deformable_body_configuration_output_port(),
        deformable_logger.deformable_body_configuration_input_port,
    )
