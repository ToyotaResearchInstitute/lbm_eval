import dataclasses
import inspect
import os
from pathlib import Path
from typing import Any

from pydrake.geometry import Meshcat
from pydrake.planning import RobotDiagramBuilder

from anzu.sim.common.deformable_logging import DeformableLogging
from anzu.sim.common.html_recording import HtmlRecording
from anzu.sim.common.keyframes_logging import KeyframesLogging


@dataclasses.dataclass(kw_only=True)
class LoggingConfig:
    # Parent directory for all log filenames that are not absolute paths. By
    # default (if empty) logs to the process's working directory. This can be a
    # relative path (relative to `$TEST_TMPDIR` in tests or the current working
    # directory otherwise) or an absolute path.
    log_dir: str = ""

    # Individual log files to write.
    logs: list[
        HtmlRecording | KeyframesLogging | DeformableLogging
    ] = dataclasses.field(default_factory=list)


class _Finisher:
    """An internal implementation detail of ApplyLoggingConfig that provides a
    composite Finish method that delegates to Finish on all of its children."""

    def __init__(self):
        self.children = []

    def Finish(self):
        for item in self.children:
            item.Finish()


def ApplyLoggingConfig(config: LoggingConfig | Any,
                       *,
                       robot_builder: RobotDiagramBuilder,
                       meshcat: Meshcat):
    """Adds the logging specified by `config` to the given `robot_builder`
    and/or `meshcat` objects.

    Every logger in `config.logs` must provide an `ApplyLogConfig` function
    defined the same module as the class of the logger's config dataclass. This
    function looks up that function by name and calls it to create the log.
    This allows the set of possible loggers to grow beyond what LoggingConfig
    knows about.

    Returns a handle for finishing and closing the log file(s). The caller must
    call `Finish()` on the returned object when the simulation is complete."""

    # Choose the path where logs should live.
    if config.log_dir:
        log_dir = Path(config.log_dir).absolute()
    else:
        log_dir = Path(os.environ.get("TEST_TMPDIR", "/tmp"))

    # Add all requested logs.
    result = _Finisher()
    for item in config.logs:
        module = inspect.getmodule(item)
        apply_function = getattr(module, "ApplyLogConfig")
        child_result = apply_function(
            config=item,
            log_dir=log_dir,
            robot_builder=robot_builder,
            meshcat=meshcat,
        )
        if child_result is not None:
            result.children.append(child_result)

    return result
