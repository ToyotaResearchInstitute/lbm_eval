import dataclasses
import pathlib

from pydrake.geometry import Meshcat

# DEBUGGING HINT:  If you set an HtmlRecording, but no recording is saved, it
# is likely that you have failed to call HardwareStationSimulation.cleanup().


@dataclasses.dataclass(kw_only=True)
class HtmlRecording:
    # Filename, relative to the `log_dir` if relative.
    filename: str = "recording.html"

    # Frequency, in fps, of meshcat recording.
    frequency_fps: float = 64.0


class _Finisher:
    """An internal implementation detail of ApplyLogConfig that provides a
    callback to finish writing the recording."""

    def __init__(self, *, meshcat, html_filename):
        self._meshcat = meshcat
        self._html_filename = html_filename

    def Finish(self):
        self._meshcat.StopRecording()
        self._meshcat.PublishRecording()
        text = self._meshcat.StaticHtml()
        self._html_filename.write_text(text, encoding="utf-8")


def ApplyLogConfig(*,
                   config: HtmlRecording,
                   log_dir: pathlib.Path,
                   meshcat: Meshcat,
                   **kwargs):
    """Enables meshcat recording on the given `meshcat`."""
    meshcat.StartRecording(config.frequency_fps)
    html_filename = log_dir / config.filename
    html_filename.parent.mkdir(parents=True, exist_ok=True)
    return _Finisher(meshcat=meshcat, html_filename=html_filename)
