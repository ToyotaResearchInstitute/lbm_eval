"""Instrumentation helpers for in-process control of profile sampling by Linux
`perf`. See README.md for an overview.
"""

from contextlib import contextmanager
from enum import Enum
import os

_enable_cmd = "enable".encode("utf-8")
_disable_cmd = "disable".encode("utf-8")
_ack_cmd = "ack\n\0".encode("utf-8")


def _fd_from_env(env_key: str) -> int:
    name = os.environ.get(env_key)
    if not name:
        return -1
    return os.open(name, os.O_RDWR)


class SamplingState(Enum):
    UNKNOWN = 0
    OFF = 1
    ON = 2


_perf_controller_singleton = None


class PerfController:
    """In-process controller class.

    This class is designed to work together with the top-level launch script
    perf_controlled_record.py. It looks for the names of two FIFO files via the
    environment variables ANZU_PERF_CTL_FIFO and ANZU_PERF_ACK_FIFO. If those
    are missing, or the files can't be opened, then control will be
    unavailable.

    There is also a corresponding C++ implementation of this controller. They
    should not be used together simultaneously. The communication protocol is
    not designed for multiple controllers, and stalls or crashes may result.

    Most users should used the singleton interface (PerfController.Singleton())
    rather than creating an instance of this class directly.
    """
    def __init__(self):
        self._ctl_fd = _fd_from_env("ANZU_PERF_CTL_FIFO")
        self._ack_fd = _fd_from_env("ANZU_PERF_ACK_FIFO")
        self._is_control_available = (self._ctl_fd >= 0 and self._ack_fd >= 0)
        self._sampling_state = SamplingState.UNKNOWN

    def pause(self):
        """Turn sampling off."""
        if not self.is_control_available():
            return
        self._send_command(_disable_cmd)
        self._sampling_state = SamplingState.OFF

    def resume(self):
        """Turn sampling on."""
        if not self.is_control_available():
            return
        self._send_command(_enable_cmd)
        self._sampling_state = SamplingState.ON

    def is_control_available(self):
        """Returns True iff sampling control is available."""
        return self._is_control_available

    def sampling_state(self):
        return self._sampling_state

    def _send_command(self, command):
        assert self.is_control_available()
        wrote_bytes = os.write(self._ctl_fd, command)
        if wrote_bytes != len(command):
            raise RuntimeError("`perf` command not completely written.")
        got = os.read(self._ack_fd, len(_ack_cmd))
        if got != _ack_cmd:
            raise RuntimeError(
                f"`perf` command acknowledgment not received:"
                f" {got} != {_ack_cmd}")

    def disconnect(self):
        """Close open channels, if any, and make control unavailable."""
        self._is_control_available = False
        if self._ctl_fd >= 0:
            os.close(self._ctl_fd)
        if self._ack_fd >= 0:
            os.close(self._ack_fd)

    @staticmethod
    def Singleton():
        """Returns the singleton PerfController."""
        global _perf_controller_singleton
        if not _perf_controller_singleton:
            _perf_controller_singleton = PerfController()
        return _perf_controller_singleton


@contextmanager
def scoped_perf_sampling(controller=PerfController.Singleton()):
    """Context manager to turn sampling on for the duration
    of a section of code.
    """
    controller.resume()
    try:
        yield
    finally:
        controller.pause()
