import abc

from pydrake.systems.framework import Context, EventStatus


class HardwareStationMonitor(abc.ABC):
    """Base class for all station simuation components that interact with
    Drake's Simulator.set_monitor feature."""

    @abc.abstractmethod
    def Monitor(self, root_context: Context) -> EventStatus:
        """Callback for Simulator.set_monitor that breaks out of the simulation
        when needed."""

    @abc.abstractmethod
    def HandleExternalUpdates(self, root_context: Context) -> None:
        """Update the simulator's context as needed."""


class CompositeMonitor(HardwareStationMonitor):
    """A composite monitor that can contain multiple monitors of different
    types. Each monitor can be added at most once. """

    def __init__(self):
        super().__init__()
        self._monitors: list[HardwareStationMonitor] = []

    def Monitor(self, root_context: Context) -> EventStatus:
        for monitor in self._monitors:
            status = monitor.Monitor(root_context)
            if status.severity() != EventStatus.Severity.kDidNothing:
                return status
        return EventStatus.DidNothing()

    def HandleExternalUpdates(self, root_context: Context) -> None:
        for monitor in self._monitors:
            monitor.HandleExternalUpdates(root_context)

    def get_monitor_by_type(self, monitor_class):
        """Return the only monitor of the given type if one exists, otherwise
        return None."""
        for monitor in self._monitors:
            if isinstance(monitor, monitor_class):
                return monitor
        return None

    def add_monitor(self, monitor: HardwareStationMonitor) -> None:
        # Throw an error if the monitor is already added
        if self.get_monitor_by_type(type(monitor)) is not None:
            raise ValueError(f"Monitor of type {type(monitor)} already added.")
        self._monitors.append(monitor)
