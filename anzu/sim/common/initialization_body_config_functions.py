from pydrake.geometry import SceneGraph
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Context, EventStatus

from anzu.sim.common.hardware_station_monitor import (
    CompositeMonitor,
    HardwareStationMonitor,
)
from anzu.sim.common.initialization_body_config import InitializationBodyConfig


class InitializationBodyMonitor(HardwareStationMonitor):
    """This class is a simulation monitor designed to (effectively) remove
    initialization bodies at a specified time.
    """

    def __init__(
        self,
        config: InitializationBodyConfig,
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
    ):
        super().__init__()
        assert plant.is_finalized()
        self._plant = plant
        self._scene_graph = scene_graph
        self._t_remove = config.t_remove
        self._instance_names: list[str] = config.names
        self._body_list = []
        self._removed = False
        # Loop over all bodies to confirm that they exist and are either
        # floating bodies or locked to the world.
        for instance_name in self._instance_names:
            model_instance = self._plant.GetModelInstanceByName(instance_name)
            body_indices = self._plant.GetBodyIndices(model_instance)
            for body_index in body_indices:
                body = self._plant.get_body(body_index)
                self._body_list.append(body)
            if not self._plant.IsAnchored(body) and not body.is_floating():
                raise ValueError(
                    f"""Initialization body {body.name()} is neither floating
                    or anchored.""")

    def Monitor(self, root_context: Context) -> EventStatus:
        """Callback for Simulator.set_monitor that computes breaks out of the
        simulation once and only once when the time for removal has reached
        and the removal hasn't happened yet.
        Users should call the LockBodiesAndRemoveGeometry() method to
        effectively remove the initialization bodies."""
        if self._needs_update(root_context):
            return EventStatus.ReachedTermination(
                None, "Remove initialization bodies"
            )
        return EventStatus.DidNothing()

    def _needs_update(self, root_context: Context) -> bool:
        """Returns True iff we need to stop the sim to remove bodies."""
        return not self._removed and root_context.get_time() >= self._t_remove

    def HandleExternalUpdates(self, root_context: Context) -> None:
        """Locks the initialization bodies to the world body (if they aren't
        already welded to world) to avoid solving for their dynamics. Removes
        geometries associated with the initialization bodies so that they don't
        interfere with contact and visualization."""
        if not self._needs_update(root_context):
            return

        # Lock the bodies if needed.
        plant_context = self._plant.GetMyMutableContextFromRoot(root_context)
        for body in self._body_list:
            if not self._plant.IsAnchored(body):
                body.Lock(plant_context)
        sg_context = self._scene_graph.GetMyMutableContextFromRoot(
            root_context
        )
        for body in self._body_list:
            collision_ids = self._plant.GetCollisionGeometriesForBody(body)
            visual_ids = self._plant.GetVisualGeometriesForBody(body)
            for g_id in collision_ids + visual_ids:
                self._scene_graph.RemoveGeometry(
                    sg_context, self._plant.get_source_id(), g_id
                )
        self._removed = True


def ApplyInitializationBodyConfig(
    config: InitializationBodyConfig,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    monitor: CompositeMonitor,
) -> None:
    """Prepares the given plant and scene_graph for removing the transient
    geometries and locking the initialiation bodies.
    Adds the InitializationBodyMonitor to the given CompositeMonitor.
    @pre The plant is finalized.
    """
    monitor.add_monitor(InitializationBodyMonitor(config, plant, scene_graph))
