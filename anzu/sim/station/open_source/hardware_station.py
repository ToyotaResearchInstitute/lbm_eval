# WARNING: This file is auto-generated; see README.md.
"""This file provides support for an in-process variant of the
hardware_station_simulation.py.
"""
from typing import Callable, List

from pydrake.geometry import Meshcat
from pydrake.manipulation import (
    ApplyDriverConfigs,
    ApplyNamedPositionsAsDefaults,
)
from pydrake.multibody.parsing import ModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import ApplyMultibodyPlantConfig
from pydrake.planning import RobotDiagramBuilder
from pydrake.systems.framework import PortDataType
from pydrake.systems.lcm import ApplyLcmBusConfig
from pydrake.visualization import ApplyVisualizationConfig

from anzu.common.anzu_model_directives import add_default_anzu_packages
from anzu.sim.camera.camera_config import AnzuCameraConfig
from anzu.sim.camera.camera_config_functions import ApplyAnzuCameraConfig
from anzu.sim.common.hardware_station_monitor import CompositeMonitor
from anzu.sim.common.initialization_body_config_functions import (
    ApplyInitializationBodyConfig,
)
from anzu.sim.common.item_locking_monitor_config_functions import (
    ApplyItemLockingMonitorConfig,
)
from anzu.sim.common.logging_config_functions import ApplyLoggingConfig
from anzu.sim.station.open_source.hardware_station_simulation_scenario import (
    Scenario as OpenSourceScenario,
)


def MakeHardwareStation(
    scenario: OpenSourceScenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: List[str] = None,
    parser_preload_callback: Callable[[RobotDiagramBuilder], None] = None,
    parser_prefinalize_callback: Callable[[RobotDiagramBuilder], None] = None,
    prebuild_callback: Callable[[RobotDiagramBuilder], None] = None,
    export_ports=True,
) -> tuple:
    """Make a diagram encapsulating a simulation of (or the communications
    interface to/from) a physical robot, including sensors and controllers.
    Note: The `simulator_config` and `deformable_sim_config` settings portion
    of the Scenario are *not* applied in this function. It is expected that the
    caller of this function will subsequently invoke ApplySimulatorConfig and
    ApplyDeformableSimConfig separately.

    Args:
        scenario: A scenario structure, populated using the yaml_load_typed
            method.

        meshcat: If not None, then this meshcat instance is passed through to
            ApplyVisualizationConfig and will be added to the subdiagram.

        package_xmls: A list of package.xml file paths that will be passed to
            the parser, using Parser.AddPackageXml().

        parser_preload_callback: A callback function that will be called after
            the Parser is created, but before any directives are processed.
            This can be used to add additional packages to the parser, or to
            add additional model directives.

        parser_prefinalize_callback: A callback function that will be called
            after the directives are processed, but before the plant is
            finalized. This can be used to add additional model directives.

        prebuild_callback: A callback function that will be called after the
            diagram builder is created, but before the diagram is built. This
            can be used to add additional systems to the diagram.

        export_ports: A boolean flag which designates whether to export all
            unconnected and unexported ports created by the scenario's builder.
            This defaults to `True` for use in a single-process simulator
            environment.
            Set this value to `False` if LCM communication is intended as the
            communication method to control the simulated environment.
    Returns:
        Returns a tuple containing:
        - The RobotDiagram populated via the directives in `scenario`.
        - The configured LoggingScope object for object lifetime purposes.
        - A CompositeMonitor if specified through the Scenario, else None.
          If it is not None, the returned monitor must be registered with an
          externally constructed Simulator object via `set_monitor()` method
          call, and the monitor's HandleExternalUpdates() method should be
          invoked before every Simulator.AdvanceTo() step.
    """
    if meshcat is None and scenario.visualization.enable_meshcat_creation:
        # NOTE:We need a meshcat instance for some of the logic, below.
        # Create it now so that all of the helper functions will share
        # the same object.  It's critical that we create _at most one_
        # meshcat instance to avoid confusing users with multiple
        # http ports or empty scenes.
        meshcat = Meshcat()
    robot_builder = RobotDiagramBuilder(
        time_step=scenario.plant_config.time_step
    )
    builder = robot_builder.builder()
    sim_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    ApplyMultibodyPlantConfig(scenario.plant_config, sim_plant)
    scene_graph.set_config(scenario.scene_graph_config)

    is_open_source = isinstance(scenario, OpenSourceScenario)

    parser = robot_builder.parser()
    if package_xmls is None:
        # TODO(jeremy-nimmer) It's not clear how this defaulting should
        # generalize when used outside of Anzu.
        add_default_anzu_packages(parser.package_map())
    else:
        for package_xml in package_xmls:
            parser.package_map().AddPackageXml(package_xml)

    if parser_preload_callback:
        parser_preload_callback(robot_builder)

    # Some scenario elements add items to the plant, or support items
    # added to the plant.  We apply those first.
    models_from_directives = ProcessModelDirectives(
        ModelDirectives(directives=scenario.directives),
        plant=sim_plant,
        parser=parser,
    )

    lcm_buses = ApplyLcmBusConfig(scenario.lcm_buses, builder)

    # Apply the camera configs.
    for camera in scenario.cameras:
        if (
            not isinstance(camera.renderer_class, str)
            and camera.renderer_class.environment_map
        ):
            env_map_url = camera.renderer_class.environment_map.texture.path
            env_map_path = parser.package_map().ResolveUrl(env_map_url)
            camera.renderer_class.environment_map.texture.path = env_map_path
        if isinstance(camera, AnzuCameraConfig):
            camera_lcm = lcm_buses.Find(
                f"Camera '{camera.name}'",
                camera.lcm_bus,
            )
            ApplyAnzuCameraConfig(
                config=camera,
                builder=builder,
                lcm_buses=None,
                plant=sim_plant,
                scene_graph=scene_graph,
                lcm=camera_lcm,
            )
        else:
            if is_open_source:
                raise NotImplementedError(camera)

    if parser_prefinalize_callback:
        parser_prefinalize_callback(robot_builder)

    sim_plant.Finalize()

    monitor = None
    if scenario.initialization_bodies:
        if monitor is None:
            monitor = CompositeMonitor()
        ApplyInitializationBodyConfig(
            config=scenario.initialization_bodies,
            plant=sim_plant,
            scene_graph=scene_graph,
            monitor=monitor,
        )

    # Configure item locking post-Finalize so we can identify free bodies.
    if scenario.item_locking:
        if monitor is None:
            monitor = CompositeMonitor()
        ApplyItemLockingMonitorConfig(
            config=scenario.item_locking,
            plant=sim_plant,
            scene_graph=scene_graph,
            monitor=monitor,
        )

    # Apply any logging configuration; return the logging object to keep
    # it alive.
    # TODO(russt): Don't require caller to hold on to the logger object for
    # lifetime purposes.
    logging = ApplyLoggingConfig(
        config=scenario.logging,
        robot_builder=robot_builder,
        meshcat=meshcat,
    )

    # The remaining scenario items apply systems to the diagram, now that
    # the plant has finalized its ports.
    ApplyVisualizationConfig(
        scenario.visualization, builder, lcm_buses, meshcat=meshcat
    )
    ApplyDriverConfigs(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        models_from_directives=models_from_directives,
        lcm_buses=lcm_buses,
        builder=builder,
    )

    # Finally, now that the diagram topology is finished, we can build
    # the initial state for the context.
    ApplyNamedPositionsAsDefaults(scenario.initial_position, sim_plant)

    if prebuild_callback:
        prebuild_callback(robot_builder)

    # Export ports.
    if export_ports:
        for system in builder.GetSystems():
            for i in range(system.num_input_ports()):
                port = system.get_input_port(i, warn_deprecated=False)
                if builder.IsConnectedOrExported(port):
                    # Skip ports that are already connected (or exported)
                    continue
                if (
                    port.get_data_type() == PortDataType.kVectorValued
                    and port.size() == 0
                ):
                    # Skip ports that cannot carry any data
                    continue
                # TODO(jeremy.nimmer) Should we skip deprecated ports here?
                builder.ExportInput(
                    port, f"{system.get_name()}.{port.get_name()}"
                )
            for i in range(system.num_output_ports()):
                port = system.get_output_port(i, warn_deprecated=False)
                if (
                    port.get_data_type() == PortDataType.kVectorValued
                    and port.size() == 0
                ):
                    # Skip ports that cannot carry any data
                    continue
                # TODO(jeremy.nimmer) Should we skip deprecated ports here?
                builder.ExportOutput(
                    port, f"{system.get_name()}.{port.get_name()}"
                )

    diagram = robot_builder.Build()
    diagram.set_name("station")
    return (diagram, logging, monitor)
