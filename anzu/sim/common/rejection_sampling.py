"""This file contains the Plant-aware portions of rejection sampling, to
separate the syntactic and kinematic parts of scenario processing.
"""

import copy

import numpy as np

from pydrake.common import RandomGenerator
from pydrake.common.yaml import yaml_dump, yaml_load_typed
from pydrake.multibody.tree import ModelInstanceIndex


def _plant_and_context_from_scenario(*, make_hardware_station, scenario):
    """Create a simulator plant from the given scenario, as well as a context
    for that plant reflecting the configured initial conditions. Certain
    irrelevant scenario elements such as cameras and drivers will be skipped.
    """
    # We'll nerf the scenario a bit, to avoid spending time creating pieces of
    # it that either we don't need, or that cause trouble when not connected.
    fast_scenario = copy.deepcopy(scenario)
    try:
        fast_scenario.cameras.clear()
    except AttributeError:
        pass
    try:
        # Rejection sampling does not support a HardwareStationMonitor, so we
        # must disable this.
        fast_scenario.item_locking = None
    except AttributeError:
        pass
    try:
        # Rejection sampling does not support a HardwareStationMonitor, so we
        # must disable this.
        fast_scenario.initialization_bodies = None
    except AttributeError:
        pass
    try:
        fast_scenario.logging.logs.clear()
    except AttributeError:
        pass
    try:
        fast_scenario.model_drivers.clear()
    except AttributeError:
        pass
    try:
        fast_scenario.visualization.enable_meshcat_creation = False
    except AttributeError:
        pass

    # We're not going to step time, so we need instantanous output port values,
    # not sampled output ports.
    fast_scenario.plant_config.use_sampled_output_ports = False

    # Create the station, based on the scenario.
    (diagram, logging, monitor) = make_hardware_station(fast_scenario)
    assert monitor is None
    diagram_context = diagram.CreateDefaultContext()
    random = RandomGenerator(scenario.random_seed)
    diagram.SetRandomContext(diagram_context, random)

    # Provide dummy values for mandatory plant input ports so that contact
    # results evaluation doesn't crash.
    plant = diagram.plant()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    for i in range(plant.num_model_instances()):
        port = plant.get_desired_state_input_port(ModelInstanceIndex(i))
        if port.size() == 0:
            continue
        port.FixValue(plant_context, np.zeros(port.size()))

    return diagram, plant, plant_context, logging


def satisfies_constraints(*, schema, make_hardware_station,
                          scenario_dict: dict):
    """Return True iff the passed-in scenario satisfies its own specified
    constraints.
    """
    assert schema is not None
    assert make_hardware_station is not None
    scenario_str = yaml_dump(scenario_dict)
    scenario = yaml_load_typed(data=scenario_str, schema=schema)
    diagram, plant, context, logging = _plant_and_context_from_scenario(
        make_hardware_station=make_hardware_station, scenario=scenario)
    return scenario.preconditions.Satisfied(plant=plant, plant_context=context)
