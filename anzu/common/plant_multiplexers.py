import numpy as np

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.primitives import Selector, SelectorParams


def _get_position_indices(plant: MultibodyPlant, i: ModelInstanceIndex):
    """Given a plant and model instance, returns an ndarray[dtype=int] with the
    indices of that model's positions in the whole-plant positions vector.
    """
    nq = plant.num_positions()
    return plant.GetPositionsFromArray(i, np.arange(nq)).astype(int)


def _get_velocity_indices(plant: MultibodyPlant, i: ModelInstanceIndex):
    """Given a plant and model instance, returns an ndarray[dtype=int] with the
    indices of that model's velocities in the whole-plant velocities vector.
    """
    nv = plant.num_velocities()
    return plant.GetVelocitiesFromArray(i, np.arange(nv)).astype(int)


def MakeCommandDemultiplexer(*,
                             plant: MultibodyPlant,
                             model_instance_names: list[str]):
    """Returns a system that demultiplexes desired position and desired
    velocity into per-model-instance outputs for the model instances named by
    `model_instance_names`. The order of individual positions (and velocities)
    on the input ports matches their relative order in the full-plant state,
    but note that the input ports are NOT necessarily full-plant-sized; they
    only contain the given model instances.

    input_ports:
    - position
    - velocity
    output_ports:
    - {model_instance_names[0]}.position
    - {model_instance_names[0]}.velocity
    - ...
    - {model_instance_names[N-1]}.position
    - {model_instance_names[N-1]}.velocity
    """
    return Selector(_MakeCommandDemultiplexerSelectorParams(
        plant=plant,
        model_instance_names=model_instance_names,
    ))


def _MakeCommandDemultiplexerSelectorParams(*,
                                            plant: MultibodyPlant,
                                            model_instance_names: list[str]):
    """Implementation logic for MakeCommandDemultiplexer, immediately above."""
    nq = plant.num_positions()
    nv = plant.num_velocities()
    num_models = plant.num_model_instances()

    # Prepare our lists with properly shaped structs filled with invalid data.
    # In our loops below, we'll replace the invalid data with the correct data.
    inputs = [
        SelectorParams.InputPortParams(name=attribute)
        for attribute in ["position", "velocity"]
    ]
    outputs = [
        SelectorParams.OutputPortParams(
            name=f"{model_instance_name}.{attribute}",
        )
        for model_instance_name in model_instance_names
        for attribute in ["position", "velocity"]
    ]

    # Build tables to help map a full-plant position (or velocity) index to the
    # local index in our position (or velocity) input port index. Each table is
    # a tally of how many elements we've skipped as of the i'th offset.
    position_adjust = np.zeros(nq, dtype=int)
    velocity_adjust = np.zeros(nv, dtype=int)
    for model_index in [ModelInstanceIndex(i) for i in range(num_models)]:
        if plant.GetModelInstanceName(model_index) not in model_instance_names:
            # Set indices which are not demultiplexed to 1.
            position_adjust[_get_position_indices(plant, model_index)] = 1
            velocity_adjust[_get_velocity_indices(plant, model_index)] = 1
    # Sum the overall adjustments cumulatively.
    position_adjust = np.cumsum(position_adjust)
    velocity_adjust = np.cumsum(velocity_adjust)

    # Loop over all models to assign the input->output mapping -- first for q
    # (the even-numbered output ports) and then for v (the odd-numbered ports).
    output_port_index = 0
    for model_instance_name in model_instance_names:
        model_index = plant.GetModelInstanceByName(model_instance_name)
        model_nq = plant.num_positions(model_index)
        input_port_index = 0
        inputs[input_port_index].size += model_nq
        output_selections = [
            SelectorParams.OutputSelection()
            for _ in range(model_nq)
        ]
        for i, q_index in enumerate(_get_position_indices(plant, model_index)):
            output_selections[i].input_port_index = input_port_index
            output_selections[i].input_offset = (
                q_index - position_adjust[q_index]
            )
        outputs[output_port_index].selections = output_selections
        output_port_index += 2
    output_port_index = 1
    for model_instance_name in model_instance_names:
        model_index = plant.GetModelInstanceByName(model_instance_name)
        model_nv = plant.num_velocities(model_index)
        input_port_index = 1
        inputs[input_port_index].size += model_nv
        output_selections = [
            SelectorParams.OutputSelection()
            for _ in range(model_nv)
        ]
        for i, v_index in enumerate(_get_velocity_indices(plant, model_index)):
            output_selections[i].input_port_index = input_port_index
            output_selections[i].input_offset = (
                v_index - velocity_adjust[v_index]
            )
        outputs[output_port_index].selections = output_selections
        output_port_index += 2

    return SelectorParams(inputs=inputs, outputs=outputs)


def MakeStateMultiplexer(*, plant: MultibodyPlant):
    """Returns a system with one input port for each model instance's state in
    `plant`. The system will have one output port for the state of the `plant`.

    input_ports:
    - {model_instance_name[0]}.state
    - ...
    - {model_instance_name[N-1]}.state
    output_ports:
    - state
    """
    return Selector(_MakeStateMultiplexerSelectorParams(plant=plant))


def _MakeStateMultiplexerSelectorParams(*, plant: MultibodyPlant):
    """Implementation logic for MakeStateMultiplexer, immediately above."""
    nq = plant.num_positions()
    nv = plant.num_velocities()
    num_models = plant.num_model_instances()

    # Prepare our lists with the proper shape but filled with invalid data.
    # In our loop below, we'll replace the invalid data with the correct data.
    inputs = [
        SelectorParams.InputPortParams()
        for _ in range(num_models)
    ]
    output_selections = [
        SelectorParams.OutputSelection()
        for _ in range(nq + nv)
    ]

    # Loop over all models to assign the input->output mapping.
    for i in [ModelInstanceIndex(i) for i in range(num_models)]:
        model_nq = plant.num_positions(i)
        model_nv = plant.num_velocities(i)
        inputs[i].name = f"{plant.GetModelInstanceName(i)}.state"
        inputs[i].size = model_nq + model_nv
        for j, k in enumerate(_get_position_indices(plant, i)):
            output_selections[k].input_port_index = i
            output_selections[k].input_offset = j
        for j, k in enumerate(_get_velocity_indices(plant, i)):
            output_selections[nq + k].input_port_index = i
            output_selections[nq + k].input_offset = model_nq + j

    # Wrap our params into the necessary return type.
    return SelectorParams(
        inputs=inputs,
        outputs=[
            SelectorParams.OutputPortParams(
                name="state",
                selections=output_selections,
            ),
        ],
    )
