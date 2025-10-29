import dataclasses

import numpy as np

from pydrake.manipulation import SchunkWsgTrajectoryGenerator
from pydrake.multibody.parsing import ModelInstanceInfo
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.framework import Context, DiagramBuilder, VectorSystem
from pydrake.systems.lcm import LcmBuses
from pydrake.systems.primitives import Gain, MatrixGain


@dataclasses.dataclass(kw_only=True)
class SchunkWsgImplicitPdDriver:
    """
    Declares a simple implicit PD Schunk gripper controller (it _never_ adds
    LCM systems). Applying this driver will add the necessary control subsystem
    and validate the model instance to which it is applied.
    """
    pass


class _MultibodyForceToHandForceSystem(VectorSystem):
    """
    Maps the individual forces reporting on the gripper's two fingers and
    combines them into a single force.
    """
    def __init__(self):
        VectorSystem.__init__(self, input_size=2, output_size=1,
                              direct_feedthrough=True)

    def DoCalcVectorOutput(self,
                           context: Context,
                           u: np.ndarray,
                           x: np.ndarray,
                           output: np.ndarray):
        self.ValidateContext(context=context)
        # gripper force = abs(-finger0 + finger1).
        output[0] = np.abs(u[0] - u[1])


def _make_multibody_state_to_hand_state_system():
    """
    Converts multibody state for the two gripper fingers, [q₀, q₁, v₀, v₁] such
    that q₀ = -q₁ and v₀ = -v₁, into the driver state [q, v] (the gripper is a
    one-dof entity) as follows:

       y = |-1 1  0 0| * |q₀ q₁ v₀ v₁| = |(-q₀ + q₁)| = |2q₁| = |-2q₀|
           | 0 0 -1 1|                   |(-v₀ + v₁)|   |2v₁|   |-2v₀|
    """
    D = np.array(((-1, 1, 0, 0), (0, 0, -1, 1)), dtype=float)
    return MatrixGain(D)


def _build_hand_control(plant: MultibodyPlant,
                        hand_instance: ModelInstanceIndex,
                        builder: DiagramBuilder):
    # Confirm that the model instance has the appropriate actuation for this
    # driver.
    actuator_indices = plant.GetJointActuatorIndices(hand_instance)
    if len(actuator_indices) == 2:
        raise RuntimeError(
            "The Schunk hand does not support implicit PD control -- it's "
            "missing a mimic joint. But the driver was configured as "
            "SchunkWsgImplicitPdDriver.")
    assert len(actuator_indices) == 1, f"Actuator count: {len(actuator_indices)}"  # noqa
    assert plant.get_joint_actuator(actuator_indices[0]).has_controller(), "No controller"  # noqa

    driver_builder = DiagramBuilder()

    # TODO(jeremy.nimmer) These constants should not be hard-coded, instead
    # they should be scraped from the plant's model instance to reflect the
    # model was actually loaded.  Since we plan to rewrite this indexing logic
    # along those lines in the future, we won't bother binding the C++
    # constants for reuse here, we'll just hard-code the same invariant values
    # as C++.
    num_schunk_positions = 2
    num_schunk_velocities = 2
    schunk_position_index = 0
    wsg_trajectory_generator = driver_builder.AddSystem(
        SchunkWsgTrajectoryGenerator(
            input_size=num_schunk_positions + num_schunk_velocities,
            position_index=schunk_position_index,
            use_force_limit=False))

    driver_builder.ExportInput(
        wsg_trajectory_generator.get_desired_position_input_port(), "position")
    multibody_state_index = driver_builder.ExportInput(
        wsg_trajectory_generator.get_state_input_port(), "multibody_state")

    # The trajectory generator outputs the desired width between the fingers,
    # and its derivative. The mimic joint measures the position of a single
    # finger joint, so we halve the desired trajectory generator output here,
    # since the fingers move symmetrically.
    half_gain = driver_builder.AddSystem(Gain(0.5, 2))
    driver_builder.Connect(wsg_trajectory_generator.get_target_output_port(),
                           half_gain.get_input_port())
    driver_builder.ExportOutput(half_gain.get_output_port(), "desired_state")

    mbp_state_to_driver_state = driver_builder.AddSystem(
        _make_multibody_state_to_hand_state_system())
    driver_builder.ConnectInput(multibody_state_index,
                                mbp_state_to_driver_state.get_input_port())
    driver_builder.ExportOutput(mbp_state_to_driver_state.get_output_port(),
                                "state")

    mbp_force_to_hand_force = driver_builder.AddSystem(
        _MultibodyForceToHandForceSystem())
    driver_builder.ExportInput(mbp_force_to_hand_force.get_input_port(),
                               "generalized_contact_forces")
    driver_builder.ExportOutput(mbp_force_to_hand_force.get_output_port(),
                                "force")

    driver = builder.AddSystem(driver_builder.Build())

    builder.Connect(plant.get_state_output_port(hand_instance),
                    driver.GetInputPort("multibody_state"))
    builder.Connect(
        plant.get_generalized_contact_forces_output_port(hand_instance),
        driver.GetInputPort("generalized_contact_forces"))
    builder.Connect(driver.GetOutputPort("desired_state"),
                    plant.get_desired_state_input_port(hand_instance))

    return driver


def ApplyDriverConfig(
    driver_config: SchunkWsgImplicitPdDriver,
    model_instance_name: str,
    sim_plant: MultibodyPlant,
    models_from_directives: dict[str, ModelInstanceInfo],
    lcms: LcmBuses,
    builder: DiagramBuilder
) -> None:
    if model_instance_name not in models_from_directives:
        raise RuntimeError(
            "SchunkWsgImplicitPdDriver could not find the hand model "
            f"directive '{model_instance_name}' to actuate")
    hand_model = models_from_directives[model_instance_name]
    driver = _build_hand_control(sim_plant, hand_model.model_instance, builder)
    driver.set_name(model_instance_name)
