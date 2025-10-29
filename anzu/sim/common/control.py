import numpy as np

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.framework import BasicVector, LeafSystem, PortDataType

import anzu.common.multibody_extras as me
import anzu.common.multibody_plant_subgraph as msub
from anzu.sim.common.primitive_systems import ConstantSource

# TODO(eric.cousineau): Keep alive fixes:
# - https://github.com/robotlocomotion/drake/pull/13075 - didn't work :(
# - InverseDynamicsController - doesn't keep control plant alive
_keep_alive = []


def _is_floating_body(body):
    # TODO(12596): The body.has_quaternion_dofs() call works around an
    # issue where manipulands can be added with explicit SE(3) joints. This
    # ought to be generalized for all applicable 6-DoF joints, not just
    # those with quaternions.
    return body.is_floating() or body.has_quaternion_dofs()


def build_with_no_control(builder, plant, model_instance):
    """
    Attaches zeros to the given `model`s actuation port on `plant`.

    Args:
        builder: DiagramBuilder instance.
        plant: MultibodyPlant instance.
        model: ModelInstanceIndex
    """
    nu = plant.num_actuated_dofs(model_instance)
    constant = builder.AddSystem(ConstantSource(np.zeros(nu)))
    builder.Connect(
        constant.get_output_port(0),
        plant.get_actuation_input_port(model_instance))


def build_with_no_control_all(builder, plant):
    for i in range(plant.num_model_instances()):
        model = ModelInstanceIndex(i)
        build_with_no_control(builder, plant, model)


def make_control_model(plant, controlled_model_instances):
    """
    Python version of `MakeIiwaControllerModel`, but more generalized.

    Args:
        plant: MultibodyPlant instance.
        controlled_model_instances: A Set[ModelInstanceIndex], where each model
            will be articulated in the resulting plant. (Ordering is the same
            as in the original plant).
    Returns:
        (control_plant, to_control):
            control_plant: MultibodyPlant for control.
            to_control: MultibodyPlantElementsMap mapping from `plant` to
                `control_plant`.
    """
    # TODO(eric.cousineau): Make `controlled_models` be a subgraph (for more
    # granularity) instead?
    control_plant = MultibodyPlant(time_step=0.)
    subgraph = msub.MultibodyPlantSubgraph(
        msub.get_elements_from_plant(plant))
    # Remove any models that have floating bodies from original plant (e.g.
    # manipulands).
    for body in me.get_bodies(plant):
        if _is_floating_body(body):
            if body.model_instance() in controlled_model_instances:
                # TODO(eric.cousineau): This may be a bit funky...
                raise RuntimeError(
                    "The control model must be welded in the original model")
            subgraph.remove_model_instance(body.model_instance())
    # Freeze any joints not part of the models we care about.
    model_instances = subgraph.elements_src.model_instances
    freeze_model_instances = model_instances - set(controlled_model_instances)
    subgraph.apply_policy(msub.FreezeJointSubgraphPolicy.from_plant(
        plant, plant.CreateDefaultContext(),
        model_instances=freeze_model_instances))
    to_control = subgraph.add_to(control_plant)
    control_plant.Finalize()
    return control_plant, to_control


class SlewRateSystem(LeafSystem):
    """
    Takes in positions and maximum velocity, and saturates commanded position
    based on maximum element-wise velocity v_upper, v_lower, and period_sec.

    Inputs:
        q: Positions.
    Outputs:
        q_limited
        v_limited
    State:
        q_prev

    Outputs rate-limited position and saturated velocities.
    """
    # TODO(eric.cousineau): Hoist to Drake C++.
    def __init__(self, period_sec, v_lower, v_upper):
        LeafSystem.__init__(self)
        v_lower = np.asarray(v_lower)
        v_upper = np.asarray(v_upper)
        assert v_lower.dtype == float
        assert v_upper.dtype == float
        assert v_lower.ndim == 1
        assert v_lower.shape == v_upper.shape
        (num_dof,) = v_lower.shape
        self._v_lower = v_lower
        self._v_upper = v_upper
        self._period_sec = period_sec
        self._q_prev_index = int(
            self.DeclareDiscreteState(np.full(num_dof, np.nan))
        )
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec, 0.0, self._update)
        self._q_input_port = self.DeclareInputPort(
            "q", PortDataType.kVectorValued, num_dof)
        self._q_limited_output_port = self.DeclareVectorOutputPort(
            "q_limited", BasicVector(num_dof), self._output_q_limited)
        self._v_limited_output_port = self.DeclareVectorOutputPort(
            "v_limited", BasicVector(num_dof), self._output_v_limited)

    def q_input_port(self):
        return self._q_input_port

    def q_limited_output_port(self):
        return self._q_limited_output_port

    def v_limited_output_port(self):
        return self._v_limited_output_port

    def _calc_q_prev(self, context):
        q_prev = context.get_discrete_state(self._q_prev_index).get_value()
        if np.all(~np.isfinite(q_prev)):
            # Will have zero velocity for first step.
            q_prev = self._q_input_port.Eval(context)
        return q_prev

    def _calc_q_and_v_limited(self, context):
        # N.B. This formulation has direct feedthrough on q for all t :(
        q = self._q_input_port.Eval(context)
        q_prev = self._calc_q_prev(context)
        assert np.isfinite(q).all()
        assert np.isfinite(q_prev).all()
        dt = self._period_sec
        # Finite difference.
        v = (q - q_prev) / dt
        v_limited = np.clip(v, self._v_lower, self._v_upper)
        # Euler integrate post-saturation.
        q_limited = q_prev + dt * v_limited
        return q_limited, v_limited

    def _output_q_limited(self, context, output):
        q_limited, _ = self._calc_q_and_v_limited(context)
        output.set_value(q_limited)

    def _output_v_limited(self, context, output):
        _, v_limited = self._calc_q_and_v_limited(context)
        output.set_value(v_limited)

    def _update(self, context, discrete_state):
        q_limited, _ = self._calc_q_and_v_limited(context)
        q_prev_value = discrete_state.get_mutable_vector(self._q_prev_index)
        q_prev_value.set_value(q_limited)
