import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import (
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    LeafSystem,
)
from pydrake.systems.primitives import Saturation


# TODO(dale.mcconachie): Switch to using the C++ implementation of this as a
# drake system.
# NOTE: Test coverage is handled in cc_py_test.py.
def head_pose_to_yaw_pitch(X_TH: RigidTransform) -> np.ndarray:
    """
    Converts a commanded head pose into a yaw-pitch command to be sent to the
    joint position stream primitive.

    """
    rpy_TH = X_TH.rotation().ToRollPitchYaw()
    return np.array([rpy_TH.yaw_angle(), rpy_TH.pitch_angle()])


class _YawPitchControl(LeafSystem):
    """Converts a commanded head pose into a yaw-pitch command to be sent
    directly to the head model driver.
    """

    def __init__(self):
        super().__init__()

        self._X_TH_input = self.DeclareAbstractInputPort(
            "X_TH", AbstractValue.Make(RigidTransform())
        )
        self._q_output = self.DeclareVectorOutputPort(
            "q_desired", BasicVector(2), self.calc_output
        )

    def calc_output(self, context: Context, output: BasicVector):
        X_TH = self._X_TH_input.Eval(context)
        output.SetFromVector(head_pose_to_yaw_pitch(X_TH))


class _MaxCommandDelta(LeafSystem):
    """Saturates the output to be within a maximum delta from the input."""

    # Note: In theory this could be implemented with a combination of Adder
    # and Saturation blocks, but this is more interpretable and easier to
    # reason about.
    def __init__(self, size: int, max_delta: float):
        super().__init__()
        self._max_delta = np.full(size, max_delta)

        self._q_current_port = self.DeclareVectorInputPort("q_current", size)
        self._q_desired = self.DeclareVectorInputPort("q_desired", size)
        self._q_commanded = self.DeclareVectorOutputPort(
            "q_commanded", size, self.calc_q_commanded
        )

    def get_q_desired_port(self):
        """Returns the input port for the desired joint positions."""
        return self._q_desired

    def get_q_current_port(self):
        """Returns the input port for the current joint positions."""
        return self._q_current_port

    def calc_q_commanded(self, context: Context, output: BasicVector):
        q_current = self._q_current_port.Eval(context)
        q_desired = self._q_desired.Eval(context)
        q_commanded = np.clip(
            q_desired,
            q_current - self._max_delta,
            q_current + self._max_delta,
        )
        output.set_value(q_commanded)


def make_head_control_system(
    plant: MultibodyPlant, head_model_name: str
) -> Diagram:
    """Creates a diagram that mimics the behavior of the head control processes
    in `rby1_control_main.cc` for use in a single-threaded simulation.
    """
    builder = DiagramBuilder()

    yaw_pitch_control = builder.AddSystem(_YawPitchControl())

    # Commanded joint position saturation block; lookup the joint limits for
    # the head model.
    head_model = plant.GetModelInstanceByName(head_model_name)
    min_q = []
    max_q = []
    for index in plant.GetJointIndices(head_model):
        joint = plant.get_joint(index)
        min_q.extend(joint.position_lower_limits())
        max_q.extend(joint.position_upper_limits())
    joint_limits = builder.AddNamedSystem(
        "JointLimits", Saturation(min_q, max_q)
    )
    num_dof = len(min_q)

    # Maximum commanded rate of change for the head yaw and pitch; mimics the
    # behavior of ControlModeDifferentialInverseKinematics::set_head_command()
    # plus MoveJointPositionStream::set_qd().
    max_command_delta = builder.AddSystem(
        # TODO(dale.mcconachie): Extract this max_delta value from some central
        # place.
        _MaxCommandDelta(size=num_dof, max_delta=0.1)
    )

    # Wire everything together and export the ports. Broadly speaking data
    # flows in the order that ports are exported and connected here.
    builder.ExportInput(max_command_delta.get_q_current_port(), "q_current")
    builder.ExportInput(yaw_pitch_control.get_input_port(), "X_TH_desired")
    builder.Connect(
        yaw_pitch_control.get_output_port(), joint_limits.get_input_port()
    )
    builder.Connect(
        joint_limits.get_output_port(), max_command_delta.get_q_desired_port()
    )
    builder.ExportOutput(max_command_delta.get_output_port(), "q_commanded")

    return builder.Build()
