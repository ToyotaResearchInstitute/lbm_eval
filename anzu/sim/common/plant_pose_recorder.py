import logging
from pathlib import Path

import numpy as np

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.framework import Context, LeafSystem


class PlantPoseRecorder(LeafSystem):
    """A leaf system that periodically records the poses of models into a text
    file throughout simulation, which then can be used, e.g. for offline
    rendering. Only the models with DoFs will be recorded.

    The output file format follows the following structure:
    - It starts with a line `time: {simulation_time}` to represent the
      simulation time at which subsequent poses are recorded.

    - The pose of each model is written on a single line starting with
      `{model_name}: ` followed by `{pose_vector}`.

    - The format for the `{pose_vector}` differs depending on whether it is a
      robot component or an object.

      For robot components, such as arms and/or grippers, it lists joint-name
      strings first followed by a `:` as a separator, then the corresponding
      joint values.

        Example:
          gripper_right: right_finger_sliding_joint left_finger_sliding_joint : -0.05 0.05  # noqa

      For objects, it takes a quaternion as rotation, then position,
      i.e. `{qw qx qy qz px py pz}`

    - At least one space will be used between items as a separator.

    TODO(jeremy-nimmer) Everything about this implementation is terrible:
    the custom syntax with floats saved in ASCII, no way to cross-check that
    the kinematic tree whose state we logged matches the kinematic tree we're
    loading into. If we're going to keep using this file, we need to completely
    re-design it from scratch."""

    def __init__(self, plant: MultibodyPlant, filename: Path, period: float):
        """Creates a recorder for the given `plant`.
        The `filename` is the path to record poses to.
        The `period` is the period in seconds to record poses."""
        LeafSystem.__init__(self)

        logging.debug("Saving keyframes to {}.", filename)

        self._plant = plant
        self._file = open(filename, "w", encoding="utf-8")
        self._model_names = []

        self.DeclareVectorInputPort(
            "plant_state", plant.num_multibody_states()
        )
        self.DeclarePeriodicPublishEvent(period, 0.0, self._record_pose)

        # For all models with DoFs, we store their model names.
        plant_context = plant.CreateDefaultContext()
        for i in range(plant.num_model_instances()):
            i = ModelInstanceIndex(i)
            model_x = self._plant.GetPositionsAndVelocities(plant_context, i)
            if len(model_x) > 0:
                model_name = plant.GetModelInstanceName(i)
                self._model_names.append(model_name)

    def _record_pose(self, context: Context):
        time = context.get_time()
        if time == 0:
            for i in range(self._plant.num_model_instances()):
                name = self._plant.GetModelInstanceName(ModelInstanceIndex(i))
                logging.info(f"model {i}: {name}")

        data = f"time: {time!r}\n"

        plant_x = self.EvalVectorInput(context, 0).get_value()
        plant_q = plant_x[0:self._plant.num_positions()]
        for model_name in self._model_names:
            data += f"{model_name}: "

            model_instance = self._plant.GetModelInstanceByName(model_name)
            joint_indices = self._plant.GetJointIndices(model_instance)

            # Construct the joint names in the same order as the joint indices.
            joint_names = ""
            model_q = []
            for joint_index in joint_indices:
                joint = self._plant.get_joint(joint_index)
                # Only log joints that contribute to the generalized position.
                if joint.num_positions() > 0:
                    joint_names += f"{joint.name()} "
                    joint_positions = plant_q[
                        joint.position_start():joint.position_start()
                        + joint.num_positions()
                    ]
                    model_q = np.append(model_q, joint_positions, axis=None)

            # Compare the number of positions collected in model_q to the
            # expected number of model positions from the plant, and ensure
            # they match.
            assert len(model_q) == self._plant.num_positions(model_instance)

            # Model instances made up of a single floating body are defined by
            # a single quaternion joint. Since we do not write floating joint
            # names for single floating bodies in the keyframes file, we
            # identify and exclude them here.
            # TODO(rcory) There is no reason to log single floating body models
            # any differently than articulated models. However, removing this
            # check changes the format of the keyframes file. This should be
            # addressed when we rewrite this file from scratch (see top
            # level TODO).
            is_floating_body = (len(joint_indices) == 1) and (
                self._plant.get_joint(joint_indices[0]).type_name()
                == "quaternion_floating"
            )
            if len(joint_indices) > 0 and not is_floating_body:
                data += f"{joint_names}: "

            data += " ".join([repr(float(q)) for q in model_q])
            data += "\n"

        self._file.write(data)
        self._file.flush()
