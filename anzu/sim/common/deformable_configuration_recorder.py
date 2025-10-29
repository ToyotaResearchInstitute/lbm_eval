from pathlib import Path

import h5py
import numpy as np

from pydrake.common.value import Value
from pydrake.geometry import GeometryConfigurationVector
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.framework import Context, LeafSystem


class DeformableConfigurationRecorder(LeafSystem):
    """
    Logs the deformable object configuration into an h5 file.
    The h5 file stores a dictionary with keys
    {
      "time",
      str(deformable_geometry_id1),
      str(deformable_geometry_id2),
      ...,
      str(deformable_geometry_idN)
    }
    data["time"] stores a numpy array of simulation time when the data is
    logged.
    data[str(deformable_geometry_id1)] is a numpy array, recording the sequence
    of configuration for deformable geometry with that ID.
    data[str(deformable_geometry_id1)][t_index] is the configuration of the
    specified deformable geometry at time data["time"][t_index].
    """

    # The key used in the h5 file for time.
    TIME_KEY = "time"

    def __init__(self, plant: MultibodyPlant, filename: Path, period: float):
        """
        @pre `filename` must ends up with `.h5` as the file extension.
        """
        super().__init__()
        # Check filename. The file extension should be h5.
        if filename.suffix != ".h5":
            raise RuntimeError(f"{filename} must have extension .h5")
        self._plant = plant
        self.deformable_body_configuration_input_port = (
            self.DeclareAbstractInputPort(
                "deformable_body_configuration",
                Value(GeometryConfigurationVector()),
            )
        )
        self.DeclarePeriodicPublishEvent(
            period, offset_sec=0.0, publish=self.record_configuration
        )
        self._deformable_geometry_ids_to_body_scoped_names = {}
        deformable_model = self._plant.deformable_model()
        for model_instance in range(self._plant.num_model_instances()):
            body_ids = self._plant.deformable_model().GetBodyIds(
                ModelInstanceIndex(model_instance)
            )
            for body_id in body_ids:
                geometry_id = deformable_model.GetGeometryId(body_id)
                body_scoped_name = (
                    deformable_model.GetBody(body_id).scoped_name().get_full()
                )
                self._deformable_geometry_ids_to_body_scoped_names[
                    geometry_id
                ] = body_scoped_name
        # The deformable_body_scoped_name cannot be TIME_KEY.
        assert (
            DeformableConfigurationRecorder.TIME_KEY
            not in self._deformable_geometry_ids_to_body_scoped_names.values()
        ), (
            "deformable body scoped name cannot be "
            f"{DeformableConfigurationRecorder.TIME_KEY}"
        )

        self.filename = filename
        # Create the empty h5 file.
        with h5py.File(self.filename, "w") as f:
            f.create_dataset(
                DeformableConfigurationRecorder.TIME_KEY,
                data=np.array([]),
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            )
            dummy_plant_context = self._plant.CreateDefaultContext()
            for (
                geometry_id,
                body_scoped_name,
            ) in self._deformable_geometry_ids_to_body_scoped_names.items():
                body_id = deformable_model.GetBodyId(geometry_id)
                deformable_configuration = deformable_model.GetPositions(
                    dummy_plant_context, body_id
                )
                f.create_dataset(
                    body_scoped_name,
                    data=np.empty((0,) + deformable_configuration.shape),
                    maxshape=(None,) + deformable_configuration.shape,
                    chunks=True,
                    dtype=deformable_configuration.dtype,
                )

    def record_configuration(self, context: Context):
        t = context.get_time()
        geometry_configuration_vector = self.EvalAbstractInput(
            context, self.deformable_body_configuration_input_port.get_index()
        ).get_value()

        def get_deformable_configuration(geometry_id) -> np.array:
            deformable_configuration_flat = (
                geometry_configuration_vector.value(geometry_id)
            )
            # In C++ code, DeformableConfigurationVector.value() returns an
            # Eigen vector, which is the flattened version of the 3-by-N matrix
            # that stores the position of each vertex. Eigen matrix defaults to
            # column major, so this deformable_configuration_flat is the
            # concantenation of each column of the 3-by-N matrix. On the other
            # hand, python's numpy defaults to row-major. So to reconstruct
            # the 3-by-N matrix in numpy, we first reshape the flatten vector
            # to an N-by-3 matrix and then take its transpose.
            deformable_configuration = deformable_configuration_flat.reshape(
                (-1, 3)
            ).T
            return deformable_configuration

        log_data = {}
        log_data[DeformableConfigurationRecorder.TIME_KEY] = t
        for (
            geometry_id,
            body_scoped_name,
        ) in self._deformable_geometry_ids_to_body_scoped_names.items():
            log_data[body_scoped_name] = get_deformable_configuration(
                geometry_id
            )

        with h5py.File(self.filename, "a") as f:
            # Since the h5 data is created with chunks=True, and we are
            # resizing along the first dimension, the amortized cost of
            # resizing is constant time.
            for key, value in log_data.items():
                old_shape = f[key].shape
                new_shape = (old_shape[0] + 1,) + old_shape[1:]
                f[key].resize(new_shape)
                f[key][-1] = value
