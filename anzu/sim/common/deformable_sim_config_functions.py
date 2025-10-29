from pydrake.common import Parallelism
from pydrake.multibody.plant import MultibodyPlant

from anzu.sim.common.deformable_sim_config import DeformableSimConfig


def ApplyDeformableSimConfig(
    config: DeformableSimConfig, plant: MultibodyPlant
) -> None:
    """Applies the deformable body configuration to the given MultibodyPlant.
    """
    deformable_model = plant.deformable_model()
    # TODO(xuchenhan-tri): Find a more robust way to set the parallelism. See
    # https://github.com/RobotLocomotion/drake/issues/23106.
    deformable_model._set_parallelism(Parallelism(config.num_threads))
