import dataclasses


@dataclasses.dataclass(kw_only=True)
class DeformableSimConfig:
    """Configuration for the deformable model in a MultibodyPlant.

    NOTE:  This is applied after scenario sampling; no element of this
    config may affect collision or any other sampling predicate.
    """

    # The number of threads to use for the simulation of a single deformable
    # body.
    num_threads: int = 1
