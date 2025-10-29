import dataclasses


@dataclasses.dataclass(kw_only=True)
class ItemLockingMonitorConfig:
    # Simulation time at which item locking can first happen. Intended to allow
    # for settling of objects at the start of a simulation.
    t_start: float = 0.1

    # Scoped or non-scoped names for the geometry that triggers (un)locking.
    # Typically, this would be geometry on the EE, e.g., the fingers.
    unlock_near_geometry: list[str] = dataclasses.field(default_factory=list)

    # Distance threshold to trigger (un)locking. Free bodies must be at least
    # this far away from the `unlock_near_geometry` to be eligible for locking.
    # Must be non-negative.
    #
    # Note: Currently, deformable bodies are never locked. As a result, when
    # deformable bodies are present, this threshold should be set to a larger
    # value to prevent locking rigid bodies that are indirectly influenced by
    # the grippers through unlocked deformable bodies.
    #
    # For example, a gripper might push a deformable body, which then pushes
    # another deformable body, eventually affecting a rigid body. Even if the
    # rigid body appears far from the gripper, it may still be dynamically
    # influenced due to this chain of interactions. Consider such transitive
    # effects when choosing the threshold value in the presence of deformable
    # bodies.
    distance_threshold: float = 0.05

    # Translational speed threshold to trigger (un)locking. Free bodies must be
    # at least this slow translationally (in units of m/s) to be eligible for
    # locking. (Any rotational velocity is ignored.) Must be non-negative.
    speed_threshold: float = 0.002
