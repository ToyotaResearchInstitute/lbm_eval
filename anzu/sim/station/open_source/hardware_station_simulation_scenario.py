# WARNING: This file is auto-generated; see README.md.
import dataclasses
import math

import numpy as np

from pydrake.geometry import SceneGraphConfig
from pydrake.lcm import DrakeLcmParams
from pydrake.manipulation import IiwaDriver, SchunkWsgDriver, ZeroForceDriver
from pydrake.multibody.parsing import ModelDirective
from pydrake.multibody.plant import MultibodyPlantConfig
from pydrake.systems.analysis import SimulatorConfig
from pydrake.visualization import VisualizationConfig

from anzu.sim.camera.camera_config import AnzuCameraConfig
from anzu.sim.common.deformable_sim_config import DeformableSimConfig
from anzu.sim.common.html_recording import HtmlRecording
from anzu.sim.common.initialization_body_config import InitializationBodyConfig
from anzu.sim.common.item_locking_monitor_config import (
    ItemLockingMonitorConfig,
)
from anzu.sim.common.logging_config_functions import LoggingConfig
from anzu.sim.common.scenario_constraints_config import ScenarioConstraints
from anzu.sim.common.schunk_wsg.schunk_wsg_implicit_pd_driver_config import (
    SchunkWsgImplicitPdDriver,
)


@dataclasses.dataclass(kw_only=True)
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.
    """

    # Random seed for any random elements in the scenario.
    #
    # Note: The seed is always deterministic in the `Scenario`; a caller who
    # wants randomness must populate this value from their own randomness.
    random_seed: int = 0

    # The maximum simulation time (in seconds).  The simulator will attempt to
    # run until this time and then terminate.
    simulation_duration: float = math.inf

    # All of the fully deterministic elements of the simulation.
    directives: list[ModelDirective] = dataclasses.field(default_factory=list)

    # Item joint locking monitor config. When enabled items that are far from
    # the fingers of the robot have their joints locked.
    item_locking: ItemLockingMonitorConfig | None = None

    # A map-of-maps {model_instance_name: {joint_name: np.ndarray}} that
    # defines the initial state of some joints in the scene. Joints not
    # mentioned will remain in their default configurations.
    #
    # Note: Positions specified here override any position configuration in the
    # model directives and files, including randomized positions.
    initial_position: dict[
        str,  # model_instance_name ->
        dict[str, np.ndarray],  # joint_name -> positions
    ] = dataclasses.field(default_factory=dict)

    # A map of {bus_name: lcm_params} for LCM transceivers to be used by
    # drivers, sensors, etc.
    lcm_buses: dict[str, DrakeLcmParams] = dataclasses.field(
        default_factory=lambda: dict(default=DrakeLcmParams()),
    )

    # Cameras to add to the scene (and optionally broadcast over LCM or ROS).
    cameras: list[AnzuCameraConfig] = dataclasses.field(default_factory=list)

    # Apply a driver stack (potentially publishing and subscribing to LCM
    # channels) to a named model.
    model_drivers: dict[
        str,
        # This one should remain first (i.e., as the default); the rest are in
        # alphabetical order.
        ZeroForceDriver
        | IiwaDriver
        | SchunkWsgDriver
        | SchunkWsgImplicitPdDriver,
    ] = dataclasses.field(default_factory=dict)

    # Multibody plant configuration (timestep and contact parameters).
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig()

    # Drake simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig(
        integration_scheme="runge_kutta3",
        max_step_size=1e-3,
        accuracy=1.0e-2,
        use_error_control=True,
        target_realtime_rate=1.0,
    )

    # Additional simulator runtime configuration for deformable objects.
    deformable_sim_config: DeformableSimConfig | None = None

    # Configuration default proximity properties, as a fallback for models that
    # don't have any.
    scene_graph_config: SceneGraphConfig = SceneGraphConfig()

    # Visualization publisher(s).
    visualization: VisualizationConfig = VisualizationConfig()

    # Configuration for logging.
    logging: LoggingConfig = dataclasses.field(
        default_factory=lambda: LoggingConfig(
            logs=[
                HtmlRecording(),
            ],
        ),
    )

    # Requirements that a scenario must meet in order to be considered valid.
    preconditions: ScenarioConstraints = dataclasses.field(
        default_factory=ScenarioConstraints,
    )

    # We sometimes add transient bodies to the scene using the `directives`
    # stanza to help with sampling which are effectively removed shortly after
    # the simulation starts. This specifies which models are removed and when.
    initialization_bodies: InitializationBodyConfig | None = None
