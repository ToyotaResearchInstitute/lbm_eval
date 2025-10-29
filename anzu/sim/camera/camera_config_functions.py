from pydrake.geometry import SceneGraph
from pydrake.lcm import DrakeLcmInterface
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import ApplyCameraConfig

from anzu.sim.camera.camera_config import AnzuCameraConfig
from anzu.sim.camera.fisheye_distortion import (
    apply_fisheye_distortion_to_plant,
)


def ApplyAnzuCameraConfig(
        config: AnzuCameraConfig,
        builder: DiagramBuilder,
        lcm_buses: dict | None = None,
        plant: MultibodyPlant | None = None,
        scene_graph: SceneGraph | None = None,
        lcm: DrakeLcmInterface | None = None,
):
    """Constructs a simulated camera sensor (rgbd sensor and publishing
    systems) within `builder`. If fisheye distortion configuration is present,
    a fisheye distortion system will be connected between the rgbd sensor and
    the publishing systems.

    Args:
        config: The camera configuration.
        builder: The diagram builder to add systems to.
        lcm_buses: (Optional) The available LCM buses to use for camera message
           publication. When not provided, uses the `lcm` interface if
           provided, or else the `config.lcm_bus` must be set to "default" in
           which case an appropriate pydrake.lcm.DrakeLcm object is constructed
           and used internally.
        plant: (Optional) The MultibodyPlant to use for kinematics. In the
            common case where a MultibodyPlant has already been added to
            `builder` using either AddMultibodyPlant() or
            AddMultibodyPlantSceneGraph(), the default value (None) here is
            suitable and generally should be preferred. When provided, it must
            be a System that's been added to the given `builder`. When not
            provided, uses the system named "plant" in the given `builder`.
        scene_graph: (Optional) The SceneGraph to use for rendering. In the
            common case where a SceneGraph has already been added to `builder`
            using either AddMultibodyPlant() or AddMultibodyPlantSceneGraph(),
            the default value (None) here is suitable and generally should
            be preferred. When provided, it must be a System that's been added
            to the given `builder`. When not provided, uses the system named
            "scene_graph" in the given `builder`.
        lcm: (Optional) The LCM interface used for visualization message
            publication. When not provided, uses the `config.lcm_bus` value to
            look up the appropriate interface from `lcm_buses`.

    Returns:
        None

    See also:
    - pydrake.multibody.plant.AddMultibodyPlant
    - pydrake.multibody.plant.AddMultibodyPlantSceneGraph
    - pydrake.systems.lcm.ApplyLcmBusConfig
    - pydrake.systems.sensors.ApplyCameraConfig
    - anzu.sim.camera.fisheye_distortion.apply_fisheye_distortion_to_plant

    """
    # Don't modify systems added outside of this function.
    prior_systems = builder.GetSystems()

    ApplyCameraConfig(
        config=config.drake,
        builder=builder,
        lcm_buses=lcm_buses,
        plant=plant,
        scene_graph=scene_graph,
        lcm=lcm,
    )
    if config.fisheye_distortion is not None:
        # Apply fisheye distortion system (if enabled). This only adds the
        # distortion LeafSystem with unconnected output ports.
        apply_fisheye_distortion_to_plant(config, builder)

        # Rewire the fisheye output into the lcm chain.
        # 1. Find the relevant systems.
        added_systems = set(builder.GetSystems()) - set(prior_systems)

        def find_system(name):
            if builder.HasSubsystemNamed(name):
                found = builder.GetSubsystemByName(name)
                assert found in added_systems
                return found
            return None

        rgbd_sensor = find_system(f"rgbd_sensor_{config.name}")
        fisheye = find_system(f"fisheye_distortion_{config.name}")
        image_to_lcm = find_system(f"image_to_lcm_{config.name}")

        # 2. Rewiring only makes sense if we have the lcm chain.
        if image_to_lcm is not None:
            # 2a. Rewire for each type of channel that is configured.
            for is_channel_configured, lcm_input, sensor_output in [
                    (config.depth, "depth", "depth_image_16u"),
                    (config.rgb, "rgb", "color_image"),
                    (config.label, "label", "label_image")]:
                if not is_channel_configured:
                    continue
                builder.Disconnect(
                    rgbd_sensor.GetOutputPort(sensor_output),
                    image_to_lcm.GetInputPort(lcm_input))
                builder.Connect(fisheye.GetOutputPort(sensor_output),
                                image_to_lcm.GetInputPort(lcm_input))
