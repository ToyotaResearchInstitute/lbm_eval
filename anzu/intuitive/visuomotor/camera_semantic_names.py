import logging

from pydrake.common.yaml import yaml_load

from anzu.common.path_util import resolve_path
from anzu.intuitive.station_defines import ARBITRARY_STATION_NAME


def _strip_prefix(prefix, s):
    assert s.startswith(prefix)
    return s[len(prefix) :]


def _is_image_sensor(camera_type):
    camera_type_name = _strip_prefix("!", camera_type["_tag"])

    image_sensor_types = [
        "SimCamera",
        "IntelRealsense",
        "FramosRealsense",
        "BgrLcmRealsense",
        "K4ASensor",
        "RosRealsenseSensor",
        "RosBgrSensor",
    ]

    return camera_type_name in image_sensor_types


def load_camera_id_to_semantic_name(station_name, image_only=True):
    """
    Load mapping from
    package://anzu/intuitive/visuomotor/demo/hardware.camera_frame_mappings.yaml
    """  # noqa
    all_mappings_file = resolve_path(
        "package://anzu/intuitive/visuomotor/demo/"
        "hardware.camera_frame_mappings.yaml"
    )
    all_mappings = yaml_load(filename=all_mappings_file)
    mappings = all_mappings["frame_mappings"][station_name]["frame_mapping"]
    camera_id_to_semantic_name = {}
    for mapping in mappings:
        camera_name = _strip_prefix("camera_", mapping["hardware_frame_name"])
        semantic_name = mapping["readable_frame_name"]
        camera_type = mapping["camera_type"]
        if _is_image_sensor(camera_type) or image_only is False:
            camera_id_to_semantic_name[camera_name] = semantic_name
    return camera_id_to_semantic_name


def resolve_camera_id_to_semantic_name(
    demonstration_info,
    station_name,
):
    """
    Attempts to extra camera semantic naming from the following (in order):
    - If data is present in `demonstration_info`, use this.
    - Otherwise, if the station name is not ARBITRARY_STATION_NAME, load from
      disk using `load_camera_id_to_semantic_name`.
    """
    # First, try to load from episode metadata.
    camera_id_to_semantic_name = demonstration_info.camera_id_to_semantic_name
    if camera_id_to_semantic_name is not None:
        logging.info("Using mapping stored in pkl file")
        return camera_id_to_semantic_name

    # Next, try to load from station name.
    if station_name == ARBITRARY_STATION_NAME:
        logging.info("Arbitrary station name, no semantic mapping")
        return None
    else:
        logging.info(f"Loading mapping from current file")
        return load_camera_id_to_semantic_name(station_name)


def rename_camera_image_set(mapping, image_set_map, *, invert=False):
    """
    Nominally, will take camera id to semantic name in `mapping`, and change
    `image_set_map` to be keyed by semantic name instead of camera id.

    If `invert` is true, will do the inverse.
    """
    if invert:
        inverted = {v: k for k, v in mapping.items()}
        assert len(inverted) == len(mapping)
        mapping = inverted

    new_image_set_map = {}
    for camera_name, image_set in image_set_map.items():
        new_name = mapping[camera_name]
        new_image_set_map[new_name] = image_set

    return new_image_set_map


def partition_camera_ids_based_on_semantic_names(camera_id_to_semantic_name):
    """
    Splits cameras into scene and wrist cameras.
    """
    scene_cameras = []
    moving_cameras = []
    # TODO(aditya.bhat): Add proper handling for cameras mounted on head,
    # chest, etc (anything that is not a scene/wrist camera).
    for camera_name, semantic_name in camera_id_to_semantic_name.items():
        if "wrist" in semantic_name:
            moving_cameras.append(camera_name)
        elif "scene" in semantic_name:
            scene_cameras.append(camera_name)
        elif "head" in semantic_name:
            scene_cameras.append(camera_name)
        elif "lidar" in semantic_name:
            pass
        else:
            print(f"Can't identify camera {semantic_name}")
            assert False
    return scene_cameras, moving_cameras
