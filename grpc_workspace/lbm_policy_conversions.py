"""
grpc_lbm_policy_conversions library module.

Contained here are conversion functions to transition back and forth between
the observation action interface and gRPC message structures.
"""

import uuid

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
import yaml

from grpc_workspace.proto.GetPolicyMetadata_pb2 import (
    PolicyMetadata as PolicyMetadataMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    CameraImageSet as CameraImageSetMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    ImageCompression as ImageCompressionMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    ImageDtype,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    MultiarmObservation as MultiarmObservationMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    PosesAndGrippers as PosesAndGrippersMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    PosesAndGrippersActualAndDesired as PosesAndGrippersActualAndDesiredMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    RigidTransform as RigidTransformMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    RobotGripperStatus as RobotGripperStatusMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import (
    RobotPoseStatus as RobotPoseStatusMsg,
)
from grpc_workspace.proto.PolicyStep_pb2 import CameraImage as CameraImageMsg
from grpc_workspace.proto.PolicyStep_pb2 import CameraInfo as CameraInfoMsg
from grpc_workspace.proto.PolicyStep_pb2 import Header as HeaderMsg
from grpc_workspace.proto.PolicyStep_pb2 import Image as ImageMsg
from grpc_workspace.proto.PolicyStep_pb2 import Time as TimeMsg
from robot_gym.multiarm_spaces import (
    CameraDepthImage,
    CameraImage,
    CameraImageSet,
    CameraImageSetMap,
    CameraRgbImage,
    MultiarmObservation,
    PosesAndGrippers,
    PosesAndGrippersActualAndDesired,
)
from robot_gym.policy import PolicyMetadata

# TODO(imcmahon): These conversion functions should be done in some automated
# fashion rather than by hand here.


# These dictionaries have key-values are similar but not symmetrical. A string
# representing the type name is used as the key to look up ImageDtype.
# Normally, numpy dtypes would be used as keys for this dictionary, but it
# is has been observed that differing versions of numpy can cause the numpy
# types to hash to different values (e.g. between the imported numpy and a
# loaded picklefile). Therefore, using numpy types as a key is unreliable,
# and we use a name string lookup instead.
STR_TO_IMAGE_DTYPE = {
    "int8": ImageDtype.DTYPE_INT8,
    "int16": ImageDtype.DTYPE_INT16,
    "uint8": ImageDtype.DTYPE_UINT8,
    "uint16": ImageDtype.DTYPE_UINT16,
}


IMAGE_DTYPE_TO_NP_DTYPE = {
    ImageDtype.DTYPE_INT8: np.int8,
    ImageDtype.DTYPE_INT16: np.int16,
    ImageDtype.DTYPE_UINT8: np.uint8,
    ImageDtype.DTYPE_UINT16: np.uint16,
}


def grpc_msg_to_uuid(msg_data: str) -> uuid.UUID:
    """Create a uuid.UUID object from a gRPC string UUID message."""
    return uuid.UUID(msg_data)


def uuid_to_grpc_msg(source: uuid.UUID) -> str:
    """Create a gRPC string UUID message from a uuid.UUID object."""
    return str(source)


def image_msg_to_array(image: ImageMsg) -> np.array:
    """Create a numpy array from a gRPC Image message object."""
    dtype = IMAGE_DTYPE_TO_NP_DTYPE[image.dtype]
    image_array = np.frombuffer(image.data, dtype=dtype)
    return np.squeeze(
        image_array.reshape((image.height, image.width, image.channels))
    )


def possibly_empty_str_to_optional_str(
    possibly_empty_string: str,
) -> str | None:
    """Return the supplied string if it is not empty, otherwise return None."""
    return possibly_empty_string or None


def optional_str_to_str(optional_string: str | None) -> str:
    """Return the supplied string if not None or create an empty string."""
    return optional_string or str()


def policy_metadata_to_grpc_msg(source: PolicyMetadata) -> PolicyMetadataMsg:
    """Create a gRPC PolicyMetadata message from a PolicyMetadata object."""
    return PolicyMetadataMsg(
        name=source.name,
        skill_type=source.skill_type,
        checkpoint_path=source.checkpoint_path,
        git_repo=optional_str_to_str(source.git_repo),
        git_sha=optional_str_to_str(source.git_sha),
        is_language_conditioned=bool(source.is_language_conditioned),
        raw_policy_config=yaml.dump(source.raw_policy_config),
        runtime_information=source.runtime_information,
    )


def grpc_msg_to_policy_metadata(msg_data: PolicyMetadataMsg) -> PolicyMetadata:
    """Create a PolicyMetadata object from a gRPC PolicyMetadata message."""
    return PolicyMetadata(
        name=msg_data.name,
        skill_type=msg_data.skill_type,
        checkpoint_path=msg_data.checkpoint_path,
        git_repo=possibly_empty_str_to_optional_str(msg_data.git_repo),
        git_sha=possibly_empty_str_to_optional_str(msg_data.git_sha),
        is_language_conditioned=msg_data.is_language_conditioned,
        raw_policy_config=yaml.safe_load(msg_data.raw_policy_config),
        runtime_information=dict(msg_data.runtime_information),
    )


def grpc_msg_to_timestamp(msg_data: TimeMsg) -> float:
    """
    Creates a float timestamp from a gRPC Time message. If the seconds (sec)
    member variable is set to -1, this will return None instead.
    """
    if msg_data.sec != -1:
        return msg_data.sec + float(msg_data.nanosec) / 10**9
    else:
        return None


def timestamp_to_grpc_msg(timestamp: float) -> TimeMsg:
    """
    Creates a gRPC Time message from a float timestamp, which will use
    seconds (sec) equal to -1 to represent `None`.
    """
    # Note: The Time.to_msg() conversion does not allow for any negative
    # value in the seconds field, but negative values for the sec field
    # in Time.mgs is valid. See the rcl_interfaces/builtin_interfaces
    # repo for details:
    # https://github.com/ros2/rcl_interfaces/blob/humble/builtin_interfaces/msg/Time.msg#L4-L5  # noqa
    if timestamp is not None and timestamp >= 0:

        # Convert timestamp (seconds) to nanoseconds:
        # See here:
        # https://github.com/ros2/rclpy/blob/7.5.0/rclpy/rclpy/time.py#L156-L163  # noqa
        conversion_constant = 10**9
        nanoseconds = timestamp * conversion_constant
        seconds = nanoseconds // conversion_constant
        nanoseconds = nanoseconds % conversion_constant

        return TimeMsg(sec=int(seconds), nanosec=int(nanoseconds))
    else:
        return TimeMsg(sec=-1)


def np_shaped_3x3_array_to_flat_array(array: np.array) -> np.array:
    """Create a flat 1x9 numpy array from a row-major, 3x3 numpy array."""
    if array.shape != (3, 3):
        raise AssertionError(
            "To convert a 3x3 numpy array to flat array, the supplied array "
            "must be a two-dimensional, nine-element, 3x3 matrix-array: "
            f"{array.shape}"
        )
    return array.reshape((9,))


def np_flat_array_to_shaped_3x3_array(array: np.array) -> np.array:
    """Create a row-major, 3x3 numpy array from a flat 1x9 numpy array."""
    if array.shape != (9,):
        raise AssertionError(
            "To convert a flat array to 3x3 numpy array, the supplied array "
            "must be a one-dimensional, nine-element array: "
            f"{array.shape}"
        )
    return array.reshape((3, 3))


def rotation_matrix_to_array(rotation_matrix: RotationMatrix) -> np.array:
    """
    Create a row-major one-dimensional array from a 3x3 Drake RotationMatrix.

    We use a 3x3 rotation matrix to avoid numeric precision loss when
    converting from rotation matrix to a quaternion, so that way
    inference via remote policy should not be different than direct
    inference, aside from real-time latency issues.

    This row-major storage order for matches those used matrices in other gRPC
    messages, such as sensor_msgs/CameraInfo.
    """
    rotation_array = np.array(
        [
            rotation_matrix.row(0),
            rotation_matrix.row(1),
            rotation_matrix.row(2),
        ]
    )
    return np_shaped_3x3_array_to_flat_array(rotation_array)


def array_to_rotation_matrix(array: np.array) -> RotationMatrix:
    """
    Create a Drake RotationMatrix from a 1x9 numpy array.

    The input one-dimensional, nine-element array represents a 3x3
    row-major rotation matrix.
    """
    if array.shape != (9,):
        raise AssertionError(
            "To convert to a RotationMatrix, supplied array must be a "
            f"one-dimentional, nine-element array: {array.shape}"
        )
    return RotationMatrix(
        [
            array[0:3].tolist(),
            array[3:6].tolist(),
            array[6:9].tolist(),
        ]
    )


def rigid_transform_to_grpc_msg(
    source: RigidTransform,
) -> RigidTransformMsg:
    """Create a gRPC RigidTransform message from a Drake RigidTransform."""
    return RigidTransformMsg(
        rotation=rotation_matrix_to_array(source.rotation()),
        translation=source.translation(),
    )


def grpc_msg_to_rigid_transform(msg_data: RigidTransformMsg) -> RigidTransform:
    """Create a Drake RigidTransform from a gRPC RigidTransform message."""
    return RigidTransform(
        R=array_to_rotation_matrix(np.array(msg_data.rotation)),
        p=np.array(msg_data.translation),
    )


def grpc_msg_to_poses_and_grippers(
    msg_data: PosesAndGrippersMsg,
) -> PosesAndGrippers:
    """Create a PosesAndGrippers object from a grpc PosesAndGrippers
    message.
    """
    poses_dict = dict()
    grippers_dict = dict()
    joint_position_dict = dict()
    joint_velocity_dict = dict()
    joint_torque_dict = dict()
    joint_torque_external_dict = dict()
    wrench_dict = dict()
    external_wrench_dict = dict()

    # The PosesAndGrippers dataclass requires that "poses" be populated,
    # but all other robot fields are optional and should be set to "None"
    # if they are empty.
    for pose_status in msg_data.pose_status:
        model = pose_status.robot_name
        poses_dict[model] = grpc_msg_to_rigid_transform(pose_status.pose)
        if len(pose_status.joint_position):
            joint_position_dict[model] = np.asarray(pose_status.joint_position)
        if len(pose_status.joint_velocity):
            joint_velocity_dict[model] = np.asarray(pose_status.joint_velocity)
        if len(pose_status.joint_torque):
            joint_torque_dict[model] = np.asarray(pose_status.joint_torque)
        if len(pose_status.joint_torque_external):
            joint_torque_external_dict[model] = np.asarray(
                pose_status.joint_torque_external
            )
        if len(pose_status.wrench):
            wrench_dict[model] = np.asarray(pose_status.wrench)
        if len(pose_status.external_wrench):
            external_wrench_dict[model] = np.asarray(
                pose_status.external_wrench
            )

    # Overwrite the dictionaries with None if empty
    def none_if_empty(data):
        return data if data else None

    joint_position_dict = none_if_empty(joint_position_dict)
    joint_velocity_dict = none_if_empty(joint_velocity_dict)
    joint_torque_dict = none_if_empty(joint_torque_dict)
    joint_torque_external_dict = none_if_empty(joint_torque_external_dict)
    wrench_dict = none_if_empty(wrench_dict)
    external_wrench_dict = none_if_empty(external_wrench_dict)

    # The PosesAndGrippers dataclass requires that "grippers" be populated.
    for gripper_status in msg_data.gripper_status:
        grippers_dict[gripper_status.gripper_name] = (
            gripper_status.gripper_position
        )

    timestamp_data = grpc_msg_to_timestamp(msg_data.timestamp_data)
    timestamp_sent = grpc_msg_to_timestamp(msg_data.timestamp_sent)
    timestamp_received = grpc_msg_to_timestamp(msg_data.timestamp_received)

    return PosesAndGrippers(
        poses=poses_dict,
        grippers=grippers_dict,
        joint_position=joint_position_dict,
        joint_velocity=joint_velocity_dict,
        joint_torque=joint_torque_dict,
        joint_torque_external=joint_torque_external_dict,
        wrench=wrench_dict,
        external_wrench=external_wrench_dict,
        timestamp_data=timestamp_data,
        timestamp_sent=timestamp_sent,
        timestamp_received=timestamp_received,
    )


def poses_and_grippers_to_grpc_msg(
    source: PosesAndGrippers,
) -> PosesAndGrippersMsg:
    """
    Create a grpc PosesAndGrippers message from a PosesAndGrippers object.

    The PosesAndGrippers grpc message requires that the optional list fields,
    which can be None in the PosesAndGripper dataclass, be set to an empty
    list if they are None for transport over grpc.
    """
    robot_pose_status_msg_list = list()
    value_or_empty_list = lambda data, key: (
        data[key] if data and (key in data) else list()
    )
    for robot_name in source.poses.keys():
        robot_pose_status_msg = RobotPoseStatusMsg(
            robot_name=robot_name,
            pose=rigid_transform_to_grpc_msg(source.poses[robot_name]),
            wrench=value_or_empty_list(
                source.wrench,
                robot_name,
            ),
            external_wrench=value_or_empty_list(
                source.external_wrench,
                robot_name,
            ),
            joint_position=value_or_empty_list(
                source.joint_position,
                robot_name,
            ),
            joint_velocity=value_or_empty_list(
                source.joint_velocity,
                robot_name,
            ),
            joint_torque=value_or_empty_list(
                source.joint_torque,
                robot_name,
            ),
            joint_torque_external=value_or_empty_list(
                source.joint_torque_external,
                robot_name,
            ),
        )
        robot_pose_status_msg_list.append(robot_pose_status_msg)

    robot_gripper_status_msg_list = list()
    for gripper_name in source.grippers.keys():
        robot_gripper_status_msg = RobotGripperStatusMsg(
            gripper_name=gripper_name,
            gripper_position=float(source.grippers[gripper_name]),
        )
        robot_gripper_status_msg_list.append(robot_gripper_status_msg)

    return PosesAndGrippersMsg(
        pose_status=robot_pose_status_msg_list,
        gripper_status=robot_gripper_status_msg_list,
        timestamp_data=timestamp_to_grpc_msg(source.timestamp_data),
        timestamp_sent=timestamp_to_grpc_msg(source.timestamp_sent),
        timestamp_received=timestamp_to_grpc_msg(source.timestamp_received),
    )


def grpc_msg_to_poses_and_grippers_actual_and_desired(
    robot: PosesAndGrippersActualAndDesiredMsg,
) -> PosesAndGrippersActualAndDesired:
    """Create a PosesAndGrippersActualAndDesired object from a grpc message."""
    return PosesAndGrippersActualAndDesired(
        actual=grpc_msg_to_poses_and_grippers(robot.actual),
        desired=grpc_msg_to_poses_and_grippers(robot.desired),
        version=float(robot.version),
    )


def grpc_msg_to_camera_image_set_map(
    visuo: list[CameraImageSetMsg],
) -> CameraImageSetMap:
    """Create CameraImageSetMap object from grpc CameraImageSet message
    list.
    """
    camera_image_set_map = dict()

    # This will iterate over a list of CameraImageSet. Within each
    # CameraImageSet is a CameraImage for each of rgb, depth, label. Within
    # each CameraImage is an Image, CameraInfo and RigidTransform.
    for camera_image_set_msg in visuo:
        rgb = CameraRgbImage(
            array=image_msg_to_array(camera_image_set_msg.camera_rgb.image),
            K=np_flat_array_to_shaped_3x3_array(
                np.array(camera_image_set_msg.camera_rgb.info.k),
            ),
            X_TC=grpc_msg_to_rigid_transform(
                camera_image_set_msg.camera_rgb.pose,
            ),
            timestamp=grpc_msg_to_timestamp(
                camera_image_set_msg.camera_rgb.info.header.stamp,
            ),
        )

        depth = None
        if camera_image_set_msg.has_depth:
            depth = CameraDepthImage(
                array=image_msg_to_array(
                    camera_image_set_msg.camera_depth.image
                ),
                K=np_flat_array_to_shaped_3x3_array(
                    np.array(camera_image_set_msg.camera_depth.info.k),
                ),
                X_TC=grpc_msg_to_rigid_transform(
                    camera_image_set_msg.camera_depth.pose,
                ),
                timestamp=grpc_msg_to_timestamp(
                    camera_image_set_msg.camera_depth.info.header.stamp,
                ),
            )

        label = None
        if camera_image_set_msg.has_label:
            # NOTE(charlie): This ought to be a CameraLabelImage, but the
            # observation we get in grpc_service_policy is of type CameraImage.
            label = CameraImage(
                array=image_msg_to_array(
                    camera_image_set_msg.camera_label.image
                ),
                K=np_flat_array_to_shaped_3x3_array(
                    np.array(camera_image_set_msg.camera_label.info.k),
                ),
                X_TC=grpc_msg_to_rigid_transform(
                    camera_image_set_msg.camera_label.pose,
                ),
                # NOTE(charlie): If we change CameraImage to CameraLabelImage,
                # we would have to reinstate the timestamp field.
                # timestamp=grpc_msg_to_timestamp(
                #     camera_image_set_msg.camera_label.info.header.stamp,
                # ),
            )

        if isinstance(camera_image_set_msg.camera_serial, str):
            camera_serial = camera_image_set_msg.camera_serial
        else:
            camera_serial = (camera_image_set_msg.camera_serial,)
        camera_image_set_map[camera_serial] = CameraImageSet(
            rgb=rgb,
            depth=depth,
            label=label,
        )

    return camera_image_set_map


def poses_and_grippers_actual_and_desired_to_grpc_msg(
    robot: PosesAndGrippersActualAndDesired,
) -> PosesAndGrippersActualAndDesiredMsg:
    """
    Creates a PosesAndGrippersActualAndDesired gRPC message from data
    in the multiarm_spaces' PosesAndGrippersActualAndDesired class.
    """
    return PosesAndGrippersActualAndDesiredMsg(
        actual=poses_and_grippers_to_grpc_msg(robot.actual),
        desired=poses_and_grippers_to_grpc_msg(robot.desired),
        version=float(robot.version) if robot.version else 0.0,
    )


def policy_observation_to_grpc_msg(
    obs: MultiarmObservation,
) -> MultiarmObservationMsg:
    """Create a grpc message from data in the MultiarmObservation object."""
    if obs.language_instruction is not None:
        use_language_instruction = True
        language_instruction = obs.language_instruction
    else:
        use_language_instruction = False
        language_instruction = str()
    return MultiarmObservationMsg(
        robot=poses_and_grippers_actual_and_desired_to_grpc_msg(obs.robot),
        visuo=camera_image_set_map_to_grpc_msg(obs.visuo),
        use_language_instruction=use_language_instruction,
        language_instruction=language_instruction,
    )


def grpc_msg_to_policy_observation(
    obs: MultiarmObservationMsg,
) -> MultiarmObservation:
    """Create an MultiarmObservation object from MultiarmObservation grpc
    message.
    """
    if obs.use_language_instruction:
        language_instruction = obs.language_instruction
    else:
        language_instruction = None
    return MultiarmObservation(
        robot=grpc_msg_to_poses_and_grippers_actual_and_desired(obs.robot),
        visuo=grpc_msg_to_camera_image_set_map(obs.visuo),
        language_instruction=language_instruction,
    )


def camera_image_set_map_to_grpc_msg(
    visuo: CameraImageSetMap,
) -> list[CameraImageSetMsg]:
    """
    Creates a list of gRPC CameraImageSetMsg messages from data
    in the supplied CameraImageSetMap dictionary.
    """
    camera_serial_msg_list = list()
    if visuo:
        camera_serial_msg_list.extend(visuo.keys())

    camera_image_set_msg_list = list()
    for camera_serial in camera_serial_msg_list:

        image_rgb = visuo[camera_serial].rgb.array

        if image_rgb.ndim != 3:
            raise ValueError(
                f"Expected RGB image to have 3 dims, got {image_rgb.ndim}"
            )

        # TODO(charlie): Make a function to populate each of the messages
        # defined in the proto file.
        has_rgb = True
        camera_rgb = CameraImageMsg(
            image=ImageMsg(
                height=image_rgb.shape[0],
                width=image_rgb.shape[1],
                channels=image_rgb.shape[2],
                data=image_rgb.tobytes(),
                dtype=STR_TO_IMAGE_DTYPE[image_rgb.dtype.name],
                compression=ImageCompressionMsg.NONE,
            ),
            info=CameraInfoMsg(
                height=image_rgb.shape[0],
                width=image_rgb.shape[1],
                distortion_model="plumb_bob",
                k=np_shaped_3x3_array_to_flat_array(
                    visuo[camera_serial].rgb.K
                ),
                header=HeaderMsg(
                    stamp=timestamp_to_grpc_msg(
                        getattr(visuo[camera_serial].rgb, "timestamp", None),
                    )
                ),
            ),
            pose=rigid_transform_to_grpc_msg(visuo[camera_serial].rgb.X_TC),
        )

        has_depth = False
        camera_depth = CameraImageMsg()
        if visuo[camera_serial].depth:
            has_depth = True
            image_depth = visuo[camera_serial].depth.array

            if image_depth.ndim != 2:
                raise ValueError(
                    f"Expected depth image to have 2 dims, got "
                    f"{image_depth.ndim}"
                )

            camera_depth = CameraImageMsg(
                image=ImageMsg(
                    height=image_depth.shape[0],
                    width=image_depth.shape[1],
                    channels=1,
                    data=image_depth.tobytes(),
                    dtype=STR_TO_IMAGE_DTYPE[image_depth.dtype.name],
                    compression=ImageCompressionMsg.NONE,
                ),
                info=CameraInfoMsg(
                    height=image_depth.shape[0],
                    width=image_depth.shape[1],
                    distortion_model="plumb_bob",
                    k=np_shaped_3x3_array_to_flat_array(
                        visuo[camera_serial].depth.K
                    ),
                    header=HeaderMsg(
                        stamp=timestamp_to_grpc_msg(
                            getattr(
                                visuo[camera_serial].depth, "timestamp", None
                            ),
                        ),
                    ),
                ),
                pose=rigid_transform_to_grpc_msg(
                    visuo[camera_serial].depth.X_TC
                ),
            )

        has_label = False
        camera_label = CameraImageMsg()
        if visuo[camera_serial].label:
            has_label = True
            image_label = visuo[camera_serial].label.array

            if image_label.ndim != 2:
                raise ValueError(
                    f"Expected label image to have 2 dims, got "
                    f"{image_label.ndim}"
                )

            camera_label = CameraImageMsg(
                image=ImageMsg(
                    height=image_label.shape[0],
                    width=image_label.shape[1],
                    channels=1,
                    data=image_label.tobytes(),
                    dtype=STR_TO_IMAGE_DTYPE[image_label.dtype.name],
                    compression=ImageCompressionMsg.NONE,
                ),
                info=CameraInfoMsg(
                    height=image_label.shape[0],
                    width=image_label.shape[1],
                    distortion_model="plumb_bob",
                    k=np_shaped_3x3_array_to_flat_array(
                        visuo[camera_serial].label.K
                    ),
                    header=HeaderMsg(
                        stamp=timestamp_to_grpc_msg(
                            getattr(
                                visuo[camera_serial].label, "timestamp", None
                            ),
                        ),
                    ),
                ),
                pose=rigid_transform_to_grpc_msg(
                    visuo[camera_serial].label.X_TC
                ),
            )

        camera_image_set_msg = CameraImageSetMsg(
            camera_serial=camera_serial,
            has_rgb=has_rgb,
            has_depth=has_depth,
            has_label=has_label,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth,
            camera_label=camera_label,
        )

        camera_image_set_msg_list.append(camera_image_set_msg)

    return camera_image_set_msg_list
