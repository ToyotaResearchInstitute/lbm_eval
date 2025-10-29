import copy
import dataclasses as dc

import numpy as np
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from robot_gym.multiarm_spaces import (
    CURRENT_VERSION,
    CameraDepthImage,
    CameraImageSet,
    CameraLabelImage,
    CameraRgbImage,
    MultiarmObservation,
    PosesAndGrippers,
    PosesAndGrippersActualAndDesired,
    RestorePosesAndGrippersConfig,
)


def _quaternion_shortest_path(q_A, q_B):
    # Returns q_B s.t. q_A and q_B have minimum dot product.
    assert isinstance(q_A, Quaternion)
    assert isinstance(q_B, Quaternion)
    if np.dot(q_A.wxyz(), q_B.wxyz()) < 0:
        return Quaternion(-q_B.wxyz())
    else:
        return q_B


def se3_interp(s, X_AB_start, X_AB_end):
    p_TP_start = X_AB_start.translation()
    p_TP_end = X_AB_end.translation()
    p_TP = p_TP_start + s * (p_TP_end - p_TP_start)
    q_TP_start = X_AB_start.rotation().ToQuaternion()
    q_TP_end = X_AB_end.rotation().ToQuaternion()
    q_TP_end = _quaternion_shortest_path(q_TP_start, q_TP_end)
    q_TP = q_TP_start.slerp(s, q_TP_end)
    X_AB = RigidTransform(q_TP, p_TP)
    return X_AB


def _interp_vec(s, x1, x2):
    x = x1 + s * (x2 - x1)
    return x


def _assert_nominal_action(action):
    assert action.joint_position is None
    assert action.joint_velocity is None
    assert action.joint_torque is None
    assert action.joint_torque_external is None
    assert action.wrench is None
    assert action.external_wrench is None


def interp_action(s, action_1, action_2):
    # TODO(eric.cousineau): Enable these checks.
    # assert _assert_nominal_action(action_1)
    # assert _assert_nominal_action(action_2)
    action = copy.deepcopy(action_1)
    for key in action.poses.keys():
        action.poses[key] = se3_interp(
            s, action_1.poses[key], action_2.poses[key]
        )
    for key in action.grippers.keys():
        action.grippers[key] = _interp_vec(
            s, action_1.grippers[key], action_2.grippers[key]
        )
    return action


def action_get_stop_info(action):
    debug = action.debugging_output
    if debug is None:
        return None
    stop_info = {}
    is_success = debug.get("is_success", None)
    if is_success is not None:
        stop_info["is_success"] = is_success
    is_retry = debug.get("is_retry", None)
    if is_retry is not None:
        stop_info["is_retry"] = is_retry
    stop_reason = debug.get("stop_reason", None)
    if stop_reason is not None:
        stop_info["stop_reason"] = stop_reason
    return stop_info


def _normalize(x, *, tol=1e-10):
    x = np.asarray(x)
    n = np.linalg.norm(x)
    assert n >= tol
    return x / n


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    NumPy version of diffusion_policy rotation_6d_to_matrix.
    """
    # Ensure we use high precision.
    d6 = d6.astype(np.float64)
    a1, a2 = d6[:3], d6[3:]
    b1 = _normalize(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3))


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    NumPy version of diffusion_policy matrix_to_rotation_6d.
    """
    return matrix[:2, :].copy().reshape((6,))


def _rigid_transform_to_xyz_rot_6d(X):
    """
    Converts rigid transform to dictionary of {"xyz": ..., "rot_6d": ...}

    See matrix_to_rotation_6d for more info about "rot_6d".
    """
    assert isinstance(X, RigidTransform)
    xyz = X.translation()
    rot_6d = matrix_to_rotation_6d(X.rotation().matrix())
    return {
        "xyz": xyz,
        "rot_6d": rot_6d,
    }


def _to_np_value(v):
    if isinstance(v, RigidTransform):
        return _rigid_transform_to_xyz_rot_6d(v)
    else:
        v = np.asarray(v)
        if v.shape == ():
            v = v.reshape(1)
        return v


def _flatten_dict(config, prefix="", delim="."):
    """
    Transforms a dict of the form:

        {"top": {"mid": {"bottom": 0.25}}}

    to the following form:

        {"top.mid.bottom": 0.25}
    """
    assert isinstance(config, dict)
    out = dict()
    for k, v in config.items():
        assert isinstance(k, str), repr(k)
        assert "." not in k, repr(k)
        if isinstance(v, dict):
            v = _flatten_dict(v, f"{prefix}{k}{delim}", delim=delim)
            for ki, vi in v.items():
                assert ki not in config
            out.update(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def _dict_to_flat_pure_np_dict(x):
    """
    Takes a nested dictionary `x` and ensures that all values contained within
    it are purely NumPy values with the correct coordinate representation.
    Additionally flattens the dictionary, e.g.
        {"pose": {"xyz": ..., "rot_6d": ...}}
    becomes
        {"pose__xyz": ..., "pose__rot_6d": ...}
    """
    keys_prev = list(x.keys())
    while True:
        x = _flatten_dict(x, delim="__")
        # Remove any values that are `None`.
        x = {k: v for k, v in x.items() if v is not None}
        # Now sort through.
        for k, v in x.items():
            x[k] = _to_np_value(v)
        keys = list(x.keys())
        if keys == keys_prev:
            break
        else:
            keys_prev = keys
    return x


def _flatten_np_dict_to_vector(x):
    x = np.concatenate([np.asarray(x).reshape(-1) for x in x.values()])
    return x


def multiarm_action_to_dp_vector(action: PosesAndGrippers) -> np.ndarray:
    raw_dict = {}
    if action.poses is not None:
        raw_dict["poses"] = action.poses
    if action.grippers is not None:
        raw_dict["grippers"] = action.grippers
    assert len(raw_dict) > 0
    flat_dict = _dict_to_flat_pure_np_dict(raw_dict)
    vector = _flatten_np_dict_to_vector(flat_dict)
    return vector


def dp_vector_to_multiarm_action(
    config: RestorePosesAndGrippersConfig, vector: np.ndarray
) -> PosesAndGrippers:
    index = 0
    # Poses.
    poses = {}
    for model_name in config.model_names:
        xyz = vector[index : index + 3]
        index += 3
        rot_6d = vector[index : index + 6]
        index += 6
        rot = rotation_6d_to_matrix(rot_6d)
        pose = RigidTransform(R=RotationMatrix(rot), p=xyz)
        poses[model_name] = pose
    # Grippers.
    grippers = {}
    for gripper_name in config.gripper_names:
        gripper_width = vector[index].item()
        index += 1
        grippers[gripper_name] = gripper_width
    # Done.
    assert index == len(vector)
    return PosesAndGrippers(poses=poses, grippers=grippers)


def multiarm_observation_to_dp_dict(
    obs: MultiarmObservation,
    *,
    rgb_only=False,
):
    """
    Converts MultiarmObservation to a dictionary for inference.
    """
    raw_dict = {}
    # Camera images.
    for camera_name, image_set in obs.visuo.items():
        if image_set.rgb is not None:
            raw_dict[camera_name] = image_set.rgb.array
        if image_set.depth is not None:
            raw_dict[f"{camera_name}_depth"] = image_set.depth.array
        if image_set.label is not None:
            raw_dict[f"{camera_name}_label"] = image_set.label.array
    # Proprioception.
    robot_dict = {"robot": dc.asdict(obs.robot)}
    robot_dict_flat = _dict_to_flat_pure_np_dict(robot_dict)
    # Root-level.
    if obs.timestamp_packaged is not None:
        robot_dict_flat["timestamp_packaged"] = _to_np_value(
            obs.timestamp_packaged
        )
    # Language.
    if obs.language_instruction is not None:
        raw_dict["language_instruction"] = obs.language_instruction

    key_overlap = set(raw_dict.keys()) & set(robot_dict_flat.keys())
    assert len(key_overlap) == 0, key_overlap
    raw_dict.update(robot_dict_flat)
    assert len(raw_dict) > 0
    return raw_dict


# TODO (dale.mcconachie) Hoist these builder functions to a more central/common
# location when they are used by a 2nd consumer.


def rpy_deg(rpy_deg):
    """Converts RPY in degress to a RotationMatrix."""
    return RotationMatrix(RollPitchYaw(np.deg2rad(rpy_deg)))


def xyz_rpy(xyz, rpy):
    """Shorthand to create an isometry from XYZ and RPY."""
    return RigidTransform(R=RotationMatrix(rpy=RollPitchYaw(rpy)), p=xyz)


def xyz_rpy_deg(xyz, rpy_deg):
    return xyz_rpy(xyz, np.deg2rad(rpy_deg))


_DEFAULT_PLACEHOLDER_CAMERAS = ["scene_right_0"]


def make_example_obs_and_act(camera_names: list[str] | None = None):
    """Constructs a manually defined observation and action pair. If
    camera_names is None, a default value of `_DEFAULT_PLACEHOLDER_CAMERAS` is
    used.
    """
    if camera_names is None:
        camera_names = _DEFAULT_PLACEHOLDER_CAMERAS

    h, w, c = (48, 64, 3)
    fake_rgb = np.zeros((h, w, c), dtype=np.uint8)
    fake_depth = np.zeros((h, w), dtype=np.uint16)
    fake_label = np.zeros((h, w), dtype=np.uint16)
    fake_K = np.eye(3)
    fake_X_TC = RigidTransform()
    model_1 = "right::panda"
    gripper_1 = "right::panda_hand"
    model_2 = "left::panda"
    gripper_2 = "left::panda_hand"

    obs = MultiarmObservation(
        robot=PosesAndGrippersActualAndDesired(
            actual=PosesAndGrippers(
                poses={
                    model_1: xyz_rpy_deg([0.1, 0.2, 0.3], [15, 30, 45]),
                    model_2: xyz_rpy_deg([-0.4, 0.5, -0.6], [-45, 60, -15]),
                },
                grippers={
                    gripper_1: 0.02,
                    gripper_2: 0.04,
                },
                joint_position={
                    model_1: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                    model_2: np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]),
                },
                joint_velocity={
                    model_1: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                    model_2: np.array([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0]),
                },
                joint_torque={
                    model_1: np.array(
                        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
                    ),
                    model_2: np.array(
                        [-10.0, 20.0, -30.0, 40.0, -50.0, 60.0, -70.0]
                    ),
                },
                joint_torque_external={
                    model_1: np.array(
                        [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
                    ),
                    model_2: np.array(
                        [-20.0, 40.0, -60.0, 80.0, -100.0, 120.0, -140.0]
                    ),
                },
                wrench={
                    model_1: np.array([0.2, 0.6, 0.8, 2, 6, 8]),
                    model_2: np.array([-0.2, 0.6, -0.8, 2, -6, 8]),
                },
                external_wrench={
                    model_1: np.array([0.02, 0.06, 0.08, 0.2, 0.6, 0.8]),
                    model_2: np.array([-0.02, 0.06, -0.08, 0.2, -0.6, 0.8]),
                },
                timestamp_data=0.123,
                timestamp_sent=None,
                timestamp_received=0.456,
            ),
            desired=PosesAndGrippers(
                poses={
                    model_1: xyz_rpy_deg(
                        [0.11, 0.21, 0.31], [15.1, 30.1, 45.1]
                    ),
                    model_2: xyz_rpy_deg(
                        [-0.41, 0.51, -0.61], [-45.1, 60.1, -15.1]
                    ),
                },
                grippers={
                    gripper_1: 0.021,
                    gripper_2: 0.041,
                },
                joint_position={
                    model_1: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                    model_2: np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]),
                },
                joint_velocity=None,
                joint_torque=None,
                joint_torque_external=None,
                timestamp_sent=0.789,
            ),
            version=CURRENT_VERSION,
        ),
        visuo={
            name: CameraImageSet(
                rgb=CameraRgbImage(fake_rgb, fake_K, fake_X_TC),
                depth=CameraDepthImage(fake_depth, fake_K, fake_X_TC),
                label=CameraLabelImage(fake_label, fake_K, fake_X_TC),
            )
            for name in camera_names
        },
        timestamp_packaged=1.234,
        language_instruction="do something",
    )
    act = PosesAndGrippers(
        poses={
            "right::panda": xyz_rpy_deg([0.1, 0.2, 0.3], [15, 30, 45]),
            "left::panda": xyz_rpy_deg([-0.4, 0.5, -0.6], [-45, 60, -15]),
        },
        grippers={
            "right::panda_hand": 0.02,
            "left::panda_hand": 0.04,
        },
        debugging_output={
            "skill": ["something"],
        },
    )
    return obs, act


def make_example_obs_and_obs_dict(camera_names: list[str] | None = None):
    """Constructs a manually transcribed version of the data returned by
    `make_example_obs_and_act()`. If camera_names is None, a default value of
    `_DEFAULT_PLACEHOLDER_CAMERAS` is used.
    """
    if camera_names is None:
        camera_names = _DEFAULT_PLACEHOLDER_CAMERAS

    # Hijack the dataclass version to get the camera images.
    obs, _ = make_example_obs_and_act(camera_names)
    rgbs = {name: obs.visuo[name].rgb.array for name in camera_names}
    depths = {
        f"{name}_depth": obs.visuo[name].depth.array for name in camera_names
    }
    labels = {
        f"{name}_label": obs.visuo[name].label.array for name in camera_names
    }

    # Populate the dict version.
    obs_dict = {
        **rgbs,
        **labels,
        **depths,
        "robot__actual__poses__right::panda__xyz": np.array([0.1, 0.2, 0.3]),
        "robot__actual__poses__right::panda__rot_6d": np.array(
            [
                0.61237244,
                -0.59150635,
                0.52451905,
                0.61237244,
                0.77451905,
                0.15849365,
            ]
        ),
        "robot__actual__poses__left::panda__xyz": np.array([-0.4, 0.5, -0.6]),
        "robot__actual__poses__left::panda__rot_6d": np.array(
            [
                0.48296291,
                -0.40849365,
                0.77451905,
                -0.12940952,
                0.84150635,
                0.52451905,
            ]
        ),
        "robot__actual__grippers__right::panda_hand": np.array([0.02]),
        "robot__actual__grippers__left::panda_hand": np.array([0.04]),
        "robot__actual__joint_position__right::panda": np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        "robot__actual__joint_position__left::panda": np.array(
            [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]
        ),
        "robot__actual__joint_velocity__right::panda": np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
        "robot__actual__joint_velocity__left::panda": np.array(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0]
        ),
        "robot__actual__joint_torque__right::panda": np.array(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        ),
        "robot__actual__joint_torque__left::panda": np.array(
            [-10.0, 20.0, -30.0, 40.0, -50.0, 60.0, -70.0]
        ),
        "robot__actual__joint_torque_external__right::panda": np.array(
            [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
        ),
        "robot__actual__joint_torque_external__left::panda": np.array(
            [-20.0, 40.0, -60.0, 80.0, -100.0, 120.0, -140.0]
        ),
        "robot__actual__wrench__right::panda": np.array(
            [0.2, 0.6, 0.8, 2.0, 6.0, 8.0]
        ),
        "robot__actual__wrench__left::panda": np.array(
            [-0.2, 0.6, -0.8, 2.0, -6.0, 8.0]
        ),
        "robot__actual__external_wrench__right::panda": np.array(
            [0.02, 0.06, 0.08, 0.2, 0.6, 0.8]
        ),
        "robot__actual__external_wrench__left::panda": np.array(
            [-0.02, 0.06, -0.08, 0.2, -0.6, 0.8]
        ),
        "robot__actual__timestamp_data": np.array([0.123]),
        "robot__actual__timestamp_received": np.array([0.456]),
        "robot__desired__poses__right::panda__xyz": np.array(
            [0.11, 0.21, 0.31]
        ),
        "robot__desired__poses__right::panda__rot_6d": np.array(
            [
                0.61068579,
                -0.59166356,
                0.52630513,
                0.61282122,
                0.77404131,
                0.1590918,
            ]
        ),
        "robot__desired__poses__left::panda__xyz": np.array(
            [-0.41, 0.51, -0.61]
        ),
        "robot__desired__poses__left::panda__rot_6d": np.array(
            [
                0.48127627,
                -0.40897299,
                0.77531558,
                -0.1298583,
                0.84146443,
                0.52447539,
            ]
        ),
        "robot__desired__grippers__right::panda_hand": np.array([0.021]),
        "robot__desired__grippers__left::panda_hand": np.array([0.041]),
        "robot__desired__joint_position__right::panda": np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        "robot__desired__joint_position__left::panda": np.array(
            [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]
        ),
        "robot__desired__timestamp_sent": np.array([0.789]),
        "robot__version": np.array([CURRENT_VERSION]),
        "timestamp_packaged": np.array([1.234]),
        "language_instruction": "do something",
    }
    return obs, obs_dict
