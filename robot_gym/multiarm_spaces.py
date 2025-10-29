import dataclasses as dc
from typing import Any, Dict, List, Optional

import numpy as np
from pydrake.math import RigidTransform

# Whenever you update this schema, please also follow the instructions in
# intuitive/visuomotor/test/test_multiarm.README.md, and update the tests
# convert_to_spartan_format_test.py and test_multiarm_episode_load_test.py.


@dc.dataclass
class PosesAndGrippers:
    # Per model.
    poses: Dict[str, RigidTransform]
    # Per gripper.
    grippers: Dict[str, float]

    # Additional per-model observation data.
    joint_position: Dict[str, np.ndarray] = None
    joint_velocity: Dict[str, np.ndarray] = None
    joint_torque: Dict[str, np.ndarray] = None
    joint_torque_external: Dict[str, np.ndarray] = None
    wrench: Dict[str, np.ndarray] = None
    external_wrench: Dict[str, np.ndarray] = None

    # Timing addons.

    # Timestamp (seconds). Wall time from the status message (`utime` for
    # actual, `command_utime` for desired). Specifically, these timestamps are
    # generated in `robot_control_main`'s `publish_status()`.
    timestamp_data: Optional[float] = None
    # Timestamp (seconds). Wall time for when this object was "sent" to the
    # robot.
    timestamp_sent: Optional[float] = None
    # Timestamp (seconds). Wall time for when this object was received by the
    # env in this process.
    timestamp_received: Optional[float] = None

    # Optional debugging information, typically used for the relevant Policy
    # (e.g., diffusion policy debug outputs we want to record).
    debugging_output: Optional[Dict[str, Any]] = None


@dc.dataclass
class PosesAndGrippersActualAndDesired:
    actual: PosesAndGrippers
    desired: PosesAndGrippers

    # This is a change-detector field for this struct + PosesAndGrippers, meant
    # to loosely indicate the "schema" we have. Notable dates:
    # - 20230831: Schema without forces. `version` field will be None.
    # - 20231218: Schema with forces. `version` field will be None.
    # - 20240422: Addition of timestamp fields. `version` field will be None.
    # - 20240503: Addition of `version` field.
    # - 20240530: Addition of semantic camera names.
    # - 20240909: Minor module / class renames.
    # - 20241007: Minor module renames for move from Anzu to LBM repo.
    # - 20241212: Addition of the language field in MultiarmObservation.
    # We set this to None so that way pickles before versioning indicate there
    # was no data present.
    version: Optional[float] = None


# N.B. We ddefer setting the current version to `multiarm.py`.
CURRENT_VERSION = float(20241212)


@dc.dataclass
class CameraImage:
    array: np.ndarray  # See more specific instantations.
    K: np.ndarray  # 3x3
    X_TC: RigidTransform


@dc.dataclass
class CameraRgbImage(CameraImage):
    array: np.ndarray  # (H, W, C), dtype=uint8
    timestamp: Optional[float] = None


@dc.dataclass
class CameraDepthImage(CameraImage):
    array: np.ndarray  # (H, W), dtype=uint16
    timestamp: Optional[float] = None


@dc.dataclass
class CameraLabelImage(CameraImage):
    array: np.ndarray  # (H, W), dtype= ?
    timestamp: Optional[float] = None


@dc.dataclass
class CameraImageSet:
    rgb: CameraRgbImage
    depth: Optional[CameraDepthImage] = None
    label: Optional[CameraLabelImage] = None


# Camera Id -> Camera images
CameraImageSetMap = dict[str, CameraImageSet]


@dc.dataclass
class MultiarmObservation:
    robot: PosesAndGrippersActualAndDesired
    visuo: Dict[str, CameraImageSet]
    # Timestamp (seconds). Wall time for when this object was assembled.
    timestamp_packaged: Optional[float] = None

    language_instruction: Optional[str] = None


@dc.dataclass
class RestorePosesAndGrippersConfig:
    # Note: This only unpacks poses and grippers. It does not attempt to unpack
    # force or joint configuration information.
    model_names: List[str]
    gripper_names: List[str]

    @staticmethod
    def make_default():
        # Any action vectors that have unspecified order should have this
        # ordering.
        return RestorePosesAndGrippersConfig(
            model_names=["right::panda", "left::panda"],
            gripper_names=["right::panda_hand", "left::panda_hand"],
        )
