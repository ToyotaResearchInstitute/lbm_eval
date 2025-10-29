"""
TODO(sfeng): everything here needs to be lifted to c++ and then pybinded.
Reason being we need to inject this fisheye distortion block in between
rgbd sensors and downstream ros / lcm publishers, which are currently all
implemented in c++ under ApplyAnzuCameraConfigWithRos()
"""

import functools

import cv2
import numpy as np

from pydrake.common.value import Value
from pydrake.systems.framework import LeafSystem
from pydrake.systems.sensors import ImageDepth16U, ImageLabel16I, ImageRgba8U

from anzu.sim.camera.camera_config import AnzuCameraConfig


def apply_fisheye_distortion_to_plant(
    camera: AnzuCameraConfig,
    builder,
):
    """
    If `camera` has fisheye parameters declared in its fisheye_distortion
    field, this function will add a FisheyeDistortionLeafSystem leaf system
    to the `builder` for `camera`'s to distort its `color_image` output
    channel. No op otherwise.
    """
    if camera.fisheye_distortion is None:
        return

    K_pinhole = np.eye(3)
    K_pinhole[0, 0] = camera.focal.x
    K_pinhole[1, 1] = camera.focal.y
    K_pinhole[0, 2] = camera.center_x
    K_pinhole[1, 2] = camera.center_y

    port_names = []
    port_names.extend(["color_image"] if camera.rgb else [])
    port_names.extend(["depth_image_16u"] if camera.depth else [])
    port_names.extend(["label_image"] if camera.label else [])

    fisheye_sys = builder.AddSystem(
        FisheyeDistortionLeafSystem(
            image_width=camera.width,
            image_height=camera.height,
            K_pinhole=K_pinhole,
            K_fisheye=camera.fisheye_distortion.get_K(),
            D_fisheye=camera.fisheye_distortion.d,
            port_names=port_names,
        )
    )
    fisheye_sys.set_name("fisheye_distortion_" + camera.name)

    pinhole_camera = builder.GetSubsystemByName("rgbd_sensor_" + camera.name)
    for port_name in port_names:
        builder.Connect(
            pinhole_camera.GetOutputPort(port_name),
            fisheye_sys.GetInputPort(port_name),
        )


def _get_distortion_mapping(
    *,
    width: int,
    height: int,
    K_pinhole: np.ndarray,
    K_fisheye: np.ndarray,
    D_fisheye: np.ndarray,
):
    """
    Generates a distortion mapping used to warp a pinhole image into a fisheye
    image.
    Args:
        width: width of the distortion image to create
        height: height of the distortion image to create
        K_pinhole: 3x3 intrinsics associated with the source pinhole camera
        K_fisheye: 3x4 desired fisheye pinhole approximation intrinsics
        D_fisheye: 1x4 desired fisheye distortion parameters
    Returns: a mapping from a pinhole camera image to a fisheye image.
    """
    meshgrid = np.meshgrid(np.arange(width), np.arange(height))
    undistorted_points = np.stack(meshgrid, axis=-1).astype(np.float32)
    # Don't get confused by the function name. We are going from undistorted to
    # distorted.
    return cv2.fisheye.undistortPoints(
        undistorted_points, K_fisheye, D_fisheye, P=K_pinhole
    )


def _do_fisheye_distortion(
    *,
    image_pinhole: np.ndarray,
    distortion_mapping: np.ndarray,
    port_name: str
):
    """
    Applies fisheye distortion.
    Args:
        distortion_mapping: (h, w) mapping from pinhole to fisheye image.
        image_pinhole: (h, w, 3) or (h, w) image to be distorted.
        port_name: the Drake RgbdSensor output port name that generated the
            image_pinhole; "color_image" causes us to use color interpolation;
            any other values are assumed to be depth or label images that
            should not be interpolated.
    Returns: distorted image.
    """
    if port_name == "color_image":
        # Color images use bilinear interpolation.
        interpolation = cv2.INTER_LINEAR
    else:
        # Depth image (or label, etc.) use nearest-neighbor, because we never
        # want to blur their edges, the depth (or label) should always be some
        # specific pixel value from the pinhole camera.
        interpolation = cv2.INTER_NEAREST

    image_distorted = cv2.remap(
        image_pinhole,
        distortion_mapping[..., 0],
        distortion_mapping[..., 1],
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
    assert image_distorted.dtype == image_pinhole.dtype, (
        image_distorted.dtype,
        image_pinhole.dtype,
    )
    assert image_pinhole.shape == image_distorted.shape, (
        image_pinhole.shape,
        image_distorted.shape,
    )
    return image_distorted


class FisheyeDistortionLeafSystem(LeafSystem):
    """
    A LeafSystem that applies fisheye distortion to the input RGB image
    and copies the result to its output.
    """

    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        K_pinhole: np.array,
        K_fisheye: np.array,
        D_fisheye: np.array,
        port_names=("color_image",),
    ):
        """
        Args:
            image_width: width of image
            image_height: height of image
            K_pinhole: 3x3 intrinsics matrix for input pinhole image.
            K_fisheye: 3x3 fisheye intrinsics matrix used for output.
            D_fisheye: 1x4 distortion parameter used for output.
            port_names: the list of port names to declare (each as both an
                input port and an output port); each name must be one of
                either 'color_image', 'label_image', or 'depth_image_16u'.
        See FisheyeDistortionConfig for more details
        """

        LeafSystem.__init__(self)
        self.K_pinhole = K_pinhole
        self.K_fisheye = K_fisheye
        self.D_fisheye = D_fisheye
        self._width = image_width
        self._height = image_height

        self._distortion_mapping = _get_distortion_mapping(
            width=self._width,
            height=self._height,
            K_pinhole=self.K_pinhole,
            K_fisheye=self.K_fisheye,
            D_fisheye=self.D_fisheye,
        )

        image_types = dict(
            color_image=ImageRgba8U,
            depth_image_16u=ImageDepth16U,
            label_image=ImageLabel16I,
        )
        for port_name in port_names:
            image_type = image_types[port_name]
            model_value = Value(
                image_type(
                    width=self._width,
                    height=self._height,
                ),
            )
            self.DeclareAbstractInputPort(
                name=port_name,
                model_value=model_value,
            )
            self.DeclareAbstractOutputPort(
                name=port_name,
                alloc=model_value.Clone,
                calc=functools.partial(
                    self._apply_distortion,
                    port_name=port_name,
                ),
            )

    def _apply_distortion(self, context, output, *, port_name):
        # Extract the image data from the input port.
        # For color images, discard the alpha channel.
        # For other images, squeeze from (h, w, 1) to just (h, w).
        drake_image_in = self.GetInputPort(port_name).Eval(context)
        if port_name == "color_image":
            image_in = drake_image_in.data[..., :3]
        else:
            image_in = drake_image_in.data[..., 0]

        # Sanity check input size.
        hw = (self._height, self._width)
        assert image_in.shape[:2] == hw, image_in.shape

        # Apply fisheye distortion.
        image_out = _do_fisheye_distortion(
            image_pinhole=image_in,
            distortion_mapping=self._distortion_mapping,
            port_name=port_name,
        )

        # Sanity check output size.
        assert image_out.shape == image_in.shape, image_out.shape
        mutable_output = output.get_mutable_value().mutable_data
        assert mutable_output.shape[:2] == hw, mutable_output.shape

        # Copy the distorted image onto the output port.
        # For color image, hard-code the alpha channel to 100%.
        if port_name == "color_image":
            mutable_output[..., :3] = image_out
            mutable_output[..., -1] = 255
        else:
            mutable_output[..., 0] = image_out
