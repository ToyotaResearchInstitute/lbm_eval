import dataclasses
import functools

import numpy as np

from pydrake.systems.sensors import CameraConfig


@dataclasses.dataclass(kw_only=True)
class FisheyeDistortionConfig:
    """Struct to hold fisheye related parameters."""

    def get_K(self):
        K = np.zeros(shape=(3, 3))
        K[0, 0] = self.focal.focal_x()
        K[1, 1] = self.focal.focal_y()
        K[0, 2] = self.center_x
        K[1, 2] = self.center_y
        return K

    focal: CameraConfig.FocalLength = dataclasses.field(
        default_factory=CameraConfig.FocalLength,
    )
    """The x and y focal lengths (in pixels)."""

    center_x: float = 0.0
    """The x-position of the principal point (in pixels)."""

    center_y: float = 0.0
    """The y-position of the principal point (in pixels)."""

    d: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(4))
    """Distortion coefficients, (k1, k2, k3, k4).
    See https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html"""


def _dataclass_from_cxx_struct(*, cxx_struct, **dataclass_kwargs):
    """Decorator that respells a C++ serializable struct bound in Python to be
    an actual Python dataclass."""

    def _default_from_cxx(*, bound_cxx_struct, bound_name):
        return getattr(bound_cxx_struct(), bound_name)

    def _decorate(cls):
        for cxx_field in cxx_struct.__fields__:
            setattr(
                cls,
                cxx_field.name,
                dataclasses.field(
                    default_factory=functools.partial(
                        _default_from_cxx,
                        bound_cxx_struct=cxx_struct,
                        bound_name=cxx_field.name,
                    ),
                ),
            )
            # This is where dataclasses seeks type annotations on fields.
            # It doesn't have a secondary mechanism, so we need to go against
            # the advice at https://docs.python.org/3.10/howto/annotations.html
            # and directly frob the __annotations__ dictionary.
            cls.__annotations__[cxx_field.name] = cxx_field.type
        return dataclasses.dataclass(**dataclass_kwargs)(cls)

    return _decorate


@_dataclass_from_cxx_struct(cxx_struct=CameraConfig, kw_only=True)
class _DrakeCameraConfig:
    """A private python dataclass identical to Drake's C++ CameraConfig struct.
    This is used as the base class for AnzuCameraConfig."""


@dataclasses.dataclass(kw_only=True)
class AnzuCameraConfig(_DrakeCameraConfig):
    """Camera configuration specific to Anzu."""

    fisheye_distortion: FisheyeDistortionConfig | None = None
    """Optional corresponding fisheye distortion params if we want to model a
    fisheye camera."""

    @property
    def drake(self):
        """Property-like `drake` attribute. Returns the C++ base config struct
        with the same values as this. This returns a copy, not a view."""
        kwargs = dataclasses.asdict(self)
        kwargs.pop("fisheye_distortion", None)
        return CameraConfig(**kwargs)

    @drake.setter
    def drake(self, new_value):
        for field in dataclasses.fields(self):
            name = field.name
            if name == "fisheye_distortion":
                continue
            setattr(self, name, getattr(new_value, name))
