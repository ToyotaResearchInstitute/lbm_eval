from pydrake.math import RigidTransform

# TODO(eric.cousineau): Make this match exactly with C++.


class PoseEstimate:
    """Stores pose estimate from a detection.
    `O` is object frame, `F` is base / expressed-in frame.
    """

    def __init__(
            self,
            model_name,
            X_FO,
            confidence=None,
            timestamp=None,
            frame_name="undefined",
            X_FO_inv_cov=None,
            raw_measurements=None,
    ):
        """
        @param X_FO_inv_cov The information (inverse covariance) matrix of
        X_FO expressed in the object frame O.
        @param raw_measurements The raw measurements used to estimate this
            pose, e.g. KOSNetMeasurement with center, north, azimuth,
            elevation, etc. in KOSNet context. This is necessary for multiview
            fusion at raw measurements level to avoid linearization errors.
        """
        self.model_name = model_name
        self.X_FO = RigidTransform(X_FO)
        self.X_FO_inv_cov = X_FO_inv_cov
        self.confidence = confidence
        self.timestamp = timestamp
        self.frame_name = frame_name
        self.raw_measurements = raw_measurements

    def difference(self, other):
        """
        Computes pose difference between this and another PoseEstimate.

        Assume this instance's frame is `E`, the other's frame is `A`, and
        we have poses in the world frame `W`.

        The rotation is computed in a fixed model frame, which assumes `A == E`
        and we use different world frames `We` and `Wa`, thus we get:
            R_WeWa = R_WeE * inv(R_WaA)
        This is useful for interpreting the rotation in the world frame.

        The translation is computed in the base frame:
            p_EA_W = p_WA - p_WE

        @param other PoseEstimate to compare to, in frame `A`.
        @return (R_WeRa, p_EA_W)
        """
        assert self.frame_name == other.frame_name
        X_WE = self.X_FO
        X_WA = other.X_FO
        R_WeWa = X_WE.rotation() @ X_WA.rotation().inverse()
        p_EA_W = X_WA.translation() - X_WE.translation()
        return (R_WeWa.matrix(), p_EA_W)

    def __repr__(self):
        return (
            "<PoseEstimate: model_name={}, timestamp={}, frame_name={}, confidence={}>"  # noqa
            .format(self.model_name, self.timestamp, self.frame_name,
                    self.confidence))

    def reexpress(self, new_frame, X_FnF_map):
        """Re-express pose of old frame F in new frame Fn."""
        if self.frame_name != new_frame:
            X_FnF = X_FnF_map[self.frame_name]
            X_FnO = X_FnF @ self.X_FO
            return PoseEstimate(
                model_name=self.model_name,
                X_FO=X_FnO,
                # Since the information matrix is expressed in the object frame
                # O, it doesn't change when the base frame changes.
                X_FO_inv_cov=self.X_FO_inv_cov,
                frame_name=new_frame,
                timestamp=self.timestamp,
                confidence=self.confidence,
                raw_measurements=self.raw_measurements,
            )
        else:
            return self
