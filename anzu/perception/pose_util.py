import numpy as np
from numpy.linalg import norm

from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix


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


def rotation_matrix_to_axang3(R):
    assert isinstance(R, RotationMatrix), repr(type(R))
    axang = AngleAxis(R.matrix())
    axang3 = axang.angle() * axang.axis()
    return axang3


def so3_vector_minus(R_WA, R_WD):
    """
    Provides an error δR_DA_W on the tangent bundle of SE(3) for an actual
    frame A and desired frame D, whose poses are expressed in common frame W.

    For a trivial first-order system with coordinates of R_WA, whose dynamics
    are defined via angular velocity:
        Ṙ_WA = ω_WA × R_WA
    We can use this error to exponentially drive A to D (when D is stationary):
        δR_DA_W = R_WA ⊖ R_WD  (what this function provides)
        ω_WA = -δR_DA_W
    Note that this is restricted to angular error magnitude within interval
    [0, π).

    The analogous error and feedback on a Euclidean quantity x ∈ Rⁿ with actual
    value xₐ and desired value xₜ:
        eₓ = xₐ - xₜ
        ẋₐ = -eₓ
    """
    # TODO(eric.cousineau): Add citations.
    # Wa is W from "actual perspective", Wd is W from "desired perspetive".
    # TODO(eric.cousineau): Better notation for this difference?
    R_WaWd = R_WA @ R_WD.inverse()
    dR_DA_W = rotation_matrix_to_axang3(R_WaWd)
    return dR_DA_W


def se3_vector_minus(X_WA, X_WD):
    """
    Extension of so3_vector_minus() from SO(3) to SE(3). Returns as a 6d
    vector that can be used for feedback via spatial velocity / acceleration:
        δX_DA_W = X_WA ⊖ X_WD
                = [δR_DA_W, p_DA_W]
    """
    # TODO(eric.cousineau): Hoist comments to Drake's
    # ComputePoseDiffInCommonFrame and rename that function.
    dX_DA_W = np.zeros(6)
    dX_DA_W[:3] = so3_vector_minus(X_WA.rotation(), X_WD.rotation())
    dX_DA_W[3:] = X_WA.translation() - X_WD.translation()
    return dX_DA_W


def axang3_to_rotation_matrix(axang3, tol=1e-5):
    """Converts a 3-element axis-angle rotation to pydrake's RotationMatrix."""
    assert isinstance(axang3, (list, np.ndarray)), repr(type(axang3))
    axang3 = np.asarray(axang3)
    assert axang3.size == 3, axang3.shape
    # TODO(eric.cousineau): Choose more intelligent tolerance.
    angle = norm(axang3)
    if angle <= tol:
        # Assume identity
        R = RotationMatrix()
    else:
        axis = axang3 / angle
        R = RotationMatrix(AngleAxis(angle=angle, axis=axis))
    return R


def rot_distance_angle(Ra, Rb):
    """
    Rotation / SO3 distance as a single (absolute) angle in radians.
    """
    Ra = RotationMatrix(Ra)
    Rb = RotationMatrix(Rb)
    ang = np.abs(AngleAxis((Ra.inverse() @ Rb).matrix()).angle())
    return ang


def se3_distance_pair(Xa, Xb, rot_distance=rot_distance_angle):
    """
    Returns [translation_dist, rotation_dist], rotation in degrees.
    """
    Xa = RigidTransform(Xa)
    Xb = RigidTransform(Xb)
    tr = norm(Xa.translation() - Xb.translation())
    rot_deg = rot_distance(Xa.rotation(), Xb.rotation()) * 180.0 / np.pi
    # N.B. While recarray's would be ideal, NumPy recarrays are really
    # confusing with scalars, esp. for default / unset values
    # (e.g. assuming 0).
    return np.array([tr, rot_deg])


def translation_distance(Xa, Xb):
    return norm(
        RigidTransform(Xa).translation() - RigidTransform(Xb).translation()
    )


def xyz_rpy(xyz, rpy):
    """Shorthand to create an isometry from XYZ and RPY."""
    return RigidTransform(R=RotationMatrix(rpy=RollPitchYaw(rpy)), p=xyz)


def xyz_rpy_deg(xyz, rpy_deg):
    return xyz_rpy(xyz, np.deg2rad(rpy_deg))


def rpy_deg(rpy_deg):
    """Converts RPY in degress to a RotationMatrix."""
    return RotationMatrix(RollPitchYaw(np.deg2rad(rpy_deg)))


def to_xyz_rpy(X):
    """Converts an isometry to a tuple of 2 e-element numpy
    arrays composed of the translation (XYZ) and rotation (RPY)
    in that order"""
    X = RigidTransform(X)
    rpy = RollPitchYaw(X.rotation()).vector()
    return (X.translation(), rpy)


def to_xyz_rpy_deg(X):
    xyz, rpy = to_xyz_rpy(X)
    return (xyz, rpy * 180 / np.pi)


def to_xyz_rpy_list(X):
    """Converts an isometry to a 6 element list composed of
    the translation (XYZ).and rotation (RPY)"""
    xyz, rpy = to_xyz_rpy(X)
    return xyz.tolist() + rpy.tolist()


def normalize(v, tol=None):
    """Normalizes with optional tolerance for zeros."""
    vn = norm(v)
    if tol is not None and vn <= tol:
        return np.zeros_like(v)
    else:
        return v / vn


def vec_to_vec_rotation(start_in, end_in, psi=0.0):
    """
    Generate a rotation matrix that rotates a vector @p start to anther vector
    @p end.
    @param psi There is a family of rotations that can rotation start to end.
        Think of this as an index to a specific rotation in the family.
        In non-degenerated case, it is the angle between the rotation axis and
        the cross product vector of start and end.
    @return a numpy rotation matrix R such that end = R @ start.
    """
    start = normalize(start_in)
    end = normalize(end_in)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if abs(start.dot(end) - 1) < 1e-5:
        R = RotationMatrix(AngleAxis(psi, start)).matrix()
    elif abs(start.dot(end) + 1) < 1e-5:
        ortho_start = np.ones(3)
        max_idx = np.abs(start).argmax()
        ortho_start[max_idx] = -start.sum() / start[max_idx] + 1
        ortho_start = normalize(ortho_start)
        if start.dot(ortho_start) > 1e-5:
            raise RuntimeError("Wrong ortho start")
        cross_vec = np.cross(start, ortho_start)
        axis = cos_psi * cross_vec + sin_psi * ortho_start
        R = RotationMatrix(AngleAxis(np.pi, axis)).matrix()
    else:
        midpoint = normalize(start + end)
        cross_vec = normalize(np.cross(start, end))
        axis = cos_psi * cross_vec + sin_psi * midpoint

        # https://math.stackexchange.com/questions/2548811/find-an-angle-to-rotate-a-vector-around-a-ray-so-that-the-vector-gets-as-close-a
        e_vec = normalize(start - (start.dot(axis)) * axis)
        f_vec = np.cross(axis, e_vec)
        angle = np.arctan2(end.dot(f_vec), end.dot(e_vec))

        R = RotationMatrix(AngleAxis(angle, axis)).matrix()
        if not np.allclose(end, R @ start):
            raise AssertionError("Computed rotation is incorrect!")

    return R


def _sample_uniform(bounds):
    return np.random.uniform(bounds[0], bounds[1])


def sample_uniform_vector_in_cone(cone_axis_in, cone_angle):
    """
    https://math.stackexchange.com/a/205589
    """
    cone_axis = normalize(cone_axis_in)
    z_N = _sample_uniform([np.cos(cone_angle / 2), 1.0])
    sqrt_1_zN2 = np.sqrt(1 - z_N**2)
    phi_N = _sample_uniform([0.0, 2.0 * np.pi])
    vec_N = np.array(
        [sqrt_1_zN2 * np.cos(phi_N), sqrt_1_zN2 * np.sin(phi_N), z_N]
    )
    R_ConeN = vec_to_vec_rotation(np.array([0.0, 0.0, 1.0]), cone_axis)
    return R_ConeN @ vec_N


def calc_se3_kabsch(p_BPset, p_WQset):
    """
    Kabsch, Procrustes, Horn, Wabha, whatevs. Use this to do crappy touch-point
    calibration for robots.

    Given points P w.r.t. frame B (p_BPset), and points Q w.r.t frame W
    (p_WQset) that correspond, find SE(3) transform X_WB that maps
    points p_BPset to p_WQset with minimum error:

        min  ∑ᵢ |X_WB * p_BPᵢ - p_WQᵢ|²
        X_WB

    It is suggested to align frame B to a physically identifiable body frame
    (e.g. table-top-center) and align frame W to a known kinematic base (e.g.
    Panda arm or IIWA arm base). Then you can record p_BPset with a measuring
    tape or CAD, and you can then move the robot arm (e.g. Panda Pilot mode, or
    IIWA pendant mode) to touch those positions with a known end-effector frame
    (generally the finger tip).

    See also:
    - https://en.wikipedia.org/wiki/Kabsch_algorithm
    - https://github.com/RobotLocomotion/drake/blob/b108c42c1/perception/estimators/dev/pose_closed_form.h#L20-L46
    """  # noqa
    p_BPset = np.array(p_BPset)
    p_WQset = np.array(p_WQset)
    assert p_BPset.ndim == 2
    assert p_BPset.shape[1] == 3
    assert p_BPset.shape == p_WQset.shape

    # Mean. Pm represents the mean of Pset, while Qm represents mean of Qset.
    p_BPm = np.mean(p_BPset, axis=0)
    p_PmPset_B = p_BPset - p_BPm
    p_WQm = np.mean(p_WQset, axis=0)
    p_QmQset_W = p_WQset - p_WQm

    # Covariance.
    H_WB = p_QmQset_W.T @ p_PmPset_B
    # N.B. It'd be nice to use `RotationMatrix.ProjectToRotationMatrix()`, but
    # it doesn't handle degenerate cases well (det(H_WB) < 0).
    U, _, Vh = np.linalg.svd(H_WB)
    V = Vh.T
    d = np.sign(np.linalg.det(V.T @ U))
    D = np.diag([1, 1, d])
    R_WB = RotationMatrix(U @ D @ V.T)

    # Compute offset. We assume that Pm and Qm correspond to the same points,
    # and thus we use them to locate the origins of B and W, respectively.
    p_PmB_W = R_WB @ -p_BPm
    p_QmW = -p_WQm
    p_WB = p_PmB_W - p_QmW
    X_WB = RigidTransform(R_WB, p_WB)
    return X_WB


def is_pose_same(X_TP, X_TP_des, pos_thresh, deg_thresh):
    """
    Checks that two poses, X_TP and X_TP_des, are within a given threshold.
    """
    assert isinstance(X_TP, RigidTransform)
    assert isinstance(X_TP_des, RigidTransform)
    xyz_diff = X_TP.translation() - X_TP_des.translation()
    rot_diff = X_TP_des.rotation().inverse() @ X_TP.rotation()
    rot_diff = rot_diff.ToAngleAxis()
    if np.linalg.norm(xyz_diff) > pos_thresh:
        return False
    if np.abs(rot_diff.angle()) > np.deg2rad(deg_thresh):
        return False
    return True


def saturate_se3_diff(X_WA, X_WAdes, *, pos_dist_max, ang_diff_max):
    """
    Takes a desired pose X_WAdes (frame Ades w.r.t. frame W), and saturates it
    to be near X_WA (A w.r.t. frame W) with the following criteria:

    - p_AAdes_W is within a ball of pos_dist_max (m).
    - R_AAdes is decomposed to angle-axis, and the angle is saturated to
    ang_diff_max (rad).

    Returns newly saturated X_WAdes.
    """
    p_AAdes_W = X_WAdes.translation() - X_WA.translation()
    dist = norm(p_AAdes_W)
    if dist > pos_dist_max:
        p_AAdes_W *= pos_dist_max / dist

    R_AAdes = X_WA.rotation().inverse() @ X_WAdes.rotation()
    axang_AAdes = AngleAxis(R_AAdes.matrix())
    assert axang_AAdes.angle() >= 0
    if axang_AAdes.angle() > ang_diff_max:
        axang_AAdes = AngleAxis(ang_diff_max, axang_AAdes.axis())
    R_AAdes = RotationMatrix(axang_AAdes)

    X_WAdes = RigidTransform(
        R=X_WA.rotation() @ R_AAdes,
        p=X_WA.translation() + p_AAdes_W,
    )
    return X_WAdes
