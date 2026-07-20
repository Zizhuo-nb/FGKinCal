import numpy as np
import gtsam

def rigid_transform(pc_R, T_last):
    points_h = np.hstack([
        pc_R,
        np.ones((pc_R.shape[0], 1))
    ])

    transformed_h = (T_last @ points_h.T).T

    return transformed_h[:, :3]



def icp_error_jacobian(p2, n, delta_T):
    R = delta_T[:3, :3]

    N = p2.shape[0]
    J = np.zeros((N, 6), order="F")

    for i in range(N):
        pi = p2[i]
        ni = n[i]

        J_rot = -ni @ R @ skew(pi)
        J_trans = ni @ R

        J[i, 0:3] = J_rot
        J[i, 3:6] = J_trans

    return J







def smooth_error_jacobian(delta_T_prev,
                          delta_T_curr,
                          Delta_T_base_prev,
                          Delta_T_base_curr):
    """
    Jacobian of:

        r = Log( X_prev^{-1} X_curr )

    where:

        X_prev = delta_T_prev @ Delta_T_base_prev
        X_curr = delta_T_curr @ Delta_T_base_curr

    GTSAM Pose3 tangent order:
        [rx, ry, rz, tx, ty, tz]

    Return:
        H_prev, H_curr
    """

    X_prev = delta_T_prev @ Delta_T_base_prev
    X_curr = delta_T_curr @ Delta_T_base_curr

    E = np.linalg.inv(X_prev) @ X_curr

    Ad_E_inv = adjoint_SE3(np.linalg.inv(E))
    Ad_Bprev_inv = adjoint_SE3(np.linalg.inv(Delta_T_base_prev))
    Ad_Bcurr_inv = adjoint_SE3(np.linalg.inv(Delta_T_base_curr))

    H_prev = -Ad_E_inv @ Ad_Bprev_inv
    H_curr = Ad_Bcurr_inv

    return np.asarray(H_prev, order="F"), np.asarray(H_curr, order="F")


#=========================#
def skew(p):
    x, y, z = p
    return np.array([
        [0.0, -z,   y],
        [z,   0.0, -x],
        [-y,  x,   0.0]
    ])

def adjoint_SE3(T):
    """
    SE(3) adjoint matrix.

    T = [R t; 0 1]

    Tangent order:
        xi = [omega, v]
           = [rx, ry, rz, tx, ty, tz]

    Ad_T xi =
        [ R omega
          [t]x R omega + R v ]
    """

    R = T[:3, :3]
    t = T[:3, 3]

    Ad = np.zeros((6, 6))

    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = skew(t) @ R
    Ad[3:6, 3:6] = R

    return Ad



#==============================cubic spline related functions========================
def icp_error_cubic_spline_jacobian(p2,n,timeL):
    N = p2.shape[0]
    J = np.zeros((N, 24), order="F")
    for i in range(N):
        pi = p2[i]
        ni = n[i]
        u = timeL[i]
        J_rot = -ni @ skew(pi)   # (3,)
        J_trans = ni             # (3,)

        J_pose = np.hstack((J_rot, J_trans))  # (6,)

        J[i, 0:6] = J_pose
        J[i, 6:12] = u * J_pose
        J[i, 12:18] = (u**2) * J_pose
        J[i, 18:24] = (u**3) * J_pose

    return J


def c0_between_factor_cubic_spline_jacobian(timeL):
    I6 = np.eye(6)
    H_left = np.hstack((
        I6,
        timeL * I6,
        timeL**2 * I6,
        timeL**3 * I6
    ))

    H_right = np.hstack((
        -I6,
        np.zeros((6, 6)),
        np.zeros((6, 6)),
        np.zeros((6, 6))
    ))

    return H_left, H_right


def c1_between_factor_cubic_spline_jacobian(timeL):
    I6 = np.eye(6)
    Z6 = np.zeros((6, 6))

    H_left = np.hstack((
        Z6,
        I6,
        2 * timeL * I6,
        3 * timeL**2 * I6
    ))

    H_right = np.hstack((
        Z6,
        -I6,
        Z6,
        Z6
    ))

    return H_left, H_right


def c2_between_factor_cubic_spline_jacobian(timeL):
    I6 = np.eye(6)
    Z6 = np.zeros((6, 6))

    H_left = np.hstack((
        Z6,
        Z6,
        2 * I6,
        6 * timeL * I6
    ))

    H_right = np.hstack((
        Z6,
        Z6,
        -2 * I6,
        Z6
    ))

    return H_left, H_right


def transform_points_with_cubic_spline(p2, coefficients, timeL):
    """
    coefficients: (24,)，顺序为 [a0, a1, a2, a3]
    每个 aj 都是6维。
    timeL: (N,)，每个 p2 点对应的局部时间。
    """

    coefficients = coefficients.reshape(4, 6)

    p2_new = np.zeros_like(p2, dtype=float)

    for i in range(p2.shape[0]):
        t = timeL[i]

        xi = (
            coefficients[0]
            + t * coefficients[1]
            + t**2 * coefficients[2]
            + t**3 * coefficients[3]
        )

        delta_T = gtsam.Pose3.Expmap(xi).matrix()

        R = delta_T[:3, :3]
        trans = delta_T[:3, 3]

        p2_new[i] = R @ p2[i] + trans

    return p2_new