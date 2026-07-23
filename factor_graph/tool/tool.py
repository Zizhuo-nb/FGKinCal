import numpy as np

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

def prior_error_jacobian():
    pass

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


    #========================================FOR SPLINE OPTIMIZATION========================================#

    def icp_spline_jacobian():
        pass

    def c0_spline_jacobian():
        pass

    def c1_spline_jacobian():
        pass    

    def c2_spline_jacobian():
        pass
    