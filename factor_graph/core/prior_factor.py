import numpy as np
from scipy.spatial.transform import Rotation

'''
input : rigid body transform matrix (4,4)
return : [rx, ry, rz, tx, ty, tz]
'''

def prior_error(T_curremt, T_prior = np.eye(4)):
    T_err = np.linalg.inv(T_prior) @ T_curremt
    R_err = T_err[0:3, 0:3]
    t_err = T_err[0:3, 3]

    rotvec = Rotation.from_matrix(R_err).as_rotvec()

    prior_residual = np.hstack((rotvec, t_err))

    return prior_residual

def prior_error_func(Delta_T_prev):
    def error_func(this, values, H):
        key = this.key()[0]
        delta_pose = values.atPose3(key)
        delta_T = delta_pose.matrix()
        Delta_T_current = delta_T @ Delta_T_prev

        residual = prior_error(Delta_T_current, np.eye(4))

        if H is not None:
            H[0] = np.zeros((6,6))

        return residual
    return error_func