import numpy as np
import gtsam
from factor_graph.tool.tool import smooth_error_jacobian


def T_to_pose3(T):
    R = gtsam.Rot3(T[:3, :3])
    t = T[:3, 3]
    return gtsam.Pose3(R, gtsam.Point3(t[0], t[1], t[2]))


def smooth_residual(delta_T_prev,
                    delta_T_curr,
                    Delta_T_base_prev,
                    Delta_T_base_curr):
    X_prev = delta_T_prev @ Delta_T_base_prev
    X_curr = delta_T_curr @ Delta_T_base_curr

    E = np.linalg.inv(X_prev) @ X_curr

    return gtsam.Pose3.Logmap(T_to_pose3(E))


def smooth_error_func(Delta_T_base_prev, Delta_T_base_curr):
    def error_func(this, values, H):
        key_prev = this.keys()[0]
        key_curr = this.keys()[1]

        delta_pose_prev = values.atPose3(key_prev)
        delta_pose_curr = values.atPose3(key_curr)

        delta_T_prev = delta_pose_prev.matrix()
        delta_T_curr = delta_pose_curr.matrix()

        residual = smooth_residual(
            delta_T_prev,
            delta_T_curr,
            Delta_T_base_prev,
            Delta_T_base_curr
        )

        if H is not None:
            H_prev, H_curr = smooth_error_jacobian(
                delta_T_prev,
                delta_T_curr,
                Delta_T_base_prev,
                Delta_T_base_curr
            )

            H[0] = H_prev
            H[1] = H_curr

        return residual

    return error_func