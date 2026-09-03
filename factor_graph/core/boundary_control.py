import gtsam
import numpy as np
from scipy.spatial.transform import Rotation


def compute_mean_center_extrinsic(windows,window_ids,name=None):
    if name == "start":
        u = 0.0
    elif name == "end":
        u = 1.0
    else:
        u = 0.5
    basis = np.array([1,u,u**2,u**3])
    center_extrinsics = []
    for window_id in window_ids:
        coefficients = np.asarray(windows[window_id]['coefficients'], dtype = float).reshape(6,4)
        extrinsics = coefficients @ basis
        center_extrinsics.append(extrinsics)
    center_extrinsics = np.asarray(center_extrinsics)
    
    mean_rotation = Rotation.from_rotvec(center_extrinsics[:, :3]).mean().as_rotvec()
    mean_translation = np.mean(center_extrinsics[:, 3:], axis=0)
    
    mean_center_extrinsic = np.concatenate([mean_rotation, mean_translation])
    return mean_center_extrinsic  #[rx,ry,rz,x,y,z]

def boundary_error_func(mean_extrinsic, name):
    """Create a boundary factor with an analytical SE(3) Jacobian."""

    if name == "start":
        basis = np.array([1.0, 0.0, 0.0, 0.0])
    elif name == "end":
        basis = np.array([1.0, 1.0, 1.0, 1.0])
    else:
        raise ValueError("name must be 'start' or 'end'.")

    # Fixed target for this optimization.
    mean_extrinsic = np.asarray(
        mean_extrinsic, dtype=float
    ).reshape(6).copy()

    mean_rotation_T = Rotation.from_rotvec(
        mean_extrinsic[:3]
    ).as_matrix().T

    mean_translation = mean_extrinsic[3:]

    jacobian_spline = np.kron(
        np.eye(6),
        basis.reshape(1, 4),
    )

    identity = np.eye(3)

    def skew(v):
        x, y, z = v
        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ])

    def right_jacobian(rotvec):
        """SO(3) right Jacobian."""
        angle2 = float(rotvec @ rotvec)
        K = skew(rotvec)

        if angle2 < 1e-8:
            a = 0.5 - angle2 / 24.0 + angle2**2 / 720.0
            b = (
                1.0 / 6.0
                - angle2 / 120.0
                + angle2**2 / 5040.0
            )
        else:
            angle = np.sqrt(angle2)
            a = (1.0 - np.cos(angle)) / angle2
            b = (
                (angle - np.sin(angle))
                / (angle2 * angle)
            )

        return identity - a * K + b * (K @ K)

    def error_func(this, values, H):
        # Read the current coefficients.
        key = this.keys()[0]
        coefficients = values.atVector(key).reshape(6, 4)

        extrinsic = coefficients @ basis
        rotvec = extrinsic[:3]
        translation = extrinsic[3:]

        rotation = Rotation.from_rotvec(rotvec).as_matrix()

        # Relative transform: T_mean^{-1} * T_boundary.
        relative_rotation = mean_rotation_T @ rotation
        d = mean_rotation_T @ (
            translation - mean_translation
        )

        omega = Rotation.from_matrix(
            relative_rotation
        ).as_rotvec()

        # Inverse SO(3) left Jacobian:
        # A = I - W / 2 + beta * W^2.
        W = skew(omega)
        W2 = W @ W
        angle2 = float(omega @ omega)

        # Taylor expansions avoid cancellation near zero.
        if angle2 < 1e-2:
            beta = (
                1.0 / 12.0
                + angle2 / 720.0
                + angle2**2 / 30240.0
                + angle2**3 / 1209600.0
            )
            beta_gradient_scale = (
                1.0 / 360.0
                + angle2 / 7560.0
                + angle2**2 / 201600.0
            )
        else:
            angle = np.sqrt(angle2)
            cot_half = 1.0 / np.tan(0.5 * angle)

            beta = (
                1.0 - 0.5 * angle * cot_half
            ) / angle2

            beta_gradient_scale = (
                -2.0 / angle2**2
                + (1.0 + cot_half**2) / (4.0 * angle2)
                + cot_half / (2.0 * angle2 * angle)
            )

        A = identity - 0.5 * W + beta * W2

        # Full SE(3) Log residual: rotation first.
        residual = np.concatenate([omega, A @ d])

        if H is not None:
            # Derivative of omega with respect to boundary rotvec.
            J_rotation = A.T @ right_jacobian(rotvec)

            # Derivative of A(omega) @ d with respect to omega.
            J_coupling = (
                0.5 * skew(d)
                - beta * (
                    skew(W @ d) + W @ skew(d)
                )
                + np.outer(
                    W2 @ d,
                    beta_gradient_scale * omega,
                )
            )

            # Derivative with respect to [rotvec, translation].
            J_extrinsic = np.zeros((6, 6))
            J_extrinsic[:3, :3] = J_rotation
            J_extrinsic[3:, :3] = J_coupling @ J_rotation
            J_extrinsic[3:, 3:] = A @ mean_rotation_T

            # Derivative with respect to the 24 spline coefficients.
            H[0] = np.asfortranarray(
                J_extrinsic @ jacobian_spline
            )

        return residual

    return error_func