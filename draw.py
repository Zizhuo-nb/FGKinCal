import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


SPLINE_CSV = (r"F:\UNIVERSITY_BONN\master_thesis\working_space\FGKinCal\output\spline_coefficients.csv")


def rotmat_x(a):
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])


def rotmat_y(a):
    return np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)]
    ])


def rotmat_z(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0, 0, 1]
    ])


def rotmat_to_euler_xyz(R):
    rx = np.arctan2(R[2, 1], R[2, 2])
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    rz = np.arctan2(R[1, 0], R[0, 0])
    return np.array([rx, ry, rz])


def build_static_right_extrinsic():
    # translation [m]
    tx = 1.259
    ty = 0.656
    tz = 0.585

    # rotation [deg]
    rx = 144.396
    ry = -0.346
    rz = 91.705

    R_BS = (
        rotmat_z(np.deg2rad(rz))
        @ rotmat_y(np.deg2rad(ry))
        @ rotmat_x(np.deg2rad(rx))
    )

    R_SB = R_BS.T

    H_static = np.eye(4)
    H_static[:3, :3] = R_SB
    H_static[:3, 3] = [tx, ty, tz]

    return H_static


def plot_dynamic_extrinsic(csv_path, num_samples=200):
    df = pd.read_csv(csv_path, header=None)

    starts = df.iloc[:, 1].to_numpy(float)
    durations = df.iloc[:, 2].to_numpy(float)
    coefficients = df.iloc[:, 3:27].to_numpy(float)

    order = np.argsort(starts)
    starts = starts[order]
    durations = durations[order]
    coefficients = coefficients[order]

    H_static = build_static_right_extrinsic()

    names = [
        "rx [deg]",
        "ry [deg]",
        "rz [deg]",
        "tx [mm]",
        "ty [mm]",
        "tz [mm]"
    ]

    fig, axes = plt.subplots(
        6, 1,
        figsize=(12, 14),
        sharex=True
    )

    for i in range(len(starts)):
        start = starts[i]
        duration = durations[i]
        end = start + duration

        t = np.linspace(start, end, num_samples)
        u = (t - start) / duration

        basis = np.column_stack([
            np.ones_like(u),
            u,
            u**2,
            u**3
        ])

        coeff = coefficients[i].reshape(6, 4)
        correction = basis @ coeff.T

        dynamic = np.zeros((num_samples, 6))

        for j in range(num_samples):
            rotvec = correction[j, 0:3]        # rad
            translation = correction[j, 3:6]  # m

            H_delta = np.eye(4)
            H_delta[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
            H_delta[:3, 3] = translation

            # 正确的完整动态外参
            H_dynamic = H_delta @ H_static

            R_BS_dynamic = H_dynamic[:3, :3].T
            euler = rotmat_to_euler_xyz(R_BS_dynamic)

            dynamic[j, 0:3] = np.rad2deg(euler)       # deg
            dynamic[j, 3:6] = H_dynamic[:3, 3] * 1000.0  # mm

        for k in range(6):
            axes[k].plot(t, dynamic[:, k], linewidth=1.2)
            axes[k].set_ylabel(names[k])
            axes[k].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("Right Scanner Dynamic Extrinsic", fontsize=14)
    plt.tight_layout()
    plt.savefig(
    "right_dynamic_extrinsic.png",
    dpi=300,
    bbox_inches="tight"
)
    plt.show()


if __name__ == "__main__":
    plot_dynamic_extrinsic(SPLINE_CSV)