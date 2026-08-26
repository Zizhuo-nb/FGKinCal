import os
import numpy as np
from scipy.spatial.transform import Rotation

def correct_full_right_cloud(
    pc_r,
    time_r,
    R_NB_r,
    t_NB_r,
    coefficients,
    window_start,
    window_duration,
):
    """Apply the time-dependent cubic-spline correction to the right cloud."""

    time_r = np.asarray(time_r).reshape(-1)
    normalized_time = (time_r - window_start) / window_duration

    basis = np.column_stack(
        (
            np.ones_like(normalized_time),
            normalized_time,
            normalized_time**2,
            normalized_time**3,
        )
    )

    coefficients = np.asarray(coefficients).reshape(6, 4)
    correction_twist = basis @ coefficients.T

    delta_rotation = Rotation.from_rotvec(
        correction_twist[:, 0:3]
    ).as_matrix()
    delta_translation = correction_twist[:, 3:6]

    # Transform global points into the instantaneous body frame.
    static_body_points = np.einsum(
        "nij,nj->ni",
        R_NB_r.transpose(0, 2, 1),
        pc_r - t_NB_r,
    )

    # Apply the spline correction in the body frame.
    corrected_body_points = (
        np.einsum(
            "nij,nj->ni",
            delta_rotation,
            static_body_points,
        )
        + delta_translation
    )

    # Transform corrected body-frame points back to the global frame.
    return (
        np.einsum(
            "nij,nj->ni",
            R_NB_r,
            corrected_body_points,
        )
        + t_NB_r
    )


def save_all_window_pointclouds(output_dir, windows):
    """Save all windows as three complete point-cloud files."""

    os.makedirs(output_dir, exist_ok=True)

    all_left = []
    all_right_original = []
    all_right_corrected = []

    # Sort by window ID so the output follows the original temporal order.
    for window_id in sorted(windows.keys()):
        data = windows[window_id]

        pc_right_corrected = correct_full_right_cloud(
            pc_r=data["pc_right"],
            time_r=data["time_right"],
            R_NB_r=data["rotation_right"],
            t_NB_r=data["translation_right"],
            coefficients=data["coefficients"],
            window_start=data["window_start"],
            window_duration=data["window_duration"],
        )

        all_left.append(data["pc_left"])
        all_right_original.append(data["pc_right"])
        all_right_corrected.append(pc_right_corrected)

    all_left = np.concatenate(all_left, axis=0)
    all_right_original = np.concatenate(all_right_original, axis=0)
    all_right_corrected = np.concatenate(all_right_corrected, axis=0)

    output_files = {
        "all_windows_left_utm.xyz": all_left,
        "all_windows_right_original_utm.xyz": all_right_original,
        "all_windows_right_corrected_utm.xyz": all_right_corrected,
    }

    for filename, point_cloud in output_files.items():
        np.savetxt(
            os.path.join(output_dir, filename),
            point_cloud,
            fmt="%.6f",
        )

    print(
        f"Saved {len(windows)} windows as complete point clouds to: "
        f"{output_dir}"
    )