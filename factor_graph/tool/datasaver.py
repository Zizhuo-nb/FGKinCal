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


# def save_all_window_pointclouds(output_dir, windows):
#     """Save all windows as three complete point-cloud files."""

#     os.makedirs(output_dir, exist_ok=True)

#     all_left = []
#     all_right_original = []
#     all_right_corrected = []

#     # Sort by window ID so the output follows the original temporal order.
#     for window_id in sorted(windows.keys()):
#         data = windows[window_id]

#         pc_right_corrected = correct_full_right_cloud(
#             pc_r=data["pc_right"],
#             time_r=data["time_right"],
#             R_NB_r=data["rotation_right"],
#             t_NB_r=data["translation_right"],
#             coefficients=data["coefficients"],
#             window_start=data["window_start"],
#             window_duration=data["window_duration"],
#         )

#         all_left.append(data["pc_left"])
#         all_right_original.append(data["pc_right"])
#         all_right_corrected.append(pc_right_corrected)

#     all_left = np.concatenate(all_left, axis=0)
#     all_right_original = np.concatenate(all_right_original, axis=0)
#     all_right_corrected = np.concatenate(all_right_corrected, axis=0)

#     output_files = {
#         "all_windows_left_utm.xyz": all_left,
#         "all_windows_right_original_utm.xyz": all_right_original,
#         "all_windows_right_corrected_utm.xyz": all_right_corrected,
#     }

#     for filename, point_cloud in output_files.items():
#         np.savetxt(
#             os.path.join(output_dir, filename),
#             point_cloud,
#             fmt="%.6f",
#         )

#     print(
#         f"Saved {len(windows)} windows as complete point clouds to: "
#         f"{output_dir}"
#     )



# def save_all_window_pointclouds(output_dir, windows):
#     """Save plant and ground point clouds separately."""

#     os.makedirs(output_dir, exist_ok=True)

#     left_plant = []
#     left_ground = []

#     right_original_plant = []
#     right_original_ground = []

#     right_corrected_plant = []
#     right_corrected_ground = []

#     for window_id in sorted(windows.keys()):
#         data = windows[window_id]

#         # Final corrected right cloud
#         pc_right_corrected = correct_full_right_cloud(
#             pc_r=data["pc_right"],
#             time_r=data["time_right"],
#             R_NB_r=data["rotation_right"],
#             t_NB_r=data["translation_right"],
#             coefficients=data["coefficients"],
#             window_start=data["window_start"],
#             window_duration=data["window_duration"],
#         )

#         # -----------------------------
#         # Left
#         # -----------------------------
#         left_plant.append(
#             data["pc_left_non_ground"]
#         )

#         left_ground.append(
#             data["pc_left_ground"]
#         )

#         # -----------------------------
#         # Right original
#         # -----------------------------
#         plant_idx = data["right_non_ground_idx"]
#         ground_idx = data["right_ground_idx"]

#         right_original_plant.append(
#             data["pc_right"][plant_idx]
#         )

#         right_original_ground.append(
#             data["pc_right"][ground_idx]
#         )

#         # -----------------------------
#         # Right corrected
#         # -----------------------------
#         right_corrected_plant.append(
#             pc_right_corrected[plant_idx]
#         )

#         right_corrected_ground.append(
#             pc_right_corrected[ground_idx]
#         )

#     # Merge all windows
#     left_plant = np.concatenate(left_plant, axis=0)
#     left_ground = np.concatenate(left_ground, axis=0)

#     right_original_plant = np.concatenate(
#         right_original_plant, axis=0
#     )
#     right_original_ground = np.concatenate(
#         right_original_ground, axis=0
#     )

#     right_corrected_plant = np.concatenate(
#         right_corrected_plant, axis=0
#     )
#     right_corrected_ground = np.concatenate(
#         right_corrected_ground, axis=0
#     )

#     output_files = {
#         "all_windows_left_plant_utm.xyz":
#             left_plant,

#         "all_windows_left_ground_utm.xyz":
#             left_ground,

#         "all_windows_right_original_plant_utm.xyz":
#             right_original_plant,

#         "all_windows_right_original_ground_utm.xyz":
#             right_original_ground,

#         "all_windows_right_corrected_plant_utm.xyz":
#             right_corrected_plant,

#         "all_windows_right_corrected_ground_utm.xyz":
#             right_corrected_ground,
#     }

#     for filename, point_cloud in output_files.items():
#         np.savetxt(
#             os.path.join(output_dir, filename),
#             point_cloud,
#             fmt="%.6f",
#         )

#     print(
#         f"Saved plant and ground point clouds separately to: "
#         f"{output_dir}"
#     )




def correct_full_cloud(
    pc,
    time,
    R_NB,
    t_NB,
    coefficients,
    window_start,
    window_duration,
    side="right",
):
    """
    Apply symmetric time-dependent cubic-spline correction.

    right:
        p' = R_delta @ p + t_delta / 2

    left:
        p' = R_delta.T @ p - t_delta / 2
    """

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    time = np.asarray(time).reshape(-1)

    # Normalize time to [0, 1]
    normalized_time = (time - window_start) / window_duration

    # Cubic spline basis
    basis = np.column_stack(
        (
            np.ones_like(normalized_time),
            normalized_time,
            normalized_time**2,
            normalized_time**3,
        )
    )

    # 24 coefficients -> 6 x 4
    coefficients = np.asarray(coefficients).reshape(6, 4)

    # [rx, ry, rz, tx, ty, tz] for every point
    correction_twist = basis @ coefficients.T

    delta_rotation = Rotation.from_rotvec(
        correction_twist[:, 0:3]
    ).as_matrix()

    delta_translation = correction_twist[:, 3:6]


    # ---------------------------------------------------------
    # Global -> instantaneous body frame
    # ---------------------------------------------------------
    static_body_points = np.einsum(
        "nij,nj->ni",
        R_NB.transpose(0, 2, 1),
        pc - t_NB,
    )


    # ---------------------------------------------------------
    # Symmetric correction
    # ---------------------------------------------------------
    if side == "right":

        corrected_body_points = (
            np.einsum(
                "nij,nj->ni",
                delta_rotation,
                static_body_points,
            )
            + delta_translation / 2.0
        )

    else:  # left

        corrected_body_points = (
            np.einsum(
                "nij,nj->ni",
                delta_rotation.transpose(0, 2, 1),
                static_body_points,
            )
            - delta_translation / 2.0
        )


    # ---------------------------------------------------------
    # Instantaneous body frame -> global
    # ---------------------------------------------------------
    corrected_global_points = (
        np.einsum(
            "nij,nj->ni",
            R_NB,
            corrected_body_points,
        )
        + t_NB
    )

    return corrected_global_points





def save_all_window_pointclouds(output_dir, windows):
    """Save all windows as complete original and corrected point clouds."""

    os.makedirs(output_dir, exist_ok=True)

    all_left_original = []
    all_right_original = []
    all_left_corrected = []
    all_right_corrected = []

    for window_id in sorted(windows.keys()):
        data = windows[window_id]

        pc_left_corrected = correct_full_cloud(
            pc=data["pc_left"],
            time=data["time_left"],
            R_NB=data["rotation_left"],
            t_NB=data["translation_left"],
            coefficients=data["coefficients"],
            window_start=data["window_start"],
            window_duration=data["window_duration"],
            side="left",
        )

        pc_right_corrected = correct_full_cloud(
            pc=data["pc_right"],
            time=data["time_right"],
            R_NB=data["rotation_right"],
            t_NB=data["translation_right"],
            coefficients=data["coefficients"],
            window_start=data["window_start"],
            window_duration=data["window_duration"],
            side="right",
        )

        all_left_original.append(data["pc_left"])
        all_right_original.append(data["pc_right"])
        all_left_corrected.append(pc_left_corrected)
        all_right_corrected.append(pc_right_corrected)

    all_left_original = np.concatenate(all_left_original, axis=0)
    all_right_original = np.concatenate(all_right_original, axis=0)
    all_left_corrected = np.concatenate(all_left_corrected, axis=0)
    all_right_corrected = np.concatenate(all_right_corrected, axis=0)

    output_files = {
        "all_windows_left_original_utm.xyz": all_left_original,
        "all_windows_right_original_utm.xyz": all_right_original,
        "all_windows_left_corrected_utm.xyz": all_left_corrected,
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