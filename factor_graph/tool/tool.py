import numpy as np
import os
import CSF
from scipy.spatial.transform import Rotation
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from src.base.base import RotmatX, RotmatY, RotmatZ




def window_data( window_index, kin_cal, idx_left, idx_right):
        """Create left/right point clouds and per-point right trajectory data."""

        # --------------------------------------------------------------
        # Select trajectory states and laser frames belonging to window
        # --------------------------------------------------------------
        left_indices = np.arange(
            idx_left[window_index][0],
            idx_left[window_index][1],
        )
        right_indices = np.arange(
            idx_right[window_index][0],
            idx_right[window_index][1],
        )

        trajectory_left = kin_cal.TL.crop_by_index(left_indices)
        laser_left = kin_cal.lmidataL.crop_by_index(left_indices)

        trajectory_right = kin_cal.TR.crop_by_index(right_indices)
        laser_right = kin_cal.lmidataR.crop_by_index(right_indices)

        # --------------------------------------------------------------
        # Direct georeferencing with the original static calibration
        # --------------------------------------------------------------
        georef_left = directgeoreferencing(
            trajectory_left,
            laser_left,
            kin_cal.calL,
        )
        point_cloud_left = georef_left.run(calibration="static")

        georef_right = directgeoreferencing(
            trajectory_right,
            laser_right,
            kin_cal.calR,
        )
        point_cloud_right = georef_right.run(calibration="static")

        pc_left = point_cloud_left.xyz
        pc_right = point_cloud_right.xyz

        time_right_list = []
        rotation_right_list = []
        translation_right_list = []

        # --------------------------------------------------------------
        # Expand each trajectory state to all laser points in its frame
        # --------------------------------------------------------------
        for frame_index, frame in enumerate(laser_right.frames):
            num_points = frame.M
            state = trajectory_right.statesall[frame_index]

            rotation_nb = (
                RotmatZ(state[9])
                @ RotmatY(state[8])
                @ RotmatX(state[7])
            )
            translation_nb = state[1:4]

            time_right_list.append(
                np.full(
                    (num_points, 1),
                    laser_right.timestamps[frame_index],
                )
            )
            rotation_right_list.append(
                np.repeat(
                    rotation_nb[None, :, :],
                    num_points,
                    axis=0,
                )
            )
            translation_right_list.append(
                np.repeat(
                    translation_nb[None, :],
                    num_points,
                    axis=0,
                )
            )

        time_right = np.vstack(time_right_list)
        rotation_right = np.concatenate(rotation_right_list, axis=0)
        translation_right = np.vstack(translation_right_list)

        return (
            pc_left,
            pc_right,
            time_right,
            rotation_right,
            translation_right,
        )





def get_non_ground_indices(points, name="cloud"):
    """Return CSF non-ground indices and downsampled ground indices."""

    points = np.asarray(points, dtype=np.float64)

    # ------------------------------------------------------------------
    # EXPERIMENT HOOK: CSF ground-segmentation parameters
    # ------------------------------------------------------------------
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.01
    csf.params.class_threshold = 0.05
    csf.params.rigidness = 1
    csf.setPointCloud(points)

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_idx = np.asarray(ground, dtype=np.int64)
    non_ground_idx = np.asarray(non_ground, dtype=np.int64)

    # ------------------------------------------------------------------
    # EXPERIMENT HOOK: ground-point downsampling
    # ------------------------------------------------------------------
    raw_ground_count = len(ground_idx)
    ground_idx = voxel_downsample_indices(
        points,
        ground_idx,
        voxel_size=0.007,
    )

    print(
        f"[CSF] {name}: "
        f"total={len(points)}, "
        f"ground_raw={raw_ground_count}, "
        f"ground_downsampled={len(ground_idx)}, "
        f"non_ground={len(non_ground_idx)}, "
        f"non_ground_ratio={100 * len(non_ground_idx) / len(points):.2f}%"
    )

    return non_ground_idx, ground_idx


def voxel_downsample_indices(points, indices, voxel_size=0.05):
    """Downsample selected points while preserving original point indices."""

    selected_points = points[indices]
    voxel_coordinates = np.floor(
        selected_points / voxel_size
    ).astype(np.int64)

    _, unique_local_indices = np.unique(
        voxel_coordinates,
        axis=0,
        return_index=True,
    )

    return indices[unique_local_indices]


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






