import numpy as np
import CSF
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





def get_non_ground_indices(points, config_threshould,ground_voxel,name="cloud"):
    """Return CSF non-ground indices and downsampled ground indices."""

    points = np.asarray(points, dtype=np.float64)

    # ------------------------------------------------------------------
    # EXPERIMENT HOOK: CSF ground-segmentation parameters
    # ------------------------------------------------------------------
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.01
    csf.params.class_threshold = config_threshould
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
        voxel_size=ground_voxel,
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