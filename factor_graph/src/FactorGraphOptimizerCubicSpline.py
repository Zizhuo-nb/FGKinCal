import os

import CSF
import numpy as np
from scipy.spatial.transform import Rotation

from factor_graph.core.cubic_factor import CubicIcpFactor
from factor_graph.core.gtsam_cubic_spline_optimizer import (
    gtsam_optimize_single_cubic_icp,
)
from src.base.base import RotmatX, RotmatY, RotmatZ
from src.config.sICPconfig import sICPconfig
from src.core.KinematicCalibration import KinematicCalibration
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing

#===========检查采样点分布以及z形状===============

from factor_graph.tool.testing_tool import plot_sampled_points_time_z,plot_full_z_and_max_z_by_timestamp,detect_plant_time_interval


#===============================================


class FactorGraphOptimizerCubicSpline:
    """Optimize cubic-spline calibration parameters for one point-cloud window."""

    WINDOW_INDEX = 14
    OUTPUT_WINDOW_ID = WINDOW_INDEX

    MAX_OUTER_ITERATIONS = 50
    OUTER_RMSE_THRESHOLD = 1e-5
    ICP_SIGMA = 0.01

    def __init__(
        self,
        parent_dir,
        output_dir,
        calibration_dir,
        configfile,
        plot_id,
        date,
    ):
        self.parent_dir = parent_dir
        self.output_dir = output_dir
        self.calibration_dir = calibration_dir
        self.configfile = configfile
        self.plot_id = plot_id
        self.date = date
        self.config = sICPconfig()

    def run(self):
        """Run ground separation, iterative matching, and spline optimization."""
        kin_cal = KinematicCalibration(
            self.parent_dir,
            self.output_dir,
            self.calibration_dir,
            self.configfile,
        )

        kin_cal.copy_data(self.plot_id, self.date)
        kin_cal.print_info()
        kin_cal.loadconfig()
        kin_cal.loadcalibration()
        kin_cal.loaddata()

        idx_left, idx_right = kin_cal.get_alignment_intervals()

        (
            pc_left,
            pc_right,
            time_right,
            rotation_right,
            translation_right,
        ) = self.window_data(
            self.WINDOW_INDEX,
            kin_cal,
            idx_left,
            idx_right,
        )

        print(f"pc_left:          {pc_left.shape}")
        print(f"pc_right:         {pc_right.shape}")
        print(f"time_right:       {time_right.shape}")
        print(f"rotation_right:   {rotation_right.shape}")
        print(f"translation_right:{translation_right.shape}")

        time_right = np.asarray(time_right).reshape(-1)
        window_start = np.min(time_right)
        window_duration = np.max(time_right) - window_start

        if window_duration <= 0:
            raise ValueError("Window duration must be positive.")

        coefficients = np.zeros(24, dtype=np.float64)
        #=============================CSF================================
        # Separate non-ground and ground points using CSF.
        left_non_ground_idx, left_ground_idx = get_non_ground_indices(
            pc_left,
            name="left",
        )
        right_non_ground_idx, right_ground_idx = get_non_ground_indices(
            pc_right,
            name="right",
        )

        pc_left_non_ground = pc_left[left_non_ground_idx]
        pc_left_ground = pc_left[left_ground_idx]
        #================================================================
        # # ============================= CSF: ground only =============================

        # _, left_ground_idx = get_non_ground_indices(
        #     pc_left,
        #     name="left",
        # )

        # _, right_ground_idx = get_non_ground_indices(
        #     pc_right,
        #     name="right",
        # )

        # # Only the left ground cloud is fixed.
        # pc_left_ground = pc_left[left_ground_idx]

        # print(
        #     f"Ground-only optimization: "
        #     f"left_ground={len(left_ground_idx)}, "
        #     f"right_ground={len(right_ground_idx)}"
        # )

        # # ============================================================================

        # # ============================= CSF: plant only =============================

        # left_non_ground_idx, _ = get_non_ground_indices(
        #     pc_left,
        #     name="left",
        # )

        # right_non_ground_idx, _ = get_non_ground_indices(
        #     pc_right,
        #     name="right",
        # )

        # # Only the left plant / non-ground cloud is fixed.
        # pc_left_non_ground = pc_left[left_non_ground_idx]

        # print(
        #     f"Plant-only optimization: "
        #     f"left_non_ground={len(left_non_ground_idx)}, "
        #     f"right_non_ground={len(right_non_ground_idx)}"
        # )

        # # ===========================================================================

        previous_rmse = None

        for outer_iteration in range(self.MAX_OUTER_ITERATIONS):
            # Apply the current spline correction to the complete right cloud.
            pc_right_corrected = correct_full_right_cloud(
                pc_r=pc_right,
                time_r=time_right,
                R_NB_r=rotation_right,
                t_NB_r=translation_right,
                coefficients=coefficients,
                window_start=window_start,
                window_duration=window_duration,
            )
            #=====================CSF=======================
            # Keep ground and non-ground matching independent.
            pc_right_non_ground = pc_right_corrected[right_non_ground_idx]
            pc_right_ground = pc_right_corrected[right_ground_idx]

            non_ground_icp = CubicIcpFactor(
                pc_left_non_ground,
                pc_right_non_ground,
            )
            ground_icp = CubicIcpFactor(
                pc_left_ground,
                pc_right_ground,
            )

            matching_non_ground, filtered_non_ground_idx = non_ground_icp.matching(
                self.config
            )
            matching_ground, filtered_ground_idx = ground_icp.matching(
                self.config
            )

            # Convert filtered local indices back to original right-cloud indices.
            right_match_idx = right_non_ground_idx[filtered_non_ground_idx]

            # The optimizer requires the original static right points.
            matching_non_ground_opt = matching_non_ground.copy()
            matching_non_ground_opt[:, 3:6] = pc_right[right_match_idx]
            

            if len(matching_ground) > 0:
                right_ground_match_idx = right_ground_idx[filtered_ground_idx]

                matching_ground_opt = matching_ground.copy()
                matching_ground_opt[:, 3:6] = pc_right[right_ground_match_idx]

                matching_all = np.concatenate(
                    [matching_non_ground_opt, matching_ground_opt],
                    axis=0,
                )
                right_match_idx_all = np.concatenate(
                    [right_match_idx, right_ground_match_idx]
                )
            else:
                matching_all = matching_non_ground_opt
                right_match_idx_all = right_match_idx
            #=====================CSF=======================

            # # ============================= Ground-only matching =========================

            # # Extract the corrected right ground cloud.
            # pc_right_ground = pc_right_corrected[right_ground_idx]

            # ground_icp = CubicIcpFactor(
            #     pc_left_ground,
            #     pc_right_ground,
            # )

            # matching_ground, filtered_ground_idx = ground_icp.matching(
            #     self.config
            # )

            # if len(matching_ground) == 0:
            #     raise RuntimeError(
            #         f"No ground matches found at outer iteration "
            #         f"{outer_iteration + 1}."
            #     )

            # # matching() returns indices relative to pc_right_ground.
            # # Convert them back to indices of the original complete right cloud.
            # right_ground_match_idx = right_ground_idx[
            #     filtered_ground_idx
            # ]

            # # The optimizer needs the original, uncorrected right points.
            # matching_all = matching_ground.copy()
            # matching_all[:, 3:6] = pc_right[
            #     right_ground_match_idx
            # ]

            # # Keep this variable name because the optimizer and visualization use it.
            # right_match_idx_all = right_ground_match_idx

            # # ============================================================================

            # # ========================== Plant-only matching ============================

            # # Extract the corrected right non-ground cloud.
            # pc_right_non_ground = pc_right_corrected[right_non_ground_idx]

            # non_ground_icp = CubicIcpFactor(
            #     pc_left_non_ground,
            #     pc_right_non_ground,
            # )

            # matching_non_ground, filtered_non_ground_idx = non_ground_icp.matching(
            #     self.config
            # )

            # if len(matching_non_ground) == 0:
            #     raise RuntimeError(
            #         f"No non-ground matches found at outer iteration "
            #         f"{outer_iteration + 1}."
            #     )

            # # Convert local non-ground indices back to the original full right cloud.
            # right_non_ground_match_idx = right_non_ground_idx[
            #     filtered_non_ground_idx
            # ]

            # # The optimizer needs the original static right points.
            # matching_all = matching_non_ground.copy()
            # matching_all[:, 3:6] = pc_right[
            #     right_non_ground_match_idx
            # ]

            # # Keep the same variable name for optimizer / visualization.
            # right_match_idx_all = right_non_ground_match_idx

            # # ===========================================================================

            # #==================================No CSF==========================
            # # Match the complete left and right point clouds without CSF.
            # full_cloud_icp = CubicIcpFactor(
            #     pc_left,
            #     pc_right_corrected,
            # )

            # matching_all, right_match_idx_all = full_cloud_icp.matching(
            #     self.config
            # )

            # # Restore the original static right points for optimization.
            # matching_all = matching_all.copy()
            # matching_all[:, 3:6] = pc_right[right_match_idx_all]
            # #==================================No CSF==========================
            

            coefficients, rmse = gtsam_optimize_single_cubic_icp(
                matching=matching_all,
                pcr_idx=right_match_idx_all,
                time_r=time_right,
                R_NB_r=rotation_right,
                t_NB_r=translation_right,
                window_start=window_start,
                window_duration=window_duration,
                coefficients_init=coefficients,
                icp_sigma=self.ICP_SIGMA,
            )
            #print info for csf
            print(
                f"Outer iteration {outer_iteration + 1}: "
                f"non_ground_matches={len(matching_non_ground)}, "
                f"ground_matches={len(matching_ground)}, "
                f"total_matches={len(matching_all)}, "
                f"RMSE={rmse:.8f}"
            )
            #print info for no csf
            # print(
            #     f"Outer iteration {outer_iteration + 1}: "
            #     f"full_cloud_matches={len(matching_all)}, "
            #     f"RMSE={rmse:.8f}"
            # )

            if previous_rmse is not None:
                rmse_change = abs(previous_rmse - rmse)

                if rmse_change < self.OUTER_RMSE_THRESHOLD:
                    print("Outer loop converged.")
                    break

            previous_rmse = rmse
        #=================打印检查z轴以及采样点分布=======================
        # Recompute the point cloud using the final spline coefficients.
        final_pc_right_corrected = correct_full_right_cloud(
            pc_r=pc_right,
            time_r=time_right,
            R_NB_r=rotation_right,
            t_NB_r=translation_right,
            coefficients=coefficients,
            window_start=window_start,
            window_duration=window_duration,
        )
        # plot_full_z_and_max_z_by_timestamp(
        #     window_start=window_start,
        #     window_end=window_start + window_duration,
        #     point_cloud=final_pc_right_corrected,
        #     point_times=time_right,
        #     window_id=self.OUTPUT_WINDOW_ID,
        #     save_path=os.path.join(
        #         kin_cal.output_dir,
        #         f"window_{self.OUTPUT_WINDOW_ID}_full_z_max_profile.png",
        #     ),
        #     show=True,
        #     show_all_points=True,
        #     time_bin_size=None,
        # )

        # Final matched right points used by the optimizer.
        final_sampled_points = final_pc_right_corrected[
            right_match_idx_all
        ]

        final_sampled_times = time_right[
            right_match_idx_all
        ]

        plot_sampled_points_time_z(
            window_start=window_start,
            window_end=window_start + window_duration,
            sampled_points=final_sampled_points,
            sampled_times=final_sampled_times,
            window_id=self.OUTPUT_WINDOW_ID,
            save_path=os.path.join(
                kin_cal.output_dir,
                f"window_{self.OUTPUT_WINDOW_ID}_sampling_time_z.png",
            ),
            show=True,
        )



        plant_interval = detect_plant_time_interval(
            point_cloud=pc_right,
            point_times=time_right,
            window_start=window_start,
        )
        if plant_interval is not None:
            plant_start_time = plant_interval[
                "plant_start_time"
            ]

            plant_end_time = plant_interval[
                "plant_end_time"
            ]
        #===============================================================
        save_single_window_pointclouds(
            output_dir=kin_cal.output_dir,
            window_id=self.OUTPUT_WINDOW_ID,
            pc_l=pc_left,
            pc_r=pc_right,
            time_r=time_right,
            R_NB_r=rotation_right,
            t_NB_r=translation_right,
            coefficients_opt=coefficients,
            window_start=window_start,
            window_duration=window_duration,
        )

    def window_data(self, window_index, kin_cal, idx_left, idx_right):
        """Create left/right point clouds and per-point right trajectory data."""
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

        # Generate statically calibrated point clouds.
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

        # Expand each trajectory state to all laser points in its frame.
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


def save_single_window_pointclouds(
    output_dir,
    window_id,
    pc_l,
    pc_r,
    time_r,
    R_NB_r,
    t_NB_r,
    coefficients_opt,
    window_start,
    window_duration,
):
    """Save left, original-right, and corrected-right point clouds."""
    os.makedirs(output_dir, exist_ok=True)

    pc_right_corrected = correct_full_right_cloud(
        pc_r=pc_r,
        time_r=time_r,
        R_NB_r=R_NB_r,
        t_NB_r=t_NB_r,
        coefficients=coefficients_opt,
        window_start=window_start,
        window_duration=window_duration,
    )

    output_files = {
        f"window_{window_id}_left_utm.xyz": pc_l,
        f"window_{window_id}_right_original_utm.xyz": pc_r,
        f"window_{window_id}_right_corrected_utm.xyz": pc_right_corrected,
    }

    for filename, point_cloud in output_files.items():
        np.savetxt(
            os.path.join(output_dir, filename),
            point_cloud,
            fmt="%.6f",
        )

    print("Point clouds saved to:", output_dir)