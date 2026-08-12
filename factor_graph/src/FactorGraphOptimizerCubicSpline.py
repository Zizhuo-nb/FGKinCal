import os

import CSF
import numpy as np
from scipy.spatial.transform import Rotation
from factor_graph.tool.spline_polt import (
    save_spline_coefficients,
    plot_splines,
)
from factor_graph.core.cubic_factor import CubicIcpFactor
from factor_graph.core.gtsam_cubic_spline_optimizer import (
    gtsam_optimize_single_cubic_icp,
)
from factor_graph.tool.testing_tool import (
    detect_plant_time_interval,
    plot_full_z_and_max_z_by_timestamp,
    plot_sampled_points_time_z,
    plot_sampled_points_time_z_stitched,
)
from src.base.base import RotmatX, RotmatY, RotmatZ
from src.config.sICPconfig import sICPconfig
from src.core.KinematicCalibration import KinematicCalibration
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing


class FactorGraphOptimizerCubicSpline:
    """Run sliding-window cubic-spline calibration for all point-cloud windows."""

    # ------------------------------------------------------------------
    # Outer-loop optimization parameters
    # ------------------------------------------------------------------
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
        """Load data, optimize all sliding windows, and save the latest results."""

        # ==============================================================
        # 1. Load dataset, calibration, configuration, and trajectories
        # ==============================================================
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

        if len(idx_left) != len(idx_right):
            raise ValueError(
                f"Left/right window counts differ: "
                f"{len(idx_left)} vs {len(idx_right)}"
            )

        # ==============================================================
        # 2. Sliding-window state
        #
        # EXPERIMENT PARAMETER:
        # Change MAX_JOINT_WINDOWS to control how many adjacent windows
        # are jointly optimized at one time.
        # ==============================================================
        MAX_JOINT_WINDOWS = 3

        # Windows currently participating in joint optimization.
        active_windows = {}

        # Windows that have left the sliding window. Their coefficients
        # are already final because they will no longer be optimized.
        finalized_windows = {}

        # ==============================================================
        # 3. Process every point-cloud window in temporal order
        # ==============================================================
        for window_index in range(len(idx_right)):
            print(
                f"\n{'=' * 70}\n"
                f"Processing window {window_index + 1}/{len(idx_right)} "
                f"(window_id={window_index})\n"
                f"{'=' * 70}"
            )

            # ----------------------------------------------------------
            # 3.1 Build the complete left/right point clouds and the
            #     per-point trajectory information for the new window
            # ----------------------------------------------------------
            (
                pc_left,
                pc_right,
                time_right,
                rotation_right,
                translation_right,
            ) = self.window_data(
                window_index,
                kin_cal,
                idx_left,
                idx_right,
            )

            print(f"pc_left:           {pc_left.shape}")
            print(f"pc_right:          {pc_right.shape}")
            print(f"time_right:        {time_right.shape}")
            print(f"rotation_right:    {rotation_right.shape}")
            print(f"translation_right: {translation_right.shape}")

            time_right = np.asarray(time_right).reshape(-1)
            window_start = np.min(time_right)
            window_duration = np.max(time_right) - window_start

            if window_duration <= 0:
                raise ValueError("Window duration must be positive.")

            # ----------------------------------------------------------
            # 3.2 Separate plant/non-ground and ground points using CSF
            #
            # EXPERIMENT HOOK:
            # Modify CSF parameters inside get_non_ground_indices().
            # Replace this block when testing another segmentation method.
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # 3.3 Register the new window in the active sliding window
            # ----------------------------------------------------------
            active_windows[window_index] = {
                # Complete point clouds and per-point trajectory data
                "pc_left": pc_left,
                "pc_right": pc_right,
                "time_right": time_right,
                "rotation_right": rotation_right,
                "translation_right": translation_right,
                "window_start": window_start,
                "window_duration": window_duration,

                # Segmented point-cloud data used for matching
                "pc_left_non_ground": pc_left_non_ground,
                "pc_left_ground": pc_left_ground,
                "right_non_ground_idx": right_non_ground_idx,
                "right_ground_idx": right_ground_idx,

                # Current 24-dimensional spline coefficients
                "coefficients": np.zeros(24, dtype=np.float64),
            }

            # ----------------------------------------------------------
            # 3.4 Remove the oldest window when the active buffer is full
            #
            # The removed window keeps the latest coefficients obtained
            # during its final participation in joint optimization.
            # It is stored in memory and exported only after all windows
            # have been processed.
            # ----------------------------------------------------------
            if len(active_windows) > MAX_JOINT_WINDOWS:
                oldest_window_id = next(iter(active_windows))
                finalized_windows[oldest_window_id] = active_windows.pop(
                    oldest_window_id
                )

            # ==========================================================
            # 4. Outer loop: rematch and jointly optimize active windows
            # ==========================================================
            previous_rmse = None

            for outer_iteration in range(self.MAX_OUTER_ITERATIONS):

                # ------------------------------------------------------
                # 4.1 Recompute correspondences for every active window
                # ------------------------------------------------------
                for active_window_id, data in active_windows.items():

                    # Apply the current spline correction before matching.
                    pc_right_corrected = correct_full_right_cloud(
                        pc_r=data["pc_right"],
                        time_r=data["time_right"],
                        R_NB_r=data["rotation_right"],
                        t_NB_r=data["translation_right"],
                        coefficients=data["coefficients"],
                        window_start=data["window_start"],
                        window_duration=data["window_duration"],
                    )

                    # Split the corrected right cloud using the CSF labels
                    # computed once when the window entered the buffer.
                    pc_right_non_ground = pc_right_corrected[
                        data["right_non_ground_idx"]
                    ]
                    pc_right_ground = pc_right_corrected[
                        data["right_ground_idx"]
                    ]

                    # --------------------------------------------------
                    # EXPERIMENT HOOK: SAMPLING / MATCHING STRATEGY
                    #
                    # Modify or replace this section when testing:
                    # - different point sampling methods,
                    # - different correspondence search methods,
                    # - independent plant/ground matching parameters,
                    # - adaptive sampling based on time or geometry.
                    # --------------------------------------------------
                    non_ground_icp = CubicIcpFactor(
                        data["pc_left_non_ground"],
                        pc_right_non_ground,
                    )
                    ground_icp = CubicIcpFactor(
                        data["pc_left_ground"],
                        pc_right_ground,
                    )

                    matching_non_ground, filtered_non_ground_idx = (
                        non_ground_icp.matching(self.config)
                    )
                    matching_ground, filtered_ground_idx = (
                        ground_icp.matching(self.config)
                    )

                    # # Convert non-ground indices from the segmented right
                    # # cloud back to indices of the complete right cloud.
                    # right_match_idx = data["right_non_ground_idx"][
                    #     filtered_non_ground_idx
                    # ]

                    # # The optimizer must receive the original static right
                    # # points, not the already corrected matching points.
                    # matching_non_ground_opt = matching_non_ground.copy()
                    # matching_non_ground_opt[:, 3:6] = data["pc_right"][
                    #     right_match_idx
                    # ]

                    # # Merge plant/non-ground and ground correspondences.
                    # if len(matching_ground) > 0:
                    #     right_ground_match_idx = data["right_ground_idx"][
                    #         filtered_ground_idx
                    #     ]

                    #     matching_ground_opt = matching_ground.copy()
                    #     matching_ground_opt[:, 3:6] = data["pc_right"][
                    #         right_ground_match_idx
                    #     ]

                    #     matching_all = np.concatenate(
                    #         [matching_non_ground_opt, matching_ground_opt],
                    #         axis=0,
                    #     )
                    #     right_match_idx_all = np.concatenate(
                    #         [right_match_idx, right_ground_match_idx]
                    #     )
                    # else:
                    #     matching_all = matching_non_ground_opt
                    #     right_match_idx_all = right_match_idx

                    # # Store the current correspondences for the joint
                    # # optimizer. These values are replaced every outer loop.
                    # data["matching"] = matching_all
                    # data["pcr_idx"] = right_match_idx_all

                    right_match_idx_non_ground = data["right_non_ground_idx"][
                        filtered_non_ground_idx
                    ]

                    matching_non_ground_opt = matching_non_ground.copy()
                    matching_non_ground_opt[:, 3:6] = data["pc_right"][
                        right_match_idx_non_ground
                    ]

                    if len(matching_ground) > 0:
                        right_match_idx_ground = data["right_ground_idx"][
                            filtered_ground_idx
                        ]

                        matching_ground_opt = matching_ground.copy()
                        matching_ground_opt[:, 3:6] = data["pc_right"][
                            right_match_idx_ground
                        ]

                        matching_all = np.concatenate(
                            [matching_non_ground_opt, matching_ground_opt],
                            axis=0,
                        )
                        right_match_idx_all = np.concatenate(
                            [right_match_idx_non_ground, right_match_idx_ground]
                        )
                    else:
                        right_match_idx_ground = np.empty(
                            0,
                            dtype=np.int64,
                        )

                        matching_all = matching_non_ground_opt
                        right_match_idx_all = right_match_idx_non_ground

                    data["matching"] = matching_all
                    data["pcr_idx"] = right_match_idx_all

                    # Save the two groups separately for later plotting
                    data["pcr_idx_non_ground"] = right_match_idx_non_ground
                    data["pcr_idx_ground"] = right_match_idx_ground

                    # Matching statistics for experiment inspection.
                    print(
                        f"[Matching] "
                        f"outer={outer_iteration + 1}, "
                        f"window={active_window_id}, "
                        f"plant={len(matching_non_ground)}, "
                        f"ground={len(matching_ground)}, "
                        f"total={len(matching_all)}"
                    )

                # ------------------------------------------------------
                # 4.2 Jointly optimize all currently active windows
                #
                # The optimizer is expected to create one spline node per
                # active window and continuity factors between neighbors.
                # ------------------------------------------------------
                coefficients_result, rmse = gtsam_optimize_single_cubic_icp(
                    windows=active_windows,
                    icp_sigma=self.ICP_SIGMA,
                )

                # Write the latest coefficients back to every active window.
                for active_window_id in active_windows:
                    active_windows[active_window_id]["coefficients"] = (
                        coefficients_result[active_window_id]
                    )

                print(
                    f"Active windows: {list(active_windows.keys())}, "
                    f"outer iteration: {outer_iteration + 1}, "
                    f"RMSE={rmse:.8f}"
                )

                # ------------------------------------------------------
                # 4.3 Outer-loop convergence check
                # ------------------------------------------------------
                if previous_rmse is not None:
                    rmse_change = abs(previous_rmse - rmse)

                    if rmse_change < self.OUTER_RMSE_THRESHOLD:
                        print("Outer loop converged.")
                        break

                previous_rmse = rmse

            # ==========================================================
            # 5. Per-window diagnostics
            #
            # These diagnostics use the newest coefficients available for
            # the window at this point. They do not control final export.
            # ==========================================================
            current_window = active_windows[window_index]
            coefficients = current_window["coefficients"]
            right_match_idx_all = current_window["pcr_idx"]

            final_pc_right_corrected = correct_full_right_cloud(
                pc_r=pc_right,
                time_r=time_right,
                R_NB_r=rotation_right,
                t_NB_r=translation_right,
                coefficients=coefficients,
                window_start=window_start,
                window_duration=window_duration,
            )

            # Optional diagnostic: complete Z profile for the window.
            # plot_full_z_and_max_z_by_timestamp(
            #     window_start=window_start,
            #     window_end=window_start + window_duration,
            #     point_cloud=final_pc_right_corrected,
            #     point_times=time_right,
            #     window_id=window_index,
            #     save_path=os.path.join(
            #         kin_cal.output_dir,
            #         f"window_{window_index}_full_z_max_profile.png",
            #     ),
            #     show=True,
            #     show_all_points=True,
            #     time_bin_size=None,
            # )

            # Diagnostic: temporal and vertical distribution of the points
            # sampled by the current matching strategy.
            # final_sampled_points = final_pc_right_corrected[
            #     right_match_idx_all
            # ]
            final_sampled_times = time_right[right_match_idx_all]

            # plot_sampled_points_time_z(
            #     window_start=window_start,
            #     window_end=window_start + window_duration,
            #     sampled_points=final_sampled_points,
            #     sampled_times=final_sampled_times,
            #     window_id=window_index,
            #     save_path=os.path.join(
            #         kin_cal.output_dir,
            #         f"window_{window_index}_sampling_time_z.png",
            #     ),
            #     show=False,
            # )

            # Optional diagnostic: detect the plant time interval.
            plant_interval = detect_plant_time_interval(
                point_cloud=pc_right,
                point_times=time_right,
                window_start=window_start,
            )

            if plant_interval is not None:
                plant_start_time = plant_interval["plant_start_time"]
                plant_end_time = plant_interval["plant_end_time"]

        # ==============================================================
        # 6. Final export after all sliding-window updates are complete
        #
        # finalized_windows:
        #   windows that already left the active buffer.
        # active_windows:
        #   the final windows still remaining in the buffer.
        #
        # Combining both dictionaries here guarantees that every window
        # is exported using its newest and final available coefficients.
        # ==============================================================
        all_windows = {}
        all_windows.update(finalized_windows)
        all_windows.update(active_windows)

        #===============================================================
        save_spline_coefficients(
            all_windows,
            kin_cal.output_dir,
        )

        plot_splines(
            all_windows,
            kin_cal.output_dir,
        )
        #===============================================================

        sampling_plot_records = []

        for window_id in sorted(all_windows.keys()):
            data = all_windows[window_id]

            window_start = data["window_start"]
            window_duration = data["window_duration"]
            window_end = window_start + window_duration

            coefficients = data["coefficients"]

            final_pc_right_corrected = correct_full_right_cloud(
                pc_r=data["pc_right"],
                time_r=data["time_right"],
                R_NB_r=data["rotation_right"],
                t_NB_r=data["translation_right"],
                coefficients=coefficients,
                window_start=window_start,
                window_duration=window_duration,
            )

            right_idx_non_ground = data.get(
                "pcr_idx_non_ground",
                np.empty(0, dtype=np.int64),
            )
            right_idx_ground = data.get(
                "pcr_idx_ground",
                np.empty(0, dtype=np.int64),
            )

            plant_points = final_pc_right_corrected[
                right_idx_non_ground
            ]
            plant_times = data["time_right"][
                right_idx_non_ground
            ].reshape(-1)

            ground_points = final_pc_right_corrected[
                right_idx_ground
            ]
            ground_times = data["time_right"][
                right_idx_ground
            ].reshape(-1)

            sampling_plot_records.append(
                {
                    "window_id": window_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "plant_points": plant_points,
                    "plant_times": plant_times,
                    "ground_points": ground_points,
                    "ground_times": ground_times,
                }
            )

        plot_sampled_points_time_z_stitched(
            window_records=sampling_plot_records,
            save_path=os.path.join(
                kin_cal.output_dir,
                "all_windows_sampling_time_z_stitched.png",
            ),
            show=False,
        )

        save_all_window_pointclouds(
            output_dir=kin_cal.output_dir,
            windows=all_windows,
        )

    def window_data(self, window_index, kin_cal, idx_left, idx_right):
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


# ----------------------------------------------------------------------
# Retained legacy interface
#
# This function is intentionally kept for single-window experiments.
# The current sliding-window run() method uses save_all_window_pointclouds()
# for the final combined export.
# ----------------------------------------------------------------------
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
    """Save one window's left, original-right, and corrected-right clouds."""

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