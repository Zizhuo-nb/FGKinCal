import numpy as np
from src.config.sICPconfig import sICPconfig
from src.core.KinematicCalibration import KinematicCalibration
from factor_graph.core.cubic_factor import CubicIcpFactor
from factor_graph.core.gtsam_cubic_spline_optimizer import (
    gtsam_optimize_single_cubic_icp,
)
from factor_graph.tool.tool import(
    window_data,
    correct_full_right_cloud,
    get_non_ground_indices,
    save_all_window_pointclouds

)
from factor_graph.tool.spline_polt import (
    save_spline_coefficients,
    plot_splines,
)


class FactorGraphOptimizerCubicSplineBatch:
    MAX_OUTER_INTERATIONS = 50
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
        all_windows = {}
       

        for window_index in range(len(idx_right)):
            print(
                f"\n{'=' * 70}\n"
                f"Processing window {window_index + 1}/{len(idx_right)} "
                f"(window_id={window_index})\n"
                f"{'=' * 70}"
            )

            (
                pc_left,
                pc_right,
                time_right,
                rotation_right,
                translation_right,
            ) = window_data(
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

            all_windows[window_index] = {
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

            print("window" , window_index)


        previous_rmse = None
        for outer_iteration in range(self.MAX_OUTER_INTERATIONS):
            print(f"[Macthing all] outer={outer_iteration +1},")
            for window_current_id, data in all_windows.items():
                pc_right_corrected = correct_full_right_cloud(
                    pc_r=data["pc_right"],
                    time_r=data["time_right"],
                    R_NB_r=data["rotation_right"],
                    t_NB_r=data["translation_right"],
                    coefficients=data["coefficients"],
                    window_start=data["window_start"],
                    window_duration=data["window_duration"],
                )
                matching_non_ground, idx_non_ground = self.match_group(
                    data["pc_left_non_ground"],
                    pc_right_corrected[data["right_non_ground_idx"]],
                    data["right_non_ground_idx"],
                    data["pc_right"],
                )

                matching_ground, idx_ground = self.match_group(
                    data["pc_left_ground"],
                    pc_right_corrected[data["right_ground_idx"]],
                    data["right_ground_idx"],
                    data["pc_right"],
                )

                if len(matching_ground) == 0:
                    raise ValueError("Ground point is not enough! Try to adjust the downsample/filtering")
                elif len(matching_non_ground) <10:
                    matching_all = matching_ground
                    idx_all = idx_ground
                    idx_non_ground = np.empty(0, dtype=np.int64)
                    print("#"*10)
                    print("Warning, plant points are not enough, bad results!")
                    print("#"*10)
                else:
                    matching_all = np.concatenate(
                        [matching_non_ground,matching_ground],
                        axis = 0,
                    )
                    idx_all = np.concatenate([idx_non_ground,idx_ground])
                data["matching"] = matching_all
                data["pcr_idx"] = idx_all
                data["pcr_idx_non_ground"] = idx_non_ground
                data["pcr_idx_ground"] = idx_ground

                print(
                    
                    f"matching window = {window_current_id},"
                    f"plant={len(matching_non_ground)},"
                    f"ground={len(matching_ground)},"
                    f"total={len(matching_all)},"
                )
            coefficients_result, rmse =gtsam_optimize_single_cubic_icp(
                windows=all_windows,
                icp_sigma=self.ICP_SIGMA,
            )
            for window_id, coefficients in coefficients_result.items():
                all_windows[window_id]["coefficients"] = coefficients

            print(
                f"Active windows: {list(all_windows.keys())}, "
                f"outer iteration: {outer_iteration + 1}, "
                f"RMSE={rmse:.8f}"
            )
            if (
                previous_rmse is not None
                and abs(previous_rmse - rmse) < self.OUTER_RMSE_THRESHOLD
            ):
                print("Outer loop converged.")
                break

            previous_rmse = rmse

            

        

        save_spline_coefficients(
            all_windows,
            kin_cal.output_dir,
        )
        
        plot_splines(
            all_windows,
            kin_cal.output_dir,
        )

        save_all_window_pointclouds(
            output_dir=kin_cal.output_dir,
            windows=all_windows,
        )























    def match_group(self,pc_left, pc_right_corrected, right_idx, pc_right_original):
        icp = CubicIcpFactor(pc_left,pc_right_corrected)
        matching, filtered_idx = icp.matching(self.config)
        original_idx = right_idx[filtered_idx]
        matching_opt = matching.copy()
        matching_opt[:, 3:6] = pc_right_original[original_idx]

        return matching_opt, original_idx     