import numpy as np
from src.core.KinematicCalibration import KinematicCalibration
from factor_graph.tool.dataloader import window_data,get_non_ground_indices
from factor_graph.tool.datasaver import correct_full_right_cloud,save_all_window_pointclouds
from factor_graph.core.gtsam_cubic_spline_optimizer import gtsam_optimize_single_cubic_icp
from factor_graph.tool.tool import save_spline_coefficients, plot_splines
from factor_graph.tool.tool import load_mean_spline_coefficients

from factor_graph.core.cubic_factor import CubicIcpFactor





class FactorGraphOptimizerCubicSplineBatch:
    
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
        self.config = kin_cal.config
        idx_left, idx_right = kin_cal.get_alignment_intervals()
        all_windows = {}
        if idx_right[1][0] < idx_right[0][1]: raise ValueError("Window cannot overlapping! Change the step size in config!!")
        
        for window_index in range(len(idx_right)):
            print( f"\n{'=' * 70}\n" f"Processing window {window_index + 1}/{len(idx_right)} " f"(window_id={window_index})\n" f"{'=' * 70}")
            
            (pc_left, pc_right, time_right, rotation_right, translation_right,) = window_data(window_index, kin_cal, idx_left, idx_right)
            
            print(f"pc_left:           {pc_left.shape}")
            print(f"pc_right:          {pc_right.shape}")
            print(f"time_right:        {time_right.shape}")
            print(f"rotation_right:    {rotation_right.shape}")
            print(f"translation_right: {translation_right.shape}")
            
            time_right = np.asarray(time_right).reshape(-1)
            window_start = np.min(time_right)
            window_duration = np.max(time_right) - window_start
            
            if window_duration <= 0: raise ValueError("Window duration must be positive!")
            
            if self.config.segment_use:
                left_non_ground_idx, left_ground_idx = get_non_ground_indices(
                    pc_left,
                    self.config.csf_shreshould,
                    self.config.ground_voxel,
                    name="left")
                right_non_ground_idx, right_ground_idx = get_non_ground_indices(
                    pc_right,
                    self.config.csf_shreshould,
                    self.config.ground_voxel,
                    name="right")
                
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
            else:
                all_windows[window_index] = {
                    # Complete point clouds and per-point trajectory data
                    "pc_left": pc_left,
                    "pc_right": pc_right,
                    "time_right": time_right,
                    "rotation_right": rotation_right,
                    "translation_right": translation_right,
                    "window_start": window_start,
                    "window_duration": window_duration,
                    # Current 24-dimensional spline coefficients
                    "coefficients": np.zeros(24, dtype=np.float64),
                }
                

                print("window" , window_index)
                
        previous_rmse = None
        for outer_iteration in range(self.config.max_iterations):
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
                
                if self.config.segment_use:
                    if (len(data["pc_left_non_ground"]) ==0 or len(data["right_non_ground_idx"]) == 0):
                        matching_non_ground = np.empty((0,6))
                        idx_non_ground = np.empty(0, dtype = np.int64)
                    else:
                        matching_non_ground, idx_non_ground = self.match_group(
                            data["pc_left_non_ground"],
                            pc_right_corrected[data["right_non_ground_idx"]],
                            data["right_non_ground_idx"],
                            data["pc_right"],
                            voxelization_use=self.config.plant_voxelization_use,
                        )
                        
                    if (len(data["pc_left_ground"]) == 0 or len(data["right_ground_idx"]) == 0):
                        raise ValueError("Ground points are not enough!")
                    
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
                        matching_all = np.concatenate([matching_non_ground,matching_ground], axis = 0)
                        idx_all = np.concatenate([idx_non_ground,idx_ground])
                    data["matching"] = matching_all
                    data["pcr_idx"] = idx_all
                    data["pcr_idx_non_ground"] = idx_non_ground
                    data["pcr_idx_ground"] = idx_ground

                    print(
                        f"matching window = {window_current_id+1},"
                        f"plant={len(matching_non_ground)},"
                        f"ground={len(matching_ground)},"
                        f"total={len(matching_all)},")
                else:
                    right_idx_all = np.arange(len(data["pc_right"]))
                    matching_all, idx_all = self.match_group(
                        data["pc_left"],
                        pc_right_corrected,
                        right_idx_all,
                        data["pc_right"],
                    )
                    data["matching"] = matching_all
                    data["pcr_idx"] = idx_all
                    print(
                        f"matching window = {window_current_id+1}, "
                        f"total={len(matching_all)},"
                    )
            
            mean_coefficients,_ = load_mean_spline_coefficients("output/spline_coefficients.csv")
            coefficients_result, rmse =gtsam_optimize_single_cubic_icp(windows=all_windows,boundary_control=self.config.boundary_control,mean_coefficients= mean_coefficients)   
            for window_id, coefficients in coefficients_result.items():
                all_windows[window_id]["coefficients"] = coefficients

            print(
                f"Active windows: {[k+1 for k in all_windows.keys()]}, "
                f"outer iteration: {outer_iteration + 1}, "
                f"RMSE={rmse:.8f}"
            )
            if (
                previous_rmse is not None
                and abs(previous_rmse - rmse) < self.config.convergence_threshold
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def match_group(self,pc_left, pc_right_corrected, right_idx, pc_right_original,voxelization_use=None,):
        icp = CubicIcpFactor(pc_left,pc_right_corrected)
        matching, filtered_idx = icp.matching(self.config,voxelization_use=voxelization_use)
        original_idx = right_idx[filtered_idx]
        matching_opt = matching.copy()
        matching_opt[:, 3:6] = pc_right_original[original_idx]

        return matching_opt, original_idx     