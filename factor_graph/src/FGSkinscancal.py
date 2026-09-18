import numpy as np

from scipy.spatial.transform import Rotation
from src.base.base import Rotmat2Euler


from src.core.KinematicCalibration import KinematicCalibration
from factor_graph.tool.dataloader import get_non_ground_indices, window_data
from factor_graph.tool.datasaver import save_all_window_pointclouds,correct_full_cloud,save_all_window_pointclouds
from factor_graph.core.gtsam_cubic_spline_optimizer import gtsam_optimize_single_cubic_icp
from factor_graph.tool.tool import save_spline_coefficients, plot_splines
from factor_graph.core.boundary_control import compute_mean_center_extrinsic
from factor_graph.core.cubic_factor import CubicIcpFactor








class FGSkinscancal:
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
            (pc_left, pc_right, time_right, rotation_right, translation_right,time_left, rotation_left, translation_left) = window_data(window_index, kin_cal, idx_left, idx_right)
            print(f"pc_left:           {pc_left.shape}")
            print(f"time_left:         {time_left.shape}")
            print(f"rotation_left:     {rotation_left.shape}")
            print(f"translation_left:  {translation_left.shape}")
            print(f"pc_right:          {pc_right.shape}")
            print(f"time_right:        {time_right.shape}") 
            print(f"rotation_right:    {rotation_right.shape}")
            print(f"translation_right: {translation_right.shape}")
            
            time_right = np.asarray(time_right).reshape(-1)
            time_left = np.asarray(time_left).reshape(-1)
            # Rwindow_start = np.min(time_right)
            # Rwindow_duration = np.max(time_right) - Rwindow_start
            # Lwindow_start = np.min(time_left)
            # Lwindow_duration = np.max(time_left) - Lwindow_start
            # window_start = min(np.min(time_left), np.min(time_right))
            # window_end = max(np.max(time_left), np.max(time_right))
            window_start = np.min(time_left)
            window_end = np.max(time_left)
            window_duration = window_end - window_start
            
            
            if self.config.segment_use:
                left_non_ground_idx, left_ground_idx = get_non_ground_indices(pc_left, self.config.CSFThreshould,self.config.SloopSmooth,self.config.cloth_resolution,self.config.rigidness)
                right_non_ground_idx, right_ground_idx = get_non_ground_indices(pc_right, self.config.CSFThreshould,self.config.SloopSmooth,self.config.cloth_resolution,self.config.rigidness)
                
                all_windows[window_index] = {
                    "pc_left": pc_left,
                    "pc_right": pc_right,
                    "time_right": time_right,
                    "rotation_right": rotation_right,
                    "translation_right": translation_right,
                    "time_left": time_left,
                    "rotation_left": rotation_left,
                    "translation_left": translation_left,
                    
                    # "Rwindow_start": Rwindow_start,
                    # "Rwindow_duration": Rwindow_duration,
                    # "Lwindow_start": Lwindow_start,
                    # "Lwindow_duration": Lwindow_duration,
                    "window_start": window_start,
                    "window_duration": window_duration,

                    "left_non_ground_idx": left_non_ground_idx,
                    "left_ground_idx": left_ground_idx,
                    "right_non_ground_idx": right_non_ground_idx,
                    "right_ground_idx": right_ground_idx,
                    "coefficients": np.zeros(24, dtype=np.float64),
                }
                
            else:
                all_windows[window_index] = {
                    "pc_left": pc_left,
                    "pc_right": pc_right,
                    "time_right": time_right,
                    "rotation_right": rotation_right,
                    "translation_right": translation_right,
                    "time_left": time_left,
                    "rotation_left": rotation_left,
                    "translation_left": translation_left,
                    # "Rwindow_start": Rwindow_start,
                    # "Rwindow_duration": Rwindow_duration,
                    # "Lwindow_start": Lwindow_start,
                    # "Lwindow_duration": Lwindow_duration,
                    "window_start": window_start,
                    "window_duration": window_duration,
                    "coefficients": np.zeros(24, dtype=np.float64),
                }
                
        previous_rmse = None
        for outer_iteration in range(self.config.max_iterations):
            print(f"[Macthing all] outer={outer_iteration +1},")
            for window_current_id, data in all_windows.items():
                pc_left_corrected = correct_full_cloud(
                    pc=data["pc_left"],
                    time=data["time_left"],
                    R_NB=data["rotation_left"],
                    t_NB=data["translation_left"],
                    coefficients=data["coefficients"],
                    # window_start=data["Lwindow_start"],
                    # window_duration=data["Lwindow_duration"],
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
                    # window_start=data["Rwindow_start"],
                    # window_duration=data["Rwindow_duration"],
                    window_start=data["window_start"],
                    window_duration=data["window_duration"],
                    
                    side="right",
                )
                
                if self.config.segment_use:
                    if (len(data["left_non_ground_idx"]) ==0 or len(data["right_non_ground_idx"]) == 0):
                        matching_non_ground = np.empty((0,9))
                        idx_left_non_ground = np.empty(0, dtype=np.int64)
                        idx_right_non_ground = np.empty(0, dtype=np.int64)
                    else:
                        matching_non_ground, idx_left_non_ground, idx_right_non_ground = self.match_group(
                            pc_left_corrected[data["left_non_ground_idx"]],
                            pc_right_corrected[data["right_non_ground_idx"]],
                            data["left_non_ground_idx"],
                            data["right_non_ground_idx"],
                            data["pc_left"],
                            data["pc_right"],
                            point_type="plant",
                        )
                    
                    
                    if (len(data["left_ground_idx"]) == 0 or len(data["right_ground_idx"]) == 0):
                        raise ValueError("Ground points are not enough!")
                    matching_ground, idx_left_ground, idx_right_ground = self.match_group(
                        pc_left_corrected[data["left_ground_idx"]],
                        pc_right_corrected[data["right_ground_idx"]],
                        data["left_ground_idx"],
                        data["right_ground_idx"],
                        data["pc_left"],
                        data["pc_right"],
                        point_type="ground",
                        disable_filters=(
                            window_current_id == min(all_windows.keys())
                            or window_current_id == max(all_windows.keys())
                        ),
                    )
                    
                    if len(matching_ground) == 0:
                        raise ValueError("Ground point is not enough! Try to adjust the downsample/filtering")
                    elif len(matching_non_ground) <10:
                        matching_all = matching_ground
                        idx_left_all = idx_left_ground
                        idx_right_all = idx_right_ground

                        idx_left_non_ground = np.empty(0, dtype=np.int64)
                        idx_right_non_ground = np.empty(0, dtype=np.int64)
                        print("#"*10)
                        print("Warning, plant points are not enough, bad results!")
                        print("#"*10)  
                    else:
                        matching_all = np.concatenate([matching_non_ground,matching_ground], axis = 0)
                        idx_left_all = np.concatenate([idx_left_non_ground, idx_left_ground])
                        idx_right_all = np.concatenate([idx_right_non_ground, idx_right_ground])
                    
                    data["matching"] = matching_all
                    data["pcl_idx"] = idx_left_all
                    data["pcr_idx"] = idx_right_all
                    data["pcl_idx_non_ground"] = idx_left_non_ground
                    data["pcr_idx_non_ground"] = idx_right_non_ground
                    data["pcl_idx_ground"] = idx_left_ground
                    data["pcr_idx_ground"] = idx_right_ground
                    print(
                        f"matching window = {window_current_id+1},"
                        f"plant={len(matching_non_ground)},"
                        f"ground={len(matching_ground)},"
                        f"total={len(matching_all)},")
                else:
                    left_idx_all = np.arange(len(data["pc_left"]))
                    right_idx_all = np.arange(len(data["pc_right"]))
                    matching_all, idx_left_all, idx_right_all = self.match_group(
                        pc_left_corrected,
                        pc_right_corrected,
                        left_idx_all,
                        right_idx_all,
                        data["pc_left"],
                        data["pc_right"],
                    )
                    data["matching"] = matching_all
                    data["pcl_idx"] = idx_left_all
                    data["pcr_idx"] = idx_right_all
                    print(
                        f"matching window = {window_current_id+1}, "
                        f"total match ={len(matching_all)},"
                    )
            # mean_coefficients,_ = load_mean_spline_coefficients("output/spline_coefficients.csv")
            #=====compte mean center extrinsic windows that I need=============
            mean_head_extrinsic = None
            mean_tail_extrinsic = None
            use_boundary_prior = (self.config.boundary_control and outer_iteration > 0)
            if use_boundary_prior:
                window_ids = sorted(all_windows.keys())
                mean_extrinsic = compute_mean_center_extrinsic(
                    all_windows,
                    window_ids,
                )

                mean_head_extrinsic = mean_extrinsic.copy()
                mean_tail_extrinsic = mean_extrinsic.copy()
                
            #==================================================================
            coefficients_result, rmse =gtsam_optimize_single_cubic_icp(windows=all_windows,
                continuity= self.config.continuity,
                boundary_control=use_boundary_prior,
                mean_head_extrinsic=mean_head_extrinsic,
                mean_tail_extrinsic=mean_tail_extrinsic,
                boundary_rotation_sigma=0.01,
                boundary_translation_sigma=0.01,)   
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
        # #============================================================
        # #消融实验
        # #============================================================
        # spline_center_results = self.get_spline_center_results(
        #     all_windows,
        #     idx_left,
        #     idx_right,
        #     kin_cal,
        # )
        # kin_cal.compute_kinematic_calibration_parameter(
        #     icp_param=spline_center_results
        # )
        
        
        # # # Only keep middle windows for final point-cloud saving
        # # left_keep = np.arange(
        # #     idx_left[1][0],
        # #     idx_left[-2][1]
        # # )

        # # right_keep = np.arange(
        # #     idx_right[1][0],
        # #     idx_right[-2][1]
        # # )

        # # kin_cal.TL = kin_cal.TL.crop_by_index(left_keep)
        # # kin_cal.lmidataL = kin_cal.lmidataL.crop_by_index(left_keep)
        # # kin_cal.kcalL.xint = kin_cal.kcalL.xint[left_keep]

        # # kin_cal.TR = kin_cal.TR.crop_by_index(right_keep)
        # # kin_cal.lmidataR = kin_cal.lmidataR.crop_by_index(right_keep)
        # # kin_cal.kcalR.xint = kin_cal.kcalR.xint[right_keep]
        
        
        # pcl, pcr = kin_cal.create_pointcloud(
        #     calibration="kinematic"
        # )

        # pcl.write_to_file(
        #     path=kin_cal.output_dir,
        #     filename="pc_left_spline_center_interpolated",
        #     offset=kin_cal.config.txyz,
        # )

        # pcr.write_to_file(
        #     path=kin_cal.output_dir,
        #     filename="pc_right_spline_center_interpolated",
        #     offset=kin_cal.config.txyz,
        # )
        
        #=============================================================
        #=============================================================
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
    def match_group(self,
        pc_left_corrected,
        pc_right_corrected,
        left_idx,
        right_idx,
        pc_left_original,
        pc_right_original,
        point_type=None,
        disable_filters=False):
            icp = CubicIcpFactor(pc_left_corrected, pc_right_corrected)
            matching, filtered_left_idx, filtered_right_idx = icp.matching(self.config,point_type=point_type,disable_filters=disable_filters)
            original_left_idx = left_idx[filtered_left_idx]
            original_right_idx = right_idx[filtered_right_idx]

            matching_opt = matching.copy()

            matching_opt[:, 0:3] = pc_left_original[original_left_idx]
            matching_opt[:, 3:6] = pc_right_original[original_right_idx]

            return matching_opt, original_left_idx, original_right_idx
        
        
        
        
    
    
    
    
    
    
    # #===================================================================
    # #消融实验
    # #===================================================================



    # def get_spline_center_results(
    #     self,
    #     all_windows,
    #     idx_left,
    #     idx_right,
    #     kin_cal,
    # ):
    #     results = []

    #     for window_id in sorted(all_windows.keys()):
    #         data = all_windows[window_id]

    #         # 1. 与离散方法完全相同的中心时间戳
    #         idxmL = round(
    #             (idx_left[window_id][0] + idx_left[window_id][1]) / 2
    #         )
    #         timei = kin_cal.TL.time[idxmL]

    #         # 2. 转成 spline 的归一化时间 u
    #         u = (
    #             (timei - data["window_start"])
    #             / data["window_duration"]
    #         )

    #         basis = np.array([1.0, u, u**2, u**3])

    #         # 3. 计算该时刻的 6DoF
    #         coefficients = np.asarray(
    #             data["coefficients"],
    #             dtype=float
    #         ).reshape(6, 4)

    #         xi = coefficients @ basis

    #         # 4. spline 前三维是 rotvec，转成离散方法使用的 Euler
    #         R = Rotation.from_rotvec(xi[:3]).as_matrix()
    #         euler = Rotmat2Euler(R)

    #         # 5. 组成和 Px.txt 一样的 11 列
    #         results.append([
    #             idx_left[window_id][0],
    #             idx_left[window_id][1],
    #             idx_right[window_id][0],
    #             idx_right[window_id][1],

    #             euler[0],
    #             euler[1],
    #             euler[2],

    #             xi[3],
    #             xi[4],
    #             xi[5],

    #             timei,
    #         ])

    #     return np.asarray(results)