from src.core.KinematicCalibration import KinematicCalibration
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from factor_graph.core.cubic_factor import CubicIcpFactor
from src.config.sICPconfig import sICPconfig
from factor_graph.core.gtsam_cubic_spline_optimizer import gtsam_optimize_single_cubic_icp
from factor_graph.core.cubic_factor import icp_error_func
from scipy.spatial.transform import Rotation
from src.base.base import RotmatX, RotmatY, RotmatZ, Rotmat2Euler
import CSF
import os
import numpy as np




class FactorGraphOptimizerCubicSpline:
    def __init__(self,
                 parent_dir,
                 output_dir,
                 calibration_dir,
                 configfile,
                 plot_id,
                 date):
        self.parent_dir = parent_dir
        self.output_dir = output_dir
        self.calibration_dir = calibration_dir
        self.configfile = configfile
        self.plot_id = plot_id
        self.date = date
        self.config = sICPconfig()
       


    def run(self):
        kin_cal = KinematicCalibration(self.parent_dir,
                                       self.output_dir,
                                       self.calibration_dir,
                                       self.configfile)
        
        kin_cal.copy_data(self.plot_id , self.date)
        kin_cal.print_info()
        kin_cal.loadconfig()
        kin_cal.loadcalibration()
        kin_cal.loaddata()

        idxL, idxR = kin_cal.get_alignment_intervals()#same as before, generating windows for "ICP"

        for i in range(len(idxR)):
            pass    
#===============================================one window=========================================
        pc_l, pc_r,time_r,R_NB_r, t_NB_r = self.window_data(2,kin_cal,idxL,idxR)
        print(pc_l.shape)
        print(pc_r.shape)
        print(time_r.shape)
        print(R_NB_r.shape)
        print(t_NB_r.shape)

        # cubic_icp = CubicIcpFactor(pc_l, pc_r)
        # matching,pcr_idx = cubic_icp.matching(self.config)
        # print(matching.shape, "====" , pcr_idx.shape)

        time_r = np.asarray(time_r).reshape(-1)
        window_start = np.min(time_r)
        window_duration = np.max(time_r) - window_start

        if window_duration <0:
             raise ValueError("window duration is not positive")




        coefficients = np.zeros(24)


             
        # 只生成用于ICP的非地面索引
        left_keep_idx = get_non_ground_indices(pc_l, "left")
        right_keep_idx = get_non_ground_indices(pc_r, "right")
        #CLOTH

        pc_l_icp = pc_l[left_keep_idx]

        #CLOTH
        
        previous_rmse = None
        rmse_threshold = 1e-5
        max_outer_iterations = 50

        for outer_iteration in range(max_outer_iterations):

            # 1. 当前样条修正完整右点云
            pc_r_corrected = correct_full_right_cloud(
                pc_r,
                time_r,
                R_NB_r,
                t_NB_r,
                coefficients,
                window_start,
                window_duration
            )

            # 2. 使用修正后的点云重新匹配
            # cubic_icp = CubicIcpFactor(
            #     pc_l,
            #     pc_r_corrected
            # )
            #TIHUAN上面的#
            pc_r_corrected_icp = pc_r_corrected[right_keep_idx]

            cubic_icp = CubicIcpFactor(
                pc_l_icp,
                pc_r_corrected_icp
            )

            #替换上面的

            # matching, pcr_idx = cubic_icp.matching(
            #     self.config
            # )

            ##替换上面的
            matching, pcr_idx_filtered = cubic_icp.matching(self.config)

            pcr_idx = right_keep_idx[pcr_idx_filtered]

            #替换上面的

            # 3. 残差函数需要原始静态右点
            matching_for_opt = matching.copy()
            matching_for_opt[:, 3:6] = pc_r[pcr_idx]

            # 4. 重新优化
            coefficients, rmse = gtsam_optimize_single_cubic_icp(
                matching=matching_for_opt,
                pcr_idx=pcr_idx,
                time_r=time_r,
                R_NB_r=R_NB_r,
                t_NB_r=t_NB_r,
                window_start=window_start,
                window_duration=window_duration,
                coefficients_init=coefficients,
                icp_sigma=0.01
            )

            print(
                f"outer iteration {outer_iteration + 1}: "
                f"matches={matching.shape[0]}, RMSE={rmse:.8f}"
            )

            # 5. RMSE变化足够小则停止
            if previous_rmse is not None:
                rmse_change = abs(previous_rmse - rmse)

                if rmse_change < rmse_threshold:
                    print("Outer loop converged.")
                    break

            previous_rmse = rmse

        save_single_window_pointclouds(
            output_dir=kin_cal.output_dir,
            window_id=1,
            pc_l=pc_l,
            pc_r=pc_r,
            time_r=time_r,
            R_NB_r=R_NB_r,
            t_NB_r=t_NB_r,
            coefficients_opt=coefficients,
            window_start=window_start,
            window_duration=window_duration
        )

#==================================================================================================




    def window_data(self,i,kin_cal,idxL,idxR):
            idxleft= np.arange(idxL[i][0],idxL[i][1])
            idxright= np.arange(idxR[i][0], idxR[i][1])
    
            TLi= kin_cal.TL.crop_by_index(idxleft) #trajectory data
            LMIl_i = kin_cal.lmidataL.crop_by_index(idxleft) #laser data
            TRi= kin_cal.TR.crop_by_index(idxright)
            LMIr_i = kin_cal.lmidataR.crop_by_index(idxright)
    
            # Run point cloud creation
            georefL = directgeoreferencing( TLi, LMIl_i, kin_cal.calL )
            pcl_i = georefL.run( calibration="static" ) 
    
            georefR = directgeoreferencing( TRi, LMIr_i, kin_cal.calR )
            pcr_i = georefR.run( calibration="static" )
    

    
            pc_l = pcl_i.xyz 
            pc_r = pcr_i.xyz #tuple(N,3)
            time_r_list = [] #(N,1)
            R_NB_r_list = [] #(N,3,3)
            t_NB_r_list = [] #(N,3)

            for idx, frame in enumerate(LMIr_i.frames):
                num_points = frame.M

                state = TRi.statesall[idx]

                R_NB = (
                    RotmatZ(state[9])
                    @ RotmatY(state[8])
                    @ RotmatX(state[7])
                )

                t_NB = state[1:4]

                time_r_list.append(
                    np.full((num_points, 1), LMIr_i.timestamps[idx])
                )

                R_NB_r_list.append(
                    np.repeat(R_NB[None, :, :], num_points, axis=0)
                )

                t_NB_r_list.append(
                    np.repeat(t_NB[None, :], num_points, axis=0)
                )

            time_r = np.vstack(time_r_list)
            R_NB_r = np.concatenate(R_NB_r_list, axis=0)
            t_NB_r = np.vstack(t_NB_r_list)

            
            return pc_l, pc_r,time_r, R_NB_r,t_NB_r






def get_non_ground_indices(points, name="cloud"):
    points = np.asarray(points, dtype=np.float64)

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.1
    csf.params.class_threshold = 0.03

    csf.setPointCloud(points)

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_idx = np.asarray(ground, dtype=np.int64)
    non_ground_idx = np.asarray(non_ground, dtype=np.int64)

    print(
        f"[CSF] {name}: total={len(points)}, "
        f"ground={len(ground_idx)}, "
        f"non-ground={len(non_ground_idx)}, "
        f"kept={100 * len(non_ground_idx) / len(points):.2f}%"
    )

    return non_ground_idx





def correct_full_right_cloud(
        pc_r,
        time_r,
        R_NB_r,
        t_NB_r,
        coefficients,
        window_start,
        window_duration
):
    time_r = np.asarray(time_r).reshape(-1)

    u = (time_r - window_start) / window_duration

    basis = np.column_stack((
        np.ones_like(u),
        u,
        u**2,
        u**3
    ))

    coefficients = np.asarray(coefficients).reshape(6, 4)
    xi = basis @ coefficients.T

    delta_R = Rotation.from_rotvec(
        xi[:, 0:3]
    ).as_matrix()

    translation = xi[:, 3:6]

    q_static_body = np.einsum(
        "nij,nj->ni",
        R_NB_r.transpose(0, 2, 1),
        pc_r - t_NB_r
    )

    q_corrected_body = (
        np.einsum(
            "nij,nj->ni",
            delta_R,
            q_static_body
        )
        + translation
    )

    pc_r_corrected = (
        np.einsum(
            "nij,nj->ni",
            R_NB_r,
            q_corrected_body
        )
        + t_NB_r
    )

    return pc_r_corrected











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
        window_duration
):
    os.makedirs(output_dir, exist_ok=True)

    # 使用样条系数修正完整右点云
    pc_r_corrected = correct_full_right_cloud(
        pc_r=pc_r,
        time_r=time_r,
        R_NB_r=R_NB_r,
        t_NB_r=t_NB_r,
        coefficients=coefficients_opt,
        window_start=window_start,
        window_duration=window_duration
    )

    np.savetxt(
        os.path.join(output_dir, f"window_{window_id}_left_utm.xyz"),
        pc_l,
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(output_dir, f"window_{window_id}_right_original_utm.xyz"),
        pc_r,
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(output_dir, f"window_{window_id}_right_corrected_utm.xyz"),
        pc_r_corrected,
        fmt="%.6f"
    )

    print("Point clouds saved to:", output_dir)