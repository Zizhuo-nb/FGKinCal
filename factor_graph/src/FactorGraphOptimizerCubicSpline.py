from factor_graph.core.cubic_spline_factor import icpFactor
from src.core.KinematicCalibration import KinematicCalibration
from src.config.sICPconfig import sICPconfig
from factor_graph.core.prior_factor import prior_error
# from factor_graph.core.smooth_factor import smooth_factor
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from src.base.base import RotmatX, RotmatY, RotmatZ, Rotmat2Euler, Euler2RotMat
# from factor_graph.core.icp_factor import icpFactor
from factor_graph.tool.tool import rigid_transform,transform_points_with_cubic_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import gtsam
from factor_graph.core.gtsam_optimizer import gtsam_optimize_sliding_icp_cubic_spline
import numpy as np


class FactorGraphSpline:
    config: sICPconfig
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
        self.icp_residual = None
        self.Delta_T_prev = np.eye(4)
        self.result = []

        self.window_buffer = []
        self.max_window_size = 5

        self.noise_model = {
            "icp_sigma": 1,
            "prior_sigmas": np.array([1, 1, 1, 1, 1, ]),
            "smooth_sigmas": np.array([1, 1, 1, 1, 1, 1])
        }

        self.config = sICPconfig()

    def run(self):
        kin_cal = KinematicCalibration(self.parent_dir,
                                    self.output_dir,
                                    self.calibration_dir,
                                    self.configfile)
        
        kin_cal.copy_data(self.plot_id , self.date)
        kin_cal.print_info()
        kin_cal.loadconfig()
        self.config = kin_cal.config
        kin_cal.loadcalibration()
        kin_cal.loaddata()

        idxL, idxR = kin_cal.get_alignment_intervals()#same as before, generating windows for "ICP"

        for i in range(len(idxL)):
            print("\n")
            print("============================================================")
            print(f"Running spline ICP window: {i + 1} / {len(idxL)}")
            print("============================================================")

            # 1. 生成当前窗口左右机体系点云、逐点时间
            (
                pc_l,
                pc_r,
                timeL,
                time_center,
                time_start,
                time_end
            ) = self.window_data(
                i=i,
                kin_cal=kin_cal,
                idxL=idxL,
                idxR=idxR
            )

            # 2. 新窗口的样条初值：
            # 使用当前最后一个窗口的优化结果
            if len(self.window_buffer) == 0:
                coeff_init = np.zeros(24)
            else:
                coeff_init = self.window_buffer[-1][
                    "spline_coefficients"
                ].copy()

            # 3. 建立新窗口
            item = {
                "window": i,
                "time": time_center,

                "time_start": time_start,
                "time_end": time_end,

                "pc_l": pc_l,
                "pc_r": pc_r,
                "timeL_all": timeL,

                "spline_coefficients": coeff_init
            }

            self.window_buffer.append(item)

            # 4. 保持窗口最大数量为5
            if len(self.window_buffer) > self.max_window_size:

                # 该窗口已经不会再参与后续优化，可以固定并保存
                finished_item = self.window_buffer.pop(0)

                self.result.append({
                    "window": finished_item["window"],
                    "time": finished_item["time"],
                    "time_start": finished_item["time_start"],
                    "time_end": finished_item["time_end"],
                    "spline_coefficients":
                        finished_item["spline_coefficients"].copy()
                })
            print(
                "active sliding windows:",
                [item["window"] + 1 for item in self.window_buffer]
            )

            print(
                "initial coefficient norm:",
                np.linalg.norm(coeff_init)
            )

            # 5. 对当前缓冲区内所有窗口执行：
            # 更新点云 -> 重新匹配 -> 联合GTSAM优化 -> 直到收敛
            self.window_buffer = self.optimize_buffer_until_converged(
                self.window_buffer
            )

            current_item = self.window_buffer[-1]

            print("------------------------------------------------------------")
            print("window optimization completed:", current_item["window"] + 1)
            print("final matches:", current_item["p1"].shape[0])
            print("final ICP RMSE before:", current_item["icp_rmse_before"])
            print("final ICP RMSE after :", current_item["icp_rmse_after"])
            print("final spline coefficients:")
            print(current_item["spline_coefficients"].reshape(4, 6))
            print("============================================================")

        # 6. 主循环结束后，保存缓冲区内剩余窗口
        for item in self.window_buffer:
            self.result.append({
                "window": item["window"],
                "time": item["time"],
                "time_start": item["time_start"],
                "time_end": item["time_end"],
                "spline_coefficients":
                    item["spline_coefficients"].copy()
            })
        

        self.export_spline_calibration_to_kincal(
            kin_cal
        )

        pcl, pcr = kin_cal.create_pointcloud(
            calibration="kinematic"
        )

        pcl.write_to_file(
            path=kin_cal.output_dir,
            filename="pcl_factor_spline_calibration",
            offset=kin_cal.config.txyz
        )

        pcr.write_to_file(
            path=kin_cal.output_dir,
            filename="pcr_factor_spline_calibration",
            offset=kin_cal.config.txyz
        )

        return self.result


    def window_data(self, i, kin_cal, idxL, idxR):
        idxleft = np.arange(idxL[i][0], idxL[i][1])
        idxright = np.arange(idxR[i][0], idxR[i][1])

        TLi = kin_cal.TL.crop_by_index(idxleft)
        LMIl_i = kin_cal.lmidataL.crop_by_index(idxleft)

        TRi = kin_cal.TR.crop_by_index(idxright)
        LMIr_i = kin_cal.lmidataR.crop_by_index(idxright)

        georefL = directgeoreferencing(
            TLi,
            LMIl_i,
            kin_cal.calL
        )
        pcl_i = georefL.run(calibration="static")

        georefR = directgeoreferencing(
            TRi,
            LMIr_i,
            kin_cal.calR
        )
        pcr_i = georefR.run(calibration="static")

        idxmL = round(
            (idxL[i][0] + idxL[i][1]) / 2
        )

        Tmil = kin_cal.TL.statesall[idxmL, :]
        time_center = float(kin_cal.TL.time[idxmL])

        pc_l = pcl_i.xyz
        pc_r = pcr_i.xyz

        point_time_r = pcr_i.time.reshape(-1)

        time_start = float(point_time_r[0])
        time_end = float(point_time_r[-1])

        # 样条内部使用的局部时间
        timeL = point_time_r - time_start

        xyz_e_left = pc_l - Tmil[1:4]
        xyz_e_right = pc_r - Tmil[1:4]

        R_B_NED = (
            RotmatZ(Tmil[9])
            @ RotmatY(Tmil[8])
            @ RotmatX(Tmil[7])
        )

        pc_l = (R_B_NED.T @ xyz_e_left.T).T
        pc_r = (R_B_NED.T @ xyz_e_right.T).T

        return (
            pc_l,
            pc_r,
            timeL,
            time_center,
            time_start,
            time_end
        )


    
    def optimize_buffer_until_converged(
        self,
        window_buffer,
        max_iterations=20,
        tolerance=1e-6
    ):
        for iteration in range(max_iterations):

            print()
            print(
                f"---- outer ICP iteration "
                f"{iteration + 1} / {max_iterations} ----"
            )

            optimization_buffer = []

            # =====================================================
            # 1. 用当前完整样条更新点云，并重新匹配
            # =====================================================
            for item in window_buffer:

                pc_r_updated = transform_points_with_cubic_spline(
                    item["pc_r"],
                    item["spline_coefficients"],
                    item["timeL_all"]
                )

                icp_factor = icpFactor(
                    item["pc_l"],
                    pc_r_updated
                )

                pc_matching, matched_timeL = icp_factor.matching(
                    self.config,
                    item["timeL_all"]
                )

                if pc_matching is None or pc_matching.shape[0] == 0:
                    raise ValueError(
                        f"Window {item['window'] + 1}: "
                        "no valid ICP matches"
                    )

                p1 = pc_matching[:, 0:3]
                p2 = pc_matching[:, 3:6]
                n = pc_matching[:, 6:9]

                residual_before = np.sum(
                    n * (p2 - p1),
                    axis=1
                )

                rmse_before = np.sqrt(
                    np.mean(residual_before**2)
                )

                print(
                    f"window {item['window'] + 1}: "
                    f"matches = {p1.shape[0]}, "
                    f"RMSE before = {rmse_before:.8f}"
                )

                optimization_buffer.append({
                    "window": item["window"],
                    "p1": p1,
                    "p2": p2,
                    "n": n,
                    "timeL": matched_timeL,
                    "timeL_all": item["timeL_all"],
                    "icp_residual": residual_before,

                    # 连续因子使用当前完整系数
                    "spline_coefficients":
                        item["spline_coefficients"].copy()
                })

            # =====================================================
            # 2. 当前所有窗口一起进行GTSAM优化
            # =====================================================
            optimization_buffer, result = (
                gtsam_optimize_sliding_icp_cubic_spline(
                    optimization_buffer,
                    self.noise_model
                )
            )

            # =====================================================
            # 3. 查看本轮优化结果并累加增量
            # =====================================================
            max_change = 0.0

            for item, optimized_item in zip(
                window_buffer,
                optimization_buffer
            ):
                delta_coeff = optimized_item[
                    "delta_coeff_opt"
                ]

                coefficient_change = np.linalg.norm(
                    delta_coeff
                )

                max_change = max(
                    max_change,
                    coefficient_change
                )

                # 在本轮固定匹配点上应用GTSAM求出的增量
                p2_after = transform_points_with_cubic_spline(
                    optimized_item["p2"],
                    delta_coeff,
                    optimized_item["timeL"]
                )

                residual_after = np.sum(
                    optimized_item["n"]
                    * (p2_after - optimized_item["p1"]),
                    axis=1
                )

                rmse_before = np.sqrt(
                    np.mean(
                        optimized_item["icp_residual"]**2
                    )
                )

                rmse_after = np.sqrt(
                    np.mean(residual_after**2)
                )

                print(
                    f"window {item['window'] + 1}: "
                    f"RMSE {rmse_before:.8f} -> "
                    f"{rmse_after:.8f}, "
                    f"delta norm = {coefficient_change:.8e}"
                )

                # 累加到当前完整样条系数
                item["spline_coefficients"] += delta_coeff

                # 保存检测结果
                item["p1"] = optimized_item["p1"]
                item["p2"] = optimized_item["p2"]
                item["n"] = optimized_item["n"]
                item["matched_timeL"] = optimized_item["timeL"]
                item["icp_residual"] = optimized_item[
                    "icp_residual"
                ]

                item["icp_rmse_before"] = rmse_before
                item["icp_rmse_after"] = rmse_after
                item["delta_coeff_norm"] = coefficient_change

            print(
                f"maximum coefficient change: "
                f"{max_change:.8e}"
            )

            # =====================================================
            # 4. 判断整个滑动窗口是否收敛
            # =====================================================
            if max_change < tolerance:
                print(
                    f"outer ICP converged after "
                    f"{iteration + 1} iterations"
                )
                break

        else:
            print(
                f"outer ICP reached maximum "
                f"{max_iterations} iterations"
            )

        return window_buffer






    def export_spline_calibration_to_kincal(self, kin_cal):
        """
        直接在每个轨迹/profile时间戳上计算优化后的样条。
        不再调用 fill_borders() 和 interpolate_cubic_spline()。
        """

        results = sorted(
            self.result,
            key=lambda item: item["time_start"]
        )

        # ==================================================
        # 静态左外参
        # ==================================================
        R_BS_L = (
            RotmatZ(np.deg2rad(kin_cal.calL.rz))
            @ RotmatY(np.deg2rad(kin_cal.calL.ry))
            @ RotmatX(np.deg2rad(kin_cal.calL.rx))
        )

        H_sbl = kin_cal.create_homogeneous_matrix(
            R_BS_L.T,
            np.array([
                kin_cal.calL.tx,
                kin_cal.calL.ty,
                kin_cal.calL.tz
            ])
        )

        # ==================================================
        # 静态右外参
        # ==================================================
        R_BS_R = (
            RotmatZ(np.deg2rad(kin_cal.calR.rz))
            @ RotmatY(np.deg2rad(kin_cal.calR.ry))
            @ RotmatX(np.deg2rad(kin_cal.calR.rx))
        )

        H_sbr = kin_cal.create_homogeneous_matrix(
            R_BS_R.T,
            np.array([
                kin_cal.calR.tx,
                kin_cal.calR.ty,
                kin_cal.calR.tz
            ])
        )

        # ==================================================
        # 左侧保持静态
        # ==================================================
        kin_cal.kcalL.xint = np.zeros(
            (len(kin_cal.TL.time), 7)
        )

        kin_cal.kcalL.xint[:, 0] = kin_cal.TL.time
        kin_cal.kcalL.xint[:, 1:4] = Rotmat2Euler(
            H_sbl[:3, :3].T
        )
        kin_cal.kcalL.xint[:, 4:7] = H_sbl[:3, 3]

        # ==================================================
        # 右侧直接计算每个时间戳对应的样条外参
        # ==================================================
        kin_cal.kcalR.xint = np.zeros(
            (len(kin_cal.TR.time), 7)
        )

        kin_cal.kcalR.xint[:, 0] = kin_cal.TR.time

        for j, timestamp in enumerate(kin_cal.TR.time):
            timestamp = float(timestamp)

            # 找出该时间对应的样条段
            selected = None

            for item in results:
                if (
                    item["time_start"]
                    <= timestamp
                    <= item["time_end"]
                ):
                    selected = item
                    break

            # 数据开头没有被窗口覆盖：使用第一段起点
            if selected is None and timestamp < results[0]["time_start"]:
                selected = results[0]
                local_time = 0.0

            # 数据结尾没有被窗口覆盖：使用最后一段末端
            elif selected is None:
                selected = results[-1]
                local_time = (
                    selected["time_end"]
                    - selected["time_start"]
                )

            else:
                local_time = (
                    timestamp
                    - selected["time_start"]
                )

            coefficients = selected[
                "spline_coefficients"
            ].reshape(4, 6)

            # 直接计算该时间上的6维修正
            xi = (
                coefficients[0]
                + local_time * coefficients[1]
                + local_time**2 * coefficients[2]
                + local_time**3 * coefficients[3]
            )

            Delta_T = gtsam.Pose3.Expmap(xi).matrix()

            # 当前设计：左侧固定，只修正右侧
            H_sb_newr = Delta_T @ H_sbr

            kin_cal.kcalR.xint[j, 1:4] = Rotmat2Euler(
                H_sb_newr[:3, :3].T
            )

            kin_cal.kcalR.xint[j, 4:7] = (
                H_sb_newr[:3, 3]
            )

        # 保存完整的逐时间外参
        kin_cal.kcalL.x = kin_cal.kcalL.xint.copy()
        kin_cal.kcalR.x = kin_cal.kcalR.xint.copy()

        kin_cal.kcalL.write_to_file(
            path_out=kin_cal.output_dir,
            fname="l"
        )

        kin_cal.kcalR.write_to_file(
            path_out=kin_cal.output_dir,
            fname="r"
        )