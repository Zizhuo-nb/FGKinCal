import gtsam
import numpy as np
from factor_graph.core.icp_factor import icp_error_func
from factor_graph.core.cubic_spline_factor import (icp_error_func_cubic_spline,
                                                   c0_between_error_func,
                                                   c1_between_error_func,
                                                   c2_between_error_func)
from factor_graph.core.smooth_factor import smooth_error_func
from factor_graph.tool.tool import (transform_points_with_cubic_spline,
                                    icp_error_cubic_spline_jacobian,
                                    c0_between_factor_cubic_spline_jacobian,
                                    c1_between_factor_cubic_spline_jacobian,
                                    c2_between_factor_cubic_spline_jacobian)

def gtsam_optimize_sliding_icp_cubic_spline(
    window_buffer,
    noise_model
):
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # =========================================================
    # 1. 为每个窗口建立24维增量节点和ICP单因子
    # =========================================================
    for item in window_buffer:
        key = gtsam.symbol("x", item["window"])

        # 当前p2已经应用了完整样条修正
        # 本轮GTSAM只估计新增量
        delta_coeff_init = np.zeros(24)

        initial_values.insert(
            key,
            delta_coeff_init
        )

        icp_noise = gtsam.noiseModel.Isotropic.Sigma(
            item["p1"].shape[0],
            noise_model["icp_sigma"]
        )

        graph.add(
            gtsam.CustomFactor(
                icp_noise,
                [key],
                icp_error_func_cubic_spline(
                    item["p1"],
                    item["p2"],
                    item["n"],
                    item["timeL"]
                )
            )
        )

    # =========================================================
    # 2. 为相邻窗口加入C0、C1、C2连续因子
    # =========================================================
    continuity_noise = gtsam.noiseModel.Diagonal.Sigmas(
        noise_model["smooth_sigmas"]
    )

    for k in range(1, len(window_buffer)):
        item_left = window_buffer[k - 1]
        item_right = window_buffer[k]

        key_left = gtsam.symbol(
            "x",
            item_left["window"]
        )

        key_right = gtsam.symbol(
            "x",
            item_right["window"]
        )

        # 左侧样条段的结束局部时间
        # 注意使用完整窗口的timeL_all，而不是匹配后的timeL
        timeL_end = float(
            np.max(item_left["timeL_all"])
        )

        # C0连续
        graph.add(
            gtsam.CustomFactor(
                continuity_noise,
                [key_left, key_right],
                c0_between_error_func(
                    item_left["spline_coefficients"],
                    item_right["spline_coefficients"],
                    timeL_end
                )
            )
        )

        # C1连续
        graph.add(
            gtsam.CustomFactor(
                continuity_noise,
                [key_left, key_right],
                c1_between_error_func(
                    item_left["spline_coefficients"],
                    item_right["spline_coefficients"],
                    timeL_end
                )
            )
        )

        # C2连续
        graph.add(
            gtsam.CustomFactor(
                continuity_noise,
                [key_left, key_right],
                c2_between_error_func(
                    item_left["spline_coefficients"],
                    item_right["spline_coefficients"],
                    timeL_end
                )
            )
        )

    # =========================================================
    # 3. 联合优化
    # =========================================================
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values
    )

    result = optimizer.optimize()

    # =========================================================
    # 4. 读取每个窗口的24维新增量
    # =========================================================
    for item in window_buffer:
        key = gtsam.symbol("x", item["window"])

        delta_coeff_opt = result.atVector(key)

        item["delta_coeff_opt"] = (
            delta_coeff_opt.copy()
        )

    return window_buffer, result