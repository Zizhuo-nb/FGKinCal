import gtsam
import numpy as np

from factor_graph.core.cubic_factor import icp_error_func,c0_error_func,c1_error_func,c2_error_func



def gtsam_optimize_single_cubic_icp(
    windows,
    icp_sigma=0.01,
    c0_sigma=0.01,
    c1_sigma=0.01,
    c2_sigma=0.01,
):
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    window_ids = sorted(windows.keys())
    keys = {}

    # 1. 每个窗口建立一个24维节点和一个ICP因子
    for window_id in window_ids:
        data = windows[window_id]

        key = gtsam.symbol("c", window_id)
        keys[window_id] = key

        coefficients_init = np.asarray(
            data["coefficients"],
            dtype=float,
        ).reshape(24)

        initial_values.insert(
            key,
            coefficients_init,
        )

        def make_icp_error_func(key_i, data_i):

            def error_func(this, values, H):
                coefficients = values.atVector(key_i)

                residual, jacobian = icp_error_func(
                    coefficients,
                    data_i["matching"],
                    data_i["pcr_idx"],
                    data_i["time_right"],
                    data_i["rotation_right"],
                    data_i["translation_right"],
                    data_i["window_start"],
                    data_i["window_duration"],
                )

                if H is not None:
                    H[0] = jacobian

                return residual

            return error_func

        matching = data["matching"]

        icp_noise = gtsam.noiseModel.Isotropic.Sigma(
            matching.shape[0],
            icp_sigma,
        )

        graph.add(
            gtsam.CustomFactor(
                icp_noise,
                [key],
                make_icp_error_func(key, data),
            )
        )

    # 2. 相邻窗口之间添加C0、C1、C2连续因子
    c0_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        c0_sigma,
    )
    c1_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        c1_sigma,
    )
    c2_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        c2_sigma,
    )

    for i in range(len(window_ids) - 1):
        left_id = window_ids[i]
        right_id = window_ids[i + 1]

        left_key = keys[left_id]
        right_key = keys[right_id]

        duration_left = windows[left_id]["window_duration"]
        duration_right = windows[right_id]["window_duration"]

        graph.add(
            gtsam.CustomFactor(
                c0_noise,
                [left_key, right_key],
                c0_error_func(),
            )
        )

        graph.add(
            gtsam.CustomFactor(
                c1_noise,
                [left_key, right_key],
                c1_error_func(
                    duration_left,
                    duration_right,
                ),
            )
        )

        graph.add(
            gtsam.CustomFactor(
                c2_noise,
                [left_key, right_key],
                c2_error_func(
                    duration_left,
                    duration_right,
                ),
            )
        )

    # 3. 联合优化
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )

    result = optimizer.optimize()

    # 4. 取出每个窗口的系数
    coefficients_result = {}

    total_squared_error = 0.0
    total_residual_count = 0

    for window_id in window_ids:
        key = keys[window_id]
        data = windows[window_id]

        coefficients_opt = result.atVector(key)

        coefficients_result[window_id] = coefficients_opt

        residual_opt, _ = icp_error_func(
            coefficients_opt,
            data["matching"],
            data["pcr_idx"],
            data["time_right"],
            data["rotation_right"],
            data["translation_right"],
            data["window_start"],
            data["window_duration"],
        )

        total_squared_error += np.sum(
            residual_opt**2
        )
    
        total_residual_count += len(residual_opt)

    rmse = np.sqrt(
        total_squared_error / total_residual_count
    )

    return coefficients_result, rmse