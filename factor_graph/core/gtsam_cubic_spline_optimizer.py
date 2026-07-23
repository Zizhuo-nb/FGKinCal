import gtsam
import numpy as np

from factor_graph.core.cubic_factor import icp_error_func


# def gtsam_optimize_single_cubic_icp(
#         matching,
#         pcr_idx,
#         time_r,
#         R_NB_r,
#         t_NB_r,
#         window_start,
#         window_duration,
#         icp_sigma=0.01
# ):
#     graph = gtsam.NonlinearFactorGraph()
#     initial_values = gtsam.Values()

#     # 一个窗口，一个24维样条系数节点
#     key = gtsam.symbol("c", 0)

#     coefficients_init = np.zeros(24)
#     initial_values.insert(key, coefficients_init)

#     # GTSAM要求的误差函数接口
#     def error_func(this, values, H):
#         coefficients = values.atVector(key)

#         residual, jacobian = icp_error_func(
#             coefficients,
#             matching,
#             pcr_idx,
#             time_r,
#             R_NB_r,
#             t_NB_r,
#             window_start,
#             window_duration
#         )

#         if H is not None:
#             H[0] = jacobian

#         return residual

#     noise = gtsam.noiseModel.Isotropic.Sigma(
#         matching.shape[0],
#         icp_sigma
#     )

#     graph.add(
#         gtsam.CustomFactor(
#             noise,
#             [key],
#             error_func
#         )
#     )

#     optimizer = gtsam.LevenbergMarquardtOptimizer(
#         graph,
#         initial_values
#     )

#     result = optimizer.optimize()

#     coefficients_opt = result.atVector(key)

#     residual_initial, _ = icp_error_func(
#         coefficients_init,
#         matching,
#         pcr_idx,
#         time_r,
#         R_NB_r,
#         t_NB_r,
#         window_start,
#         window_duration
#     )

#     residual_opt, _ = icp_error_func(
#         coefficients_opt,
#         matching,
#         pcr_idx,
#         time_r,
#         R_NB_r,
#         t_NB_r,
#         window_start,
#         window_duration
#     )

#     print("initial RMSE:", np.sqrt(np.mean(residual_initial ** 2)))
#     print("optimized RMSE:", np.sqrt(np.mean(residual_opt ** 2)))
#     print("optimized coefficients:")
#     print(coefficients_opt.reshape(6, 4))

#     return coefficients_opt, result


def gtsam_optimize_single_cubic_icp(
        matching,
        pcr_idx,
        time_r,
        R_NB_r,
        t_NB_r,
        window_start,
        window_duration,
        coefficients_init,
        icp_sigma=0.01
):
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    key = gtsam.symbol("c", 0)

    coefficients_init = np.asarray(
        coefficients_init,
        dtype=float
    ).reshape(24)

    initial_values.insert(key, coefficients_init)

    def error_func(this, values, H):
        coefficients = values.atVector(key)

        residual, jacobian = icp_error_func(
            coefficients,
            matching,
            pcr_idx,
            time_r,
            R_NB_r,
            t_NB_r,
            window_start,
            window_duration
        )

        if H is not None:
            H[0] = jacobian

        return residual

    noise = gtsam.noiseModel.Isotropic.Sigma(
        matching.shape[0],
        icp_sigma
    )

    graph.add(
        gtsam.CustomFactor(
            noise,
            [key],
            error_func
        )
    )

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values
    )

    result = optimizer.optimize()
    coefficients_opt = result.atVector(key)

    residual_opt, _ = icp_error_func(
        coefficients_opt,
        matching,
        pcr_idx,
        time_r,
        R_NB_r,
        t_NB_r,
        window_start,
        window_duration
    )

    rmse = np.sqrt(np.mean(residual_opt**2))

    return coefficients_opt, rmse