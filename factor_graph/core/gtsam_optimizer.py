import gtsam
import numpy as np
from factor_graph.core.icp_factor import icp_error_func
from factor_graph.core.smooth_factor import smooth_error_func


def T_to_pose3(T):
    R = gtsam.Rot3(T[:3,:3])
    t = T[:3, 3]
    return gtsam.Pose3(R, gtsam.Point3(t[0], t[1], t[2]))

def pose3_to_T(pose):
    return pose.matrix()




def gtsam_optimize_sliding_icp(window_buffer, noise_model):
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(
        noise_model["smooth_sigmas"]
    )

    for item in window_buffer:
        key = gtsam.symbol("x", item["window"])

        initial_values.insert(
            key,
            T_to_pose3(item["delta_T_init"])
        )

        icp_noise = gtsam.noiseModel.Isotropic.Sigma(
            item["p1"].shape[0],
            noise_model["icp_sigma"]
        )

        graph.add(
            gtsam.CustomFactor(
                icp_noise,
                [key],
                icp_error_func(item["p1"], item["p2"], item["n"])
            )
        )
    #========================================================================
    #turn this on if you want to add smooth factors between consecutive windows

    # for k in range(1, len(window_buffer)):
    #     item_prev = window_buffer[k - 1]
    #     item_curr = window_buffer[k]

    #     key_prev = gtsam.symbol("x", item_prev["window"])
    #     key_curr = gtsam.symbol("x", item_curr["window"])

    #     graph.add(
    #         gtsam.CustomFactor(
    #             smooth_noise,
    #             [key_prev, key_curr],
    #             smooth_error_func(
    #                 item_prev["Delta_T_base"],
    #                 item_curr["Delta_T_base"]
    #             )
    #         )
    #     )
    #========================================================================

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values
    )

    result = optimizer.optimize()

    for item in window_buffer:
        key = gtsam.symbol("x", item["window"])
        delta_T_opt = pose3_to_T(result.atPose3(key))

        item["delta_T_opt"] = delta_T_opt
        item["delta_T_init"] = delta_T_opt

        item["Delta_T"] = delta_T_opt @ item["Delta_T_base"]

    return window_buffer, result