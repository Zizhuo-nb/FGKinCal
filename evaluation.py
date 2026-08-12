# import numpy as np
# import matplotlib.pyplot as plt


# def analyze_m3c2(file_path):
#     data = np.loadtxt(file_path, comments="/")

#     uncertainty = data[:, 4]
#     d = data[:, 5]   # M3C2_distance

#     # remove NaN / inf
#     valid = np.isfinite(d) & np.isfinite(uncertainty)

#     print("========== M3C2 Statistics ==========")
#     print("file:", file_path)
#     print("total points:", len(d))
#     print("valid points:", np.sum(valid))
#     print("invalid / nan points:", len(d) - np.sum(valid))
#     print()

#     d = d[valid]
#     uncertainty = uncertainty[valid]

#     abs_d = np.abs(d)

#     print("Mean distance:", np.mean(d), "m")
#     print("Std distance :", np.std(d), "m")
#     print("RMS distance :", np.sqrt(np.mean(d ** 2)), "m")
#     print("Mean |dist|  :", np.mean(abs_d), "m")
#     print("Median |dist|:", np.median(abs_d), "m")
#     print("95% |dist|   :", np.percentile(abs_d, 95), "m")
#     print("Max |dist|   :", np.max(abs_d), "m")
#     print()

#     print("Mean distance:", np.mean(d) * 100, "cm")
#     print("Std distance :", np.std(d) * 100, "cm")
#     print("RMS distance :", np.sqrt(np.mean(d ** 2)) * 100, "cm")
#     print("Mean |dist|  :", np.mean(abs_d) * 100, "cm")
#     print()

#     print("|d| > 2 cm ratio :", np.mean(abs_d > 0.02) * 100, "%")
#     print("|d| > 5 cm ratio :", np.mean(abs_d > 0.05) * 100, "%")
#     print("|d| > 10 cm ratio:", np.mean(abs_d > 0.10) * 100, "%")
#     print()

#     print("|d| > uncertainty ratio:", np.mean(abs_d > uncertainty) * 100, "%")

#     plt.figure()
#     plt.hist(d, bins=100)
#     plt.xlabel("M3C2 distance [m]")
#     plt.ylabel("Count")
#     plt.title("M3C2 distance histogram")
#     plt.grid(True)
#     plt.show()



# if __name__ == "__main__":
#     analyze_m3c2(r"F:\UNIVERSITY_BONN\master_thesis\working_space\evaluat_file\FOR_P151\M3C2_static_pointCloud.txt")


# import open3d as o3d

# pcd1 = o3d.t.io.read_point_cloud(r"F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal\output\pcl_kinematic_calibration.las")
# pcd2 = o3d.t.io.read_point_cloud(r"F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal\output\pcr_kinematic_calibration.las")

# params = o3d.t.geometry.MetricParameters()

# result = pcd1.compute_metrics(
#     pcd2,
#     [o3d.t.geometry.Metric.ChamferDistance],
#     params
# )

# print(result)




# import laspy
# import numpy as np
# import open3d as o3d


# def read_las_as_o3d(path):
#     las = laspy.read(path)

#     points = np.column_stack([
#         las.x,
#         las.y,
#         las.z
#     ]).astype(np.float32)

#     pcd = o3d.t.geometry.PointCloud(
#         o3d.core.Tensor(points)
#     )

#     return pcd


# pcd1 = read_las_as_o3d(
#     r"F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal\output\pcl_kinematic_calibration.las"
# )

# pcd2 = read_las_as_o3d(
#     r"F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal\output\pcr_kinematic_calibration.las"
# )

# params = o3d.t.geometry.MetricParameters()

# result = pcd1.compute_metrics(
#     pcd2,
#     [o3d.t.geometry.Metric.ChamferDistance],
#     params
# )

# print(result)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_spline_csv(csv_path, num_samples=200, save_path="spline_6dof.png"):
    # 读取 csv
    # 默认格式：
    # col 0: window_id
    # col 1: window_start
    # col 2: window_duration
    # col 3~26: 24个系数
    df = pd.read_csv(csv_path, header=None)

    window_ids = df.iloc[:, 0].to_numpy(dtype=int)
    window_starts = df.iloc[:, 1].to_numpy(dtype=float)
    window_durations = df.iloc[:, 2].to_numpy(dtype=float)
    coefficients_all = df.iloc[:, 3:27].to_numpy(dtype=float)

    # 按 window_start 排序，防止顺序乱
    order = np.argsort(window_starts)
    window_ids = window_ids[order]
    window_starts = window_starts[order]
    window_durations = window_durations[order]
    coefficients_all = coefficients_all[order]

    dof_names = ["rotvec_x", "rotvec_y", "rotvec_z", "tx", "ty", "tz"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    for i in range(len(window_ids)):
        start = window_starts[i]
        duration = window_durations[i]
        end = start + duration

        t = np.linspace(start, end, num_samples)
        u = (t - start) / duration

        basis = np.column_stack([
            np.ones_like(u),
            u,
            u**2,
            u**3
        ])   # (num_samples, 4)

        coeff = coefficients_all[i].reshape(6, 4)   # (6, 4)
        values = basis @ coeff.T                    # (num_samples, 6)

        for k in range(6):
            axes[k].plot(t, values[:, k], linewidth=1.2)
            axes[k].axvline(start, linestyle="--", alpha=0.3)
            axes[k].axvline(end, linestyle="--", alpha=0.3)
            axes[k].set_ylabel(dof_names[k])
            axes[k].grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("6-DoF Cubic Spline Over All Windows", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    csv_path = "spline_coefficients.csv"   # 改成你的文件路径
    plot_spline_csv(csv_path)