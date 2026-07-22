import numpy as np
import matplotlib.pyplot as plt


def analyze_m3c2(file_path):
    data = np.loadtxt(file_path, comments="/")

    uncertainty = data[:, 4]
    d = data[:, 5]   # M3C2_distance

    # remove NaN / inf
    valid = np.isfinite(d) & np.isfinite(uncertainty)

    print("========== M3C2 Statistics ==========")
    print("file:", file_path)
    print("total points:", len(d))
    print("valid points:", np.sum(valid))
    print("invalid / nan points:", len(d) - np.sum(valid))
    print()

    d = d[valid]
    uncertainty = uncertainty[valid]

    abs_d = np.abs(d)

    print("Mean distance:", np.mean(d), "m")
    print("Std distance :", np.std(d), "m")
    print("RMS distance :", np.sqrt(np.mean(d ** 2)), "m")
    print("Mean |dist|  :", np.mean(abs_d), "m")
    print("Median |dist|:", np.median(abs_d), "m")
    print("95% |dist|   :", np.percentile(abs_d, 95), "m")
    print("Max |dist|   :", np.max(abs_d), "m")
    print()

    print("Mean distance:", np.mean(d) * 100, "cm")
    print("Std distance :", np.std(d) * 100, "cm")
    print("RMS distance :", np.sqrt(np.mean(d ** 2)) * 100, "cm")
    print("Mean |dist|  :", np.mean(abs_d) * 100, "cm")
    print()

    print("|d| > 2 cm ratio :", np.mean(abs_d > 0.02) * 100, "%")
    print("|d| > 5 cm ratio :", np.mean(abs_d > 0.05) * 100, "%")
    print("|d| > 10 cm ratio:", np.mean(abs_d > 0.10) * 100, "%")
    print()

    print("|d| > uncertainty ratio:", np.mean(abs_d > uncertainty) * 100, "%")

    plt.figure()
    plt.hist(d, bins=100)
    plt.xlabel("M3C2 distance [m]")
    plt.ylabel("Count")
    plt.title("M3C2 distance histogram")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    analyze_m3c2(r"F:\UNIVERSITY_BONN\master_thesis\working_space\evaluat_file\M3C2_factor_smaller_window.txt")