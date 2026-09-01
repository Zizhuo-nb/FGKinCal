import os
import numpy as np
import matplotlib.pyplot as plt


def save_spline_coefficients(windows, output_dir):
    records = []

    for window_id in sorted(windows):
        data = windows[window_id]

        records.append([
            window_id,
            data["window_start"],
            data["window_duration"],
            *data["coefficients"],
        ])

    np.savetxt(
        os.path.join(output_dir, "spline_coefficients.csv"),
        np.asarray(records),
        delimiter=",",
    )


def plot_splines(windows, output_dir, num_samples=200):
    dof_names = [
        "rotvec_x",
        "rotvec_y",
        "rotvec_z",
        "tx",
        "ty",
        "tz",
    ]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(12, 14),
        sharex=True,
    )

    # 按 window_start 排序
    window_ids = sorted(
        windows,
        key=lambda window_id: windows[window_id]["window_start"]
    )

    for window_id in window_ids:
        data = windows[window_id]

        start = data["window_start"]
        duration = data["window_duration"]
        end = start + duration

        t = np.linspace(start, end, num_samples)
        u = (t - start) / duration

        basis = np.column_stack([
            np.ones_like(u),
            u,
            u**2,
            u**3,
        ])

        coeff = np.asarray(
            data["coefficients"],
            dtype=float,
        ).reshape(6, 4)

        values = basis @ coeff.T

        for k in range(6):
            axes[k].plot(
                t,
                values[:, k],
                linewidth=1.2,
            )

            axes[k].axvline(
                start,
                linestyle="--",
                alpha=0.3,
            )

            axes[k].axvline(
                end,
                linestyle="--",
                alpha=0.3,
            )

            axes[k].set_ylabel(dof_names[k])
            axes[k].grid(True)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        "6-DoF Cubic Spline Over All Windows",
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "spline_6dof.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()
    
    
    
    
    
def load_mean_spline_coefficients(csv_path):
    data = np.loadtxt(
        csv_path,
        delimiter=",",
    )
    # Handle the case of only one window
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # First 3 columns are:
    # window_id, window_start, window_duration
    coefficients = data[:, 3:]

    if coefficients.shape[1] != 24:
        raise ValueError(
            f"Expected 24 spline coefficients, "
            f"but got {coefficients.shape[1]}"
        )

    # Mean over all windows
    mean_coefficients = np.mean(
        coefficients,
        axis=0,
    )

    mean_matrix = mean_coefficients.reshape(6, 4)

    return mean_coefficients, mean_matrix
    