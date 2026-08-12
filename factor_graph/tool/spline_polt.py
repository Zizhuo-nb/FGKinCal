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


def plot_splines(windows, output_dir):
    names = ["rx", "ry", "rz", "tx", "ty", "tz"]

    fig, axes = plt.subplots(
        6, 1,
        figsize=(12, 14),
        sharex=True
    )

    for window_id in sorted(windows):
        data = windows[window_id]

        start = data["window_start"]
        duration = data["window_duration"]

        t = np.linspace(start, start + duration, 100)
        u = (t - start) / duration

        basis = np.column_stack([
            np.ones_like(u),
            u,
            u**2,
            u**3,
        ])

        coeff = np.asarray(
            data["coefficients"]
        ).reshape(6, 4)

        values = basis @ coeff.T

        for i in range(6):
            axes[i].plot(t, values[:, i])
            axes[i].set_ylabel(names[i])
            axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "spline_6dof.png"),
        dpi=200
    )

    plt.close()