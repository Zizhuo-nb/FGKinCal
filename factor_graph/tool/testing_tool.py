import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import medfilt, savgol_filter
def plot_sampled_points_time_z(
    window_start,
    window_end,
    sampled_points,
    sampled_times,
    window_id,
    save_path=None,
    show=True,
):
    """
    Plot the temporal and vertical distribution of sampled points.

    x-axis:
        Global Z coordinate.

    y-axis:
        Relative point time inside the current window.

    Parameters
    ----------
    window_start : float
        Start timestamp of the current window.

    window_end : float
        End timestamp of the current window.

    sampled_points : np.ndarray
        Sampled point coordinates with shape (N, 3).

    sampled_times : np.ndarray
        Per-point timestamps with shape (N,) or (N, 1).

    window_id : int
        Current window identifier.

    save_path : str or None
        Optional path for saving the figure.

    show : bool
        Whether to display the figure.
    """
    sampled_points = np.asarray(
        sampled_points,
        dtype=np.float64,
    )

    sampled_times = np.asarray(
        sampled_times,
        dtype=np.float64,
    ).reshape(-1)

    if sampled_points.ndim != 2 or sampled_points.shape[1] != 3:
        raise ValueError(
            "sampled_points must have shape (N, 3)."
        )

    if len(sampled_points) != len(sampled_times):
        raise ValueError(
            "The number of sampled points and timestamps must match."
        )

    if window_end <= window_start:
        raise ValueError(
            "window_end must be greater than window_start."
        )

    if len(sampled_points) == 0:
        print(
            f"[Sampling plot] Window {window_id}: no sampled points."
        )
        return

    relative_time = sampled_times - window_start
    z_values = sampled_points[:, 2]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.scatter(
        relative_time,  # x-axis: time
        z_values,       # y-axis: Z
        s=10,
        alpha=0.7,
    )

    axis.set_xlabel("Relative time in window [s]")
    axis.set_ylabel("Global Z coordinate [m]")

    axis.set_title(
        f"Window {window_id}: sampled-point distribution\n"
        f"Number of points: {len(sampled_points)}"
    )

    axis.set_xlim(
        0.0,
        window_end - window_start,
    )

    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    if save_path is not None:
        output_parent = os.path.dirname(save_path)

        if output_parent:
            os.makedirs(
                output_parent,
                exist_ok=True,
            )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"[Sampling plot] Saved to: {save_path}"
        )

    if show:
        plt.show()

    plt.close(figure)











def plot_full_z_and_max_z_by_timestamp(
window_start,
window_end,
point_cloud,
point_times,
window_id,
save_path=None,
show=True,
show_all_points=True,
max_background_points=200000,
time_bin_size=None,
):
    """
    Plot the full vertical distribution and maximum Z value per timestamp.

    x-axis:
        Relative time inside the current window.

    y-axis:
        Global Z coordinate.

    Parameters
    ----------
    window_start : float
        Start timestamp of the window.

    window_end : float
        End timestamp of the window.

    point_cloud : np.ndarray
        Complete right-window point cloud with shape (N, 3).

    point_times : np.ndarray
        Timestamp of every right-window point.
        Supported shapes: (N,) or (N, 1).

    window_id : int
        Window identifier.

    save_path : str or None
        Optional output image path.

    show : bool
        Whether to display the figure.

    show_all_points : bool
        Whether to draw the complete Z distribution in the background.

    max_background_points : int
        Maximum number of background points to draw.
        This only affects visualization, not maximum-Z calculation.

    time_bin_size : float or None
        If None, group points by their exact timestamps.
        Otherwise, group points into time bins of this size in seconds.
    """
    point_cloud = np.asarray(
        point_cloud,
        dtype=np.float64,
    )

    point_times = np.asarray(
        point_times,
        dtype=np.float64,
    ).reshape(-1)

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(
            "point_cloud must have shape (N, 3)."
        )

    if len(point_cloud) != len(point_times):
        raise ValueError(
            "The number of points and timestamps must match."
        )

    if window_end <= window_start:
        raise ValueError(
            "window_end must be greater than window_start."
        )

    # Keep only valid points inside the current window.
    valid_mask = (
        np.isfinite(point_times)
        & np.isfinite(point_cloud[:, 2])
        & (point_times >= window_start)
        & (point_times <= window_end)
    )

    valid_points = point_cloud[valid_mask]
    valid_times = point_times[valid_mask]

    if len(valid_points) == 0:
        print(
            f"[Maximum-Z plot] Window {window_id}: "
            "no valid points."
        )
        return

    z_values = valid_points[:, 2]
    relative_times = valid_times - window_start

    # Create the group identifier.
    if time_bin_size is None:
        # Your current data uses exactly the same timestamp
        # for all points belonging to one laser frame.
        group_ids = valid_times
    else:
        if time_bin_size <= 0:
            raise ValueError(
                "time_bin_size must be positive."
            )

        group_ids = np.floor(
            relative_times / time_bin_size
        ).astype(np.int64)

    # Sort once so maximum Z can be calculated efficiently.
    sorting_indices = np.argsort(group_ids)

    sorted_group_ids = group_ids[sorting_indices]
    sorted_times = valid_times[sorting_indices]
    sorted_z = z_values[sorting_indices]

    # Find the first index of every timestamp or time bin.
    group_start_indices = np.concatenate(
        (
            np.array([0]),
            np.flatnonzero(
                sorted_group_ids[1:] != sorted_group_ids[:-1]
            ) + 1,
        )
    )

    group_end_indices = np.concatenate(
        (
            group_start_indices[1:],
            np.array([len(sorted_group_ids)]),
        )
    )

    group_counts = (
        group_end_indices - group_start_indices
    )

    # Maximum Z value in each timestamp group.
    maximum_z = np.maximum.reduceat(
        sorted_z,
        group_start_indices,
    )

    # Use the mean timestamp as the horizontal location.
    group_time_sum = np.add.reduceat(
        sorted_times,
        group_start_indices,
    )

    representative_times = (
        group_time_sum / group_counts
    )

    representative_relative_times = (
        representative_times - window_start
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    # Draw the complete vertical point distribution.
    if show_all_points:
        if len(valid_points) > max_background_points:
            sampling_step = int(
                np.ceil(
                    len(valid_points)
                    / max_background_points
                )
            )

            display_indices = np.arange(
                0,
                len(valid_points),
                sampling_step,
            )
        else:
            display_indices = np.arange(
                len(valid_points)
            )

        axis.scatter(
            relative_times[display_indices],
            z_values[display_indices],
            s=1,
            alpha=0.15,
            label="Complete right-window Z distribution",
        )

    # Draw the upper plant envelope.
    axis.plot(
        representative_relative_times,
        maximum_z,
        linewidth=1.5,
        label="Maximum Z per timestamp",
    )

    axis.scatter(
        representative_relative_times,
        maximum_z,
        s=8,
    )

    axis.set_xlabel(
        "Relative time in window [s]"
    )

    axis.set_ylabel(
        "Global Z coordinate [m]"
    )

    axis.set_title(
        f"Window {window_id}: complete Z distribution "
        f"and maximum-Z envelope\n"
        f"Points: {len(valid_points)}, "
        f"timestamps/groups: {len(maximum_z)}"
    )

    axis.set_xlim(
        0.0,
        window_end - window_start,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    if save_path is not None:
        output_parent = os.path.dirname(
            save_path
        )

        if output_parent:
            os.makedirs(
                output_parent,
                exist_ok=True,
            )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"[Maximum-Z plot] Saved to: {save_path}"
        )

    if show:
        plt.show()

    plt.close(figure)

    return (
        representative_relative_times,
        maximum_z,
    )



def detect_plant_time_interval(
    point_cloud,
    point_times,
    window_start,
    median_kernel=5,
    savgol_window=9,
    savgol_order=2,
    derivative_sigma=3.0,
    height_sigma=3.0,
    persistence=3,
):
    """
    Detect the plant interval from the maximum-Z profile.

    The function returns:
        1. first reliable positive Z transition;
        2. last reliable negative Z transition.

    Parameters
    ----------
    point_cloud : np.ndarray
        Complete right-window point cloud, shape (N, 3).

    point_times : np.ndarray
        Timestamp of each right-window point, shape (N,) or (N, 1).

    window_start : float
        Absolute start time of the window.

    median_kernel : int
        Kernel size used by the median filter.
        Must be odd.

    savgol_window : int
        Window length used by the Savitzky-Golay filter.
        Must be odd.

    savgol_order : int
        Polynomial order of the Savitzky-Golay filter.

    derivative_sigma : float
        Sensitivity of positive/negative transition detection.

    height_sigma : float
        Sensitivity used to decide whether the profile is above ground level.

    persistence : int
        Number of consecutive timestamps required to validate a transition.

    Returns
    -------
    result : dict
        Plant start/end timestamps and diagnostic profiles.
    """
    point_cloud = np.asarray(
        point_cloud,
        dtype=np.float64,
    )

    point_times = np.asarray(
        point_times,
        dtype=np.float64,
    ).reshape(-1)

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(
            "point_cloud must have shape (N, 3)."
        )

    if len(point_cloud) != len(point_times):
        raise ValueError(
            "The number of points and timestamps must match."
        )

    valid_mask = (
        np.isfinite(point_times)
        & np.isfinite(point_cloud[:, 2])
    )

    valid_times = point_times[valid_mask]
    valid_z = point_cloud[valid_mask, 2]

    if len(valid_times) == 0:
        raise ValueError("No valid points are available.")

    # Sort all points by timestamp.
    sort_idx = np.argsort(valid_times)
    valid_times = valid_times[sort_idx]
    valid_z = valid_z[sort_idx]

    # Group points with exactly the same timestamp.
    unique_times, group_start_idx = np.unique(
        valid_times,
        return_index=True,
    )

    # Maximum Z value for every timestamp.
    max_z = np.maximum.reduceat(
        valid_z,
        group_start_idx,
    )

    if len(unique_times) < 5:
        raise ValueError(
            "Too few unique timestamps for interval detection."
        )

    # Ensure valid odd filter sizes.
    median_kernel = min(
        median_kernel,
        len(max_z) if len(max_z) % 2 == 1 else len(max_z) - 1,
    )

    median_kernel = max(3, median_kernel)

    if median_kernel % 2 == 0:
        median_kernel -= 1

    savgol_window = min(
        savgol_window,
        len(max_z) if len(max_z) % 2 == 1 else len(max_z) - 1,
    )

    savgol_window = max(
        savgol_order + 2,
        savgol_window,
    )

    if savgol_window % 2 == 0:
        savgol_window -= 1

    # First remove isolated spikes, then smooth the plant profile.
    max_z_median = medfilt(
        max_z,
        kernel_size=median_kernel,
    )

    max_z_smooth = savgol_filter(
        max_z_median,
        window_length=savgol_window,
        polyorder=savgol_order,
        mode="interp",
    )

    # Estimate the ground/background height from the lower profile.
    lower_profile = max_z_smooth[
        max_z_smooth
        <= np.percentile(max_z_smooth, 35)
    ]

    ground_level = np.median(lower_profile)

    ground_mad = np.median(
        np.abs(lower_profile - ground_level)
    )

    ground_sigma = 1.4826 * ground_mad

    # Avoid a zero threshold when the ground is extremely flat.
    ground_sigma = max(
        ground_sigma,
        0.005,
    )

    height_threshold = (
        ground_level
        + height_sigma * ground_sigma
    )

    # Numerical derivative dz/dt.
    dz_dt = np.gradient(
        max_z_smooth,
        unique_times,
    )

    derivative_center = np.median(dz_dt)

    derivative_mad = np.median(
        np.abs(dz_dt - derivative_center)
    )

    derivative_scale = 1.4826 * derivative_mad

    derivative_scale = max(
        derivative_scale,
        1e-6,
    )

    positive_threshold = (
        derivative_center
        + derivative_sigma * derivative_scale
    )

    negative_threshold = (
        derivative_center
        - derivative_sigma * derivative_scale
    )

    above_plant_height = (
        max_z_smooth > height_threshold
    )

    start_index = None

    # Find the first positive transition followed by persistent high values.
    for index in range(
        1,
        len(unique_times) - persistence,
    ):
        persistent_high = np.all(
            above_plant_height[
                index:index + persistence
            ]
        )

        positive_transition = (
            dz_dt[index] > positive_threshold
        )

        crossed_height_threshold = (
            max_z_smooth[index - 1] <= height_threshold
            and max_z_smooth[index] > height_threshold
        )

        if persistent_high and (
            positive_transition
            or crossed_height_threshold
        ):
            start_index = index
            break

    end_index = None

    # Find the last negative transition followed by persistent low values.
    for index in range(
        len(unique_times) - persistence - 1,
        0,
        -1,
    ):
        persistent_low = np.all(
            ~above_plant_height[
                index:index + persistence
            ]
        )

        negative_transition = (
            dz_dt[index] < negative_threshold
        )

        crossed_height_threshold = (
            max_z_smooth[index - 1] > height_threshold
            and max_z_smooth[index] <= height_threshold
        )

        if persistent_low and (
            negative_transition
            or crossed_height_threshold
        ):
            end_index = index
            break

    if start_index is None or end_index is None:
        print(
            "[Plant detection] A complete plant interval "
            "could not be detected."
        )

        return None

    if end_index <= start_index:
        print(
            "[Plant detection] Detected end time is not "
            "later than start time."
        )

        return None

    plant_start_time = unique_times[start_index]
    plant_end_time = unique_times[end_index]

    print(
        "[Plant detection]\n"
        f"  start absolute time: {plant_start_time:.8f}\n"
        f"  end absolute time:   {plant_end_time:.8f}\n"
        f"  start relative time: "
        f"{plant_start_time - window_start:.6f} s\n"
        f"  end relative time:   "
        f"{plant_end_time - window_start:.6f} s\n"
        f"  plant duration:      "
        f"{plant_end_time - plant_start_time:.6f} s\n"
        f"  height threshold:    "
        f"{height_threshold:.4f} m"
    )

    return {
        "plant_start_time": plant_start_time,
        "plant_end_time": plant_end_time,
        "plant_start_relative": (
            plant_start_time - window_start
        ),
        "plant_end_relative": (
            plant_end_time - window_start
        ),
        "start_index": start_index,
        "end_index": end_index,
        "unique_times": unique_times,
        "max_z": max_z,
        "max_z_smooth": max_z_smooth,
        "dz_dt": dz_dt,
        "ground_level": ground_level,
        "height_threshold": height_threshold,
        "positive_derivative_threshold": positive_threshold,
        "negative_derivative_threshold": negative_threshold,
    }