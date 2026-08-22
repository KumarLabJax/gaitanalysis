"""
Extends gaitanalysis's own per-stride lateral-displacement amplitude/phase
computation with a second, curve-averaging method, and a phase-curve
export for plotting.

Two ways to summarize a landmark's swing across a recording's strides:

PerStride (gaitinference's own method)
    cyclic-spline-interpolate each stride individually (spline_interpolate), 
    measure its own peak/trough (lat_disp_amp / lat_disp_phase), then 
    aggregate those per-stride numbers across strides -- arithmetic mean 
    for amplitude, circular mean/variance for phase (phase wraps at 0%/100%).

AverageFirst (this one)
    Average every stride within a video first, then read amplitude and phases 
    off that one averaged curve.

"""

import os
import urllib.parse as urlparse

import h5py
import numpy as np
import pandas as pd
import scipy.stats

import gaitinference as ginf


LANDMARKS = {
    "Base Tail": ginf.BASE_TAIL_INDEX,
    "Nose": ginf.NOSE_INDEX,
    "Tip Tail": ginf.TIP_TAIL_INDEX,
}

DEFAULT_NUM_INTERP_FRAMES = 60

OFFICIAL_INTERP_FRAMES = 360

def phase_axis(num_interp_frames):
    """
    The AverageFirst method's phase axis, 0-100%.

    The interpolation itself spans the full stride (both endpoints
    included), so num_interp_frames points conceptually run 0%-100%.
    Historically these were labeled 100 * arange(n) / n instead (e.g.
    0, 1.667, ..., 98.333 for n=60); kept here for compatibility with
    existing phase-curve plots and tables.
    """
    return 100.0 * np.arange(num_interp_frames) / num_interp_frames


def get_optional_attr(group, name):
    """Safely retrieve an HDF5 attribute."""
    if name in group.attrs:
        return group.attrs[name]

    return np.nan


def center_strides(lateral_values):
    """
    Center each stride around its own mean lateral position.

    lateral_values : ndarray, shape (n_strides, n_phase_points, ...)
    """
    stride_means = np.mean(
        lateral_values,
        axis=1,
        keepdims=True,
    )

    return lateral_values - stride_means


def _per_stride_interpolate(stride):
    """
    Cyclic-spline-interpolate one stride to OFFICIAL_INTERP_FRAMES points
    across all 12 landmarks/2 dims, matching
    gaitinference.add_lateral_displacement_to_strides(). 
    """
    try:
        frame_count, point_count, dim_count = stride.shape

        interpolated = np.empty(
            (OFFICIAL_INTERP_FRAMES, point_count, dim_count),
            dtype=np.double,
        )

        for point_index in range(point_count):
            for dim_index in range(dim_count):
                interpolated[:, point_index, dim_index] = ginf.spline_interpolate(
                    stride[:, point_index, dim_index],
                    OFFICIAL_INTERP_FRAMES,
                    cyclic=True,
                )

        return interpolated

    except TypeError:
        return stride


def summarize_strides_dual(strides, num_interp_frames=DEFAULT_NUM_INTERP_FRAMES):
    """
    Compute AverageFirst and PerStride Amplitude/PeakOffset/TroughOffset
    for all three tracked landmarks from one recording/bin's raw strides.

    Parameters
    ----------
    strides : list of ndarray
        Raw (frames x 12 x 2) stride arrays for one recording/bin.

    Returns
    -------
    dict
        Keyed by landmark name ("Base Tail", "Nose", "Tip Tail") -> dict
        with "Amplitude_AverageFirst", "PeakOffset_AverageFirst",
        "TroughOffset_AverageFirst", "PeakOffset_PerStride",
        "PeakOffset_PerStride_Variance", "TroughOffset_PerStride",
        "TroughOffset_PerStride_Variance".
    """
    n_strides = len(strides)

    values = phase_axis(num_interp_frames)

    # ---- AverageFirst: linear interpolation + averaging, all landmarks ----

    interpolated = np.stack(
        [
            ginf.interpolate_stride_points(stride, num_interp_frames)
            for stride in strides
        ],
        axis=0,
    )

    # Shape: n_strides x n_phase_points x 12 landmarks
    mean_curve = np.mean(
        center_strides(interpolated[:, :, :, 1]),
        axis=0,
    )

    peak_idx_avg = np.argmax(mean_curve, axis=0)
    trough_idx_avg = np.argmin(mean_curve, axis=0)

    landmark_range = np.arange(mean_curve.shape[1])
    peak_val_avg = mean_curve[peak_idx_avg, landmark_range]
    trough_val_avg = mean_curve[trough_idx_avg, landmark_range]

    # ---- PerStride: gaitinference's own cyclic-spline peak/trough phase ----

    peak_phases = np.empty((n_strides, 12))
    trough_phases = np.empty((n_strides, 12))

    for stride_index, stride in enumerate(strides):

        interpolated_stride = _per_stride_interpolate(stride)

        peak_phases[stride_index] = ginf.lat_disp_phase(interpolated_stride)
        trough_phases[stride_index] = ginf.lat_disp_trough_phase(interpolated_stride)

    peak_offset_per_stride = scipy.stats.circmean(
        peak_phases, axis=0, low=0.0, high=1.0,
    )
    trough_offset_per_stride = scipy.stats.circmean(
        trough_phases, axis=0, low=0.0, high=1.0,
    )
    peak_offset_variance = scipy.stats.circvar(
        peak_phases, axis=0, low=0.0, high=1.0,
    )
    trough_offset_variance = scipy.stats.circvar(
        trough_phases, axis=0, low=0.0, high=1.0,
    )

    results = {}

    for landmark_name, landmark_index in LANDMARKS.items():

        results[landmark_name] = {
            "Amplitude_AverageFirst": (
                peak_val_avg[landmark_index] - trough_val_avg[landmark_index]
            ) / 2.0,
            "PeakOffset_AverageFirst": values[peak_idx_avg[landmark_index]],
            "TroughOffset_AverageFirst": values[trough_idx_avg[landmark_index]],
            "PeakOffset_PerStride": peak_offset_per_stride[landmark_index] * 100.0,
            "PeakOffset_PerStride_Variance": peak_offset_variance[landmark_index],
            "TroughOffset_PerStride": trough_offset_per_stride[landmark_index] * 100.0,
            "TroughOffset_PerStride_Variance": trough_offset_variance[landmark_index],
        }

    return results


def _summarize_landmark_curve(interpolated_strides, landmark_index, values):
    """
    Recording-level mean/SD/SEM across strides at each phase point, for
    one landmark. `values` is the phase axis (see phase_axis()).
    """
    lateral_y = center_strides(
        interpolated_strides[:, :, landmark_index, 1]
    )

    n_strides = lateral_y.shape[0]

    mean_displacement = np.mean(lateral_y, axis=0)

    if n_strides > 1:

        sd_displacement = np.std(lateral_y, axis=0, ddof=1)
        sem_displacement = sd_displacement / np.sqrt(n_strides)

    else:

        sd_displacement = np.full(len(values), np.nan)
        sem_displacement = np.full(len(values), np.nan)

    return pd.DataFrame({
        "PhaseIndex": np.arange(len(values)),
        "PercentStride": values,
        "MeanDisplacement": mean_displacement,
        "SDDisplacement": sd_displacement,
        "SEMDisplacement": sem_displacement,
        "NStrides": n_strides,
    })


def export_phase_curves(
    gait_h5,
    speed_bin,
    output_path,
    num_interp_frames=DEFAULT_NUM_INTERP_FRAMES,
):
    """
    Write the recording-level, long-format phase-curve CSV (one row per
    recording x landmark x phase point) for one speed bin, given an
    already-open gait HDF5 file.
    """
    output_tables = []

    if "angular_velocity_bin_size" not in gait_h5.attrs:
        raise ValueError(
            "HDF5 file does not contain 'angular_velocity_bin_size'."
        )

    angular_velocity_bin_size = float(gait_h5.attrs["angular_velocity_bin_size"])

    speed_bin_size = get_optional_attr(gait_h5, "speed_bin_size")

    # The bin centered around zero angular velocity, e.g. for the default
    # 40 deg/sec bin size, bin key -20 corresponds to -20 <= av < 20.
    center_av_bin = int(-angular_velocity_bin_size // 2)

    bin_name = ginf.speed_av_bin_tup_to_str((speed_bin, center_av_bin))

    if not pd.isna(speed_bin_size):
        speed_bin_upper = float(speed_bin) + float(speed_bin_size)
    else:
        speed_bin_upper = np.nan

    angular_velocity_bin_lower = float(center_av_bin)
    angular_velocity_bin_upper = angular_velocity_bin_lower + angular_velocity_bin_size

    values = phase_axis(num_interp_frames)

    for escaped_filename in gait_h5.keys():

        network_filename = urlparse.unquote(escaped_filename)

        recording_group = gait_h5[escaped_filename]

        bin_path = escaped_filename + "/bins/" + bin_name

        if bin_path not in gait_h5:
            continue

        bin_group = gait_h5[bin_path]

        if "normalized_stride_points" not in bin_group:
            continue

        stored_stride_points = list(bin_group["normalized_stride_points"])

        if not stored_stride_points:
            continue

        strides = list(ginf.restore_stride_points_shape(stored_stride_points))

        interpolated = np.stack(
            [
                ginf.interpolate_stride_points(stride, num_interp_frames)
                for stride in strides
            ],
            axis=0,
        )

        body_length_cm = get_optional_attr(recording_group, "median_body_length_cm")

        for landmark_name, landmark_index in LANDMARKS.items():

            landmark_table = _summarize_landmark_curve(
                interpolated, landmark_index, values,
            )

            landmark_table.insert(0, "Landmark", landmark_name)
            landmark_table.insert(0, "MedianBodyLengthCm", body_length_cm)
            landmark_table.insert(0, "AngularVelocityBinUpper", angular_velocity_bin_upper)
            landmark_table.insert(0, "AngularVelocityBinLower", angular_velocity_bin_lower)
            landmark_table.insert(0, "AngularVelocityBin", center_av_bin)
            landmark_table.insert(0, "SpeedBinUpper", speed_bin_upper)
            landmark_table.insert(0, "SpeedBinLower", float(speed_bin))
            landmark_table.insert(0, "SpeedBin", int(speed_bin))
            landmark_table.insert(0, "Recording", network_filename)

            output_tables.append(landmark_table)

    if not output_tables:
        raise RuntimeError(
            "No phase-curve data were found for speed bin "
            + str(speed_bin)
            + ". Check the HDF5 file."
        )

    output = pd.concat(output_tables, ignore_index=True)

    output = output.sort_values(
        ["Recording", "Landmark", "PhaseIndex"],
        kind="stable",
    ).reset_index(drop=True)

    output.to_csv(output_path, index=False)


def export_phase_curves_all_bins(
    gait_h5,
    speed_bins,
    output_dir,
    num_interp_frames=DEFAULT_NUM_INTERP_FRAMES,
):
    """
    Write one phase-curve CSV per speed bin into output_dir (created if
    needed), named phase-curves-speed-<bin>.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    for speed_bin in speed_bins:

        output_path = os.path.join(
            output_dir,
            "phase-curves-speed-{}.csv".format(speed_bin),
        )

        export_phase_curves(
            gait_h5,
            speed_bin,
            output_path,
            num_interp_frames=num_interp_frames,
        )
