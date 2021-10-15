import argparse
import csv
import functools
import h5py
import multiprocessing as mp
import numpy as np
import os
import urllib.parse as urlparse
import yaml

import gaitinference as ginf


# Example use:
#
#   python gengaitstats.py \
#           --batch-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10_exists.txt' \
#           --root-dir '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10'
#
#   python gengaitstats.py \
#           --batch-file 'strain-survey.txt' \
#           --root-dir '.' \
#           --out-file output/strain-survey-gait.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-01-15.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutant-output.h5
#
#
#   # now we generate with the extra controls
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-with-extra-ctrls-2019-01-15.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutant-with-extra-ctrls-output.h5
#
#   # next version of gait mutants
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-01-23.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-2019-01-23-output-test.h5
#
#   # need to regenerate the strain survey output
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-01-24.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#           --out-file output/strain-survey-gait-2019-01-24.h5 \
#           --max-duration-minutes 55
#
#   # generate output for autistic models (these do include novel objects and injections so
#   # we need to limit the data to only the first 55 min)
#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-04-01.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-02-25_exists.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-output-2019-05-31.h5
#
#   # need to regenerate the strain survey output
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-05-29.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/strain-survey-gait-2019-06-03.h5 \
#           --max-duration-minutes 55
#
#   # generate output for autistic models (these do include novel objects and injections so
#   # we need to limit the data to only the first 55 min)
#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-06-05.h5 \
#           --max-duration-minutes 55

####  updated runs for high res model after using labels from Megan 2019-07-02 ####

#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-07-02.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-02-25_exists.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-output-2019-07-02.h5
#
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-05-29.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/strain-survey-gait-2019-07-02.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py \
#           --batch-file data/metadata/CCF1-day1-2019-07-02.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/CCF1-day1-2019-07-02.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py \
#           --batch-file data/metadata/KOMP-curated-2019-07-02.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/KOMP-curated-2019-07-02.h5

####  updated output and added phase 2019-08-02 ####

#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-08-02.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-02-25_exists.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-output-2019-08-02.h5
#
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-05-29.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/strain-survey-gait-2019-08-02.h5 \
#           --max-duration-minutes 55

####  improved phase with interpolation 2019-08-19 ####

#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-08-19.h5 \
#           --max-duration-minutes 55
#
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-02-25_exists.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-output-2019-08-19.h5
#
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-05-29.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/strain-survey-gait-2019-08-19.h5 \
#           --max-duration-minutes 55

#### Formalin (nociception)

#   python gengaitstats.py --batch-file ./data/metadata/FormalinGaitFileNames.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/FormalinGait-2019-08-20.h5

#### 

#   python gengaitstats.py --batch-file ./data/metadata/AgedB6-2019-10-21.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/AgedB6-2019-10-21.h5

#### changed accum function for cleaner hildebrand plots 2019-11-12

#   echo "========== AUTISM RUN =========="
#   python gengaitstats.py \
#           --batch-file data/metadata/autism-kos-2019-04-01.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/autism-kos-2019-11-12.h5 \
#           --max-duration-minutes 55
#
#   echo "========== GAIT MUTANT RUN =========="
#   python gengaitstats.py --batch-file ./data/metadata/gait-mutants-2019-02-25_exists.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/gait-mutants-output-2019-11-12.h5
#
#   echo "========== STRAIN SURVEY RUN =========="
#   python gengaitstats.py \
#           --batch-file data/metadata/strain-survey-batch-2019-05-29.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/strain-survey-gait-2019-11-12.h5 \
#           --max-duration-minutes 55
#
#   echo "========== KOMP RUN =========="
#   python gengaitstats.py \
#           --batch-file data/metadata/KOMP-curated-2019-07-02.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file output/KOMP-curated-2019-11-12.h5

#### CFA for Kyunyin 2019-11-22

#   echo "========== CFA RUN =========="
#   python gengaitstats.py --batch-file ./data/metadata/cfa-kk-all-batch.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file ./output/cfa-kk-all-batch-2019-11-22.h5

#### CFA for Leinani 2020-01-13

# python gengaitstats.py --batch-file cfa-project/baseline_CFA.txt \
#           --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#           --out-file cfa-project/output/baseline_CFA.h5

#   echo "========== KOMP RUN =========="
#   python gengaitstats.py \
#           --batch-file data/metadata/KOMP-curated-2019-07-02.txt \
#           --root-dir ~/smb/labshare \
#           --cm-per-pixel-mapping data/metadata/KOMP_map.txt \
#           --out-file output/KOMP-curated-2019-11-12.h5

#   python gengaitstats.py \
#           --batch-file data/metadata/KOMP-curated-2019-07-02.txt \
#           --root-dir ~/smb/labshare \
#           --out-file output/KOMP-curated-nomaping-2019-11-12.h5

#   python gengaitstats.py \
#           --batch-file data/metadata/KOMP-curated-2019-07-02.txt \
#           --root-dir ~/smb/labshare \
#           --cm-per-pixel-mapping data/metadata/KOMP_map.txt \
#           --out-file output/KOMP-curated-2020-08-20.h5

#### Gait for Leinani on 2020-08-12 (note temp2 to be moved to USB drive)

#   python gengaitstats.py \
#           --batch-file ~/temp2/leinani-2020-07-31.txt \
#           --root-dir ~/smb/labshare \
#           --out-file ~/temp2/leinani-2020-07-31-gait.h5

#### Gait for Vivek Kohar

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/'
# python gengaitstats.py \
#       --batch-file data/metadata/networkPointNotFoundBXD.txt \
#       --root-dir "${share_root}" \
#       --out-file output/networkPointNotFoundBXD-2020-11-17.h5

#### BackTracked_Data_ForJAABA Gait for Vivek Kohar

# python gengaitstats.py \
#       --batch-file '/media/sheppk/TOSHIBA EXT/BackTracked_Data_ForJAABA/BackTracked_Data_ForJAABA.txt' \
#       --root-dir '/media/sheppk/TOSHIBA EXT/BackTracked_Data_ForJAABA' \
#       --out-file output/BackTracked_Data_ForJAABA-2020-12-07.h5

#### B6 and B6J strain survey gait with stride position and proportional position

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/'
# python gengaitstats.py \
#       --batch-file data/metadata/strain-survey-b6j-bjnj-only-batch-2021-01-18.txt \
#       --root-dir "${share_root}" \
#       --out-file output/strain-survey-b6j-bjnj-only-batch-2021-01-18.h5

def _any_good_strides(tracks):
    for track in tracks:
        for _ in track.good_strides:
            return True

    return False


def gen_gait_stats(net_fname, data_file_name, corner_file_name, root_dir,
                   base_tail_smooth, max_duration_frames, sb_size, avb_size,
                   cm_per_px):

    def limit_frames(xs):
        return xs[:max_duration_frames]

    data_file_path = os.path.join(root_dir, data_file_name)
    data_file = None
    try:
        data_file = h5py.File(data_file_path, 'r')
    except OSError:
        print('ERROR: FAILED TO OPEN DATA FILE', data_file_path)
        return None

    # calculate the open field dimensions if there is a valid corners file
    corner_file_path = os.path.join(root_dir, corner_file_name)
    open_field_dims = None
    try:
        if os.path.exists(corner_file_path):
            with open(corner_file_path, 'r') as corner_file:
                doc = yaml.safe_load(corner_file)

                # the order of coorinates is: upper left, lower left,
                # upper right, lower right
                xs = doc['corner_coords']['xs']
                ys = doc['corner_coords']['ys']

                min_x = np.mean((xs[0], xs[1]))
                width = np.mean((xs[2], xs[3])) - min_x
                min_y = np.mean((ys[0], ys[2]))
                height = np.mean((ys[1], ys[3])) - min_y

                open_field_dims = {
                    'min_x': min_x,
                    'width': width,
                    'min_y': min_y,
                    'height': height,
                }
    except Exception:
        print('failed to parse corners file:', corner_file_path)

    video_dict = None

    for group in data_file.values():
        base_tail_speed = limit_frames(ginf.calc_speed(
            group,
            ginf.BASE_TAIL_INDEX,
            smoothing_window=base_tail_smooth,
            cm_per_px=cm_per_px))
        left_rear_paw_speed = limit_frames(ginf.calc_speed(group, ginf.LEFT_REAR_PAW_INDEX, cm_per_px=cm_per_px))
        right_rear_paw_speed = limit_frames(ginf.calc_speed(group, ginf.RIGHT_REAR_PAW_INDEX, cm_per_px=cm_per_px))

        angle_deg = limit_frames(ginf.calc_angle_deg(group))
        angular_speed = list(ginf.calc_angle_speed_deg(angle_deg, smoothing_window=5))

        tracks = list(ginf.trackstridedet(
            left_rear_paw_speed,
            right_rear_paw_speed,
            base_tail_speed,
            angular_speed,
            cm_per_px=cm_per_px))

        if tracks:
            left_rear_paw_xy = limit_frames(ginf.get_xy_pos(group, ginf.LEFT_REAR_PAW_INDEX))
            right_rear_paw_xy = limit_frames(ginf.get_xy_pos(group, ginf.RIGHT_REAR_PAW_INDEX))
            base_tail_xy = limit_frames(ginf.get_xy_pos(group, ginf.BASE_TAIL_INDEX))

            lr_paw_conf = limit_frames(ginf.get_conf(group, ginf.LEFT_REAR_PAW_INDEX))
            rr_paw_conf = limit_frames(ginf.get_conf(group, ginf.RIGHT_REAR_PAW_INDEX))
            base_tail_conf = limit_frames(ginf.get_conf(group, ginf.BASE_TAIL_INDEX))

            points = limit_frames(group['points'][:]).astype(np.double)
            ginf.add_median_xy_pos_to_strides(tracks, points, open_field_dims=open_field_dims)
            del points
            ginf.add_xy_pos_to_strides(tracks, left_rear_paw_xy, right_rear_paw_xy)
            ginf.add_conf_to_strides(group, tracks)
            ginf.add_conf_to_tracks(tracks, lr_paw_conf, rr_paw_conf, base_tail_conf)
            ginf.mark_bad_strides(tracks, group, cm_per_px=cm_per_px)

            if _any_good_strides(tracks):
                body_len_cm = ginf.median_body_length_cm(group, tracks, cm_per_px=cm_per_px)
                ginf.add_lateral_displacement_to_strides(group, tracks, body_len_cm, cm_per_px=cm_per_px)
                duration_frames, distance_traveled_cm = ginf.get_distance_traveled_cm(
                    base_tail_xy, base_tail_conf, smoothing_window=5, cm_per_px=cm_per_px)

                print('{}: found {} tracks'.format(net_fname, len(tracks)))

                all_strides_summary, binned_strides_summary = ginf.summarize_gait_dict(
                    group,
                    tracks,
                    speed_bin_size=sb_size,
                    angular_velocity_bin_size=avb_size,
                    stride_resolution=100,
                    body_length_cm=body_len_cm,
                    cm_per_px=cm_per_px,
                )

                if video_dict is not None:
                    print('ERROR: FOUND MULTIPLE POSE TRACKING GROUPS IN HDF5 FILE:', net_fname)
                    video_dict = None
                    break
                else:
                    video_dict = {
                        'network_filename': net_fname,
                        'all_strides_summary': all_strides_summary,
                        'binned_strides_summary': binned_strides_summary,
                        'median_body_length_cm': body_len_cm,
                        'distance_traveled_cm': distance_traveled_cm,
                        'duration_secs': duration_frames / ginf.FRAMES_PER_SECOND,
                    }

            else:
                print('WARNING: FAILED TO FIND ANY GOOD STRIDES FOR:', net_fname)
        else:
            print('WARNING: FAILED TO FIND ANY TRACKS FOR:', net_fname)

    return video_dict


def _gen_gait_stats(data_file_dict, root_dir,
                    base_tail_smooth, max_duration_frames, sb_size, avb_size,
                    cm_per_px_mapping):

    try:
        cm_per_px = ginf.CM_PER_PIXEL
        if cm_per_px_mapping is not None:
            if data_file_dict['net_file_name'] in cm_per_px_mapping:
                cm_per_px = cm_per_px_mapping[data_file_dict['net_file_name']]
                print('PIX MAPPING {}: {}'.format(
                    data_file_dict['net_file_name'], cm_per_px))
            else:
                print('NO MAPPING FOR:', data_file_dict['net_file_name'])

        return gen_gait_stats(
            data_file_dict['net_file_name'],
            data_file_dict['data_file_name'],
            data_file_dict['corner_file_name'],
            root_dir,
            base_tail_smooth,
            max_duration_frames,
            sb_size,
            avb_size,
            cm_per_px)
    except:
        # print which video so we have an idea of what caused the problem
        print('Exception while processing:', data_file_dict['net_file_name'])
        raise


def gen_all_gait_stats(data_file_names, root_dir,
                       base_tail_smooth, max_duration_frames, sb_size, avb_size,
                       cm_per_px_mapping, num_procs):

    gen_gait_stats_partial = functools.partial(
        _gen_gait_stats,
        root_dir=root_dir,
        base_tail_smooth=base_tail_smooth,
        max_duration_frames=max_duration_frames,
        sb_size=sb_size,
        avb_size=avb_size,
        cm_per_px_mapping=cm_per_px_mapping,
    )

    if num_procs == 1:
        for data_file_dict in data_file_names:
            video_dict = gen_gait_stats_partial(data_file_dict)
            if video_dict is not None:
                yield video_dict
    else:
        with mp.Pool(num_procs) as p:
            for video_dict in p.imap_unordered(gen_gait_stats_partial, data_file_names):
                if video_dict is not None:
                    yield video_dict


def write_summary(summary, summary_group):

    export_attr_names = [
        'avg_speed_cm_per_sec',
        'median_speed_cm_per_sec',
        'avg_limb_duty_factor',
        'median_limb_duty_factor',
        'avg_temporal_symmetry',
        'median_temporal_symmetry',
        'avg_step_width',
        'median_step_width',
        'avg_step_length1',
        'median_step_length1',
        'avg_step_length2',
        'median_step_length2',
        'avg_stride_length',
        'median_stride_length',
        'avg_angular_velocity',
        'median_angular_velocity',
        'avg_nose_lateral_displacement',
        'median_nose_lateral_displacement',
        'avg_base_tail_lateral_displacement',
        'median_base_tail_lateral_displacement',
        'avg_tip_tail_lateral_displacement',
        'median_tip_tail_lateral_displacement',
        'avg_nose_lateral_displacement_phase',
        'avg_base_tail_lateral_displacement_phase',
        'avg_tip_tail_lateral_displacement_phase',
        'stride_count',
    ]

    export_dataset_names = [
        'speed_cm_per_sec',
        'limb_duty_factor',
        'start_frame',
        # 'frame_count',
        'temporal_symmetry',
        'step_width',
        'step_length1',
        'step_length2',
        'stride_length',
        'angular_velocity',
        'median_position_xy',
        'median_position_proportional_xy',
        'nose_lateral_displacement',
        'base_tail_lateral_displacement',
        'tip_tail_lateral_displacement',
        'nose_lateral_change',
        'base_tail_lateral_change',
        'tip_tail_lateral_change',
        'nose_lateral_displacement_phase',
        'base_tail_lateral_displacement_phase',
        'tip_tail_lateral_displacement_phase',
        'nose_confidence',
        'left_ear_confidence',
        'right_ear_confidence',
        'base_neck_confidence',
        'left_front_paw_confidence',
        'right_front_paw_confidence',
        'center_spine_confidence',
        'left_rear_paw_confidence',
        'right_rear_paw_confidence',
        'base_tail_confidence',
        'mid_tail_confidence',
        'tip_tail_confidence',
    ]

    type_map = {
        'start_frame': np.uint64,
    }

    summary_group['left_rear_hildebrand'] = summary.left_rear_hildebrand
    summary_group['right_rear_hildebrand'] = summary.right_rear_hildebrand

    for export_attr_name in export_attr_names:
        summary_group.attrs[export_attr_name] = getattr(
            summary,
            export_attr_name)

    if summary.all_strides is not None:
        for ds_name in export_dataset_names:
            summary_group[ds_name] = np.array(
                [getattr(s, ds_name) for s in summary.all_strides],
                dtype=(type_map[ds_name] if ds_name in type_map else np.double),
            )

        summary_group['frame_count'] = np.array(
            [len(s) for s in summary.all_strides],
            dtype=np.int32,
        )

    norm_stride_dset = summary_group.create_dataset(
        'normalized_stride_points',
        (len(summary.normalized_stride_points),),
        h5py.special_dtype(vlen=np.double))
    for i, curr_stride_points in enumerate(summary.normalized_stride_points):
        norm_stride_dset[i] = curr_stride_points.flatten()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-file',
        help='the batch file to process.',
        required=True,
    )
    parser.add_argument(
        '--root-dir',
        help='the root directory for the batch file',
        default='.',
    )
    parser.add_argument(
        '--base-tail-smooth',
        help='The window size that should be used for smoothing base tail speed.'
             ' Base tail speed acts as a surrogate for overall mouse speed'
             ' and this smoothing is used to reduce the effect of jitter on'
             ' our estimate of speed.',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--stride-count-thresh',
        help='Tracks must have at least this number of strides to be included'
             ' in analysis.',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--speed-bin-size',
        help='what bin size should be used for mouse speed',
        type=float,
        default=5,
    )
    parser.add_argument(
        '--speed-bin-start',
        help='what speed bin index to we start at (inclusive)',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--speed-bin-stop',
        help='what speed bin index to we stop at (exclusive)',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--angular-velocity-bin-size',
        help='what bin size should be used for mouse angular velocity',
        type=float,
        default=40,
    )
    parser.add_argument(
        '--angular-velocity-bin-count',
        help='how many bins should we use for angular velocity',
        type=float,
        default=11,
    )
    parser.add_argument(
        '--min-strides',
        help='the minimum number of strides for a valid bin',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--out-file',
        help='the output HDF5 file to use',
        default=os.path.join('output', 'gait.h5'),
    )
    parser.add_argument(
        '--stride-resolution',
        help='the resolution to use for strides',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--max-duration-minutes',
        help='do not consider data after the given duration in minutes',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        help="the minimum confidence threshold for strides",
        default=0.3,
    )
    parser.add_argument(
        '--num-procs',
        help='the number of processes to use',
        default=12,
        type=int,
    )
    parser.add_argument(
        '--cm-per-pixel-mapping',
        help='path to the tab delimited file that maps video network IDs to the'
             ' CM_PER_PIXEL value that should be for each respective video. The'
             ' file must consist of two columns and not use a header row. The'
             ' first column is network ID and the second will be the corresponding'
             ' CM_PER_PIXEL value. If this option is not specified, then a default'
             ' value is used of 19.5 * 2.54 / 400 for all videos.'
    )

    args = parser.parse_args()
    ginf.MIN_CONF_THRESH = args.conf_thresh

    sb_size = args.speed_bin_size
    sb_start = args.speed_bin_start
    sb_stop = args.speed_bin_stop
    avb_size = args.angular_velocity_bin_size
    avb_count = args.angular_velocity_bin_count
    bin_tuples_set = set(ginf.gen_speed_and_av_bins(
        sb_size, sb_start, sb_stop,
        avb_size, avb_count))

    max_duration_frames = None
    if args.max_duration_minutes is not None:
        max_duration_frames = ginf.FRAMES_PER_SECOND * 60 * args.max_duration_minutes

    data_file_names = []
    with open(args.batch_file, newline='') as batch_file:
        batch_reader = csv.reader(batch_file, delimiter='\t')
        for row in batch_reader:
            if row:
                net_file_name = row[0]
                if len(row) == 1:
                    data_file_base, _ = os.path.splitext(net_file_name)
                    data_file_name = data_file_base + '_pose_est_v2.h5'
                    corner_file_name = data_file_base + '_corners_v2.yaml'
                    data_file_names.append({
                        'net_file_name': net_file_name,
                        'data_file_name': data_file_name,
                        'corner_file_name': corner_file_name,
                    })
                elif len(row) == 2:
                    data_file_name = row[1]
                    data_file_names.append({
                        'net_file_name': net_file_name,
                        'data_file_name': data_file_name,
                        'corner_file_name': None,
                    })

    cm_per_px_mapping = None
    if args.cm_per_pixel_mapping is not None:
        cm_per_px_mapping = dict()
        with open(args.cm_per_pixel_mapping, newline='') as cm_per_pixel_mapping_file:
            cm_per_pixel_mapping_reader = csv.reader(cm_per_pixel_mapping_file, delimiter='\t')
            for row in cm_per_pixel_mapping_reader:
                if len(row) == 2:
                    net_file_name = row[0]
                    cm_per_pixel = float(row[1])

                    cm_per_px_mapping[net_file_name] = cm_per_pixel

    outdir = os.path.dirname(args.out_file)
    if outdir:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    with h5py.File(args.out_file, 'w') as gait_h5:

        gait_h5.attrs['speed_bin_size'] = args.speed_bin_size
        gait_h5.attrs['speed_bin_start'] = args.speed_bin_start
        gait_h5.attrs['speed_bin_stop'] = args.speed_bin_stop
        gait_h5.attrs['angular_velocity_bin_size'] = args.angular_velocity_bin_size
        gait_h5.attrs['angular_velocity_bin_count'] = args.angular_velocity_bin_count

        for video_dict in gen_all_gait_stats(data_file_names, args.root_dir,
                                             args.base_tail_smooth, max_duration_frames,
                                             sb_size, avb_size, cm_per_px_mapping, args.num_procs):

            escaped_file_name = urlparse.quote(video_dict['network_filename'], safe='')
            vid_grp = gait_h5.create_group(escaped_file_name)
            vid_grp.attrs['median_body_length_cm'] = video_dict['median_body_length_cm']
            vid_grp.attrs['distance_traveled_cm'] = video_dict['distance_traveled_cm']
            vid_grp.attrs['duration_secs'] = video_dict['duration_secs']

            all_strides_group = gait_h5.create_group(escaped_file_name  + '/all_strides')
            write_summary(video_dict['all_strides_summary'], all_strides_group)
            for bin_tuple, curr_summary in video_dict['binned_strides_summary'].items():

                if bin_tuple not in bin_tuples_set:
                    continue

                bin_str = ginf.speed_av_bin_tup_to_str(bin_tuple)
                bin_grp = gait_h5.create_group(escaped_file_name  + '/bins/' + bin_str)

                write_summary(curr_summary, bin_grp)


if __name__ == '__main__':
    main()
