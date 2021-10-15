import argparse
import h5py
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl

# if __name__ == "__main__":
#     mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from svgwrite import px
import svgwrite
import urllib.parse as urlparse

import gaitinference as ginf


# Define the Tuples For Mutant vs. Control w.r.t. Gait Mutants
mutant_ctrl_pairs = [
    (
        ('Dixdc1 KO', 'HOM'),
        ('C57BL/6NJ', 'WT'),
    ),
    (
        ('Mecp2-', 'HEMI'),
        ('Mecp2-', 'WT'),
    ),
    (
        ('Mecp2-', 'HET'),
        ('Mecp2-', 'WT'),
    ),
    (
        ('B6.SOD1-G93A', 'HEMI'),
        ('B6.SOD1-G93A', 'WT'),
    ),
    (
        ('B6C3.B-Ts65Dn', 'Trisomic'),
        ('B6EiC3Sn.BLiAF1/J', 'F1 Ctrl'),
    ),
]


def plot_lateral_disp(data, title, condition='Mouse Line'):
    ax = sns.lineplot(
        x='Percent Stride',
        y='Displacement',
        hue=condition,
        data=data,
    ).set_title(title)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return ax


def plot_lateral_disp2(data, title, condition='Mouse Line'):
    ax = sns.lineplot(
        x='Percent Stride',
        y='Displacement',
        hue='Speed',
        hue_order=sorted(set(data['Speed'])),
        style=condition,
        data=data,
        palette=sns.cubehelix_palette(len(set(data['Speed']))),
    ).set_title(title)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return ax


# plotting functions we'll use throughout
def strip_box_plt(x, y, hue, data, title, order=None, figsize=(15.0, 6.0)):
    plt.figure(figsize=figsize)

    data = data.copy()
    bp = sns.boxplot(x, y, data=data, showfliers=False, order=order)
    bp.set_title(title)
    handles, labels = bp.get_legend_handles_labels()
    stripplt = sns.stripplot(x, y, data=data, jitter=True, dodge=True, linewidth=1, edgecolor='gray', order=order)
    stripplt.set_xticklabels(stripplt.get_xticklabels(), rotation=90)
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)    


def strip_box_plt_attr(metadata, gait_h5, title, bin_str, attr_key, norm=False):
    md_copy = metadata.copy()
    md_copy[bin_str + '-' + attr_key] = float('nan')
    for grp_name, grp in gait_h5.items():
        if 'bins/' + bin_str in grp:
            if norm:
                md_copy.loc[md_copy['NetworkFilename'] == urlparse.unquote(grp_name), bin_str + '-' + attr_key] = (
                    grp['bins/' + bin_str].attrs[attr_key] / grp.attrs['median_body_length_cm']
                )
            else:
                md_copy.loc[md_copy['NetworkFilename'] == urlparse.unquote(grp_name), bin_str + '-' + attr_key] = (
                    grp['bins/' + bin_str].attrs[attr_key]
                )

    mut_by_geno_by_sex_by_test = (
        md_copy['Mutant'] + 'x' + md_copy['Genotype'] + 'x'
        + md_copy['Sex'] + 'x' + md_copy['TestNum'].map(str)
    )
    md_copy['Mutant x Genotype x Sex x Test#'] = mut_by_geno_by_sex_by_test

    strip_box_plt(
        x='Mutant x Genotype x Sex x Test#',
        y=bin_str + '-' + attr_key,
        hue='Genotype',
        data=md_copy,
        title=title,
        order=list(sorted(set(mut_by_geno_by_sex_by_test))))


def strip_box_plt_mut_ctrl(metadata, gait_h5, title, bin_str, attr_key, norm=False):
    md_copy = metadata.copy()
    md_copy[bin_str + '-' + attr_key] = float('nan')
    mut_by_geno_by_sex_by_test = (
        md_copy['Mutant'] + 'x' + md_copy['Genotype'] + 'x'
        + md_copy['Sex'] + 'x' + md_copy['TestNum'].map(str)
    )
    md_copy['Mutant x Genotype x Sex x Test#'] = mut_by_geno_by_sex_by_test

    for grp_name, grp in gait_h5.items():
        if 'bins/' + bin_str in grp:
            if norm:
                md_copy.loc[md_copy['NetworkFilename'] == urlparse.unquote(grp_name), bin_str + '-' + attr_key] = (
                    grp['bins/' + bin_str].attrs[attr_key] / grp.attrs['median_body_length_cm']
                )
            else:
                md_copy.loc[md_copy['NetworkFilename'] == urlparse.unquote(grp_name), bin_str + '-' + attr_key] = (
                    grp['bins/' + bin_str].attrs[attr_key]
                )

    for sex in ['M', 'F']:
        for mut_tuple, ctrl_tuple in mutant_ctrl_pairs:
            mut_line, mut_geno = mut_tuple
            ctrl_line, ctrl_geno = ctrl_tuple

            mut_count = ((md_copy['Sex'] == sex) & (md_copy['Mutant'] == mut_line) & (md_copy['Genotype'] == mut_geno)).sum()
            print('Mut Count:', mut_count)
            ctrl_count = ((md_copy['Sex'] == sex) & (md_copy['Mutant'] == ctrl_line) & (md_copy['Genotype'] == ctrl_geno)).sum()
            print('Ctrl Count:', ctrl_count)

            if mut_count >= 1 and ctrl_count >= 1:
                filter_cond = (md_copy['Sex'] == sex) & (
                    ((md_copy['Mutant'] == mut_line) & (md_copy['Genotype'] == mut_geno))
                    | ((md_copy['Mutant'] == ctrl_line) & (md_copy['Genotype'] == ctrl_geno))
                )
                data = md_copy.loc[filter_cond, :]
                strip_box_plt(
                    x='Mutant x Genotype x Sex x Test#',
                    #x='Mutant',
                    y=bin_str + '-' + attr_key,
                    hue='Genotype',
                    data=data,
                    title=title,
                    #order=list(sorted(set(mut_by_geno_by_sex_by_test)))
                    figsize=(6.0, 6.0),
                )
                plt.show()


PLOT_WINDOW_SIZE = 100
SMOOTHING_WINDOW = 5

def plot_ang_vel(data_group, frame_index, fig_width, fig_height, fig_title):

    base_tail_speed = ginf.calc_speed(
        data_group,
        ginf.BASE_TAIL_INDEX,
        smoothing_window=SMOOTHING_WINDOW)
    left_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    # left_fore_paw_speed = ginf.calc_speed(
    #     data_group,
    #     ginf.LEFT_FRONT_PAW_INDEX)
    # right_fore_paw_speed = ginf.calc_speed(
    #     data_group,
    #     ginf.RIGHT_FRONT_PAW_INDEX)
    angle_deg = ginf.calc_angle_deg(data_group)
    angular_speed = list(ginf.calc_angle_speed_deg(
        angle_deg,
        smoothing_window=SMOOTHING_WINDOW))

    left_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    base_tail_conf = ginf.get_conf(
        data_group,
        ginf.BASE_TAIL_INDEX)

    tracks = list(ginf.trackstridedet(
        left_rear_paw_speed,
        right_rear_paw_speed,
        base_tail_speed,
        angular_speed))
    ginf.add_conf_to_strides(data_group, tracks)
    ginf.add_conf_to_tracks(
        tracks,
        left_rear_paw_conf,
        right_rear_paw_conf,
        base_tail_conf)
    ginf.mark_bad_strides(tracks, data_group)

    frame_count = len(base_tail_speed)
    plot_range_start = max(0, frame_index - PLOT_WINDOW_SIZE // 2)
    plot_range_stop = min(frame_count, frame_index + PLOT_WINDOW_SIZE // 2)

    # Create the figure we want to add to an existing canvas
    fig = mpl.figure.Figure(
        figsize=(fig_width / 100, fig_height / 100),
        dpi=100)

    ax = fig.add_subplot(111)
    ax.set_xlim([plot_range_start, plot_range_stop])
    ax.plot(
        np.arange(plot_range_start, plot_range_stop),
        angular_speed[plot_range_start:plot_range_stop],
    )
    ax.axvline(x=frame_index, color='r')
    ax.axhline(y=0, color='k')
    # ax.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    ax.set_title(fig_title)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Angular Velocity (deg/sec)')

    return fig


def plot_paw_speeds(data_group, frame_index, fig_width, fig_height, fig_title, fore_paws=False):

    base_tail_speed = ginf.calc_speed(
        data_group,
        ginf.BASE_TAIL_INDEX,
        smoothing_window=SMOOTHING_WINDOW)
    left_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    left_fore_paw_speed = ginf.calc_speed(
        data_group,
        ginf.LEFT_FRONT_PAW_INDEX)
    right_fore_paw_speed = ginf.calc_speed(
        data_group,
        ginf.RIGHT_FRONT_PAW_INDEX)
    angle_deg = ginf.calc_angle_deg(data_group)
    angular_speed = list(ginf.calc_angle_speed_deg(
        angle_deg,
        smoothing_window=SMOOTHING_WINDOW))

    left_rear_paw_xy = ginf.get_xy_pos(data_group, ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_xy = ginf.get_xy_pos(data_group, ginf.RIGHT_REAR_PAW_INDEX)

    left_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    base_tail_conf = ginf.get_conf(
        data_group,
        ginf.BASE_TAIL_INDEX)

    tracks = list(ginf.trackstridedet(
        left_rear_paw_speed,
        right_rear_paw_speed,
        base_tail_speed,
        angular_speed))

    if tracks:
        ginf.add_xy_pos_to_strides(tracks, left_rear_paw_xy, right_rear_paw_xy)
        ginf.add_conf_to_strides(data_group, tracks)
        ginf.add_conf_to_tracks(
            tracks,
            left_rear_paw_conf,
            right_rear_paw_conf,
            base_tail_conf)
        ginf.mark_bad_strides(tracks, data_group)

    frame_count = len(base_tail_speed)
    plot_range_start = max(0, frame_index - PLOT_WINDOW_SIZE // 2)
    plot_range_stop = min(frame_count, frame_index + PLOT_WINDOW_SIZE // 2)

    # Create the figure we want to add to an existing canvas
    fig = mpl.figure.Figure(
        figsize=(fig_width / 100, fig_height / 100),
        dpi=100)

    if fore_paws:
        speed_points = np.transpose(np.stack((
            right_fore_paw_speed[plot_range_start:plot_range_stop],
            left_fore_paw_speed[plot_range_start:plot_range_stop],
            base_tail_speed[plot_range_start:plot_range_stop],
        )))
    else:
        speed_points = np.transpose(np.stack((
            right_rear_paw_speed[plot_range_start:plot_range_stop],
            left_rear_paw_speed[plot_range_start:plot_range_stop],
            base_tail_speed[plot_range_start:plot_range_stop],
        )))

    # ax = fig.add_subplot(212)
    ax = fig.add_subplot(111)

    ax.set_title(fig_title)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Speed (cm/sec)')

    ax.set_xlim([plot_range_start, plot_range_stop])
    ax.plot(
        np.arange(plot_range_start, plot_range_stop),
        speed_points,
    )
    ax.axvline(x=frame_index, color='r')

    for track in tracks:
        if track.stop_frame_exclu > plot_range_start and track.start_frame < plot_range_stop:
            ax.axvspan(xmin=track.start_frame, xmax=track.stop_frame_exclu, color='0.8')

            for step in track.lrp_steps:
                ax.axvspan(
                    xmin=step.start_frame, xmax=step.stop_frame_exclu,
                    # ymin=0.51, ymax=0.75,
                    ymin=0.0, ymax=0.05,
                    color='k')

            for step in track.rrp_steps:
                ax.axvspan(
                    xmin=step.start_frame, xmax=step.stop_frame_exclu,
                    # ymin=0.25, ymax=0.49,
                    ymin=0.05, ymax=0.1,
                    color='k')

            for stride in track.good_strides:
                ax.axvline(x=stride.start_frame, color='black')

    return fig


def plot_path_svg(dwg, data_group, frame_index, render_base_tail=True):

    base_tail_speed = ginf.calc_speed(
        data_group,
        ginf.BASE_TAIL_INDEX,
        smoothing_window=SMOOTHING_WINDOW)
    left_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_speed = ginf.calc_speed(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    angle_deg = ginf.calc_angle_deg(data_group)
    angular_speed = list(ginf.calc_angle_speed_deg(
        angle_deg,
        smoothing_window=SMOOTHING_WINDOW))

    left_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.LEFT_REAR_PAW_INDEX)
    right_rear_paw_conf = ginf.get_conf(
        data_group,
        ginf.RIGHT_REAR_PAW_INDEX)
    base_tail_conf = ginf.get_conf(
        data_group,
        ginf.BASE_TAIL_INDEX)

    tracks = list(ginf.trackstridedet(
        left_rear_paw_speed,
        right_rear_paw_speed,
        base_tail_speed,
        angular_speed))
    ginf.add_conf_to_strides(data_group, tracks)
    ginf.add_conf_to_tracks(
        tracks,
        left_rear_paw_conf,
        right_rear_paw_conf,
        base_tail_conf)
    ginf.mark_bad_strides(tracks, data_group)

    frame_count = len(base_tail_speed)
    plot_range_start = max(0, frame_index - PLOT_WINDOW_SIZE // 2)
    plot_range_stop = min(frame_count, frame_index + PLOT_WINDOW_SIZE // 2)

    # point_dists = ginf.point_distances(
    #     data_group['points'][plot_range_start : plot_range_stop, ...])
    # lat_dist = point_dists['lat_dist']
    # paw_dist = point_dists['paw_dist']

    rrp_xy_pos = data_group['points'][plot_range_start : plot_range_stop, ginf.RIGHT_REAR_PAW_INDEX, :]
    rrp_path_commands = []
    for x_pos, y_pos in rrp_xy_pos:
        if rrp_path_commands:
            rrp_path_commands.append('L {},{}'.format(y_pos, x_pos))
        else:
            rrp_path_commands.append('M {},{}'.format(y_pos, x_pos))
    dwg.add(dwg.path(d=' '.join(rrp_path_commands), stroke='#ff7f0e', stroke_width=2, fill='none'))

    lrp_xy_pos = data_group['points'][plot_range_start : plot_range_stop, ginf.LEFT_REAR_PAW_INDEX, :]
    lrp_path_commands = []
    for x_pos, y_pos in lrp_xy_pos:
        if lrp_path_commands:
            lrp_path_commands.append('L {},{}'.format(y_pos, x_pos))
        else:
            lrp_path_commands.append('M {},{}'.format(y_pos, x_pos))
    dwg.add(dwg.path(d=' '.join(lrp_path_commands), stroke='#1f77b4', stroke_width=2, fill='none'))

    if render_base_tail:
        basetail_xy_pos = data_group['points'][plot_range_start : plot_range_stop, ginf.BASE_TAIL_INDEX, :]
        basetail_path_commands = []
        for x_pos, y_pos in basetail_xy_pos:
            if basetail_path_commands:
                basetail_path_commands.append('L {},{}'.format(y_pos, x_pos))
            else:
                basetail_path_commands.append('M {},{}'.format(y_pos, x_pos))
        dwg.add(dwg.path(d=' '.join(basetail_path_commands), stroke='#2ca02c', stroke_width=2, fill='none'))

    # pose_est_points_grp = dwg.add(dwg.g(id='pose_est_points', stroke='#000', stroke_width=2, fill='none'))
    # curr_frame_points = data_group['points'][frame_index, :, :]

    # pose_indexes = [
    #     ginf.NOSE_INDEX,
    #     ginf.BASE_NECK_INDEX,
    #     ginf.CENTER_SPINE_INDEX,
    #     ginf.BASE_TAIL_INDEX,
    #     ginf.RIGHT_REAR_PAW_INDEX,
    #     ginf.LEFT_REAR_PAW_INDEX,
    #     ginf.RIGHT_FRONT_PAW_INDEX,
    #     ginf.LEFT_FRONT_PAW_INDEX,
    # ]
    # for pose_index in pose_indexes:
    #     pose_est_points_grp.add(dwg.circle(
    #         center=(
    #             curr_frame_points[pose_index, 1] * px,
    #             curr_frame_points[pose_index, 0] * px,
    #         ),
    #         r=4 * px,
    #     ))

    stance_points_grp = dwg.add(dwg.g(id='stance_points', stroke='#000', stroke_width=2, fill='none'))

    for track in tracks:
        if track.stop_frame_exclu > plot_range_start and track.start_frame < plot_range_stop:
            for stances in track.lrp_stances:
                if (stances.start_frame >= plot_range_start
                        and stances.start_frame < plot_range_stop):

                    curr_frame_points = data_group['points'][stances.start_frame, :, :]
                    stance_points_grp.add(dwg.circle(
                        center=(
                            curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 1] * px,
                            curr_frame_points[ginf.LEFT_REAR_PAW_INDEX, 0] * px,
                        ),
                        r=4 * px,
                    ))

            for stances in track.rrp_stances:
                if (stances.start_frame >= plot_range_start
                        and stances.start_frame < plot_range_stop):

                    curr_frame_points = data_group['points'][stances.start_frame, :, :]
                    stance_points_grp.add(dwg.circle(
                        center=(
                            curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 1] * px,
                            curr_frame_points[ginf.RIGHT_REAR_PAW_INDEX, 0] * px,
                        ),
                        r=4 * px,
                    ))


# python gaitplt.py \
#   --hdf5-file data/all_inferences/LL1-B2B+2016-04-08_SPD+LL1-4_100007-M-AX4-8-42402-4_APR_04-02-2018_inference.h5 \
#   --video-file data/video/LL1-4_100007-M-AX4-8-42402-4.avi --out-dir output --frame-number 3281

# python gaitplt.py \
#   --hdf5-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151_pose_est.h5' \
#   --video-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151.avi' \
#   --out-dir output \
#   --frame-number 1365

# share_root="/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar"
# python -u gaitplt.py \
#   --hdf5-file "${share_root}/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151_pose_est_v2.h5" \
#   --video-file "${share_root}/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151.avi" \
#   --out-dir temp \
#   --frame-number 1365 \
#   --plot-width 600 --plot-height 200

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hdf5-file',
        help='HDF5 input file that gives the mouse positions for each frame',
        required=True,
    )
    parser.add_argument(
        '--video-file',
        help='video file to open (you can also just open from the file menu)',
        required=True,
    )
    parser.add_argument(
        '--out-dir',
        help='output dir that all plot files are written to',
        required=True,
    )
    parser.add_argument(
        '--plot-width',
        type=int,
        help='the width to use for plots',
        default=1200,
    )
    parser.add_argument(
        '--plot-height',
        type=int,
        help='the height to use for plots',
        default=400,
    )
    parser.add_argument(
        '--frame-number',
        type=int,
        help='the frame number',
        required=True,
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        help="the minimum confidence threshold for strides",
        # default=75.0,
        default=0.3,
    )

    args = parser.parse_args()
    ginf.MIN_CONF_THRESH = args.conf_thresh

    os.makedirs(args.out_dir, exist_ok=True)

    gait_h5 = h5py.File(args.hdf5_file, 'r')
    vid_grp = next(iter(gait_h5.values()))

    plt.rcParams['svg.fonttype'] = 'none'
    # title_font = {
    #     'fontname':'Arial',
    #     'size':'16',
    #     'color':'black',
    #     'weight':'normal',
    #     'verticalalignment':'bottom',
    # }
    # axis_font = {'fontname':'Arial', 'size':'14'}

    fig = plot_paw_speeds(vid_grp, args.frame_number, args.plot_width, args.plot_height, 'Forepaw Speed', True)
    canvas = FigureCanvas(fig)
    canvas.print_figure(os.path.join(args.out_dir, 'forepaw-speed.svg'))

    fig = plot_paw_speeds(vid_grp, args.frame_number, args.plot_width, args.plot_height, 'Hind Paw Speed')
    canvas = FigureCanvas(fig)
    canvas.print_figure(os.path.join(args.out_dir, 'hind-paw-speed.svg'))

    fig = plot_ang_vel(vid_grp, args.frame_number, args.plot_width, args.plot_height, 'Angular Velocity')
    canvas = FigureCanvas(fig)
    canvas.print_figure(os.path.join(args.out_dir, 'angular-speed.svg'))

    image_reader = imageio.get_reader(args.video_file)
    image = image_reader.get_data(args.frame_number)
    print(os.path.join(args.out_dir, 'frame.png'))
    imageio.imwrite(os.path.join(args.out_dir, 'frame.png'), image)

    dwg = svgwrite.Drawing(os.path.join(args.out_dir, 'stride-overlay.svg'), size=image.shape[:2])
    plot_path_svg(dwg, vid_grp, args.frame_number)
    dwg.save()


if __name__ == "__main__":
    main()
