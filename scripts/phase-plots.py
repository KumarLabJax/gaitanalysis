import os
import re
import sys
#os.chdir("/Users/sabnig/Documents/Projects/Komp/Temp")
import csv
import h5py
import itertools
import math
import matplotlib
if __name__ == "__main__":
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.parse as urlparse
import gaitinference as ginf
import gaitplt as gplt

NUM_INTERP_FRAMES = 60
#NUM_INTERP_FRAMES = 2

def main():

    cmap_tups = [
        (ginf.LEFT_FRONT_PAW_INDEX, 'Greens'),
        (ginf.RIGHT_FRONT_PAW_INDEX, 'Greens'),
        (ginf.LEFT_REAR_PAW_INDEX, 'Oranges'),
        (ginf.RIGHT_REAR_PAW_INDEX, 'Oranges'),
        (ginf.TIP_TAIL_INDEX, 'Blues'),
        (ginf.MID_TAIL_INDEX, 'Reds'),
        (ginf.BASE_TAIL_INDEX, 'Purples'),
        (ginf.NOSE_INDEX, 'Reds'),
        (ginf.BASE_NECK_INDEX, 'Blues'),
        (ginf.CENTER_SPINE_INDEX, 'Reds'), #'Greys'),
    ]
    cmap_dict = dict(((str(k), v) for k, v in cmap_tups))

    lines = [
        [ginf.TIP_TAIL_INDEX, ginf.MID_TAIL_INDEX, ginf.BASE_TAIL_INDEX,
        ginf.CENTER_SPINE_INDEX, ginf.BASE_NECK_INDEX, ginf.NOSE_INDEX],
        [ginf.LEFT_FRONT_PAW_INDEX, ginf.CENTER_SPINE_INDEX, ginf.RIGHT_FRONT_PAW_INDEX],
        [ginf.LEFT_REAR_PAW_INDEX, ginf.BASE_TAIL_INDEX, ginf.RIGHT_REAR_PAW_INDEX],
    ]


    # gait_h5 = h5py.File('../Data/KOMP-curated-2020-08-20.h5','r')
    # metadata = pd.read_excel('../Data/KOMP-LinkedData.xlsx')
    gait_h5 = h5py.File('output/KOMP-curated-2020-08-20.h5','r')
    metadata = pd.read_excel('data/metadata/KOMP-LinkedData.xlsx')

    metadata = metadata.loc[metadata['NetworkFilename'].isin(urlparse.unquote(k) for k in gait_h5.keys()), :]
    metadata['median_body_length_cm'] = float('nan')
    for grp_name, grp in gait_h5.items():
        metadata.loc[metadata['NetworkFilename'] == urlparse.unquote(grp_name), 'median_body_length_cm'] = grp.attrs['median_body_length_cm']

    stride_resolution = 100
    def stride_bin_to_rad(stride_bin):
        return math.radians(3.6 * stride_bin)

    angular_velocity_bin_size = gait_h5.attrs['angular_velocity_bin_size']

    center_av_bin = -angular_velocity_bin_size // 2
    speed = 30

    # we only want to look at the centered angular velocity bin here
    # angular_velocity_bin_count = 1

    bin_count_dict = dict()
    net_filenames_dict = dict()
    norm_stride_pts_dict = dict()

    # we need to segregate the files by strain
    metadata = metadata.rename(columns = {"OFA_Genotype":"Strain","Mouse.ID":"MouseID"})
    metadata.Strain = metadata.Strain.fillna('C57BL/6NJ')
    metadata.Strain = [re.sub("<.*>","",x) for x in metadata.Strain]
    metadata.Strain = [re.sub(" ","",x) for x in metadata.Strain]
    metadata_grps = metadata.groupby('Strain')
    i = 0
    for grp_key, metadata_grp in metadata_grps:
        curr_speed = 20
        curr_av = center_av_bin
        group_dict_key = grp_key
        bin_str = ginf.speed_av_bin_tup_to_str((curr_speed, curr_av))
        for net_filename in metadata_grp['NetworkFilename']:
            escaped_file_name = urlparse.quote(net_filename, safe='')
            bin_path = escaped_file_name + '/bins/' + bin_str
            if bin_path in gait_h5:
                bin_grp = gait_h5[bin_path]
                both_paw_hild = np.stack([bin_grp['left_rear_hildebrand'], bin_grp['right_rear_hildebrand']])
                if group_dict_key not in bin_count_dict:
                    bin_count_dict[group_dict_key] = 1
                    norm_stride_pts_dict[group_dict_key] = [list(bin_grp['normalized_stride_points'])]
                    net_filenames_dict[group_dict_key] = [net_filename]
                else:
                    bin_count_dict[group_dict_key] += 1
                    norm_stride_pts_dict[group_dict_key].append(list(bin_grp['normalized_stride_points']))
                    net_filenames_dict[group_dict_key].append(net_filename)
        
        i += 1
        if i == 50:
            break

    ordered_keys = sorted(bin_count_dict.keys())


    tip_tail_lat_df_dict = dict()
    nose_lat_df_dict = dict()
    base_tail_lat_df_dict = dict()

    for curr_key in ordered_keys:
        # speed, ang_vel, strain = curr_key
        
        # if ang_vel != center_av_bin:
        #     continue
        
        # if speed not in (10,15,20,25,30):
        #     continue
        
        # if strain not in ('Stmn4-/-','C57BL/6NJ'):
        #     continue

        # print('=================================')
        # print('speed: {}; strain: {}; count: {}'.format(
        #     speed, strain, bin_count_dict[curr_key]))
        strain = curr_key
        print('strain:', strain)


        if strain == 'C57BL/6NJ':
            curr_strides = ginf.restore_stride_points_shape(list(itertools.chain.from_iterable(norm_stride_pts_dict[curr_key])))
            interp_strides = np.stack(ginf.interpolate_stride_points(s, NUM_INTERP_FRAMES) for s in curr_strides)
            tip_tail_ys = interp_strides[:, :, 11, 1]
            tip_tail_ys -= np.repeat(np.mean(tip_tail_ys, axis=1), tip_tail_ys.shape[1]).reshape(tip_tail_ys.shape)
            nose_ys = interp_strides[:, :, 0, 1]
            nose_ys -= np.repeat(np.mean(nose_ys, axis=1), nose_ys.shape[1]).reshape(nose_ys.shape)
            base_tail_ys = interp_strides[:, :, 9, 1]
            base_tail_ys -= np.repeat(np.mean(base_tail_ys, axis=1), base_tail_ys.shape[1]).reshape(base_tail_ys.shape)
            num_strides = tip_tail_ys.shape[0]
            time_point = np.tile(np.arange(NUM_INTERP_FRAMES), num_strides)
            
            curr_tip_tail_lat_df = pd.DataFrame({
                'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                'Displacement': tip_tail_ys.flatten(),
                'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                'Mouse Line': strain,
                'label': strain,
            })
            curr_nose_lat_df = pd.DataFrame({
                'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                'Displacement': nose_ys.flatten(),
                'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                'Mouse Line': strain,
                'label': strain,
            })
            curr_base_tail_lat_df = pd.DataFrame({
                'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                'Displacement': base_tail_ys.flatten(),
                'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                'Mouse Line': strain,
                'label': strain,
            })

            tip_tail_lat_df_dict[strain] = curr_tip_tail_lat_df
            base_tail_lat_df_dict[strain] = curr_base_tail_lat_df
            nose_lat_df_dict[strain] = curr_nose_lat_df
        else:
            for i, curr_norm_stride_pts in enumerate(norm_stride_pts_dict[curr_key]):
                curr_strides = ginf.restore_stride_points_shape(curr_norm_stride_pts)
                interp_strides = np.stack(ginf.interpolate_stride_points(s, NUM_INTERP_FRAMES) for s in curr_strides)
                tip_tail_ys = interp_strides[:, :, 11, 1]
                tip_tail_ys -= np.repeat(np.mean(tip_tail_ys, axis=1), tip_tail_ys.shape[1]).reshape(tip_tail_ys.shape)
                nose_ys = interp_strides[:, :, 0, 1]
                nose_ys -= np.repeat(np.mean(nose_ys, axis=1), nose_ys.shape[1]).reshape(nose_ys.shape)
                base_tail_ys = interp_strides[:, :, 9, 1]
                base_tail_ys -= np.repeat(np.mean(base_tail_ys, axis=1), base_tail_ys.shape[1]).reshape(base_tail_ys.shape)
                num_strides = tip_tail_ys.shape[0]
                time_point = np.tile(np.arange(NUM_INTERP_FRAMES), num_strides)
                
                curr_tip_tail_lat_df = pd.DataFrame({
                    'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                    'Displacement': tip_tail_ys.flatten(),
                    'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                    'Mouse Line': strain,
                    'label': net_filenames_dict[strain][i],
                })
                curr_nose_lat_df = pd.DataFrame({
                    'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                    'Displacement': nose_ys.flatten(),
                    'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                    'Mouse Line': strain,
                    'label': net_filenames_dict[strain][i],
                })
                curr_base_tail_lat_df = pd.DataFrame({
                    'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                    'Displacement': base_tail_ys.flatten(),
                    'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                    'Mouse Line': strain,
                    'label': net_filenames_dict[strain][i],
                })

                if i == 0:
                    tip_tail_lat_df_dict[strain] = [curr_tip_tail_lat_df]
                    base_tail_lat_df_dict[strain] = [curr_base_tail_lat_df]
                    nose_lat_df_dict[strain] = [curr_nose_lat_df]
                else:
                    tip_tail_lat_df_dict[strain].append(curr_tip_tail_lat_df)
                    base_tail_lat_df_dict[strain].append(curr_base_tail_lat_df)
                    nose_lat_df_dict[strain].append(curr_nose_lat_df)

            tip_tail_lat_df_dict[strain] = pd.concat(tip_tail_lat_df_dict[strain], ignore_index=True)
            base_tail_lat_df_dict[strain] = pd.concat(base_tail_lat_df_dict[strain], ignore_index=True)
            nose_lat_df_dict[strain] = pd.concat(nose_lat_df_dict[strain], ignore_index=True)

    for strain in ordered_keys:
        if strain == 'C57BL/6NJ':
            continue

        print('Plots for strain:', strain)

        ctrl_tip_tail_lat_dfs = [v for k, v in tip_tail_lat_df_dict.items() if k in (strain, 'C57BL/6NJ')]
        plt.figure()
        ax = gplt.plot_lateral_disp(
            pd.concat(ctrl_tip_tail_lat_dfs, ignore_index=True),
            'Tip of Tail Lateral Displacement at {} cm/sec'.format(speed),
            condition='label',
        )
        file_name = 'tip_tail_lat_disp_{0}.pdf'.format(re.sub("./.","",strain))
        file_name = os.path.join('temp-komp-plots', file_name)
        ax.get_figure().savefig(file_name, bbox_inches='tight')

        ctrl_nose_lat_dfs = [v for k, v in nose_lat_df_dict.items() if k in (strain, 'C57BL/6NJ')]
        plt.figure()
        ax = gplt.plot_lateral_disp(
            pd.concat(ctrl_nose_lat_dfs, ignore_index=True),
            'Nose Lateral Displacement at {} cm/sec'.format(speed),
            condition='label',
        )
        file_name = 'nose_lat_disp_{0}.pdf'.format(re.sub("./.","",strain))
        file_name = os.path.join('temp-komp-plots', file_name)
        ax.get_figure().savefig(file_name, bbox_inches='tight')

        ctrl_base_tail_lat_dfs = [v for k, v in base_tail_lat_df_dict.items() if k in (strain, 'C57BL/6NJ')]
        ctrl_base_tail_lat_dfs = pd.concat(ctrl_base_tail_lat_dfs, ignore_index=True)
        plt.figure()
        ax = gplt.plot_lateral_disp(
            ctrl_base_tail_lat_dfs,
            'Base of Tail Lateral Displacement at {} cm/sec'.format(speed),
            condition='label',
        )
        file_name = 'base_tail_lat_disp_{}.pdf'.format(re.sub("./.","",strain))
        file_name = os.path.join('temp-komp-plots', file_name)
        ax.get_figure().savefig(file_name, bbox_inches='tight')


if __name__ == '__main__':
    main()
