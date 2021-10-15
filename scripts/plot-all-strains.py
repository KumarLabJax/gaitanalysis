# some initial imports and plotting setup

import csv
import argparse
import math
import h5py
# from IPython.display import HTML
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpngw
import numpy as np
import os
import pandas as pd
import seaborn as sns
import urllib.parse as urlparse
import warnings
import pylab


import gaitinference as ginf

# cmap_tups = [
#     (ginf.LEFT_FRONT_PAW_INDEX, 'Greens'),
#     (ginf.RIGHT_FRONT_PAW_INDEX, 'Greens'),
#     (ginf.LEFT_REAR_PAW_INDEX, 'Oranges'),
#     (ginf.RIGHT_REAR_PAW_INDEX, 'Oranges'),
#     (ginf.TIP_TAIL_INDEX, 'Blues'),
#     (ginf.MID_TAIL_INDEX, 'Reds'),
#     (ginf.BASE_TAIL_INDEX, 'Purples'),
#     (ginf.NOSE_INDEX, 'Reds'),
#     (ginf.BASE_NECK_INDEX, 'Blues'),
#     (ginf.CENTER_SPINE_INDEX, 'Reds')]

# cmap_dict = dict(((str(k), v) for k, v in cmap_tups))

# lines = [
#     [ginf.TIP_TAIL_INDEX, ginf.MID_TAIL_INDEX, ginf.BASE_TAIL_INDEX,
#      ginf.CENTER_SPINE_INDEX, ginf.BASE_NECK_INDEX, ginf.NOSE_INDEX],
#     [ginf.LEFT_FRONT_PAW_INDEX, ginf.CENTER_SPINE_INDEX, ginf.RIGHT_FRONT_PAW_INDEX],
#     [ginf.LEFT_REAR_PAW_INDEX, ginf.BASE_TAIL_INDEX, ginf.RIGHT_REAR_PAW_INDEX],
# ]

NUM_INTERP_FRAMES = 60

plotAll = True
# plotFounders = False
# plotFoundersComparison = False
# plotIndividual = False
# plotIndividualAnimals = False
# plotIndividualStrides = False
# plotSameCCLine = False
calcAmpPS = False
calcCCAmpPS = False
calcAmpPerStride = False
plotOverlay = False
plot6j = False
plotBySex = False


def main():
    """parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_files',
        help='the HDF5 file(s) to use for gait inference',
        required=True,
    )

    parser.add_argument(
        '--meta_data',
        help='the meta data Excel',
        default='MergedMetaList_2019-07-09.tsv',
    )


    parser.add_argument(
        '--base_tail_smooth',
        help='The window size that should be used for smoothing base tail speed.'
             ' Base tail speed acts as a surrogate for overall mouse speed'
             ' and this smoothing is used to reduce the effect of jitter on'
             ' our estimate of speed.',
        type=int,
        default=5,
    )

    args = parser.parse_args()"""

    # gait_h5 = h5py.File('strain-survey-gait-2019-11-12.h5', 'r')
    gait_h5 = h5py.File('output/strain-survey-gait-2019-11-12.h5', 'r')

    # metadata = pd.read_csv('MergedMetaList_2019-07-16.tsv', sep='\t')
    metadata = pd.read_csv('data/metadata/MergedMetaList_2021-03-02.tsv', sep='\t')
    metadata_grps = metadata.groupby('Strain')
    
    df = pd.DataFrame()

    taildf = pd.DataFrame()
    strain_t_df = pd.DataFrame()
    # for every strain and info in the meta data group
    for grp_key, metadata_grp in metadata_grps:
        if grp_key in ('NOR/LtJ', 'C57BL/6J', 'AKR/J', 'BALB/cByJ', 'C57BL/6NJ', 'BTBR T<+>ltpr3<tf>/J', 'CAF1/J', 'CBA/CaJ', 'CBA/J', 'LG/J'):
            print(grp_key)
            i = 0
            numberanimals=-1
            while i < len(metadata_grp):
                
                if metadata_grp.loc[:,'TestDataset'].iloc[i] == "StrainSurvey":
                    
                    # in file name
                    for grp_name, grp in gait_h5.items():
                        
                        str_grp_name = str(grp_name).replace("%","/")
                        str_grp_name = str_grp_name.replace("2F", "")
        
                        if metadata_grp.loc[:,'NetworkFilename'].iloc[i] == str_grp_name:
                        
                            print(str_grp_name)
                        
                            if "bins" in grp:
                                
                                # in bins
                                for bins in grp["bins"].values():
                                    str_bins = str(bins)
                                    str_bins = str_bins[len(str_bins)-31:len(str_bins)-15]
                                    if str_bins == "20_ang_vel_neg20":
                                        
                                        numberanimals = numberanimals+1
                                        for bingrp in bins.values():
                                            newbingrp = str(bingrp)
                                            if newbingrp[15:39] == "normalized_stride_points":
                                                curr_strides = ginf.restore_stride_points_shape(bingrp)
                                                interp_strides = np.stack([ginf.interpolate_stride_points(s, NUM_INTERP_FRAMES) for s in curr_strides])
                                                tip_tail_ys = interp_strides[:, :, 11, 1]
                                                tip_tail_ys -= np.repeat(np.mean(tip_tail_ys, axis=1), tip_tail_ys.shape[1]).reshape(tip_tail_ys.shape)
                                                nose_ys = interp_strides[:, :, 0, 1]
                                                nose_ys -= np.repeat(np.mean(nose_ys, axis=1), nose_ys.shape[1]).reshape(nose_ys.shape)
                                                num_strides = tip_tail_ys.shape[0]
                                                time_point = np.tile(np.arange(NUM_INTERP_FRAMES), num_strides)
                                                
                                                curr_tip_tail_lat_df = pd.DataFrame({
                                                    'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                                                    'Displacement': tip_tail_ys.flatten(),
                                                    'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                                                    'Mouse Line': grp_key,
                                                    'Speed': 20,
                                                    'Animal Number':numberanimals,
                                                    'Sex': metadata_grp.loc[:,'Sex'].iloc[i],
                                                })
                                                curr_nose_lat_df = pd.DataFrame({
                                                    'Percent Stride': np.tile(100.0 * np.arange(NUM_INTERP_FRAMES) / NUM_INTERP_FRAMES, num_strides),
                                                    'Displacement': nose_ys.flatten(),
                                                    'stride_num': np.repeat(np.arange(num_strides), NUM_INTERP_FRAMES),
                                                    'Mouse Line': grp_key,
                                                    'Speed': 20,
                                                    'Animal Number':numberanimals,
                                                    'Sex': metadata_grp.loc[:,'Sex'].iloc[i],
                                                })
                                            
                                                    
                                                if grp_key == 'C57BL/6J':
    
                                                    wt_panel_key = (20, -20, 'B6', 'Founder')                                            
                                                    wt_speed, wt_ang_vel, wt_geno, wt_mut = wt_panel_key
                                                    wt_strides = curr_strides
                                                    wt_interp_strides = interp_strides
    
                                            
                                                if grp_key == 'NOR/LtJ':
                                                    mut_panel_key = (20, -20, 'NOR', 'Founder')
                                                    mut_speed, mut_ang_vel, mut_geno, mut_mut = mut_panel_key
                                                    mut_strides = curr_strides
                                                    mut_interp_strides = interp_strides
                                                
    
    
                                                df = df.append(curr_nose_lat_df, ignore_index=True)
                                                taildf = taildf.append(curr_tip_tail_lat_df, ignore_index=True)
                                                        
                i = i+1


    print('Plotting for tail')
    ax = sns.lineplot(x="Percent Stride", y="Displacement", hue="Mouse Line", data=taildf)
    ax.get_legend().remove()
    ax.set(xlabel='Percent Stride', ylabel='Lateral Tip of Tail Displacement')
    ax.set_title('Strain Survey Variation in Lateral Tip of Tail Displacement')
    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    file_name = "newtail-disp-plot-legend.pdf"
    ax.get_figure().savefig(file_name, bbox_inches='tight')
    plt.close()
    
                        
    print('Plotting for nose')
    ax = sns.lineplot(x="Percent Stride", y="Displacement", hue="Mouse Line", data=df)
    ax.get_legend().remove()
    ax.set(xlabel='Percent Stride', ylabel='Lateral Nose Displacement')
    ax.set_title('Strain Survey Variation in Lateral Nose Displacement')
    file_name = "nose-disp-plot.pdf"
    ax.get_figure().savefig(file_name, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
