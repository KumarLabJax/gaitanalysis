import argparse
import csv
import h5py
import numpy as np
import urllib.parse as urlparse


ALL_STRIDES_MEASURES_TO_SUMMARIZE = [
    'speed_cm_per_sec',
]

BINNED_MEASURES_TO_SUMMARIZE = [
    'angular_velocity',
    'base_tail_lateral_displacement',
    # 'base_tail_lateral_displacement_phase',
    'limb_duty_factor',
    'nose_lateral_displacement',
    # 'nose_lateral_displacement_phase',
    'step_length1',
    # 'step_length2',
    'step_width',
    'stride_length',
    'temporal_symmetry',
    'tip_tail_lateral_displacement',
    # 'tip_tail_lateral_displacement_phase',
]

VIDEO_ATTRIBUTES = [
    'distance_traveled_cm',
    # 'duration_secs',
    'median_body_length_cm',
]


NAME_MAPPINGS = {
    'angular_velocity': 'Angular Velocity',
    'base_tail_lateral_displacement': 'Amplitude Tail Base',
    # 'base_tail_lateral_displacement_phase',
    'limb_duty_factor': 'Limb Duty',
    'nose_lateral_displacement': 'Amplitude Nose',
    # 'nose_lateral_displacement_phase',
    'speed_cm_per_sec': 'Speed',
    'step_length1': 'Step Length',
    # 'step_length2',
    'step_width': 'Step Width',
    'stride_length': 'Stride Length',
    'temporal_symmetry': 'Temporal Symmetry',
    'tip_tail_lateral_displacement': 'Amplitude Tail Tip',
    # 'tip_tail_lateral_displacement_phase',
    'distance_traveled_cm': 'Distance Traveled',
    'median_body_length_cm': 'Body Length',
}

# Example usage:
#
#   for speed in 10 15 20 25
#   do
#       echo "${speed} cm/sec"
#       python -u summarizegaitcsv.py \
#           --gait-h5 output/strain-survey-gait-2019-11-12.h5 \
#           --speed "${speed}" > "output/strain-survey-gait-2019-11-12_summary_${speed}cm_per_sec.csv"
#   done

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gait-h5',
        help='input HDF5 file that contains gait data',
        required=True,
    )

    parser.add_argument(
        '--speed',
        help='the speed bin to use (in cm/sec)',
        required=True,
        choices=['10', '15', '20', '25']
    )

    args = parser.parse_args()

    bin_name = 'speed_{}_ang_vel_neg20'.format(args.speed)

    header = ['NetworkFilename', 'Stride Count'] + [NAME_MAPPINGS[a] for a in VIDEO_ATTRIBUTES]
    for measure in ALL_STRIDES_MEASURES_TO_SUMMARIZE + BINNED_MEASURES_TO_SUMMARIZE:
        header.append(NAME_MAPPINGS[measure])
        # header.append(measure + '_median')
        header.append(NAME_MAPPINGS[measure] + ' Variance')

    print(','.join(header))

    with h5py.File(args.gait_h5, 'r') as gait_h5:
        for grp_name, group in gait_h5.items():
            if 'bins' in group and bin_name in group['bins']:
                grp_name = urlparse.unquote(grp_name)
                curr_row = [grp_name]
                curr_row.append(str(group['bins'][bin_name].attrs['stride_count']))

                for attr in VIDEO_ATTRIBUTES:
                    curr_row.append(str(group.attrs[attr]))

                for measure in ALL_STRIDES_MEASURES_TO_SUMMARIZE:
                    values = group['all_strides'][measure][:]
                    curr_row.append(str(np.mean(values)))
                    # curr_row.append(str(np.median(values)))
                    curr_row.append(str(np.var(values)))

                for measure in BINNED_MEASURES_TO_SUMMARIZE:
                    values = group['bins'][bin_name][measure][:]
                    curr_row.append(str(np.mean(values)))
                    # curr_row.append(str(np.median(values)))
                    curr_row.append(str(np.var(values)))

                print(','.join(curr_row))


if __name__ == '__main__':
    main()
