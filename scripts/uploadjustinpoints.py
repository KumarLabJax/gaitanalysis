import argparse
import csv
import os
import re
import shutil


# Here we will generate a two-column file that maps input file names to physical
# file locations.
#
# Example use:
#
#   python remapjustinsfilenames.py \
#           data/metadata/MergedMetaList_2018-10-22.csv \
#           justin-xsede/all_inferences strain-survey.txt

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'merged_meta_in',
        help='merged metadata input file',
    )
    parser.add_argument(
        'point_inf_dir',
        help='directory used to store point inference files',
    )
    # parser.add_argument(
    #     'mapping_out',
    #     help='the mapping output file',
    # )
    parser.add_argument(
        'out_dir',
        help='the output directory',
    )

    args = parser.parse_args()

    network_filename_set = set()
    mouse_id_dict = dict()
    with open(args.merged_meta_in, newline='') as merged_meta_in:
        merged_meta_reader = csv.reader(merged_meta_in)
        header = next(merged_meta_reader)

        # assert len(header) >= 1
        # assert header[0] == 'NetworkFilename'
        net_fn_index = header.index('NetworkFilename')
        mouse_id_index = header.index('MouseID')

        for row in merged_meta_reader:
            net_fn = row[net_fn_index].strip()
            mouse_id = row[mouse_id_index].strip()

            network_filename_set.add(net_fn)
            mouse_id_dict[mouse_id] = net_fn

    net_id_to_path_dict = dict()
    justin_regex = re.compile(r'(.+)_(feb|APR)_\d\d-\d\d-\d\d\d\d_inference\.h5')
    for f in os.listdir(args.point_inf_dir):
        m = justin_regex.fullmatch(f)
        if m:

            id_grp = m.group(1)
            net_filename_str = None
            net_filename_guess_str = id_grp.replace('+', '/') + '.avi'
            filepath = os.path.join(args.point_inf_dir, f)

            if net_filename_guess_str in network_filename_set:
                assert net_filename_guess_str not in net_id_to_path_dict
                net_id_to_path_dict[net_filename_guess_str] = filepath
                # mapping_out_writer.writerow([net_filename_str, filepath])
                # print(net_filename_str, '|', filepath)
                net_filename_str = net_filename_guess_str
            elif id_grp in mouse_id_dict:
                assert mouse_id_dict[id_grp] not in net_id_to_path_dict
                net_id_to_path_dict[mouse_id_dict[id_grp]] = filepath
                # mapping_out_writer.writerow([mouse_id_dict[id_grp], filepath])
                # print(mouse_id_dict[id_grp], '|', filepath)
                net_filename_str = mouse_id_dict[id_grp]
            else:
                print('CANT FIND', net_filename_guess_str, 'from', f)

            if net_filename_str is not None:
                net_filename_path = os.path.join(args.out_dir, net_filename_str)
                # print('exists???', os.path.isfile(net_filename_path))
                if os.path.isfile(net_filename_path):
                    net_filename_root, _ = os.path.splitext(net_filename_path)
                    dest_path = net_filename_root + '_pose_est.h5'

                    print('copying from "{}" to "{}"'.format(filepath, dest_path))
                    shutil.copyfile(filepath, dest_path)


if __name__ == "__main__":
    main()
