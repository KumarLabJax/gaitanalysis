import argparse
import csv
import h5py
import re
import urllib.parse as urlparse


# map cage ID to genotype
geno_dict = {
    'K1808151': 'WT',
    'K1808152': 'WT',
    'K1808153': 'HEMI',
    'K1808154': 'HEMI',
    'K1808155': 'HET',
    'K1808156': 'WT',
    'K1808157': 'HEMI',

    'K1808291': 'WT',
    'K1808292': 'WT',

    'K1809131': 'HOM',
    'K1809132': 'HOM',

    'K1809271': 'Trisomic',
    'K1809272': 'Trisomic',
    'K1809273': 'F1_ctrl',
    'K1809274': 'F1_ctrl',
    'K1809275': 'F1_ctrl',
}


def bin_label_to_speed_av(bin_label):
    bin_re = re.compile(r'speed_(\d+)_ang_vel_(neg)?(\d+)')
    m = bin_re.match(bin_label)

    if m.group(2) is None:
        return m.group(1), m.group(3)
    else:
        return m.group(1), '-' + m.group(3)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--default-mutation',
        default='',
        help='the default string to use for the mutation column'
             ' if the cage is not found in the hardcoded geno_dict',
    )
    parser.add_argument(
        'merged_meta_in',
        help='merged metadata input file',
    )
    parser.add_argument(
        'gait_stats_h5',
        help='gait statistics input HDF5 file',
    )
    parser.add_argument(
        'vid_table_out',
        help='output file',
    )

    args = parser.parse_args()

    net_fn_dict = dict()
    with open(args.merged_meta_in, newline='') as merged_meta_in:
        merged_meta_reader = csv.reader(merged_meta_in, delimiter='\t')
        header = next(merged_meta_reader)

        net_fn_index = header.index('NetworkFilename')
        mouse_id_index = header.index('MouseID')
        strain_index = header.index('Strain')
        dob_index = header.index('DOB')
        test_date_index = header.index('TestDate')
        cage_id_index = header.index('Cage')
        test_num_index = header.index('TestNum')
        sex_index = header.index('Sex')
        weight_index = header.index('Weight')

        for row in merged_meta_reader:
            net_fn = row[net_fn_index]
            net_fn_dict[net_fn] = row

    stats_h5 = h5py.File(args.gait_stats_h5, 'r')
    print('creating table:', args.vid_table_out)
    with open(args.vid_table_out, 'w', newline='') as vid_table_out_file:
        merged_out_writer = csv.writer(vid_table_out_file, delimiter='\t')
        merged_out_writer.writerow([
            'NetworkFilename',
            'BodyLength',
            'MouseID',
            'Strain',
            'Mutant',
            'DOB',
            'TestDate',
            'Cage',
            'TestNum',
            'Sex',
            'Weight',
            'BinSpeed',
            'BinAngularVelocity',
            'TraitName',
            'TraitValue',
        ])

        for quoted_net_fn, vid_grp in stats_h5.items():
            net_fn = urlparse.unquote(quoted_net_fn)
            print(net_fn)
            merged_meta_row = net_fn_dict[net_fn]
            assert len(header) == len(net_fn_dict[net_fn])

            body_len = float('nan')
            if 'median_body_length_cm' in vid_grp.attrs:
                body_len = vid_grp.attrs['median_body_length_cm']

            if 'bins' in vid_grp:
                for bin_name, bin_grp in vid_grp['bins'].items():
                    bin_speed, bin_av = bin_label_to_speed_av(bin_name)

                    for attr_name, attr_val in bin_grp.attrs.items():
                        merged_out_writer.writerow([
                            net_fn,
                            body_len,
                            merged_meta_row[mouse_id_index],
                            merged_meta_row[strain_index],
                            geno_dict.get(merged_meta_row[cage_id_index], args.default_mutation),
                            merged_meta_row[dob_index],
                            merged_meta_row[test_date_index],
                            merged_meta_row[cage_id_index],
                            merged_meta_row[test_num_index],
                            merged_meta_row[sex_index],
                            merged_meta_row[weight_index],
                            bin_speed,
                            bin_av,
                            attr_name,
                            attr_val,
                        ])

if __name__ == "__main__":
    main()
