import argparse
import csv


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'merged_meta_in',
        help='merged metadata input file',
    )
    parser.add_argument(
        'gait_mutants_in',
        help='gait mutant file provided Sean',
    )
    parser.add_argument(
        'test_summary_out',
        help='the mapping output file',
    )

    args = parser.parse_args()

    mouse_id_dict = dict()
    with open(args.merged_meta_in, newline='') as merged_meta_in:
        merged_meta_reader = csv.reader(merged_meta_in)
        header = next(merged_meta_reader)

        mouse_id_index = header.index('MouseID')

        net_fn_index = header.index('NetworkFilename')
        mouse_id_index = header.index('MouseID')
        strain_index = header.index('Strain')
        test_date_index = header.index('TestDate')
        cage_id_index1 = header.index('Cage')
        test_num_index = header.index('TestNum')
        sex_index = header.index('Sex')

        for row in merged_meta_reader:

            mouse_id = row[mouse_id_index].strip()
            if mouse_id not in mouse_id_dict:
                mouse_id_dict[mouse_id] = [row]
            else:
                mouse_id_dict[mouse_id].append(row)

    cage_id_dict = dict()
    with open(args.gait_mutants_in) as gait_mutants_in:

        gait_mutants_reader = csv.reader(gait_mutants_in)
        header = next(gait_mutants_reader)

        cage_id_index2 = header.index('CageID')
        mutant_index = header.index('Mutant')
        genotype_index = header.index('Genotype')
        sex_index = header.index('Sex')

        for row in gait_mutants_reader:
            cage_id = row[cage_id_index2].strip()
            mutant = row[mutant_index]
            geno = row[genotype_index]
            sex = row[sex_index]
            cage_id_dict[cage_id] = (mutant, geno, sex)

    geno_count_dict = dict()
    for mouse_id, rows in mouse_id_dict.items():
        test_nums = {row[test_num_index] for row in rows}
        if test_nums != {'1', '2'}:
            print(mouse_id)

        for row in rows:
            test_num = row[test_num_index]
            cage_id = row[cage_id_index1]
            cage_tuple = cage_id_dict[cage_id]

            if cage_tuple in geno_count_dict:
                test_num_dict = geno_count_dict[cage_tuple]
            else:
                test_num_dict = {
                    '1': 0,
                    '2': 0,
                }
                geno_count_dict[cage_tuple] = test_num_dict

            test_num_dict[test_num] += 1

    print(geno_count_dict)

    with open(args.test_summary_out, 'w', newline='') as test_summary_out:

        test_summary_writer = csv.writer(test_summary_out, delimiter='\t')
        test_summary_writer.writerow(['Mutant', 'Genotype', 'Sex', 'Test 1 Count', 'Test 2 Count'])

        for key, test_num_dict in geno_count_dict.items():
            mut, geno, sex = key
            test_summary_writer.writerow([mut, geno, sex, test_num_dict['1'], test_num_dict['2']])


if __name__ == "__main__":
    main()
