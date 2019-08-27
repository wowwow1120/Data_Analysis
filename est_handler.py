#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from lib import data as data_lib
from lib import learning
import argparse
import configparser

EST = True


def make_est_result_file(est_csv_file, model_bin_file, result_file, ml_cfg):

    file_to_dict = data_lib.read_csv_file_as_dict(est_csv_file)
    general_cfg = ml_cfg['general']
    ml_method = general_cfg['method']

    with open(result_file, 'a+') as new_file:
        col_name = [key for key, val in file_to_dict.items()]
        for i in col_name:
            new_file.write('{},'.format(i))
        new_file.write('{}_{}\n'.format("OUT", col_name[0]))

        correct_count = 0
        for i in range(len(file_to_dict[list(file_to_dict)[0]])):
            est_dict = {key: val[i] for key, val in file_to_dict.items()}
            real_y_value = est_dict['RH_BEFORE_DEO_O2']
            est_thresh = ml_cfg[ml_method]['est_thresh']

            if ml_method == 'RFR':
                est_result = learning.rfr_estimation(model_bin_file, est_dict)

                if abs(float(est_result)-float(real_y_value)) <= float(est_thresh):
                    correct_count += 1

                line = [val[i] for key, val in file_to_dict.items()]
                for i in line:
                    new_file.write('{},'.format(i))
                new_file.write('{}\n'.format(est_result))

            elif ml_method == 'XGBR':
                est_result = learning.xgbr_estimation(model_bin_file, est_dict)

                if abs(float(est_result)-float(real_y_value)) <= float(est_thresh):
                    correct_count += 1

                line = [val[i] for key, val in file_to_dict.items()]
                for i in line:
                    new_file.write('{},'.format(i))
                new_file.write('{}\n'.format(est_result))

            elif ml_method == 'LGBMR':
                est_result = learning.lgbmr_estimation(model_bin_file, est_dict)

                if abs(float(est_result)-float(real_y_value)) <= float(est_thresh):
                    correct_count += 1

                line = [val[i] for key, val in file_to_dict.items()]
                for i in line:
                    new_file.write('{},'.format(i))
                new_file.write('{}\n'.format(est_result))

        accuracy = (correct_count / len(file_to_dict[list(file_to_dict)[0]])) * 100
        print('accuracy : {:4.2f} %'.format(accuracy))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--est_csv_file", required=True, help="csv file for estimation")
    parser.add_argument("--ml_ini_file", required=True, help="ini file for machine learner")
    parser.add_argument("--model_bin_file", required=True, help="model bin file for training")
    parser.add_argument("--result_file", required=True, help="estimation result file")

    args = parser.parse_args()
    ml_cfg = configparser.ConfigParser()
    ml_cfg.read(args.ml_ini_file)

    make_est_result_file(args.est_csv_file, args.model_bin_file, args.result_file, ml_cfg)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        if EST:
            sys.argv.extend([
                             "--est_csv_file", "../sys_component/analysis/DB/rh_macro_model5_dataset_all_DB_425.csv",
                             "--ml_ini_file", "../sys_component/config_machine_learning/rh_macro_m5.ini",
                             "--model_bin_file", "../sys_component/model/RH_MACRO_M5/LGBMR/190823",
                             "--result_file", "../sys_component/analysis/result.csv"
                             ])
        else:
            sys.argv.extend(["--~help"])
    main()
