from handlers.dataset_handler import PoscoTempEstModel
from sys_component.feature_selector import FeatureSelector
from lib import data as data_lib
from lib import sys_lib
import sys
import argparse
import configparser
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

CAT = 'rh'
TAP_WORK_DATE = 'TAP_WORK_DATE'
IMPORTANT_FEATURE_NUM = 5
DIV_CHAR_NUM = 120
DIV_CHAR = '*'
DIV_LINE = ''
for _ in range(DIV_CHAR_NUM):
    DIV_LINE += DIV_CHAR
DIV_PATTERN = "\n\n" + DIV_LINE + "\n"


def statistics(feature, dat_mtx):
    dat_mtx = feature.update_dataset_by_null_is(dat_mtx)
    feature.init_statistical_properties(dat_mtx)
    feature.analyze_dataset(dat_mtx)
    feature.print_analytics_result(console=True)


def refine_dataset(feature, var_ini, dat_mtx, g_logger):
    statistics(feature, dat_mtx=dat_mtx)
    feature.refine_dataset_model(model_num=var_ini['model_info']['model_number'], feature=feature, logger=g_logger)
    feature.refine_dataset_col_by_dataset_order()
    feature.refine_dataset_col_by_null_replacement()
    feature.refine_dataset_col_by_null(var_ini)
    feature.refine_dataset_group_by_operation_pattern(cat=var_ini['model_info']['model_name'])
    feature.refine_dataset_row_by_nan_replacement()
    feature.refine_dataset_row_by_range()
    sys_lib.print_and_write(feature.log)
    feature.refine_dataset_row_by_process_type(cat=var_ini['model_info']['model_name'])
    feature.split_dataset_by_type_and_dataset_order()
    sys_lib.print_and_write(feature.log)


def filter_none_from_list(list):
    filtered_list = list(filter(None, list))
    return filtered_list


def get_statistics_from_(train_data, est_data):
    cnt_list = []
    min_list = []
    max_list = []
    mean_list = []
    std_list = []
    for i in train_data, est_data:
        cnt_list.append(len(i))
        min_list.append(min(i))
        max_list.append(max(i))
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    lresult = stats.levene(train_data, est_data)
    if lresult.pvalue <= 0.05:  #confidence_level=95%
        ttest_result = stats.ttest_ind(train_data, est_data, equal_var=False)

    else:
        ttest_result = stats.ttest_ind(train_data, est_data, equal_var=True)

    return cnt_list, min_list, max_list, mean_list, std_list, ttest_result.pvalue


def gen_plot(x, y, method, x_label=None, y_lable=None, title=None):
    kde_x = stats.gaussian_kde(x)
    kde_y = stats.gaussian_kde(y)

    if min(x) < min(y):
        if max(x) > max(y):
            grid = np.linspace(min(x), max(x))
        else:
            grid = np.linspace(min(x), max(y))

    else:
        if max(x) > max(y):
            grid = np.linspace(min(y), max(x))
        else:
            grid = np.linspace(min(y), max(y))

    plt.plot(grid, kde_x(grid), label=x_label)
    plt.plot(grid, kde_y(grid), label=y_lable)
    plt.plot(grid, abs(kde_x(grid)-kde_y(grid)), label='difference')
    plt.title(title)

    plt.legend()
    # plt.show()
    plt.savefig('{}_{}.png'.format(method, title))
    plt.clf()


def gen_time_series_plot(train_x, train_y, est_x, est_y, method, title):
    train_time = train_x
    train_value = train_y
    est_time = est_x
    est_value = est_y
    plt.scatter(train_time, train_value)
    plt.scatter(est_time, est_value)
    plt.savefig('{}_{}.png'.format(method, title))
    plt.clf()


def feature_selection(feature, var_ini, dat_mtx, g_logger):

    vt_feature_list = []
    fr_feature_list = []
    mir_feature_list = []
    rfc_feature_list = []

    refine_dataset(feature, var_ini, dat_mtx=dat_mtx, g_logger=g_logger)
    feature_selector = FeatureSelector(feature.datasets['in_num'], feature.datasets['out_num'])

    vt_features = feature_selector.variance_threshold_method(thresh=float(var_ini['refinement']['FS_var_threshold']))
    for i in range(IMPORTANT_FEATURE_NUM):
        vt_feature_list.append(vt_features[i][0])
    sys_lib.print_and_write(DIV_PATTERN + feature_selector.get_log())

    fr_features = feature_selector.f_regression_method(p_thresh=float(var_ini['refinement']
                                                                      ['FS_f_regression_p_threshold']),
                                                       selection_=False, log_=True)
    for i in range(IMPORTANT_FEATURE_NUM):
        fr_feature_list.append(fr_features[i][0])
    sys_lib.print_and_write(DIV_PATTERN + feature_selector.get_log())

    mir_features = feature_selector.mutual_info_regression_method(mi_thresh=float(var_ini['refinement']
                                                                                  ['FS_mi_regression_threshold']),
                                                                  selection_=False, log_=True)
    for i in range(IMPORTANT_FEATURE_NUM):
        mir_feature_list.append(mir_features[i][0])
    sys_lib.print_and_write(DIV_PATTERN + feature_selector.get_log())

    rfc_features = feature_selector.random_forest_classifier_method(thresh=float(var_ini['refinement']
                                                                                 ['FS_random_forest_threshold']),
                                                                    selection_=False, log_=True)
    for i in range(IMPORTANT_FEATURE_NUM):
        rfc_feature_list.append(rfc_features[i][0])
    sys_lib.print_and_write(DIV_PATTERN + feature_selector.get_log())

    return vt_feature_list, fr_feature_list, mir_feature_list, rfc_feature_list


def make_features_dict(train_dict, est_dict, feature, var_ini, dat_mtx, g_logger):

    vt_feature_list, fr_feature_list, mir_feature_list, rfc_feature_list = feature_selection(feature, var_ini,
                                                                                             dat_mtx=dat_mtx,
                                                                                             g_logger=g_logger)
    vt_feature_dict_train = {}
    vt_feature_dict_est = {}
    fr_feature_dict_train = {}
    fr_feature_dict_est = {}
    mir_feature_dict_train = {}
    mir_feature_dict_est = {}
    rfc_feature_dict_train = {}
    rfc_feature_dict_est = {}
    date_dict_train = {}
    date_dict_est = {}

    for i in range(len(vt_feature_list)):
        vt_feature_dict_train[vt_feature_list[i]] = train_dict[vt_feature_list[i]]
        vt_feature_dict_est[vt_feature_list[i]] = est_dict[vt_feature_list[i]]

    for i in range(len(fr_feature_list)):
        fr_feature_dict_train[fr_feature_list[i]] = train_dict[fr_feature_list[i]]
        fr_feature_dict_est[fr_feature_list[i]] = est_dict[fr_feature_list[i]]

    for i in range(len(mir_feature_list)):
        mir_feature_dict_train[mir_feature_list[i]] = train_dict[mir_feature_list[i]]
        mir_feature_dict_est[mir_feature_list[i]] = est_dict[mir_feature_list[i]]

    for i in range(len(rfc_feature_list)):
        rfc_feature_dict_train[rfc_feature_list[i]] = train_dict[rfc_feature_list[i]]
        rfc_feature_dict_est[rfc_feature_list[i]] = est_dict[rfc_feature_list[i]]

    date_dict_train[TAP_WORK_DATE] = train_dict[TAP_WORK_DATE]
    date_dict_est[TAP_WORK_DATE] = est_dict[TAP_WORK_DATE]

    return vt_feature_dict_train, vt_feature_dict_est, fr_feature_dict_train, fr_feature_dict_est, \
           mir_feature_dict_train, mir_feature_dict_est, rfc_feature_dict_train, rfc_feature_dict_est, date_dict_train,\
           date_dict_est


def compare_feature_statistics(train_dict, est_dict, feature, var_ini, dat_mtx, g_logger, args):

    vt_feature_dict_train, vt_feature_dict_est, fr_feature_dict_train, fr_feature_dict_est, \
    mir_feature_dict_train, mir_feature_dict_est, rfc_feature_dict_train, rfc_feature_dict_est, date_dict_train, \
    date_dict_est = make_features_dict(train_dict, est_dict, feature, var_ini, dat_mtx, g_logger)

    vt_feature_list_train = [[k, v] for k, v in vt_feature_dict_train.items()]
    vt_feature_list_est = [[k, v] for k, v in vt_feature_dict_est.items()]
    fr_feature_list_train = [[k, v] for k, v in fr_feature_dict_train.items()]
    fr_feature_list_est = [[k, v] for k, v in fr_feature_dict_est.items()]
    mir_feature_list_train = [[k, v] for k, v in mir_feature_dict_train.items()]
    mir_feature_list_est = [[k, v] for k, v in mir_feature_dict_est.items()]
    rfc_feature_list_train = [[k, v] for k, v in rfc_feature_dict_train.items()]
    rfc_feature_list_est = [[k, v] for k, v in rfc_feature_dict_est.items()]
    date_list_train = [[k, v] for k, v in date_dict_train.items()]
    date_list_est = [[k, v] for k, v in date_dict_est.items()]

    msg = '\n'

    if args.time_series_plot_:
        for i in range(len(vt_feature_dict_train)):
            feature_name = list(vt_feature_dict_train.keys())[i]
            train_value_list = vt_feature_dict_train[feature_name]
            train_time_list = date_list_train[0][1]
            est_value_list = vt_feature_dict_est[feature_name]
            est_time_list = date_list_est[0][1]
            gen_time_series_plot(train_time_list, train_value_list, est_time_list, est_value_list, 'vt',
                                 'time_series_plot of {}'.format(feature_name))

        for i in range(len(fr_feature_dict_train)):
            feature_name = list(fr_feature_dict_train.keys())[i]
            train_value_list = fr_feature_dict_train[feature_name]
            train_time_list = date_list_train[0][1]
            est_value_list = fr_feature_dict_est[feature_name]
            est_time_list = date_list_est[0][1]
            gen_time_series_plot(train_time_list, train_value_list, est_time_list, est_value_list, 'fr',
                                 'time_series_plot of {}'.format(feature_name))

        for i in range(len(mir_feature_dict_train)):
            feature_name = list(mir_feature_dict_train.keys())[i]
            train_value_list = mir_feature_dict_train[feature_name]
            train_time_list = date_list_train[0][1]
            est_value_list = mir_feature_dict_est[feature_name]
            est_time_list = date_list_est[0][1]
            gen_time_series_plot(train_time_list, train_value_list, est_time_list, est_value_list, 'vt',
                                 'time_series_plot of {}'.format(feature_name))

        for i in range(len(rfc_feature_dict_train)):
            feature_name = list(rfc_feature_dict_train.keys())[i]
            train_value_list = rfc_feature_dict_train[feature_name]
            train_time_list = date_list_train[0][1]
            est_value_list = rfc_feature_dict_est[feature_name]
            est_time_list = date_list_est[0][1]
            gen_time_series_plot(train_time_list, train_value_list, est_time_list, est_value_list, 'vt',
                                 'time_series_plot of {}'.format(feature_name))

    if args.variance_threshold_method_:
        for i in range(len(vt_feature_dict_train)):
            feature_name = list(vt_feature_dict_train.keys())[i]
            refine_vt_feature_list_train = ['' if x == '#N/A' or x == 'NULL' else x for x in vt_feature_list_train[i][1]]
            refine_vt_feature_list_est = ['' if x == '#N/A' or x == 'NULL' else x for x in vt_feature_list_est[i][1]]
            vt_feature_arr_train = np.array(list(filter(None, refine_vt_feature_list_train))).astype(np.float)
            vt_feature_arr_est = np.array(list(filter(None, refine_vt_feature_list_est))).astype(np.float)
            gen_plot(vt_feature_arr_train, vt_feature_arr_est, 'vt', '{}_{}'.format(feature_name, 'train'),
                     '{}_{}'.format(feature_name, 'est'), '{}_{}'.format(feature_name, 'distribution_graph'))
            vt_cnt, vt_min, vt_max, vt_mean, vt_std, vt_result = get_statistics_from_(vt_feature_arr_train,
                                                                                      vt_feature_arr_est)
            msg += "\n   > variance_threshold_method_result_statistics of {}".format(feature_name)
            msg += "\n     trainset cnt : {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                vt_cnt[0], vt_min[0], vt_max[0], vt_mean[0], vt_std[0])
            msg += "\n     testset cnt :  {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                vt_cnt[1], vt_min[1], vt_max[1], vt_mean[1], vt_std[1])
            msg += "\n     p-value of trainset and testset : {}".format(vt_result)
            msg += "\n"

    if args.f_regression_method_:
        for i in range(len(fr_feature_dict_train)):
            feature_name = list(fr_feature_dict_train.keys())[i]
            refine_fr_feature_list_train = ['' if x == '#N/A' or x == 'NULL' else x for x in fr_feature_list_train[i][1]]
            refine_fr_feature_list_est = ['' if x == '#N/A' or x == 'NULL' else x for x in fr_feature_list_est[i][1]]
            fr_feature_arr_train = np.array(list(filter(None, refine_fr_feature_list_train))).astype(np.float)
            fr_feature_arr_est = np.array(list(filter(None, refine_fr_feature_list_est))).astype(np.float)
            gen_plot(fr_feature_arr_train, fr_feature_arr_est, 'fr', '{}_{}'.format(feature_name, 'train'),
                     '{}_{}'.format(feature_name, 'est'), '{}_{}'.format(feature_name, 'distribution_graph'))
            fr_cnt, fr_min, fr_max, fr_mean, fr_std, fr_result = get_statistics_from_(fr_feature_arr_train, fr_feature_arr_est)
            msg += "\n\n   > f_regression_method_result_statistics of {}".format(feature_name)
            msg += "\n     trainset cnt : {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                fr_cnt[0], fr_min[0], fr_max[0], fr_mean[0], fr_std[0])
            msg += "\n     testset cnt :  {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                fr_cnt[1], fr_min[1], fr_max[1], fr_mean[1], fr_std[1])
            msg += "\n     p-value of trainset and testset : {}".format(fr_result)
            msg += "\n"

    if args.mutual_info_regression_method_:
        for i in range(len(mir_feature_dict_train)):
            feature_name = list(mir_feature_dict_train.keys())[i]
            refine_mir_feature_list_train = ['' if x == '#N/A' or x == 'NULL' else x for x in mir_feature_list_train[i][1]]
            refine_mir_feature_list_est = ['' if x == '#N/A' or x == 'NULL' else x for x in mir_feature_list_est[i][1]]
            mir_feature_arr_train = np.array(list(filter(None, refine_mir_feature_list_train))).astype(np.float)
            mir_feature_arr_est = np.array(list(filter(None, refine_mir_feature_list_est))).astype(np.float)
            gen_plot(mir_feature_arr_train, mir_feature_arr_est, 'mir', '{}_{}'.format(feature_name, 'train'),
                     '{}_{}'.format(feature_name, 'est'), '{}_{}'.format(feature_name, 'distribution_graph'))
            mir_cnt, mir_min, mir_max, mir_mean, mir_std, mir_result = get_statistics_from_(mir_feature_arr_train, mir_feature_arr_est)
            msg += "\n\n   > mutual_info_regression_method_result_statistics of {}".format(feature_name)
            msg += "\n     trainset cnt : {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                mir_cnt[0], mir_min[0], mir_max[0], mir_mean[0], mir_std[0])
            msg += "\n     testset cnt :  {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                mir_cnt[1], mir_min[1], mir_max[1], mir_mean[1], mir_std[1])
            msg += "\n     p-value of trainset and testset : {}".format(mir_result)
            msg += "\n"

    if args.random_forest_classifier_method_:
        for i in range(len(rfc_feature_dict_train)):
            feature_name = list(rfc_feature_dict_train.keys())[i]
            refine_rfc_feature_list_train = ['' if x == '#N/A' or x == 'NULL' else x for x in rfc_feature_list_train[i][1]]
            refine_rfc_feature_list_est = ['' if x == '#N/A' or x == 'NULL' else x for x in rfc_feature_list_est[i][1]]
            rfc_feature_arr_train = np.array(list(filter(None, refine_rfc_feature_list_train))).astype(np.float)
            rfc_feature_arr_est = np.array(list(filter(None, refine_rfc_feature_list_est))).astype(np.float)
            gen_plot(rfc_feature_arr_train, rfc_feature_arr_est, 'rfc', '{}_{}'.format(feature_name, 'train'),
                     '{}_{}'.format(feature_name, 'est'), '{}_{}'.format(feature_name, 'distribution_graph'))
            rfc_cnt, rfc_min, rfc_max, rfc_mean, rfc_std, rfc_result = get_statistics_from_(rfc_feature_arr_train, rfc_feature_arr_est)
            msg += "\n\n   > random_forest_classifier_method_result_statistics of {}".format(feature_name)
            msg += "\n     trainset cnt : {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                rfc_cnt[0], rfc_min[0], rfc_max[0], rfc_mean[0], rfc_std[0])
            msg += "\n     testset cnt :  {:8d} || min : {:<8} || max : {:8.2f} || mean : {:8.2f} || std : {:8.2f}".format(
                rfc_cnt[1], rfc_min[1], rfc_max[1], rfc_mean[1], rfc_std[1])
            msg += "\n     p-value of trainset and testset : {}".format(rfc_result)
            msg += "\n"

    print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_file", required=True, help="csv file used for training")
    parser.add_argument("--est_csv_file", required=True, help="csv file for estimation")
    parser.add_argument("--var_csv_file", required=True, help="csv file for variables")
    parser.add_argument("--var_ini_file", required=True, help="ini file for configuration")
    parser.add_argument("--variance_threshold_method_", default=False, action='store_true',
                        help="Flag to run variance_threshold_method")
    parser.add_argument("--f_regression_method_", default=False, action='store_true',
                        help="Flag to run f_regression_method")
    parser.add_argument("--mutual_info_regression_method_", default=False, action='store_true',
                        help="Flag to run mutual_info_regression_method")
    parser.add_argument("--random_forest_classifier_method_", default=False, action='store_true',
                        help="Flag to run random_forest_classifier_method")
    parser.add_argument("--time_series_plot_", default=False, action='store_true',
                        help="Flag to generate scatter plot")

    args = parser.parse_args()

    g_logger = sys_lib.setup_logger(CAT, '_', folder='log', console_=True)

    var_ini = configparser.ConfigParser()
    var_ini.read(args.var_ini_file)

    var_mtx = data_lib.read_csv_file(args.var_csv_file)
    model_num_str = var_ini['model_info']['model_number']
    dat_mtx = data_lib.read_csv_file(var_ini['dataset' + '_' + model_num_str]['raw_dataset_csv_filename'])

    start_pos, end_pos = sys_lib.calc_crop_info_from_ini(var_ini['var_info_csv'])
    roi_var_mtx = data_lib.crop_mtx(var_mtx, start_pos, end_pos)
    feature = PoscoTempEstModel(roi_var_mtx)
    feature.init_feat_class(var_ini, offset=start_pos[0])

    train_file_to_dict = data_lib.read_csv_file_as_dict(args.train_csv_file)
    est_file_to_dict = data_lib.read_csv_file_as_dict(args.est_csv_file)

    compare_feature_statistics(train_file_to_dict, est_file_to_dict, feature, var_ini, dat_mtx=dat_mtx,
                               g_logger=g_logger, args=args)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv.extend([
            "--train_csv_file", "../sys_component/analysis/NEW_latest/NEW_RH_Macro_raw_latest_add_date.csv",
            "--est_csv_file", "../sys_component/dataset_raw/190809_db/190809_RH_EVAL_DB_refined.csv",
            "--var_csv_file", "../sys_component/rh_macro_var.csv",
            "--var_ini_file", "../sys_component/rh_macro_var.ini",

            # "--var_csv_file", "../sys_component/cvt_macro_var.csv",
            # "--var_ini_file", "../sys_component/cvt_macro_var.ini",

            "--variance_threshold_method_",
            "--f_regression_method_",
            "--mutual_info_regression_method_",
            "--random_forest_classifier_method_",
            # "--time_series_plot_"

        ])

    main()
