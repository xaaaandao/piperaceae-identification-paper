import os
import time

import numpy as np

from classifier import find_best_classifier_and_params, list_classifiers
from data import merge_all_files_of_dir, get_samples_with_patch, get_info, get_cv, show_info_data, \
    show_info_data_train_test
from data import get_x_y
from result import calculate_test, insert_result_fold_and_time
from save import save, create_path_base, save_info_samples
from save_model import save_best_model


def non_handcraft(cfg, current_datetime, labels, list_data_input, list_extractor, metric):
    list_dir = [d for d in list_data_input if os.path.isdir(d) and len(os.listdir(d)) > 0]
    list_data = load_data(cfg, list_extractor, list_dir)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data, metric)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                x_test, x_train, y_test, y_train = split_train_test(data, index_test, index_train)

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_info_samples(fold, labels, index_train, index_test, data['n_patch'], path, data['y'], y_train, y_test)
                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = get_result(data, fold, labels, y_pred, y_test)

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule, result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)

            save(best['params'], cfg, classifier_name, data, labels, list_result_fold, list_time, metric, path)


def split_train_test(data, index_test, index_train, handcraft=False):
    if handcraft:
        x = data['x']
        y = data['y']
        x_train, y_train = x[index_train], y[index_train]
        x_test, y_test = x[index_test], y[index_test]
    else:
        x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, data['n_patch'])
        x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, data['n_patch'])
    return x_test, x_train, y_test, y_train


def load_data(cfg, list_extractor, list_inputs, handcraft=False):
    list_data = []
    if handcraft:
        for file in list_inputs:
            dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(file)
            get_x_y(cfg, color_mode, np.loadtxt(file), dataset, extractor, file, image_size, list_data, list_extractor, 1, segmented, slice_patch)
    else:
        for d in list_inputs:
            dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(d)
            data, n_patch = merge_all_files_of_dir(d)
            get_x_y(cfg, color_mode, np.array(data), dataset, extractor, d, image_size, list_data, list_extractor, n_patch, segmented, slice_patch)
    return list_data


def get_result(data, fold, labels, y_pred, y_test, handcraft=False):
    if handcraft:
        return calculate_test(fold, labels, y_pred, y_test)
    return calculate_test(fold, labels, y_pred, y_test, n_patch=int(data['n_patch']))