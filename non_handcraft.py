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
    list_data = []
    list_only_dir = [d for d in list_data_input if os.path.isdir(d) and len(os.listdir(d)) > 0]

    for d in list_only_dir:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(d)
        data, n_patch = merge_all_files_of_dir(d)
        get_x_y(cfg, color_mode, np.array(data), dataset, extractor, d, image_size, list_data, list_extractor, n_patch,
                segmented, slice_patch)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data, metric)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, data['n_patch'])
                x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, data['n_patch'])

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_info_samples(fold, labels, index_train, index_test, data['n_patch'], path, data['y'], y_train, y_test)
                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, labels, y_pred, y_test, n_patch=int(data['n_patch']))

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule, result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)

            save(best['params'], cfg, classifier_name, data, labels, list_result_fold, list_time, metric, path)


