import os
import time

import numpy as np

from classifier import find_best_classifier_and_params, list_classifiers
from data import get_info, get_cv, show_info_data, show_info_data_train_test
from non_handcraft import get_x_y
from result import calculate_test, insert_result_fold_and_time
from save import save, create_path_base
from save_model import save_best_model


def handcraft(cfg, current_datetime, labels, list_data_input, list_extractor):
    n_patch = None
    list_data = []
    list_only_file = [file for file in list_data_input if os.path.isfile(file)]

    for file in list_only_file:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(file)
        get_x_y(cfg, color_mode, np.loadtxt(file), dataset, extractor, file, image_size, list_data, list_extractor,
                n_patch, segmented, slice_patch)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)
            split = get_cv(cfg, data)
            x = data['x']
            y = data['y']
            for fold, (index_train, index_test) in enumerate(split):
                x_train, y_train = x[index_train], y[index_train]
                x_test, y_test = x[index_test], y[index_test]

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, data['n_labels'], y_pred, y_test)

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule,
                                            result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)
            save(best['params'], cfg, classifier_name, data, labels, list_result_fold, list_time, path)



