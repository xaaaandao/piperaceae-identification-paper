import os
import time

from classifier import find_best_classifier_and_params, list_classifiers
from data import get_info, get_cv, show_info_data, show_info_data_train_test
from non_handcraft import load_data, split_train_test, get_result
from result import calculate_test, insert_result_fold_and_time
from save import save, create_path_base, save_info_samples
from save_model import save_best_model


def handcraft(cfg, current_datetime, labels, list_data_input, list_extractor, metric):
    list_files = [file for file in list_data_input if os.path.isfile(file)]
    list_data = load_data(cfg, list_extractor, list_files, handcraft=True)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            best, classifier_name, time_find_best_params = find_best_classifier_and_params(cfg, classifier, data, metric)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)
            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                x_test, x_train, y_test, y_train = split_train_test(data, index_test, index_train, handcraft=True)

                show_info_data_train_test(classifier_name, fold, x_test, x_train, y_test, y_train)

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_info_samples(fold, labels, index_train, index_test, data['n_patch'], path, data['y'], y_train, y_test)
                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = get_result(data, fold, labels, y_pred, y_test, handcraft=True)

                end_time_train_valid = time.time()
                insert_result_fold_and_time(end_time_train_valid, fold, list_result_fold, list_time, result_max_rule,
                                            result_prod_rule, result_sum_rule, start_time_train_valid, time_find_best_params)
            save(best['params'], cfg, classifier_name, data, labels, list_result_fold, list_time, metric, path)



