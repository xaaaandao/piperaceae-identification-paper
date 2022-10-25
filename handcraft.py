import collections
import os
import pathlib
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from classifier import find_best_classifier_and_params, list_classifiers
from data import get_info, add_data, get_cv, show_info_data
from non_handcraft import data_with_pca
from result import calculate_test
from save import save, create_path_base
from save_model import save_best_model


def handcraft(cfg, current_datetime, labels, list_data_input, list_extractor):
    n_patch = None
    list_data = []
    list_only_file = [file for file in list_data_input if os.path.isfile(file)]
    for file in list_only_file:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(file)

        data = np.loadtxt(file)
        n_samples, n_features = data.shape
        x, y = data[0:, 0:n_features - 1], data[:, n_features - 1]
        n_labels = len(np.unique(y))

        if np.isnan(x).any():
            raise ValueError(f'{file} contain is nan')

        x_normalized = StandardScaler().fit_transform(x)
        list_data.append(add_data(color_mode, dataset, file, extractor, image_size, n_features - 1, n_labels, n_patch,
                                  n_samples, segmented, slice_patch, x_normalized, y))

        data_with_pca(cfg, color_mode, file, dataset, extractor, image_size, list_data, list_extractor, n_features,
                      n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y)

    for data in list_data:
        show_info_data(data)

        for classifier in list_classifiers:
            classifier_name = classifier.__class__.__name__

            best, time_search_best_params = find_best_classifier_and_params(
                cfg,
                classifier,
                classifier_name,
                data)

            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)

            print(f'classifier: {classifier}')
            print(f'path: {path}')

            split = get_cv(cfg, data)
            x = data['x']
            y = data['y']
            for fold, (index_train, index_test) in enumerate(split):
                x_train, y_train = x[index_train], y[index_train]
                x_test, y_test = x[index_test], y[index_test]

                print(fold, classifier_name, x_train.shape, x_test.shape)
                print('train')
                print(sorted(list(collections.Counter(y_train).items())))
                print('test')
                print(sorted(list(collections.Counter(y_test).items())))

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, data['n_labels'], y_pred, y_test)
                end_time_train_valid = time.time()
                time_train_valid = end_time_train_valid - start_time_train_valid

                list_result_fold.append(result_max_rule)
                list_result_fold.append(result_prod_rule)
                list_result_fold.append(result_sum_rule)
                list_time.append({
                    'fold': fold,
                    'time_train_valid': time_train_valid,
                    'time_search_best_params': time_search_best_params
                })
            save(best['params'], cfg, classifier_name, data, labels, list_result_fold, list_time, path)
