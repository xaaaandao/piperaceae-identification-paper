import collections
import os
import pathlib
import time

import numpy as np
import sklearn.preprocessing

from classifier import find_best_classifier_and_params, list_classifiers
from data import merge_all_files, data_with_pca, get_samples_with_patch, get_info
from result import calculate_test
from save import save


def non_handcraft(cfg, current_datetime, kf, list_data_input, list_extractor):
    list_only_dir = [dir for dir in list_data_input if os.path.isdir(dir) and len(os.listdir(dir)) > 0]
    for dir in list_only_dir:

        n_patch = -1
        dataset, color_mode, segmented, dim, extractor, slice_patch = get_info(dir)

        list_data = []
        n_patch = merge_all_files(dir, list_data, n_patch)

        new_data = np.array(list_data)
        n_samples, n_features = new_data.shape
        x, y = new_data[0:, 0:n_features - 1], new_data[:, n_features - 1]
        n_labels = len(np.unique(y))

        print(dataset, color_mode, segmented, dim, extractor)
        print(n_samples, n_features, n_labels, collections.Counter(y))

        x_normalized = sklearn.preprocessing.StandardScaler().fit_transform(x)
        list_data_pca = data_with_pca(cfg, extractor, list_extractor, x_normalized, y)

        for data in list_data_pca:
            for classifier in list_classifiers:
                classifier_name = classifier.__class__.__name__

                best, time_search_best_params = find_best_classifier_and_params(cfg,
                                                                                classifier,
                                                                                classifier_name,
                                                                                data)

                list_result_fold = []
                list_time = []

                path = create_path_base(cfg, classifier_name, color_mode, current_datetime, data, dataset, dim,
                                        extractor, n_patch, segmented)

                for fold, (index_train, index_test) in enumerate(kf.split(np.random.rand(n_samples, ))):
                    x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, int(n_patch))
                    x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, int(n_patch))

                    print(fold, classifier_name, x_train.shape, x_test.shape)
                    all_labels = collections.Counter(y)

                    for key, value in collections.Counter(y_train).items():
                        print('train', key, value, round((value * 100) / all_labels[key], 2))

                    for key, value in collections.Counter(y_test).items():
                        print('test', key, value, round((value * 100) / all_labels[key], 2))

                    start_time_train_valid = time.time()
                    best['classifier'].fit(x_train, y_train)
                    y_pred = best['classifier'].predict_proba(x_test)

                    result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, n_labels, y_pred, y_test,
                                                                                        n_patch=int(n_patch))
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

                save(best['params'], cfg, classifier_name, color_mode, data, dataset, dim, extractor, dir,
                     list_result_fold, list_time, n_patch, path, slice_patch)


def create_path_base(cfg, classifier_name, color_mode, current_datetime, data, dataset, dim, extractor, n_patch,
                     segmented):
    path = os.path.join(cfg['dir_output'], current_datetime, dataset, segmented, color_mode, dim, extractor,
                        classifier_name, f'patch={n_patch}', str(data['pca']))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path
