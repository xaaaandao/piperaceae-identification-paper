import collections
import os
import time

import numpy as np
import sklearn.preprocessing

from classifier import find_best_classifier_and_params, list_classifiers
from data import merge_all_files_of_dir, get_samples_with_patch, get_info, add_data, get_cv
from result import calculate_test
from save import save, create_path_base
from save_model import save_best_model


def non_handcraft(cfg, current_datetime, filename_labels, list_data_input, list_extractor):
    list_only_dir = [d for d in list_data_input if os.path.isdir(d)]
    list_only_dir = [d for d in list_only_dir if len(os.listdir(d)) > 0]

    list_data = []
    for d in list_only_dir:
        dataset, color_mode, segmented, image_size, extractor, slice_patch = get_info(d)
        data, n_patch = merge_all_files_of_dir(d)

        new_data = np.array(data)
        n_samples, n_features = new_data.shape
        x, y = new_data[0:, 0:n_features - 1], new_data[:, n_features - 1]
        n_labels = len(np.unique(y))

        if np.isnan(x).any():
            raise ValueError(f'{d} contain is nan')

        x_normalized = sklearn.preprocessing.StandardScaler().fit_transform(x)
        list_data.append(add_data(color_mode, dataset, d, extractor, image_size, n_features-1, n_labels, n_patch,
                                  n_samples, segmented, slice_patch, x_normalized, y))

        data_with_pca(cfg, color_mode, d, dataset, extractor, image_size, list_data, list_extractor, n_features,
                      n_labels, n_patch, n_samples, segmented, slice_patch, x_normalized, y)

    # for data in list_data:
    #     print(f'dataset: {data["dataset"]} color_mode: {data["color_mode"]}')
    #     print(f'segmented: {data["segmented"]} image_size: {data["image_size"]} extractor: {data["extractor"]}')
    #     print(f'n_samples/patch: {int(data["n_samples"]) / int(data["n_patch"])}')
    #     print(f'n_samples: {data["n_samples"]} n_features: {data["n_features"]}')
    #     print(f'n_labels: {data["n_labels"]} samples_per_labels: {collections.Counter(data["y"])}')

    for classifier in list_classifiers:
        for data in list_data:
            classifier_name = classifier.__class__.__name__
            best, time_find_best_params = find_best_classifier_and_params(cfg, classifier, classifier_name, data)
            list_result_fold = []
            list_time = []

            path = create_path_base(cfg, classifier_name, current_datetime, data)

            print(f'classifier: {classifier}')
            print(f'path: {path}')

            split = get_cv(cfg, data)

            for fold, (index_train, index_test) in enumerate(split):
                x_train, y_train = get_samples_with_patch(data['x'], data['y'], index_train, data['n_patch'])
                x_test, y_test = get_samples_with_patch(data['x'], data['y'], index_test, data['n_patch'])

                print(fold, classifier_name, x_train.shape, x_test.shape)
                print('train')
                print(sorted(list(collections.Counter(y_train).items())))
                print('test')
                print(sorted(list(collections.Counter(y_test).items())))

                start_time_train_valid = time.time()
                best['classifier'].fit(x_train, y_train)
                y_pred = best['classifier'].predict_proba(x_test)

                save_best_model(best['classifier'], fold, path)

                result_max_rule, result_prod_rule, result_sum_rule = calculate_test(fold, data['n_labels'], y_pred, y_test,
                                                                                    n_patch=int(data['n_patch']))

                end_time_train_valid = time.time()
                time_train_valid = end_time_train_valid - start_time_train_valid

                list_result_fold.append(result_max_rule)
                list_result_fold.append(result_prod_rule)
                list_result_fold.append(result_sum_rule)
                list_time.append({
                    'fold': fold,
                    'time_train_valid': time_train_valid,
                    'time_search_best_params': time_find_best_params
                })

            save(best['params'], cfg, classifier_name, data, filename_labels, list_result_fold, list_time, path)


def data_with_pca(cfg, color_mode, d, dataset, extractor, image_size, list_data, list_extractor, n_features, n_labels,
                  n_patch, n_samples, segmented, slice_patch, x_normalized, y):
    for pca in list_extractor[extractor]:
        if pca < n_features - 1:
            x = sklearn.decomposition.PCA(n_components=pca, random_state=cfg['seed']).fit_transform(x_normalized)
            list_data.append(
                add_data(color_mode, dataset, d, extractor, image_size, x.shape[1], n_labels, n_patch, n_samples,
                         segmented, slice_patch, x, y))


